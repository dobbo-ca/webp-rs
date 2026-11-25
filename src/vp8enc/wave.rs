//! The macroblock pass as a wavefront over macroblock rows. Each row is encoded by one worker,
//! left to right, and macroblock (x, y) starts once row y - 1 has finished macroblock x + 1 (all
//! of row y - 1 in the last column). A macroblock then reads the same reconstruction, contexts
//! and tables as in a serial pass, so the bytes do not depend on the thread count, and one thread
//! runs the same code with no pool.
//!
//! A row passes what the row below reads through a ring of two slots per column: the bottom pixel
//! rows of each macroblock, the nonzero context it leaves, its bottom four i4 modes and the
//! libwebp path's chroma DC errors. Two slots are enough: row y + 2 writes column x only after
//! row y + 1 has finished macroblock x + 1, so row y + 1 has done its reads of x and x + 1. A
//! row's progress is published with Release after its slot is written, and read with Acquire
//! before the slot is.
//!
//! The libwebp path refreshes its probabilities and costs every so many macroblocks from the
//! statistics of all those before. Its frame runs as a series of segments, one wavefront each,
//! with the statistics recorded and the tables refreshed between them on the calling thread.

use super::libwebp::{LwScratch, LwState};
use super::predict::{Above, PRED_DC, Ybr, prepare_above};
use super::quant::Quant;
use super::rdo::Lambdas;
use super::tokens::{NzCtx, TokBuf};
use super::{Enc, MbInfo, reserve_tokens};
use crate::Error;
use crate::limits::{Budget, check_cancel};
use crate::parallel::guard;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A frame of fewer macroblocks than this runs on one worker: the handover to a pool worker and
/// the waits at the start and end of the wavefront would cost more than the work saves.
const MIN_WAVE_MBS: usize = 64;

#[cfg(test)]
thread_local! {
    /// Makes the frames this thread encodes fill a worker's scratch with garbage at the start of
    /// every row, to show that nothing left by an earlier row is read.
    pub(super) static POISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The bytes a macroblock passes to the one below it, as 7 words: its bottom luma row (0..16),
/// Cb and Cr rows (16..24, 24..32), the nonzero context it leaves (32..41), its bottom four i4
/// modes (41..45) and its chroma DC errors (45..49).
const EDGE: usize = 56;
type Edge = [u8; EDGE];
type Slot = [AtomicU64; EDGE / 8];

fn put(s: &Slot, e: &Edge) {
    for (a, c) in s.iter().zip(e.as_chunks::<8>().0) {
        a.store(u64::from_le_bytes(*c), Relaxed);
    }
}

fn get(s: &Slot) -> Edge {
    let mut e = [0u8; EDGE];
    for (c, a) in e.as_chunks_mut::<8>().0.iter_mut().zip(s) {
        *c = a.load(Relaxed).to_le_bytes();
    }
    e
}

/// `m` locked; a worker that panicked holding it leaves nothing half written that is read.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How many workers a frame of `mbw` by `mbh` macroblocks takes at most: one per row up to half
/// the width (the rows in flight are two macroblocks apart), and one for a frame of a single row
/// or under `MIN_WAVE_MBS`. Under `cfg(test)` and `--cfg fuzzing` every frame of two rows or more
/// takes one per row, so that small images reach the wavefront.
pub(crate) fn workers(mbw: usize, mbh: usize, max_threads: usize) -> usize {
    let small = cfg!(any(test, fuzzing));
    let n = if mbh < 2 || (!small && mbw * mbh < MIN_WAVE_MBS) {
        1
    } else if small {
        mbh
    } else {
        mbh.min((mbw / 2).max(1))
    };
    n.min(max_threads)
}

/// One worker's buffers, kept for the next call in `Bufs`.
#[derive(Default)]
pub(crate) struct WorkBuf {
    pub tok: TokBuf,
    tmp: Option<Box<LwScratch>>,
}

/// The macroblocks of one row that one worker encoded in one segment: their tokens are
/// `wk[buf].tok[a..b]`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Piece {
    pub buf: usize,
    pub row: usize,
    pub a: usize,
    pub b: usize,
}

/// The buffers of the macroblock pass, kept for the next call in `Bufs`.
#[derive(Default)]
pub(crate) struct Wave {
    pub wk: Vec<WorkBuf>,
    /// The token runs in bitstream order: by segment, then by row.
    pub pieces: Vec<Piece>,
    /// Two slots per column, row y in slot y % 2.
    ring: Vec<Slot>,
    /// The macroblocks each row has finished.
    prog: Vec<AtomicUsize>,
}

/// Makes `v` hold `n` default elements for one call and charges it at its capacity, keeping a
/// retained `v` whose capacity is within 2x of `n`.
fn fit_with<T: Default>(budget: &Budget, v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    let cap = v.capacity();
    if (n..=2 * n).contains(&cap) {
        budget.grow((cap * size_of::<T>()) as u64)?;
    } else {
        *v = Vec::new();
        budget.grow((n * size_of::<T>()) as u64)?;
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    }
    v.clear();
    v.resize_with(n, T::default);
    Ok(())
}

impl Wave {
    /// Whether any buffer is held from an earlier call. A kept worker is charged with its token
    /// buffer and libwebp scratch, so the room for the workers covers them.
    pub(crate) fn warm(&self) -> bool {
        self.wk.capacity() > 0
            || self.pieces.capacity() > 0
            || self.ring.capacity() > 0
            || self.prog.capacity() > 0
    }

    /// Gets the buffers of `n` workers for one frame: kept ones of more workers are dropped, the
    /// others emptied and charged at their capacity. Each gets a libwebp scratch if `lw`, and
    /// gives up a kept one if not.
    pub(crate) fn start(&mut self, budget: &Budget, n: usize, lw: bool) -> Result<(), Error> {
        self.wk.truncate(n);
        budget.grow((self.wk.capacity().max(n) * size_of::<WorkBuf>()) as u64)?;
        self.wk
            .try_reserve_exact(n - self.wk.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.wk.resize_with(n, WorkBuf::default);
        for w in &mut self.wk {
            w.tok.clear();
            budget.grow(2 * w.tok.capacity() as u64)?;
            if lw {
                budget.grow(size_of::<LwScratch>() as u64)?;
                if w.tmp.is_none() {
                    w.tmp = Some(Box::default());
                }
            } else {
                w.tmp = None;
            }
        }
        Ok(())
    }

    /// Charges and gets the ring and the row progress of a frame of `mbw` by `mbh` macroblocks.
    pub(crate) fn edges(&mut self, budget: &Budget, mbw: usize, mbh: usize) -> Result<(), Error> {
        fit_with(budget, &mut self.ring, 2 * mbw)?;
        fit_with(budget, &mut self.prog, mbh)
    }

    /// The tokens of piece `p`.
    pub(crate) fn toks(&self, p: &Piece) -> &[u16] {
        &self.wk[p.buf].tok[p.a..p.b]
    }
}

/// The state of a row that a segment ends inside, for the worker that goes on with it in the
/// next segment.
#[derive(Clone, Copy)]
pub(crate) struct Carry {
    z: Ybr,
    left: NzCtx,
    left_derr: [[i8; 2]; 2],
}

/// What the workers of one segment share: read-only settings, the ring and the progress, and
/// what they hand back.
pub(crate) struct Shared<'a> {
    pub method: usize,
    pub gray: bool,
    pub mbw: usize,
    pub seg_q: &'a [Quant; 4],
    pub seg_lam: &'a [Lambdas; 4],
    /// The libwebp state with the segment's tables, on that path.
    pub lw: Option<&'a LwState>,
    budget: &'a Budget,
    cancel: &'a AtomicBool,
    ring: &'a [Slot],
    prog: &'a [AtomicUsize],
    /// The segment: macroblocks `k0..k1` in raster order.
    k0: usize,
    k1: usize,
    carry_in: Option<Carry>,
    carry_out: Mutex<Option<Carry>>,
    pieces: Mutex<Vec<Piece>>,
    nskip: AtomicUsize,
    max_edge: [AtomicI64; 4],
    /// Set when a worker fails, so that no other waits for its row.
    abort: AtomicBool,
    /// The failure of the lowest row, which is what a serial pass reports.
    err: Mutex<Option<(usize, Error)>>,
    #[cfg(test)]
    poison: bool,
}

impl Shared<'_> {
    fn fail(&self, row: usize, e: Error) {
        self.abort.store(true, Relaxed);
        let mut err = lock(&self.err);
        if err.as_ref().is_none_or(|(r, _)| row < *r) {
            *err = Some((row, e));
        }
    }
}

/// One worker of the macroblock pass: the prediction workspace and levels the macroblock code
/// works in, its token buffer and scratch, the contexts of the macroblock being encoded and the
/// row it is in.
pub(crate) struct Worker<'s> {
    pub sh: &'s Shared<'s>,
    pub z: Ybr,
    pub q: Quant,
    pub lam: Lambdas,
    pub lv: [[i16; 16]; 25],
    pub tok: &'s mut TokBuf,
    pub tmp: Option<&'s mut LwScratch>,
    /// The nonzero context from above, updated as the macroblock is coded.
    pub top: NzCtx,
    /// The bottom four i4 modes of the macroblock above, DC in the first row.
    pub above: [u8; 4],
    pub top_derr: [[i8; 2]; 2],
    pub left_derr: [[i8; 2]; 2],
    pub max_edge: [i64; 4],
    nskip: usize,
    /// The row's macroblocks, and its 16 luma and 8 chroma lines.
    pub mbs: &'s mut [MbInfo],
    pub y: &'s mut [u8],
    pub cb: &'s mut [u8],
    pub cr: &'s mut [u8],
    pub ys: usize,
    pub cs: usize,
    buf: usize,
}

/// One MB row as a worker takes it: its index, its luma, Cb and Cr lines and its macroblocks.
type Row<'b> = (
    usize,
    (
        ((&'b mut [u8], &'b mut [u8]), &'b mut [u8]),
        &'b mut [MbInfo],
    ),
);

/// A worker: takes a buffer, then rows in order until none is left or another worker failed.
fn work<'s, 'b: 's>(
    sh: &'s Shared<'s>,
    rows: &Mutex<impl Iterator<Item = Row<'b>>>,
    bufs: &Mutex<impl Iterator<Item = (usize, &'b mut WorkBuf)>>,
    (ys, cs): (usize, usize),
) -> Result<(), Error> {
    let Some((buf, wb)) = lock(bufs).next() else {
        return Ok(());
    };
    let mut w = Worker {
        sh,
        z: [[0; 32]; 26],
        q: Quant::default(),
        lam: Lambdas::default(),
        lv: [[0; 16]; 25],
        tok: &mut wb.tok,
        tmp: wb.tmp.as_deref_mut(),
        top: [0; 9],
        above: [PRED_DC; 4],
        top_derr: [[0; 2]; 2],
        left_derr: [[0; 2]; 2],
        max_edge: [0; 4],
        nskip: 0,
        mbs: Default::default(),
        y: Default::default(),
        cb: Default::default(),
        cr: Default::default(),
        ys,
        cs,
        buf,
    };
    while !sh.abort.load(Relaxed) {
        let Some((mby, (((y, cb), cr), mbs))) = lock(rows).next() else {
            break;
        };
        match guard(|| w.row(mby, y, cb, cr, mbs)) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                sh.fail(mby, e);
                break;
            }
        }
    }
    sh.nskip.fetch_add(w.nskip, Relaxed);
    for (m, &e) in sh.max_edge.iter().zip(&w.max_edge) {
        m.fetch_max(e, Relaxed);
    }
    Ok(())
}

impl<'s> Worker<'s> {
    /// Encodes the segment's macroblocks of row `mby`. Returns false if another worker failed
    /// while this one waited.
    fn row(
        &mut self,
        mby: usize,
        y: &'s mut [u8],
        cb: &'s mut [u8],
        cr: &'s mut [u8],
        mbs: &'s mut [MbInfo],
    ) -> Result<bool, Error> {
        let sh = self.sh;
        check_cancel(sh.cancel)?;
        let (mbw, start) = (sh.mbw, mby * sh.mbw);
        let x0 = sh.k0.saturating_sub(start);
        let x1 = (sh.k1 - start).min(mbw);
        (self.y, self.cb, self.cr, self.mbs) = (y, cb, cr, mbs);
        let mut left: NzCtx = [0; 9];
        self.left_derr = [[0; 2]; 2];
        if x0 > 0
            && let Some(c) = sh.carry_in
        {
            self.z = c.z;
            left = c.left;
            self.left_derr = c.left_derr;
        }
        // A row resumed inside it (x0 > 0) keeps the prediction workspace from the carry.
        #[cfg(test)]
        if sh.poison {
            self.poison(mby, x0 == 0);
        }
        let a = self.tok.len();
        for mbx in x0..x1 {
            if !self.wait(mby, mbx) {
                return Ok(false);
            }
            reserve_tokens(sh.budget, self.tok)?;
            if sh.lw.is_some() {
                self.encode_mb_lw(mbx, mby, &mut left);
            } else if self.encode_mb(mbx, mby, &mut left) {
                self.nskip += 1;
            }
            self.publish(mby, mbx);
        }
        if x1 < mbw {
            *lock(&sh.carry_out) = Some(Carry {
                z: self.z,
                left,
                left_derr: self.left_derr,
            });
        }
        lock(&sh.pieces).push(Piece {
            buf: self.buf,
            row: mby,
            a,
            b: self.tok.len(),
        });
        Ok(true)
    }

    /// Waits until macroblock `mbx` of row `mby` may start: row `mby - 1` has finished
    /// macroblock `mbx + 1`, or its last. Spins briefly, then yields. Returns false if a worker
    /// failed.
    fn wait(&self, mby: usize, mbx: usize) -> bool {
        if mby == 0 {
            return true;
        }
        let sh = self.sh;
        let need = (mbx + 2).min(sh.mbw);
        let p = &sh.prog[mby - 1];
        let mut spins = 0u32;
        while p.load(Acquire) < need {
            if sh.abort.load(Relaxed) {
                return false;
            }
            if spins < 64 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        true
    }

    /// Fills z's edges and takes the contexts from above: from the ring, or the frame's edge in
    /// the first row.
    pub(crate) fn prepare(&mut self, mbx: usize, mby: usize) {
        if mby == 0 {
            self.top = [0; 9];
            self.above = [PRED_DC; 4];
            self.top_derr = [[0; 2]; 2];
            prepare_above(&mut self.z, mbx, None);
            return;
        }
        let sh = self.sh;
        let slot = ((mby - 1) & 1) * sh.mbw;
        let e = get(&sh.ring[slot + mbx]);
        let mut a = Above {
            y: [0; 20],
            u: [0; 8],
            v: [0; 8],
        };
        a.y[..16].copy_from_slice(&e[..16]);
        if mbx + 1 < sh.mbw {
            let r = sh.ring[slot + mbx + 1][0].load(Relaxed).to_le_bytes();
            a.y[16..].copy_from_slice(&r[..4]);
        } else {
            a.y[16..].fill(e[15]);
        }
        a.u.copy_from_slice(&e[16..24]);
        a.v.copy_from_slice(&e[24..32]);
        self.top.copy_from_slice(&e[32..41]);
        self.above.copy_from_slice(&e[41..45]);
        for ch in 0..2 {
            for i in 0..2 {
                self.top_derr[ch][i] = e[45 + 2 * ch + i] as i8;
            }
        }
        prepare_above(&mut self.z, mbx, Some(&a));
    }

    /// Passes macroblock `mbx`'s bottom rows and contexts to the row below, then counts it done.
    fn publish(&mut self, mby: usize, mbx: usize) {
        let sh = self.sh;
        let mut e = [0u8; EDGE];
        let (yo, co) = (15 * self.ys + 16 * mbx, 7 * self.cs + 8 * mbx);
        e[..16].copy_from_slice(&self.y[yo..yo + 16]);
        e[16..24].copy_from_slice(&self.cb[co..co + 8]);
        e[24..32].copy_from_slice(&self.cr[co..co + 8]);
        e[32..41].copy_from_slice(&self.top);
        e[41..45].copy_from_slice(&self.mbs[mbx].modes[12..]);
        for ch in 0..2 {
            for i in 0..2 {
                e[45 + 2 * ch + i] = self.top_derr[ch][i] as u8;
            }
        }
        put(&sh.ring[(mby & 1) * sh.mbw + mbx], &e);
        sh.prog[mby].store(mbx + 1, Release);
    }

    /// Fills the scratch with garbage, the prediction workspace too if `z`.
    #[cfg(test)]
    fn poison(&mut self, mby: usize, z: bool) {
        let k = (mby * 37 + 11) as u8;
        if z {
            for (j, row) in self.z.iter_mut().enumerate() {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = k ^ (j as u8).wrapping_mul(29) ^ (i as u8).wrapping_mul(13);
                }
            }
        }
        for (j, b) in self.lv.iter_mut().enumerate() {
            b.fill(-1000 - j as i16 * 7);
        }
        self.q = self.sh.seg_q[(mby + 1) & 3];
        self.lam = self.sh.seg_lam[(mby + 2) & 3];
        if let Some(t) = self.tmp.as_deref_mut() {
            t.poison(k);
        }
    }
}

impl Enc<'_> {
    /// Encodes macroblocks `k0..k1` (raster order) on up to `workers` threads, on the libwebp
    /// path with `lw` (and the state's current tables). `carry` is the state of the row the
    /// segment before ended in; the result's is the row this one ends in. Also returns the
    /// skipped macroblocks and the libwebp path's largest edge of each segment. The pieces of
    /// tokens go to `wave.pieces` in bitstream order.
    pub(super) fn encode_rows(
        &mut self,
        k0: usize,
        k1: usize,
        lw: bool,
        carry: Option<Carry>,
        workers: usize,
    ) -> Result<(Option<Carry>, usize, [i64; 4]), Error> {
        let (mbw, ys, cs) = (self.mbw, self.img.ys, self.img.cs);
        let (r0, r1) = (k0 / mbw, (k1 - 1) / mbw);
        let w = &mut self.wave;
        let p0 = w.pieces.len();
        let sh = Shared {
            method: self.method,
            gray: self.gray,
            mbw,
            seg_q: &self.seg_q,
            seg_lam: &self.seg_lam,
            lw: if lw { self.lw.as_deref() } else { None },
            budget: self.budget,
            cancel: self.cancel,
            ring: &w.ring,
            prog: &w.prog,
            k0,
            k1,
            carry_in: carry,
            carry_out: Mutex::new(None),
            pieces: Mutex::new(std::mem::take(&mut w.pieces)),
            nskip: AtomicUsize::new(0),
            max_edge: Default::default(),
            abort: AtomicBool::new(false),
            err: Mutex::new(None),
            #[cfg(test)]
            poison: POISON.get(),
        };
        let rows = self
            .img
            .y
            .chunks_mut(16 * ys)
            .zip(self.img.cb.chunks_mut(8 * cs))
            .zip(self.img.cr.chunks_mut(8 * cs))
            .zip(self.mbs.chunks_mut(mbw))
            .enumerate()
            .skip(r0)
            .take(r1 + 1 - r0);
        let rows = Mutex::new(rows);
        let bufs = Mutex::new(w.wk.iter_mut().enumerate());
        let r = self
            .budget
            .par_workers(workers, || work(&sh, &rows, &bufs, (ys, cs)));
        let Shared {
            carry_out,
            pieces,
            nskip,
            max_edge,
            err,
            ..
        } = sh;
        w.pieces = pieces.into_inner().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, e)) = err.into_inner().unwrap_or_else(PoisonError::into_inner) {
            return Err(e);
        }
        r?;
        w.pieces[p0..].sort_unstable_by_key(|p| p.row);
        Ok((
            carry_out
                .into_inner()
                .unwrap_or_else(PoisonError::into_inner),
            nskip.into_inner(),
            max_edge.map(AtomicI64::into_inner),
        ))
    }
}
