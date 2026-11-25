//! VP8 key frame encoder. Two paths share the transforms and the entropy coder: the default one
//! (own mode search, two token partitions, skip flags) and, with `match_libwebp` at methods 3 to 6,
//! libwebp's RD_OPT_BASIC with its decisions and bitstream layout, so the output equals cwebp's.
//! Ported from webp-go's internal/vp8enc, which follows libwebp (BSD-3-Clause, see NOTICE) and
//! golang.org/x/image/vp8.

mod boolenc;
mod header;
mod libwebp;
mod mb;
pub(crate) mod predict;
pub(crate) mod quant;
mod rdo;
mod segment;
#[cfg(test)]
mod tests;
mod tokens;
mod transform;
mod wave;

use crate::Error;
use crate::encode::{Options, Source};
use crate::limits::{Budget, Ledger, Limits, check_cancel, zeroed};
use crate::tables::PRED_PROB;
use crate::vp8l::AlphaScratch;
use boolenc::BoolEncoder;
use header::{frame_tag, put_frame_header, put_i4_mode, put_i16_mode, put_uv_mode};
use libwebp::LwState;
use predict::Planes;
use quant::{Quant, new_quant};
use rdo::{Lambdas, new_lambdas};
use segment::Segs;
use std::sync::atomic::AtomicBool;
use tokens::{MAX_MB_TOK, NzCtx, TokBuf, emit, emit2, finalize_probs, record_chroma, record_luma};
use transform::{Dst, rgba_to_ycbcr};
use wave::{Piece, Wave};

#[derive(Clone, Copy, Default)]
pub(crate) struct MbInfo {
    pub skip: bool,
    pub i16: bool,
    pub modes: [u8; 16],
    pub uv: u8,
    pub seg: u8,
}

/// One encode's state.
pub(crate) struct Enc<'a> {
    cancel: &'a AtomicBool,
    /// Shared with the ALPH thread.
    budget: &'a Budget,
    w: usize,
    h: usize,
    /// Source, then reconstruction, padded to the macroblock grid.
    img: Planes,
    mbw: usize,
    mbh: usize,
    /// The w by h alpha plane when `has_alpha`.
    alpha: Vec<u8>,
    has_alpha: bool,
    /// A gray source: chroma is flat 128, so UV analysis and search are skipped.
    gray: bool,
    seg_q: [Quant; 4],
    seg_lam: [Lambdas; 4],
    method: usize,
    quality: f32,
    segments: i32,
    lw_on: bool,
    lw: Option<Box<LwState>>,
    alpha16: bool,
    legacy_alpha: bool,
    mbs: Vec<MbInfo>,
    /// The workers' tokens and the MB pass's buffers.
    wave: Wave,
    /// The premultiplied-source strip and the partitions' bytes, from `Bufs`.
    strip: Vec<u8>,
    pbufs: [Vec<u8>; 3],
    /// What the ALPH encoders keep, from `Bufs`.
    ascr: AlphaScratch,
}

/// Whether the w by h plane of n-byte pixels at `off` is one colour: an overlapping compare
/// shows a row repeats one pixel.
pub(crate) fn flat(p: &[u8], off: usize, stride: usize, w: usize, h: usize, n: usize) -> bool {
    let r0 = &p[off..off + w * n];
    if r0[n..] != r0[..r0.len() - n] {
        return false;
    }
    (1..h).all(|y| p[off + y * stride..off + y * stride + w * n] == *r0)
}

/// The working buffers an encode takes at its start and gives back at its end, so the next call
/// reuses them (the public `Encoder`). A reused buffer is charged to the next call's budget at its
/// full capacity; stale contents are always overwritten before they are read.
#[derive(Default)]
pub(crate) struct Bufs {
    img: Planes,
    mbs: Vec<MbInfo>,
    wave: Wave,
    lw: Option<Box<LwState>>,
    alpha: Vec<u8>,
    strip: Vec<u8>,
    parts: [Vec<u8>; 3],
    ascr: AlphaScratch,
}

impl Bufs {
    /// Whether anything is held from an earlier call that a later call charges.
    pub(crate) fn warm(&self) -> bool {
        let caps = [
            &self.img.y,
            &self.img.cb,
            &self.img.cr,
            &self.alpha,
            &self.strip,
        ]
        .into_iter()
        .chain(&self.parts)
        .any(|v| v.capacity() > 0);
        caps || self.mbs.capacity() > 0 || self.lw.is_some() || self.wave.warm() || self.ascr.warm()
    }
}

/// Encodes one image to a WebP file written to `out` (cleared first, its capacity reused),
/// reusing and refilling `bufs`.
pub(crate) fn encode(
    w: usize,
    h: usize,
    src: &Source,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    bufs: &mut Bufs,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let budget = Budget::new(limits);
    let mut e = Enc::new(w, h, opts, &budget, cancel);
    e.take(bufs);
    let r = e.run(src, out);
    e.give(bufs);
    r
}

/// Makes `v` hold `n` elements for one call and charges it. A retained `v` is kept if its
/// capacity is within 2x of `n` (its stale contents stay) and charged at its capacity; else it is
/// dropped and a zeroed one is allocated.
fn fit<T: Copy + Default>(budget: &Budget, v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    let cap = v.capacity();
    if (n..=2 * n).contains(&cap) {
        budget.grow((cap * size_of::<T>()) as u64)?;
        v.truncate(n);
        v.resize(n, T::default());
    } else {
        *v = Vec::new();
        budget.grow((n * size_of::<T>()) as u64)?;
        *v = zeroed(n)?;
    }
    Ok(())
}

impl<'a> Enc<'a> {
    fn new(w: usize, h: usize, opts: &Options, budget: &'a Budget, cancel: &'a AtomicBool) -> Self {
        Enc {
            cancel,
            budget,
            w,
            h,
            img: Planes::default(),
            mbw: w.div_ceil(16),
            mbh: h.div_ceil(16),
            alpha: Vec::new(),
            has_alpha: false,
            gray: false,
            seg_q: [Quant::default(); 4],
            seg_lam: [Lambdas::default(); 4],
            method: opts.method.clamp(0, 6) as usize,
            quality: opts.quality.clamp(0.0, 100.0),
            segments: opts.segments,
            lw_on: opts.match_libwebp,
            lw: None,
            alpha16: opts.alpha_libwebp16,
            legacy_alpha: opts.legacy_alpha,
            mbs: Vec::new(),
            wave: Wave::default(),
            strip: Vec::new(),
            pbufs: Default::default(),
            ascr: AlphaScratch::default(),
        }
    }

    fn take(&mut self, b: &mut Bufs) {
        self.img = std::mem::take(&mut b.img);
        self.mbs = std::mem::take(&mut b.mbs);
        self.wave = std::mem::take(&mut b.wave);
        self.lw = b.lw.take();
        self.alpha = std::mem::take(&mut b.alpha);
        self.strip = std::mem::take(&mut b.strip);
        self.pbufs = std::mem::take(&mut b.parts);
        self.ascr = std::mem::take(&mut b.ascr);
    }

    fn give(self, b: &mut Bufs) {
        b.img = self.img;
        b.mbs = self.mbs;
        b.wave = self.wave;
        b.lw = self.lw;
        b.alpha = self.alpha;
        b.strip = self.strip;
        b.parts = self.pbufs;
        b.ascr = self.ascr;
    }
}

impl Enc<'_> {
    /// Whether this encode takes the libwebp path.
    fn lw_path(&self) -> bool {
        self.lw_on && self.method >= 3
    }

    fn run(&mut self, src: &Source, out: &mut Vec<u8>) -> Result<(), Error> {
        if let Some((y, u, v, a)) = solid(src, self.w, self.h)
            && !self.lw_path()
        {
            self.alloc_mbs()?;
            if a != 255 {
                self.has_alpha = true;
                fit(self.budget, &mut self.alpha, self.w * self.h)?;
                self.alpha.fill(a);
            }
            return self.finish(|e| e.encode_solid(y, u, v), out);
        }
        self.load(src)?;
        self.finish(
            |e| {
                let mut s = if e.segments == 1 {
                    e.one_segment(e.quality)
                } else {
                    e.analyze(e.quality, e.method)?
                };
                check_cancel(e.cancel)?;
                e.encode_frame(&mut s)
            },
            out,
        )
    }

    /// Makes the VP8 frame with `frame` and, on a second thread if the budget has one, the ALPH
    /// chunk, then wraps both in RIFF into `out`. With one thread the frame comes first, then ALPH.
    fn finish(
        &mut self,
        frame: impl FnOnce(&mut Self) -> Result<Parts, Error>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let budget = self.budget;
        let alpha = std::mem::take(&mut self.alpha);
        let mut ascr = std::mem::take(&mut self.ascr);
        let r = if self.has_alpha {
            let job = AlphaJob {
                w: self.w,
                h: self.h,
                lw: self.lw_path() && self.method <= 5 && !self.legacy_alpha,
                alpha16: self.alpha16,
                method: self.method,
            };
            let cancel = self.cancel;
            budget.join(
                (self.w * self.h) as u64,
                || frame(self),
                || encode_alpha(budget, &alpha, &job, cancel, &mut ascr),
            )
        } else {
            frame(self).map(|p| (p, Vec::new()))
        };
        self.alpha = alpha;
        self.ascr = ascr;
        let (parts, alph) = r?;
        let riff = self.write_riff(&parts, &alph, out);
        if self.has_alpha {
            self.ascr.out = alph;
        }
        let n = parts.n;
        for (slot, e) in self.pbufs.iter_mut().zip(parts.enc).take(n) {
            *slot = e.buf;
        }
        riff
    }

    /// Charges and allocates the per-macroblock arrays.
    fn alloc_mbs(&mut self) -> Result<(), Error> {
        let n = self.mbw * self.mbh;
        fit(self.budget, &mut self.mbs, n)?;
        self.mbs.fill(MbInfo::default());
        // A piece of tokens per row, and one more per libwebp refresh inside a row: one every
        // 97 MBs at most.
        fit(self.budget, &mut self.wave.pieces, self.mbh + n / 96 + 2)?;
        self.wave.pieces.clear();
        Ok(())
    }

    /// Charges and allocates the planes, padded to the macroblock grid.
    fn alloc_planes(&mut self) -> Result<(), Error> {
        self.alloc_mbs()?;
        let (ys, cs) = (self.mbw * 16, self.mbw * 8);
        let (yn, cn) = (ys * self.mbh * 16, cs * self.mbh * 8);
        let mut im = std::mem::take(&mut self.img);
        // The planes are not cleared: conversion and `pad` write every byte.
        fit(self.budget, &mut im.y, yn)?;
        fit(self.budget, &mut im.cb, cn)?;
        fit(self.budget, &mut im.cr, cn)?;
        (im.ys, im.cs, im.w, im.h) = (ys, cs, self.w, self.h);
        self.img = im;
        Ok(())
    }

    fn load(&mut self, src: &Source) -> Result<(), Error> {
        self.alloc_planes()?;
        let (w, h) = (self.w, self.h);
        match *src {
            Source::Rgba { pix, stride } => {
                if !self.convert(pix, stride, None) {
                    self.load_premul(pix, stride)?;
                }
            }
            Source::Nrgba { pix, stride } => {
                if !self.convert(pix, stride, None) {
                    self.has_alpha = true;
                    let mut a = std::mem::take(&mut self.alpha);
                    fit(self.budget, &mut a, w * h)?;
                    self.convert(pix, stride, Some(&mut a));
                    self.alpha = a;
                }
            }
            Source::Gray { pix, stride } => {
                self.gray = true;
                for y in 0..h {
                    let row = &pix[y * stride..y * stride + w];
                    let dst = &mut self.img.y[y * self.img.ys..y * self.img.ys + w];
                    for (d, &g) in dst.iter_mut().zip(row) {
                        *d = gray_y(g);
                    }
                }
                self.img.cb.fill(128);
                self.img.cr.fill(128);
            }
            Source::Ycbcr420 {
                y,
                cb,
                cr,
                y_stride,
                c_stride,
            } => {
                for r in 0..h {
                    let d = r * self.img.ys;
                    self.img.y[d..d + w].copy_from_slice(&y[r * y_stride..r * y_stride + w]);
                }
                let cw = w.div_ceil(2);
                for r in 0..h.div_ceil(2) {
                    let d = r * self.img.cs;
                    self.img.cb[d..d + cw].copy_from_slice(&cb[r * c_stride..r * c_stride + cw]);
                    self.img.cr[d..d + cw].copy_from_slice(&cr[r * c_stride..r * c_stride + cw]);
                }
            }
        }
        if self.has_alpha {
            self.cleanup();
        }
        self.pad();
        Ok(())
    }

    /// `rgba_to_ycbcr` over the whole image.
    fn convert(&mut self, pix: &[u8], stride: usize, a: Option<&mut [u8]>) -> bool {
        let im = &mut self.img;
        let mut d = Dst {
            y: &mut im.y,
            ys: im.ys,
            cb: &mut im.cb,
            cr: &mut im.cr,
            cs: im.cs,
            w: im.w,
            h: im.h,
        };
        rgba_to_ycbcr(&mut d, pix, stride, a)
    }

    /// Converts premultiplied RGBA in 16-row NRGBA strips, as color.NRGBAModel does.
    fn load_premul(&mut self, pix: &[u8], stride: usize) -> Result<(), Error> {
        let (w, h) = (self.w, self.h);
        let budget = self.budget;
        let mut alpha = std::mem::take(&mut self.alpha);
        fit(budget, &mut alpha, w * h)?;
        let mut strip = std::mem::take(&mut self.strip);
        fit(budget, &mut strip, 4 * w * 16)?;
        for y in (0..h).step_by(16) {
            let sh = 16.min(h - y);
            for j in 0..sh {
                let src = &pix[(y + j) * stride..(y + j) * stride + 4 * w];
                let dst = &mut strip[4 * w * j..4 * w * (j + 1)];
                for i in (0..4 * w).step_by(4) {
                    let a = u32::from(src[i + 3]);
                    match a {
                        255 => dst[i..i + 3].copy_from_slice(&src[i..i + 3]),
                        0 => dst[i..i + 3].fill(0),
                        _ => {
                            let a16 = a * 0x101;
                            for c in 0..3 {
                                dst[i + c] =
                                    ((u32::from(src[i + c]) * 0x101 * 0xffff / a16) >> 8) as u8;
                            }
                        }
                    }
                    dst[i + 3] = a as u8;
                }
            }
            let im = &mut self.img;
            let mut d = Dst {
                y: &mut im.y[y * im.ys..],
                ys: im.ys,
                cb: &mut im.cb[(y / 2) * im.cs..],
                cr: &mut im.cr[(y / 2) * im.cs..],
                cs: im.cs,
                w,
                h: sh,
            };
            rgba_to_ycbcr(
                &mut d,
                &strip[..4 * w * sh],
                4 * w,
                Some(&mut alpha[y * w..(y + sh) * w]),
            );
        }
        self.strip = strip;
        self.has_alpha = true;
        self.alpha = alpha;
        Ok(())
    }

    /// WebPCleanupTransparentArea on YUVA, picture_tools_enc.c:104-181: fully transparent 8x8
    /// blocks take the first colour of their run, and transparent pixels of other blocks take the
    /// mean luma of the opaque ones.
    fn cleanup(&mut self) {
        const S: usize = 8;
        let (w, h) = (self.w, self.h);
        let (ys, cs) = (self.img.ys, self.img.cs);
        let a = &self.alpha;
        let img = &mut self.img;
        let mut y = 0;
        while y + S <= h {
            let mut reset = true;
            let (mut vy, mut vu, mut vv) = (0u8, 0u8, 0u8);
            let mut x = 0;
            while x + S <= w {
                if !smoothen(&a[y * w + x..], w, &mut img.y[y * ys + x..], ys, S, S) {
                    reset = true;
                    x += S;
                    continue;
                }
                let co = y / 2 * cs + x / 2;
                if reset {
                    vy = img.y[y * ys + x];
                    vu = img.cb[co];
                    vv = img.cr[co];
                    reset = false;
                }
                for j in 0..S {
                    img.y[(y + j) * ys + x..(y + j) * ys + x + S].fill(vy);
                }
                for j in 0..S / 2 {
                    for i in 0..S / 2 {
                        img.cb[co + j * cs + i] = vu;
                        img.cr[co + j * cs + i] = vv;
                    }
                }
                x += S;
            }
            if x < w {
                smoothen(&a[y * w + x..], w, &mut img.y[y * ys + x..], ys, w - x, S);
            }
            y += S;
        }
        if y < h {
            let mut x = 0;
            while x + S <= w {
                smoothen(&a[y * w + x..], w, &mut img.y[y * ys + x..], ys, S, h - y);
                x += S;
            }
            if x < w {
                smoothen(
                    &a[y * w + x..],
                    w,
                    &mut img.y[y * ys + x..],
                    ys,
                    w - x,
                    h - y,
                );
            }
        }
    }

    /// Edge replication to the macroblock grid.
    fn pad(&mut self) {
        let (w, h) = (self.w, self.h);
        let im = &mut self.img;
        pad_plane(&mut im.y, im.ys, w, h);
        pad_plane(&mut im.cb, im.cs, w.div_ceil(2), h.div_ceil(2));
        pad_plane(&mut im.cr, im.cs, w.div_ceil(2), h.div_ceil(2));
    }

    /// Runs the MB pass as a wavefront over the MB rows (see `wave`). On the libwebp path it runs
    /// segment by segment: before each the tables are refreshed from the tokens of all MBs before
    /// it, as libwebp refreshes them before every so many MBs.
    fn encode_frame(&mut self, s: &mut Segs) -> Result<Parts, Error> {
        for i in 0..4 {
            self.seg_q[i] = new_quant(s.quant[i] as usize, s.dq_uv_ac);
            self.seg_lam[i] = new_lambdas(&self.seg_q[i]);
        }
        let lw = self.lw_path();
        let workers = wave::workers(self.mbw, self.mbh, self.budget.max_threads());
        self.wave.start(self.budget, workers, lw)?;
        self.wave.edges(self.budget, self.mbw, self.mbh)?;
        if lw {
            // The state's buffers at their capacity, a kept one's too.
            self.budget
                .grow(LwState::bytes(self.lw.as_deref()) as u64)?;
            match self.lw.as_mut() {
                Some(l) => l.reset(self.mbw, self.mbh, self.quality),
                None => self.lw = Some(LwState::new(self.mbw, self.mbh, self.quality)),
            }
        }
        let n = self.mbw * self.mbh;
        let mut nskip = 0;
        if lw {
            let (mut k0, mut carry) = (0, None);
            while k0 < n {
                let len = self.lw.as_mut().map_or(n - k0, |l| l.segment(n - k0));
                let p0 = self.wave.pieces.len();
                let (c, _, edge) = self.encode_rows(k0, k0 + len, true, carry, workers)?;
                carry = c;
                if let Some(l) = self.lw.as_mut() {
                    // VP8RecordStats in raster order, as the halving makes the order count.
                    for p in &self.wave.pieces[p0..] {
                        l.record(self.wave.toks(p));
                    }
                    for (m, e) in l.max_edge.iter_mut().zip(edge) {
                        *m = (*m).max(e);
                    }
                }
                k0 += len;
            }
            self.adjust_filter(s);
        } else {
            nskip = self.encode_rows(0, n, false, None, workers)?.1;
        }
        self.write_frame(s, nskip)
    }

    /// Exact at qi 0; DC predicts the rest.
    fn encode_solid(&mut self, y: u8, u: u8, v: u8) -> Result<Parts, Error> {
        self.wave.start(self.budget, 1, false)?;
        self.mbs.fill(MbInfo {
            skip: true,
            i16: true,
            ..MbInfo::default()
        });
        let mut nskip = self.mbs.len();
        let tok = &mut self.wave.wk[0].tok;
        if y != 128 || u != 128 || v != 128 {
            reserve_tokens(self.budget, tok)?;
            let mut lv = [[0i16; 16]; 25];
            lv[24][0] = 8 * (i16::from(y) - 128);
            for c in 16..20 {
                lv[c][0] = 2 * (i16::from(u) - 128);
                lv[c + 4][0] = 2 * (i16::from(v) - 128);
            }
            let (mut l, mut t): (NzCtx, NzCtx) = ([0; 9], [0; 9]);
            record_luma(tok, true, &lv, &mut l, &mut t);
            record_chroma(tok, &lv, &mut l, &mut t);
            self.mbs[0].skip = false;
            nskip -= 1;
        }
        let end = tok.len();
        for row in 0..self.mbh {
            let a = if row == 0 { 0 } else { end };
            self.wave.pieces.push(Piece {
                buf: 0,
                row,
                a,
                b: end,
            });
        }
        self.write_frame(
            &Segs {
                n: 1,
                ..Segs::default()
            },
            nskip,
        )
    }

    /// Writes the header, mode bits and tokens for nskip skipped MBs.
    fn write_frame(&mut self, s: &Segs, nskip: usize) -> Result<Parts, Error> {
        let n = self.mbw * self.mbh;
        let skip_prob = ((n - nskip) * 255 / n) as u8;
        let lw = self.lw_path();
        let probs = match self.lw.as_mut() {
            Some(l) if lw => {
                l.finalize();
                l.tabs.probs
            }
            _ => finalize_probs(self.wave.wk.iter().map(|w| &w.tok[..])),
        };
        let ntok: usize = self.wave.pieces.iter().map(|p| p.b - p.a).sum();
        // Partitions kept from an earlier call are held for the whole call, so they are charged
        // at their capacity; one too small for this call is charged again as it grows.
        let kept: usize = self.pbufs.iter().map(Vec::capacity).sum();
        self.budget.grow(kept as u64)?;

        let mut p0 = BoolEncoder {
            buf: std::mem::take(&mut self.pbufs[0]),
            ..BoolEncoder::default()
        };
        p0.reset();
        // Partition 0 codes to at most 105 bytes a MB plus the header, and the tokens to at most
        // 7/8 byte each: buffers reserved at those bounds never grow, so a call after one of the
        // same size does not allocate.
        reserve_part(self.budget, &mut p0.buf, 105 * n + (1 << 16))?;
        put_frame_header(&mut p0, s, skip_prob, &probs, lw);
        for mby in 0..self.mbh {
            for mbx in 0..self.mbw {
                let mb = self.mbs[mby * self.mbw + mbx];
                // tree_enc.c:278-281.
                if s.update_map {
                    if p0.put_bit(mb.seg >= 2, s.probs[0]) {
                        p0.put_bit(mb.seg & 1 == 1, s.probs[2]);
                    } else {
                        p0.put_bit(mb.seg & 1 == 1, s.probs[1]);
                    }
                }
                if !lw {
                    p0.put_bit(mb.skip, skip_prob);
                }
                if p0.put_bit(mb.i16, 145) {
                    put_i16_mode(&mut |b, p| p0.put_bit(b, p), mb.modes[0]);
                } else {
                    for n in 0..16 {
                        let (above, left) = self.ctx_modes(mbx, mby, n);
                        put_i4_mode(
                            &mut |b, p| p0.put_bit(b, p),
                            mb.modes[n],
                            &PRED_PROB[above as usize][left as usize],
                        );
                    }
                }
                put_uv_mode(&mut |b, p| p0.put_bit(b, p), mb.uv);
            }
        }
        let a_len = p0.finish().len();
        if a_len >= 1 << 19 {
            return Err(Error::Unsupported("first partition too large"));
        }
        let tag = frame_tag(self.w, self.h, a_len);
        let mut p1 = BoolEncoder {
            buf: std::mem::take(&mut self.pbufs[1]),
            ..BoolEncoder::default()
        };
        p1.reset();
        let tok_bytes = 7 * ntok / 8 + 16;
        reserve_part(self.budget, &mut p1.buf, tok_bytes)?;
        let wave = &self.wave;
        if lw {
            // One partition, and the chunk size counts the pad, syntax_enc.c:333-341.
            for pc in &wave.pieces {
                emit(&mut p1, wave.toks(pc), &probs);
            }
            p1.finish();
            return Ok(Parts {
                pad_size: true,
                tag,
                psz: [0; 3],
                enc: [p0, p1, BoolEncoder::default()],
                n: 2,
            });
        }
        let mut p2 = BoolEncoder {
            buf: std::mem::take(&mut self.pbufs[2]),
            ..BoolEncoder::default()
        };
        p2.reset();
        reserve_part(self.budget, &mut p2.buf, tok_bytes)?;
        // One piece per row here, in row order.
        emit2(
            &mut p1,
            &mut p2,
            self.mbh,
            |r| wave.toks(&wave.pieces[r]),
            &probs,
        );
        let b_len = p1.finish().len();
        p2.finish();
        if b_len >= 1 << 24 {
            return Err(Error::Unsupported("token partition too large"));
        }
        let psz = [b_len as u8, (b_len >> 8) as u8, (b_len >> 16) as u8];
        Ok(Parts {
            pad_size: false,
            tag,
            psz,
            enc: [p0, p1, p2],
            n: 3,
        })
    }

    /// Wraps the frame in RIFF, with VP8X and the ALPH chunk data `alph` when the image has alpha.
    /// `pad_size` counts the pad in the VP8 chunk size, as libwebp does.
    fn write_riff(&mut self, parts: &Parts, alph: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        let pad_size = parts.pad_size;
        let e = &parts.enc;
        let none = &parts.tag[..0];
        let chunk: [&[u8]; 5] = if pad_size {
            [&parts.tag[..], &e[0].buf, &e[1].buf, none, none]
        } else {
            [
                &parts.tag[..],
                &e[0].buf,
                &parts.psz[..],
                &e[1].buf,
                &e[2].buf,
            ]
        };
        let n: usize = chunk.iter().map(|c| c.len()).sum();
        let pad = n & 1;
        let head = if self.has_alpha { 20 + 18 + 8 } else { 20 };
        let apad = alph.len() & 1;
        let total = head + alph.len() + apad + n + pad + if self.has_alpha { 8 } else { 0 };
        self.budget.grow(total as u64)?;
        out.clear();
        out.try_reserve_exact(total)
            .map_err(|_| Error::OutOfMemory)?;
        let sz = if pad_size { n + pad } else { n };
        if self.has_alpha {
            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(
                &((4 + 18 + 8 + alph.len() + apad + 8 + n + pad) as u32).to_le_bytes(),
            );
            out.extend_from_slice(b"WEBPVP8X");
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&[0x10, 0, 0, 0]); // alpha
            out.extend_from_slice(&((self.w - 1) as u32).to_le_bytes()[..3]);
            out.extend_from_slice(&((self.h - 1) as u32).to_le_bytes()[..3]);
            out.extend_from_slice(b"ALPH");
            out.extend_from_slice(&(alph.len() as u32).to_le_bytes());
            out.extend_from_slice(alph);
            out.extend(std::iter::repeat_n(0, apad));
            out.extend_from_slice(b"VP8 ");
        } else {
            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(&((4 + 8 + n + pad) as u32).to_le_bytes());
            out.extend_from_slice(b"WEBPVP8 ");
        }
        out.extend_from_slice(&(sz as u32).to_le_bytes());
        for c in &chunk {
            out.extend_from_slice(c);
        }
        out.extend(std::iter::repeat_n(0, pad));
        Ok(())
    }
}

/// Makes room in `tok` for one more macroblock's tokens, charging what the buffer grows by. The
/// grown buffer is charged whole while the old one is still live (the copy holds both), then the
/// old one is given back.
fn reserve_tokens(budget: &Budget, tok: &mut TokBuf) -> Result<(), Error> {
    if tok.room() < MAX_MB_TOK {
        // Grows by half the tokens held, so the copies stay few.
        let (held, extra) = (tok.held(), (tok.len() / 2).max(MAX_MB_TOK));
        let cap = tok.capacity();
        budget.grow(2 * (held + extra) as u64)?;
        if tok.grow(extra).is_err() {
            // The old buffer is still live: give back the new one's charge only.
            budget.release(2 * (held + extra) as u64);
            return Err(Error::OutOfMemory);
        }
        budget.release(2 * cap as u64);
        // The allocator may have given more than was asked for.
        let more = tok.capacity() - (held + extra);
        budget.grow(2 * more as u64)?;
    }
    Ok(())
}

/// Makes the empty partition buffer `buf` hold at least `need` bytes. Its capacity is charged
/// already; if it must grow, the new buffer is charged while the old one is still live (the
/// reallocation holds both), then the old one is given back.
fn reserve_part(budget: &Budget, buf: &mut Vec<u8>, need: usize) -> Result<(), Error> {
    let cap = buf.capacity();
    if cap < need {
        budget.grow(need as u64)?;
        if buf.try_reserve_exact(need).is_err() {
            // The old buffer is still live: give back the new one's charge only.
            budget.release(need as u64);
            return Err(Error::OutOfMemory);
        }
        budget.release(cap as u64);
        // The allocator may have given more than was asked for.
        budget.grow((buf.capacity() - need) as u64)?;
    }
    Ok(())
}

/// The VP8 frame as the partitions it is written in: the frame tag, the first partition and the
/// token partitions (a 3 byte size of the first token partition goes between them unless `pad_size`).
struct Parts {
    pad_size: bool,
    tag: [u8; 10],
    psz: [u8; 3],
    /// The partitions in order; the first `n` are used.
    enc: [BoolEncoder; 3],
    n: usize,
}

/// What the ALPH encoder needs to know of the encode.
struct AlphaJob {
    w: usize,
    h: usize,
    /// libwebp's ALPH writer, else webp-go's original one of method 6 and `legacy_alpha`: lossless, no
    /// filter, no level reduction.
    lw: bool,
    alpha16: bool,
    method: usize,
}

/// Makes the ALPH chunk data of the w by h plane `a`, in the buffers of `scr`. The encoder
/// charges the kept buffers at their capacity and each buffer it allocates through a `Ledger`
/// and checks `cancel` once a row or a pass; the charges are released when it ends, except
/// those of the output and of the scratch kept in `scr`, which stay allocated. Runs on its own
/// thread when the budget has one.
fn encode_alpha(
    budget: &Budget,
    a: &[u8],
    j: &AlphaJob,
    cancel: &AtomicBool,
    scr: &mut AlphaScratch,
) -> Result<Vec<u8>, Error> {
    check_cancel(cancel)?;
    let (w, h) = (j.w, j.h);
    let ledger = Ledger::new(budget, cancel);
    let alph = scr.encode(&ledger, j.lw, j.alpha16, &a[..w * h], w, h, j.method)?;
    // The output stays allocated at its capacity (the libwebp method-3 path reserves more than
    // it writes), so that is what stays charged.
    budget.grow(alph.capacity() as u64)?;
    Ok(alph)
}

/// Limited-range luma from an 8-bit gray sample.
fn gray_y(g: u8) -> u8 {
    ((56318 * u32::from(g) + (1 << 15) + (16 << 16)) >> 16) as u8
}

/// SmoothenBlock: whether the block is fully transparent.
fn smoothen(a: &[u8], a_stride: usize, l: &mut [u8], l_stride: usize, w: usize, h: usize) -> bool {
    let (mut sum, mut n) = (0usize, 0usize);
    for j in 0..h {
        for i in 0..w {
            if a[j * a_stride + i] != 0 {
                n += 1;
                sum += usize::from(l[j * l_stride + i]);
            }
        }
    }
    if n > 0 && n < w * h {
        let avg = (sum / n) as u8;
        for j in 0..h {
            for i in 0..w {
                if a[j * a_stride + i] == 0 {
                    l[j * l_stride + i] = avg;
                }
            }
        }
    }
    n == 0
}

fn pad_plane(p: &mut [u8], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        let row = &mut p[y * stride..(y + 1) * stride];
        let v = row[w - 1];
        row[w..].fill(v);
    }
    let (head, tail) = p.split_at_mut(h * stride);
    let last = &head[(h - 1) * stride..];
    for row in tail.chunks_exact_mut(stride) {
        row.copy_from_slice(last);
    }
}

/// The one colour and alpha of the image, if it has one: (Y, Cb, Cr, alpha).
fn solid(src: &Source, w: usize, h: usize) -> Option<(u8, u8, u8, u8)> {
    match *src {
        Source::Rgba { pix, stride } => solid_rgba(pix, stride, w, h, true),
        Source::Nrgba { pix, stride } => solid_rgba(pix, stride, w, h, false),
        Source::Gray { pix, stride } => {
            if !flat(pix, 0, stride, w, h, 1) {
                return None;
            }
            Some((gray_y(pix[0]), 128, 128, 255))
        }
        Source::Ycbcr420 {
            y,
            cb,
            cr,
            y_stride,
            c_stride,
        } => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            if !flat(y, 0, y_stride, w, h, 1)
                || !flat(cb, 0, c_stride, cw, ch, 1)
                || !flat(cr, 0, c_stride, cw, ch, 1)
            {
                return None;
            }
            Some((y[0], cb[0], cr[0], 255))
        }
    }
}

/// solid's RGBA and NRGBA case; `pre` unpremultiplies RGBA. A uniform alpha weights chroma the
/// same in every 2x2 block.
fn solid_rgba(
    pix: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    pre: bool,
) -> Option<(u8, u8, u8, u8)> {
    if !flat(pix, 0, stride, w, h, 4) {
        return None;
    }
    let mut p = [pix[0], pix[1], pix[2], pix[3]];
    if pre && p[3] != 255 {
        // color.NRGBAModel.Convert of a color.RGBA.
        // Alpha 0 gives NRGBA{0, 0, 0, 0}.
        let a = u32::from(p[3]) * 0x101;
        for c in p.iter_mut().take(3) {
            *c = (u32::from(*c) * 0x101 * 0xffff)
                .checked_div(a)
                .map_or(0, |v| (v >> 8) as u8);
        }
    }
    let (mut y, mut cb, mut cr) = ([0u8; 1], [0u8; 1], [0u8; 1]);
    {
        let mut d = Dst {
            y: &mut y,
            ys: 1,
            cb: &mut cb,
            cr: &mut cr,
            cs: 1,
            w: 1,
            h: 1,
        };
        if !rgba_to_ycbcr(&mut d, &p, 4, None) {
            let mut ab = [0u8; 1];
            rgba_to_ycbcr(&mut d, &p, 4, Some(&mut ab));
        }
    }
    Some((y[0], cb[0], cr[0], p[3]))
}
