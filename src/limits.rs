use crate::Error;
use crate::parallel::{ThreadPool, guard, pool_for};
use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

/// Caps one call. Every public entry point takes a `&Limits`; there is no unlimited call. The
/// defaults are the Go package's, with `max_alloc` finite.
///
/// ```
/// use webp::Limits;
/// // For example, a caller limited to 2 CPUs and 1 GiB.
/// let limits = Limits { max_threads: 2, max_alloc: 768 << 20, ..Limits::default() };
/// assert_eq!(limits.max_pixels, 1 << 26);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// The most pixels (width x height) a decode or encode may handle. Sides are always capped at
    /// 16383 (a side over it is `Error::Width` or `Error::Height` from the decoders, and
    /// `Error::InvalidArgument` from `encode`). Default `1 << 26`. Over it is `Error::MaxPixels`.
    pub max_pixels: u64,
    /// The peak working memory in bytes of one call, all threads together: every allocation that
    /// scales with the input is charged here before it is made and released when it is freed, and
    /// the output the call returns stays charged until it returns. Default `1 << 30` (1 GiB).
    /// Over it is `Error::MemoryBudget`. Allocations of a fixed size (tables
    /// of a few KiB, thread stacks) are not charged.
    ///
    /// With `max_threads` 1 (or 0) the charges are the same on every run, so whether a call fits
    /// is deterministic. With 2 or more, the VP8 frame and the ALPH chunk of an image with alpha
    /// are encoded (or decoded) at the same time, and the encoder's macroblock rows run on
    /// several threads, each with its own token buffer; all charge the one budget, which counts
    /// the peak of all threads together. A call that fits at `max_threads` 1 can therefore fail with
    /// `Error::MemoryBudget` at 2, and with 2 threads a call whose peak is within a few buffers
    /// of `max_alloc` may return `Ok` on one run and `Error::MemoryBudget` on another, depending
    /// on how the threads interleave. The bytes of an `Ok` result do not depend on it.
    pub max_alloc: usize,
    /// The most OS threads one call uses, the calling thread included. Default
    /// `std::thread::available_parallelism()`, or 1 if that is unknown. 1 (and 0, which is
    /// treated as 1) runs everything on the calling thread and never touches a pool. The output
    /// does not depend on it. Work that runs on two threads at once needs its working memory at
    /// once, so a call can fit `max_alloc` with 1 thread and not with more (see `max_alloc`).
    pub max_threads: usize,
    /// The worker threads the call runs its parallel stages on, taking at most `max_threads - 1`
    /// of them besides the calling thread. Default `None`: a call with `max_threads` above 1 then
    /// uses the crate's shared pool, built on the first such call with `available_parallelism`
    /// threads and never more (if it cannot be built, the call runs on the calling thread). A
    /// pool given here always wins and the shared one is not built for the call. Create one with
    /// [`ThreadPool::new`] and keep it. The output does not depend on it.
    pub pool: Option<ThreadPool>,
    /// The most EXIF and XMP bytes `metadata` may return together, each chunk also costing the 32
    /// bytes of its slice entry. Default `1 << 24`. Over it is `Error::MetadataBytes`. The
    /// chunk entries `metadata` holds are also charged against `max_alloc`.
    pub max_metadata_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_pixels: 1 << 26,
            max_alloc: 1 << 30,
            max_threads: thread::available_parallelism().map_or(1, |n| n.get()),
            pool: None,
            max_metadata_bytes: 1 << 24,
        }
    }
}

/// The most a side may have.
pub(crate) const MAX_SIDE: u64 = 16383;

/// The memory and threads one call has taken: a ledger of the bytes held against `max_alloc` (and
/// their peak) and of the threads live against `max_threads`. It is shared by reference across the
/// threads of the call.
pub(crate) struct Budget {
    max: u64,
    held: AtomicU64,
    peak: AtomicU64,
    max_threads: usize,
    /// Live threads, the caller's included.
    threads: AtomicUsize,
    peak_threads: AtomicUsize,
    /// The caller's pool, if it gave one.
    caller_pool: Option<ThreadPool>,
    /// Where `join`, `par_rows` and `par_workers` hand work, resolved by `pool()` at the first
    /// stage that has work for a second thread, so a call that never has (metadata, lossless
    /// decode, a small encode, a bad file) never builds the shared pool. `None` runs everything
    /// on the calling thread.
    pool: OnceLock<Option<ThreadPool>>,
}

/// Bytes taken with [`Budget::charge`], released on drop (the ledger tests use it).
#[cfg(test)]
pub(crate) struct Charge<'a> {
    b: &'a Budget,
    n: u64,
}

#[cfg(test)]
impl Drop for Charge<'_> {
    fn drop(&mut self) {
        self.b.release(self.n);
    }
}

/// A thread slot taken with `Budget::take_slot`, given back on drop.
struct Slot<'a>(&'a Budget);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.threads.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A band of a parallel RGB conversion has at least this many bytes of output. The conversion
/// (fancy upsampling and the YUV to RGB arithmetic) costs about 0.5 to 1 ns a byte of RGBA, so
/// 64 KiB is some 30 to 60 us of work, where handing a task to a running pool worker costs a few
/// microseconds. A call whose output is under two bands stays on the calling thread. The
/// cost per byte is an estimate, not a measurement; the bench times only the default and the
/// single-threaded runs, not other thresholds.
const MIN_BAND_BYTES: usize = 64 << 10;

/// `join` hands the second job to a pool worker only for an image of at least this many pixels.
/// The ALPH decode or encode and the VP8 work take roughly 5 to 20 ns a pixel, so at 4K pixels
/// (64x64) the shorter side is some 20 to 80 us, against a few microseconds to wake a worker; a
/// smaller image has less work than that and stays on the calling thread.
///
/// Under `--cfg fuzzing` it is 64 pixels, so that the small images a fuzz input can afford reach
/// the threaded paths.
pub(crate) const MIN_JOIN_PIXELS: u64 = if cfg!(fuzzing) { 1 << 6 } else { 1 << 12 };

/// What `par_workers` allocates for each worker it hands work to, outside the buffers its
/// callers charge: the worker's thread slot and result (8 and 32 bytes), rayon's boxed job for
/// the spawn (the closure and its scope, under 100 bytes) and its share of the injector queue's
/// blocks (24 bytes a job). Rounded up to 256 and charged per worker for the stage, so the peak
/// stays within `max_alloc` at any `max_threads`.
const PER_WORKER: u64 = 256;

impl Budget {
    pub(crate) fn new(l: &Limits) -> Budget {
        Budget {
            max: l.max_alloc as u64,
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            max_threads: l.max_threads.max(1),
            threads: AtomicUsize::new(1),
            peak_threads: AtomicUsize::new(1),
            caller_pool: l.pool.clone(),
            pool: OnceLock::new(),
        }
    }

    /// Charges `n` bytes until the call ends, before they are allocated.
    pub(crate) fn grow(&self, n: u64) -> Result<(), Error> {
        let mut cur = self.held.load(Ordering::Relaxed);
        loop {
            let got = cur.saturating_add(n);
            if got > self.max {
                return Err(Error::MemoryBudget { got, max: self.max });
            }
            match self
                .held
                .compare_exchange_weak(cur, got, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(got, Ordering::Relaxed);
                    return Ok(());
                }
                Err(c) => cur = c,
            }
        }
    }

    /// Charges `n` bytes until the returned guard is dropped.
    #[cfg(test)]
    pub(crate) fn charge(&self, n: u64) -> Result<Charge<'_>, Error> {
        self.grow(n)?;
        Ok(Charge { b: self, n })
    }

    /// Gives back `n` bytes charged with `grow`.
    pub(crate) fn release(&self, n: u64) {
        self.held.fetch_sub(n, Ordering::Relaxed);
    }

    /// The most bytes held at once so far.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// The bytes held now.
    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }

    /// The most threads live at once so far, the caller's included.
    #[cfg(test)]
    pub(crate) fn peak_threads(&self) -> usize {
        self.peak_threads.load(Ordering::Relaxed)
    }

    /// The most threads the call may use, the caller's included (`max_threads`, at least 1).
    pub(crate) fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// The pool this call hands work to: the caller's, else (with `max_threads` above 1) the
    /// shared one, built now if this is the first call in the process that needs it.
    fn pool(&self) -> Option<&ThreadPool> {
        if self.max_threads <= 1 {
            return None;
        }
        self.pool
            .get_or_init(|| pool_for(&self.caller_pool, self.max_threads))
            .as_ref()
    }

    /// Takes one more thread if `max_threads` allows.
    fn take_slot(&self) -> Option<Slot<'_>> {
        let mut cur = self.threads.load(Ordering::Relaxed);
        loop {
            if cur >= self.max_threads {
                return None;
            }
            match self.threads.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak_threads.fetch_max(cur + 1, Ordering::Relaxed);
                    return Some(Slot(self));
                }
                Err(c) => cur = c,
            }
        }
    }

    /// Runs `a` on the calling thread and `b` on a pool worker if the image has `pixels >=
    /// MIN_JOIN_PIXELS` and `max_threads` has a thread to give, else `a` then `b` on the calling
    /// thread. This, `par_rows` and `par_workers` are the only places that hand work to the pool.
    /// An error of `a` comes before one of `b`. On the pool a panic in `a` or `b` is an
    /// `Error::Internal`.
    pub(crate) fn join<A, B: Send>(
        &self,
        pixels: u64,
        a: impl FnOnce() -> Result<A, Error>,
        b: impl FnOnce() -> Result<B, Error> + Send,
    ) -> Result<(A, B), Error> {
        let pool = if pixels >= MIN_JOIN_PIXELS {
            self.pool()
        } else {
            None
        };
        let (Some(pool), Some(slot)) = (pool, pool.and_then(|_| self.take_slot())) else {
            return Ok((a()?, b()?));
        };
        let mut rb = None;
        let ra = pool.0.in_place_scope(|s| {
            s.spawn(|_| {
                let _slot = slot;
                rb = Some(guard(b));
            });
            guard(a)
        });
        let rb = rb.unwrap_or_else(|| Err(Error::Internal("task did not run".into())));
        Ok((ra?, rb?))
    }

    /// Calls `f(band, first_row, end_row)` over `dst`, whose rows are `row_len` bytes, in bands on
    /// as many threads as `max_threads` gives (at least `MIN_BAND_BYTES` of `dst` each, and no
    /// more workers than the pool has). `f` must give the same bytes for any split. The error is
    /// the first band's, as a serial run gives; a panic in `f` on the pool is an
    /// `Error::Internal`.
    pub(crate) fn par_rows(
        &self,
        dst: &mut [u8],
        row_len: usize,
        f: impl Fn(&mut [u8], usize, usize) -> Result<(), Error> + Sync,
    ) -> Result<(), Error> {
        let rows = dst.len() / row_len.max(1);
        let bands = (dst.len() / MIN_BAND_BYTES).saturating_sub(1);
        let Some(pool) = self.pool().filter(|_| bands > 0) else {
            return f(dst, 0, rows);
        };
        let mut slots = Vec::new();
        let want = bands.min(pool.threads());
        while slots.len() < want {
            match self.take_slot() {
                Some(s) => slots.push(s),
                None => break,
            }
        }
        if slots.is_empty() {
            return f(dst, 0, rows);
        }
        let band = rows.div_ceil(slots.len() + 1);
        let mut res = vec![Ok(()); rows.div_ceil(band)];
        let f = &f;
        pool.0.in_place_scope(|s| {
            let mut slots = slots.into_iter();
            let mut bands = dst
                .chunks_mut(band * row_len)
                .zip(res.iter_mut())
                .enumerate();
            let first = bands.next();
            for (i, (chunk, out)) in bands {
                let (r0, r1) = (i * band, i * band + chunk.len() / row_len);
                match slots.next() {
                    Some(slot) => s.spawn(move |_| {
                        let _slot = slot;
                        *out = guard(|| f(chunk, r0, r1));
                    }),
                    None => *out = guard(|| f(chunk, r0, r1)),
                }
            }
            if let Some((_, (chunk, out))) = first {
                let r1 = chunk.len() / row_len;
                *out = guard(|| f(chunk, 0, r1));
            }
        });
        res.into_iter().collect()
    }

    /// Runs `f` on up to `n` threads at once: on the calling thread, and on as many pool workers
    /// as `max_threads` and the pool give, at most `n - 1`. With `n` 1, or no thread to give, `f`
    /// runs once on the calling thread and the pool is not touched. The workers share the work
    /// through `f` itself (each takes what is left until nothing is), so `f` must finish whether
    /// it runs once or `n` times, and a worker that has started must never wait for one that has
    /// not. The error is the calling thread's, else the first worker's; a panic in `f` on the
    /// pool is an `Error::Internal`.
    pub(crate) fn par_workers(
        &self,
        n: usize,
        f: impl Fn() -> Result<(), Error> + Sync,
    ) -> Result<(), Error> {
        let pool = if n > 1 { self.pool() } else { None };
        let Some(pool) = pool else {
            return f();
        };
        let want = (n - 1).min(pool.threads());
        let fixed = want as u64 * PER_WORKER;
        if self.grow(fixed).is_err() {
            return f();
        }
        let mut slots = Vec::new();
        while slots.len() < want {
            match self.take_slot() {
                Some(s) => slots.push(s),
                None => break,
            }
        }
        if slots.is_empty() {
            self.release(fixed);
            return f();
        }
        let mut res = vec![Ok(()); slots.len()];
        let f = &f;
        let first = pool.0.in_place_scope(|s| {
            for (slot, out) in slots.into_iter().zip(res.iter_mut()) {
                s.spawn(move |_| {
                    let _slot = slot;
                    *out = guard(f);
                });
            }
            guard(f)
        });
        let r = first.and_then(|()| res.into_iter().collect());
        self.release(fixed);
        r
    }
}

/// `n` zeroed elements, or `OutOfMemory`; the caller has charged them.
pub(crate) fn zeroed<T: Copy + Default>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, T::default());
    Ok(v)
}

/// The bytes `v` holds, at capacity.
pub(crate) fn cap_bytes<T>(v: &Vec<T>) -> u64 {
    (v.capacity() * size_of::<T>()) as u64
}

/// Makes `v` hold `n` zeroed elements, reusing its capacity. The capacity `v` has was charged to
/// `budget` already (when its owner's call began); a capacity short of `n` is dropped, given back
/// and `n` elements are charged and allocated instead.
pub(crate) fn fit_zeroed<T: Copy + Default>(
    budget: &Budget,
    v: &mut Vec<T>,
    n: usize,
) -> Result<(), Error> {
    let cap = v.capacity();
    if cap < n {
        let size = size_of::<T>() as u64;
        *v = Vec::new();
        budget.release(cap as u64 * size);
        budget.grow(n as u64 * size)?;
    }
    v.clear();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, T::default());
    Ok(())
}

pub(crate) fn check_cancel(c: &AtomicBool) -> Result<(), Error> {
    if c.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// The charges of one ALPH encode: every allocation that scales with the plane goes through it,
/// is charged to the call's [`Budget`] before it is made, and is released when the ledger drops
/// (the encoder's buffers live as long as the encode). It also carries the cancel flag. A refused
/// charge, a failed allocation and a set cancel flag come back as an `Err` that the encoder
/// functions pass up with `?`.
pub(crate) struct Ledger<'a> {
    b: &'a Budget,
    cancel: &'a AtomicBool,
    held: Cell<u64>,
}

impl Drop for Ledger<'_> {
    fn drop(&mut self) {
        self.b.release(self.held.get());
    }
}

impl<'a> Ledger<'a> {
    pub(crate) fn new(b: &'a Budget, cancel: &'a AtomicBool) -> Ledger<'a> {
        Ledger {
            b,
            cancel,
            held: Cell::new(0),
        }
    }

    /// `Err(Error::Cancelled)` if the flag is set.
    #[inline]
    pub(crate) fn check(&self) -> Result<(), Error> {
        check_cancel(self.cancel)
    }

    /// Charges `n` bytes the caller is about to allocate, until they are given back or the
    /// ledger drops.
    pub(crate) fn take(&self, n: u64) -> Result<(), Error> {
        self.b.grow(n)?;
        self.held.set(self.held.get() + n);
        Ok(())
    }

    /// Gives back `n` bytes charged with `take` that the caller has freed.
    pub(crate) fn give(&self, n: u64) {
        let n = n.min(self.held.get());
        self.b.release(n);
        self.held.set(self.held.get() - n);
    }

    /// Leaves `n` bytes charged with `take` charged to the call when the ledger drops, for
    /// buffers that outlive it.
    pub(crate) fn keep(&self, n: u64) {
        self.held.set(self.held.get() - n.min(self.held.get()));
    }

    /// Makes the capacity of `v` at least `n` elements, charging the new block before it is
    /// allocated and giving the old one back after the copy (both are live during it).
    pub(crate) fn reserve<T>(&self, v: &mut Vec<T>, n: usize) -> Result<(), Error> {
        let cap = v.capacity();
        if cap >= n {
            return Ok(());
        }
        let size = size_of::<T>() as u64;
        let (new, old) = (n as u64 * size, cap as u64 * size);
        self.take(new)?;
        v.try_reserve_exact(n - v.len())
            .map_err(|_| Error::OutOfMemory)?;
        // A block this ledger did not charge (the `vec![1]` the legacy writer starts from) is
        // given back as far as it was.
        self.give(old);
        Ok(())
    }

    /// A new vector of `n` copies of `x`.
    pub(crate) fn alloc<T: Clone>(&self, n: usize, x: T) -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        self.resize(&mut v, n, x)?;
        Ok(v)
    }

    /// `v` cleared, then holding `n` copies of `x`.
    pub(crate) fn resize<T: Clone>(&self, v: &mut Vec<T>, n: usize, x: T) -> Result<(), Error> {
        v.clear();
        self.reserve(v, n)?;
        v.resize(n, x);
        Ok(())
    }

    /// `v.push(x)`, growing by doubling through `reserve`.
    #[inline]
    pub(crate) fn push<T>(&self, v: &mut Vec<T>, x: T) -> Result<(), Error> {
        if v.len() == v.capacity() {
            let n = (2 * v.capacity()).max(16);
            self.reserve(v, n)?;
        }
        v.push(x);
        Ok(())
    }

    /// `v.extend_from_slice(s)`.
    pub(crate) fn extend_from_slice<T: Clone>(&self, v: &mut Vec<T>, s: &[T]) -> Result<(), Error> {
        let n = v.len() + s.len();
        self.reserve(v, n)?;
        v.extend_from_slice(s);
        Ok(())
    }
}

/// Runs `f` with a ledger over a default budget that is never cancelled, for tests that call the
/// ALPH encoders directly.
#[cfg(test)]
pub(crate) fn with_ledger<R>(f: impl FnOnce(&Ledger) -> R) -> R {
    let budget = Budget::new(&Limits::default());
    let never = AtomicBool::new(false);
    f(&Ledger::new(&budget, &never))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim(max_alloc: usize, max_threads: usize) -> Limits {
        Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        }
    }

    /// Charges add up to the limit, a charge over it fails and changes nothing, and a released
    /// charge can be taken again; the peak is the most held at once.
    #[test]
    fn ledger_charges_release_and_peak() {
        let b = Budget::new(&lim(100, 1));
        let a = b.charge(60).unwrap();
        assert_eq!(
            b.charge(41).err(),
            Some(Error::MemoryBudget { got: 101, max: 100 })
        );
        b.grow(40).unwrap();
        assert_eq!(b.peak(), 100);
        assert!(b.grow(1).is_err());
        drop(a);
        b.release(40);
        let c = b.charge(70).unwrap();
        drop(c);
        assert_eq!(b.peak(), 100);
        assert!(b.charge(100).is_ok());
        assert!(b.charge(101).is_err());
    }

    /// `join` spawns only for at least `MIN_JOIN_PIXELS` pixels and if `max_threads` has a thread
    /// to give (the caller counts as one), runs `a` on the calling thread, and gives every slot
    /// back.
    #[test]
    fn join_spawns_within_max_threads() {
        let main = thread::current().id();
        for max in [0usize, 1, 2, 8] {
            let b = Budget::new(&lim(1 << 20, max));
            let (a, c) = b
                .join(
                    MIN_JOIN_PIXELS,
                    || Ok(thread::current().id()),
                    || Ok(thread::current().id()),
                )
                .unwrap();
            assert_eq!(a, main);
            assert_eq!(c != main, max >= 2, "max_threads {max}");
            assert_eq!(b.peak_threads(), if max >= 2 { 2 } else { 1 });
            assert_eq!(b.threads.load(Ordering::Relaxed), 1);
            // Under the threshold nothing is spawned.
            let b = Budget::new(&lim(1 << 20, max));
            let (a, c) = b
                .join(
                    MIN_JOIN_PIXELS - 1,
                    || Ok(thread::current().id()),
                    || Ok(thread::current().id()),
                )
                .unwrap();
            assert_eq!(
                (a, c, b.peak_threads()),
                (main, main, 1),
                "max_threads {max}"
            );
        }
    }

    /// `par_rows` gives the same bytes for every thread count, never has more threads live than
    /// `max_threads`, and does not spawn with one.
    #[test]
    fn par_rows_within_max_threads() {
        let row = 1024;
        let fill = |band: &mut [u8], r0: usize, r1: usize| {
            assert_eq!(band.len(), (r1 - r0) * row);
            for (i, r) in band.chunks_exact_mut(row).enumerate() {
                r.fill(((r0 + i) % 251) as u8);
            }
            Ok(())
        };
        let mut want = vec![0u8; 8 << 20];
        Budget::new(&lim(1 << 20, 1))
            .par_rows(&mut want, row, fill)
            .unwrap();
        for (i, r) in want.chunks_exact(row).enumerate() {
            assert!(r.iter().all(|&v| v == (i % 251) as u8), "row {i}");
        }
        for max in [0usize, 1, 2, 3, 64] {
            let b = Budget::new(&lim(1 << 20, max));
            let mut got = vec![0u8; 8 << 20];
            b.par_rows(&mut got, row, fill).unwrap();
            assert!(got == want, "max_threads {max}");
            // The caller and a worker for each of the other bands, up to what the pool has.
            let pool = b.pool().map_or(0, ThreadPool::threads);
            assert_eq!(
                b.peak_threads(),
                max.max(1).min(1 + pool),
                "max_threads {max}"
            );
            assert_eq!(b.threads.load(Ordering::Relaxed), 1);
        }
        // Under two bands of output is not worth a thread.
        let b = Budget::new(&lim(1 << 20, 8));
        b.par_rows(&mut vec![0u8; 2 * MIN_BAND_BYTES - row], row, fill)
            .unwrap();
        assert_eq!(b.peak_threads(), 1);
    }

    /// A caller's pool is the one used (and the shared one is not needed), a panic in a band is an
    /// `Error::Internal`, and the slots come back.
    #[test]
    fn caller_pool_and_panics() {
        let pool = ThreadPool::new(2).unwrap();
        let l = Limits {
            pool: Some(pool.clone()),
            ..lim(1 << 20, 3)
        };
        let b = Budget::new(&l);
        assert!(b.pool() == Some(&pool));
        let r = b.par_rows(&mut vec![0u8; 4 << 20], 1024, |_, r0, _| {
            assert!(r0 == 0, "injected at {r0}");
            Ok(())
        });
        assert!(
            matches!(r, Err(Error::Internal(ref m)) if m.contains("injected")),
            "{r:?}"
        );
        assert_eq!(b.threads.load(Ordering::Relaxed), 1);
        let r = b.join(
            MIN_JOIN_PIXELS,
            || Ok(()),
            || -> Result<(), Error> { panic!("injected in b") },
        );
        assert!(
            matches!(r, Err(Error::Internal(ref m)) if m.contains("injected in b")),
            "{r:?}"
        );
        assert_eq!(b.threads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_failing_band_is_an_error() {
        let b = Budget::new(&lim(1 << 20, 4));
        let r = b.par_rows(&mut vec![0u8; 4 << 20], 1024, |_, r0, _| {
            if r0 > 0 {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert_eq!(r, Err(Error::Cancelled));
        assert_eq!(b.threads.load(Ordering::Relaxed), 1);
    }
}
