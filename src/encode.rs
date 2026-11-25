//! The public encoder API: the options, the pixel sources and `encode`.

use crate::limits::{Limits, MAX_SIDE, check_cancel};
use crate::{Error, vp8enc};
use std::sync::atomic::AtomicBool;

/// Controls the encoder. `Options::default()` is Go's nil `*Options`: quality 75, method 4.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// 0 to 100; clamped. A NaN quality is kept as the Go package does: an image of one colour
    /// is unaffected, and any other encodes as at quality 100.
    pub quality: f32,
    /// 0 to 6 like `cwebp -m`; clamped.
    pub method: i32,
    /// Makes methods 3 to 6 decide and write exactly as libwebp 1.6.0 (and 1.2.2) does with
    /// default settings, so the output bytes equal `cwebp -q quality -m method`. It costs speed:
    /// every intra mode gets a full rate-distortion trial. Methods 0 to 2 ignore it. ALPH matches
    /// libwebp 1.2.2, or 1.6.0 with `alpha_libwebp16`.
    pub match_libwebp: bool,
    /// With `match_libwebp` at methods 3 to 5, writes ALPH as libwebp 1.6.0 does; the default is
    /// 1.2.2.
    pub alpha_libwebp16: bool,
    /// With `match_libwebp`, writes ALPH with webp-go's original writer: same decoded alpha, other
    /// bytes, about 4x faster.
    pub legacy_alpha: bool,
    /// 1 is one segment and no analysis pass, `cwebp -segments 1` with `match_libwebp` at methods
    /// 3 to 6. 0 and 4 are the default 4; any other value is an error.
    pub segments: i32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            quality: 75.0,
            method: 4,
            match_libwebp: false,
            alpha_libwebp16: false,
            legacy_alpha: false,
            segments: 0,
        }
    }
}

/// The pixels to encode. Rows are `stride` bytes apart; the image size is given to [`encode`].
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// 4 bytes a pixel, alpha-premultiplied (Go `image.RGBA`).
    Rgba { pix: &'a [u8], stride: usize },
    /// 4 bytes a pixel, not premultiplied (Go `image.NRGBA`).
    Nrgba { pix: &'a [u8], stride: usize },
    /// 1 byte a pixel (Go `image.Gray`).
    Gray { pix: &'a [u8], stride: usize },
    /// 4:2:0 planes, as decoded by [`Planar`](crate::Planar) (Go `image.YCbCr`); the chroma
    /// planes hold `(width + 1) / 2` by `(height + 1) / 2` samples.
    Ycbcr420 {
        y: &'a [u8],
        cb: &'a [u8],
        cr: &'a [u8],
        y_stride: usize,
        c_stride: usize,
    },
}

/// The bytes a plane of `rows` rows of `row` samples, `stride` apart, needs.
fn need(rows: usize, row: usize, stride: usize) -> Option<usize> {
    (rows - 1).checked_mul(stride)?.checked_add(row)
}

/// Checks that the planes hold the image.
fn check_source(src: &Source, w: usize, h: usize) -> Result<(), Error> {
    let short = Error::InvalidArgument("pixel plane shorter than the image");
    let ok = |len: usize, row: usize, stride: usize, rows: usize| {
        stride >= row && need(rows, row, stride).is_some_and(|n| len >= n)
    };
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let good = match *src {
        Source::Rgba { pix, stride } | Source::Nrgba { pix, stride } => {
            ok(pix.len(), 4 * w, stride, h)
        }
        Source::Gray { pix, stride } => ok(pix.len(), w, stride, h),
        Source::Ycbcr420 {
            y,
            cb,
            cr,
            y_stride,
            c_stride,
        } => {
            ok(y.len(), w, y_stride, h)
                && ok(cb.len(), cw, c_stride, ch)
                && ok(cr.len(), cw, c_stride, ch)
        }
    };
    if good { Ok(()) } else { Err(short) }
}

/// Encodes a `width` by `height` image to a WebP file: lossy VP8, with a lossless ALPH chunk when
/// the image has alpha. The output does not depend on the machine or on `max_threads`.
///
/// Limits enforced: `max_pixels` and the side cap of 16383; `max_alloc`, the peak of the planes,
/// the per-macroblock arrays, each worker's token buffer, the partitions, the ALPH encoder's
/// buffers and the output, each charged before it is allocated (the ALPH encoder's one by one, as
/// it allocates them) and released when freed; `max_threads` and `pool`: the VP8 frame's analysis
/// and macroblock rows run on up to `max_threads` threads (the rows as a wavefront, each row two
/// macroblocks behind the one above, for an image of 64 macroblocks or more), and the ALPH chunk
/// of an image with alpha is encoded on one of them while the VP8 frame is. `cancel` is checked
/// once a macroblock row by the VP8 encoder and once a row, a pass or a tile row by the ALPH
/// encoder, on whichever thread it runs, so a set flag ends the call promptly with every
/// `max_threads`.
///
/// This is `Encoder::new().encode(..)`. To encode many images, keep one [`Encoder`] and call it
/// for each.
///
/// # Errors
///
/// `Error::InvalidArgument` for sides outside 1 to 16383, a plane shorter than the image or
/// `segments` other than 0, 1 and 4; `Error::MaxPixels` or `Error::MemoryBudget` for a limit;
/// `Error::OutOfMemory` if the allocator refuses; `Error::Cancelled`; `Error::Internal` for a
/// panic on a pool thread; and `Error::Unsupported` for a partition too large to write.
pub fn encode(
    width: usize,
    height: usize,
    src: &Source,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, Error> {
    Encoder::new().encode(width, height, src, opts, limits, cancel)
}

/// An encoder that keeps its working buffers between calls (Go's `webp.Encoder`), so a series of
/// encodes does not allocate, clear and page in the planes, the token buffer and the
/// partitions each time. The output of a call never depends on the earlier calls: it equals
/// [`encode`]'s.
///
/// # Allocations
///
/// With `max_threads` 1, [`Encoder::encode_into`] allocates nothing once a call of the same size
/// has run (two calls for an image with alpha, as the ALPH encoder swaps the roles of some
/// buffers), apart from the 1.6.0 ALPH writer (`alpha_libwebp16`), which makes its own scratch
/// in each call. With more threads a call allocates a little each time: the task of the ALPH
/// chunk, the wavefront workers and the analysis bands. [`Encoder::encode`] also allocates the returned `Vec`.
///
/// # Resource limits
///
/// `Limits` apply to each call as they do to [`encode`]: the peak working memory of the call,
/// including the buffers this encoder kept from earlier calls and the call uses (the planes, the
/// per-macroblock arrays, the token buffers, the partitions and the libwebp-matching state, each
/// charged at its capacity before it is used, the ALPH encoder's scratch too), stays under
/// `max_alloc`. The memory the encoder holds between calls is not charged to any call and is the
/// caller's choice: it is about the working set of the largest image encoded, and drop the
/// encoder to free it. The planes and the per-macroblock arrays are dropped rather than reused
/// when they are more than 2x too big for the call; the token buffers, the partitions and the
/// ALPH scratch keep their largest capacity and are charged whole, and one too small for the call
/// is charged at its new size while it grows. The libwebp-matching state has a fixed size and is
/// always kept and reset. If the kept buffers make a call exceed `max_alloc` they are dropped and
/// the call runs again from scratch, so with `max_threads` 1 a call fits whenever [`encode`]
/// would. With 2 or more threads the macroblock rows and the ALPH chunk charge the budget at the
/// same time, so a call that fits at `max_threads` 1 can fail at 2, and the outcome of a call whose
/// peak is near `max_alloc` depends on how the threads interleave, for `Encoder` as for
/// [`encode`]; see [`Limits::max_alloc`].
///
/// # Errors
///
/// The same as [`encode`].
///
/// ```
/// use std::sync::atomic::AtomicBool;
/// use webp::{Encoder, Limits, Options, Source};
///
/// # fn main() -> Result<(), webp::Error> {
/// let (w, h) = (16, 16);
/// let pix = vec![200u8; 4 * w * h];
/// let (limits, cancel) = (Limits::default(), AtomicBool::new(false));
/// let src = Source::Nrgba { pix: &pix, stride: 4 * w };
/// let mut enc = Encoder::new();
/// let a = enc.encode(w, h, &src, &Options::default(), &limits, &cancel)?;
/// let b = enc.encode(w, h, &src, &Options::default(), &limits, &cancel)?;
/// assert_eq!(a, b);
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct Encoder {
    bufs: vp8enc::Bufs,
}

impl Encoder {
    /// An encoder that holds nothing yet.
    pub fn new() -> Encoder {
        Encoder::default()
    }

    /// Encodes a `width` by `height` image like [`encode`], reusing the buffers of earlier calls.
    ///
    /// Limits enforced, as in [`encode`]: `max_pixels` and the side cap of 16383, `max_alloc` (the
    /// buffers this encoder kept are charged at their capacity, see the struct's "Resource
    /// limits"), `max_threads`, `pool` and `cancel`.
    ///
    /// # Errors
    ///
    /// As [`encode`]: `Error::InvalidArgument` for sides outside 1 to 16383, a plane shorter than
    /// the image or `segments` other than 0, 1 and 4; `Error::MaxPixels` or `Error::MemoryBudget`;
    /// `Error::OutOfMemory` if the allocator refuses, `Error::Internal` for a panic on a pool thread;
    /// `Error::Cancelled`; and `Error::Unsupported` for a partition too large to write. After an
    /// error the encoder is still usable.
    pub fn encode(
        &mut self,
        width: usize,
        height: usize,
        src: &Source,
        opts: &Options,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        self.encode_into(width, height, src, opts, limits, cancel, &mut out)?;
        Ok(out)
    }

    /// Encodes like [`Encoder::encode`], but writes the file into `out`, which is cleared first
    /// and keeps its capacity, so a caller that passes the same `Vec` each time allocates nothing
    /// for the output. `out` is charged against `max_alloc` at the size of the file, as the
    /// returned `Vec` of `encode` is. After an error `out` is empty.
    ///
    /// Limits enforced, as in [`encode`]: `max_pixels` and the side cap of 16383, `max_alloc` (the
    /// buffers this encoder kept are charged at their capacity, see the struct's "Resource
    /// limits"), `max_threads`, `pool` and `cancel`.
    ///
    /// # Errors
    ///
    /// The same as [`Encoder::encode`].
    pub fn encode_into(
        &mut self,
        width: usize,
        height: usize,
        src: &Source,
        opts: &Options,
        limits: &Limits,
        cancel: &AtomicBool,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        out.clear();
        if !(1..=MAX_SIDE as usize).contains(&width) || !(1..=MAX_SIDE as usize).contains(&height) {
            return Err(Error::InvalidArgument("dimensions out of range"));
        }
        let px = width as u64 * height as u64;
        if px > limits.max_pixels {
            return Err(Error::MaxPixels {
                got: px,
                max: limits.max_pixels,
            });
        }
        if !matches!(opts.segments, 0 | 1 | 4) {
            return Err(Error::InvalidArgument("Segments must be 0, 1 or 4"));
        }
        check_source(src, width, height)?;
        check_cancel(cancel)?;
        let warm = self.bufs.warm();
        let r = vp8enc::encode(
            width,
            height,
            src,
            opts,
            limits,
            cancel,
            &mut self.bufs,
            out,
        );
        if warm && matches!(r, Err(Error::MemoryBudget { .. })) {
            // The kept buffers may be what did not fit: try again as a fresh encoder would.
            self.bufs = vp8enc::Bufs::default();
            return vp8enc::encode(
                width,
                height,
                src,
                opts,
                limits,
                cancel,
                &mut self.bufs,
                out,
            );
        }
        r
    }
}
