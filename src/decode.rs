//! Decoding still WebP, and the first frame of animated files.

use crate::Error;
use crate::limits::{Budget, Limits, MAX_SIDE, cap_bytes, fit_zeroed};
use crate::riff::{self, File};
use crate::{vp8dec, vp8ldec};
use std::sync::atomic::AtomicBool;

/// What a WebP header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    /// The VP8X alpha flag or, for a bare VP8L image, its alpha-is-used bit. `decode` returns the
    /// alpha the stream codes.
    pub has_alpha: bool,
    pub lossless: bool,
    /// How many frames of an animated file `decode` leaves out: all but the first. 0 for a still
    /// image.
    pub frames_dropped: usize,
}

/// A decoded image: non-premultiplied RGBA, 4 bytes a pixel, rows `width * 4` bytes apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pix: Vec<u8>,
}

fn parse(data: &[u8]) -> Result<(File<'_>, Config), Error> {
    let mut f = riff::parse(data)?;
    let mut c = Config {
        width: 0,
        height: 0,
        has_alpha: f.alpha,
        lossless: f.lossless,
        frames_dropped: 0,
    };
    let (w, h) = if f.lossless {
        let (w, h, a) = vp8ldec::header(f.image).map_err(|_| Error::Format)?;
        c.has_alpha = c.has_alpha || a;
        (w, h)
    } else {
        vp8dec::header(f.image).map_err(|_| Error::Format)?
    };
    if w < 1 || h < 1 {
        return Err(Error::Format);
    }
    if f.frames == 0 {
        if f.width != 0 && (f.width != w || f.height != h) {
            return Err(Error::Format); // VP8X canvas and image disagree
        }
        f.fw = w;
        f.fh = h;
    } else if f.fw != w || f.fh != h {
        return Err(Error::Format); // ANMF and image disagree
    }
    c.width = w;
    c.height = h;
    if f.frames > 0 {
        c.width = f.width; // the canvas
        c.height = f.height;
        c.frames_dropped = f.frames - 1;
    }
    Ok((f, c))
}

impl Limits {
    fn check(&self, c: &Config) -> Result<(), Error> {
        let (w, h) = (c.width as u64, c.height as u64);
        if w > MAX_SIDE {
            return Err(Error::Width {
                got: w,
                max: MAX_SIDE,
            });
        }
        if h > MAX_SIDE {
            return Err(Error::Height {
                got: h,
                max: MAX_SIDE,
            });
        }
        if w * h > self.max_pixels {
            return Err(Error::MaxPixels {
                got: w * h,
                max: self.max_pixels,
            });
        }
        Ok(())
    }
}

/// Reads the image size from the headers, walking the chunks like [`decode`] does. It allocates
/// nothing and spawns no thread, so it enforces none of `limits` (as Go's `DecodeConfig`, it
/// reports the size of a file `decode` would refuse; check it against `Limits::max_pixels`
/// yourself).
///
/// # Errors
///
/// `Error::Format` or `Error::Truncated` if the container or the image header is invalid.
pub fn decode_config(data: &[u8], _limits: &Limits) -> Result<Config, Error> {
    parse(data).map(|(_, c)| c)
}

/// Decodes a still WebP, or the first frame of an animated one placed at its offset on a
/// transparent canvas of the file's size (as libwebp's `WebPAnimDecoder` does;
/// `Config::frames_dropped` counts the rest). The output is bit exact with dwebp: libwebp's
/// default fancy chroma upsampling, no dithering, alpha from an ALPH chunk with its filters and
/// compression, and the same for every `max_threads`.
///
/// Limits enforced: `max_pixels` and the side cap of 16383 (on the canvas); `max_alloc`, the peak
/// of the pixels (4 bytes each), the decoder's tables and, for a lossy image, its planes, each
/// charged before it is allocated and released when freed; `max_threads` and `pool`, with which a lossy image
/// decodes its ALPH chunk alongside the VP8 frame and converts bands of rows to RGBA in parallel.
/// `cancel` is checked once a pixel row, a macroblock row and a transform row, by every thread.
///
/// # Errors
///
/// `Error::Width`, `Error::Height`, `Error::MaxPixels` or `Error::MemoryBudget` for a limit,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled`, `Error::Internal` for a
/// panic on a pool thread, and `Error::Format`,
/// `Truncated` or the `Vp8*` and `PrefixCode` errors for a bad file.
pub fn decode(data: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Image, Error> {
    decode_in(data, limits, &Budget::new(limits), cancel)
}

fn decode_in(
    data: &[u8],
    limits: &Limits,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Image, Error> {
    let mut d = Decoder::new();
    d.decode_with(data, limits, budget, cancel)?;
    Ok(d.image)
}

/// A decoder that keeps its buffers between calls, so a series of decodes does not allocate,
/// clear and page in the pixels, the planes and the decoders' tables each time. The output of a
/// call never depends on the earlier calls: it equals [`decode`]'s and [`decode_planar`]'s.
///
/// The image of a call is valid until the next call, which reuses its pixels.
///
/// # Allocations
///
/// With `max_threads` 1, a call allocates nothing once a call with the same file (or one no
/// bigger in every buffer) has run: the pixels, the canvas of an animated file, the alpha plane,
/// the planes and tables of the VP8 and VP8L decoders. With more threads a lossy image with
/// alpha decodes its ALPH chunk on a pool thread, which allocates a task for it, and the
/// conversion to RGBA in bands allocates a little too. After an error some buffers may have been
/// dropped and are allocated again by the next call.
///
/// # Resource limits
///
/// `Limits` apply to each call as they do to [`decode`]: the peak working memory of the call stays
/// under `max_alloc`, and every buffer this decoder kept from earlier calls is charged at its
/// capacity at the start of the call, whether the call uses it or not (the kept buffers of a
/// big image count against a small call). A buffer that has to grow is charged at its new size
/// and the old size given back. If the kept buffers make a call exceed `max_alloc` they are
/// dropped and the call runs again from scratch, so a call fits whenever [`decode`] would. The
/// memory the decoder holds between calls is not charged to any call and is the caller's choice:
/// drop the decoder to free it.
///
/// ```
/// use std::sync::atomic::AtomicBool;
/// use webp::{Decoder, Limits, Options, Source};
///
/// # fn main() -> Result<(), webp::Error> {
/// let (w, h) = (16, 16);
/// let pix = vec![200u8; 4 * w * h];
/// let (limits, cancel) = (Limits::default(), AtomicBool::new(false));
/// let src = Source::Nrgba { pix: &pix, stride: 4 * w };
/// let file = webp::encode(w, h, &src, &Options::default(), &limits, &cancel)?;
/// let mut dec = Decoder::new();
/// let a = dec.decode(&file, &limits, &cancel)?.clone();
/// let b = dec.decode(&file, &limits, &cancel)?;
/// assert_eq!(&a, b);
/// # Ok(())
/// # }
/// ```
pub struct Decoder {
    /// The last RGBA image. Its pixels are the canvas for an animated file.
    image: Image,
    /// The first frame of an animated file before it is placed on the canvas.
    frame_pix: Vec<u8>,
    /// The ALPH plane.
    alpha: Vec<u8>,
    /// The last planar image; its planes and alpha plane go back to the buffers on the next call.
    planar: Planar,
    vp8: vp8dec::Scratch,
    vp8l: vp8ldec::Scratch,
}

impl Default for Decoder {
    fn default() -> Decoder {
        Decoder::new()
    }
}

impl Decoder {
    /// A decoder that holds nothing yet.
    pub fn new() -> Decoder {
        Decoder {
            image: Image {
                width: 0,
                height: 0,
                pix: Vec::new(),
            },
            frame_pix: Vec::new(),
            alpha: Vec::new(),
            planar: Planar {
                width: 0,
                height: 0,
                alpha: None,
                frame: vp8dec::Frame::default(),
            },
            vp8: vp8dec::Scratch::default(),
            vp8l: vp8ldec::Scratch::default(),
        }
    }

    /// Decodes like [`decode`], reusing the buffers of earlier calls. The image is valid until the
    /// next call.
    ///
    /// Limits enforced, as in [`decode`]: `max_pixels` and the side cap of 16383, `max_alloc` (the
    /// buffers this decoder kept are charged at their capacity, see the struct's "Resource
    /// limits"), `max_threads`, `pool` and `cancel`.
    ///
    /// # Errors
    ///
    /// As [`decode`]. After an error the decoder is still usable.
    pub fn decode(
        &mut self,
        data: &[u8],
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<&Image, Error> {
        self.retrying(limits, |d, budget| {
            d.decode_with(data, limits, budget, cancel)
        })?;
        Ok(&self.image)
    }

    /// Decodes like [`decode_planar`], reusing the buffers of earlier calls. The planes are valid
    /// until the next call.
    ///
    /// Limits enforced, as in [`decode_planar`]: `max_pixels` and the side cap of 16383,
    /// `max_alloc` (the buffers this decoder kept are charged at their capacity, see the struct's
    /// "Resource limits"), `max_threads`, `pool` and `cancel`.
    ///
    /// # Errors
    ///
    /// As [`decode_planar`]. After an error the decoder is still usable.
    pub fn decode_planar(
        &mut self,
        data: &[u8],
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<&Planar, Error> {
        self.retrying(limits, |d, budget| {
            d.planar_with(data, limits, budget, cancel)
        })?;
        Ok(&self.planar)
    }

    /// Runs `f` over a budget made from `limits`; if the kept buffers made it fail on
    /// `max_alloc`, drops them and runs it again as a fresh decoder would.
    fn retrying(
        &mut self,
        limits: &Limits,
        f: impl Fn(&mut Decoder, &Budget) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.recycle();
        let warm = self.kept_bytes() > 0;
        let r = f(self, &Budget::new(limits));
        if warm && matches!(r, Err(Error::MemoryBudget { .. })) {
            *self = Decoder::new();
            return f(self, &Budget::new(limits));
        }
        r
    }

    /// Takes the planes and the alpha plane of the last planar image back into the buffers.
    fn recycle(&mut self) {
        if let Some(a) = self.planar.alpha.take() {
            self.alpha = a;
        }
        if self.planar.frame.y.capacity() > 0 {
            self.vp8.frame = std::mem::take(&mut self.planar.frame);
        }
    }

    /// The bytes the buffers hold, at capacity.
    fn kept_bytes(&self) -> u64 {
        cap_bytes(&self.image.pix)
            + cap_bytes(&self.frame_pix)
            + cap_bytes(&self.alpha)
            + self.vp8.bytes()
            + self.vp8l.bytes()
    }

    /// `decode` over `budget`, leaving the image in `self.image`.
    fn decode_with(
        &mut self,
        data: &[u8],
        limits: &Limits,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.recycle();
        let (f, c) = parse(data)?;
        limits.check(&c)?;
        if !f.lossless && f.alph.is_some_and(bad_alph) {
            return Err(Error::Format);
        }
        budget.grow(self.kept_bytes())?;
        let (fw, fh) = (f.fw, f.fh);
        let Decoder {
            image,
            frame_pix,
            alpha,
            vp8,
            vp8l,
            ..
        } = self;
        image.width = c.width;
        image.height = c.height;
        if fw == c.width && fh == c.height {
            fit_zeroed(budget, &mut image.pix, 4 * fw * fh)?;
            return decode_frame(&f, &mut image.pix, alpha, vp8, vp8l, budget, cancel);
        }
        fit_zeroed(budget, frame_pix, 4 * fw * fh)?;
        decode_frame(&f, frame_pix, alpha, vp8, vp8l, budget, cancel)?;
        fit_zeroed(budget, &mut image.pix, 4 * c.width * c.height)?;
        for y in 0..fh {
            let d = 4 * ((f.y + y) * c.width + f.x);
            image.pix[d..d + 4 * fw].copy_from_slice(&frame_pix[4 * y * fw..4 * (y + 1) * fw]);
        }
        Ok(())
    }

    /// `decode_planar` over `budget`, leaving the planes in `self.planar`.
    fn planar_with(
        &mut self,
        data: &[u8],
        limits: &Limits,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        self.recycle();
        let (f, c) = parse(data)?;
        limits.check(&c)?;
        if f.frames > 0 {
            return Err(Error::Animated);
        }
        if f.lossless {
            return Err(Error::NotLossy);
        }
        if f.alph.is_some_and(bad_alph) {
            return Err(Error::Format);
        }
        budget.grow(self.kept_bytes())?;
        let Decoder {
            alpha,
            planar,
            vp8,
            vp8l,
            ..
        } = self;
        let n = c.width as u64 * c.height as u64;
        match f.alph {
            Some(a) => {
                fit_zeroed(budget, alpha, c.width * c.height)?;
                let plane = &mut alpha[..];
                budget.join(
                    n,
                    || read_alph(a, c.width, c.height, plane, vp8l, budget, cancel),
                    || vp8dec::decode(f.image, budget, cancel, vp8),
                )?;
                planar.alpha = Some(std::mem::take(alpha));
            }
            None => vp8dec::decode(f.image, budget, cancel, vp8)?,
        }
        planar.width = c.width;
        planar.height = c.height;
        planar.frame = std::mem::take(&mut vp8.frame);
        Ok(())
    }
}

/// Decodes the image of `f` to RGBA into `pix`.
fn decode_frame(
    f: &File<'_>,
    pix: &mut [u8],
    alpha: &mut Vec<u8>,
    vp8: &mut vp8dec::Scratch,
    vp8l: &mut vp8ldec::Scratch,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    if f.lossless {
        vp8ldec::decode(f.image, pix, budget, cancel, vp8l)
    } else {
        decode_lossy(
            f.image, f.alph, f.fw, f.fh, pix, alpha, vp8, vp8l, budget, cancel,
        )
    }
}

fn bad_alph(a: &[u8]) -> bool {
    a.len() < 2 || a[0] >> 6 != 0 || (a[0] >> 4) & 3 > 1 || a[0] & 3 > 1
}

/// Decodes a VP8 frame to YUV planes, then to RGBA in `pix`, with the alpha of an ALPH chunk if
/// there is one. The ALPH chunk and the frame decode on two threads if `budget` has them and the
/// image has at least `MIN_JOIN_PIXELS` pixels, and the conversion runs in bands of rows; both
/// stages take their workers from the pool the budget holds.
fn decode_lossy(
    data: &[u8],
    alph: Option<&[u8]>,
    w: usize,
    h: usize,
    pix: &mut [u8],
    alpha: &mut Vec<u8>,
    vp8: &mut vp8dec::Scratch,
    vp8l: &mut vp8ldec::Scratch,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    if let Some(a) = alph {
        fit_zeroed(budget, alpha, w * h)?;
        let plane = &mut alpha[..];
        budget.join(
            w as u64 * h as u64,
            || read_alph(a, w, h, plane, vp8l, budget, cancel),
            || vp8dec::decode(data, budget, cancel, vp8),
        )?;
    } else {
        vp8dec::decode(data, budget, cancel, vp8)?;
    }
    let (fr, alpha) = (&vp8.frame, &alpha[..]);
    budget.par_rows(pix, 4 * w, |band, r0, r1| {
        fr.rows(band, r0, r1, 4, cancel)?;
        if alph.is_some() {
            for (i, px) in band.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[r0 * w + i];
            }
        }
        Ok(())
    })
}

/// Reads an ALPH chunk (WebP container spec, "Alpha") into `plane`, one byte per pixel.
fn read_alph(
    alph: &[u8],
    w: usize,
    h: usize,
    plane: &mut [u8],
    sc: &mut vp8ldec::Scratch,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let filter = (alph[0] >> 2) & 3;
    if alph[0] & 3 == 0 {
        if alph.len() - 1 < w * h {
            return Err(Error::Truncated);
        }
        plane.copy_from_slice(&alph[1..1 + w * h]);
    } else {
        match vp8ldec::decode_alpha(&alph[1..], w, h, plane, budget, cancel, sc) {
            Ok(()) => {}
            Err(e @ (Error::Cancelled | Error::MemoryBudget { .. } | Error::OutOfMemory)) => {
                return Err(e);
            }
            Err(_) => return Err(Error::Format),
        }
    }
    unfilter(plane, w, h, filter);
    Ok(())
}

/// Undoes the alpha prediction filters (0 none, 1 horizontal, 2 vertical, 3 gradient) in place.
/// The first row, and the first pixel of every row but for the vertical and gradient filters,
/// predict from the left or are raw, as libwebp's HorizontalUnfilter, VerticalUnfilter and
/// GradientUnfilter do.
fn unfilter(a: &mut [u8], w: usize, h: usize, filter: u8) {
    if filter == 0 {
        return;
    }
    for y in 0..h {
        let (before, rest) = a.split_at_mut(y * w);
        let row = &mut rest[..w];
        let prev = if y > 0 {
            Some(&before[(y - 1) * w..])
        } else {
            None
        };
        let Some(prev) = prev.filter(|_| filter != 1) else {
            let mut pred = prev.map_or(0, |p| p[0]);
            for v in row.iter_mut() {
                *v = v.wrapping_add(pred);
                pred = *v;
            }
            continue;
        };
        if filter == 2 {
            for (v, &p) in row.iter_mut().zip(prev) {
                *v = v.wrapping_add(p);
            }
            continue;
        }
        let (mut left, mut tl) = (prev[0], prev[0]);
        for i in 0..w {
            let top = prev[i];
            let g = i32::from(left) + i32::from(top) - i32::from(tl);
            row[i] = row[i].wrapping_add(g.clamp(0, 255) as u8);
            left = row[i];
            tl = top;
        }
    }
}

/// A lossy image as the 4:2:0 planes the VP8 frame decodes to; see [`decode_planar`].
pub struct Planar {
    pub width: usize,
    pub height: usize,
    /// One byte a pixel, `width` per row; `None` if there is no ALPH chunk.
    pub alpha: Option<Vec<u8>>,
    frame: vp8dec::Frame,
}

impl Planar {
    /// The luma plane, `y_stride()` bytes a row, padded to a multiple of 16 rows.
    pub fn y(&self) -> &[u8] {
        &self.frame.y
    }

    pub fn cb(&self) -> &[u8] {
        &self.frame.u
    }

    pub fn cr(&self) -> &[u8] {
        &self.frame.v
    }

    pub fn y_stride(&self) -> usize {
        self.frame.y_stride
    }

    pub fn c_stride(&self) -> usize {
        self.frame.c_stride
    }

    /// Writes rows `y0` to `y1` (exclusive) as 3 bytes a pixel into `dst`, which must hold exactly
    /// those rows. The conversion is `decode`'s: libwebp's fancy upsampling, so a whole image in
    /// bands equals the one-shot RGBA.
    ///
    /// Limits enforced: `max_threads` and `pool`, with which a large band converts in parallel
    /// (the bytes are the same for any count). It allocates nothing, so `max_alloc` does not
    /// apply. `cancel` is checked once a row, by every thread.
    ///
    /// # Errors
    ///
    /// `Error::InvalidArgument` if the rows are out of range or `dst` is not exactly their size,
    /// `Error::Cancelled`, and `Error::Internal` for a panic on a pool thread.
    pub fn rgb(
        &self,
        dst: &mut [u8],
        y0: usize,
        y1: usize,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        if y0 > y1 || y1 > self.height || dst.len() != 3 * self.width * (y1 - y0) {
            return Err(Error::InvalidArgument(
                "rgb rows out of range or dst of the wrong size",
            ));
        }
        Budget::new(limits).par_rows(dst, 3 * self.width, |band, r0, r1| {
            self.frame.rows(band, y0 + r0, y0 + r1, 3, cancel)
        })
    }
}

/// Decodes a lossy still WebP to its Y, Cb and Cr planes, which take 1.5 bytes a pixel of the
/// `max_alloc` budget (the sides rounded up to 16) where `decode` takes 5.5. [`Planar::rgb`] then
/// converts a band of rows at a time, bit exact with `decode`, into memory the caller holds.
/// Alpha from an ALPH chunk is a 1 byte plane (2.5 bytes a pixel in all); a compressed one also
/// holds 4 bytes a pixel while it decodes. Check [`decode_config`] first.
///
/// Limits enforced: `max_pixels` and the side cap of 16383; `max_alloc`, the peak of the planes,
/// the alpha plane and the decoders' scratch, charged before they are allocated (the planes and
/// the alpha plane stay charged until return); `max_threads` and `pool`, with which the ALPH chunk
/// decodes alongside the VP8 frame. `cancel` is checked once a macroblock row and a transform row, by
/// every thread.
///
/// # Errors
///
/// `Error::NotLossy` for a lossless image, `Error::Animated` for an animated file,
/// `Error::Width`, `Error::Height`, `Error::MaxPixels` or `Error::MemoryBudget`,
/// `Error::OutOfMemory`, `Error::Cancelled`, `Error::Internal` for a panic on a pool thread, and
/// the format errors of [`decode`].
pub fn decode_planar(data: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Planar, Error> {
    let mut d = Decoder::new();
    d.planar_with(data, limits, &Budget::new(limits), cancel)?;
    Ok(d.planar)
}

/// Returns every EXIF and XMP chunk of a WebP file, still or animated, in file order. The bytes
/// alias `data`. A chunk whose declared size runs past the end of the file or the RIFF size is
/// returned cut there with `truncated` set. If the bytes of all the chunks together, plus 32 for
/// each chunk, are over `limits.max_metadata_bytes` it stops walking and returns a limit error
/// and no chunks. It allocates only the two slices of chunks, which that limit bounds and which
/// are charged against `max_alloc` as they grow, and spawns no thread, so `max_threads` and
/// `max_pixels` do not apply.
///
/// # Errors
///
/// `Error::MetadataBytes` or `Error::MemoryBudget`, and `Error::Format` or `Error::Truncated` for a
/// bad container.
pub fn metadata<'a>(data: &'a [u8], limits: &Limits) -> Result<riff::Meta<'a>, Error> {
    riff::metadata(data, limits)
}

#[cfg(test)]
mod tests {
    use super::{Budget, Limits, decode_in, unfilter};
    use crate::ThreadPool;
    use crate::testutil::read;
    use std::sync::atomic::AtomicBool;

    static NO: AtomicBool = AtomicBool::new(false);

    fn with(n: usize) -> Limits {
        Limits {
            max_threads: n,
            ..Limits::default()
        }
    }

    /// A lossy image with alpha holes of `w` by `h` pixels.
    fn lossy_alpha(w: usize, h: usize) -> Vec<u8> {
        let mut pix = vec![0u8; 4 * w * h];
        for (i, p) in pix.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (x, y) = (i % w, i / w);
            p.copy_from_slice(&[
                (x / 3) as u8,
                (y / 3) as u8,
                (x ^ y) as u8,
                (x / 20 * 40) as u8,
            ]);
        }
        let src = crate::Source::Nrgba {
            pix: &pix,
            stride: 4 * w,
        };
        crate::encode(w, h, &src, &crate::Options::default(), &with(1), &NO).unwrap()
    }

    /// A decode never has more threads live than `max_threads` (no other thread with 1 or 0), lossy
    /// with alpha uses a second one from `MIN_JOIN_PIXELS` pixels, on the shared pool or a
    /// caller's, and the pixels are the same for every count and pool.
    #[test]
    fn decode_threads_stay_within_max_threads() {
        let big = lossy_alpha(300, 230);
        const _: () = assert!(300 * 230 >= crate::limits::MIN_JOIN_PIXELS);
        for (name, data, alpha) in [
            (
                "lossy_alpha_photoa_raw.webp",
                read("lossy_alpha_photoa_raw.webp"),
                false,
            ),
            (
                "lossy_alpha_photoa_best.webp",
                read("lossy_alpha_photoa_best.webp"),
                false,
            ),
            ("300x230 lossy+alpha", big, true),
            ("he_768.webp", read("he_768.webp"), false),
            ("ll_photo_z4.webp", read("ll_photo_z4.webp"), false),
        ] {
            let mut want = None;
            for (n, own_pool) in [0usize, 1, 2, 3, 64]
                .into_iter()
                .flat_map(|n| [(n, false), (n, true)])
            {
                let l = Limits {
                    pool: own_pool.then(|| ThreadPool::new(3).unwrap()),
                    ..with(n)
                };
                let b = Budget::new(&l);
                let img = decode_in(&data, &l, &b, &NO).unwrap();
                assert!(
                    b.peak_threads() <= n.max(1),
                    "{name}: {} threads of {n}",
                    b.peak_threads()
                );
                if n <= 1 {
                    assert_eq!(b.peak_threads(), 1, "{name}");
                } else if alpha {
                    // The ALPH chunk takes a second thread, and the 276 KB of RGBA are up to four bands
                    // of 64 KiB (`MIN_BAND_BYTES`).
                    assert!(b.peak_threads() >= 2, "{name}");
                }
                match &want {
                    Some(w) => assert!(&img == w, "{name}, {n} threads, own pool {own_pool}"),
                    None => want = Some(img),
                }
            }
        }
    }

    /// With one thread the charges are deterministic: the peak the ledger saw is the smallest
    /// `max_alloc` that passes, so nothing is charged that is not held and nothing held is missed.
    #[test]
    fn decode_ledger_peak_is_the_smallest_budget() {
        for name in [
            "lossy_alpha_photoa_raw.webp",
            "lossy_alpha_photoa_best.webp",
            "he_768.webp",
            "ll_photo_z4.webp",
            "anim_lossy_full.webp",
        ] {
            let data = read(name);
            let l = with(1);
            let b = Budget::new(&l);
            decode_in(&data, &l, &b, &NO).unwrap();
            let peak = b.peak() as usize;
            let at = |m: usize| {
                let l = Limits {
                    max_alloc: m,
                    ..with(1)
                };
                decode_in(&data, &l, &Budget::new(&l), &NO)
            };
            assert!(at(peak).is_ok(), "{name}: peak {peak} does not pass");
            assert!(at(peak - 1).is_err(), "{name}: peak {peak} - 1 passes");
        }
    }

    /// Filters a random plane the way libwebp's encoder does and checks that `unfilter` gives it
    /// back, for every filter and odd shapes.
    #[test]
    fn unfilter_inverts_the_encoder_filters() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        };
        for (w, h) in [(1usize, 1usize), (1, 7), (7, 1), (13, 9)] {
            let want: Vec<u8> = (0..w * h).map(|_| next()).collect();
            for f in 1..4u8 {
                let mut got = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let p = if y == 0 && x == 0 {
                            0
                        } else if y == 0 || (f == 1 && x > 0) {
                            i32::from(want[y * w + x - 1])
                        } else if x == 0 || f == 2 {
                            i32::from(want[(y - 1) * w + x])
                        } else {
                            (i32::from(want[y * w + x - 1]) + i32::from(want[(y - 1) * w + x])
                                - i32::from(want[(y - 1) * w + x - 1]))
                            .clamp(0, 255)
                        };
                        got[y * w + x] = want[y * w + x].wrapping_sub(p as u8);
                    }
                }
                unfilter(&mut got, w, h, f);
                assert_eq!(got, want, "filter {f}, {w}x{h}: not inverted");
            }
        }
    }
}
