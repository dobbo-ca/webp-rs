//! The RIFF container: the walk Decode does, and the EXIF/XMP walk of `metadata`.

use crate::Error;
use crate::limits::{Budget, Ledger, Limits};
use std::sync::atomic::AtomicBool;

/// What Parse finds: the first image (or the first animation frame) of a file.
#[derive(Default)]
pub(crate) struct File<'a> {
    /// Image is a VP8L chunk, else VP8.
    pub lossless: bool,
    /// Canvas from VP8X, else 0 (the image's own size is in its header).
    pub width: usize,
    pub height: usize,
    /// The VP8X alpha flag; VP8L's own flag is in its header.
    pub alpha: bool,
    pub image: &'a [u8],
    /// Last ALPH chunk before a VP8 chunk, as libwebp reads it.
    pub alph: Option<&'a [u8]>,
    /// ANMF frames in the file; 0 for a still image.
    pub frames: usize,
    /// The first frame's offset and size on the canvas.
    pub x: usize,
    pub y: usize,
    pub fw: usize,
    pub fh: usize,
}

fn le32(b: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u24(b: &[u8], i: usize) -> usize {
    usize::from(b[i]) | usize::from(b[i + 1]) << 8 | usize::from(b[i + 2]) << 16
}

/// Walks the chunks of a still or animated file.
pub(crate) fn parse(data: &[u8]) -> Result<File<'_>, Error> {
    let mut f = File::default();
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(Error::Format);
    }
    let end = le32(&data[4..8]) + 8;
    if end > data.len() as u64 {
        return Err(Error::Truncated);
    }
    if end < 12 {
        return Err(Error::Format);
    }
    let data = &data[..end as usize];
    let (mut pos, mut vp8x, mut anim) = (12usize, false, false);
    while pos + 8 <= data.len() {
        let tag = &data[pos..pos + 4];
        let n = le32(&data[pos + 4..pos + 8]);
        if n > (data.len() - pos - 8) as u64 {
            return Err(Error::Truncated);
        }
        let n = n as usize;
        let body = &data[pos + 8..pos + 8 + n];
        pos += 8 + n + (n & 1);
        match tag {
            b"VP8X" => {
                if vp8x || n != 10 {
                    return Err(Error::Format);
                }
                vp8x = true;
                anim = body[0] & 0x02 != 0;
                f.alpha = body[0] & 0x10 != 0;
                f.width = 1 + u24(body, 4);
                f.height = 1 + u24(body, 7);
            }
            b"ALPH" => {
                if anim {
                    return Err(Error::Format); // an animation's ALPH lives inside its ANMF
                }
                if vp8x {
                    f.alph = Some(body);
                }
            }
            b"ANMF" => {
                if !anim {
                    return Err(Error::Format);
                }
                f.frames += 1;
                if f.frames > 1 {
                    let mut g = File {
                        width: f.width,
                        height: f.height,
                        ..File::default()
                    };
                    first_frame(body, &mut g)?;
                } else {
                    first_frame(body, &mut f)?;
                }
            }
            b"VP8 " | b"VP8L" => {
                if anim {
                    return Err(Error::Format);
                }
                f.lossless = tag == b"VP8L";
                f.image = body;
                if f.lossless {
                    f.alph = None; // ALPH belongs to VP8 only
                }
                return Ok(f);
            }
            _ => {}
        }
        if !vp8x && tag != b"VP8X" {
            return Err(Error::Format); // the first chunk is VP8, VP8L or VP8X
        }
    }
    if f.frames > 0 {
        return Ok(f);
    }
    Err(Error::Format)
}

/// Reads an ANMF body: the frame's offset and size (checked to lie in the canvas), then its ALPH
/// and VP8 or VP8L chunks.
fn first_frame<'a>(b: &'a [u8], f: &mut File<'a>) -> Result<(), Error> {
    if b.len() < 16 {
        return Err(Error::Format);
    }
    f.x = 2 * u24(b, 0);
    f.y = 2 * u24(b, 3);
    f.fw = 1 + u24(b, 6);
    f.fh = 1 + u24(b, 9);
    if f.x + f.fw > f.width || f.y + f.fh > f.height {
        return Err(Error::Format);
    }
    let mut pos = 16usize;
    while pos + 8 <= b.len() {
        let tag = &b[pos..pos + 4];
        let n = le32(&b[pos + 4..pos + 8]);
        if n > (b.len() - pos - 8) as u64 {
            return Err(Error::Truncated);
        }
        let n = n as usize;
        let body = &b[pos + 8..pos + 8 + n];
        pos += 8 + n + (n & 1);
        match tag {
            b"ALPH" => f.alph = Some(body),
            b"VP8 " | b"VP8L" => {
                f.lossless = tag == b"VP8L";
                f.image = body;
                if f.lossless {
                    f.alph = None;
                }
                return Ok(());
            }
            _ => {}
        }
    }
    Err(Error::Format)
}

/// What each collected metadata chunk is charged on top of its bytes: the Go `riff.Chunk` it takes
/// in a slice (24 bytes of slice header and a flag, padded to 32), so empty chunks cannot be
/// collected for free.
pub(crate) const CHUNK_COST: u64 = 32;

/// One EXIF or XMP chunk body. `data` holds only the bytes inside the file and the RIFF size;
/// `truncated` is set when the chunk's declared size is more than that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaChunk<'a> {
    pub data: &'a [u8],
    pub truncated: bool,
}

/// Every EXIF and XMP chunk of a file, each in file order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Meta<'a> {
    pub exif: Vec<MetaChunk<'a>>,
    pub xmp: Vec<MetaChunk<'a>>,
}

/// Walks all top-level chunks, past the image, and collects the EXIF and XMP ones. Each costs
/// `CHUNK_COST` plus its bytes; as soon as the total is over `max` the walk stops with a limit
/// error carrying the total so far. The chunk entries are charged against `max_alloc` as the two
/// vectors grow.
pub(crate) fn metadata<'a>(data: &'a [u8], limits: &Limits) -> Result<Meta<'a>, Error> {
    let budget = Budget::new(limits);
    let never = AtomicBool::new(false);
    let cx = Ledger::new(&budget, &never);
    walk(data, limits.max_metadata_bytes, &cx)
}

fn walk<'a>(data: &'a [u8], max: u64, cx: &Ledger) -> Result<Meta<'a>, Error> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(Error::Format);
    }
    let mut end = le32(&data[4..8]) + 8;
    if end < 12 {
        return Err(Error::Format);
    }
    end = end.min(data.len() as u64);
    let over = |got: u64| Error::MetadataBytes { got, max };
    let mut m = Meta::default();
    let mut got = 0u64;
    let mut pos = 12u64;
    while pos + 8 <= end {
        let p = pos as usize;
        let tag = &data[p..p + 4];
        let mut n = le32(&data[p + 4..p + 8]);
        pos += 8;
        let truncated = n > end - pos;
        if truncated {
            n = end - pos;
        }
        let chunk = MetaChunk {
            data: &data[pos as usize..(pos + n) as usize],
            truncated,
        };
        if tag == b"EXIF" || tag == b"XMP " {
            got += CHUNK_COST + n;
            if got > max {
                return Err(over(got));
            }
        }
        match tag {
            b"EXIF" => cx.push(&mut m.exif, chunk)?,
            b"XMP " => cx.push(&mut m.xmp, chunk)?,
            _ => {}
        }
        if truncated {
            return Ok(m);
        }
        pos += n + (n & 1);
    }
    // Under 8 bytes left: a header cut off after the start of an EXIF or XMP tag is that chunk,
    // cut, with no data.
    if pos < end {
        let t = &data[pos as usize..(pos + 4).min(end) as usize];
        for tag in [b"EXIF", b"XMP "] {
            if tag.starts_with(t) {
                got += CHUNK_COST;
                if got > max {
                    return Err(over(got));
                }
                let c = MetaChunk {
                    data: &[],
                    truncated: true,
                };
                if tag == b"EXIF" {
                    cx.push(&mut m.exif, c)?;
                } else {
                    cx.push(&mut m.xmp, c)?;
                }
            }
        }
    }
    Ok(m)
}
