//! Shared by `oracle` and `bench`: the `.enc` input of tests/parity/go/oracle (its format is in the
//! comment of `encFormat` in `tests/parity/go/oracle/main.go`) and the limits both programs use.
#![allow(dead_code)]

use std::sync::atomic::AtomicBool;
use webp::{Error, Limits, Options, Source};

/// Caps every decode and encode of the oracles; the Go oracle's `maxPixels`.
pub const MAX_PIXELS: u64 = 1 << 22;

pub fn limits() -> Limits {
    Limits {
        max_pixels: MAX_PIXELS,
        ..Limits::default()
    }
}

/// The class token of an error, as the Go oracle prints it.
pub fn class(e: Error) -> String {
    e.class().to_string()
}

/// FNV-1a 64 and the length of `b`, as a line: `digest` of the Go oracle.
pub fn digest(b: &[u8]) -> Vec<u8> {
    let h = b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
        (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{h:016x} {}\n", b.len()).into_bytes()
}

/// One parsed `.enc` input.
pub struct Enc<'a> {
    kind: u8,
    method: i32,
    flags: u8,
    segments: i32,
    quality: f32,
    pub width: usize,
    pub height: usize,
    pix: &'a [u8],
}

/// Parses a `.enc` input; `None` is Go's `bad-input`.
pub fn parse_enc(d: &[u8]) -> Option<Enc<'_>> {
    let (head, pix) = d.split_first_chunk::<12>()?;
    let width = usize::from(u16::from_le_bytes([head[8], head[9]]));
    let height = usize::from(u16::from_le_bytes([head[10], head[11]]));
    if (width * height) as u64 > MAX_PIXELS {
        return None;
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let need = match head[0] {
        0 | 1 => 4 * width * height,
        2 => width * height,
        3 => width * height + 2 * cw * ch,
        _ => return None,
    };
    if pix.len() != need {
        return None;
    }
    Some(Enc {
        kind: head[0],
        method: i32::from(head[1] as i8),
        flags: head[2],
        segments: i32::from(head[3] as i8),
        quality: f32::from_le_bytes([head[4], head[5], head[6], head[7]]),
        width,
        height,
        pix,
    })
}

/// Encodes the input with `webp::encode`.
pub fn encode(c: &Enc, l: &Limits) -> Result<Vec<u8>, String> {
    encode_with(&mut webp::Encoder::new(), c, l)
}

/// Encodes the input with `e`, which keeps its buffers for the next call.
pub fn encode_with(e: &mut webp::Encoder, c: &Enc, l: &Limits) -> Result<Vec<u8>, String> {
    let (w, h) = (c.width, c.height);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let src = match c.kind {
        0 => Source::Nrgba {
            pix: c.pix,
            stride: 4 * w,
        },
        1 => Source::Rgba {
            pix: c.pix,
            stride: 4 * w,
        },
        2 => Source::Gray {
            pix: c.pix,
            stride: w,
        },
        _ => Source::Ycbcr420 {
            y: &c.pix[..w * h],
            cb: &c.pix[w * h..w * h + cw * ch],
            cr: &c.pix[w * h + cw * ch..],
            y_stride: w,
            c_stride: cw,
        },
    };
    let o = if c.flags & 8 != 0 {
        Options::default()
    } else {
        Options {
            quality: c.quality,
            method: c.method,
            match_libwebp: c.flags & 1 != 0,
            alpha_libwebp16: c.flags & 2 != 0,
            legacy_alpha: c.flags & 4 != 0,
            segments: c.segments,
        }
    };
    e.encode(w, h, &src, &o, l, &AtomicBool::new(false))
        .map_err(class)
}
