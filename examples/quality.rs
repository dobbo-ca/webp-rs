//! Encodes a directory of images and decodes the results: `quality <images> <webp-out> <dec-out>`
//! encodes every `.ppm` and `.pgm` of the user-provided images directory under the five bench settings (quality 75, segments 0: plain
//! methods 0, 4 and 6, match_libwebp methods 4 and 6; the tags `m0 m4 m6 lw4 lw6`) with one reused
//! `Encoder`, writes `<webp-out>/<stem>_<tag>.webp`, decodes it and writes
//! `<dec-out>/<stem>_<tag>.ppm`. An error writes `<stem>_<tag>.err` with its class token. It is
//! `oracle -qenc` of `tests/parity/go/oracle`; `diff -r` of the two output trees compares them.

mod common;

use common::{MAX_PIXELS, class, limits};
use std::sync::atomic::AtomicBool;
use std::{env, fs, process};
use webp::{Options, Source};

/// The settings: tag, method, match_libwebp.
const SETTINGS: [(&str, i32, bool); 5] = [
    ("m0", 0, false),
    ("m4", 4, false),
    ("m6", 6, false),
    ("lw4", 4, true),
    ("lw6", 6, true),
];

fn must<T, E: std::fmt::Display>(r: Result<T, E>) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("quality: {e}");
        process::exit(1)
    })
}

/// A cell: gray samples, or NRGBA with alpha 255.
struct Cell {
    w: usize,
    h: usize,
    gray: bool,
    pix: Vec<u8>,
}

/// The header's separators, as in qenc.go: space, tab, CR and LF (not form feed).
fn sep(c: &u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n')
}

fn token<'a>(d: &'a [u8], pos: &mut usize) -> &'a [u8] {
    while d.get(*pos).is_some_and(sep) {
        *pos += 1;
    }
    let s = *pos;
    while *pos - s < 16 && d.get(*pos).is_some_and(|c| !sep(c)) {
        *pos += 1;
    }
    &d[s..*pos]
}

/// A plain decimal number, as strconv.Atoi in qenc.go reads it (a leading '+' too; a '-' fails
/// here and gives a side below 1 there, bad input either way).
fn num(t: &[u8]) -> Option<usize> {
    std::str::from_utf8(t).ok()?.parse().ok()
}

/// Reads a binary PGM (P5) or PPM (P6) with maxval 255; `None` is Go's `bad-input`. It accepts
/// exactly the headers qenc.go's pnmImage does: no comments, the separators of `sep`.
fn pnm(d: &[u8]) -> Option<Cell> {
    let mut pos = 0;
    let magic = token(d, &mut pos);
    let w = num(token(d, &mut pos))?;
    let h = num(token(d, &mut pos))?;
    let max = num(token(d, &mut pos))?;
    pos += 1; // the single whitespace byte after maxval
    let ch = match magic {
        b"P5" => 1,
        b"P6" => 3,
        _ => return None,
    };
    if max != 255 || w < 1 || h < 1 || w > 1 << 16 || h > 1 << 16 || (w * h) as u64 > MAX_PIXELS {
        return None;
    }
    let pix = d.get(pos..)?;
    if pix.len() != ch * w * h {
        return None;
    }
    if ch == 1 {
        return Some(Cell {
            w,
            h,
            gray: true,
            pix: pix.to_vec(),
        });
    }
    let mut rgba = Vec::with_capacity(4 * w * h);
    for p in pix.as_chunks::<3>().0 {
        rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    Some(Cell {
        w,
        h,
        gray: false,
        pix: rgba,
    })
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let [_, cells, webp_out, dec_out] = a.as_slice() else {
        eprintln!("usage: quality <images> <webp-out> <dec-out>");
        process::exit(2);
    };
    must(fs::create_dir_all(webp_out));
    must(fs::create_dir_all(dec_out));
    let mut names: Vec<String> = Vec::new();
    for e in must(fs::read_dir(cells)) {
        let n = must(e).file_name().to_string_lossy().into_owned();
        if n.ends_with(".ppm") || n.ends_with(".pgm") {
            names.push(n);
        }
    }
    names.sort();
    let limits = limits();
    let cancel = AtomicBool::new(false);
    let mut enc = webp::Encoder::new();
    for name in &names {
        let stem = name.rsplit_once('.').map_or(name.as_str(), |(s, _)| s);
        let cell = pnm(&must(fs::read(format!("{cells}/{name}"))));
        for (tag, method, match_libwebp) in SETTINGS {
            let base = format!("{stem}_{tag}");
            let opts = Options {
                quality: 75.0,
                method,
                match_libwebp,
                ..Options::default()
            };
            let r = match &cell {
                None => Err("bad-input".to_string()),
                Some(c) => {
                    let src = if c.gray {
                        Source::Gray {
                            pix: &c.pix[..],
                            stride: c.w,
                        }
                    } else {
                        Source::Nrgba {
                            pix: &c.pix[..],
                            stride: 4 * c.w,
                        }
                    };
                    enc.encode(c.w, c.h, &src, &opts, &limits, &cancel)
                        .map_err(class)
                }
            };
            let bytes = match r {
                Ok(b) => b,
                Err(t) => {
                    must(fs::write(
                        format!("{webp_out}/{base}.err"),
                        format!("{t}\n"),
                    ));
                    continue;
                }
            };
            must(fs::write(format!("{webp_out}/{base}.webp"), &bytes));
            match webp::decode(&bytes, &limits, &cancel) {
                Ok(i) => {
                    let mut b = format!("P6\n{} {}\n255\n", i.width, i.height).into_bytes();
                    for p in i.pix.as_chunks::<4>().0 {
                        b.extend_from_slice(&p[..3]);
                    }
                    must(fs::write(format!("{dec_out}/{base}.ppm"), b));
                }
                Err(e) => must(fs::write(
                    format!("{dec_out}/{base}.err"),
                    format!("{}\n", class(e)),
                )),
            }
        }
    }
}
