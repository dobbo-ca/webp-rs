//! Ports of webp-go's alpha encode tests (internal/webp/alpha_test.go) and
//! TestDecodePlanarCompressedAlphaBudget.
#![allow(clippy::needless_range_loop)]

mod common;

use common::pcg::Pcg;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use webp::{Error, Limits, Options, Source};

static NO_CANCEL: AtomicBool = AtomicBool::new(false);

fn enc(w: usize, h: usize, src: &Source, o: &Options) -> Vec<u8> {
    webp::encode(w, h, src, o, &Limits::default(), &NO_CANCEL).unwrap()
}

/// An NRGBA tile with transparent holes and soft alpha edges.
fn holes(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut r = Pcg::new(seed, 1);
    let mut m = vec![0u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            let a = match (x / 20 + y / 13) % 5 {
                0 => 0,
                1 => (x * 7) as u8,
                _ => 255,
            };
            let o = 4 * (y * w + x);
            m[o..o + 4].copy_from_slice(&[r.uint32() as u8, x as u8, y as u8, a]);
        }
    }
    m
}

/// Checks the file is VP8X then ALPH, and returns the decoded alpha plane.
fn decode_alpha(b: &[u8], w: usize, h: usize) -> Vec<u8> {
    assert_eq!(&b[12..16], b"VP8X");
    assert_eq!(&b[30..34], b"ALPH");
    let p = webp::decode_planar(b, &Limits::default(), &NO_CANCEL).unwrap();
    assert_eq!((p.width, p.height), (w, h));
    p.alpha.unwrap()
}

fn alph_chunk(b: &[u8], w: usize, h: usize) -> Vec<u8> {
    decode_alpha(b, w, h);
    let n = u32::from_le_bytes([b[34], b[35], b[36], b[37]]) as usize;
    b[38..38 + n].to_vec()
}

#[test]
fn encode_alpha() {
    for (w, h) in [(1, 1), (7, 5), (33, 17), (770, 770), (101, 250)] {
        let src = holes(w, h, w as u64);
        let b = enc(
            w,
            h,
            &Source::Nrgba {
                pix: &src,
                stride: 4 * w,
            },
            &Options {
                quality: 90.0,
                method: 4,
                ..Options::default()
            },
        );
        let a = decode_alpha(&b, w, h);
        for i in 0..w * h {
            assert_eq!(a[i], src[4 * i + 3], "{w}x{h}: pixel {i}");
        }
    }
}

/// A clear tile stays one colour with alpha 0.
#[test]
fn encode_solid_alpha() {
    for a in [0u8, 1, 128, 254] {
        let mut m = vec![0u8; 4 * 770 * 770];
        for p in m.as_chunks_mut::<4>().0.iter_mut() {
            p.copy_from_slice(&[10, 200, 90, a]);
        }
        let b = enc(
            770,
            770,
            &Source::Nrgba {
                pix: &m,
                stride: 4 * 770,
            },
            &Options {
                quality: 90.0,
                method: 4,
                ..Options::default()
            },
        );
        let al = decode_alpha(&b, 770, 770);
        assert!(al.iter().all(|&v| v == a), "alpha {a}");
    }
}

/// Premultiplied RGBA keeps its alpha; opaque NRGBA stays a simple VP8 file.
#[test]
fn encode_alpha_kinds() {
    let src = holes(64, 48, 3);
    let mut pre = src.clone();
    for p in pre.as_chunks_mut::<4>().0.iter_mut() {
        let a = u32::from(p[3]);
        for c in &mut p[..3] {
            let c16 = u32::from(*c) * 0x101;
            *c = ((c16 * a / 0xff) >> 8) as u8;
        }
    }
    let b = enc(
        64,
        48,
        &Source::Rgba {
            pix: &pre,
            stride: 256,
        },
        &Options::default(),
    );
    let a = decode_alpha(&b, 64, 48);
    for i in 0..64 * 48 {
        assert_eq!(a[i], src[4 * i + 3], "RGBA pixel {i}");
    }

    let mut op = holes(64, 48, 3);
    for p in op.as_chunks_mut::<4>().0.iter_mut() {
        p[3] = 255;
    }
    let b = enc(
        64,
        48,
        &Source::Nrgba {
            pix: &op,
            stride: 256,
        },
        &Options::default(),
    );
    assert_eq!(&b[12..16], b"VP8 ");
}

/// Segments changes only VP8: ALPH stays, 4 is 0, others fail.
#[test]
fn segments_alpha_and_errors() {
    let src = holes(101, 250, 5);
    let s = Source::Nrgba {
        pix: &src,
        stride: 404,
    };
    let e = |match_libwebp, segments| {
        enc(
            101,
            250,
            &s,
            &Options {
                quality: 90.0,
                method: 4,
                match_libwebp,
                segments,
                ..Options::default()
            },
        )
    };
    for m in [false, true] {
        let b0 = e(m, 0);
        assert_eq!(
            alph_chunk(&b0, 101, 250),
            alph_chunk(&e(m, 1), 101, 250),
            "match {m}: Segments 1 changed ALPH"
        );
        assert_eq!(b0, e(m, 4), "match {m}: Segments 4 differs from 0");
    }
    for n in [-1, 2, 3, 5] {
        let o = Options {
            segments: n,
            ..Options::default()
        };
        assert!(matches!(
            webp::encode(101, 250, &s, &o, &Limits::default(), &NO_CANCEL),
            Err(Error::InvalidArgument(_))
        ));
    }
}

/// The ALPH of the default path is compressed, and DecodePlanar fits its planes, the alpha plane and
/// the decoder's ring and tables in the budget.
#[test]
fn decode_planar_compressed_alpha_budget() {
    let (w, h) = (1500usize, 1500usize);
    let mut src = vec![0u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            let a = match (x / 40 + y / 30) % 4 {
                0 => 0,
                1 => (x / 3 + y / 5) as u8,
                _ => 255,
            };
            let o = 4 * (y * w + x);
            src[o..o + 4].copy_from_slice(&[(x / 6) as u8, (y / 6) as u8, (x + y) as u8, a]);
        }
    }
    let b = enc(
        w,
        h,
        &Source::Nrgba {
            pix: &src,
            stride: 4 * w,
        },
        &Options {
            quality: 75.0,
            method: 0,
            ..Options::default()
        },
    );
    assert_ne!(b[12 + 4 + 4 + 10 + 4 + 4] & 3, 0, "ALPH is raw");
    // Planes and alpha, a ring of 4 * ring_px and 2 MiB for the tables and rows.
    let planes = ((w + 15) & !15) * ((h + 15) & !15) * 3 / 2;
    let l = Limits {
        max_alloc: planes + w * h + 4 * ((1 << 20) + (1 << 15)) + (2 << 20),
        ..Limits::default()
    };
    let p = webp::decode_planar(&b, &l, &NO_CANCEL).unwrap();
    let a = p.alpha.unwrap();
    for i in 0..w * h {
        assert_eq!(a[i], src[4 * i + 3], "pixel {i}");
    }
}

/// A flag set from another thread while a large ALPH encode runs ends the call with
/// `Error::Cancelled` soon after, with one thread and with two (where the join waits for the ALPH
/// thread, which used to run to the end), for both libwebp ALPH writers.
#[test]
fn cancel_stops_the_alph_encode() {
    let (w, h) = (2048, 1024);
    let mut r = Pcg::new(3, 4);
    let pix: Vec<u8> = (0..4 * w * h).map(|_| r.uint32() as u8).collect();
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    for threads in [1, 2] {
        for alpha_libwebp16 in [false, true] {
            let o = Options {
                match_libwebp: true,
                method: 5,
                alpha_libwebp16,
                ..Options::default()
            };
            let l = Limits {
                max_threads: threads,
                ..Limits::default()
            };
            let cancel = AtomicBool::new(false);
            let (res, took) = std::thread::scope(|s| {
                s.spawn(|| {
                    std::thread::sleep(Duration::from_millis(100));
                    cancel.store(true, Ordering::Relaxed);
                });
                let t = Instant::now();
                let res = webp::encode(w, h, &src, &o, &l, &cancel);
                (res, t.elapsed())
            });
            assert_eq!(
                res.err(),
                Some(Error::Cancelled),
                "{threads} threads, alpha_libwebp16 {alpha_libwebp16}"
            );
            assert!(
                took < Duration::from_secs(3),
                "{threads} threads, alpha_libwebp16 {alpha_libwebp16}: took {took:?} to stop"
            );
        }
    }
}
