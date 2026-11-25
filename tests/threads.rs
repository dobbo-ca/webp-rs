//! The output does not depend on `max_threads` or on the pool: 0, 1, 2 and the default, on the
//! shared pool or a caller's, give the same bytes for every call that can use a second thread
//! (the ALPH chunk from 4K pixels, bands of RGB from 128 KiB of output, the encoder's MB rows
//! from 64 macroblocks).

mod common;

use common::read;
use std::sync::atomic::AtomicBool;
use webp::{Limits, Options, Source, ThreadPool};

static NO: AtomicBool = AtomicBool::new(false);

fn counts() -> Vec<usize> {
    vec![0, 1, 2, Limits::default().max_threads]
}

fn with(n: usize) -> Limits {
    Limits {
        max_threads: n,
        ..Limits::default()
    }
}

/// `with(n)` once on the shared pool and once on a pool of the caller's.
fn each(n: usize) -> [Limits; 2] {
    [
        with(n),
        Limits {
            pool: Some(ThreadPool::new(3).unwrap()),
            ..with(n)
        },
    ]
}

/// A textured NRGBA image with alpha holes and soft edges.
fn image(w: usize, h: usize) -> Vec<u8> {
    let mut pix = vec![0u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            let a = match (x / 20 + y / 13) % 5 {
                0 => 0,
                1 => (x * 7) as u8,
                _ => 255,
            };
            let n = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 7;
            let o = 4 * (y * w + x);
            pix[o..o + 4].copy_from_slice(&[(x / 3 + n) as u8, (y / 3) as u8, (x ^ y) as u8, a]);
        }
    }
    pix
}

/// A lossy file big enough (over 1 MiB of RGBA) for the RGB conversion to split into bands, with
/// or without an ALPH chunk.
fn big_lossy(alpha: bool) -> Vec<u8> {
    let (w, h) = (1024, 700);
    let mut pix = image(w, h);
    if !alpha {
        for a in pix.iter_mut().skip(3).step_by(4) {
            *a = 255;
        }
    }
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    webp::encode(w, h, &src, &Options::default(), &with(2), &NO).unwrap()
}

fn check_decode(name: &str, data: &[u8]) {
    let want = webp::decode(data, &with(1), &NO).unwrap_or_else(|e| panic!("{name}: {e}"));
    for n in counts() {
        for l in each(n) {
            let got = webp::decode(data, &l, &NO).unwrap();
            assert!(got == want, "{name}, {n} threads: pixels differ");
        }
    }
}

#[test]
fn decode_is_the_same_for_every_thread_count() {
    for name in [
        "lossy_alpha_photoa_raw.webp",
        "lossy_alpha_photoa_best.webp",
        "lossy_alpha_vfilter.webp",
        "he_768.webp",
        "ll_photoa_z4.webp",
        "anim_lossy_full.webp",
    ] {
        check_decode(name, &read(name));
    }
    check_decode("big lossy", &big_lossy(false));
    check_decode("big lossy+alpha", &big_lossy(true));
}

#[test]
fn planar_is_the_same_for_every_thread_count() {
    let files = [
        ("photoa raw", read("lossy_alpha_photoa_raw.webp")),
        ("photoa best", read("lossy_alpha_photoa_best.webp")),
        ("big lossy", big_lossy(false)),
        ("big lossy+alpha", big_lossy(true)),
    ];
    for (name, data) in &files {
        let one = webp::decode_planar(data, &with(1), &NO).unwrap();
        let mut want = vec![0u8; 3 * one.width * one.height];
        one.rgb(&mut want, 0, one.height, &with(1), &NO).unwrap();
        for n in counts() {
            for l in each(n) {
                let p = webp::decode_planar(data, &l, &NO).unwrap();
                assert!(
                    p.y() == one.y()
                        && p.cb() == one.cb()
                        && p.cr() == one.cr()
                        && p.alpha == one.alpha,
                    "{name}, {n} threads: planes differ"
                );
                let mut got = vec![0u8; want.len()];
                p.rgb(&mut got, 0, p.height, &l, &NO).unwrap();
                assert!(got == want, "{name}, {n} threads: rgb differs");
            }
        }
    }
}

#[test]
fn encode_is_the_same_for_every_thread_count() {
    // Over 4K pixels, so that the ALPH chunk does run on a second thread.
    let (w, h) = (300, 230);
    let pix = image(w, h);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let lw = |method| Options {
        match_libwebp: true,
        method,
        ..Options::default()
    };
    for (name, o) in [
        ("default", Options::default()),
        ("libwebp method 4", lw(4)),
        ("libwebp method 6", lw(6)),
        (
            "legacy alpha",
            Options {
                legacy_alpha: true,
                ..lw(4)
            },
        ),
    ] {
        let want = webp::encode(w, h, &src, &o, &with(1), &NO).unwrap();
        for n in counts() {
            for l in each(n) {
                let got = webp::encode(w, h, &src, &o, &l, &NO).unwrap();
                assert!(got == want, "{name}, {n} threads: output differs");
            }
        }
    }

    // An opaque 768x512 image: only the MB rows (and the analysis) run on several threads, as
    // many as 3 and 8 too.
    let (w, h) = (768, 512);
    let mut pix = image(w, h);
    for a in pix.iter_mut().skip(3).step_by(4) {
        *a = 255;
    }
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    for (name, o) in [
        (
            "768x512 method 0",
            Options {
                method: 0,
                ..Options::default()
            },
        ),
        ("768x512 default", Options::default()),
        ("768x512 libwebp method 4", lw(4)),
    ] {
        let want = webp::encode(w, h, &src, &o, &with(1), &NO).unwrap();
        for n in counts().into_iter().chain([3, 8]) {
            for l in each(n) {
                let got = webp::encode(w, h, &src, &o, &l, &NO).unwrap();
                assert!(got == want, "{name}, {n} threads: output differs");
            }
        }
    }
}
