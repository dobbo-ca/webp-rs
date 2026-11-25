//! Ports of webp-go's encoder tests (internal/webp/encode_test.go, libwebp_test.go,
//! golden_test.go) that need no cwebp or dwebp. The MatchLibwebp hashes were pinned from cwebp.

#![allow(clippy::needless_range_loop)]

mod common;

use common::pcg::Pcg;
use common::png::decode_png;
use common::{read, sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use webp::{Error, Limits, Options, Source};

static NO_CANCEL: AtomicBool = AtomicBool::new(false);

fn enc(w: usize, h: usize, src: &Source, o: &Options) -> Vec<u8> {
    webp::encode(w, h, src, o, &Limits::default(), &NO_CANCEL).unwrap()
}

/// An owned test image.
enum Img {
    Gray(usize, usize, Vec<u8>),
    Nrgba(usize, usize, Vec<u8>),
}

impl Img {
    fn size(&self) -> (usize, usize) {
        match self {
            Img::Gray(w, h, _) | Img::Nrgba(w, h, _) => (*w, *h),
        }
    }

    fn encode(&self, o: &Options) -> Vec<u8> {
        let (w, h) = self.size();
        match self {
            Img::Gray(_, _, p) => enc(w, h, &Source::Gray { pix: p, stride: w }, o),
            Img::Nrgba(_, _, p) => enc(
                w,
                h,
                &Source::Nrgba {
                    pix: p,
                    stride: 4 * w,
                },
                o,
            ),
        }
    }
}

fn load_png(name: &str) -> Img {
    let p = decode_png(&read(name));
    match p.channels {
        1 => Img::Gray(p.width, p.height, p.pix),
        3 => {
            let mut v = Vec::with_capacity(4 * p.width * p.height);
            for c in p.pix.as_chunks::<3>().0 {
                v.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            Img::Nrgba(p.width, p.height, v)
        }
        _ => Img::Nrgba(p.width, p.height, p.pix),
    }
}

fn load_fixture(name: &str) -> Img {
    if name.ends_with(".webp") {
        // he_768.webp is lossless, which x/image decodes to NRGBA.
        let m = dec(&read(name));
        Img::Nrgba(m.width, m.height, m.pix)
    } else {
        load_png(name)
    }
}

/// One 2x2 box downsample, as a DZI pyramid level.
fn halve(w: usize, h: usize, m: &[u8]) -> (usize, usize, Vec<u8>) {
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    let mut o = vec![0u8; 4 * ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut s = [0usize; 4];
            let mut n = 0;
            for j in 2 * y..(2 * y + 2).min(h) {
                for i in 2 * x..(2 * x + 2).min(w) {
                    for c in 0..4 {
                        s[c] += usize::from(m[4 * (j * w + i) + c]);
                    }
                    n += 1;
                }
            }
            for c in 0..4 {
                o[4 * (y * ow + x) + c] = ((s[c] + n / 2) / n) as u8;
            }
        }
    }
    (ow, oh, o)
}

/// The fixtures plus small textured DZI-like levels.
fn lw_cases() -> Vec<(String, Img)> {
    let mut c = vec![
        ("fl_768".to_string(), load_fixture("fl_768.png")),
        ("he_768".to_string(), load_fixture("he_768.webp")),
        ("kodim05".to_string(), load_fixture("kodim05_192.png")),
        ("kodim23".to_string(), load_fixture("kodim23_192.png")),
    ];
    let mut r = Pcg::new(21, 22);
    let (mut w, mut h) = (1600, 1200);
    let mut m = vec![0u8; 4 * w * h];
    for (i, b) in m.iter_mut().enumerate() {
        *b = r.uint32() as u8;
        if i % 4 == 3 {
            *b = 255;
        }
    }
    let mut l = 1;
    while w > 10 {
        (w, h, m) = halve(w, h, &m);
        if w <= 200 {
            c.push((format!("noise_l{l}_{w}x{h}"), Img::Nrgba(w, h, m.clone())));
        }
        l += 1;
    }
    c
}

fn lw_options(q: f32, method: i32, segments: i32) -> Options {
    Options {
        quality: q,
        method,
        match_libwebp: true,
        segments,
        ..Options::default()
    }
}

/// Pinned from cwebp 1.6.0 -q 90 -m 4, cwebp 1.6.0 -q 50 -m 6 (the trellis path) and cwebp 1.2.2
/// -q 90 -m 4 -segments 1: name, then the three sha256 sums.
const LW_GOLDEN: [(&str, [&str; 3]); 10] = [
    (
        "fl_768",
        [
            "d732b9b0b937854b759e04200d67816190e375f0a168289bf354ac57d4e8ccd6",
            "7fa20c4be7e5c88a50d62fce58885a451687ce57433ee5533aa1b1f1402274d3",
            "416833f4c4d18ff8101f6c65775d1f880c0bf714ba60e3172909e230de30e063",
        ],
    ),
    (
        "he_768",
        [
            "71604f41092ebfa7b95957e4558f1130637ecad8c065dc03f22116a050f84d7a",
            "8d1e62889a75dbf172fec6f867476e3388482ef93ee74bcbdd4b2305f5295357",
            "1ef72d885d37f0c0718b777c942a003d3776008bb2b6c680893cde3f3393228b",
        ],
    ),
    (
        "kodim05",
        [
            "bbba0f72850c386c1c46d695a718319a8c666cb6756a255faef0735eb99c7618",
            "3b5c141d6c3a8e596f7df13e5aa068a841ab2ea27c2f62d6363c297a8ded740b",
            "2fbcd2cd95f350315cbceaa94ba9152233bf8878df236781b4a9b0a5559a63cc",
        ],
    ),
    (
        "kodim23",
        [
            "394ef074fa130f0c11513f02bbbef9180fd8a730a53917370512679ddda83269",
            "db8de2c5d68068cf4b0b01a0ec02699187d98d763b4e22dec52973594f51c9c5",
            "ba9cf1fc61b97d179735d395654637f95ffee483685438c8d37832fc47600fa3",
        ],
    ),
    (
        "noise_l3_200x150",
        [
            "049d30a2197a890cf7806bfe1130fabf906aaa644fa6374a0c89f55d278c5d22",
            "c758cf5cd279fd39f3a959e059d1b0c115b0a3c6a86a7c837ae884804baedc6b",
            "a34614659cd6f1557b2ec7e027b9ab1109c9f43853a2ce1f0e96b374c3410147",
        ],
    ),
    (
        "noise_l4_100x75",
        [
            "7a3176003bfb936e6d4ceace9e2183eaa499413ae3d4a2ebec90a5be9ad28c0d",
            "52d3608161b11a119743dfa0f5e0408dfa1691eee218e764e74645de42d4943b",
            "2ef5a2149fdc148753dc1b09e9e72dd8817da4ce40b5e771e968efc341e3a019",
        ],
    ),
    (
        "noise_l5_50x38",
        [
            "adf7aafac9376212a9ca3ee2d49ec6045a49c775e424b6969188819445c266f6",
            "c50ee3397b4f6a0f9809ab7d9e8e8702c463473ccc5b79da71cd660dcac2c4bd",
            "4c7715058e22dfb402206e602269166cf5b3cdf1c170e49b3fdb6760fdf55731",
        ],
    ),
    (
        "noise_l6_25x19",
        [
            "b034753a6843d713c04cd4d4b4872b2489c9c200ab3dead3acc41b20817a8b9b",
            "0d467cb00854f19b3b878191bd6935560e1f733ced895774eae2ce40ac2ad9bb",
            "2c6e2cd9420e8cedf440acda0a0714cc46c7ce15212efb01a39f90a6996730ca",
        ],
    ),
    (
        "noise_l7_13x10",
        [
            "70e9ee48ba8d359c160af1efcc1bf092e744d7c41143a651b2d93c556e90ebf6",
            "9a7f53d86974b6366b290471f911ce4fd3f0fc5d74732e0bad4a86ff294b02d5",
            "70e9ee48ba8d359c160af1efcc1bf092e744d7c41143a651b2d93c556e90ebf6",
        ],
    ),
    (
        "noise_l8_7x5",
        [
            "14803174589c733c8042d0fac04d486501f54f9bdafe5be4e7f2545a4fe1fdf6",
            "f3ffedad76f51d541a8d8e7e8685829c97ccbb536bf13eb6115712ea1ab05665",
            "14803174589c733c8042d0fac04d486501f54f9bdafe5be4e7f2545a4fe1fdf6",
        ],
    ),
];

/// MatchLibwebp output equals cwebp's byte for byte (1.2.2 and 1.6.0 agree).
#[test]
fn match_libwebp_golden() {
    let cases = lw_cases();
    assert_eq!(cases.len(), LW_GOLDEN.len());
    for (name, img) in &cases {
        let want = LW_GOLDEN
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name}: no golden"))
            .1;
        let got = [
            sha256(&img.encode(&lw_options(90.0, 4, 0))),
            sha256(&img.encode(&lw_options(50.0, 6, 0))),
            sha256(&img.encode(&lw_options(90.0, 4, 1))),
        ];
        for (k, what) in ["q90 m4", "q50 m6", "q90 m4 segments 1"].iter().enumerate() {
            assert_eq!(got[k], want[k], "{name} {what}");
        }
    }
}

fn flat_gray(w: usize, h: usize) -> Vec<u8> {
    // 130 is Y 128 under the luma formula.
    vec![130; w * h]
}

fn dec(data: &[u8]) -> webp::Image {
    webp::decode(data, &Limits::default(), &NO_CANCEL).unwrap()
}

#[test]
fn encode_flat() {
    for (w, h) in [(1, 1), (16, 16), (17, 33), (768, 768), (1000, 1)] {
        let g = flat_gray(w, h);
        let b = enc(
            w,
            h,
            &Source::Gray { pix: &g, stride: w },
            &Options::default(),
        );
        let p = webp::decode_planar(&b, &Limits::default(), &NO_CANCEL).unwrap();
        assert_eq!((p.width, p.height), (w, h));
        for y in 0..h {
            for x in 0..w {
                assert_eq!(p.y()[y * p.y_stride() + x], 128, "{w}x{h}: Y({x},{y})");
                let c = (y / 2) * p.c_stride() + x / 2;
                assert_eq!(p.cb()[c], 128, "{w}x{h}: Cb({x},{y})");
                assert_eq!(p.cr()[c], 128, "{w}x{h}: Cr({x},{y})");
            }
        }
    }
}

#[test]
fn encode_sizes() {
    let g = flat_gray(768, 768);
    let b = enc(
        768,
        768,
        &Source::Gray {
            pix: &g,
            stride: 768,
        },
        &Options::default(),
    );
    // 1030, not 1048: the solid fast path codes at qi 0, not the quality-75 qi the slow path
    // picked when this size was set. Plus 4 for the second token partition.
    assert_eq!(b.len(), 1034);

    let g = flat_gray(1, 1);
    let b = enc(
        1,
        1,
        &Source::Gray { pix: &g, stride: 1 },
        &Options::default(),
    );
    assert_eq!(b.len(), 48);
    assert_eq!(u32::from_le_bytes([b[16], b[17], b[18], b[19]]), 27);
}

#[test]
fn encode_bad_arguments() {
    let g = vec![0u8; 16384];
    let src = Source::Gray {
        pix: &g,
        stride: 16384,
    };
    let o = Options::default();
    let l = Limits::default();
    let e = |w, h, s: &Source, o: &Options| webp::encode(w, h, s, o, &l, &NO_CANCEL);
    assert!(matches!(e(0, 0, &src, &o), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        e(16384, 1, &src, &o),
        Err(Error::InvalidArgument(_))
    ));
    for segments in [-1, 2, 3, 5] {
        let o = Options {
            segments,
            ..Options::default()
        };
        assert!(matches!(e(8, 8, &src, &o), Err(Error::InvalidArgument(_))));
    }
    // Planes shorter than the image.
    let short = Source::Gray {
        pix: &g[..63],
        stride: 8,
    };
    assert!(matches!(
        e(8, 8, &short, &o),
        Err(Error::InvalidArgument(_))
    ));
    let nrgba = Source::Nrgba {
        pix: &g[..4 * 8 * 8 - 1],
        stride: 32,
    };
    assert!(matches!(
        e(8, 8, &nrgba, &o),
        Err(Error::InvalidArgument(_))
    ));
    let ycc = Source::Ycbcr420 {
        y: &g[..64],
        cb: &g[..15],
        cr: &g[..16],
        y_stride: 8,
        c_stride: 4,
    };
    assert!(matches!(e(8, 8, &ycc, &o), Err(Error::InvalidArgument(_))));
}

#[test]
fn encode_limits_and_cancel() {
    let g = flat_gray(64, 64);
    let src = Source::Gray {
        pix: &g,
        stride: 64,
    };
    let o = Options::default();
    let small = Limits {
        max_pixels: 100,
        ..Limits::default()
    };
    assert!(matches!(
        webp::encode(64, 64, &src, &o, &small, &NO_CANCEL),
        Err(Error::MaxPixels { .. })
    ));
    let tiny = Limits {
        max_alloc: 1000,
        ..Limits::default()
    };
    // A textured image, so the planes and tokens are charged.
    let noise: Vec<u8> = (0..64 * 64)
        .map(|i: u32| (i.wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    let src = Source::Gray {
        pix: &noise,
        stride: 64,
    };
    assert!(matches!(
        webp::encode(64, 64, &src, &o, &tiny, &NO_CANCEL),
        Err(Error::MemoryBudget { .. })
    ));
    let stop = AtomicBool::new(true);
    assert_eq!(
        webp::encode(64, 64, &src, &o, &Limits::default(), &stop),
        Err(Error::Cancelled)
    );
    stop.store(false, Ordering::Relaxed);
    assert!(webp::encode(64, 64, &src, &o, &Limits::default(), &stop).is_ok());
}

/// PSNR of an RGBA decode against opaque NRGBA pixels.
fn psnr(w: usize, h: usize, src: &[u8], got: &[u8]) -> f64 {
    let mut sse = 0f64;
    for i in 0..w * h {
        for c in 0..3 {
            let d = f64::from(src[4 * i + c]) - f64::from(got[4 * i + c]);
            sse += d * d;
        }
    }
    if sse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 * 3.0 * (w * h) as f64 / sse).log10()
}

/// Every method decodes to the right size and close to the source (golden_test.go's checks that
/// need no external tool).
#[test]
fn encode_decodes_close() {
    for name in ["kodim05_192.png", "kodim23_192.png"] {
        let Img::Nrgba(w, h, pix) = load_png(name) else {
            panic!("{name} is not RGB");
        };
        for method in [0, 2, 4, 6] {
            let b = enc(
                w,
                h,
                &Source::Nrgba {
                    pix: &pix,
                    stride: 4 * w,
                },
                &Options {
                    quality: 90.0,
                    method,
                    ..Options::default()
                },
            );
            let d = dec(&b);
            assert_eq!((d.width, d.height), (w, h), "{name} m{method}");
            let p = psnr(w, h, &pix, &d.pix);
            assert!(p > 33.0, "{name} m{method}: PSNR {p:.2} dB");
        }
    }
}

/// Gray encodes as the same image as gray NRGBA, and EncodeGray's pixels as Encode's.
#[test]
fn gray_matches_rgba() {
    let Img::Gray(w, h, g) = load_png("fl_768.png") else {
        panic!("fl_768.png is not gray");
    };
    let mut n = Vec::with_capacity(4 * w * h);
    for &v in &g {
        n.extend_from_slice(&[v, v, v, 255]);
    }
    let a = enc(
        w,
        h,
        &Source::Gray { pix: &g, stride: w },
        &Options::default(),
    );
    let b = enc(
        w,
        h,
        &Source::Nrgba {
            pix: &n,
            stride: 4 * w,
        },
        &Options::default(),
    );
    assert_eq!(a, b, "gray {} B and NRGBA {} B differ", a.len(), b.len());
}

#[test]
fn encode_ycbcr() {
    let (w, h): (usize, usize) = (37, 21);
    let y: Vec<u8> = (0..w * h).map(|i| (i * 7) as u8).collect();
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let c = vec![128u8; cw * ch];
    let b = enc(
        w,
        h,
        &Source::Ycbcr420 {
            y: &y,
            cb: &c,
            cr: &c,
            y_stride: w,
            c_stride: cw,
        },
        &Options::default(),
    );
    let d = dec(&b);
    assert_eq!((d.width, d.height), (w, h));
}

#[test]
fn method_clamp() {
    let m = load_png("kodim23_192.png");
    let q = |q: f32, method: i32| {
        m.encode(&Options {
            quality: q,
            method,
            ..Options::default()
        })
    };
    // Methods 2..4 share one path by design; 0 and 2 must differ.
    assert_ne!(
        q(75.0, 0),
        q(75.0, 2),
        "Method 0 and 2 produce the same bytes"
    );
    assert_eq!(q(75.0, -1), q(75.0, 0), "Method -1 differs from 0");
    assert_eq!(q(75.0, 7), q(75.0, 6), "Method 7 differs from 6");
    assert_eq!(
        m.encode(&Options::default()),
        q(75.0, 4),
        "default Options differs from Quality 75 Method 4"
    );
}

/// A premultiplied RGBA image of opaque pixels encodes like the NRGBA one.
#[test]
fn rgba_opaque_matches_nrgba() {
    let Img::Nrgba(w, h, pix) = load_png("kodim05_192.png") else {
        panic!()
    };
    let a = enc(
        w,
        h,
        &Source::Rgba {
            pix: &pix,
            stride: 4 * w,
        },
        &Options::default(),
    );
    let b = enc(
        w,
        h,
        &Source::Nrgba {
            pix: &pix,
            stride: 4 * w,
        },
        &Options::default(),
    );
    assert_eq!(a, b);
}

/// A NaN quality encodes as the Go package does, not as an error: a solid image never reads the
/// quality, and any other gets quality index 0 and no error diffusion, which is what quality 100
/// gives, on the default path and on the libwebp one, with and without the analysis pass.
#[test]
fn nan_quality_is_quality_100() {
    let (w, h) = (48, 40);
    let textured: Vec<u8> = (0..4 * w * h)
        .map(|i: usize| {
            if i % 4 == 3 {
                255
            } else {
                (i.wrapping_mul(2654435761) >> 13) as u8 ^ (i / 7) as u8
            }
        })
        .collect();
    let solid: Vec<u8> = [10u8, 200, 90, 255].repeat(w * h);
    for (name, pix) in [("textured", &textured), ("solid", &solid)] {
        let src = Source::Nrgba { pix, stride: 4 * w };
        for (match_libwebp, segments) in [(false, 0), (true, 0), (true, 1)] {
            let o = |quality| Options {
                quality,
                match_libwebp,
                segments,
                ..Options::default()
            };
            let nan = enc(w, h, &src, &o(f32::NAN));
            let full = enc(w, h, &src, &o(100.0));
            assert!(
                nan == full,
                "{name}, match_libwebp {match_libwebp}, segments {segments}: NaN differs from 100"
            );
            if name == "solid" && !match_libwebp {
                assert!(nan == enc(w, h, &src, &o(30.0)), "solid reads the quality");
            }
        }
    }
}
