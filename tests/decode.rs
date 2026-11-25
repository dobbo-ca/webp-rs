//! Ports of webp-go's decode tests (internal/webp/decode_test.go).
#![allow(clippy::type_complexity)]

mod common;

use common::{read, read_lines, sha256, write_alpha};
use std::sync::atomic::AtomicBool;
use webp::{Error, Limits};

static NO_CANCEL: AtomicBool = AtomicBool::new(false);

fn dec(data: &[u8]) -> Result<webp::Image, Error> {
    webp::decode(data, &Limits::default(), &NO_CANCEL)
}

fn class(r: Result<webp::Image, Error>) -> &'static str {
    match r {
        Ok(_) => "ok",
        Err(e) => e.class(),
    }
}

/// Lossless output against sha256 sums of what dwebp -pam wrote (tests/data/gen_decode.sh).
#[test]
fn decode_oracle() {
    let lines = read_lines("decode_oracle.txt");
    assert!(!lines.is_empty());
    for line in &lines {
        let (name, want) = line.split_once(' ').unwrap();
        let data = read(name);
        let m = dec(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(sha256(&m.pix), want, "{name}: pixels differ from dwebp");
        let c = webp::decode_config(&data, &Limits::default()).unwrap();
        assert_eq!((c.width, c.height), (m.width, m.height), "{name}");
        assert_eq!(c.lossless, !name.starts_with("lossy_"), "{name}");
    }
}

#[test]
fn decode_refusals() {
    let good = read("ll_photo_z4.webp");
    let patch = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = good.clone();
        f(&mut b);
        b
    };
    let animated =
        b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let with_alph =
        |alph: &[u8]| write_alpha(alph, 97, 61, false, &read("lossy_photo_a.webp")[20..]);
    let small = Limits {
        max_pixels: 10,
        ..Limits::default()
    };
    let cases: Vec<(&str, Vec<u8>, Option<&Limits>, &str)> = vec![
        ("empty", vec![], None, "format"),
        ("not riff", b"RIFX0000WEBPVP8L".to_vec(), None, "format"),
        (
            "riff size past file",
            good[..good.len() - 1].to_vec(),
            None,
            "truncated",
        ),
        (
            "chunk size past riff",
            patch(&|b| b[16..20].copy_from_slice(&(1u32 << 30).to_le_bytes())),
            None,
            "truncated",
        ),
        ("animated without frames", animated.to_vec(), None, "format"),
        ("bad signature", patch(&|b| b[20] = 0), None, "format"),
        ("pixels over cap", good.clone(), Some(&small), "limit"),
        ("raw ALPH too short", with_alph(&[0, 0]), None, "truncated"),
        ("ALPH method 2", with_alph(&[2, 0]), None, "format"),
        ("ALPH reserved bits", with_alph(&[0x40, 0]), None, "format"),
        ("ALPH header only", with_alph(&[0]), None, "format"),
        ("ALPH VP8L stream cut", with_alph(&[1, 0]), None, "format"),
    ];
    for (name, data, l, want) in cases {
        let r = webp::decode(&data, l.unwrap_or(&Limits::default()), &NO_CANCEL);
        assert_eq!(class(r), want, "{name}");
    }
}

/// The first frame, at its ANMF offset on a transparent canvas, against sha256 sums of what dwebp
/// -pam wrote for it (tests/data/gen_anim.sh), and the later frames only counted.
#[test]
fn decode_animated() {
    for line in read_lines("anim_oracle.txt") {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (name, dropped, want) = (f[0], f[1].parse::<usize>().unwrap(), f[2]);
        let data = read(name);
        let m = dec(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            sha256(&m.pix),
            want,
            "{name}: pixels differ from dwebp's frame 1"
        );
        let c = webp::decode_config(&data, &Limits::default()).unwrap();
        assert_eq!(
            (c.width, c.height, c.frames_dropped),
            (m.width, m.height, dropped),
            "{name}"
        );
        let r = webp::decode_planar(&data, &Limits::default(), &NO_CANCEL);
        assert_eq!(r.err(), Some(Error::Animated), "{name}");
    }
}

fn find(b: &[u8], tag: &str) -> usize {
    b.windows(tag.len())
        .position(|w| w == tag.as_bytes())
        .unwrap()
}

const STRAY_ALPH: [u8; 12] = [b'A', b'L', b'P', b'H', 4, 0, 0, 0, 0, 0, 0, 0];

/// Inserts `chunk` before the nth tag in `b`, or at the end for tag "", and fixes the RIFF size.
fn splice(b: &[u8], tag: &str, chunk: &[u8], nth: usize) -> Vec<u8> {
    let mut at = b.len();
    if !tag.is_empty() {
        at = 0;
        for _ in 0..=nth {
            at += find(&b[at..], tag) + 1;
        }
        at -= 1;
    }
    let mut out = b[..at].to_vec();
    out.extend_from_slice(chunk);
    out.extend_from_slice(&b[at..]);
    let n = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&n.to_le_bytes());
    out
}

#[test]
fn decode_animated_refusals() {
    let good = read("anim_lossless_off.webp"); // canvas 64 x 48, frame 1 40 x 24 at 6, 2
    let patch = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = good.clone();
        f(&mut b);
        b
    };
    let anmf = find(&good, "ANMF");
    let vp8x = find(&good, "VP8X");
    let tight = Limits {
        max_pixels: 64 * 48 - 1,
        ..Limits::default()
    };
    let mut no_image = b"ANMF\x10\x00\x00\x00".to_vec();
    no_image.extend_from_slice(&[0; 16]);
    let cases: Vec<(&str, Vec<u8>, Option<&Limits>, &str)> = vec![
        (
            "frame past canvas width",
            patch(&|b| b[anmf + 8] = 13),
            None,
            "format",
        ),
        (
            "frame past canvas height",
            patch(&|b| b[anmf + 11] = 14),
            None,
            "format",
        ),
        (
            "frame size not the image's",
            patch(&|b| b[anmf + 14] += 1),
            None,
            "format",
        ),
        (
            "ANMF without the flag",
            patch(&|b| b[vp8x + 8] &= !2),
            None,
            "format",
        ),
        (
            "canvas over the pixel cap",
            good.clone(),
            Some(&tight),
            "limit",
        ),
        (
            "canvas side over the cap",
            patch(&|b| {
                b[vp8x + 14] = 0xff;
                b[vp8x + 15] = 0x3f;
            }),
            None,
            "limit",
        ),
        (
            "stray ALPH before the first ANMF",
            splice(&read("anim_lossy_full.webp"), "ANMF", &STRAY_ALPH, 0),
            None,
            "format",
        ),
        (
            "stray ALPH after the first ANMF",
            splice(&read("anim_alpha_off.webp"), "ANMF", &STRAY_ALPH, 1),
            None,
            "format",
        ),
        (
            "later ANMF without an image",
            splice(&good, "", &no_image, 0),
            None,
            "format",
        ),
        (
            "ANMF cut short",
            patch(&|b| b[anmf + 4..anmf + 8].copy_from_slice(&8u32.to_le_bytes())),
            None,
            "format",
        ),
    ];
    for (name, data, l, want) in cases {
        let r = webp::decode(&data, l.unwrap_or(&Limits::default()), &NO_CANCEL);
        assert_eq!(class(r), want, "{name}");
    }
}

/// libwebp rejects a prefix code whose max_symbol is past its alphabet (vp8l_dec.c
/// ReadHuffmanCode); tests/data/maxsym_bad.webp has 1002 for 280.
#[test]
fn decode_max_symbol_refused() {
    let r = dec(&read("maxsym_bad.webp"));
    assert_eq!(class(r), "vp8l-bitstream");
}

/// Every truncation of a good file fails, with an error, not a panic.
#[test]
fn decode_truncated() {
    for name in ["ll_photo_z9", "lossy_photo_a", "lossy_ours_photo"] {
        let good = read(&format!("{name}.webp"));
        for n in (0..good.len()).step_by(5) {
            let mut b = good[..n].to_vec();
            if n >= 8 {
                b[4..8].copy_from_slice(&((n - 8) as u32).to_le_bytes()); // RIFF size matches
            }
            if b.len() >= 20 {
                b[16..20].copy_from_slice(&((n - 20) as u32).to_le_bytes()); // so does the chunk
            }
            assert!(dec(&b).is_err(), "{name}: {n} bytes decoded");
        }
    }
}

#[test]
fn decode_budget() {
    let data = read("ll_photo_z4.webp");
    let l = Limits {
        max_alloc: 1000,
        ..Limits::default()
    };
    match webp::decode(&data, &l, &NO_CANCEL) {
        Err(Error::MemoryBudget { .. }) => {}
        r => panic!("over the whole budget: {:?}", r.err()),
    }
    // The pixels alone fit, the transform images and prefix codes do not.
    let l = Limits {
        max_alloc: 4 * 97 * 61,
        ..Limits::default()
    };
    assert_eq!(
        webp::decode(&data, &l, &NO_CANCEL).err().map(|e| e.class()),
        Some("limit")
    );
    assert!(dec(&data).is_ok());
}

/// A side over 16383 and a pixel count over `max_pixels` are each their own `Error` variant, which a
/// caller can match, all in the parity class `limit`.
#[test]
fn limit_errors_are_distinct_variants() {
    // A VP8L header only: the limits are checked before the stream is read.
    let file = |w: u32, h: u32| -> Vec<u8> {
        let v = (w - 1) | ((h - 1) << 14);
        let mut b = b"RIFF\x12\x00\x00\x00WEBPVP8L\x05\x00\x00\x00".to_vec();
        b.extend_from_slice(&[
            0x2f,
            v as u8,
            (v >> 8) as u8,
            (v >> 16) as u8,
            (v >> 24) as u8,
            0,
        ]);
        b
    };
    assert_eq!(
        dec(&file(16384, 1)).err(),
        Some(Error::Width {
            got: 16384,
            max: 16383
        })
    );
    assert_eq!(
        dec(&file(1, 16384)).err(),
        Some(Error::Height {
            got: 16384,
            max: 16383
        })
    );
    let l = Limits {
        max_pixels: 99,
        ..Limits::default()
    };
    assert_eq!(
        webp::decode(&file(10, 10), &l, &NO_CANCEL).err(),
        Some(Error::MaxPixels { got: 100, max: 99 })
    );
}

#[test]
fn decode_cancelled() {
    let stop = AtomicBool::new(true);
    for name in ["ll_photo_z9.webp", "lossy_photo_a.webp"] {
        let r = webp::decode(&read(name), &Limits::default(), &stop);
        assert_eq!(r.err(), Some(Error::Cancelled), "{name}");
    }
}

/// `decode_planar` reserves the planes only, and its RGB bands add up to the one-shot RGBA that
/// `decode` makes.
#[test]
fn decode_planar() {
    let data = read("lossy_kodim05.webp");
    let want = dec(&data).unwrap();
    let c = webp::decode_config(&data, &Limits::default()).unwrap();
    let (pw, ph) = ((c.width + 15) & !15, (c.height + 15) & !15);
    let planes = pw * ph * 3 / 2 + (pw / 16) * 160 + 4096;
    let l = Limits {
        max_alloc: planes,
        ..Limits::default()
    };
    assert!(
        webp::decode(&data, &l, &NO_CANCEL).is_err(),
        "decode fits the planes-only budget"
    );
    let p = webp::decode_planar(&data, &l, &NO_CANCEL).unwrap();
    let mut band = vec![0u8; 3 * c.width * 5];
    let mut y = 0;
    while y < c.height {
        let y1 = (y + 5).min(c.height);
        let n = 3 * c.width * (y1 - y);
        p.rgb(&mut band[..n], y, y1, &Limits::default(), &NO_CANCEL)
            .unwrap();
        for i in 0..c.width * (y1 - y) {
            let o = 4 * (y * c.width + i);
            assert_eq!(
                band[3 * i..3 * i + 3],
                want.pix[o..o + 3],
                "row {y} px {i} differs"
            );
        }
        y = y1;
    }
    assert!(
        p.rgb(&mut band, 0, 1, &Limits::default(), &NO_CANCEL)
            .is_err(),
        "wrong dst size accepted"
    );
    let r = webp::decode_planar(&read("ll_photo_z4.webp"), &Limits::default(), &NO_CANCEL);
    assert_eq!(r.err(), Some(Error::NotLossy));
}

/// The alpha plane and RGB of `decode_planar` equal `decode`'s, for raw and compressed ALPH, and
/// planes plus a raw alpha fit a budget `decode` does not.
#[test]
fn decode_planar_alpha() {
    let mut names: Vec<String> =
        std::fs::read_dir(format!("{}/tests/data", env!("CARGO_MANIFEST_DIR")))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.starts_with("lossy_alpha_") && n.ends_with(".webp"))
            .collect();
    names.sort();
    assert!(!names.is_empty());
    for name in &names {
        let data = read(name);
        let want = dec(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        let p = webp::decode_planar(&data, &Limits::default(), &NO_CANCEL).unwrap();
        let alpha = p.alpha.as_ref().unwrap();
        let mut rgb = vec![0u8; 3 * p.width * p.height];
        p.rgb(&mut rgb, 0, p.height, &Limits::default(), &NO_CANCEL)
            .unwrap();
        for (i, &a) in alpha.iter().enumerate() {
            assert!(
                a == want.pix[4 * i + 3] && rgb[3 * i..3 * i + 3] == want.pix[4 * i..4 * i + 3],
                "{name}: pixel {i} differs"
            );
        }
    }
    let data = read("lossy_alpha_photoa_raw.webp");
    let c = webp::decode_config(&data, &Limits::default()).unwrap();
    let (pw, ph) = ((c.width + 15) & !15, (c.height + 15) & !15);
    let n = pw * ph * 3 / 2 + c.width * c.height + (pw / 16) * 160 + 4096;
    let l = Limits {
        max_alloc: n,
        ..Limits::default()
    };
    assert!(
        webp::decode(&data, &l, &NO_CANCEL).is_err(),
        "decode fits the planar budget"
    );
    webp::decode_planar(&data, &l, &NO_CANCEL).unwrap();
}
