//! Buffer reuse: an `Encoder` and a `Decoder` that have handled a file once handle it again, at
//! `max_threads` 1, without one call to the allocator. This binary counts every allocation
//! (`alloc`, `alloc_zeroed` and `realloc`), so it holds exactly one test: another test running
//! beside it would be counted too. The exception is the 1.6.0 ALPH writer (`alpha_libwebp16`),
//! which makes its scratch in each call: its output is checked, its allocations are not counted.
//!
//! Each case runs one call before the counted one, so the counted call is the second. Encodes
//! with alpha are the exception: they run two calls first and count the third, as the ALPH
//! encoder swaps the roles of its reference buffers while it works, so the capacities of the
//! pair settle only after two calls.

mod common;

use common::png::decode_png;
use common::read;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use webp::{Decoder, Encoder, Limits, Options, Source};

/// Counts the calls that ask the allocator for memory.
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// How many allocator calls of a measured window `LOG` keeps.
const LOG_LEN: usize = 16;
/// `ALLOCS` when the window opened.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// The first `LOG_LEN` calls of the window as `(size << 2) | kind`; atomics, so the allocator
/// itself never allocates to record them.
static LOG: [AtomicUsize; LOG_LEN] = [const { AtomicUsize::new(0) }; LOG_LEN];
const KINDS: [&str; 3] = ["alloc", "alloc_zeroed", "realloc"];

/// Counts one allocator call of `size` bytes (`kind` indexes `KINDS`) and files it in `LOG`.
fn note(size: usize, kind: usize) {
    let k = ALLOCS
        .fetch_add(1, Relaxed)
        .wrapping_sub(BASE.load(Relaxed));
    if k < LOG_LEN {
        LOG[k].store((size << 2) | kind, Relaxed);
    }
}

/// Opens a measured window and returns the count at its start.
fn open_window() -> usize {
    let now = ALLOCS.load(Relaxed);
    BASE.store(now, Relaxed);
    now
}

/// The first allocator calls of the window that just closed with `n` of them, by kind and size.
fn first_allocs(n: usize) -> String {
    (0..n.min(LOG_LEN))
        .map(|k| {
            let v = LOG[k].load(Relaxed);
            format!("{} {}", KINDS[v & 3], v >> 2)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// SAFETY: every method forwards to `System` with the caller's arguments, so System's contract
// carries over; the counter is an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size(), 0);
        // SAFETY: the caller upholds `alloc`'s contract, which `System.alloc` has too.
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size(), 1);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from System with it.
        unsafe { System.dealloc(p, l) };
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new, 2);
        // SAFETY: `p` came from this allocator with layout `l`, and `new` is valid for it.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The image of a PNG as NRGBA, its alpha set by `alpha(x, y)`.
fn image(name: &str, alpha: impl Fn(usize, usize) -> u8) -> (usize, usize, Vec<u8>) {
    let p = decode_png(&read(name));
    assert_eq!(p.channels, 3, "{name}");
    let mut pix = Vec::with_capacity(4 * p.width * p.height);
    for (i, c) in p.pix.as_chunks::<3>().0.iter().enumerate() {
        pix.extend_from_slice(&[c[0], c[1], c[2], alpha(i % p.width, i / p.width)]);
    }
    (p.width, p.height, pix)
}

#[test]
fn second_call_allocates_nothing() {
    let mut rows: Vec<(String, usize)> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    let limits = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);

    // Encode: an opaque image, one with four alpha values (the palette writer) and one with many
    // (filters, the predictor and the histograms), at the methods and writers of each ALPH path.
    let images = [
        ("opaque", image("kodim05_192.png", |_, _| 255)),
        (
            "alpha4",
            image("kodim05_192.png", |x, y| {
                [0, 85, 170, 255][(x / 12 + y / 9) % 4]
            }),
        ),
        (
            "alpha-many",
            image("kodim23_192.png", |x, y| {
                if (x / 8 + y / 8) % 5 == 0 {
                    0
                } else {
                    (x * 5 + y * 3) as u8
                }
            }),
        ),
    ];
    // (method, match_libwebp, legacy_alpha, alpha_libwebp16, counted)
    let cases = [
        (0, false, false, false, true),
        (4, false, false, false, true),
        (6, false, false, false, true),
        (4, true, false, false, true),
        (6, true, false, false, true),
        (4, true, true, false, true),
        (4, true, false, true, false),
    ];
    for (name, (w, h, pix)) in &images {
        let (w, h) = (*w, *h);
        let src = Source::Nrgba { pix, stride: 4 * w };
        for (method, match_libwebp, legacy_alpha, alpha_libwebp16, counted) in cases {
            let o = Options {
                method,
                match_libwebp,
                legacy_alpha,
                alpha_libwebp16,
                ..Options::default()
            };
            let what = format!(
                "encode_into {name} m{method} match {match_libwebp} legacy {legacy_alpha} a16 {alpha_libwebp16}"
            );
            let want = webp::encode(w, h, &src, &o, &limits, &cancel).unwrap();
            let mut enc = Encoder::new();
            let mut out = Vec::new();
            // The ALPH encoder's swapped buffers need a second warm-up call; see the top.
            let warmups = if *name == "opaque" { 1 } else { 2 };
            for _ in 0..warmups {
                enc.encode_into(w, h, &src, &o, &limits, &cancel, &mut out)
                    .unwrap();
            }
            let before = open_window();
            let r = enc.encode_into(w, h, &src, &o, &limits, &cancel, &mut out);
            let n = ALLOCS.load(Relaxed) - before;
            let first = first_allocs(n);
            if let Err(e) = r {
                bad.push(format!("{what}: {e:?}"));
                continue;
            }
            if out != want || enc.encode(w, h, &src, &o, &limits, &cancel).unwrap() != want {
                bad.push(format!("{what}: output differs from encode"));
            }
            if counted && n != 0 {
                bad.push(format!("{what}: {n} allocations, first: [{first}]"));
            }
            rows.push((what, n));
        }
    }

    // Decode: lossy, lossy with a raw and with a compressed ALPH chunk, lossless (with and
    // without alpha) and the first frame of an animated file placed on a canvas (lossy with
    // alpha, lossless).
    let files = [
        "lossy_kodim05.webp",
        "lossy_alpha_photoa_raw.webp",
        "lossy_alpha_photoa_best.webp",
        "ll_photo_z4.webp",
        "ll_photoa_z4.webp",
        "ll_alpha_z4.webp",
        "he_768.webp",
        "anim_alpha_off.webp",
        "anim_lossless_off.webp",
    ];
    for name in files {
        let data = read(name);
        let what = format!("decode {name}");
        let want = webp::decode(&data, &limits, &cancel).unwrap();
        let mut dec = Decoder::new();
        dec.decode(&data, &limits, &cancel).unwrap();
        let before = open_window();
        let r = dec.decode(&data, &limits, &cancel);
        let n = ALLOCS.load(Relaxed) - before;
        let first = first_allocs(n);
        match r {
            Ok(got) => {
                if *got != want {
                    bad.push(format!("{what}: output differs from decode"));
                }
            }
            Err(e) => {
                bad.push(format!("{what}: {e:?}"));
                continue;
            }
        }
        if n != 0 {
            bad.push(format!("{what}: {n} allocations, first: [{first}]"));
        }
        rows.push((what, n));
    }

    // Planar decode of lossy files, with and without alpha.
    for name in [
        "lossy_kodim05.webp",
        "lossy_alpha_photoa_raw.webp",
        "lossy_alpha_photoa_best.webp",
    ] {
        let data = read(name);
        let what = format!("decode_planar {name}");
        let want = webp::decode_planar(&data, &limits, &cancel).unwrap();
        let mut dec = Decoder::new();
        dec.decode_planar(&data, &limits, &cancel).unwrap();
        let before = open_window();
        let r = dec.decode_planar(&data, &limits, &cancel);
        let n = ALLOCS.load(Relaxed) - before;
        let first = first_allocs(n);
        match r {
            Ok(got) => {
                let same = got.y() == want.y()
                    && got.cb() == want.cb()
                    && got.cr() == want.cr()
                    && got.alpha == want.alpha
                    && (got.width, got.height) == (want.width, want.height);
                if !same {
                    bad.push(format!("{what}: planes differ from decode_planar"));
                }
            }
            Err(e) => {
                bad.push(format!("{what}: {e:?}"));
                continue;
            }
        }
        if n != 0 {
            bad.push(format!("{what}: {n} allocations, first: [{first}]"));
        }
        rows.push((what, n));
    }

    for (what, n) in &rows {
        eprintln!("{what}: {n} allocations");
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
