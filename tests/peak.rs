//! `max_alloc` is the enforced peak: under a counting global allocator, every call either fails
//! with a limit error or allocates at most `max_alloc` plus `SLACK` bytes at once, for several
//! `max_alloc` values and thread counts: the lossless and lossy decoders (the lossy one with
//! bands of rows on several threads), `metadata`, the encoder with every ALPH writer, and a
//! reused `Encoder` and `Decoder`. A reallocation counts as the old and the new block together, as
//! it holds both while it copies. One test fn, so no other test shares the counters.

mod common;

use common::read;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use webp::{Error, Limits, Options, Source};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

fn add(n: usize) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s contract
// holds; the counters are only read and written, never used as a pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the `GlobalAlloc::alloc` contract.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the `GlobalAlloc::alloc_zeroed` contract.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` and `l` are those of the allocation, by the caller.
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // The old and the new block are both live while it copies: count both from the call.
        add(new);
        // SAFETY: the caller upholds the `GlobalAlloc::realloc` contract.
        let q = unsafe { System.realloc(p, l, new) };
        LIVE.fetch_sub(if q.is_null() { new } else { l.size() }, Relaxed);
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// What a call may allocate beyond `max_alloc`: tables and buffers of a fixed size that no input
/// scales (prefix code roots, the VP8 decoder's and encoder's macroblock state, the test harness's
/// own threads, formatting of errors).
const SLACK: usize = 256 << 10;

static NO: AtomicBool = AtomicBool::new(false);

/// Runs `f`, and returns its result with the most bytes allocated at once while it ran, counted
/// from `base` (the bytes live before anything the call holds was allocated).
fn measure<T>(base: usize, f: impl FnOnce() -> T) -> (T, usize) {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
    let r = f();
    (r, PEAK.load(Relaxed).saturating_sub(base))
}

/// `check_with` for a call that needs no state.
fn check<T>(what: &str, run: impl Fn(&Limits) -> Result<T, Error>) {
    check_with(what, |_| (), |_, l| run(l));
}

/// Calls `run` with `max_alloc` of every rung of a ladder around the unconstrained peak and of a
/// bisection to the smallest one that passes, for each thread count; asserts the peak at each.
/// Before each call `prep` makes the state `run` takes (a warmed `Encoder` or `Decoder`). What it
/// keeps is counted in the peak: the call charges the kept buffers against `max_alloc`, so they
/// and what the call allocates on top of them must fit together, but until a call charges a kept
/// buffer (or drops them all to start again) it is live beside the call's charges, so the bound
/// is `max_alloc` plus what was kept. `prep` must keep nothing the
/// call does not hold (a warmed output `Vec`, say), as that would count too.
fn check_with<S, T>(
    what: &str,
    prep: impl Fn(&Limits) -> S,
    run: impl Fn(&mut S, &Limits) -> Result<T, Error>,
) {
    let mut counts = vec![1, 2, 3, Limits::default().max_threads];
    counts.sort_unstable();
    counts.dedup();
    for threads in counts {
        let limits = |m: usize| Limits {
            max_threads: threads,
            max_alloc: m,
            ..Limits::default()
        };
        let base = LIVE.load(Relaxed);
        let mut s = prep(&limits(1 << 30));
        let (r, real) = measure(base, || run(&mut s, &limits(1 << 30)));
        assert!(r.is_ok(), "{what}: fails unconstrained");
        drop((r, s));
        let probe = |m: usize| -> bool {
            let base = LIVE.load(Relaxed);
            let mut s = prep(&limits(1 << 30));
            // What the state holds before the call is the caller's memory (docs/limits.md): the
            // call charges a kept buffer when it uses it, and drops them all and starts again when
            // they do not fit. Until then they are live beside what the call has charged, so the
            // bound is their sum; an uncharged allocation still shows above it.
            let kept = LIVE.load(Relaxed).saturating_sub(base);
            let (r, peak) = measure(base, || run(&mut s, &limits(m)));
            assert!(
                peak <= m + kept + SLACK,
                "{what}, {threads} threads, max_alloc {m}, {kept} kept: allocated {peak} at once"
            );
            match r {
                Ok(_) => true,
                Err(e) if e.class() == "limit" => false,
                Err(e) => panic!("{what}: {e}"),
            }
        };
        for m in [real / 8, real / 4, real / 2, real, 2 * real, 4 * real] {
            probe(m);
        }
        let (mut lo, mut hi) = (0, 1 << 30);
        while hi - lo > 1 + hi / 256 {
            let mid = lo + (hi - lo) / 2;
            if probe(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        eprintln!("{what}, {threads} threads: unconstrained peak {real}, passes from {hi}");
    }
}

/// `image` with alpha of four values only, so the ALPH encoders take their palette paths.
fn palette_image(w: usize, h: usize) -> Vec<u8> {
    let mut pix = image(w, h, true);
    for (i, px) in pix.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = (i % w, i / w);
        px[3] = [0, 85, 170, 255][(x / 24 + y / 17) % 4];
    }
    pix
}

/// A textured image, with alpha holes if `alpha`.
fn image(w: usize, h: usize, alpha: bool) -> Vec<u8> {
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut pix = vec![0u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = if !alpha {
                255
            } else {
                match (x / 24 + y / 17) % 4 {
                    0 => 0,
                    1 => (x * 3) as u8,
                    _ => 255,
                }
            };
            let o = 4 * (y * w + x);
            let n = (s >> 60) as u8;
            pix[o..o + 4].copy_from_slice(&[
                (x / 2) as u8 ^ n,
                (y / 2) as u8,
                (x + y) as u8 ^ n,
                a,
            ]);
        }
    }
    pix
}

#[test]
fn peak_is_within_max_alloc() {
    let ll = read("ll_photoa_z4.webp");
    check("decode lossless", |l| webp::decode(&ll, l, &NO));
    let he = read("he_768.webp");
    check("decode lossless he768", |l| webp::decode(&he, l, &NO));

    let (w, h) = (448, 336);
    let pix = image(w, h, true);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let lossy_alpha =
        webp::encode(w, h, &src, &Options::default(), &Limits::default(), &NO).unwrap();
    check("decode lossy+alpha", |l| webp::decode(&lossy_alpha, l, &NO));
    check("decode_planar lossy+alpha", |l| {
        webp::decode_planar(&lossy_alpha, l, &NO).map(|p| p.y().len())
    });

    // 4 MiB of RGBA: the conversion splits into bands on up to 3 threads.
    let (bw, bh) = (1024, 1024);
    let bpix = image(bw, bh, false);
    let bsrc = Source::Nrgba {
        pix: &bpix,
        stride: 4 * bw,
    };
    let lossy = webp::encode(bw, bh, &bsrc, &Options::default(), &Limits::default(), &NO).unwrap();
    check("decode lossy, bands", |l| webp::decode(&lossy, l, &NO));
    check("decode_planar lossy", |l| {
        webp::decode_planar(&lossy, l, &NO).map(|p| p.y().len())
    });
    // A reused Decoder, warmed by one call with the same file.
    let warm_dec = |data: &[u8], l: &Limits| {
        let mut d = webp::Decoder::new();
        d.decode(data, l, &NO).unwrap();
        d
    };
    check_with(
        "Decoder::decode lossy+alpha",
        |l| warm_dec(&lossy_alpha[..], l),
        |d: &mut webp::Decoder, l| d.decode(&lossy_alpha, l, &NO).map(|i| i.pix.len()),
    );
    check_with(
        "Decoder::decode_planar lossy+alpha",
        |l| warm_dec(&lossy_alpha[..], l),
        |d: &mut webp::Decoder, l| d.decode_planar(&lossy_alpha, l, &NO).map(|p| p.y().len()),
    );
    check_with(
        "Decoder::decode lossless",
        |l| warm_dec(&ll[..], l),
        |d: &mut webp::Decoder, l| d.decode(&ll, l, &NO).map(|i| i.pix.len()),
    );

    // 524288 empty EXIF chunks: 4 MiB of file, 12 MiB of entries.
    let mut meta = b"RIFF\0\0\0\0WEBP".to_vec();
    for _ in 0..1 << 19 {
        meta.extend_from_slice(b"EXIF\0\0\0\0");
    }
    let n = (meta.len() - 8) as u32;
    meta[4..8].copy_from_slice(&n.to_le_bytes());
    check("metadata, many chunks", |l| {
        webp::metadata(&meta, l).map(|m| m.exif.len())
    });

    let opaque = image(w, h, false);
    let osrc = Source::Nrgba {
        pix: &opaque,
        stride: 4 * w,
    };
    let o = Options::default();
    check("encode", |l| webp::encode(w, h, &osrc, &o, l, &NO));
    // A reused Encoder, warmed by one call of the larger 1024x1024 image: some kept buffers are
    // dropped as too large, the rest are charged at their capacity. The output Vec is new, as a
    // warmed one keeps a capacity the call does not charge.
    let warm_enc = |src: &Source, l: &Limits| {
        let mut enc = webp::Encoder::new();
        enc.encode(bw, bh, src, &o, l, &NO).unwrap();
        enc
    };
    let small_call = |enc: &mut webp::Encoder, l: &Limits| {
        let mut out = Vec::new();
        enc.encode_into(w, h, &osrc, &o, l, &NO, &mut out)
            .map(|()| out)
    };
    check_with(
        "Encoder::encode_into, warmed by a larger image",
        |l| warm_enc(&bsrc, l),
        small_call,
    );
    // Warmed by a 1024x1024 solid image: partition 0 is kept larger than the textured call
    // needs, and the token partitions are kept tiny and must grow.
    let solid = [200u8, 50, 50, 255].repeat(bw * bh);
    let ssrc = Source::Nrgba {
        pix: &solid,
        stride: 4 * bw,
    };
    check_with(
        "Encoder::encode_into, warmed by a larger solid image",
        |l| warm_enc(&ssrc, l),
        small_call,
    );
    check("encode with alpha", |l| {
        webp::encode(w, h, &src, &o, l, &NO)
    });
    // Every ALPH writer (the default one, webp-go's original, and libwebp 1.2.2 and 1.6.0 at methods 3, 4
    // and 5, the last with 4x4 tiles, so up to 2600 histograms), on a textured plane and on one
    // that takes the palette path. Method 3 with 16 alpha values or fewer reserves an output
    // larger than it writes: its capacity must stay charged. 66K pixels: over `MIN_JOIN_PIXELS` (4K), so the ALPH chunk gets a thread.
    let (aw, ah) = (288, 232);
    let (tex, pal) = (image(aw, ah, true), palette_image(aw, ah));
    for (what, pix) in [("textured alpha", &tex), ("palette alpha", &pal)] {
        let s = Source::Nrgba {
            pix,
            stride: 4 * aw,
        };
        let mut opts = vec![
            ("default writer".to_string(), Options::default()),
            (
                "legacy_alpha".to_string(),
                Options {
                    match_libwebp: true,
                    legacy_alpha: true,
                    ..Options::default()
                },
            ),
        ];
        for alpha_libwebp16 in [false, true] {
            for method in [3, 4, 5] {
                opts.push((
                    format!("alpha_libwebp16 {alpha_libwebp16}, method {method}"),
                    Options {
                        match_libwebp: true,
                        method,
                        alpha_libwebp16,
                        ..Options::default()
                    },
                ));
            }
        }
        for (name, o) in &opts {
            check(&format!("encode, {what}, {name}"), |l| {
                webp::encode(aw, ah, &s, o, l, &NO)
            });
        }
    }

    reused_encoder_holds_only_what_it_charged();
}

/// What an `Encoder` keeps after a call under a small `max_alloc` is within that cap: the buffers
/// kept from an earlier, much larger image are charged at their capacity, so a call that cannot
/// afford them drops them (and runs again from scratch) instead of reusing them for free.
fn reused_encoder_holds_only_what_it_charged() {
    let (bw, bh) = (800, 600);
    let big = image(bw, bh, true);
    let bsrc = Source::Nrgba {
        pix: &big,
        stride: 4 * bw,
    };
    let (w, h) = (64, 48);
    let small = image(w, h, true);
    let ssrc = Source::Nrgba {
        pix: &small,
        stride: 4 * w,
    };
    let o = Options::default();
    let cap = 512 << 10;
    let limits = Limits {
        max_alloc: cap,
        max_threads: 1,
        ..Limits::default()
    };

    let before = LIVE.load(Relaxed);
    let mut enc = webp::Encoder::new();
    drop(
        enc.encode(bw, bh, &bsrc, &o, &Limits::default(), &NO)
            .unwrap(),
    );
    let kept = LIVE.load(Relaxed) - before;
    assert!(
        kept > cap + SLACK,
        "the encoder keeps only {kept} after the large call"
    );

    let out = enc.encode(w, h, &ssrc, &o, &limits, &NO).unwrap();
    drop(out);
    // What the encoder holds now, from before the large call: the small call dropped what was
    // too large for it, so this is below `kept`.
    let held = LIVE.load(Relaxed) - before;
    assert!(
        held <= cap + SLACK,
        "the encoder holds {held} after a call under max_alloc {cap}"
    );
}
