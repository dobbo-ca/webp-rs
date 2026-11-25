#![no_main]
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;
use webp::{Encoder, Error, Limits, Options, Source};

#[path = "../common.rs"]
mod common;

// data = 3 x [kind][method][flags][segments][quality][w][h][max_alloc: u16 LE][pixels, cycled]. One
// Encoder serves the three calls, which differ in size, source kind, options and limits (flags
// bit 7: a tight max_alloc of 64 * the u16 bytes, bit 6: the cancel flag set). Each result must
// equal the result of a fresh `webp::encode` of the same call: stale buffers, the ALPH state kept
// between calls, the retry after a MemoryBudget error and the use after an error must not show.
// The one-thread call writes into an output Vec kept across the three calls (`encode_into`).
fuzz_target!(|data: &[u8]| {
    common::check_tier();
    const H: usize = 9;
    if data.len() < 3 * H {
        return;
    }
    let (heads, rest) = data.split_at(3 * H);
    let mut enc = Encoder::new();
    let mut out = Vec::new();
    for hd in heads.chunks_exact(H) {
        let [kind, method, flags, seg, q, wb, hb, a0, a1] = hd else {
            unreachable!()
        };
        let (w, h) = (1 + usize::from(wb % 96), 1 + usize::from(hb % 96));
        let o = Options {
            quality: f32::from(*q),
            method: i32::from(method % 7),
            match_libwebp: flags & 1 != 0,
            alpha_libwebp16: flags & 2 != 0,
            legacy_alpha: flags & 4 != 0,
            segments: [0, 1, 4][usize::from(seg % 3)],
        };
        let n = 6 * w * h;
        let buf: Vec<u8> = if rest.is_empty() {
            vec![0; n]
        } else {
            (0..n).map(|i| rest[i % rest.len()]).collect()
        };
        let src = match kind & 3 {
            0 => Source::Nrgba {
                pix: &buf,
                stride: 4 * w,
            },
            1 => Source::Rgba {
                pix: &buf,
                stride: 4 * w,
            },
            2 => Source::Gray {
                pix: &buf,
                stride: w,
            },
            _ => Source::Ycbcr420 {
                y: &buf,
                cb: &buf,
                cr: &buf,
                y_stride: w,
                c_stride: w.div_ceil(2),
            },
        };
        let limits = Limits {
            max_alloc: if flags & 0x80 != 0 {
                usize::from(u16::from_le_bytes([*a0, *a1])) * 64
            } else {
                256 << 20
            },
            max_threads: 1,
            ..Limits::default()
        };
        let cancel = AtomicBool::new(flags & 0x40 != 0);
        let got = enc
            .encode_into(w, h, &src, &o, &limits, &cancel, &mut out)
            .map(|()| out.clone());
        assert!(
            got.is_ok() || out.is_empty(),
            "encode_into leaves output after an error"
        );
        let want = webp::encode(w, h, &src, &o, &limits, &cancel);
        // One thread: the budget charge, and so Ok or Err, is deterministic for each side. The
        // two sides charge differently near the limit: a fresh encode charges the old and the new
        // block together while its buffers grow, and a reused Encoder charges its kept buffers at
        // their capacity (and retries from scratch if they do not fit). The Encoder promises that
        // it fits whenever `encode` fits, not the reverse, so a reused Ok against a fresh
        // MemoryBudget is allowed (seed tests/fuzz/seeds/encode_reuse/reused_fits_fresh_does_not). The
        // `got` of two budget errors can differ: only the variant is compared. Every other pair
        // of results must be equal.
        match (&got, &want) {
            (Err(Error::MemoryBudget { .. }), Err(Error::MemoryBudget { .. }))
            | (Ok(_), Err(Error::MemoryBudget { .. })) => {}
            _ => assert_eq!(got, want, "reused encoder differs from a fresh one"),
        }
        // Two threads: with a tight max_alloc, Ok or MemoryBudget depends on thread timing, so
        // only two Ok outputs are compared.
        let limits2 = Limits {
            max_threads: 2,
            ..limits
        };
        let got2 = enc.encode(w, h, &src, &o, &limits2, &cancel);
        let want2 = webp::encode(w, h, &src, &o, &limits2, &cancel);
        if let (Ok(g), Ok(w2)) = (&got2, &want2) {
            assert_eq!(g, w2, "reused threaded encoder differs from a fresh one");
        }
    }
});
