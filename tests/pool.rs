//! The shared pool is built only when a call needs it: not for `max_threads` 1, not when the
//! caller passes a pool, not for a call with no work for a second thread (metadata, lossless
//! decode, a small opaque encode, a bad file), and then on the first call with `max_threads`
//! above 1, no pool, and work to hand out. One
//! test fn, in its own binary, because the pool is process-wide and the order matters.

mod common;

use common::read;
use std::sync::atomic::AtomicBool;
use webp::{Limits, Options, Source, ThreadPool, shared_pool_started};

static NO: AtomicBool = AtomicBool::new(false);

#[test]
fn shared_pool_is_lazy_and_a_callers_pool_wins() {
    // Over 4K pixels with alpha and over 128 KiB of RGBA, so both stages could use the pool.
    let (w, h) = (300, 230);
    let pix: Vec<u8> = (0..4 * w * h).map(|i| (i * 31 / 7) as u8).collect();
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let lossy = read("lossy_alpha_photoa_best.webp");
    let limits = |max_threads, pool| Limits {
        max_threads,
        pool,
        ..Limits::default()
    };
    assert!(!shared_pool_started());

    // One thread (and 0): serial, no pool.
    let want = webp::encode(w, h, &src, &Options::default(), &limits(1, None), &NO).unwrap();
    let img = webp::decode(&lossy, &limits(1, None), &NO).unwrap();
    webp::decode(&lossy, &limits(0, None), &NO).unwrap();
    let p = webp::decode_planar(&lossy, &limits(1, None), &NO).unwrap();
    let mut rgb = vec![0u8; 3 * p.width * p.height];
    p.rgb(&mut rgb, 0, p.height, &limits(1, None), &NO).unwrap();
    assert!(
        !shared_pool_started(),
        "max_threads 1 built the shared pool"
    );

    // A caller's pool, with and without a limit of 1: the shared pool stays unbuilt.
    let mine = ThreadPool::new(2).unwrap();
    assert_eq!(mine.threads(), 2);
    for n in [1, 2, 8] {
        let l = limits(n, Some(mine.clone()));
        let got = webp::encode(w, h, &src, &Options::default(), &l, &NO).unwrap();
        assert!(got == want, "encode, {n} threads, caller pool");
        assert!(
            webp::decode(&lossy, &l, &NO).unwrap() == img,
            "decode, {n} threads"
        );
        let mut again = vec![0u8; rgb.len()];
        p.rgb(&mut again, 0, p.height, &l, &NO).unwrap();
        assert!(again == rgb, "rgb, {n} threads, caller pool");
    }
    assert!(!shared_pool_started(), "a caller's pool did not win");

    // More than one thread but nothing to hand to a worker: the shared pool stays unbuilt. An
    // opaque image of fewer than 64 macroblocks (here 7x6) encodes its MB rows on one thread.
    let (ow, oh) = (112, 96);
    let opaque: Vec<u8> = pix[..4 * ow * oh]
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b, 255])
        .collect();
    let osrc = Source::Nrgba {
        pix: &opaque,
        stride: 4 * ow,
    };
    webp::encode(ow, oh, &osrc, &Options::default(), &limits(8, None), &NO).unwrap();
    webp::metadata(&lossy, &limits(8, None)).unwrap();
    webp::decode(&read("ll_photo_z4.webp"), &limits(8, None), &NO).unwrap();
    assert!(webp::decode(b"RIFF\0\0\0\0WEBPjunk", &limits(8, None), &NO).is_err());
    let tiny = vec![9u8; 4 * 16 * 16];
    let tsrc = Source::Nrgba {
        pix: &tiny,
        stride: 4 * 16,
    };
    webp::encode(16, 16, &tsrc, &Options::default(), &limits(8, None), &NO).unwrap();
    assert!(
        !shared_pool_started(),
        "a call with no work for a second thread built the shared pool"
    );

    // No pool and more than one thread: the shared pool is built, and the output is the same.
    let got = webp::encode(w, h, &src, &Options::default(), &limits(2, None), &NO).unwrap();
    assert!(got == want);
    assert!(shared_pool_started());
    assert!(webp::decode(&lossy, &limits(2, None), &NO).unwrap() == img);
}
