#![no_main]
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;
use webp::{Decoder, Error, Limits, Planar};

static NO: AtomicBool = AtomicBool::new(false);

// data = a WebP file. Its last three bytes (still part of the file, so seeds stay valid) choose the
// limits: bit 0 of the first byte picks tight ones (max_pixels up to 255,000, max_alloc up to 4 MiB),
// else generous ones. Decode, planar decode and the header agree: the planes convert, in bands of
// seven rows, to the RGB and alpha `decode` returns, and no limit is passed. A reused `Decoder`,
// warmed on files of the other kinds (lossless, lossy with alpha, a canvas), gives what a fresh
// call gives, for `decode` and `decode_planar`, with one thread so that the budget is deterministic.
const WARM: [&[u8]; 3] = [
    include_bytes!("../../data/ll_photo_z4.webp"),
    include_bytes!("../../data/lossy_alpha_photoa_best.webp"),
    include_bytes!("../../data/anim_lossy_full.webp"),
];

/// The planes, strides and alpha of a `Planar` as one comparable value.
fn planes(
    p: &Planar,
) -> (
    usize,
    usize,
    &[u8],
    &[u8],
    &[u8],
    usize,
    usize,
    Option<&[u8]>,
) {
    (
        p.width,
        p.height,
        p.y(),
        p.cb(),
        p.cr(),
        p.y_stride(),
        p.c_stride(),
        p.alpha.as_deref(),
    )
}

/// A reused result equals the fresh one, but for the budget: the reused decoder fits whenever the
/// fresh one does, not the reverse (and the two `MemoryBudget` errors can differ in their numbers).
fn same<T: PartialEq + std::fmt::Debug>(got: Result<T, Error>, want: Result<T, Error>) {
    match (&got, &want) {
        (Err(Error::MemoryBudget { .. }), Err(Error::MemoryBudget { .. }))
        | (Ok(_), Err(Error::MemoryBudget { .. })) => {}
        _ => assert_eq!(got, want, "reused decoder differs from a fresh one"),
    }
}

fuzz_target!(|data: &[u8]| {
    let k = data.last_chunk::<3>().copied().unwrap_or([0; 3]);
    let limits = if k[0] & 1 != 0 {
        Limits {
            max_pixels: u64::from(k[1]) * 1000,
            max_alloc: usize::from(u16::from_le_bytes([k[1], k[2]])) * 64,
            max_threads: 2,
            pool: None,
            max_metadata_bytes: 1 << 16,
        }
    } else {
        Limits {
            max_pixels: 1 << 20,
            max_alloc: 64 << 20,
            max_threads: 2,
            pool: None,
            max_metadata_bytes: 1 << 16,
        }
    };
    let cfg = webp::decode_config(data, &limits);
    if let Ok(c) = &cfg {
        // The canvas of a VP8X file goes up to 2^24 and `decode_config` enforces no limit (as in
        // Go): the side cap of 16383 applies to `decode`, below.
        assert!(c.width >= 1 && c.height >= 1);
    }
    let img = webp::decode(data, &limits, &NO);
    let planar = webp::decode_planar(data, &limits, &NO);
    if let Ok(i) = &img {
        let c = cfg
            .as_ref()
            .expect("decode succeeded, so the header parses");
        assert_eq!((i.width, i.height), (c.width, c.height));
        assert_eq!(i.pix.len(), 4 * i.width * i.height);
        assert!((i.width * i.height) as u64 <= limits.max_pixels);
        assert!(i.width <= 16383 && i.height <= 16383);
    }
    for e in [img.as_ref().err(), planar.as_ref().err()]
        .into_iter()
        .flatten()
    {
        assert!(*e != Error::Cancelled, "cancelled with the flag clear");
    }
    if let Ok(p) = &planar {
        let i = match &img {
            Ok(i) => i,
            // decode needs 5.5 bytes a pixel where planar needs 1.5: the tight limits may stop it.
            Err(e) if e.class() == "limit" => return,
            Err(e) => panic!("planar decoded, decode failed: {e}"),
        };
        let (w, h) = (p.width, p.height);
        assert_eq!((w, h), (i.width, i.height));
        let mut y0 = 0;
        while y0 < h {
            let y1 = (y0 + 7).min(h);
            let mut band = vec![0u8; 3 * w * (y1 - y0)];
            p.rgb(&mut band, y0, y1, &limits, &NO)
                .expect("rgb of decoded planes");
            for (n, px) in band.chunks_exact(3).enumerate() {
                assert_eq!(px, &i.pix[4 * (y0 * w + n)..][..3]);
            }
            y0 = y1;
        }
        for n in 0..w * h {
            assert_eq!(p.alpha.as_ref().map_or(255, |a| a[n]), i.pix[4 * n + 3]);
        }
        // Rows out of range or a buffer of the wrong size are an error, not a panic.
        assert!(p.rgb(&mut [], 1, 0, &limits, &NO).is_err());
        assert!(p.rgb(&mut [0u8; 1], 0, h, &limits, &NO).is_err());
    }
    // A set cancel flag stops the decodes without a panic.
    let yes = AtomicBool::new(true);
    let _ = webp::decode(data, &limits, &yes);
    let _ = webp::decode_planar(data, &limits, &yes);

    let one = Limits {
        max_threads: 1,
        ..limits
    };
    let mut dec = Decoder::new();
    for warm in WARM {
        let _ = dec.decode(warm, &Limits::default(), &NO);
        let _ = dec.decode_planar(warm, &Limits::default(), &NO);
    }
    for _ in 0..2 {
        same(
            dec.decode(data, &one, &NO).cloned(),
            webp::decode(data, &one, &NO),
        );
        let want = webp::decode_planar(data, &one, &NO);
        match (dec.decode_planar(data, &one, &NO), &want) {
            (Ok(g), Ok(w)) => assert_eq!(planes(g), planes(w), "reused planes differ"),
            (g, w) => same(g.map(|_| ()), w.as_ref().map(|_| ()).map_err(Clone::clone)),
        }
    }
});
