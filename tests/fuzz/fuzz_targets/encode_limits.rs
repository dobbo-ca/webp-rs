#![no_main]
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;
use webp::{Error, Limits, Options, Source};

#[path = "../common.rs"]
mod common;

static NO: AtomicBool = AtomicBool::new(false);

// data = [kind][method][flags][segments][quality][w][h][max_alloc: u16 LE][max_pixels: u16 LE][pixels,
// cycled]. Images of 64 pixels and more take the threaded path (MIN_JOIN_PIXELS under fuzzing). With the
// limits tight the call either gives the bytes of the call with the default limits or a limit
// error (always one when the image is over max_pixels), and with the cancel flag set it gives
// Cancelled.
fuzz_target!(|data: &[u8]| {
    common::check_tier();
    let Some((&[kind, method, flags, seg, q, wb, hb, a0, a1, p0, p1], rest)) =
        data.split_first_chunk::<11>()
    else {
        return;
    };
    let (w, h) = (1 + usize::from(wb % 64), 1 + usize::from(hb % 64));
    let o = Options {
        quality: f32::from(q),
        method: i32::from(method % 7),
        match_libwebp: flags & 1 != 0,
        alpha_libwebp16: flags & 2 != 0,
        legacy_alpha: flags & 4 != 0,
        segments: [0, 1, 4][usize::from(seg % 3)],
    };
    let n = 4 * w * h + 2 * w * h;
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
    let tight = Limits {
        max_pixels: u64::from(u16::from_le_bytes([p0, p1])),
        max_alloc: usize::from(u16::from_le_bytes([a0, a1])) * 64,
        max_threads: 2,
        pool: None,
        max_metadata_bytes: 0,
    };
    let full = Limits::default();
    let want = webp::encode(w, h, &src, &o, &full, &NO);
    let got = webp::encode(w, h, &src, &o, &tight, &NO);
    match &got {
        Ok(_) => assert_eq!(got, want, "tight limits changed the output"),
        Err(e) if e.class() == "limit" => {}
        Err(e) => panic!("{e}"),
    }
    if ((w * h) as u64) > tight.max_pixels {
        assert!(
            matches!(got, Err(Error::MaxPixels { .. })),
            "over max_pixels: {got:?}"
        );
    }
    let yes = AtomicBool::new(true);
    assert_eq!(
        webp::encode(w, h, &src, &o, &full, &yes),
        Err(Error::Cancelled)
    );
});
