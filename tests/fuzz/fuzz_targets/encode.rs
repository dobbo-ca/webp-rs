#![no_main]
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;
use webp::{Error, Limits, Options, Source};

#[path = "../common.rs"]
mod common;

static NO: AtomicBool = AtomicBool::new(false);

// data = [kind][method][flags][segments][quality: f32 LE][w][h][pixels, cycled to fit]. kind 0 to 3
// is NRGBA, RGBA, Gray and YCbCr 4:2:0 (kind & 3); bit 7 makes the size hostile (up to 24,735 a
// side, with the few pixels given, so the call must refuse it before it allocates from it). The
// method and segments are any i8, the quality any f32 (NaN and infinities too), flags bits 0 to
// 2 are the options, bits 6 and 7 the stride padding of a row. Whatever the options, a valid call
// encodes to a file that decodes to the same size, and encoding again gives the same bytes.
fuzz_target!(|data: &[u8]| {
    common::check_tier();
    let Some((&[kind, method, flags, seg, q0, q1, q2, q3, wb, hb], rest)) =
        data.split_first_chunk::<10>()
    else {
        return;
    };
    let hostile = kind & 0x80 != 0;
    let (w, h) = if hostile {
        (usize::from(wb) * 97, usize::from(hb) * 97)
    } else {
        (1 + usize::from(wb % 80), 1 + usize::from(hb % 80))
    };
    let o = Options {
        quality: f32::from_bits(u32::from_le_bytes([q0, q1, q2, q3])),
        method: i32::from(method as i8),
        match_libwebp: flags & 1 != 0,
        alpha_libwebp16: flags & 2 != 0,
        legacy_alpha: flags & 4 != 0,
        segments: i32::from(seg as i8),
    };
    let pad = usize::from(flags >> 6);
    let fill = |n: usize| -> Vec<u8> {
        if rest.is_empty() {
            vec![0; n]
        } else {
            (0..n).map(|i| rest[i % rest.len()]).collect()
        }
    };
    // The bytes a plane of `rows` rows of `row` samples, `stride` apart, needs.
    let plane = |rows: usize, row: usize, stride: usize| (rows.max(1) - 1) * stride + row;
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let row = if kind & 3 < 2 { 4 * w } else { w };
    let (stride, cstride) = (row + pad, cw + pad);
    let bufs = [
        fill(if hostile {
            rest.len()
        } else {
            plane(h, stride, stride)
        }),
        fill(if hostile {
            0
        } else {
            plane(ch, cstride, cstride)
        }),
        fill(if hostile {
            0
        } else {
            plane(ch, cstride, cstride)
        }),
    ];
    let src = match kind & 3 {
        0 => Source::Nrgba {
            pix: &bufs[0],
            stride,
        },
        1 => Source::Rgba {
            pix: &bufs[0],
            stride,
        },
        2 => Source::Gray {
            pix: &bufs[0],
            stride,
        },
        _ => Source::Ycbcr420 {
            y: &bufs[0],
            cb: &bufs[1],
            cr: &bufs[2],
            y_stride: stride,
            c_stride: cstride,
        },
    };
    let limits = Limits {
        max_alloc: 256 << 20,
        max_threads: 2,
        ..Limits::default()
    };
    match webp::encode(w, h, &src, &o, &limits, &NO) {
        Ok(file) => {
            let img = webp::decode(&file, &Limits::default(), &NO).expect("own output decodes");
            assert_eq!((img.width, img.height), (w, h));
            let again = webp::encode(w, h, &src, &o, &limits, &NO);
            assert_eq!(again.as_deref(), Ok(&file[..]), "not deterministic");
        }
        Err(Error::InvalidArgument(_)) => {
            assert!(
                hostile || !matches!(o.segments, 0 | 1 | 4),
                "valid call refused"
            );
        }
        Err(Error::MemoryBudget { .. } | Error::MaxPixels { .. }) => {
            assert!(hostile, "limit error for a small image");
        }
        Err(e) => panic!("{e}"),
    }
});
