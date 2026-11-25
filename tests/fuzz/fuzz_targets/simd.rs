#![no_main]
use libfuzzer_sys::fuzz_target;

/// libwebp's `VP8RGBToY` with its rounding: the definition `luma_row` must equal.
fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((16839 * i64::from(r) + 33059 * i64::from(g) + 6420 * i64::from(b) + (1 << 15) + (16 << 16))
        >> 16) as u8
}

/// The definition of the sum of squared differences, one byte at a time.
fn sse(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i64::from(x) - i64::from(y);
            (d * d) as u64
        })
        .sum()
}

/// A runner without AVX2 (or NEON on arm64) would fuzz the scalar tier only: fail instead. Logs
/// every kernel that is fuzzed.
fn check_tier() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        #[cfg(target_arch = "x86_64")]
        assert!(
            is_x86_feature_detected!("avx2"),
            "no AVX2: scalar tier only"
        );
        #[cfg(target_arch = "aarch64")]
        assert!(
            std::arch::is_aarch64_feature_detected!("neon"),
            "no NEON: scalar tier only"
        );
        let names: Vec<_> = webp::fuzz_api::kernels().iter().map(|k| k.0).collect();
        assert!(names.len() >= 2, "kernels() lacks a SIMD tier: {names:?}");
        eprintln!("kernels fuzzed: {names:?}");
    });
}

// data = [kernel: u8][...]. Kernel 0..=5 are the encoder kernel groups (`enc_i4`, `enc_pred`,
// `enc_block`, `enc_collect`, `enc_chroma`, `enc_cost`): the rest decodes into one kernel's inputs, and every
// tier this CPU has must equal the scalar reference. Any other value is the colour conversion and
// the plain SSE: [start: u8][bytes], every tier on the bytes from `start % 32` (so any alignment):
// `luma_row` over the RGBA pixels equals the definition, and `sse_row` over the two halves equals
// the definition.
fuzz_target!(|data: &[u8]| {
    check_tier();
    let Some((&kernel, data)) = data.split_first() else {
        return;
    };
    if kernel % 8 < 6 {
        return webp::fuzz_api::kernel(kernel % 8, data);
    }
    let Some((&start, rest)) = data.split_first() else {
        return;
    };
    let bytes = &rest[usize::from(start % 32).min(rest.len())..];
    let px = bytes.len() / 4;
    let src = &bytes[..4 * px];
    let want: Vec<u8> = src
        .chunks_exact(4)
        .map(|p| luma(p[0], p[1], p[2]))
        .collect();
    let h = bytes.len() / 2;
    let (a, b) = (&bytes[..h], &bytes[h..2 * h]);
    let want_sse = sse(a, b);
    for (name, luma_row, sse_row) in webp::fuzz_api::kernels() {
        let mut got = vec![0u8; px];
        luma_row(&mut got, src);
        assert_eq!(got, want, "{name} luma");
        assert_eq!(sse_row(a, b), want_sse, "{name} sse");
    }
});
