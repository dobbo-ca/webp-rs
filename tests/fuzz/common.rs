//! Shared by the encode targets (`#[path = "../common.rs"] mod common;`).

/// Once per process: on x86_64 unless `WEBP_SIMD=scalar`, the CPU must have AVX2, or the `default`
/// leg would fuzz the scalar kernels only. Logs which tier runs.
pub fn check_tier() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let scalar = std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "scalar");
        #[cfg(target_arch = "x86_64")]
        if !scalar {
            assert!(
                is_x86_feature_detected!("avx2"),
                "no AVX2: the AVX2 kernels would go unfuzzed"
            );
        }
        eprintln!(
            "encoder kernels fuzzed: {}",
            if scalar || !cfg!(target_arch = "x86_64") {
                "scalar"
            } else {
                "avx2"
            }
        );
    });
}
