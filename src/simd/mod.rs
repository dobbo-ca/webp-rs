//! SIMD kernels, each beside its scalar reference: AVX2 and AVX-512 on x86_64, NEON on aarch64. The
//! tier is picked once, at first use, from the CPU; `WEBP_SIMD=scalar` forces the scalar one and
//! `WEBP_SIMD=avx2` rules out AVX-512; `WEBP_SIMD=avx512` requires it and panics on a CPU without
//! it (for CI under an emulator). The AVX-512 tier (`avx512.rs`) holds only the kernels that
//! gain from 512-bit registers and takes every other kernel from AVX2.
//!
//! The first kernels are the RGBA to luma row of the encoder's colour conversion and the sum of
//! squared differences of a prediction. The encoder kernel groups (`enc_i4`, `enc_pred`,
//! `enc_block`, `enc_collect`, `enc_chroma`, `enc_cost`) follow Go's block kernels (IDCT, quantizer, costs,
//! i4 trials, analysis bins), which the bench shows do pay.
//!
//! Layout of a tier (`avx2.rs`, `avx512.rs`, `neon.rs`), so that `unsafe` stays in two small places:
//!
//! - The kernels are safe `#[target_feature(enable = "avx2")]` (NEON: `"neon"`) fns with no
//!   `unsafe` inside. Non-pointer intrinsics are safe calls there; slices become fixed-size arrays
//!   with `as_chunks`/`first_chunk` in safe code.
//! - `avx2/mem.rs`, `avx512/mem.rs` and `neon/mem.rs` hold the load and store helpers (`load256(&[u8; 32])`,
//!   `storel64(&mut [u8; 8], v)`, `vld1q_u8` wrappers, and the i16/i32 forms). The array reference
//!   type proves the bounds; each `unsafe` block has a SAFETY comment. These modules and the
//!   dispatch fns are the only `#[allow(unsafe_code)]` in the crate.
//! - The dispatch: `detect` in the tier file checks the CPU feature and returns fn pointers to
//!   two small wrappers, which call the kernels inside `unsafe` with a SAFETY comment that names
//!   the detection. It runs once (`selected`).

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;

pub(crate) mod enc_block;
pub(crate) mod enc_chroma;
pub(crate) mod enc_collect;
pub(crate) mod enc_cost;
pub(crate) mod enc_i4;
pub(crate) mod enc_pred;
pub(crate) mod kbench;

use std::sync::OnceLock;

/// One tier's kernels. `luma_row` gets `src` of exactly four bytes a `dst` byte; `sse_row` gets
/// slices of equal length.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) luma_row: fn(&mut [u8], &[u8]),
    // Read by the tests and the fuzz API only; the encoder's block SSE is in `enc_pred`.
    #[allow(dead_code)]
    pub(crate) sse_row: fn(&[u8], &[u8]) -> u64,
}

pub(crate) const SCALAR: Kernels = Kernels {
    luma_row: scalar::luma_row,
    sse_row: scalar::sse_row,
};

/// The SIMD tier this CPU supports, if any: AVX-512 when the CPU has it and `WEBP_SIMD=avx2` does
/// not rule it out, else AVX2.
#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Kernels> {
    avx512_wanted()
        .then(avx512::detect)
        .flatten()
        .or_else(avx2::detect)
}
#[cfg(target_arch = "aarch64")]
use neon::detect;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Kernels> {
    None
}

/// Whether `WEBP_SIMD=scalar` forces the scalar tiers. Read once.
pub(crate) fn scalar_forced() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "scalar"))
}

/// Whether the AVX-512 kernels run here: false on a target without them.
fn avx512_kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx512::detect().is_some()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether `WEBP_SIMD=avx512` requires the AVX-512 tier (for CI under an emulator). It panics on a
/// CPU without the AVX-512 kernels, so a test run cannot fall back to AVX2 without a word. Read once.
fn avx512_forced() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        let forced = std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "avx512");
        assert!(
            !forced || avx512_kernels(),
            "WEBP_SIMD=avx512, but this CPU lacks AVX-512 F, BW and VL"
        );
        forced
    })
}

/// Whether the AVX-512 tier may be picked: `WEBP_SIMD=avx2` rules it out. Read once.
#[cfg(target_arch = "x86_64")]
fn avx512_wanted() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        avx512_forced() || !std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "avx2")
    })
}

fn selected() -> Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    *K.get_or_init(|| {
        avx512_forced();
        if scalar_forced() {
            SCALAR
        } else {
            detect().unwrap_or(SCALAR)
        }
    })
}

/// `dst[x]` is the luma of the RGBA pixel `src[4x..4x + 4]`, libwebp's `VP8RGBToY` with rounding
/// 16 << 16 + 1 << 15. `src` holds at least `4 * dst.len()` bytes.
pub(crate) fn luma_row(dst: &mut [u8], src: &[u8]) {
    let src = &src[..4 * dst.len()];
    (selected().luma_row)(dst, src);
}

/// The sum of (a - b)^2 over equal-length slices.
#[allow(dead_code)]
pub(crate) fn sse_row(a: &[u8], b: &[u8]) -> u64 {
    (selected().sse_row)(a, b)
}

/// Every SIMD tier this CPU can run, whichever `selected` picks. A CPU without AVX-512 lists AVX2
/// only.
pub(crate) fn tiers() -> Vec<(&'static str, Kernels)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.extend(avx2::detect().map(|k| ("avx2", k)));
        v.extend(avx512::detect().map(|k| ("avx512", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(neon::detect().map(|k| ("neon", k)));
    v
}

/// The bytes a fuzz target decodes its kernel inputs from; reads past the end give zeros.
#[cfg(fuzzing)]
pub(crate) struct Input<'a>(pub(crate) &'a [u8]);

#[cfg(fuzzing)]
impl<'a> Input<'a> {
    pub(crate) fn u8(&mut self) -> u8 {
        let Some((&b, rest)) = self.0.split_first() else {
            return 0;
        };
        self.0 = rest;
        b
    }
    pub(crate) fn fill(&mut self, out: &mut [u8]) {
        out.iter_mut().for_each(|o| *o = self.u8());
    }
    pub(crate) fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = vec![0; n];
        self.fill(&mut v);
        v
    }
    pub(crate) fn rest(&mut self) -> &'a [u8] {
        std::mem::take(&mut self.0)
    }
    pub(crate) fn u32(&mut self) -> u32 {
        u32::from_le_bytes(std::array::from_fn(|_| self.u8()))
    }
    /// The four row words of a 4x4 block.
    pub(crate) fn rows(&mut self) -> [u32; 4] {
        std::array::from_fn(|_| self.u32())
    }
    pub(crate) fn z(&mut self) -> [[u8; 32]; 26] {
        std::array::from_fn(|_| std::array::from_fn(|_| self.u8()))
    }
    /// A block of coefficients: all zero, DC only, small, up to +-2048, or the full i16 range.
    pub(crate) fn coeffs(&mut self) -> [i16; 16] {
        let kind = self.u8() % 5;
        let mut c = [0i16; 16];
        for (i, v) in c.iter_mut().enumerate() {
            *v = match kind {
                0 => 0,
                1 if i > 0 => 0,
                2 => i16::from(self.u8() as i8),
                3 => 16 * i16::from(self.u8() as i8),
                _ => i16::from_le_bytes([self.u8(), self.u8()]),
            };
        }
        c
    }
}

/// Runs the differential check of the kernel group `group % 6` on `data`: every SIMD tier this
/// CPU has against the scalar reference, bit for bit. Panics on a difference.
#[cfg(fuzzing)]
pub(crate) fn fuzz(group: u8, data: &[u8]) {
    let r = &mut Input(data);
    match group % 6 {
        0 => enc_i4::fuzz(r),
        1 => enc_pred::fuzz(r),
        2 => enc_block::fuzz(r),
        3 => enc_collect::fuzz(r),
        4 => enc_chroma::fuzz(r),
        _ => enc_cost::fuzz(r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u8 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 24) as u8
    }

    /// `luma_row` and `sse_row` run the SIMD tier unless `WEBP_SIMD=scalar` says otherwise. Also
    /// logs the tier in use (`--nocapture`), so a CI log shows what the run tested.
    #[test]
    fn dispatch_picks_tier() {
        let scalar = std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "scalar");
        let sel = selected();
        // The tier in use: the last of `tiers()` whose kernels `selected` holds, else scalar.
        let in_use = tiers()
            .into_iter()
            .rev()
            .find(|(_, k)| k.luma_row as usize == sel.luma_row as usize)
            .map_or("scalar", |(name, _)| name);
        eprintln!("SIMD tier: {in_use}, AVX-512 kernels: {}", avx512_kernels());
        if std::env::var_os("WEBP_SIMD").is_some_and(|v| v == "avx512") {
            assert_eq!(
                in_use, "avx512",
                "WEBP_SIMD=avx512, but the tier in use differs"
            );
        }
        assert_eq!(
            sel.luma_row as usize == SCALAR.luma_row as usize,
            scalar || detect().is_none()
        );
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(scalar || sel.luma_row as usize != SCALAR.luma_row as usize);
    }

    #[test]
    fn simd_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        assert!(avx2::detect().is_some(), "AVX2 expected on x86_64");
        #[cfg(target_arch = "aarch64")]
        assert!(neon::detect().is_some(), "NEON expected on aarch64");
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let rand: Vec<u8> = (0..4 * 70_000 + 8).map(|_| xorshift(&mut st)).collect();
        let ff = vec![0xffu8; rand.len()];
        let zero = vec![0u8; rand.len()];
        let lens = [
            0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 63, 64, 65, 100, 255, 256,
            257, 4096, 65535, 65536, 65537, 65600, 70_000,
        ];
        for (name, k) in tiers() {
            for buf in [&rand, &ff, &zero] {
                for off in [0, 1, 3] {
                    for &n in &lens {
                        let src = &buf[off..off + 4 * n];
                        let (mut want, mut got) = (vec![0u8; n], vec![0u8; n]);
                        scalar::luma_row(&mut want, src);
                        (k.luma_row)(&mut got, src);
                        assert_eq!(got, want, "{name} luma off {off} n {n}");
                        let (a, b) = (&buf[off..off + n], &rand[3 + off..3 + off + n]);
                        assert_eq!(
                            (k.sse_row)(a, b),
                            scalar::sse_row(a, b),
                            "{name} sse off {off} n {n}"
                        );
                    }
                }
            }
            // The largest possible sums: all 255 against all 0.
            for n in [8, 16, 65536, 70_000] {
                assert_eq!(
                    (k.sse_row)(&ff[..n], &zero[..n]),
                    255 * 255 * n as u64,
                    "{name} sse max n {n}"
                );
            }
            for n in 0..300 {
                let (a, b) = (&rand[n..2 * n], &rand[7..7 + n]);
                assert_eq!((k.sse_row)(a, b), scalar::sse_row(a, b), "{name} sse n {n}");
            }
        }
    }
}
