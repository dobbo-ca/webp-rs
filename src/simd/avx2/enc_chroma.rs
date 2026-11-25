//! AVX2 form of the encoder kernels of package E, the opaque chroma row conversion: Go's
//! `chromaAVX2Asm` (kernel_amd64.s), 8 blocks a pass.

use super::mem::*;
use crate::simd::enc_chroma::Kernels;
use std::arch::x86_64::*;
use std::sync::OnceLock;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels { chroma })
}

/// Dispatch: the kernel is a safe `#[target_feature(enable = "avx2")]` fn, so calling it from here
/// needs `unsafe`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn chroma(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { chroma_kernel(cb, cr, p0, p1, to_linear, to_gamma) }
}

/// linearToGamma[0..32] and its steps linearToGamma[p + 1] - linearToGamma[p], as bytes: the low
/// and high 16 entries of each. Derived once from libwebp's table, whose entries and steps fit.
fn tables(to_gamma: &[i32; 33]) -> &'static [[u8; 16]; 4] {
    static T: OnceLock<[[u8; 16]; 4]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0u8; 16]; 4];
        for p in 0..32 {
            t[p / 16][p % 16] = to_gamma[p] as u8;
            t[2 + p / 16][p % 16] = (to_gamma[p + 1] - to_gamma[p]) as u8;
        }
        t
    })
}

/// Two 16-bit constants as the i32 lane `madd_epi16` multiplies pairs by.
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as i32) << 16)
}

/// linToGamma of the eight i32 sums in `s`, as 4 * L[p] + (D[p] * x + 64) >> 7 with p = s >> 9
/// and x = s & 511.
#[target_feature(enable = "avx2")]
fn lin_to_gamma(s: __m256i, t: &[__m256i; 4]) -> __m256i {
    let p = _mm256_srli_epi32::<9>(s);
    // Byte 0 of each lane selects from the low table when p < 16 and the high one otherwise: the
    // other bytes and the wrong table's index have bit 7 set, which pshufb turns into zero.
    let lo = _mm256_add_epi8(p, _mm256_set1_epi32(0x8080_8070u32 as i32));
    let hi = _mm256_add_epi8(p, _mm256_set1_epi32(0x8080_80f0u32 as i32));
    let l = _mm256_or_si256(_mm256_shuffle_epi8(t[0], lo), _mm256_shuffle_epi8(t[1], hi));
    let d = _mm256_or_si256(_mm256_shuffle_epi8(t[2], lo), _mm256_shuffle_epi8(t[3], hi));
    let x = _mm256_and_si256(s, _mm256_set1_epi32(511));
    let dx = _mm256_madd_epi16(x, d);
    let r = _mm256_srli_epi32::<7>(_mm256_add_epi32(dx, _mm256_set1_epi32(64)));
    _mm256_add_epi32(r, _mm256_slli_epi32::<2>(l))
}

/// linToGamma of one channel (the byte at bit `S`) summed over the 2x2 blocks of the 16 pixel
/// pairs a0/a1 (row 0, pixels 0-7 and 8-15) and b0/b1 (row 1). The dwords are the blocks
/// 0 1 4 5 | 2 3 6 7.
#[target_feature(enable = "avx2")]
fn channel<const S: i32>(
    a: [__m256i; 2],
    b: [__m256i; 2],
    to_linear: &[i32; 256],
    t: &[__m256i; 4],
) -> __m256i {
    let s0 = _mm256_add_epi32(
        gather256_i32(to_linear, _mm256_srli_epi32::<S>(a[0])),
        gather256_i32(to_linear, _mm256_srli_epi32::<S>(b[0])),
    );
    let s1 = _mm256_add_epi32(
        gather256_i32(to_linear, _mm256_srli_epi32::<S>(a[1])),
        gather256_i32(to_linear, _mm256_srli_epi32::<S>(b[1])),
    );
    lin_to_gamma(_mm256_hadd_epi32(s0, s1), t)
}

/// One chroma value of the eight blocks: (R, G) and B weighted, rounded and shifted.
#[target_feature(enable = "avx2")]
fn weigh(rg: __m256i, b: __m256i, k_rg: __m256i, k_b: __m256i, round: __m256i) -> __m256i {
    let s = _mm256_add_epi32(_mm256_madd_epi16(rg, k_rg), _mm256_madd_epi16(b, k_b));
    _mm256_srai_epi32::<18>(_mm256_add_epi32(s, round))
}

/// The chroma of `cb.len()` blocks, 8 a pass.
#[target_feature(enable = "avx2")]
fn chroma_kernel(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    let tb = tables(to_gamma);
    let t = [
        _mm256_broadcastsi128_si256(load128(&tb[0])),
        _mm256_broadcastsi128_si256(load128(&tb[1])),
        _mm256_broadcastsi128_si256(load128(&tb[2])),
        _mm256_broadcastsi128_si256(load128(&tb[3])),
    ];
    let (k_cb_rg, k_cb_b) = (
        _mm256_set1_epi32(pair(-9719, -19081)),
        _mm256_set1_epi32(28800),
    );
    let (k_cr_rg, k_cr_b) = (
        _mm256_set1_epi32(pair(28800, -24116)),
        _mm256_set1_epi32(pair(-4684, 0)),
    );
    let round = _mm256_set1_epi32((1 << 17) + (128 << 18));
    let mut alpha = _mm256_set1_epi32(-1);
    let rows = p0
        .as_chunks::<32>()
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .zip(p1.as_chunks::<32>().0.as_chunks::<2>().0.iter());
    let outs = cb
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(cr.as_chunks_mut::<8>().0.iter_mut());
    for ((r0, r1), (o_cb, o_cr)) in rows.zip(outs) {
        let a = [load256(&r0[0]), load256(&r0[1])];
        let b = [load256(&r1[0]), load256(&r1[1])];
        for v in [a[0], a[1], b[0], b[1]] {
            alpha = _mm256_and_si256(alpha, v);
        }
        let r = channel::<0>(a, b, to_linear, &t);
        let g = channel::<8>(a, b, to_linear, &t);
        let bl = channel::<16>(a, b, to_linear, &t);
        let rg = _mm256_or_si256(r, _mm256_slli_epi32::<16>(g));
        // Cb and Cr lie in [16, 240], so the packs are exact. The blocks come out 0 1 4 5 | 2 3 6 7.
        let c0 = _mm256_permute4x64_epi64::<0xd8>(weigh(rg, bl, k_cb_rg, k_cb_b, round));
        let c1 = _mm256_permute4x64_epi64::<0xd8>(weigh(rg, bl, k_cr_rg, k_cr_b, round));
        let w = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(c0, c1));
        let bytes = _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w));
        storel64(o_cb, bytes);
        storel64(o_cr, _mm_unpackhi_epi64(bytes, bytes));
    }
    let mut a = [0u8; 32];
    store256(&mut a, alpha);
    a.iter().skip(3).step_by(4).fold(255, |x, &v| x & v)
}
