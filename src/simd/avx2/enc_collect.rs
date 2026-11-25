//! AVX2 form of the encoder kernels of package E, the segment analysis bins. The fdct of all
//! blocks of a macroblock runs at once, one block to a 16 bit lane, so that the row and column
//! passes are plain lane-wise arithmetic on registers that each hold one pixel position of every
//! block (Go's `collectAsm` does one block to a 128 bit lane instead).

use super::mem::*;
use crate::simd::enc_collect::{BinStats, Kernels, Z};
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels {
        collect16,
        collect_uv,
        bin_stats,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "avx2")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn collect16(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { collect16_kernel(src, stride, z) }
}

/// Dispatch, as `collect16`.
#[allow(unsafe_code)]
fn collect_uv(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { collect_uv_kernel(u, v, stride, z) }
}

/// Dispatch, as `collect16`.
#[allow(unsafe_code)]
fn bin_stats(b: &[u8]) -> BinStats {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { bin_stats_kernel(b) }
}

/// Two 16-bit constants as the i32 lane `madd_epi16` multiplies pairs by.
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as i32) << 16)
}

/// The 16 bytes of `s` from `o`.
fn at16(s: &[u8], o: usize) -> &[u8; 16] {
    &s[o..o + 16].as_chunks::<16>().0[0]
}

/// The 8 bytes of `s` from `o`.
fn at8(s: &[u8], o: usize) -> &[u8; 8] {
    &s[o..o + 8].as_chunks::<8>().0[0]
}

/// `a2 * k0 + a3 * k1 + round` shifted right by `S`, for the 16 word lanes, as words.
#[target_feature(enable = "avx2")]
fn rot<const S: i32>(a2: __m256i, a3: __m256i, k: i32, round: i32) -> __m256i {
    let (k, round) = (_mm256_set1_epi32(k), _mm256_set1_epi32(round));
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a2, a3), k);
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a2, a3), k);
    _mm256_packs_epi32(
        _mm256_srai_epi32::<S>(_mm256_add_epi32(lo, round)),
        _mm256_srai_epi32::<S>(_mm256_add_epi32(hi, round)),
    )
}

/// The bins before the clamp, `|c| >> 3`, of the fdct of the blocks in the lanes, from their
/// differences `d[j][c]` at row j, column c. Register `4 * c + k` is coefficient k of column c.
#[target_feature(enable = "avx2")]
fn fdct_bins(d: &[[__m256i; 4]; 4]) -> [__m256i; 16] {
    let (k_a, k_b) = (pair(2217, 5352), pair(-5352, 2217));
    // Row pass: t[j][c], dsp/enc.c FTransform_C.
    let mut t = [[_mm256_setzero_si256(); 4]; 4];
    for (t, d) in t.iter_mut().zip(d) {
        let (a0, a1) = (_mm256_add_epi16(d[0], d[3]), _mm256_add_epi16(d[1], d[2]));
        let (a2, a3) = (_mm256_sub_epi16(d[1], d[2]), _mm256_sub_epi16(d[0], d[3]));
        t[0] = _mm256_slli_epi16::<3>(_mm256_add_epi16(a0, a1));
        t[1] = rot::<9>(a2, a3, k_a, 1812);
        t[2] = _mm256_slli_epi16::<3>(_mm256_sub_epi16(a0, a1));
        t[3] = rot::<9>(a2, a3, k_b, 937);
    }
    // Column pass, then the bin.
    let zero = _mm256_setzero_si256();
    let (seven, one) = (_mm256_set1_epi16(7), _mm256_set1_epi16(1));
    let mut out = [zero; 16];
    for c in 0..4 {
        let (a0, a1) = (
            _mm256_add_epi16(t[0][c], t[3][c]),
            _mm256_add_epi16(t[1][c], t[2][c]),
        );
        let (a2, a3) = (
            _mm256_sub_epi16(t[1][c], t[2][c]),
            _mm256_sub_epi16(t[0][c], t[3][c]),
        );
        let c0 = _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(a0, a1), seven));
        let c2 = _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_sub_epi16(a0, a1), seven));
        // One more when a3 is not zero: 1 + (-1 where a3 is zero).
        let bump = _mm256_add_epi16(one, _mm256_cmpeq_epi16(a3, zero));
        let c1 = _mm256_add_epi16(rot::<16>(a2, a3, k_a, 12000), bump);
        let c3 = rot::<16>(a2, a3, k_b, 51000);
        for (k, v) in [c0, c1, c2, c3].into_iter().enumerate() {
            out[4 * c + k] = _mm256_srli_epi16::<3>(_mm256_abs_epi16(v));
        }
    }
    out
}

/// Row `j` of the four 4x4 blocks of a chroma plane (`src`, with the prediction at column `bx` of
/// `z`): the 16 bytes [block row 0: block 0 | block 1][block row 1: ...] of the source in the low
/// lane and of the prediction in the high one, turned from [block][column] to [column][block].
#[target_feature(enable = "avx2")]
fn uv_plane(src: &[u8], stride: usize, z: &Z, bx: usize, j: usize) -> __m256i {
    let s = |by: usize| loadl64(at8(src, (4 * by + j) * stride));
    let p = |by: usize| loadl64(at8(&z[18 + 4 * by + j], bx));
    _mm256_shuffle_epi8(
        _mm256_set_m128i(
            _mm_unpacklo_epi64(p(0), p(1)),
            _mm_unpacklo_epi64(s(0), s(1)),
        ),
        transpose4_pattern(),
    )
}

/// Bytes `4 * c + k` from bytes `4 * k + c` of each 128-bit lane.
#[target_feature(enable = "avx2")]
fn transpose4_pattern() -> __m256i {
    _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10,
        14, 3, 7, 11, 15,
    )
}

#[target_feature(enable = "avx2")]
fn collect16_kernel(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    let pat = transpose4_pattern();
    let zero = _mm256_setzero_si256();
    let mut d = [[zero; 4]; 4];
    for (j, d) in d.iter_mut().enumerate() {
        // Block row by of row j: source bytes in the low lane, prediction in the high one, laid
        // out as [block bx][column c] and turned into [column c][block bx].
        let mut x = [zero; 4];
        for (by, x) in x.iter_mut().enumerate() {
            let s = load128(at16(src, (4 * by + j) * stride));
            let p = load128(at16(&z[1 + 4 * by + j], 8));
            *x = _mm256_shuffle_epi8(_mm256_set_m128i(p, s), pat);
        }
        // Transpose the four dwords of the four block rows: register c gets column c of every
        // block, block 4 * by + bx in byte 4 * by + bx.
        let u0 = _mm256_unpacklo_epi32(x[0], x[1]);
        let u1 = _mm256_unpackhi_epi32(x[0], x[1]);
        let u2 = _mm256_unpacklo_epi32(x[2], x[3]);
        let u3 = _mm256_unpackhi_epi32(x[2], x[3]);
        let cols = [
            _mm256_unpacklo_epi64(u0, u2),
            _mm256_unpackhi_epi64(u0, u2),
            _mm256_unpacklo_epi64(u1, u3),
            _mm256_unpackhi_epi64(u1, u3),
        ];
        for (d, v) in d.iter_mut().zip(cols) {
            *d = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)),
                _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v)),
            );
        }
    }
    let w = fdct_bins(&d);
    let k31 = _mm256_set1_epi8(31);
    let mut out = [0u8; 256];
    for (o, w) in out
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(w.as_chunks::<2>().0)
    {
        store256(o, _mm256_min_epu8(_mm256_packus_epi16(w[0], w[1]), k31));
    }
    out
}

#[target_feature(enable = "avx2")]
fn collect_uv_kernel(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    let zero = _mm256_setzero_si256();
    let mut d = [[zero; 4]; 4];
    for (j, d) in d.iter_mut().enumerate() {
        let (pu, pv) = (uv_plane(u, stride, z, 8, j), uv_plane(v, stride, z, 24, j));
        // Register c: U blocks 0-3 then V blocks 4-7 of column c, in bytes 0-7.
        let (lo, hi) = (_mm256_unpacklo_epi32(pu, pv), _mm256_unpackhi_epi32(pu, pv));
        for (c, d) in d.iter_mut().enumerate() {
            let q = if c < 2 { lo } else { hi };
            let (s, p) = (_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
            let (s, p) = if c & 1 == 0 {
                (s, p)
            } else {
                (_mm_srli_si128::<8>(s), _mm_srli_si128::<8>(p))
            };
            *d = _mm256_zextsi128_si256(_mm_sub_epi16(_mm_cvtepu8_epi16(s), _mm_cvtepu8_epi16(p)));
        }
    }
    let w = fdct_bins(&d);
    let k31 = _mm_set1_epi8(31);
    let mut out = [0u8; 128];
    for (o, w) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(w.as_chunks::<2>().0)
    {
        let packed = _mm_packus_epi16(_mm256_castsi256_si128(w[0]), _mm256_castsi256_si128(w[1]));
        store128(o, _mm_min_epu8(packed, k31));
    }
    out
}

/// Sum of the four u64 lanes of a `sad_epu8` result.
#[target_feature(enable = "avx2")]
fn sad_total(v: __m256i) -> i32 {
    let mut a = [0i32; 8];
    store256_i32(&mut a, _mm256_sad_epu8(v, _mm256_setzero_si256()));
    a[0] + a[2] + a[4] + a[6]
}

/// Go's `binStatsAsm`: the largest byte and the counts of the bytes 0 to 3.
#[target_feature(enable = "avx2")]
fn bin_stats_kernel(b: &[u8]) -> BinStats {
    let zero = _mm256_setzero_si256();
    let ks = [
        zero,
        _mm256_set1_epi8(1),
        _mm256_set1_epi8(2),
        _mm256_set1_epi8(3),
    ];
    let (chunks, tail) = b.as_chunks::<32>();
    let mut max = zero;
    let mut count = [0i32; 4];
    // Byte counters take 255 steps.
    for run in chunks.chunks(255) {
        let mut acc = [zero; 4];
        for c in run {
            let v = load256(c);
            max = _mm256_max_epu8(max, v);
            for (a, k) in acc.iter_mut().zip(ks) {
                *a = _mm256_sub_epi8(*a, _mm256_cmpeq_epi8(v, k));
            }
        }
        for (n, a) in count.iter_mut().zip(acc) {
            *n += sad_total(a);
        }
    }
    let mut m = [0u8; 32];
    store256(&mut m, max);
    let mut last = m.iter().copied().max().unwrap_or(0);
    for &v in tail {
        last = last.max(v);
        if v < 4 {
            count[usize::from(v)] += 1;
        }
    }
    (i32::from(last), count[0], count[1], count[2], count[3])
}
