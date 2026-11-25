//! AVX2 kernels: luma of 8 pixels a step, squared differences of 16 bytes a step (8 for a tail).

use super::{Kernels, scalar};
use std::arch::x86_64::*;

// The load/store helpers: the only `unsafe` in this tier besides the dispatch below.
#[allow(unsafe_code, dead_code)]
mod mem;
use mem::*;

pub(super) mod enc_block;
pub(super) mod enc_chroma;
pub(super) mod enc_collect;
pub(super) mod enc_cost;
pub(super) mod enc_i4;
pub(super) mod enc_pred;

/// Bytes per accumulation block of `sse_kernel`: the i32 lanes take at most 2 * 255^2 a step, so
/// 4096 steps stay below 2^31.
const BLOCK: usize = 16 * 4096;

pub(super) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels { luma_row, sse_row })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "avx2")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn luma_row(dst: &mut [u8], src: &[u8]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { luma_kernel(dst, src) }
}

/// Dispatch, as `luma_row`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn sse_row(a: &[u8], b: &[u8]) -> u64 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { sse_kernel(a, b) }
}

/// Sum of the u32 lanes, as u64.
#[target_feature(enable = "avx2")]
fn hsum(v: __m256i) -> u64 {
    let mut a = [0i32; 8];
    store256_i32(&mut a, v);
    a.iter().map(|&x| u64::from(x as u32)).sum()
}

#[target_feature(enable = "avx2")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len() / 4) & !7;
    let (rk, gk, bk) = (
        _mm256_set1_epi32(16839),
        _mm256_set1_epi32(33059),
        _mm256_set1_epi32(6420),
    );
    let round = _mm256_set1_epi32((1 << 15) + (16 << 16));
    let m255 = _mm256_set1_epi32(255);
    // Byte 0 of each dword to the low four bytes of its 128-bit lane.
    let pick = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    for (d, s) in dst[..n]
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<32>().0.iter())
    {
        let v = load256(s);
        let r = _mm256_and_si256(v, m255);
        let g = _mm256_and_si256(_mm256_srli_epi32::<8>(v), m255);
        let b = _mm256_and_si256(_mm256_srli_epi32::<16>(v), m255);
        let mut y = _mm256_add_epi32(_mm256_mullo_epi32(r, rk), round);
        y = _mm256_add_epi32(y, _mm256_mullo_epi32(g, gk));
        y = _mm256_add_epi32(y, _mm256_mullo_epi32(b, bk));
        let p = _mm256_shuffle_epi8(_mm256_srli_epi32::<16>(y), pick);
        let q = _mm_unpacklo_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p));
        storel64(d, q);
    }
    scalar::luma_row(&mut dst[n..], &src[4 * n..]);
}

#[target_feature(enable = "avx2")]
fn sse_kernel(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let n16 = n & !15;
    let mut total = 0u64;
    for (ba, bb) in a[..n16].chunks(BLOCK).zip(b[..n16].chunks(BLOCK)) {
        let mut acc = _mm256_setzero_si256();
        for (ca, cb) in ba
            .as_chunks::<16>()
            .0
            .iter()
            .zip(bb.as_chunks::<16>().0.iter())
        {
            let (x, y) = (load128(ca), load128(cb));
            let d = _mm256_sub_epi16(_mm256_cvtepu8_epi16(x), _mm256_cvtepu8_epi16(y));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
        }
        total += hsum(acc);
    }
    let mut i = n16;
    if let (Some(ca), Some(cb)) = (a[i..].first_chunk::<8>(), b[i..].first_chunk::<8>()) {
        let (x, y) = (loadl64(ca), loadl64(cb));
        let d = _mm_sub_epi16(_mm_cvtepu8_epi16(x), _mm_cvtepu8_epi16(y));
        total += hsum(_mm256_zextsi128_si256(_mm_madd_epi16(d, d)));
        i += 8;
    }
    total + scalar::sse_row(&a[i..], &b[i..])
}
