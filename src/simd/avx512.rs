//! AVX-512 kernels: luma of 16 pixels a step, squared differences of 64 bytes a step. The tier
//! holds only the kernels that gain from 512-bit registers (the groups' `detect` take the rest
//! from AVX2), and needs AVX-512 F, BW and VL: the kernels are safe `#[target_feature]` fns that
//! enable them together with AVX2, whose intrinsics some of them also use.

use super::Kernels;
use std::arch::x86_64::*;

// The load/store helpers: the only `unsafe` in this tier besides the dispatch below.
#[allow(unsafe_code, dead_code)]
mod mem;
use mem::*;

pub(super) mod enc_block;
pub(super) mod enc_chroma;
pub(super) mod enc_pred;

/// Bytes per accumulation block of `sse_kernel`: an i32 lane takes at most 4 * 255^2 a step, so
/// 4096 steps stay below 2^31.
const BLOCK: usize = 64 * 4096;

/// Whether this CPU has the features the kernels of this tier enable.
fn available() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
}

pub(super) fn detect() -> Option<Kernels> {
    available().then_some(Kernels { luma_row, sse_row })
}

/// Dispatch: the kernels are safe `#[target_feature]` fns, so calling them from here needs
/// `unsafe`. Only reachable through `detect`, which checked AVX2, AVX-512 F, BW and VL.
#[allow(unsafe_code)]
fn luma_row(dst: &mut [u8], src: &[u8]) {
    // SAFETY: `detect` returned this fn only after `available()` checked the features of `luma_kernel`.
    unsafe { luma_kernel(dst, src) }
}

/// Dispatch, as `luma_row`.
#[allow(unsafe_code)]
fn sse_row(a: &[u8], b: &[u8]) -> u64 {
    // SAFETY: `detect` returned this fn only after `available()` checked the features of `sse_kernel`.
    unsafe { sse_kernel(a, b) }
}

/// Sum of the u32 lanes, as u64.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn hsum(v: __m512i) -> u64 {
    let mut a = [0i32; 16];
    store512_i32(&mut a, v);
    a.iter().map(|&x| u64::from(x as u32)).sum()
}

/// The luma of the 16 RGBA pixels in `s`, as 16 bytes.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn luma16(s: &[u8; 64]) -> __m128i {
    let (rk, gk, bk) = (
        _mm512_set1_epi32(16839),
        _mm512_set1_epi32(33059),
        _mm512_set1_epi32(6420),
    );
    let round = _mm512_set1_epi32((1 << 15) + (16 << 16));
    let m255 = _mm512_set1_epi32(255);
    let v = load512(s);
    let r = _mm512_and_si512(v, m255);
    let g = _mm512_and_si512(_mm512_srli_epi32::<8>(v), m255);
    let b = _mm512_and_si512(_mm512_srli_epi32::<16>(v), m255);
    let mut y = _mm512_add_epi32(_mm512_mullo_epi32(r, rk), round);
    y = _mm512_add_epi32(y, _mm512_mullo_epi32(g, gk));
    y = _mm512_add_epi32(y, _mm512_mullo_epi32(b, bk));
    // The low byte of each lane, like `as u8`.
    _mm512_cvtepi32_epi8(_mm512_srli_epi32::<16>(y))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len() / 4);
    let n16 = n & !15;
    for (d, s) in dst[..n16]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<64>().0.iter())
    {
        store128(d, luma16(s));
    }
    // The last pixels through a zero-padded copy.
    let (mut s, mut d) = ([0u8; 64], [0u8; 16]);
    for (t, &v) in s.iter_mut().zip(&src[4 * n16..4 * n]) {
        *t = v;
    }
    store128(&mut d, luma16(&s));
    for (t, &v) in dst[n16..n].iter_mut().zip(&d) {
        *t = v;
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn sse64(acc: __m512i, a: &[u8; 64], b: &[u8; 64]) -> __m512i {
    let (x, y) = (load512(a), load512(b));
    let zero = _mm512_setzero_si512();
    // |x - y| as bytes, then as words: the squares are madd pairs.
    let d = _mm512_or_si512(_mm512_subs_epu8(x, y), _mm512_subs_epu8(y, x));
    let lo = _mm512_unpacklo_epi8(d, zero);
    let hi = _mm512_unpackhi_epi8(d, zero);
    _mm512_add_epi32(
        acc,
        _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi)),
    )
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn sse_kernel(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let n64 = n & !63;
    let mut total = 0u64;
    for (ba, bb) in a[..n64].chunks(BLOCK).zip(b[..n64].chunks(BLOCK)) {
        let mut acc = _mm512_setzero_si512();
        for (ca, cb) in ba
            .as_chunks::<64>()
            .0
            .iter()
            .zip(bb.as_chunks::<64>().0.iter())
        {
            acc = sse64(acc, ca, cb);
        }
        total += hsum(acc);
    }
    // The last bytes through zero-padded copies: a zero against a zero adds nothing.
    let (mut ta, mut tb) = ([0u8; 64], [0u8; 64]);
    for (t, &v) in ta.iter_mut().zip(&a[n64..]) {
        *t = v;
    }
    for (t, &v) in tb.iter_mut().zip(&b[n64..]) {
        *t = v;
    }
    total + hsum(sse64(_mm512_setzero_si512(), &ta, &tb))
}
