//! AVX-512 form of the encoder kernels of package B that gain from 512-bit registers: the SSE of
//! ten predictions (`sse4x10`, five predictions a register). The other kernels are AVX2's
//! (`sse_block` in AVX-512 was slower than AVX2's, see docs/safety.md).

use super::mem::*;
use crate::simd::enc_pred::Kernels;
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    if !super::available() {
        return None;
    }
    let k = crate::simd::avx2::enc_pred::detect()?;
    Some(Kernels { sse4x10, ..k })
}

/// Dispatch: the kernels are safe `#[target_feature]` fns, so calling them from here needs
/// `unsafe`. Only reachable through `detect`, which checked `available()`.
#[allow(unsafe_code)]
fn sse4x10(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    // SAFETY: `detect` returned this fn only after `available()` checked AVX2 and AVX-512 F, BW and VL.
    unsafe { sse4x10_kernel(pr, src) }
}

/// The SSE of the source block against each prediction, four u32 partial sums to a prediction
/// (the dwords of its 128-bit lane), five predictions a register.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn sse4x10_kernel(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    let s = _mm512_broadcast_i32x4(load128_u32(src));
    let zero = _mm512_setzero_si512();
    let (chunks, tail) = pr.as_flattened().as_chunks::<16>();
    // The last two predictions, padded to a register.
    let mut t = [0u32; 16];
    for (t, &v) in t.iter_mut().zip(tail) {
        *t = v;
    }
    let mut part = [[0i32; 16]; 3];
    for (o, c) in part
        .iter_mut()
        .zip(chunks.iter().chain(std::iter::once(&t)))
    {
        let m = load512_u32(c);
        let d = _mm512_or_si512(_mm512_subs_epu8(s, m), _mm512_subs_epu8(m, s));
        let lo = _mm512_unpacklo_epi8(d, zero);
        let hi = _mm512_unpackhi_epi8(d, zero);
        store512_i32(
            o,
            _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi)),
        );
    }
    let mut r = [0i32; 10];
    for (r, c) in r.iter_mut().zip(part.as_flattened().as_chunks::<4>().0) {
        *r = c.iter().sum();
    }
    r
}
