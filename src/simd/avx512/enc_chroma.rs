//! AVX-512 form of the encoder kernels of package E, the opaque chroma row conversion: 16 blocks a
//! pass. The 33-entry gamma table is looked up with `vpermw` (two 32-entry u16 tables, one for
//! each of the entry and its step to the next one) instead of the AVX2 form's four `pshufb`s, and
//! the `to_linear` lookups are 16-lane gathers.

use super::mem::*;
use crate::simd::enc_chroma::Kernels;
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    super::available().then_some(Kernels { chroma })
}

/// Dispatch: the kernel is a safe `#[target_feature]` fn, so calling it from here needs `unsafe`.
/// Only reachable through `detect`, which checked `available()`.
#[allow(unsafe_code)]
fn chroma(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    // SAFETY: `detect` returned this fn only after `available()` checked AVX2 and AVX-512 F, BW and VL.
    unsafe { chroma_kernel(cb, cr, p0, p1, to_linear, to_gamma) }
}

/// Two 16-bit constants as the i32 lane `madd_epi16` multiplies pairs by.
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as i32) << 16)
}

/// linToGamma of the sixteen i32 sums in `s`, as 4 * L[p] + (D[p] * x + 64) >> 7 with p = s >> 9
/// and x = s & 511. `tl` and `td` hold L and D (the step to the next entry) for p below 32, as
/// words: libwebp's table is below 256 and rises by less than 256 a step, and a sum is at most
/// 4 * 4095, so p is below 32.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn lin_to_gamma(s: __m512i, tl: __m512i, td: __m512i) -> __m512i {
    let p = _mm512_srli_epi32::<9>(s);
    // The word index is p in the low word of each dword and 0 in the high one; the high word of
    // the result is then entry 0, which the mask drops.
    let m = _mm512_set1_epi32(0xFFFF);
    let l = _mm512_and_si512(_mm512_permutexvar_epi16(p, tl), m);
    let d = _mm512_and_si512(_mm512_permutexvar_epi16(p, td), m);
    let x = _mm512_and_si512(s, _mm512_set1_epi32(511));
    let r = _mm512_srli_epi32::<7>(_mm512_add_epi32(
        _mm512_mullo_epi32(x, d),
        _mm512_set1_epi32(64),
    ));
    _mm512_add_epi32(r, _mm512_slli_epi32::<2>(l))
}

/// The sums over each 2x2 block of one channel (the byte at bit `shift`) of the pixel pairs `a` and
/// `b`: `a[0]`/`b[0]` hold pixels 0..16 of the two rows, `a[1]`/`b[1]` pixels 16..32. Lane k is
/// block k.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn channel(
    shift: i32,
    a: [__m512i; 2],
    b: [__m512i; 2],
    to_linear: &[i32; 256],
    tl: __m512i,
    td: __m512i,
) -> __m512i {
    let sh = _mm512_set1_epi32(shift);
    let v0 = _mm512_add_epi32(
        gather512_i32(to_linear, _mm512_srlv_epi32(a[0], sh)),
        gather512_i32(to_linear, _mm512_srlv_epi32(b[0], sh)),
    );
    let v1 = _mm512_add_epi32(
        gather512_i32(to_linear, _mm512_srlv_epi32(a[1], sh)),
        gather512_i32(to_linear, _mm512_srlv_epi32(b[1], sh)),
    );
    // The low dword of each qword gets the sum of the two pixels of a block.
    let h0 = _mm512_add_epi32(v0, _mm512_srli_epi64::<32>(v0));
    let h1 = _mm512_add_epi32(v1, _mm512_srli_epi64::<32>(v1));
    let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    lin_to_gamma(_mm512_permutex2var_epi32(h0, even, h1), tl, td)
}

/// One chroma value of the sixteen blocks: (R, G) and B weighted, rounded and shifted.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn weigh(rg: __m512i, b: __m512i, k_rg: __m512i, k_b: __m512i, round: __m512i) -> __m512i {
    let s = _mm512_add_epi32(_mm512_madd_epi16(rg, k_rg), _mm512_madd_epi16(b, k_b));
    _mm512_srai_epi32::<18>(_mm512_add_epi32(s, round))
}

/// The tables of `lin_to_gamma` as words: `to_gamma[p]` and `to_gamma[p + 1] - to_gamma[p]`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn tables(to_gamma: &[i32; 33]) -> (__m512i, __m512i) {
    let l: [i16; 32] = std::array::from_fn(|p| to_gamma[p] as i16);
    let d: [i16; 32] = std::array::from_fn(|p| (to_gamma[p + 1] - to_gamma[p]) as i16);
    (load512_i16(&l), load512_i16(&d))
}

/// The chroma of 16 blocks, `r0` and `r1` the 128 bytes of each row. Returns Cb and Cr as bytes
/// and ANDs the pixels into `alpha`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn pass(
    r0: &[u8; 128],
    r1: &[u8; 128],
    to_linear: &[i32; 256],
    tl: __m512i,
    td: __m512i,
    alpha: &mut __m512i,
) -> (__m128i, __m128i) {
    let ([a0, a1], [b0, b1]) = (r0.as_chunks::<64>().0, r1.as_chunks::<64>().0) else {
        return (_mm_setzero_si128(), _mm_setzero_si128());
    };
    let a = [load512(a0), load512(a1)];
    let b = [load512(b0), load512(b1)];
    for v in [a[0], a[1], b[0], b[1]] {
        *alpha = _mm512_and_si512(*alpha, v);
    }
    let r = channel(0, a, b, to_linear, tl, td);
    let g = channel(8, a, b, to_linear, tl, td);
    let bl = channel(16, a, b, to_linear, tl, td);
    let rg = _mm512_or_si512(r, _mm512_slli_epi32::<16>(g));
    let round = _mm512_set1_epi32((1 << 17) + (128 << 18));
    let cb = weigh(
        rg,
        bl,
        _mm512_set1_epi32(pair(-9719, -19081)),
        _mm512_set1_epi32(28800),
        round,
    );
    let cr = weigh(
        rg,
        bl,
        _mm512_set1_epi32(pair(28800, -24116)),
        _mm512_set1_epi32(pair(-4684, 0)),
        round,
    );
    // The low byte of each lane, like `as u8`.
    (_mm512_cvtepi32_epi8(cb), _mm512_cvtepi32_epi8(cr))
}

/// The chroma of `cb.len()` blocks, 16 a pass; the last 8 blocks, when there are, go through a
/// copy padded with 0xff bytes, which leave the alpha AND alone.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn chroma_kernel(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    let (tl, td) = tables(to_gamma);
    let mut alpha = _mm512_set1_epi32(-1);
    let rows = p0
        .as_chunks::<128>()
        .0
        .iter()
        .zip(p1.as_chunks::<128>().0.iter());
    let outs = cb
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(cr.as_chunks_mut::<16>().0.iter_mut());
    for ((r0, r1), (o_cb, o_cr)) in rows.zip(outs) {
        let (c0, c1) = pass(r0, r1, to_linear, tl, td, &mut alpha);
        store128(o_cb, c0);
        store128(o_cr, c1);
    }
    let blocks = cb.len().min(cr.len()).min(p0.len() / 8).min(p1.len() / 8);
    let done = blocks & !15;
    if blocks > done {
        let (mut t0, mut t1) = ([0xffu8; 128], [0xffu8; 128]);
        for (t, &v) in t0.iter_mut().zip(&p0[8 * done..8 * blocks]) {
            *t = v;
        }
        for (t, &v) in t1.iter_mut().zip(&p1[8 * done..8 * blocks]) {
            *t = v;
        }
        let (c0, c1) = pass(&t0, &t1, to_linear, tl, td, &mut alpha);
        let (mut b0, mut b1) = ([0u8; 16], [0u8; 16]);
        store128(&mut b0, c0);
        store128(&mut b1, c1);
        for (o, &v) in cb[done..blocks].iter_mut().zip(&b0) {
            *o = v;
        }
        for (o, &v) in cr[done..blocks].iter_mut().zip(&b1) {
            *o = v;
        }
    }
    let mut a = [0u8; 64];
    store512(&mut a, alpha);
    a.iter().skip(3).step_by(4).fold(255, |x, &v| x & v)
}
