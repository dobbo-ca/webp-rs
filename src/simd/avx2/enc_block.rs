//! AVX2 form of the encoder kernels of the block transform and quantizer: a 4x4 block in 32-bit
//! lanes through the same integer steps as the scalar reference, so the results are bit-exact for
//! every input. (Go's 16-bit IDCT path and two-blocks-a-pass quantizer are not ported.)

use super::mem::*;
use crate::simd::enc_block::{Block, Rows, chroma_pos, get_rows, luma_pos, put_rows, rows_at};
use crate::tables::ZIGZAG;
use crate::vp8enc::predict::Ybr;
use crate::vp8enc::quant::QuantMatrix;
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Block> {
    is_x86_feature_detected!("avx2").then_some(Block {
        fdct4: fdct4_d,
        quantize_sse: quantize_sse_d,
        quantize_n: quantize_n_d,
        idct_add4: idct_add4_d,
        residuals_y: residuals_y_d,
        residuals_uv: residuals_uv_d,
        idct_adds_y: idct_adds_y_d,
        idct_adds_uv: idct_adds_uv_d,
    })
}

// Dispatch: one safe wrapper per kernel. Each is reachable only through `detect`, which checked
// AVX2, and the kernels are safe `#[target_feature(enable = "avx2")]` fns.

#[allow(unsafe_code)]
fn fdct4_d(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { fdct(src, pred, out) }
}

#[allow(unsafe_code)]
fn quantize_sse_d(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { quant::<true>(m, in_, out) }
}

#[allow(unsafe_code)]
fn quantize_n_d(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { quant_n(m, in_, out) }
}

#[allow(unsafe_code)]
fn idct_add4_d(c: &[i16; 16], rows: &mut Rows) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { idct(c, rows) }
}

#[allow(unsafe_code)]
fn residuals_y_d(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { residuals_y(z, src, dq) }
}

#[allow(unsafe_code)]
fn residuals_uv_d(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { residuals_uv(z, u, v, dq) }
}

#[allow(unsafe_code)]
fn idct_adds_y_d(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { idct_adds_y(z, dq) }
}

#[allow(unsafe_code)]
fn idct_adds_uv_d(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { idct_adds_uv(z, dq) }
}

/// Transposes four vectors of four i32 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(a, b);
    let t1 = _mm_unpacklo_epi32(c, d);
    let t2 = _mm_unpackhi_epi32(a, b);
    let t3 = _mm_unpackhi_epi32(c, d);
    [
        _mm_unpacklo_epi64(t0, t1),
        _mm_unpackhi_epi64(t0, t1),
        _mm_unpacklo_epi64(t2, t3),
        _mm_unpackhi_epi64(t2, t3),
    ]
}

/// The fdct of `src` minus `pred`. Lanes are the block's rows in the row pass and its columns in
/// the column pass, with a transpose between them.
#[target_feature(enable = "avx2")]
fn fdct(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    let (s, p) = (load128_u32(src), load128_u32(pred));
    let d0 = _mm_sub_epi32(_mm_cvtepu8_epi32(s), _mm_cvtepu8_epi32(p));
    let d1 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<4>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<4>(p)),
    );
    let d2 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<8>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<8>(p)),
    );
    let d3 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<12>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<12>(p)),
    );
    let (k2217, k5352) = (_mm_set1_epi32(2217), _mm_set1_epi32(5352));
    // Row pass: e_i holds column i of every row.
    let [e0, e1, e2, e3] = transpose(d0, d1, d2, d3);
    let a0 = _mm_add_epi32(e0, e3);
    let a1 = _mm_add_epi32(e1, e2);
    let a2 = _mm_sub_epi32(e1, e2);
    let a3 = _mm_sub_epi32(e0, e3);
    let t0 = _mm_slli_epi32::<3>(_mm_add_epi32(a0, a1));
    let t1 = _mm_srai_epi32::<9>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a2, k2217), _mm_mullo_epi32(a3, k5352)),
        _mm_set1_epi32(1812),
    ));
    let t2 = _mm_slli_epi32::<3>(_mm_sub_epi32(a0, a1));
    let t3 = _mm_srai_epi32::<9>(_mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(a3, k2217), _mm_mullo_epi32(a2, k5352)),
        _mm_set1_epi32(937),
    ));
    // Column pass: y_r holds row r's four transformed values.
    let [y0, y1, y2, y3] = transpose(t0, t1, t2, t3);
    let a0 = _mm_add_epi32(y0, y3);
    let a1 = _mm_add_epi32(y1, y2);
    let a2 = _mm_sub_epi32(y1, y2);
    let a3 = _mm_sub_epi32(y0, y3);
    let o0 = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_add_epi32(a0, a1), _mm_set1_epi32(7)));
    let v = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a2, k2217), _mm_mullo_epi32(a3, k5352)),
        _mm_set1_epi32(12000),
    ));
    // Plus one where a3 != 0 (the compare is -1 where a3 == 0).
    let o1 = _mm_add_epi32(
        _mm_add_epi32(v, _mm_set1_epi32(1)),
        _mm_cmpeq_epi32(a3, _mm_setzero_si128()),
    );
    let o2 = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_sub_epi32(a0, a1), _mm_set1_epi32(7)));
    let o3 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(a3, k2217), _mm_mullo_epi32(a2, k5352)),
        _mm_set1_epi32(51000),
    ));
    // The low 16 bits of each lane, like `as i16`.
    let m = _mm_set1_epi32(0xFFFF);
    let lo = _mm_packus_epi32(_mm_and_si128(o0, m), _mm_and_si128(o1, m));
    let hi = _mm_packus_epi32(_mm_and_si128(o2, m), _mm_and_si128(o3, m));
    if let [a, b] = out.as_chunks_mut::<8>().0 {
        store128_i16(a, lo);
        store128_i16(b, hi);
    }
}

/// One half of the quantizer: levels and dequantized values of eight raster coefficients.
#[target_feature(enable = "avx2")]
#[inline]
fn quant8(
    v: __m256i,
    sharp: __m256i,
    iq: __m256i,
    bias: __m256i,
    q: __m256i,
) -> (__m256i, __m256i) {
    let s = _mm256_srai_epi32::<31>(v);
    let a = _mm256_add_epi32(_mm256_abs_epi32(v), sharp);
    let l = _mm256_srli_epi32::<17>(_mm256_add_epi32(_mm256_mullo_epi32(a, iq), bias));
    let l = _mm256_sub_epi32(_mm256_xor_si256(l, s), s);
    (l, _mm256_mullo_epi32(l, q))
}

/// The low 16 bits of 16 i32 lanes (like `as i16`) as 16 i16 lanes, in order.
#[target_feature(enable = "avx2")]
#[inline]
fn pack16(lo: __m256i, hi: __m256i) -> __m256i {
    let m = _mm256_set1_epi32(0xFFFF);
    let p = _mm256_packus_epi32(_mm256_and_si256(lo, m), _mm256_and_si256(hi, m));
    _mm256_permute4x64_epi64::<0xD8>(p)
}

/// Lane 0 from `dc`, the rest from `ac`.
#[target_feature(enable = "avx2")]
#[inline]
fn dc_ac(dc: u32, ac: u32) -> __m256i {
    _mm256_blend_epi32::<1>(_mm256_set1_epi32(ac as i32), _mm256_set1_epi32(dc as i32))
}

/// `quantize_sse` (with the SSE when `D`, else 0).
#[target_feature(enable = "avx2")]
fn quant<const D: bool>(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    let x = load256_i16(in_);
    let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(x));
    let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(x));
    let [sh0, sh1] = m.sharpen.as_chunks::<8>().0 else {
        return (0, 0);
    };
    let (iq1, bias1, q1) = (
        _mm256_set1_epi32(m.iq[1] as i32),
        _mm256_set1_epi32(m.bias[1] as i32),
        _mm256_set1_epi32(m.q[1] as i32),
    );
    let (l0, d0) = quant8(
        lo,
        load256_i32(sh0),
        dc_ac(m.iq[0], m.iq[1]),
        dc_ac(m.bias[0], m.bias[1]),
        dc_ac(m.q[0], m.q[1]),
    );
    let (l1, d1) = quant8(hi, load256_i32(sh1), iq1, bias1, q1);
    store256_i16(in_, pack16(d0, d1));
    let mut r = [0i16; 16];
    store256_i16(&mut r, pack16(l0, l1));
    let mut last = -1;
    for n in 0..16 {
        let l = r[(ZIGZAG[n] & 15) as usize];
        out[n] = l;
        if l != 0 {
            last = n as i32;
        }
    }
    if !D {
        return (last, 0);
    }
    // Each square is below 2^32: |v - dq| is below 2^16 (a level is within one step of |v|).
    let (e0, e1) = (_mm256_sub_epi32(lo, d0), _mm256_sub_epi32(hi, d1));
    let (mut s0, mut s1) = ([0i32; 8], [0i32; 8]);
    store256_i32(&mut s0, _mm256_mullo_epi32(e0, e0));
    store256_i32(&mut s1, _mm256_mullo_epi32(e1, e1));
    let d = s0.iter().chain(&s1).map(|&v| i64::from(v as u32)).sum();
    (last, d)
}

#[target_feature(enable = "avx2")]
fn quant_n(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    let mut nz = 0;
    for (n, (i, o)) in in_.iter_mut().zip(out.iter_mut()).enumerate() {
        if quant::<false>(m, i, o).0 >= 0 {
            nz |= 1 << n;
        }
    }
    nz
}

/// (x * k) >> 16 on i32 lanes, wrapping like Go's int32.
#[target_feature(enable = "avx2")]
#[inline]
fn mh(x: __m128i, k: __m128i) -> __m128i {
    _mm_srai_epi32::<16>(_mm_mullo_epi32(x, k))
}

/// Adds the inverse DCT of `c` to `rows`. The first pass runs down the columns of `c` with a lane
/// per column, the second down the rows, with a transpose between and one after.
#[target_feature(enable = "avx2")]
fn idct(c: &[i16; 16], rows: &mut Rows) {
    let [c01, c23] = c.as_chunks::<8>().0 else {
        return;
    };
    let (x01, x23) = (load128_i16(c01), load128_i16(c23));
    if _mm_testz_si128(_mm_or_si128(x01, x23), _mm_or_si128(x01, x23)) != 0 {
        return;
    }
    let r0 = _mm_cvtepi16_epi32(x01);
    let r1 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(x01));
    let r2 = _mm_cvtepi16_epi32(x23);
    let r3 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(x23));
    let (c1, c2) = (_mm_set1_epi32(85627), _mm_set1_epi32(35468));
    let a = _mm_add_epi32(r0, r2);
    let b = _mm_sub_epi32(r0, r2);
    let cc = _mm_sub_epi32(mh(r1, c2), mh(r3, c1));
    let d = _mm_add_epi32(mh(r1, c1), mh(r3, c2));
    let [x0, x1, x2, x3] = transpose(
        _mm_add_epi32(a, d),
        _mm_add_epi32(b, cc),
        _mm_sub_epi32(b, cc),
        _mm_sub_epi32(a, d),
    );
    let dc = _mm_add_epi32(x0, _mm_set1_epi32(4));
    let a = _mm_add_epi32(dc, x2);
    let b = _mm_sub_epi32(dc, x2);
    let cc = _mm_sub_epi32(mh(x1, c2), mh(x3, c1));
    let d = _mm_add_epi32(mh(x1, c1), mh(x3, c2));
    let [p0, p1, p2, p3] = transpose(
        _mm_srai_epi32::<3>(_mm_add_epi32(a, d)),
        _mm_srai_epi32::<3>(_mm_add_epi32(b, cc)),
        _mm_srai_epi32::<3>(_mm_sub_epi32(b, cc)),
        _mm_srai_epi32::<3>(_mm_sub_epi32(a, d)),
    );
    let px = load128_u32(rows);
    let s0 = _mm_add_epi32(p0, _mm_cvtepu8_epi32(px));
    let s1 = _mm_add_epi32(p1, _mm_cvtepu8_epi32(_mm_srli_si128::<4>(px)));
    let s2 = _mm_add_epi32(p2, _mm_cvtepu8_epi32(_mm_srli_si128::<8>(px)));
    let s3 = _mm_add_epi32(p3, _mm_cvtepu8_epi32(_mm_srli_si128::<12>(px)));
    // Saturate to i16, then to u8: the clamp to 0..=255.
    let res = _mm_packus_epi16(_mm_packs_epi32(s0, s1), _mm_packs_epi32(s2, s3));
    store128_u32(rows, res);
}

#[target_feature(enable = "avx2")]
fn residuals_y(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    for (n, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = luma_pos(n);
        fdct(&rows_at(src, 16, off), &get_rows(z, y, x), blk);
    }
}

#[target_feature(enable = "avx2")]
fn residuals_uv(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    for (c, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = chroma_pos(c);
        let src = if c >= 4 { v } else { u };
        fdct(&rows_at(src, 8, off), &get_rows(z, y, x), blk);
    }
}

#[target_feature(enable = "avx2")]
fn idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    for (n, c) in dq.iter().enumerate() {
        let (y, x, _) = luma_pos(n);
        let mut r = get_rows(z, y, x);
        idct(c, &mut r);
        put_rows(z, &r, y, x);
    }
}

#[target_feature(enable = "avx2")]
fn idct_adds_uv(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    for (c, b) in dq.iter().enumerate() {
        let (y, x, _) = chroma_pos(c);
        let mut r = get_rows(z, y, x);
        idct(b, &mut r);
        put_rows(z, &r, y, x);
    }
}
