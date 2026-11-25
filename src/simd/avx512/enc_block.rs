//! AVX-512 form of the encoder kernels of the block transform and quantizer that gain from 512-bit
//! registers: `quantize_n` (one block in 16 i32 lanes, the zigzag reorder and the nonzero test in
//! registers) and the luma batches `residuals_y` and `idct_adds_y` (the four blocks of a block row
//! in the four 128-bit lanes, through the same 32-bit steps as the AVX2 kernels, so bit-exact for
//! every input). The other kernels are AVX2's.

use super::mem::*;
use crate::simd::enc_block::Block;
use crate::simd::enc_pred::row;
use crate::tables::ZIGZAG;
use crate::vp8enc::predict::Ybr;
use crate::vp8enc::quant::QuantMatrix;
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Block> {
    if !super::available() {
        return None;
    }
    let k = crate::simd::avx2::enc_block::detect()?;
    Some(Block {
        quantize_n: quantize_n_d,
        residuals_y: residuals_y_d,
        idct_adds_y: idct_adds_y_d,
        ..k
    })
}

// Dispatch: one safe wrapper per kernel. Each is reachable only through `detect`, which checked
// `available()`, and the kernels are safe `#[target_feature]` fns enabling those features.

#[allow(unsafe_code)]
fn quantize_n_d(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    // SAFETY: `detect` returned this fn only after `available()` checked AVX2 and AVX-512 F, BW and VL.
    unsafe { quant_n(m, in_, out) }
}

#[allow(unsafe_code)]
fn residuals_y_d(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `available()` checked AVX2 and AVX-512 F, BW and VL.
    unsafe { residuals_y(z, src, dq) }
}

#[allow(unsafe_code)]
fn idct_adds_y_d(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `available()` checked AVX2 and AVX-512 F, BW and VL.
    unsafe { idct_adds_y(z, dq) }
}

/// Lane 0 from `dc`, the rest from `ac`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn dc_ac(dc: u32, ac: u32) -> __m512i {
    _mm512_mask_set1_epi32(_mm512_set1_epi32(ac as i32), 1, dc as i32)
}

/// `quantize_sse` of each block, without the SSE: the levels in zigzag order, the coefficients
/// dequantized in place. Bit n of the result is set when block n has a nonzero level.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn quant_n(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    // Lane n of the reordered levels is raster lane `ZIGZAG[n] & 15`.
    let zz: [i16; 16] = std::array::from_fn(|n| i16::from(ZIGZAG[n] & 15));
    let zz = load256_i16(&zz);
    let sharp = load512_i32(&m.sharpen);
    let (iq, bias, q) = (
        dc_ac(m.iq[0], m.iq[1]),
        dc_ac(m.bias[0], m.bias[1]),
        dc_ac(m.q[0], m.q[1]),
    );
    let mut nz = 0;
    for (n, (i, o)) in in_.iter_mut().zip(out.iter_mut()).enumerate() {
        let v = _mm512_cvtepi16_epi32(load256_i16(i));
        let s = _mm512_srai_epi32::<31>(v);
        let a = _mm512_add_epi32(_mm512_abs_epi32(v), sharp);
        let l = _mm512_srli_epi32::<17>(_mm512_add_epi32(_mm512_mullo_epi32(a, iq), bias));
        let l = _mm512_sub_epi32(_mm512_xor_si512(l, s), s);
        // The low 16 bits of each lane, like `as i16`.
        store256_i16(i, _mm512_cvtepi32_epi16(_mm512_mullo_epi32(l, q)));
        let lv = _mm256_permutexvar_epi16(zz, _mm512_cvtepi32_epi16(l));
        store256_i16(o, lv);
        if _mm256_cmpneq_epi16_mask(lv, _mm256_setzero_si256()) != 0 {
            nz |= 1 << n;
        }
    }
    nz
}

/// Transposes the four 4x4 i32 matrices of four registers, one matrix to a 128-bit lane.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn transpose(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; 4] {
    let t0 = _mm512_unpacklo_epi32(a, b);
    let t1 = _mm512_unpacklo_epi32(c, d);
    let t2 = _mm512_unpackhi_epi32(a, b);
    let t3 = _mm512_unpackhi_epi32(c, d);
    [
        _mm512_unpacklo_epi64(t0, t1),
        _mm512_unpackhi_epi64(t0, t1),
        _mm512_unpacklo_epi64(t2, t3),
        _mm512_unpackhi_epi64(t2, t3),
    ]
}

/// The fdct of four blocks at once. `d[j]` holds row j of the differences, block b in lane b and
/// column c in dword c of the lane. The result is the 16 coefficients of each block as i16, in
/// raster order: blocks 0 and 1, then blocks 2 and 3.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn fdct4(d: [__m512i; 4]) -> [__m512i; 2] {
    let (k2217, k5352) = (_mm512_set1_epi32(2217), _mm512_set1_epi32(5352));
    // Row pass: e_i holds column i of every row.
    let [e0, e1, e2, e3] = transpose(d[0], d[1], d[2], d[3]);
    let a0 = _mm512_add_epi32(e0, e3);
    let a1 = _mm512_add_epi32(e1, e2);
    let a2 = _mm512_sub_epi32(e1, e2);
    let a3 = _mm512_sub_epi32(e0, e3);
    let t0 = _mm512_slli_epi32::<3>(_mm512_add_epi32(a0, a1));
    let t1 = _mm512_srai_epi32::<9>(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(a2, k2217), _mm512_mullo_epi32(a3, k5352)),
        _mm512_set1_epi32(1812),
    ));
    let t2 = _mm512_slli_epi32::<3>(_mm512_sub_epi32(a0, a1));
    let t3 = _mm512_srai_epi32::<9>(_mm512_add_epi32(
        _mm512_sub_epi32(_mm512_mullo_epi32(a3, k2217), _mm512_mullo_epi32(a2, k5352)),
        _mm512_set1_epi32(937),
    ));
    // Column pass: y_r holds row r's four transformed values.
    let [y0, y1, y2, y3] = transpose(t0, t1, t2, t3);
    let a0 = _mm512_add_epi32(y0, y3);
    let a1 = _mm512_add_epi32(y1, y2);
    let a2 = _mm512_sub_epi32(y1, y2);
    let a3 = _mm512_sub_epi32(y0, y3);
    let o0 = _mm512_srai_epi32::<4>(_mm512_add_epi32(
        _mm512_add_epi32(a0, a1),
        _mm512_set1_epi32(7),
    ));
    let v = _mm512_srai_epi32::<16>(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(a2, k2217), _mm512_mullo_epi32(a3, k5352)),
        _mm512_set1_epi32(12000),
    ));
    // Plus one where a3 != 0: add one, then take it off again where a3 == 0.
    let one = _mm512_set1_epi32(1);
    let v1 = _mm512_add_epi32(v, one);
    let o1 = _mm512_mask_sub_epi32(
        v1,
        _mm512_cmpeq_epi32_mask(a3, _mm512_setzero_si512()),
        v1,
        one,
    );
    let o2 = _mm512_srai_epi32::<4>(_mm512_add_epi32(
        _mm512_sub_epi32(a0, a1),
        _mm512_set1_epi32(7),
    ));
    let o3 = _mm512_srai_epi32::<16>(_mm512_add_epi32(
        _mm512_sub_epi32(_mm512_mullo_epi32(a3, k2217), _mm512_mullo_epi32(a2, k5352)),
        _mm512_set1_epi32(51000),
    ));
    // The low 16 bits of each lane, like `as i16`: lane b of `lo` is block b's coefficients 0..8
    // and lane b of `hi` its coefficients 8..16.
    let m = _mm512_set1_epi32(0xFFFF);
    let lo = _mm512_packus_epi32(_mm512_and_si512(o0, m), _mm512_and_si512(o1, m));
    let hi = _mm512_packus_epi32(_mm512_and_si512(o2, m), _mm512_and_si512(o3, m));
    // Interleave the halves, block by block: qword 8 and up of the index are `hi`.
    let first = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let second = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    [
        _mm512_permutex2var_epi64(lo, first, hi),
        _mm512_permutex2var_epi64(lo, second, hi),
    ]
}

/// The 16 bytes of `b`.
fn row16(b: &[u8]) -> &[u8; 16] {
    row(b)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn residuals_y(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    // Block row br holds blocks 4 * br..4 * br + 4: rows 1 + 4 * br + r of `z` from column 8, and
    // the 16 bytes at 64 * br + 16 * r of the source.
    for (br, o) in dq
        .as_flattened_mut()
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .enumerate()
    {
        let mut d = [_mm512_setzero_si512(); 4];
        for (r, d) in d.iter_mut().enumerate() {
            let s = load128(row16(&src[64 * br + 16 * r..]));
            let p = load128(row16(&z[1 + 4 * br + r][8..]));
            *d = _mm512_sub_epi32(_mm512_cvtepu8_epi32(s), _mm512_cvtepu8_epi32(p));
        }
        let [a, b] = fdct4(d);
        if let [o0, o1] = o.as_chunks_mut::<32>().0 {
            store512_i16(o0, a);
            store512_i16(o1, b);
        }
    }
}

/// (x * k) >> 16 on i32 lanes, wrapping like Go's int32.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn mh(x: __m512i, k: __m512i) -> __m512i {
    _mm512_srai_epi32::<16>(_mm512_mullo_epi32(x, k))
}

/// Row `k` of each of four blocks (`a01` holds blocks 0 and 1, `a23` blocks 2 and 3) as i32 lanes:
/// row k of block b is qword 4 * b + k of the 16 qwords of the two registers.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
#[inline]
fn block_row(a01: __m512i, a23: __m512i, k: i64) -> __m512i {
    let idx = _mm512_setr_epi64(k, 4 + k, 8 + k, 12 + k, 0, 0, 0, 0);
    _mm512_cvtepi16_epi32(_mm512_castsi512_si256(_mm512_permutex2var_epi64(
        a01, idx, a23,
    )))
}

/// The inverse DCT of four blocks at once, `c` their 64 coefficients, as four registers: lane b of
/// register j is row j of block b's result, one pixel offset to a dword.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn idct4(c: &[i16; 64]) -> [__m512i; 4] {
    let [c01, c23] = c.as_chunks::<32>().0 else {
        return [_mm512_setzero_si512(); 4];
    };
    let (a01, a23) = (load512_i16(c01), load512_i16(c23));
    let (r0, r1, r2, r3) = (
        block_row(a01, a23, 0),
        block_row(a01, a23, 1),
        block_row(a01, a23, 2),
        block_row(a01, a23, 3),
    );
    let (c1, c2) = (_mm512_set1_epi32(85627), _mm512_set1_epi32(35468));
    let a = _mm512_add_epi32(r0, r2);
    let b = _mm512_sub_epi32(r0, r2);
    let cc = _mm512_sub_epi32(mh(r1, c2), mh(r3, c1));
    let d = _mm512_add_epi32(mh(r1, c1), mh(r3, c2));
    let [x0, x1, x2, x3] = transpose(
        _mm512_add_epi32(a, d),
        _mm512_add_epi32(b, cc),
        _mm512_sub_epi32(b, cc),
        _mm512_sub_epi32(a, d),
    );
    let dc = _mm512_add_epi32(x0, _mm512_set1_epi32(4));
    let a = _mm512_add_epi32(dc, x2);
    let b = _mm512_sub_epi32(dc, x2);
    let cc = _mm512_sub_epi32(mh(x1, c2), mh(x3, c1));
    let d = _mm512_add_epi32(mh(x1, c1), mh(x3, c2));
    transpose(
        _mm512_srai_epi32::<3>(_mm512_add_epi32(a, d)),
        _mm512_srai_epi32::<3>(_mm512_add_epi32(b, cc)),
        _mm512_srai_epi32::<3>(_mm512_sub_epi32(b, cc)),
        _mm512_srai_epi32::<3>(_mm512_sub_epi32(a, d)),
    )
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    let zero = _mm512_setzero_si512();
    for (br, c) in dq.as_flattened().as_chunks::<64>().0.iter().enumerate() {
        let p = idct4(c);
        for (j, p) in p.into_iter().enumerate() {
            let y = 1 + 4 * br + j;
            let px = _mm512_cvtepu8_epi32(load128(row16(&z[y][8..])));
            // Clamp to 0..=255: negative sums to zero, then the unsigned saturating narrow.
            let s = _mm512_max_epi32(_mm512_add_epi32(p, px), zero);
            if let Some(d) = z[y][8..].first_chunk_mut::<16>() {
                store128(d, _mm512_cvtusepi32_epi8(s));
            }
        }
    }
}
