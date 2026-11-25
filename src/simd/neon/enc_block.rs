//! NEON form of the encoder kernels of the block transform and quantizer: a 4x4 block in 32-bit
//! lanes through the same integer steps as the scalar reference (and as the AVX2 tier, which this
//! is written from), so the results are bit-exact for every input. Go's NEON kernels use
//! 16-bit lanes and are not byte-for-byte this arithmetic.

use super::mem::*;
use crate::simd::enc_block::{Block, Rows, chroma_pos, get_rows, luma_pos, put_rows, rows_at};
use crate::tables::ZIGZAG;
use crate::vp8enc::predict::Ybr;
use crate::vp8enc::quant::QuantMatrix;
use std::arch::aarch64::*;

type V = int32x4_t;

pub(in crate::simd) fn detect() -> Option<Block> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Block {
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
// NEON, and the kernels are safe `#[target_feature(enable = "neon")]` fns.

#[allow(unsafe_code)]
fn fdct4_d(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { fdct(src, pred, out) }
}

#[allow(unsafe_code)]
fn quantize_sse_d(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { quant::<true>(m, in_, out) }
}

#[allow(unsafe_code)]
fn quantize_n_d(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { quant_n(m, in_, out) }
}

#[allow(unsafe_code)]
fn idct_add4_d(c: &[i16; 16], rows: &mut Rows) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { idct(c, rows) }
}

#[allow(unsafe_code)]
fn residuals_y_d(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { residuals_y(z, src, dq) }
}

#[allow(unsafe_code)]
fn residuals_uv_d(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { residuals_uv(z, u, v, dq) }
}

#[allow(unsafe_code)]
fn idct_adds_y_d(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { idct_adds_y(z, dq) }
}

#[allow(unsafe_code)]
fn idct_adds_uv_d(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { idct_adds_uv(z, dq) }
}

/// Transposes four vectors of four i32 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn transpose(a: V, b: V, c: V, d: V) -> [V; 4] {
    let t0 = vreinterpretq_s64_s32(vtrn1q_s32(a, b));
    let t1 = vreinterpretq_s64_s32(vtrn2q_s32(a, b));
    let t2 = vreinterpretq_s64_s32(vtrn1q_s32(c, d));
    let t3 = vreinterpretq_s64_s32(vtrn2q_s32(c, d));
    [
        vreinterpretq_s32_s64(vtrn1q_s64(t0, t2)),
        vreinterpretq_s32_s64(vtrn1q_s64(t1, t3)),
        vreinterpretq_s32_s64(vtrn2q_s64(t0, t2)),
        vreinterpretq_s32_s64(vtrn2q_s64(t1, t3)),
    ]
}

/// The low 16 bits of the lanes of `lo` then `hi` (like `as i16`) as eight i16 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn narrow(lo: V, hi: V) -> int16x8_t {
    vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
}

/// The fdct of `src` minus `pred`. Lanes are the block's rows in the row pass and its columns in
/// the column pass, with a transpose between them.
#[target_feature(enable = "neon")]
fn fdct(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    let (s, p) = (
        vreinterpretq_u8_u32(load128_u32(src)),
        vreinterpretq_u8_u32(load128_u32(pred)),
    );
    // Row differences, two rows a vector, then widened to one row (four columns) a vector.
    let d01 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(p)));
    let d23 = vreinterpretq_s16_u16(vsubl_high_u8(s, p));
    let d0 = vmovl_s16(vget_low_s16(d01));
    let d1 = vmovl_high_s16(d01);
    let d2 = vmovl_s16(vget_low_s16(d23));
    let d3 = vmovl_high_s16(d23);
    // Row pass: e_i holds column i of every row.
    let [e0, e1, e2, e3] = transpose(d0, d1, d2, d3);
    let a0 = vaddq_s32(e0, e3);
    let a1 = vaddq_s32(e1, e2);
    let a2 = vsubq_s32(e1, e2);
    let a3 = vsubq_s32(e0, e3);
    let t0 = vshlq_n_s32::<3>(vaddq_s32(a0, a1));
    let t1 = vshrq_n_s32::<9>(vaddq_s32(
        vaddq_s32(vmulq_n_s32(a2, 2217), vmulq_n_s32(a3, 5352)),
        vdupq_n_s32(1812),
    ));
    let t2 = vshlq_n_s32::<3>(vsubq_s32(a0, a1));
    let t3 = vshrq_n_s32::<9>(vaddq_s32(
        vsubq_s32(vmulq_n_s32(a3, 2217), vmulq_n_s32(a2, 5352)),
        vdupq_n_s32(937),
    ));
    // Column pass: y_r holds row r's four transformed values.
    let [y0, y1, y2, y3] = transpose(t0, t1, t2, t3);
    let a0 = vaddq_s32(y0, y3);
    let a1 = vaddq_s32(y1, y2);
    let a2 = vsubq_s32(y1, y2);
    let a3 = vsubq_s32(y0, y3);
    let o0 = vshrq_n_s32::<4>(vaddq_s32(vaddq_s32(a0, a1), vdupq_n_s32(7)));
    let v = vshrq_n_s32::<16>(vaddq_s32(
        vaddq_s32(vmulq_n_s32(a2, 2217), vmulq_n_s32(a3, 5352)),
        vdupq_n_s32(12000),
    ));
    // Plus one where a3 != 0 (the compare is -1 where a3 == 0).
    let o1 = vaddq_s32(
        vaddq_s32(v, vdupq_n_s32(1)),
        vreinterpretq_s32_u32(vceqzq_s32(a3)),
    );
    let o2 = vshrq_n_s32::<4>(vaddq_s32(vsubq_s32(a0, a1), vdupq_n_s32(7)));
    let o3 = vshrq_n_s32::<16>(vaddq_s32(
        vsubq_s32(vmulq_n_s32(a3, 2217), vmulq_n_s32(a2, 5352)),
        vdupq_n_s32(51000),
    ));
    if let [a, b] = out.as_chunks_mut::<8>().0 {
        store128_s16(a, narrow(o0, o1));
        store128_s16(b, narrow(o2, o3));
    }
}

/// One group of four raster coefficients: levels and dequantized values.
#[target_feature(enable = "neon")]
#[inline]
fn quant4(v: V, sharp: V, iq: V, bias: V, q: V) -> (V, V) {
    let s = vshrq_n_s32::<31>(v);
    let a = vaddq_s32(vabsq_s32(v), sharp);
    let l = vreinterpretq_s32_u32(vshrq_n_u32::<17>(vreinterpretq_u32_s32(vaddq_s32(
        vmulq_s32(a, iq),
        bias,
    ))));
    let l = vsubq_s32(veorq_s32(l, s), s);
    (l, vmulq_s32(l, q))
}

/// Lane 0 from `dc`, the rest from `ac`.
#[target_feature(enable = "neon")]
#[inline]
fn dc_ac(dc: u32, ac: u32) -> V {
    vsetq_lane_s32::<0>(dc as i32, vdupq_n_s32(ac as i32))
}

/// `quantize_sse` (with the SSE when `D`, else 0).
#[target_feature(enable = "neon")]
fn quant<const D: bool>(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    let [ia, ib] = in_.as_chunks::<8>().0 else {
        return (0, 0);
    };
    let (xa, xb) = (load128_s16(ia), load128_s16(ib));
    let v0 = vmovl_s16(vget_low_s16(xa));
    let v1 = vmovl_high_s16(xa);
    let v2 = vmovl_s16(vget_low_s16(xb));
    let v3 = vmovl_high_s16(xb);
    let [sh0, sh1, sh2, sh3] = m.sharpen.as_chunks::<4>().0 else {
        return (0, 0);
    };
    let (iq1, bias1, q1) = (
        vdupq_n_s32(m.iq[1] as i32),
        vdupq_n_s32(m.bias[1] as i32),
        vdupq_n_s32(m.q[1] as i32),
    );
    let (l0, d0) = quant4(
        v0,
        load128_s32(sh0),
        dc_ac(m.iq[0], m.iq[1]),
        dc_ac(m.bias[0], m.bias[1]),
        dc_ac(m.q[0], m.q[1]),
    );
    let (l1, d1) = quant4(v1, load128_s32(sh1), iq1, bias1, q1);
    let (l2, d2) = quant4(v2, load128_s32(sh2), iq1, bias1, q1);
    let (l3, d3) = quant4(v3, load128_s32(sh3), iq1, bias1, q1);
    if let [a, b] = in_.as_chunks_mut::<8>().0 {
        store128_s16(a, narrow(d0, d1));
        store128_s16(b, narrow(d2, d3));
    }
    let mut r = [0i16; 16];
    if let [a, b] = r.as_chunks_mut::<8>().0 {
        store128_s16(a, narrow(l0, l1));
        store128_s16(b, narrow(l2, l3));
    }
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
    let s = vaddq_u64(
        vaddq_u64(sq(v0, d0), sq(v1, d1)),
        vaddq_u64(sq(v2, d2), sq(v3, d3)),
    );
    (last, vaddvq_u64(s) as i64)
}

#[target_feature(enable = "neon")]
fn quant_n(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    let mut nz = 0;
    for (n, (i, o)) in in_.iter_mut().zip(out.iter_mut()).enumerate() {
        if quant::<false>(m, i, o).0 >= 0 {
            nz |= 1 << n;
        }
    }
    nz
}

/// The squares of `v - d`, summed pairwise into two u64 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn sq(v: V, d: V) -> uint64x2_t {
    let e = vsubq_s32(v, d);
    vpaddlq_u32(vreinterpretq_u32_s32(vmulq_s32(e, e)))
}

/// (x * k) >> 16 on i32 lanes, wrapping like Go's int32.
#[target_feature(enable = "neon")]
#[inline]
fn mh(x: V, k: i32) -> V {
    vshrq_n_s32::<16>(vmulq_n_s32(x, k))
}

/// Two rows of u8 pixels as two i32 vectors of four.
#[target_feature(enable = "neon")]
#[inline]
fn widen(v: uint16x8_t) -> (V, V) {
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))),
        vreinterpretq_s32_u32(vmovl_high_u16(v)),
    )
}

/// Adds the inverse DCT of `c` to `rows`. The first pass runs down the columns of `c` with a lane
/// per column, the second down the rows, with a transpose between and one after.
#[target_feature(enable = "neon")]
fn idct(c: &[i16; 16], rows: &mut Rows) {
    let [c01, c23] = c.as_chunks::<8>().0 else {
        return;
    };
    let (x01, x23) = (load128_s16(c01), load128_s16(c23));
    if vmaxvq_u16(vreinterpretq_u16_s16(vorrq_s16(x01, x23))) == 0 {
        return;
    }
    let r0 = vmovl_s16(vget_low_s16(x01));
    let r1 = vmovl_high_s16(x01);
    let r2 = vmovl_s16(vget_low_s16(x23));
    let r3 = vmovl_high_s16(x23);
    let (c1, c2) = (85627, 35468);
    let a = vaddq_s32(r0, r2);
    let b = vsubq_s32(r0, r2);
    let cc = vsubq_s32(mh(r1, c2), mh(r3, c1));
    let d = vaddq_s32(mh(r1, c1), mh(r3, c2));
    let [x0, x1, x2, x3] = transpose(
        vaddq_s32(a, d),
        vaddq_s32(b, cc),
        vsubq_s32(b, cc),
        vsubq_s32(a, d),
    );
    let dc = vaddq_s32(x0, vdupq_n_s32(4));
    let a = vaddq_s32(dc, x2);
    let b = vsubq_s32(dc, x2);
    let cc = vsubq_s32(mh(x1, c2), mh(x3, c1));
    let d = vaddq_s32(mh(x1, c1), mh(x3, c2));
    let [p0, p1, p2, p3] = transpose(
        vshrq_n_s32::<3>(vaddq_s32(a, d)),
        vshrq_n_s32::<3>(vaddq_s32(b, cc)),
        vshrq_n_s32::<3>(vsubq_s32(b, cc)),
        vshrq_n_s32::<3>(vsubq_s32(a, d)),
    );
    let px = vreinterpretq_u8_u32(load128_u32(rows));
    let w01 = vmovl_u8(vget_low_u8(px));
    let w23 = vmovl_high_u8(px);
    let ((q0, q1), (q2, q3)) = (widen(w01), widen(w23));
    // Saturate to i16, then to u8: the clamp to 0..=255.
    let lo = vcombine_s16(vqmovn_s32(vaddq_s32(p0, q0)), vqmovn_s32(vaddq_s32(p1, q1)));
    let hi = vcombine_s16(vqmovn_s32(vaddq_s32(p2, q2)), vqmovn_s32(vaddq_s32(p3, q3)));
    let res = vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
    store128_u32(rows, vreinterpretq_u32_u8(res));
}

#[target_feature(enable = "neon")]
fn residuals_y(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    for (n, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = luma_pos(n);
        fdct(&rows_at(src, 16, off), &get_rows(z, y, x), blk);
    }
}

#[target_feature(enable = "neon")]
fn residuals_uv(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    for (c, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = chroma_pos(c);
        let src = if c >= 4 { v } else { u };
        fdct(&rows_at(src, 8, off), &get_rows(z, y, x), blk);
    }
}

#[target_feature(enable = "neon")]
fn idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    for (n, c) in dq.iter().enumerate() {
        let (y, x, _) = luma_pos(n);
        let mut r = get_rows(z, y, x);
        idct(c, &mut r);
        put_rows(z, &r, y, x);
    }
}

#[target_feature(enable = "neon")]
fn idct_adds_uv(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    for (c, b) in dq.iter().enumerate() {
        let (y, x, _) = chroma_pos(c);
        let mut r = get_rows(z, y, x);
        idct(b, &mut r);
        put_rows(z, &r, y, x);
    }
}
