//! NEON form of the rate costs. NEON has no gather, so, as in the AVX2 form, the 16 table indices
//! of a block come out of one vector computation and are stored to an array, and sixteen plain
//! loads read the table (Go's `resCostAsm` and `costGridAsm`, kernel_arm64.s, do the same).
//! Positions outside `first..=last` read entry 127, which is 0 in every row.

use super::mem::*;
use crate::simd::enc_cost::{Kernels, Tab, scalar_res_cost};
use std::arch::aarch64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels {
        res_cost,
        cost_grid,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "neon")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn res_cost(t: &Tab, ctx: usize, first: usize, last: i32, lv: &[i16; 16]) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { res_cost_kernel(t, ctx, first, last, lv) }
}

/// Dispatch, as `res_cost`.
#[allow(unsafe_code)]
fn cost_grid(t: &Tab, first: usize, lv: &[[i16; 16]], cols: usize, lbits: u32, tbits: u32) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { cost_grid_kernel(t, first, lv, cols, lbits, tbits) }
}

static LANES_LO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
static LANES_HI: [u16; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
static BITS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

#[target_feature(enable = "neon")]
fn res_cost_kernel(t: &Tab, ctx: usize, first: usize, last: i32, lv: &[i16; 16]) -> i32 {
    if last < 0 {
        return t.empty[ctx];
    }
    block(t, ctx, first, last as usize, lv)
}

/// The table indices of the eight lanes of one half of a block: n * 512 + ctx * 128 + min(|v|, 67),
/// all ones outside `first..=last`. `cv` is the context of each lane.
#[target_feature(enable = "neon")]
#[inline]
fn indices(
    lane: uint16x8_t,
    cv: uint16x8_t,
    c67: uint16x8_t,
    first: uint16x8_t,
    last: uint16x8_t,
) -> uint16x8_t {
    let idx = vaddq_u16(vaddq_u16(vshlq_n_u16::<9>(lane), vshlq_n_u16::<7>(cv)), c67);
    vorrq_u16(
        idx,
        vorrq_u16(vcgtq_u16(lane, last), vcgtq_u16(first, lane)),
    )
}

/// The cost of a block with a level at `last >= first`.
#[target_feature(enable = "neon")]
#[inline]
fn block(t: &Tab, ctx: usize, first: usize, last: usize, lv: &[i16; 16]) -> i32 {
    let (Ok(flat), [lv_lo, lv_hi]) = (
        <&[u16; 8192]>::try_from(t.cost.as_flattened().as_flattened()),
        lv.as_chunks::<8>().0,
    ) else {
        return scalar_res_cost(t, ctx, first, last as i32, lv);
    };
    let (lane_lo, lane_hi) = (load128_u16(&LANES_LO), load128_u16(&LANES_HI));
    // |v| as u16 (i16::MIN stays 32768, as in the AVX2 form), then its clamps.
    let a_lo = vreinterpretq_u16_s16(vabsq_s16(load128_s16(lv_lo)));
    let a_hi = vreinterpretq_u16_s16(vabsq_s16(load128_s16(lv_hi)));
    let (two, k67) = (vdupq_n_u16(2), vdupq_n_u16(67));
    let (c2_lo, c2_hi) = (vminq_u16(a_lo, two), vminq_u16(a_hi, two));
    let (c67_lo, c67_hi) = (vminq_u16(a_lo, k67), vminq_u16(a_hi, k67));
    // The context of lane n is min(|lv[n - 1]|, 2): c2 shifted up a lane across the two halves,
    // with `ctx` at lane `first`.
    let up_lo = vextq_u16::<7>(vdupq_n_u16(0), c2_lo);
    let up_hi = vextq_u16::<7>(c2_lo, c2_hi);
    let (f, l) = (vdupq_n_u16(first as u16), vdupq_n_u16(last as u16));
    let cx = vdupq_n_u16(ctx as u16);
    let cv_lo = vbslq_u16(vceqq_u16(lane_lo, f), cx, up_lo);
    let cv_hi = vbslq_u16(vceqq_u16(lane_hi, f), cx, up_hi);
    let (mut ix_lo, mut ix_hi) = ([0u16; 8], [0u16; 8]);
    store128_u16(&mut ix_lo, indices(lane_lo, cv_lo, c67_lo, f, l));
    store128_u16(&mut ix_hi, indices(lane_hi, cv_hi, c67_hi, f, l));
    let (mut s0, mut s1) = (0i32, 0i32);
    for (&i, &j) in ix_lo.iter().zip(&ix_hi) {
        s0 += i32::from(flat[usize::from(i) & 8191]);
        s1 += i32::from(flat[usize::from(j) & 8191]);
    }
    let mut r = s0 + s1 + if ctx == 0 { t.nz0 } else { 0 };
    let same = vandq_u16(vceqq_u16(a_lo, c67_lo), vceqq_u16(a_hi, c67_hi));
    if vminvq_u16(same) != u16::MAX {
        // Some level is past 67 (or outside first..=last: the loop below skips those): the fixed
        // costs of the levels past the table.
        for l in &lv[first..=last] {
            let v = usize::from(l.unsigned_abs());
            if v > 67 {
                r += t.fixed[v & 2047] - t.fixed[67];
            }
        }
    }
    if last < 15 {
        r += i32::from(t.eob[last][usize::from(lv[last].unsigned_abs()).min(2)]);
    }
    r
}

/// Bit n set when `b[n]` is not zero.
#[target_feature(enable = "neon")]
#[inline]
fn nonzero_bits(b: &[i16; 16]) -> u32 {
    let [lo, hi] = b.as_chunks::<8>().0 else {
        return 0;
    };
    let w = load128_u16(&BITS);
    let m_lo = vandq_u16(vmvnq_u16(vceqzq_s16(load128_s16(lo))), w);
    let m_hi = vandq_u16(vmvnq_u16(vceqzq_s16(load128_s16(hi))), w);
    u32::from(vaddvq_u16(m_lo)) | u32::from(vaddvq_u16(m_hi)) << 8
}

#[target_feature(enable = "neon")]
fn cost_grid_kernel(
    t: &Tab,
    first: usize,
    lv: &[[i16; 16]],
    cols: usize,
    mut lbits: u32,
    mut tbits: u32,
) -> i32 {
    let mut r = 0;
    for (i, b) in lv.iter().enumerate() {
        let (y, x) = (i / cols, i % cols);
        let ctx = (((lbits >> y) & 1) + ((tbits >> x) & 1)) as usize;
        // The nonzero lanes at or after `first`.
        let nz = nonzero_bits(b) >> first << first;
        let last = 31 - nz.leading_zeros() as i32;
        r += if last < 0 {
            t.empty[ctx]
        } else {
            block(t, ctx, first, last as usize, b)
        };
        let has = u32::from(last >= 0);
        lbits = lbits & !(1 << y) | has << y;
        tbits = tbits & !(1 << x) | has << x;
    }
    r
}
