//! AVX2 form of the rate costs. The 16 table indices of a block come out of one vector
//! computation (Go's `cost4Asm` gathers them with `VPGATHERDD`, which is slow on most cores;
//! here sixteen plain loads do it), and positions outside `first..=last` read entry 127, which is
//! 0 in every row.

use super::mem::*;
use crate::simd::enc_cost::{Kernels, Tab, scalar_res_cost};
use std::arch::x86_64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels {
        res_cost,
        cost_grid,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "avx2")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn res_cost(t: &Tab, ctx: usize, first: usize, last: i32, lv: &[i16; 16]) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { res_cost_kernel(t, ctx, first, last, lv) }
}

/// Dispatch, as `res_cost`.
#[allow(unsafe_code)]
fn cost_grid(t: &Tab, first: usize, lv: &[[i16; 16]], cols: usize, lbits: u32, tbits: u32) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { cost_grid_kernel(t, first, lv, cols, lbits, tbits) }
}

static LANES: [i16; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[target_feature(enable = "avx2")]
fn res_cost_kernel(t: &Tab, ctx: usize, first: usize, last: i32, lv: &[i16; 16]) -> i32 {
    if last < 0 {
        return t.empty[ctx];
    }
    block(t, ctx, first, last as usize, lv)
}

/// The cost of a block with a level at `last >= first`.
#[target_feature(enable = "avx2")]
#[inline]
fn block(t: &Tab, ctx: usize, first: usize, last: usize, lv: &[i16; 16]) -> i32 {
    let Ok(flat) = <&[u16; 8192]>::try_from(t.cost.as_flattened().as_flattened()) else {
        return scalar_res_cost(t, ctx, first, last as i32, lv);
    };
    let lane = load256_i16(&LANES);
    let a = _mm256_abs_epi16(load256_i16(lv));
    let c2 = _mm256_min_epu16(a, _mm256_set1_epi16(2));
    let c67 = _mm256_min_epu16(a, _mm256_set1_epi16(67));
    // The context of lane n is min(|lv[n - 1]|, 2): c2 shifted up a word across the register,
    // with `ctx` at lane `first`.
    let up = _mm256_alignr_epi8::<14>(c2, _mm256_permute2x128_si256::<0x08>(c2, c2));
    let at_first = _mm256_cmpeq_epi16(lane, _mm256_set1_epi16(first as i16));
    let cv = _mm256_blendv_epi8(up, _mm256_set1_epi16(ctx as i16), at_first);
    // idx = n * 512 + ctx * 128 + min(|v|, 67); lanes outside first..=last become all ones, which
    // the mask below turns into entry 127 of the last row.
    let idx = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_slli_epi16::<9>(lane), _mm256_slli_epi16::<7>(cv)),
        c67,
    );
    let outside = _mm256_or_si256(
        _mm256_cmpgt_epi16(lane, _mm256_set1_epi16(last as i16)),
        _mm256_cmpgt_epi16(_mm256_set1_epi16(first as i16), lane),
    );
    let mut ix = [0i16; 16];
    store256_i16(&mut ix, _mm256_or_si256(idx, outside));
    let (lo, hi) = ix.split_at(8);
    let (mut s0, mut s1) = (0i32, 0i32);
    for (&i, &j) in lo.iter().zip(hi) {
        s0 += i32::from(flat[i as u16 as usize & 8191]);
        s1 += i32::from(flat[j as u16 as usize & 8191]);
    }
    let mut r = s0 + s1 + if ctx == 0 { t.nz0 } else { 0 };
    if _mm256_movemask_epi8(_mm256_cmpeq_epi16(a, c67)) != -1 {
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

#[target_feature(enable = "avx2")]
fn cost_grid_kernel(
    t: &Tab,
    first: usize,
    lv: &[[i16; 16]],
    cols: usize,
    mut lbits: u32,
    mut tbits: u32,
) -> i32 {
    let zero = _mm256_setzero_si256();
    let mut r = 0;
    for (i, b) in lv.iter().enumerate() {
        let (y, x) = (i / cols, i % cols);
        let ctx = (((lbits >> y) & 1) + ((tbits >> x) & 1)) as usize;
        // Two bits a lane; the nonzero lanes at or after `first`.
        let zeros = _mm256_movemask_epi8(_mm256_cmpeq_epi16(load256_i16(b), zero)) as u32;
        let nz = !zeros >> (2 * first) << (2 * first);
        let last = 15 - (nz.leading_zeros() >> 1) as i32;
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
