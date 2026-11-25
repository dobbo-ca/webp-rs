//! NEON form of the encoder kernels of package B, the intra predictors and SSE.

use super::mem::{load64, load128, load128_u32};
use crate::simd::enc_pred::{Kernels, pred4_all_scalar, residual_quant3_scalar, row};
use std::arch::aarch64::*;

type Z = [[u8; 32]; 26];

pub(in crate::simd) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels {
        sse4x10,
        edge_sse,
        sse_block,
        // The scalar forms until the follow-up ports Go's kernel_arm64.s residualQuant3Asm and
        // pred4AllAsm.
        residual_quant3: residual_quant3_scalar,
        pred4_all: pred4_all_scalar,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "neon")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn sse4x10(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { sse4x10_kernel(pr, src) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn edge_sse(src: &[u8], stride: usize, n: usize, z: &Z, y: usize, x: usize, dc: i32) -> [i32; 4] {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { edge_sse_kernel(src, stride, n, z, y, x, dc) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn sse_block(src: &[u8], stride: usize, z: &Z, y: usize, x: usize, w: usize, h: usize) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { sse_block_kernel(src, stride, z, y, x, w, h) }
}

/// Adds the squares of the 16 byte differences `d` to the u32 lanes of `acc`. A square is at
/// most 65025, so it fits the u16 products.
#[target_feature(enable = "neon")]
#[inline]
fn add_sq(acc: uint32x4_t, d: uint8x16_t) -> uint32x4_t {
    let acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
    vpadalq_u16(acc, vmull_high_u8(d, d))
}

#[target_feature(enable = "neon")]
fn sse4x10_kernel(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    let s = vreinterpretq_u8_u32(load128_u32(src));
    let mut d = [0i32; 10];
    for (d, p) in d.iter_mut().zip(pr) {
        let a = vabdq_u8(s, vreinterpretq_u8_u32(load128_u32(p)));
        *d = vaddvq_u32(add_sq(vdupq_n_u32(0), a)) as i32;
    }
    d
}

/// Adds the squared differences of the DC, TM, VE and HE predictions against `s` to `acc`. Each
/// register holds the bytes of one 16-pixel row, or two 8-pixel rows.
#[target_feature(enable = "neon")]
#[inline]
fn edge_step(
    acc: &mut [uint32x4_t; 4],
    s: uint8x16_t,
    t: uint8x16_t,
    l: uint8x16_t,
    a: uint8x16_t,
    dc: uint8x16_t,
) {
    // clip8(l - a + t): add the positive part of l - a, subtract the negative one, both saturating.
    let tm = vqsubq_u8(vqaddq_u8(t, vqsubq_u8(l, a)), vqsubq_u8(a, l));
    for (acc, p) in acc.iter_mut().zip([dc, tm, t, l]) {
        *acc = add_sq(*acc, vabdq_u8(s, p));
    }
}

#[target_feature(enable = "neon")]
fn edge_sse_kernel(
    src: &[u8],
    stride: usize,
    n: usize,
    z: &Z,
    y: usize,
    x: usize,
    dc: i32,
) -> [i32; 4] {
    let mut acc = [vdupq_n_u32(0); 4];
    let a = vdupq_n_u8(z[y - 1][x - 1]);
    let dc = vdupq_n_u8(dc as u8);
    if n == 16 {
        let t = load128(row(&z[y - 1][x..]));
        for j in 0..16 {
            let s = load128(row(&src[j * stride..]));
            edge_step(&mut acc, s, t, vdupq_n_u8(z[y + j][x - 1]), a, dc);
        }
    } else {
        let t8 = load64(row(&z[y - 1][x..]));
        let t = vcombine_u8(t8, t8);
        for j in (0..8).step_by(2) {
            let s0 = load64(row(&src[j * stride..]));
            let s1 = load64(row(&src[(j + 1) * stride..]));
            let l = vcombine_u8(vdup_n_u8(z[y + j][x - 1]), vdup_n_u8(z[y + j + 1][x - 1]));
            edge_step(&mut acc, vcombine_u8(s0, s1), t, l, a, dc);
        }
    }
    let mut d = [0i32; 4];
    for (d, v) in d.iter_mut().zip(acc) {
        *d = vaddvq_u32(v) as i32;
    }
    d
}

#[target_feature(enable = "neon")]
fn sse_block_kernel(
    src: &[u8],
    stride: usize,
    z: &Z,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut acc = vdupq_n_u32(0);
    for j in 0..h {
        let (r, p) = (&src[j * stride..], &z[y + j][x..]);
        if w == 16 {
            let s = load128(row(r));
            let p = load128(row(p));
            acc = add_sq(acc, vabdq_u8(s, p));
        } else {
            let s = load64(row(r));
            let p = load64(row(p));
            let d = vabd_u8(s, p);
            acc = vpadalq_u16(acc, vmull_u8(d, d));
        }
    }
    vaddvq_u32(acc) as i32
}
