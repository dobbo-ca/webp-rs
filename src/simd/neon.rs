//! NEON kernels: luma of 16 pixels a step, squared differences of 16 bytes a step (8 for a tail).

use super::{Kernels, scalar};
use std::arch::aarch64::*;

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

/// Bytes per accumulation block of `sse_kernel`: the u32 lanes take at most 4 * 255^2 a step, so
/// 4096 steps stay below 2^32.
const BLOCK: usize = 16 * 4096;

pub(super) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels { luma_row, sse_row })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "neon")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn luma_row(dst: &mut [u8], src: &[u8]) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { luma_kernel(dst, src) }
}

/// Dispatch, as `luma_row`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn sse_row(a: &[u8], b: &[u8]) -> u64 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { sse_kernel(a, b) }
}

/// Luma of 8 pixels from their R, G and B bytes, as u16 lanes.
#[target_feature(enable = "neon")]
fn y8(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    let (r, g, b) = (vmovl_u8(r), vmovl_u8(g), vmovl_u8(b));
    let round = vdupq_n_u32((1 << 15) + (16 << 16));
    let lo = vmlal_n_u16(
        vmlal_n_u16(
            vmlal_n_u16(round, vget_low_u16(r), 16839),
            vget_low_u16(g),
            33059,
        ),
        vget_low_u16(b),
        6420,
    );
    let hi = vmlal_n_u16(
        vmlal_n_u16(
            vmlal_n_u16(round, vget_high_u16(r), 16839),
            vget_high_u16(g),
            33059,
        ),
        vget_high_u16(b),
        6420,
    );
    vcombine_u16(vshrn_n_u32::<16>(lo), vshrn_n_u32::<16>(hi))
}

#[target_feature(enable = "neon")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len() / 4) & !15;
    for (d, s) in dst[..n]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<64>().0.iter())
    {
        let px = load4x16(s);
        let lo = y8(vget_low_u8(px.0), vget_low_u8(px.1), vget_low_u8(px.2));
        let hi = y8(vget_high_u8(px.0), vget_high_u8(px.1), vget_high_u8(px.2));
        store128(d, vcombine_u8(vmovn_u16(lo), vmovn_u16(hi)));
    }
    scalar::luma_row(&mut dst[n..], &src[4 * n..]);
}

#[target_feature(enable = "neon")]
fn sse_kernel(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let n16 = n & !15;
    let mut total = 0u64;
    for (ba, bb) in a[..n16].chunks(BLOCK).zip(b[..n16].chunks(BLOCK)) {
        let mut acc = vdupq_n_u32(0);
        for (ca, cb) in ba
            .as_chunks::<16>()
            .0
            .iter()
            .zip(bb.as_chunks::<16>().0.iter())
        {
            let d = vabdq_u8(load128(ca), load128(cb));
            acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
            acc = vpadalq_u16(acc, vmull_high_u8(d, d));
        }
        total += vaddlvq_u32(acc);
    }
    let mut i = n16;
    if let (Some(ca), Some(cb)) = (a[i..].first_chunk::<8>(), b[i..].first_chunk::<8>()) {
        let d = vabd_u8(load64(ca), load64(cb));
        total += vaddlvq_u32(vpadalq_u16(vdupq_n_u32(0), vmull_u8(d, d)));
        i += 8;
    }
    total + scalar::sse_row(&a[i..], &b[i..])
}
