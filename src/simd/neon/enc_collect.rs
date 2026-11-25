//! NEON form of the encoder kernels of package E, the segment analysis bins. The fdct runs on 8
//! blocks at once, one block to a 16 bit lane, so that the row and column passes are plain
//! lane-wise arithmetic on registers that each hold one pixel position of every block (Go's
//! `collectAsm` does one block to a register instead).

use super::mem::{load64, load128, store128};
use crate::simd::enc_collect::{BinStats, Kernels, Z};
use std::arch::aarch64::*;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels {
        collect16,
        collect_uv,
        bin_stats,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "neon")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn collect16(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { collect16_kernel(src, stride, z) }
}

/// Dispatch, as `collect16`.
#[allow(unsafe_code)]
fn collect_uv(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { collect_uv_kernel(u, v, stride, z) }
}

/// Dispatch, as `collect16`.
#[allow(unsafe_code)]
fn bin_stats(b: &[u8]) -> BinStats {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { bin_stats_kernel(b) }
}

/// The 16 bytes of `s` from `o`.
fn at16(s: &[u8], o: usize) -> &[u8; 16] {
    &s[o..o + 16].as_chunks::<16>().0[0]
}

/// The 8 bytes of `s` from `o`.
fn at8(s: &[u8], o: usize) -> &[u8; 8] {
    &s[o..o + 8].as_chunks::<8>().0[0]
}

/// `a2 * k0 + a3 * k1 + round` shifted right by `S`, for the 8 word lanes, as saturated words.
#[target_feature(enable = "neon")]
#[inline]
fn rot<const S: i32>(a2: int16x8_t, a3: int16x8_t, k0: i16, k1: i16, round: i32) -> int16x8_t {
    let r = vdupq_n_s32(round);
    let lo = vmlal_n_s16(vmlal_n_s16(r, vget_low_s16(a2), k0), vget_low_s16(a3), k1);
    let hi = vmlal_high_n_s16(vmlal_high_n_s16(r, a2, k0), a3, k1);
    vcombine_s16(
        vqmovn_s32(vshrq_n_s32::<S>(lo)),
        vqmovn_s32(vshrq_n_s32::<S>(hi)),
    )
}

/// The bins, `min(|c| >> 3, 31)`, of the fdct of the blocks in the lanes, from their differences
/// `d[j][c]` at row j, column c. Register `4 * c + k` is coefficient k of column c.
#[target_feature(enable = "neon")]
fn fdct_bins(d: &[[int16x8_t; 4]; 4]) -> [uint16x8_t; 16] {
    // Row pass: t[j][c], dsp/enc.c FTransform_C.
    let mut t = [[vdupq_n_s16(0); 4]; 4];
    for (t, d) in t.iter_mut().zip(d) {
        let (a0, a1) = (vaddq_s16(d[0], d[3]), vaddq_s16(d[1], d[2]));
        let (a2, a3) = (vsubq_s16(d[1], d[2]), vsubq_s16(d[0], d[3]));
        t[0] = vshlq_n_s16::<3>(vaddq_s16(a0, a1));
        t[1] = rot::<9>(a2, a3, 2217, 5352, 1812);
        t[2] = vshlq_n_s16::<3>(vsubq_s16(a0, a1));
        t[3] = rot::<9>(a2, a3, -5352, 2217, 937);
    }
    // Column pass, then the bin.
    let (seven, one, k31) = (vdupq_n_s16(7), vdupq_n_s16(1), vdupq_n_u16(31));
    let mut out = [vdupq_n_u16(0); 16];
    for c in 0..4 {
        let (a0, a1) = (vaddq_s16(t[0][c], t[3][c]), vaddq_s16(t[1][c], t[2][c]));
        let (a2, a3) = (vsubq_s16(t[1][c], t[2][c]), vsubq_s16(t[0][c], t[3][c]));
        let c0 = vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(a0, a1), seven));
        let c2 = vshrq_n_s16::<4>(vaddq_s16(vsubq_s16(a0, a1), seven));
        // One more when a3 is not zero: 1 + (-1 where a3 is zero).
        let bump = vaddq_s16(one, vreinterpretq_s16_u16(vceqzq_s16(a3)));
        let c1 = vaddq_s16(rot::<16>(a2, a3, 2217, 5352, 12000), bump);
        let c3 = rot::<16>(a2, a3, -5352, 2217, 51000);
        for (k, v) in [c0, c1, c2, c3].into_iter().enumerate() {
            let bin = vshrq_n_u16::<3>(vreinterpretq_u16_s16(vabsq_s16(v)));
            out[4 * c + k] = vminq_u16(bin, k31);
        }
    }
    out
}

/// Bytes `4 * c + k` from bytes `4 * k + c`: [block][column] into [column][block].
#[target_feature(enable = "neon")]
#[inline]
fn transpose4_pattern() -> uint8x16_t {
    load128(&[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
}

/// From two registers y0, y1 of bytes [column][block] (4 blocks each), the four columns as 8
/// bytes: the 4 bytes of y0 then the 4 of y1.
#[target_feature(enable = "neon")]
#[inline]
fn cols(y0: uint8x16_t, y1: uint8x16_t) -> [uint8x8_t; 4] {
    let (a, b) = (vreinterpretq_u32_u8(y0), vreinterpretq_u32_u8(y1));
    let (z1, z2) = (
        vreinterpretq_u8_u32(vzip1q_u32(a, b)),
        vreinterpretq_u8_u32(vzip2q_u32(a, b)),
    );
    [
        vget_low_u8(z1),
        vget_high_u8(z1),
        vget_low_u8(z2),
        vget_high_u8(z2),
    ]
}

/// Source minus prediction per column, as words.
#[target_feature(enable = "neon")]
#[inline]
fn diff(s: [uint8x8_t; 4], p: [uint8x8_t; 4]) -> [int16x8_t; 4] {
    std::array::from_fn(|c| vreinterpretq_s16_u16(vsubl_u8(s[c], p[c])))
}

/// Packs the 16 registers of bins into 128 bytes.
#[target_feature(enable = "neon")]
fn store_bins(out: &mut [u8; 128], w: &[uint16x8_t; 16]) {
    for (o, w) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(w.as_chunks::<2>().0)
    {
        store128(o, vcombine_u8(vmovn_u16(w[0]), vmovn_u16(w[1])));
    }
}

#[target_feature(enable = "neon")]
fn collect16_kernel(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    let pat = transpose4_pattern();
    let mut out = [0u8; 256];
    for (h, o) in out.as_chunks_mut::<128>().0.iter_mut().enumerate() {
        // Half h: the blocks of block rows 2h and 2h + 1, block 4 * (by - 2h) + bx in lane.
        let zero = vdupq_n_s16(0);
        let mut d = [[zero; 4]; 4];
        for (j, d) in d.iter_mut().enumerate() {
            let y = |by: usize| {
                let by = 2 * h + by;
                (
                    vqtbl1q_u8(load128(at16(src, (4 * by + j) * stride)), pat),
                    vqtbl1q_u8(load128(at16(&z[1 + 4 * by + j], 8)), pat),
                )
            };
            let ((s0, p0), (s1, p1)) = (y(0), y(1));
            *d = diff(cols(s0, s1), cols(p0, p1));
        }
        store_bins(o, &fdct_bins(&d));
    }
    out
}

#[target_feature(enable = "neon")]
fn collect_uv_kernel(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    let pat = transpose4_pattern();
    let zero = vdupq_n_s16(0);
    let mut d = [[zero; 4]; 4];
    for (j, d) in d.iter_mut().enumerate() {
        // The 16 bytes of a plane's block rows 0 and 1, [block][column], turned into
        // [column][block].
        let plane = |src: &[u8], bx: usize| {
            let s = vcombine_u8(
                load64(at8(src, j * stride)),
                load64(at8(src, (4 + j) * stride)),
            );
            let p = vcombine_u8(load64(at8(&z[18 + j], bx)), load64(at8(&z[22 + j], bx)));
            (vqtbl1q_u8(s, pat), vqtbl1q_u8(p, pat))
        };
        let ((su, pu), (sv, pv)) = (plane(u, 8), plane(v, 24));
        // Lane 4 * plane + block: U blocks 0-3 then V blocks 4-7.
        *d = diff(cols(su, sv), cols(pu, pv));
    }
    let mut out = [0u8; 128];
    store_bins(&mut out, &fdct_bins(&d));
    out
}

/// Go's `binStatsAsm`: the largest byte and the counts of the bytes 0 to 3.
#[target_feature(enable = "neon")]
fn bin_stats_kernel(b: &[u8]) -> BinStats {
    let ks = [vdupq_n_u8(0), vdupq_n_u8(1), vdupq_n_u8(2), vdupq_n_u8(3)];
    let (chunks, tail) = b.as_chunks::<16>();
    let mut max = vdupq_n_u8(0);
    let mut count = [0i32; 4];
    // Byte counters take 255 steps.
    for run in chunks.chunks(255) {
        let mut acc = [vdupq_n_u8(0); 4];
        for c in run {
            let v = load128(c);
            max = vmaxq_u8(max, v);
            for (a, k) in acc.iter_mut().zip(ks) {
                *a = vsubq_u8(*a, vceqq_u8(v, k));
            }
        }
        for (n, a) in count.iter_mut().zip(acc) {
            *n += i32::from(vaddlvq_u8(a));
        }
    }
    let mut last = vmaxvq_u8(max);
    for &v in tail {
        last = last.max(v);
        if v < 4 {
            count[usize::from(v)] += 1;
        }
    }
    (i32::from(last), count[0], count[1], count[2], count[3])
}
