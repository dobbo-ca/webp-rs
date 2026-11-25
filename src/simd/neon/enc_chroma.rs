//! NEON form of the encoder kernels of package E, the opaque chroma row conversion: Go's
//! `chromaAsm` (kernel_arm64.s), 8 blocks a pass. gammaToLinear is a byte table lookup, not a
//! gather: the low byte by four 64 byte tables, the high byte by one 16 byte table (see
//! `Tabs::h`).

use super::mem::{load4x16, load128, store64, store128};
use crate::simd::enc_chroma::Kernels;
use std::arch::aarch64::*;
use std::sync::OnceLock;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels { chroma })
}

/// Dispatch: the kernel is a safe `#[target_feature(enable = "neon")]` fn, so calling it from
/// here needs `unsafe`. Only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn chroma(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { chroma_kernel(cb, cr, p0, p1, to_linear, to_gamma) }
}

/// The lookup tables, derived once from libwebp's tables, which are constants.
struct Tabs {
    /// `(h[v >> 4] + v) >> 4`, with the byte add wrapping, is `to_linear[v] >> 8`. to_linear(255)
    /// is 4095, so the high byte is at most 15. A run of 16 inputs crosses at most one multiple
    /// of 256 (to_linear is concave and its first runs span under 512), and with the step at
    /// offset s, `h[q] = 16 H + 16 - s - 16 q` gives `16 H + 16 - s + r` in `[16 H, 16 H + 30]`.
    h: [u8; 16],
    /// The low bytes of `to_linear`, by quarter.
    g: [[[u8; 16]; 4]; 4],
    /// `to_gamma[p]` and the step `to_gamma[p + 1] - to_gamma[p]`, p below 32, whose steps fit a
    /// byte.
    l: [[u8; 16]; 2],
    d: [[u8; 16]; 2],
}

fn tabs(to_linear: &[i32; 256], to_gamma: &[i32; 33]) -> &'static Tabs {
    static T: OnceLock<Tabs> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tabs {
            h: [0; 16],
            g: [[[0; 16]; 4]; 4],
            l: [[0; 16]; 2],
            d: [[0; 16]; 2],
        };
        for (v, &g) in to_linear.iter().enumerate() {
            t.g[v / 64][v % 64 / 16][v % 16] = g as u8;
        }
        let hi = |v: usize| to_linear[v] >> 8;
        for (q, h) in t.h.iter_mut().enumerate() {
            let s = (1..16)
                .find(|&r| hi(16 * q + r) != hi(16 * q))
                .unwrap_or(16);
            *h = (16 * hi(16 * q) + 16 - s as i32 - 16 * q as i32) as u8;
        }
        for p in 0..32 {
            t.l[p / 16][p % 16] = to_gamma[p] as u8;
            t.d[p / 16][p % 16] = (to_gamma[p + 1] - to_gamma[p]) as u8;
        }
        t
    })
}

/// to_linear of the 16 bytes in `v`, as two registers of eight u16.
#[target_feature(enable = "neon")]
#[inline]
fn linear(v: uint8x16_t, g: &[uint8x16x4_t; 4], h: uint8x16_t) -> [uint16x8_t; 2] {
    // Quarter q of the table answers for the bytes 64 q to 64 q + 63; the others index past it,
    // which `vqtbx4q_u8` leaves as they are.
    let mut lo = vqtbl4q_u8(g[0], v);
    for (q, g) in g.iter().enumerate().skip(1) {
        lo = vqtbx4q_u8(lo, *g, vsubq_u8(v, vdupq_n_u8(64 * q as u8)));
    }
    let hi = vshrq_n_u8::<4>(vaddq_u8(vqtbl1q_u8(h, vshrq_n_u8::<4>(v)), v));
    [
        vreinterpretq_u16_u8(vzip1q_u8(lo, hi)),
        vreinterpretq_u16_u8(vzip2q_u8(lo, hi)),
    ]
}

/// The sums of the 2x2 blocks of one channel: the 16 pixels `a` of row 0 and `b` of row 1, as the
/// eight sums of blocks 0 to 7.
#[target_feature(enable = "neon")]
#[inline]
fn sums(a: uint8x16_t, b: uint8x16_t, g: &[uint8x16x4_t; 4], h: uint8x16_t) -> uint16x8_t {
    let (a, b) = (linear(a, g, h), linear(b, g, h));
    vpaddq_u16(vaddq_u16(a[0], b[0]), vaddq_u16(a[1], b[1]))
}

/// linToGamma of the eight sums in `s`, as 4 * L[p] + (D[p] * x + 64) >> 7 with p = s >> 9 and
/// x = s & 511. D[p] * x fits 16 bits: the steps are small.
#[target_feature(enable = "neon")]
#[inline]
fn lin_to_gamma(s: uint16x8_t, l: uint8x16x2_t, d: uint8x16x2_t) -> int16x8_t {
    let p = vmovn_u16(vshrq_n_u16::<9>(s));
    let (l, d) = (vmovl_u8(vqtbl2_u8(l, p)), vmovl_u8(vqtbl2_u8(d, p)));
    let x = vandq_u16(s, vdupq_n_u16(511));
    let r = vrshrq_n_u16::<7>(vmulq_u16(d, x));
    vreinterpretq_s16_u16(vaddq_u16(r, vshlq_n_u16::<2>(l)))
}

/// One chroma value of the eight blocks: the channels weighted, rounded and shifted.
#[target_feature(enable = "neon")]
#[inline]
fn weigh(c: [int16x8_t; 3], k: [i16; 3]) -> uint8x8_t {
    let bias = vdupq_n_s32((1 << 17) + (128 << 18));
    let mut lo = bias;
    let mut hi = bias;
    for (c, k) in c.into_iter().zip(k) {
        lo = vmlal_n_s16(lo, vget_low_s16(c), k);
        hi = vmlal_high_n_s16(hi, c, k);
    }
    // Cb and Cr lie in [16, 240], so the narrowing is exact.
    vqmovun_s16(vcombine_s16(
        vmovn_s32(vshrq_n_s32::<18>(lo)),
        vmovn_s32(vshrq_n_s32::<18>(hi)),
    ))
}

#[target_feature(enable = "neon")]
#[inline]
fn tab4(t: &[[u8; 16]; 4]) -> uint8x16x4_t {
    uint8x16x4_t(
        load128(&t[0]),
        load128(&t[1]),
        load128(&t[2]),
        load128(&t[3]),
    )
}

#[target_feature(enable = "neon")]
#[inline]
fn tab2(t: &[[u8; 16]; 2]) -> uint8x16x2_t {
    uint8x16x2_t(load128(&t[0]), load128(&t[1]))
}

/// The chroma of `cb.len()` blocks, 8 a pass.
#[target_feature(enable = "neon")]
fn chroma_kernel(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    let t = tabs(to_linear, to_gamma);
    let g = [tab4(&t.g[0]), tab4(&t.g[1]), tab4(&t.g[2]), tab4(&t.g[3])];
    let (h, l, d) = (load128(&t.h), tab2(&t.l), tab2(&t.d));
    let mut alpha = vdupq_n_u8(255);
    let rows = p0.as_chunks::<64>().0.iter().zip(p1.as_chunks::<64>().0);
    let outs = cb
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(cr.as_chunks_mut::<8>().0.iter_mut());
    for ((r0, r1), (o_cb, o_cr)) in rows.zip(outs) {
        let (a, b) = (load4x16(r0), load4x16(r1));
        alpha = vandq_u8(vandq_u8(alpha, a.3), b.3);
        let r = lin_to_gamma(sums(a.0, b.0, &g, h), l, d);
        let gr = lin_to_gamma(sums(a.1, b.1, &g, h), l, d);
        let bl = lin_to_gamma(sums(a.2, b.2, &g, h), l, d);
        store64(o_cb, weigh([r, gr, bl], [-9719, -19081, 28800]));
        store64(o_cr, weigh([r, gr, bl], [28800, -24116, -4684]));
    }
    let mut a = [0u8; 16];
    store128(&mut a, alpha);
    a.iter().fold(255, |x, &v| x & v)
}
