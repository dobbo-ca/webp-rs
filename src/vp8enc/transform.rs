//! Forward transforms (libwebp enc.c, BSD-3-Clause), the inverse WHT (x/image vp8 idct.go,
//! BSD-3-Clause) and the RGB to YUV conversion (libwebp picture_csp_enc.c, yuv.c).

use super::predict::Ybr;
use crate::fmath::pow;
use crate::simd::enc_block::{self, get_rows, rows_at};
use std::sync::OnceLock;

/// The fdct of `src` minus the prediction at (y, x) of `z`.
pub(crate) fn residual(
    z: &Ybr,
    blk: &mut [i16; 16],
    src: &[u8],
    stride: usize,
    y: usize,
    x: usize,
) {
    enc_block::fdct4(&rows_at(src, stride, 0), &get_rows(z, y, x), blk);
}

/// Rounding that makes x/image's iWHT exact.
const WHT_ROUND: [i32; 16] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];

fn fwht_with(dc: &mut [i16; 16], round: &[i32; 16]) {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let (p0, p1, p2, p3) = (
            i32::from(dc[i * 4]),
            i32::from(dc[i * 4 + 1]),
            i32::from(dc[i * 4 + 2]),
            i32::from(dc[i * 4 + 3]),
        );
        let (a0, a1, a2, a3) = (p0 + p2, p1 + p3, p1 - p3, p0 - p2);
        t[i * 4] = a0 + a1;
        t[i * 4 + 1] = a3 + a2;
        t[i * 4 + 2] = a3 - a2;
        t[i * 4 + 3] = a0 - a1;
    }
    for i in 0..4 {
        let (a0, a1, a2, a3) = (
            t[i] + t[8 + i],
            t[4 + i] + t[12 + i],
            t[4 + i] - t[12 + i],
            t[i] - t[8 + i],
        );
        dc[i] = ((a0 + a1 + round[i]) >> 1) as i16;
        dc[4 + i] = ((a3 + a2 + round[4 + i]) >> 1) as i16;
        dc[8 + i] = ((a3 - a2 + round[8 + i]) >> 1) as i16;
        dc[12 + i] = ((a0 - a1 + round[12 + i]) >> 1) as i16;
    }
}

/// libwebp FTransformWHT_C plus `WHT_ROUND`.
pub(crate) fn fwht(dc: &mut [i16; 16]) {
    fwht_with(dc, &WHT_ROUND);
}

/// FTransformWHT_C: `fwht` without the x/image rounding.
pub(crate) fn fwht_plain(dc: &mut [i16; 16]) {
    fwht_with(dc, &[0; 16]);
}

/// x/image inverseWHT16, raster order out.
pub(crate) fn iwht(c: &[i16; 16]) -> [i16; 16] {
    let mut m = [0i32; 16];
    for i in 0..4 {
        let a0 = i32::from(c[i]) + i32::from(c[12 + i]);
        let a1 = i32::from(c[4 + i]) + i32::from(c[8 + i]);
        let a2 = i32::from(c[4 + i]) - i32::from(c[8 + i]);
        let a3 = i32::from(c[i]) - i32::from(c[12 + i]);
        m[i] = a0 + a1;
        m[8 + i] = a0 - a1;
        m[4 + i] = a3 + a2;
        m[12 + i] = a3 - a2;
    }
    let mut o = [0i16; 16];
    for i in 0..4 {
        let dc = m[i * 4] + 3;
        let a0 = dc + m[3 + i * 4];
        let a1 = m[1 + i * 4] + m[2 + i * 4];
        let a2 = m[1 + i * 4] - m[2 + i * 4];
        let a3 = dc - m[3 + i * 4];
        o[i * 4] = ((a0 + a1) >> 3) as i16;
        o[i * 4 + 1] = ((a3 + a2) >> 3) as i16;
        o[i * 4 + 2] = ((a0 - a1) >> 3) as i16;
        o[i * 4 + 3] = ((a3 - a2) >> 3) as i16;
    }
    o
}

/// Adds the inverse DCT of c at (y, x).
pub(crate) fn idct_add(z: &mut Ybr, c: &[i16; 16], y: usize, x: usize) {
    enc_block::idct_add(z, c, y, x);
}

/// The fdct of the 16 luma blocks of src against the prediction in z.
pub(crate) fn residuals_y(z: &Ybr, dq: &mut [[i16; 16]; 16], src: &[u8; 256]) {
    enc_block::residuals_y(z, src, dq);
}

/// The fdct of the 8 chroma blocks against the prediction in z.
pub(crate) fn residuals_uv(z: &Ybr, dq: &mut [[i16; 16]; 8], u: &[u8; 64], v: &[u8; 64]) {
    enc_block::residuals_uv(z, u, v, dq);
}

/// Adds the 16 luma blocks to z.
pub(crate) fn idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    enc_block::idct_adds_y(z, dq);
}

/// Adds the 8 chroma blocks to z.
pub(crate) fn idct_adds_uv(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    enc_block::idct_adds_uv(z, dq);
}

/// gammaToLinear and linearToGamma, libwebp yuv.c:237-251.
pub(super) struct Gamma {
    pub(super) to_linear: [i32; 256],
    pub(super) to_gamma: [i32; 33],
}

pub(super) fn gamma() -> &'static Gamma {
    static G: OnceLock<Gamma> = OnceLock::new();
    G.get_or_init(|| {
        let mut g = Gamma {
            to_linear: [0; 256],
            to_gamma: [0; 33],
        };
        for v in 0..256 {
            g.to_linear[v] = (pow(v as f64 * (1.0 / 255.0), 0.8) * 4095.0 + 0.5) as i32;
        }
        for v in 0..33 {
            g.to_gamma[v] = (255.0 * pow(v as f64 * (128.0 / 4095.0), 1.25) + 0.5) as i32;
        }
        g
    })
}

/// Interpolate plus LinearToGamma, yuv.c:257-276.
fn lin_to_gamma(g: &Gamma, s: i32) -> i32 {
    let (p, x) = ((s >> 9) as usize, s & 511);
    (g.to_gamma[p + 1] * x + g.to_gamma[p] * (512 - x) + 64) >> 7
}

fn chroma_of(sr: i32, sg: i32, sb: i32) -> (u8, u8) {
    (
        ((-9719 * sr - 19081 * sg + 28800 * sb + (1 << 17) + (128 << 18)) >> 18) as u8,
        ((28800 * sr - 24116 * sg - 4684 * sb + (1 << 17) + (128 << 18)) >> 18) as u8,
    )
}

/// The destination of a colour conversion: planes with their strides and the size of the area.
pub(crate) struct Dst<'a> {
    pub y: &'a mut [u8],
    pub ys: usize,
    pub cb: &'a mut [u8],
    pub cr: &'a mut [u8],
    pub cs: usize,
    pub w: usize,
    pub h: usize,
}

/// libwebp ImportYUVAFromRGBA, default gamma. With `a` None it assumes opaque pixels and reports
/// whether they were; with `a` (stride w) it copies alpha there and weights the chroma of partly
/// transparent 2x2 blocks by alpha, picture_csp_enc.c:362-437.
pub(crate) fn rgba_to_ycbcr(
    dst: &mut Dst,
    pix: &[u8],
    stride: usize,
    a: Option<&mut [u8]>,
) -> bool {
    let (w, h) = (dst.w, dst.h);
    for y in 0..h {
        crate::simd::luma_row(
            &mut dst.y[y * dst.ys..y * dst.ys + w],
            &pix[y * stride..y * stride + 4 * w],
        );
    }
    let g = gamma();
    if let Some(a) = a {
        for y in 0..h {
            let (row, p) = (
                &mut a[y * w..y * w + w],
                &pix[y * stride..y * stride + 4 * w],
            );
            for (x, r) in row.iter_mut().enumerate() {
                *r = p[4 * x + 3];
            }
        }
        for cy in 0..h.div_ceil(2) {
            let (r0, r1) = (2 * cy, (2 * cy + 1).min(h - 1));
            let p0 = &pix[r0 * stride..r0 * stride + 4 * w];
            let p1 = &pix[r1 * stride..r1 * stride + 4 * w];
            for cx in 0..w.div_ceil(2) {
                let (o0, mut o1) = (8 * cx, 8 * cx + 4);
                if o1 >= p0.len() {
                    o1 = o0;
                }
                let (a, b) = (&p0[o0..o0 + 4], &p0[o1..o1 + 4]);
                let (c, d) = (&p1[o0..o0 + 4], &p1[o1..o1 + 4]);
                let t = u32::from(a[3]) + u32::from(b[3]) + u32::from(c[3]) + u32::from(d[3]);
                let ch = |i: usize| {
                    if t == 4 * 255 || t == 0 {
                        lin_to_gamma(
                            g,
                            g.to_linear[a[i] as usize]
                                + g.to_linear[b[i] as usize]
                                + g.to_linear[c[i] as usize]
                                + g.to_linear[d[i] as usize],
                        )
                    } else {
                        weighted(g, a, b, c, d, i, t)
                    }
                };
                let (cb, cr) = chroma_of(ch(0), ch(1), ch(2));
                dst.cb[cy * dst.cs + cx] = cb;
                dst.cr[cy * dst.cs + cx] = cr;
            }
        }
        return false;
    }
    let mut and = 255u8;
    for cy in 0..h.div_ceil(2) {
        let (r0, r1) = (2 * cy, (2 * cy + 1).min(h - 1));
        let p0 = &pix[r0 * stride..r0 * stride + 4 * w];
        let p1 = &pix[r1 * stride..r1 * stride + 4 * w];
        let cw = w.div_ceil(2);
        let (n, an) = crate::simd::enc_chroma::chroma_row(
            &mut dst.cb[cy * dst.cs..cy * dst.cs + cw],
            &mut dst.cr[cy * dst.cs..cy * dst.cs + cw],
            p0,
            p1,
            &g.to_linear,
            &g.to_gamma,
        );
        and &= an;
        for cx in n..cw {
            let (o0, mut o1) = (8 * cx, 8 * cx + 4);
            if o1 >= p0.len() {
                o1 = o0;
            }
            let (a, b) = (&p0[o0..o0 + 4], &p0[o1..o1 + 4]);
            let (c, d) = (&p1[o0..o0 + 4], &p1[o1..o1 + 4]);
            and &= a[3] & b[3] & c[3] & d[3];
            let s = |i: usize| {
                lin_to_gamma(
                    g,
                    g.to_linear[a[i] as usize]
                        + g.to_linear[b[i] as usize]
                        + g.to_linear[c[i] as usize]
                        + g.to_linear[d[i] as usize],
                )
            };
            let (cb, cr) = chroma_of(s(0), s(1), s(2));
            dst.cb[cy * dst.cs + cx] = cb;
            dst.cr[cy * dst.cs + cx] = cr;
        }
    }
    and == 255
}

/// LinearToGammaWeighted for channel i with total alpha t. The arithmetic wraps like the C's
/// uint32; kInvAlpha[t] is (1<<19)/t.
fn weighted(g: &Gamma, a: &[u8], b: &[u8], c: &[u8], d: &[u8], i: usize, t: u32) -> i32 {
    let term = |p: &[u8]| u32::from(p[3]).wrapping_mul(g.to_linear[p[i] as usize] as u32);
    let s = term(a)
        .wrapping_add(term(b))
        .wrapping_add(term(c))
        .wrapping_add(term(d));
    lin_to_gamma(g, (s.wrapping_mul((1u32 << 19) / t) >> 17) as i32)
}
