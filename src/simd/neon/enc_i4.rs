//! NEON form of the i4 mode trials: four lanes, each its own mode (or block), so every step is
//! lane-wise 32-bit math with no cross-lane steps. Same arithmetic as the scalar reference in
//! `simd::enc_i4`, which the tests compare against bit for bit. Go has no NEON form of these
//! kernels that matches the pure-Go definition (its dot-product trial is a different kernel), so
//! this is written from the AVX2 tier.

use super::mem::{load128_s32, store128_s32};
use crate::simd::enc_i4::{Kernels, WEIGHT_Y};
use crate::tables::ZIGZAG;
use crate::vp8enc::quant::QuantMatrix;
use std::arch::aarch64::*;

type V = int32x4_t;

/// `x >> N` on the lanes as unsigned.
macro_rules! srl {
    ($x:expr, $n:literal) => {
        vreinterpretq_s32_u32(vshrq_n_u32::<$n>(vreinterpretq_u32_s32($x)))
    };
}

pub(in crate::simd) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels {
        i4_trials,
        rec_metrics16,
        tt_mb,
    })
}

/// Dispatch, as `luma_row` in the parent: only reachable through `detect`, which checked NEON.
#[allow(unsafe_code)]
fn i4_trials(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { i4_trials_kernel(q, pr, src, lv, rec, res) }
}

/// Dispatch, as `i4_trials`.
#[allow(unsafe_code)]
fn rec_metrics16(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { rec_metrics16_kernel(c, z, src, st) }
}

/// Dispatch, as `i4_trials`.
#[allow(unsafe_code)]
fn tt_mb(src: &[u8; 256]) -> [i32; 16] {
    // SAFETY: `detect` returned this fn only after `is_aarch64_feature_detected!("neon")`.
    unsafe { tt_mb_kernel(src) }
}

#[target_feature(enable = "neon")]
#[inline]
fn k(x: i32) -> V {
    vdupq_n_s32(x)
}

#[target_feature(enable = "neon")]
#[inline]
fn add(a: V, b: V) -> V {
    vaddq_s32(a, b)
}

#[target_feature(enable = "neon")]
#[inline]
fn sub(a: V, b: V) -> V {
    vsubq_s32(a, b)
}

/// `a * x` per lane, low 32 bits.
#[target_feature(enable = "neon")]
#[inline]
fn mulk(a: V, x: i32) -> V {
    vmulq_s32(a, k(x))
}

/// All ones in the lanes where `a == b`.
#[target_feature(enable = "neon")]
#[inline]
fn eq(a: V, b: V) -> V {
    vreinterpretq_s32_u32(vceqq_s32(a, b))
}

/// The low 16 bits of each lane, sign-extended: `as i16` of the scalar code.
#[target_feature(enable = "neon")]
#[inline]
fn s16(a: V) -> V {
    vshrq_n_s32::<16>(vshlq_n_s32::<16>(a))
}

/// `(a * x) >> 16`, the IDCT's multiplications.
#[target_feature(enable = "neon")]
#[inline]
fn mulsh(a: V, x: i32) -> V {
    vshrq_n_s32::<16>(mulk(a, x))
}

/// The 16 byte values of four row words per lane.
#[target_feature(enable = "neon")]
#[inline]
fn unpack(w: &[V; 4]) -> [V; 16] {
    let m = k(255);
    let mut o = [k(0); 16];
    for j in 0..4 {
        o[4 * j] = vandq_s32(w[j], m);
        o[4 * j + 1] = vandq_s32(srl!(w[j], 8), m);
        o[4 * j + 2] = vandq_s32(srl!(w[j], 16), m);
        o[4 * j + 3] = srl!(w[j], 24);
    }
    o
}

/// The inverse of `unpack`.
#[target_feature(enable = "neon")]
#[inline]
fn pack(b: &[V; 16]) -> [V; 4] {
    let mut o = [k(0); 4];
    for j in 0..4 {
        o[j] = vorrq_s32(
            vorrq_s32(b[4 * j], vshlq_n_s32::<8>(b[4 * j + 1])),
            vorrq_s32(
                vshlq_n_s32::<16>(b[4 * j + 2]),
                vshlq_n_s32::<24>(b[4 * j + 3]),
            ),
        );
    }
    o
}

/// The fdct, see `fdct` in the scalar reference.
#[target_feature(enable = "neon")]
#[inline]
fn fdct(d: &[V; 16]) -> [V; 16] {
    let mut t = [k(0); 16];
    for j in 0..4 {
        let (d0, d1, d2, d3) = (d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
        let (a0, a1, a2, a3) = (add(d0, d3), add(d1, d2), sub(d1, d2), sub(d0, d3));
        t[4 * j] = vshlq_n_s32::<3>(add(a0, a1));
        t[4 * j + 1] = vshrq_n_s32::<9>(add(add(mulk(a2, 2217), mulk(a3, 5352)), k(1812)));
        t[4 * j + 2] = vshlq_n_s32::<3>(sub(a0, a1));
        t[4 * j + 3] = vshrq_n_s32::<9>(add(sub(mulk(a3, 2217), mulk(a2, 5352)), k(937)));
    }
    let mut o = [k(0); 16];
    for i in 0..4 {
        let (t0, t1, t2, t3) = (t[i], t[4 + i], t[8 + i], t[12 + i]);
        let (a0, a1, a2, a3) = (add(t0, t3), add(t1, t2), sub(t1, t2), sub(t0, t3));
        // +1 where a3 != 0: the compare gives -1 where it is 0.
        let one = add(k(1), eq(a3, k(0)));
        let v = vshrq_n_s32::<16>(add(add(mulk(a2, 2217), mulk(a3, 5352)), k(12000)));
        o[i] = s16(vshrq_n_s32::<4>(add(add(a0, a1), k(7))));
        o[4 + i] = s16(add(v, one));
        o[8 + i] = s16(vshrq_n_s32::<4>(add(sub(a0, a1), k(7))));
        o[12 + i] = s16(vshrq_n_s32::<16>(add(
            sub(mulk(a3, 2217), mulk(a2, 5352)),
            k(51000),
        )));
    }
    o
}

/// The quantizer of the scalar reference: zigzag levels into `lv`, the dequantized raster
/// coefficients into `dq`. Returns (last nonzero zigzag index + 1, nonzero AC count).
#[target_feature(enable = "neon")]
#[inline]
fn quantize(q: &QuantMatrix, blk: &[V; 16], lv: &mut [V; 16], dq: &mut [V; 16]) -> (V, V) {
    let (mut last1, mut nz) = (k(0), k(0));
    for n in 0..16 {
        let j = (ZIGZAG[n] & 15) as usize;
        let c = usize::from(n > 0);
        let v = blk[j];
        let s = vshrq_n_s32::<31>(v);
        let a = add(sub(veorq_s32(v, s), s), k(q.sharpen[j]));
        let l = srl!(add(mulk(a, q.iq[c] as i32), k(q.bias[c] as i32)), 17);
        let l = sub(veorq_s32(l, s), s);
        lv[n] = l;
        dq[j] = s16(mulk(l, q.q[c] as i32));
        let zero = eq(l, k(0));
        last1 = vmaxq_s32(last1, vbicq_s32(k(n as i32 + 1), zero));
        if n > 0 {
            // zero is -1 where the level is 0.
            nz = add(nz, add(k(1), zero));
        }
    }
    (last1, nz)
}

/// TTransform of 16 pixel lanes.
#[target_feature(enable = "neon")]
#[inline]
fn tt(px: &[V; 16]) -> V {
    let mut t = [k(0); 16];
    for i in 0..4 {
        let (r0, r1, r2, r3) = (px[4 * i], px[4 * i + 1], px[4 * i + 2], px[4 * i + 3]);
        let (a0, a1, a2, a3) = (add(r0, r2), add(r1, r3), sub(r1, r3), sub(r0, r2));
        t[4 * i] = add(a0, a1);
        t[4 * i + 1] = add(a3, a2);
        t[4 * i + 2] = sub(a3, a2);
        t[4 * i + 3] = sub(a0, a1);
    }
    let mut sum = k(0);
    for i in 0..4 {
        let (a0, a1) = (add(t[i], t[8 + i]), add(t[4 + i], t[12 + i]));
        let (a2, a3) = (sub(t[4 + i], t[12 + i]), sub(t[i], t[8 + i]));
        sum = add(sum, mulk(vabsq_s32(add(a0, a1)), WEIGHT_Y[i]));
        sum = add(sum, mulk(vabsq_s32(add(a3, a2)), WEIGHT_Y[4 + i]));
        sum = add(sum, mulk(vabsq_s32(sub(a3, a2)), WEIGHT_Y[8 + i]));
        sum = add(sum, mulk(vabsq_s32(sub(a0, a1)), WEIGHT_Y[12 + i]));
    }
    sum
}

/// The IDCT of the dequantized `c` added to `pred` and clipped, then its SSE against `src` and
/// its TTransform: (reconstruction, SSE, TT).
#[target_feature(enable = "neon")]
#[inline]
fn recon(c: &[V; 16], pred: &[V; 16], src: &[V; 16]) -> ([V; 16], V, V) {
    const C1: i32 = 85627; // 65536 * cos(pi/8) * sqrt(2).
    const C2: i32 = 35468; // 65536 * sin(pi/8) * sqrt(2).
    let mut m = [[k(0); 4]; 4];
    for i in 0..4 {
        let (a, b) = (add(c[i], c[i + 8]), sub(c[i], c[i + 8]));
        let cc = sub(mulsh(c[i + 4], C2), mulsh(c[i + 12], C1));
        let d = add(mulsh(c[i + 4], C1), mulsh(c[i + 12], C2));
        m[i] = [add(a, d), add(b, cc), sub(b, cc), sub(a, d)];
    }
    let (zero, m255) = (k(0), k(255));
    let mut rec = [k(0); 16];
    for j in 0..4 {
        let dc = add(m[0][j], k(4));
        let (a, b) = (add(dc, m[2][j]), sub(dc, m[2][j]));
        let cc = sub(mulsh(m[1][j], C2), mulsh(m[3][j], C1));
        let d = add(mulsh(m[1][j], C1), mulsh(m[3][j], C2));
        let o = [add(a, d), add(b, cc), sub(b, cc), sub(a, d)];
        for i in 0..4 {
            let v = add(pred[4 * j + i], vshrq_n_s32::<3>(o[i]));
            rec[4 * j + i] = vminq_s32(vmaxq_s32(v, zero), m255);
        }
    }
    let mut sse = k(0);
    for p in 0..16 {
        let e = sub(src[p], rec[p]);
        sse = add(sse, vmulq_s32(e, e));
    }
    (rec, sse, tt(&rec))
}

#[target_feature(enable = "neon")]
fn i4_trials_kernel(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    let sb = unpack(&[
        k(src[0] as i32),
        k(src[1] as i32),
        k(src[2] as i32),
        k(src[3] as i32),
    ]);
    // Modes 0 to 3, 4 to 7, then 8 and 9 in the lanes 0 and 1.
    for (g, cnt) in [(0usize, 4usize), (4, 4), (8, 2)] {
        let mut w = [[0i32; 4]; 4];
        for l in 0..cnt {
            for j in 0..4 {
                w[j][l] = pr[g + l][j] as i32;
            }
        }
        let pb = unpack(&[
            load128_s32(&w[0]),
            load128_s32(&w[1]),
            load128_s32(&w[2]),
            load128_s32(&w[3]),
        ]);
        let mut d = [k(0); 16];
        for p in 0..16 {
            d[p] = sub(sb[p], pb[p]);
        }
        let blk = fdct(&d);
        let (mut zz, mut dq) = ([k(0); 16], [k(0); 16]);
        let (last1, nz) = quantize(q, &blk, &mut zz, &mut dq);
        let (rb, sse, t) = recon(&dq, &pb, &sb);
        let rw = pack(&rb);
        // Two levels to a lane word, so a mode's 16 levels are 8 words.
        let m16 = k(0xffff);
        let mut lw = [[0i32; 4]; 8];
        for h in 0..8 {
            let v = vorrq_s32(vandq_s32(zz[2 * h], m16), vshlq_n_s32::<16>(zz[2 * h + 1]));
            store128_s32(&mut lw[h], v);
        }
        let mut rwa = [[0i32; 4]; 4];
        for j in 0..4 {
            store128_s32(&mut rwa[j], rw[j]);
        }
        let (mut la, mut sa, mut ta, mut na) = ([0i32; 4], [0i32; 4], [0i32; 4], [0i32; 4]);
        store128_s32(&mut la, sub(last1, k(1)));
        store128_s32(&mut sa, sse);
        store128_s32(&mut ta, t);
        store128_s32(&mut na, nz);
        for l in 0..cnt {
            let m = g + l;
            for h in 0..8 {
                lv[m][2 * h] = lw[h][l] as i16;
                lv[m][2 * h + 1] = (lw[h][l] >> 16) as i16;
            }
            for j in 0..4 {
                rec[m][j] = rwa[j][l] as u32;
            }
            res[m] = [la[l], sa[l], ta[l], na[l]];
        }
    }
}

#[target_feature(enable = "neon")]
fn rec_metrics16_kernel(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    let (mut sse, mut sd) = (0, 0);
    // Blocks 4g to 4g + 3, one per lane.
    for g in 0..4 {
        let mut cw = [[0i32; 4]; 16];
        let (mut pw, mut sw) = ([[0i32; 4]; 4], [[0i32; 4]; 4]);
        for l in 0..4 {
            let n = 4 * g + l;
            let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
            for p in 0..16 {
                cw[p][l] = i32::from(c[n][p]);
            }
            for j in 0..4 {
                pw[j][l] = i32::from_le_bytes(z[y + j].as_chunks::<4>().0[x / 4]);
                sw[j][l] =
                    i32::from_le_bytes(src.as_chunks::<4>().0[(y - 1 + j) * 4 + (x - 8) / 4]);
            }
        }
        let mut cv = [k(0); 16];
        for p in 0..16 {
            cv[p] = load128_s32(&cw[p]);
        }
        let pb = unpack(&[
            load128_s32(&pw[0]),
            load128_s32(&pw[1]),
            load128_s32(&pw[2]),
            load128_s32(&pw[3]),
        ]);
        let sb = unpack(&[
            load128_s32(&sw[0]),
            load128_s32(&sw[1]),
            load128_s32(&sw[2]),
            load128_s32(&sw[3]),
        ]);
        let (rb, k_sse, t) = recon(&cv, &pb, &sb);
        let rv = pack(&rb);
        let mut rwa = [[0i32; 4]; 4];
        for j in 0..4 {
            store128_s32(&mut rwa[j], rv[j]);
        }
        let (mut sa, mut ta) = ([0i32; 4], [0i32; 4]);
        store128_s32(&mut sa, k_sse);
        store128_s32(&mut ta, t);
        for l in 0..4 {
            let n = 4 * g + l;
            let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
            for j in 0..4 {
                z[y + j][x..x + 4].copy_from_slice(&rwa[j][l].to_le_bytes());
            }
            sse += sa[l];
            sd += (ta[l] - st[n]).abs() >> 5;
        }
    }
    (sse, sd)
}

#[target_feature(enable = "neon")]
fn tt_mb_kernel(src: &[u8; 256]) -> [i32; 16] {
    let mut st = [0i32; 16];
    for g in 0..4 {
        let mut sw = [[0i32; 4]; 4];
        for l in 0..4 {
            let n = 4 * g + l;
            for j in 0..4 {
                sw[j][l] =
                    i32::from_le_bytes(src.as_chunks::<4>().0[(4 * (n / 4) + j) * 4 + n % 4]);
            }
        }
        let sb = unpack(&[
            load128_s32(&sw[0]),
            load128_s32(&sw[1]),
            load128_s32(&sw[2]),
            load128_s32(&sw[3]),
        ]);
        let mut a = [0i32; 4];
        store128_s32(&mut a, tt(&sb));
        st[4 * g..4 * g + 4].copy_from_slice(&a);
    }
    st
}
