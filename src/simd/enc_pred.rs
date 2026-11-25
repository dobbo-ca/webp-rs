//! Encoder kernels of package B, the intra predictors and SSE: scalar reference, tier dispatch and
//! cross-tier tests. Ports of Go's `sse4x10`, `edgeSSE` and `predSSE` (internal/vp8enc).

use crate::vp8enc::predict;
use crate::vp8enc::quant::QuantMatrix;
use std::sync::OnceLock;

/// The encoder's prediction workspace, `vp8enc::predict::Ybr`.
type Z = [[u8; 32]; 26];

type EdgeSse = fn(&[u8], usize, usize, &Z, usize, usize, i32) -> [i32; 4];
type SseBlock = fn(&[u8], usize, &Z, usize, usize, usize, usize) -> i32;
type ResidualQuant3 = fn(
    &QuantMatrix,
    &[[u32; 4]; 10],
    [usize; 3],
    &[u32; 4],
    &mut [[i16; 16]; 3],
    &mut [[i16; 16]; 3],
) -> [(i32, i64); 3];

/// The first N bytes of `b` as an array. The wrappers below checked that `b` is long enough, so
/// the zeros are never returned.
pub(super) fn row<const N: usize>(b: &[u8]) -> &[u8; N] {
    b.first_chunk().unwrap_or(&[0; N])
}

/// One tier's kernels. The wrappers below validate every offset and length before calling them,
/// so a kernel may index `z` and `src` without checks of its own.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(super) sse4x10: fn(&[[u32; 4]; 10], &[u32; 4]) -> [i32; 10],
    /// `src`, `stride`, `n`, `z`, `y`, `x`, `dc`.
    pub(super) edge_sse: EdgeSse,
    /// `src`, `stride`, `z`, `y`, `x`, `w`, `h`.
    pub(super) sse_block: SseBlock,
    /// `m`, the ten predictions, three mode numbers below 10, the source rows, the coefficients
    /// out, the levels out.
    pub(super) residual_quant3: ResidualQuant3,
    /// The edge bytes of `pred4_all`, the ten predictions out.
    pub(super) pred4_all: fn(&[u8; 16], &mut [[u32; 4]; 10]),
}

pub(super) const SCALAR: Kernels = Kernels {
    sse4x10: sse4x10_scalar,
    edge_sse: edge_sse_scalar,
    sse_block: sse_block_scalar,
    residual_quant3: residual_quant3_scalar,
    pred4_all: pred4_all_scalar,
};

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Kernels> {
    super::avx512_wanted()
        .then(super::avx512::enc_pred::detect)
        .flatten()
        .or_else(super::avx2::enc_pred::detect)
}
#[cfg(target_arch = "aarch64")]
fn detect() -> Option<Kernels> {
    super::neon::enc_pred::detect()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Kernels> {
    None
}

/// Every SIMD tier this CPU can run, whichever `selected` picks.
pub(crate) fn tiers() -> Vec<(&'static str, Kernels)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.extend(super::avx2::enc_pred::detect().map(|k| ("avx2", k)));
        v.extend(super::avx512::enc_pred::detect().map(|k| ("avx512", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_pred::detect().map(|k| ("neon", k)));
    v
}

fn selected() -> Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    *K.get_or_init(|| {
        if super::scalar_forced() {
            SCALAR
        } else {
            detect().unwrap_or(SCALAR)
        }
    })
}

/// Go's `sse4x10Go`: the SSE of the 4x4 source block, four rows of four bytes as little-endian
/// words, against each of the ten predictions.
pub(crate) fn sse4x10(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    (selected().sse4x10)(pr, src)
}

/// Go's `residualQuant3`: for each k, the fdct of `src` minus prediction `modes[k]` quantized with
/// `m`. `blk[k]` becomes the dequantized raster coefficients and `lv[k]` the zigzag levels; the
/// result is the last nonzero index (or -1) and the raw coefficient SSE.
pub(crate) fn residual_quant3(
    m: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    modes: [usize; 3],
    src: &[u32; 4],
    blk: &mut [[i16; 16]; 3],
    lv: &mut [[i16; 16]; 3],
) -> [(i32, i64); 3] {
    (selected().residual_quant3)(m, pr, modes.map(|m| m.min(9)), src, blk, lv)
}

/// All ten 4x4 predictions as little-endian rows, Go's `pred4All`. `e` is the edge: byte 0 the
/// top-left pixel, 1..9 the eight pixels above (z[y-1][x..x+8]), 9..13 the four to the left, the
/// rest unused.
pub(crate) fn pred4_all(e: &[u8; 16], pr: &mut [[u32; 4]; 10]) {
    (selected().pred4_all)(e, pr);
}

/// The DC prediction value of an n by n block at (y, x), Go's `dcAvg`. `edges` has bit 0 set when
/// the left edge exists and bit 1 when the top one does.
fn dc_value(z: &Z, n: usize, y: usize, x: usize, edges: u8) -> i32 {
    let log2n = if n == 16 { 4 } else { 3 };
    let top: usize = z[y - 1][x..x + n].iter().map(|&v| usize::from(v)).sum();
    let left: usize = (0..n).map(|j| usize::from(z[y + j][x - 1])).sum();
    (match edges & 3 {
        0 => 128,
        1 => (left + n / 2) >> log2n,
        2 => (top + n / 2) >> log2n,
        _ => (top + left + n) >> (log2n + 1),
    }) as i32
}

/// The SSE of the DC, TM, VE and HE predictions of the n by n block at (y, x) of `z` against
/// `src` (rows `stride` apart), n 8 or 16, Go's `edgeSSE`. `edges` as for `dc_value`.
pub(crate) fn edge_sse(
    src: &[u8],
    stride: usize,
    n: usize,
    z: &Z,
    y: usize,
    x: usize,
    edges: u8,
) -> [i32; 4] {
    assert!(n == 8 || n == 16);
    assert!(y >= 1 && x >= 1 && y + n <= 26 && x + n <= 32);
    assert!(src.len() >= (n - 1) * stride + n);
    let dc = dc_value(z, n, y, x, edges);
    (selected().edge_sse)(src, stride, n, z, y, x, dc)
}

/// The SSE between `src` and the w by h block at (y, x) of `z`, w 8 or 16, Go's `predSSE`.
pub(crate) fn sse_block(
    src: &[u8],
    stride: usize,
    z: &Z,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
) -> i32 {
    assert!(w == 8 || w == 16);
    assert!(h >= 1 && y + h <= 26 && x + w <= 32);
    assert!(src.len() >= (h - 1) * stride + w);
    (selected().sse_block)(src, stride, z, y, x, w, h)
}

fn sse4x10_scalar(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    let mut d = [0i32; 10];
    for (d, p) in d.iter_mut().zip(pr) {
        for (&r, &w) in src.iter().zip(p) {
            for k in 0..4 {
                let v = i32::from((r >> (8 * k)) as u8) - i32::from((w >> (8 * k)) as u8);
                *d += v * v;
            }
        }
    }
    d
}

fn clip8(v: i32) -> i32 {
    v.clamp(0, 255)
}

fn edge_sse_scalar(
    src: &[u8],
    stride: usize,
    n: usize,
    z: &Z,
    y: usize,
    x: usize,
    dc: i32,
) -> [i32; 4] {
    let a = i32::from(z[y - 1][x - 1]);
    let top = &z[y - 1][x..x + n];
    let mut d = [0i32; 4];
    for j in 0..n {
        let l = i32::from(z[y + j][x - 1]);
        for (i, &v) in src[j * stride..j * stride + n].iter().enumerate() {
            let s = i32::from(v);
            let t = i32::from(top[i]);
            let (k0, k1) = (s - dc, s - clip8(l - a + t));
            let (k2, k3) = (s - t, s - l);
            d[0] += k0 * k0;
            d[1] += k1 * k1;
            d[2] += k2 * k2;
            d[3] += k3 * k3;
        }
    }
    d
}

fn sse_block_scalar(
    src: &[u8],
    stride: usize,
    z: &Z,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut d = 0;
    for j in 0..h {
        for (&s, &p) in src[j * stride..j * stride + w]
            .iter()
            .zip(&z[y + j][x..x + w])
        {
            let k = i32::from(s) - i32::from(p);
            d += k * k;
        }
    }
    d
}

pub(super) fn residual_quant3_scalar(
    m: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    modes: [usize; 3],
    src: &[u32; 4],
    blk: &mut [[i16; 16]; 3],
    lv: &mut [[i16; 16]; 3],
) -> [(i32, i64); 3] {
    std::array::from_fn(|k| {
        super::enc_block::fdct4(src, &pr[modes[k]], &mut blk[k]);
        m.quantize_sse(&mut blk[k], &mut lv[k])
    })
}

fn avg2(a: i32, b: i32) -> u32 {
    (a + b + 1) as u32 >> 1
}

fn avg3(a: i32, b: i32, c: i32) -> u32 {
    (a + 2 * b + c + 2) as u32 >> 2
}

fn word(a: u32, b: u32, c: u32, d: u32) -> u32 {
    a | (b << 8) | (c << 16) | (d << 24)
}

/// The mode numbers of `vp8enc::predict`, in `pr` order.
pub(super) const DC: usize = predict::PRED_DC as usize;
pub(super) const TM: usize = predict::PRED_TM as usize;
pub(super) const VE: usize = predict::PRED_VE as usize;
pub(super) const HE: usize = predict::PRED_HE as usize;
pub(super) const RD: usize = predict::PRED_RD as usize;
pub(super) const VR: usize = predict::PRED_VR as usize;
pub(super) const LD: usize = predict::PRED_LD as usize;
pub(super) const VL: usize = predict::PRED_VL as usize;
pub(super) const HD: usize = predict::PRED_HD as usize;
pub(super) const HU: usize = predict::PRED_HU as usize;

pub(super) fn pred4_all_scalar(e: &[u8; 16], pr: &mut [[u32; 4]; 10]) {
    let g = |i: usize| i32::from(e[i]);
    let a = g(0);
    let (t0, t1, t2, t3) = (g(1), g(2), g(3), g(4));
    let (t4, t5, t6, t7) = (g(5), g(6), g(7), g(8));
    let (l0, l1, l2, l3) = (g(9), g(10), g(11), g(12));
    const B: u32 = 0x01010101;
    let clip8 = |v: i32| v.clamp(0, 255) as u32;

    let dc = ((t0 + t1 + t2 + t3 + l0 + l1 + l2 + l3 + 4) as u32 >> 3) * B;
    pr[DC] = [dc; 4];
    for (j, l) in [l0, l1, l2, l3].into_iter().enumerate() {
        let d = l - a;
        pr[TM][j] = word(clip8(d + t0), clip8(d + t1), clip8(d + t2), clip8(d + t3));
    }
    let ve = word(
        avg3(a, t0, t1),
        avg3(t0, t1, t2),
        avg3(t1, t2, t3),
        avg3(t2, t3, t4),
    );
    pr[VE] = [ve; 4];
    pr[HE] = [
        avg3(a, l0, l1) * B,
        avg3(l0, l1, l2) * B,
        avg3(l1, l2, l3) * B,
        avg3(l2, l3, l3) * B,
    ];

    // Left and top-left three-tap filters.
    let (srq, rqp, qpa, pab) = (
        avg3(l3, l2, l1),
        avg3(l2, l1, l0),
        avg3(l1, l0, a),
        avg3(l0, a, t0),
    );
    // Top three-tap filters, abc centred on T0.
    let (abc, bcd, cde) = (avg3(a, t0, t1), avg3(t0, t1, t2), avg3(t1, t2, t3));
    pr[RD] = [
        word(pab, abc, bcd, cde),
        word(qpa, pab, abc, bcd),
        word(rqp, qpa, pab, abc),
        word(srq, rqp, qpa, pab),
    ];

    let (ab, bc, cd, de) = (avg2(a, t0), avg2(t0, t1), avg2(t1, t2), avg2(t2, t3));
    pr[VR] = [
        word(ab, bc, cd, de),
        word(pab, abc, bcd, cde),
        word(qpa, ab, bc, cd),
        word(rqp, pab, abc, bcd),
    ];

    // LD and VL filter only the top row, t0 centred on T1.
    let (u0, u1, u2, u3) = (
        avg3(t0, t1, t2),
        avg3(t1, t2, t3),
        avg3(t2, t3, t4),
        avg3(t3, t4, t5),
    );
    let (u4, u5, u6) = (avg3(t4, t5, t6), avg3(t5, t6, t7), avg3(t6, t7, t7));
    pr[LD] = [
        word(u0, u1, u2, u3),
        word(u1, u2, u3, u4),
        word(u2, u3, u4, u5),
        word(u3, u4, u5, u6),
    ];
    let (v0, v1, v2, v3) = (avg2(t0, t1), avg2(t1, t2), avg2(t2, t3), avg2(t3, t4));
    pr[VL] = [
        word(v0, v1, v2, v3),
        word(u0, u1, u2, u3),
        word(v1, v2, v3, u4),
        word(u1, u2, u3, u5),
    ];

    let (sr, rq, qp, pa) = (avg2(l3, l2), avg2(l2, l1), avg2(l1, l0), avg2(l0, a));
    pr[HD] = [
        word(pa, pab, abc, bcd),
        word(qp, qpa, pa, pab),
        word(rq, rqp, qp, qpa),
        word(sr, srq, rq, rqp),
    ];

    let (pq, qr, rs) = (avg2(l0, l1), avg2(l1, l2), avg2(l2, l3));
    let (pqr, qrs, rss, s) = (
        avg3(l0, l1, l2),
        avg3(l1, l2, l3),
        avg3(l2, l3, l3),
        l3 as u32,
    );
    pr[HU] = [
        word(pq, pqr, qr, qrs),
        word(qr, qrs, rs, rss),
        word(rs, rss, s, s),
        s * B,
    ];
}

/// The fuzz target's check, one kernel picked by the next byte: every tier equals the scalar
/// reference on arbitrary bytes, with the offsets the wrappers accept.
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    let tiers: Vec<Kernels> = tiers().into_iter().map(|(_, k)| k).collect();
    match r.u8() % 5 {
        0 => {
            let src = r.rows();
            let pr: [[u32; 4]; 10] = std::array::from_fn(|_| r.rows());
            for k in &tiers {
                assert_eq!((k.sse4x10)(&pr, &src), sse4x10_scalar(&pr, &src), "sse4x10");
            }
        }
        1 => {
            let n = if r.u8() & 1 == 0 { 8 } else { 16 };
            let (y, x) = (
                1 + usize::from(r.u8()) % (26 - n),
                1 + usize::from(r.u8()) % (32 - n),
            );
            let stride = n + usize::from(r.u8() % 40);
            let edges = r.u8() & 3;
            let z = r.z();
            let src = r.bytes((n - 1) * stride + n);
            let dc = dc_value(&z, n, y, x, edges);
            let want = edge_sse_scalar(&src, stride, n, &z, y, x, dc);
            for k in &tiers {
                let got = (k.edge_sse)(&src, stride, n, &z, y, x, dc);
                assert_eq!(got, want, "edge_sse n {n} y {y} x {x} stride {stride}");
            }
        }
        3 => {
            let q = crate::vp8enc::quant::new_quant(usize::from(r.u8() % 128), 0);
            let src = r.rows();
            let pr: [[u32; 4]; 10] = std::array::from_fn(|_| r.rows());
            let modes: [usize; 3] = std::array::from_fn(|_| usize::from(r.u8() % 10));
            let (mut wb, mut wl) = ([[1i16; 16]; 3], [[2i16; 16]; 3]);
            let want = residual_quant3_scalar(&q.y1, &pr, modes, &src, &mut wb, &mut wl);
            for k in &tiers {
                let (mut gb, mut gl) = ([[3i16; 16]; 3], [[4i16; 16]; 3]);
                let got = (k.residual_quant3)(&q.y1, &pr, modes, &src, &mut gb, &mut gl);
                assert_eq!((got, gb, gl), (want, wb, wl), "residual_quant3");
            }
        }
        4 => {
            let mut e = [0u8; 16];
            r.fill(&mut e);
            let mut want = [[0u32; 4]; 10];
            pred4_all_scalar(&e, &mut want);
            for k in &tiers {
                let mut got = [[1u32; 4]; 10];
                (k.pred4_all)(&e, &mut got);
                assert_eq!(got, want, "pred4_all");
            }
        }
        _ => {
            let w = if r.u8() & 1 == 0 { 8 } else { 16 };
            let h = 1 + usize::from(r.u8()) % 25;
            let y = 1 + usize::from(r.u8()) % (26 - h);
            let x = usize::from(r.u8()) % (32 - w + 1);
            let stride = w + usize::from(r.u8() % 40);
            let z = r.z();
            let src = r.bytes((h - 1) * stride + w);
            let want = sse_block_scalar(&src, stride, &z, y, x, w, h);
            for k in &tiers {
                let got = (k.sse_block)(&src, stride, &z, y, x, w, h);
                assert_eq!(
                    got, want,
                    "sse_block w {w} h {h} y {y} x {x} stride {stride}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u8 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 24) as u8
    }

    /// Fills with xorshift bytes, all 0, all 255, or alternating 0 and 255.
    fn fill(kind: usize, st: &mut u64, n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| match kind {
                0 => xorshift(st),
                1 => 0,
                2 => 255,
                _ => [0, 255][i & 1],
            })
            .collect()
    }

    fn workspace(kind: usize, st: &mut u64) -> Z {
        let mut z = [[0u8; 32]; 26];
        for row in z.iter_mut() {
            row.copy_from_slice(&fill(kind, st, 32));
        }
        z
    }

    #[test]
    fn simd_matches_scalar() {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(!tiers().is_empty(), "a SIMD tier expected");
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for (name, k) in tiers() {
            for kz in 0..4 {
                for ks in 0..4 {
                    let z = workspace(kz, &mut st);
                    let src = fill(ks, &mut st, 16 * 40 + 16);
                    let mut pr = [[0u32; 4]; 10];
                    for p in pr.iter_mut().flatten() {
                        *p = u32::from_le_bytes(std::array::from_fn(|_| match kz {
                            0 => xorshift(&mut st),
                            1 => 0,
                            _ => 255,
                        }));
                    }
                    let s4: [u32; 4] = std::array::from_fn(|i| {
                        u32::from_le_bytes(std::array::from_fn(|j| src[16 * i + j]))
                    });
                    assert_eq!(
                        (k.sse4x10)(&pr, &s4),
                        sse4x10_scalar(&pr, &s4),
                        "{name} sse4x10 {kz} {ks}"
                    );
                    for (n, y, x) in [(16, 1, 8), (8, 18, 8), (8, 18, 24), (8, 1, 1), (16, 9, 16)] {
                        for stride in [n, 16, 33] {
                            for edges in 0..4 {
                                let dc = dc_value(&z, n, y, x, edges);
                                assert_eq!(
                                    (k.edge_sse)(&src, stride, n, &z, y, x, dc),
                                    edge_sse_scalar(&src, stride, n, &z, y, x, dc),
                                    "{name} edge n {n} y {y} x {x} stride {stride} e {edges}"
                                );
                            }
                        }
                    }
                    for (w, h, y, x) in [
                        (8, 8, 18, 8),
                        (8, 8, 18, 24),
                        (16, 16, 1, 8),
                        (16, 3, 5, 16),
                    ] {
                        assert_eq!(
                            (k.sse_block)(&src, 40, &z, y, x, w, h),
                            sse_block_scalar(&src, 40, &z, y, x, w, h),
                            "{name} block w {w} h {h}"
                        );
                    }
                }
            }
        }
    }

    /// The public wrappers against the scalar reference, whichever tier `selected` picks.
    #[test]
    fn wrappers_match_scalar() {
        let mut st = 0x1234_5678_9abc_def1u64;
        let z = workspace(0, &mut st);
        let src = fill(0, &mut st, 16 * 16);
        for edges in 0..4 {
            let dc = dc_value(&z, 16, 1, 8, edges);
            assert_eq!(
                edge_sse(&src, 16, 16, &z, 1, 8, edges),
                edge_sse_scalar(&src, 16, 16, &z, 1, 8, dc)
            );
        }
        assert_eq!(
            sse_block(&src, 8, &z, 18, 8, 8, 8),
            sse_block_scalar(&src, 8, &z, 18, 8, 8, 8)
        );
    }

    fn quant_at(qi: usize) -> QuantMatrix {
        crate::vp8enc::quant::new_quant(qi, 0).y1
    }

    fn mk_rows(kind: usize, st: &mut u64) -> [u32; 4] {
        std::array::from_fn(|i| {
            u32::from_le_bytes(std::array::from_fn(|j| match kind {
                0 | 4 => xorshift(st),
                1 => 0,
                2 => 255,
                _ => [0, 255][(i + j) & 1],
            }))
        })
    }

    #[test]
    fn residual_quant3_matches_scalar() {
        let mut st = 0x0dd5_eed5_1ce4_b0b1u64;
        let mut subsets = vec![];
        for a in 0..10 {
            for b in a + 1..10 {
                for c in b + 1..10 {
                    subsets.push([a, b, c]);
                }
            }
        }
        assert_eq!(subsets.len(), 120);
        for (name, k) in tiers() {
            for qi in 0..=127 {
                let m = quant_at(qi);
                for kind in 0..6usize {
                    let src = mk_rows(kind, &mut st);
                    let pr: [[u32; 4]; 10] = std::array::from_fn(|_| mk_rows(kind, &mut st));
                    let pr = if kind == 4 { [src; 10] } else { pr };
                    let r = |st: &mut u64| usize::from(xorshift(st)) % 10;
                    let random = [r(&mut st), r(&mut st), r(&mut st)];
                    for modes in [subsets[(qi * 6 + kind) % 120], random] {
                        let (mut wb, mut wl) = ([[0x1111i16; 16]; 3], [[0x2222i16; 16]; 3]);
                        let want = residual_quant3_scalar(&m, &pr, modes, &src, &mut wb, &mut wl);
                        let (mut gb, mut gl) = ([[0x3333i16; 16]; 3], [[0x4444i16; 16]; 3]);
                        let got = (k.residual_quant3)(&m, &pr, modes, &src, &mut gb, &mut gl);
                        assert_eq!(
                            (got, gb, gl),
                            (want, wb, wl),
                            "{name} qi {qi} kind {kind} modes {modes:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pred4_all_matches_scalar() {
        let mut st = 0x7e57_0000_cafe_f00du64;
        let mut edges: Vec<[u8; 16]> =
            vec![[0; 16], [255; 16], std::array::from_fn(|i| 17 * i as u8)];
        edges.push(std::array::from_fn(|i| 255 - 17 * i as u8));
        edges.push(std::array::from_fn(|i| [0, 255][i & 1]));
        edges.push(std::array::from_fn(|i| [255, 0][i & 1]));
        edges.extend((0..4000).map(|_| std::array::from_fn(|_| xorshift(&mut st))));
        let mut want = [[0u32; 4]; 10];
        for (name, k) in tiers() {
            for e in &edges {
                pred4_all_scalar(e, &mut want);
                let mut got = [[0xdead_beefu32; 4]; 10];
                (k.pred4_all)(e, &mut got);
                assert_eq!(got, want, "{name} {e:?}");
            }
        }
    }

    /// The wrapper clamps the mode numbers.
    #[test]
    fn residual_quant3_wrapper() {
        let m = quant_at(40);
        let src = [0x0102_0304u32; 4];
        let pr: [[u32; 4]; 10] = std::array::from_fn(|i| [0x0101_0101 * i as u32; 4]);
        let (mut wb, mut wl) = ([[0i16; 16]; 3], [[0i16; 16]; 3]);
        let want = residual_quant3_scalar(&m, &pr, [9, 9, 3], &src, &mut wb, &mut wl);
        let (mut gb, mut gl) = ([[0i16; 16]; 3], [[0i16; 16]; 3]);
        let got = residual_quant3(&m, &pr, [200, 10, 3], &src, &mut gb, &mut gl);
        assert_eq!((got, gb, gl), (want, wb, wl));
    }

    /// `dc_value` against Go's `dcAvg` cases.
    #[test]
    fn dc_cases() {
        let mut z = [[0u8; 32]; 26];
        z[0][8..24].fill(10);
        for row in z.iter_mut().skip(1).take(16) {
            row[7] = 30;
        }
        assert_eq!(dc_value(&z, 16, 1, 8, 0), 128);
        assert_eq!(dc_value(&z, 16, 1, 8, 1), 30);
        assert_eq!(dc_value(&z, 16, 1, 8, 2), 10);
        assert_eq!(dc_value(&z, 16, 1, 8, 3), 20);
    }
}
