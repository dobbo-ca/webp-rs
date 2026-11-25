//! Encoder kernels of package E, the segment analysis bins (Go's `collectAsm`, `collectUVAsm`,
//! `binStatsAsm` and `binsAlpha`): scalar reference, tier dispatch and cross-tier tests.

use std::sync::OnceLock;

/// A 32 byte wide, 26 row prediction buffer, as `vp8enc::predict::Ybr`: the luma prediction is at
/// rows 1..17, columns 8..24, and the chroma ones at rows 18..26, columns 8..16 (U) and 24..32 (V).
pub(crate) type Z = [[u8; 32]; 26];

/// The largest byte and the counts of the bytes 0, 1, 2 and 3.
pub(crate) type BinStats = (i32, i32, i32, i32, i32);

/// One tier's kernels. `collect16` and `collect_uv` return the bins of all coefficients, in an
/// order that differs between tiers (the histogram does not see it).
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) collect16: fn(&[u8], usize, &Z) -> [u8; 256],
    pub(crate) collect_uv: fn(&[u8], &[u8], usize, &Z) -> [u8; 128],
    pub(crate) bin_stats: fn(&[u8]) -> BinStats,
}

pub(crate) const SCALAR: Kernels = Kernels {
    collect16: scalar_collect16,
    collect_uv: scalar_collect_uv,
    bin_stats: scalar_bin_stats,
};

/// One fdct row pass, libwebp `FTransform_C` (dsp/enc.c).
fn fdct_row(s: &[u8], p: &[u8]) -> [i32; 4] {
    let d = |i: usize| i32::from(s[i]) - i32::from(p[i]);
    let (a0, a1, a2, a3) = (d(0) + d(3), d(1) + d(2), d(1) - d(2), d(0) - d(3));
    [
        (a0 + a1) * 8,
        (a2 * 2217 + a3 * 5352 + 1812) >> 9,
        (a0 - a1) * 8,
        (a3 * 2217 - a2 * 5352 + 937) >> 9,
    ]
}

/// The 16 bins of the fdct of the 4x4 block `src` (rows `stride` apart) minus the prediction rows
/// `pred`: `CollectHistogram_C`'s bin of each coefficient, dsp/enc.c:64-84.
fn block_bins(src: &[u8], stride: usize, pred: [&[u8]; 4], out: &mut [u8]) {
    let t: [[i32; 4]; 4] = std::array::from_fn(|j| fdct_row(&src[j * stride..], pred[j]));
    for c in 0..4 {
        let (a0, a1, a2, a3) = (
            t[0][c] + t[3][c],
            t[1][c] + t[2][c],
            t[1][c] - t[2][c],
            t[0][c] - t[3][c],
        );
        let mut v = (a2 * 2217 + a3 * 5352 + 12000) >> 16;
        if a3 != 0 {
            v += 1;
        }
        let coef = [
            (a0 + a1 + 7) >> 4,
            v,
            (a0 - a1 + 7) >> 4,
            (a3 * 2217 - a2 * 5352 + 51000) >> 16,
        ];
        for (k, v) in coef.into_iter().enumerate() {
            out[4 * c + k] = (v.abs() >> 3).min(31) as u8;
        }
    }
}

fn scalar_collect16(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    let mut out = [0u8; 256];
    for (n, o) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
        let pred = std::array::from_fn(|j| &z[y + j][x..x + 4]);
        block_bins(&src[(y - 1) * stride + x - 8..], stride, pred, o);
    }
    out
}

fn scalar_collect_uv(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    let mut out = [0u8; 128];
    for (n, o) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let (src, bx) = if n >= 4 { (v, 24) } else { (u, 8) };
        let (oy, ox) = (4 * ((n & 3) >> 1), 4 * (n & 1));
        let pred = std::array::from_fn(|j| &z[18 + oy + j][bx + ox..bx + ox + 4]);
        block_bins(&src[oy * stride + ox..], stride, pred, o);
    }
    out
}

/// The largest byte of `b` and the counts of the bytes 0, 1, 2 and 3.
fn scalar_bin_stats(b: &[u8]) -> BinStats {
    let (mut last, mut c) = (0u8, [0i32; 4]);
    for &v in b {
        last = last.max(v);
        if v < 4 {
            c[usize::from(v)] += 1;
        }
    }
    (i32::from(last), c[0], c[1], c[2], c[3])
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Kernels> {
    // No AVX-512 form: its `bin_stats` was slower than AVX2's (docs/safety.md).
    super::avx2::enc_collect::detect()
}
#[cfg(target_arch = "aarch64")]
use super::neon::enc_collect::detect;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Kernels> {
    None
}

/// Every SIMD tier this CPU can run, whichever `selected` picks.
pub(crate) fn tiers() -> Vec<(&'static str, Kernels)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    v.extend(super::avx2::enc_collect::detect().map(|k| ("avx2", k)));
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_collect::detect().map(|k| ("neon", k)));
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

/// The bins of the 256 coefficients of the 16 luma blocks of `src` (16 rows `stride` apart)
/// minus the luma prediction in `z`: bin(c) = min(|c| >> 3, 31) of the fdct. `src` holds at least
/// `15 * stride + 16` bytes.
pub(crate) fn collect16(src: &[u8], stride: usize, z: &Z) -> [u8; 256] {
    (selected().collect16)(src, stride, z)
}

/// `collect16` over the 8 chroma blocks, U then V, each plane `7 * stride + 8` bytes at least.
pub(crate) fn collect_uv(u: &[u8], v: &[u8], stride: usize, z: &Z) -> [u8; 128] {
    (selected().collect_uv)(u, v, stride, z)
}

/// GetAlpha, analysis_enc.c:116-125, over a CollectHistogram bin count.
pub(crate) fn alpha_of(d: &[i32; 32]) -> i32 {
    let (mut max_v, mut last) = (0, 1);
    for (k, &v) in d.iter().enumerate() {
        if v > 0 {
            max_v = max_v.max(v);
            last = k as i32;
        }
    }
    if max_v <= 1 {
        return 0;
    }
    510 * last / max_v
}

/// Go's `binsAlpha`: `alpha_of` of the histogram of the bins `b` (each at most 31). When bins 0-3
/// hold a count at least the size of everything above bin 3, that count is the largest and the
/// histogram is not needed.
pub(crate) fn bins_alpha(b: &[u8]) -> i32 {
    bins_alpha_with(&selected(), b)
}

fn bins_alpha_with(k: &Kernels, b: &[u8]) -> i32 {
    let (last, c0, c1, c2, c3) = (k.bin_stats)(b);
    let m = c0.max(c1).max(c2).max(c3);
    if m > 1 && m >= b.len() as i32 - (c0 + c1 + c2 + c3) {
        return 510 * last / m;
    }
    // Four histograms break the chain of read-modify-writes.
    let mut h = [[0i32; 32]; 4];
    for q in b.chunks(4) {
        for (h, &v) in h.iter_mut().zip(q) {
            h[usize::from(v & 31)] += 1;
        }
    }
    let mut d = [0i32; 32];
    for (k, d) in d.iter_mut().enumerate() {
        *d = h[0][k] + h[1][k] + h[2][k] + h[3][k];
    }
    alpha_of(&d)
}

/// The fuzz target's check, one kernel picked by the next byte: every tier equals the scalar
/// reference on arbitrary bytes. The bins of a block come in a tier's own order, so the collect
/// kernels are compared by their histograms, as the encoder uses them.
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    fn hist(b: &[u8]) -> [i32; 32] {
        let mut d = [0i32; 32];
        for &v in b {
            d[usize::from(v)] += 1;
        }
        d
    }
    let tiers: Vec<Kernels> = tiers().into_iter().map(|(_, k)| k).collect();
    match r.u8() % 4 {
        0 => {
            let stride = 16 + usize::from(r.u8());
            let z = r.z();
            let src = r.bytes(15 * stride + 16);
            let want = hist(&scalar_collect16(&src, stride, &z));
            for k in &tiers {
                assert_eq!(hist(&(k.collect16)(&src, stride, &z)), want, "collect16");
            }
        }
        1 => {
            let stride = 8 + usize::from(r.u8());
            let z = r.z();
            let (u, v) = (r.bytes(7 * stride + 8), r.bytes(7 * stride + 8));
            let want = hist(&scalar_collect_uv(&u, &v, stride, &z));
            for k in &tiers {
                assert_eq!(
                    hist(&(k.collect_uv)(&u, &v, stride, &z)),
                    want,
                    "collect_uv"
                );
            }
        }
        2 => {
            let b = r.rest();
            for k in &tiers {
                assert_eq!((k.bin_stats)(b), scalar_bin_stats(b), "bin_stats");
            }
        }
        _ => {
            // Bins are at most 31.
            let b: Vec<u8> = r.rest().iter().map(|v| v & 31).collect();
            let want = alpha_of(&hist(&b));
            for k in &tiers {
                assert_eq!(bins_alpha_with(k, &b), want, "bins_alpha");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u8 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 24) as u8
    }

    /// The bin counts, the only thing the order-free outputs show.
    fn hist(b: &[u8]) -> [i32; 32] {
        let mut d = [0i32; 32];
        for &v in b {
            d[usize::from(v)] += 1;
        }
        d
    }

    /// Pixels of one of five kinds: random, 0, 255, 0 or 255, and near 128.
    fn fill(st: &mut u64, kind: usize, n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| match kind {
                0 => xorshift(st),
                1 => 0,
                2 => 255,
                3 => [0, 255][usize::from(xorshift(st) & 1)],
                _ => 128u8
                    .wrapping_add(xorshift(st) % 5)
                    .wrapping_add((i % 7) as u8),
            })
            .collect()
    }

    #[test]
    fn collect_matches_scalar() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for (name, k) in tiers() {
            for src_kind in 0..5 {
                for z_kind in 0..5 {
                    for stride in [16usize, 17, 24, 40, 100] {
                        for _ in 0..4 {
                            let src = fill(&mut st, src_kind, 15 * stride + 16 + 3);
                            let mut z: Z = [[0; 32]; 26];
                            for row in z.iter_mut() {
                                row.copy_from_slice(&fill(&mut st, z_kind, 32));
                            }
                            let (want, got) = (
                                scalar_collect16(&src, stride, &z),
                                (k.collect16)(&src, stride, &z),
                            );
                            assert_eq!(hist(&got), hist(&want), "{name} luma {src_kind} {z_kind}");
                            let u = fill(&mut st, src_kind, 7 * stride + 8 + 2);
                            let v = fill(&mut st, (src_kind + 1) % 5, 7 * stride + 8 + 2);
                            let (want, got) = (
                                scalar_collect_uv(&u, &v, stride, &z),
                                (k.collect_uv)(&u, &v, stride, &z),
                            );
                            assert_eq!(hist(&got), hist(&want), "{name} uv {src_kind} {z_kind}");
                        }
                    }
                }
            }
        }
    }

    /// A block moved to another position changes the result only where the pixels differ: each
    /// block's own bins are tested by making a single block differ from its prediction.
    #[test]
    fn collect_single_block() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for (name, k) in tiers() {
            for n in 0..16 {
                let z: Z = [[100; 32]; 26];
                let mut src = vec![100u8; 15 * 16 + 16];
                let (by, bx) = (n / 4, n % 4);
                for j in 0..4 {
                    for i in 0..4 {
                        src[(4 * by + j) * 16 + 4 * bx + i] = xorshift(&mut st);
                    }
                }
                let (want, got) = (scalar_collect16(&src, 16, &z), (k.collect16)(&src, 16, &z));
                assert_eq!(hist(&got), hist(&want), "{name} block {n}");
            }
            for n in 0..8 {
                let z: Z = [[100; 32]; 26];
                let (mut u, mut v) = (vec![100u8; 7 * 8 + 8], vec![100u8; 7 * 8 + 8]);
                let (p, by, bx) = (if n < 4 { &mut u } else { &mut v }, (n & 3) >> 1, n & 1);
                for j in 0..4 {
                    for i in 0..4 {
                        p[(4 * by + j) * 8 + 4 * bx + i] = xorshift(&mut st);
                    }
                }
                let (want, got) = (
                    scalar_collect_uv(&u, &v, 8, &z),
                    (k.collect_uv)(&u, &v, 8, &z),
                );
                assert_eq!(hist(&got), hist(&want), "{name} uv block {n}");
            }
        }
    }

    #[test]
    fn bin_stats_matches_scalar() {
        let mut st = 0x1234_5678_9abc_def1u64;
        for (name, k) in tiers() {
            for len in (0..300).chain([8160, 8161, 8192, 16_384, 70_000]) {
                for kind in 0..4 {
                    let b: Vec<u8> = (0..len)
                        .map(|_| match kind {
                            0 => xorshift(&mut st),
                            1 => xorshift(&mut st) & 3,
                            2 => 3,
                            _ => xorshift(&mut st) & 31,
                        })
                        .collect();
                    assert_eq!((k.bin_stats)(&b), scalar_bin_stats(&b), "{name} len {len}");
                }
            }
        }
    }

    /// `bins_alpha` equals `alpha_of` of the histogram, on random and adversarial bins.
    #[test]
    fn bins_alpha_matches_alpha_of() {
        let mut st = 0x0123_4567_89ab_cdefu64;
        let mut cases: Vec<Vec<u8>> = vec![vec![0; 256], vec![31; 256], vec![0; 1], vec![5; 2]];
        for len in [1usize, 2, 3, 4, 5, 9, 10, 11, 16, 128, 256] {
            for _ in 0..200 {
                let spread = 1 + usize::from(xorshift(&mut st) % 32);
                let low = usize::from(xorshift(&mut st) % 101);
                cases.push(
                    (0..len)
                        .map(|_| {
                            if usize::from(xorshift(&mut st) % 100) < low {
                                xorshift(&mut st) & 3
                            } else {
                                (usize::from(xorshift(&mut st)) % spread) as u8
                            }
                        })
                        .collect(),
                );
            }
        }
        // The fast path's edge: bins 0-3 hold exactly as many as everything above.
        for lows in 0..=64usize {
            let mut b = vec![0u8; lows];
            b.extend((0..64 - lows).map(|i| 4 + (i % 28) as u8));
            cases.push(b);
        }
        for b in &cases {
            assert_eq!(bins_alpha(b), alpha_of(&hist(b)), "{b:?}");
            for (name, k) in tiers() {
                assert_eq!(bins_alpha_with(&k, b), alpha_of(&hist(b)), "{name} {b:?}");
            }
        }
    }
}
