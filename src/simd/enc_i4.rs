//! Encoder kernels of package A, the i4 mode trials of the `match_libwebp` path: scalar reference,
//! tier dispatch and cross-tier tests.
//!
//! A trial is Go's `i4TrialGo`: the fdct of `src - pred`, the quantizer, the IDCT added to the
//! prediction, then the metrics of the reconstruction. The SIMD tiers run one mode, or one
//! block, per lane, so the arithmetic is plain 32-bit lane math with no cross-lane steps. The
//! source rows and predictions travel as four little-endian `u32`, one per row of 4 bytes.

use crate::tables::ZIGZAG;
use crate::vp8enc::quant::QuantMatrix;
use std::sync::OnceLock;

/// kWeightY, quant_enc.c:517.
pub(super) const WEIGHT_Y: [i32; 16] = [38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];

type I4Trials = fn(
    &QuantMatrix,
    &[[u32; 4]; 10],
    &[u32; 4],
    &mut [[i16; 16]; 10],
    &mut [[u32; 4]; 10],
    &mut [[i32; 4]; 10],
);
type RecMetrics16 = fn(&[[i16; 16]; 16], &mut [[u8; 32]; 26], &[u8; 256], &[i32; 16]) -> (i32, i32);
type TtMb = fn(&[u8; 256]) -> [i32; 16];

/// One tier's kernels.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) i4_trials: I4Trials,
    pub(crate) rec_metrics16: RecMetrics16,
    pub(crate) tt_mb: TtMb,
}

pub(super) const SCALAR: Kernels = Kernels {
    i4_trials: i4_trials_scalar,
    rec_metrics16: rec_metrics16_scalar,
    tt_mb: tt_mb_scalar,
};

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Kernels> {
    super::avx2::enc_i4::detect()
}
#[cfg(target_arch = "aarch64")]
fn detect() -> Option<Kernels> {
    super::neon::enc_i4::detect()
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
        v.extend(super::avx2::enc_i4::detect().map(|k| ("avx2", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_i4::detect().map(|k| ("neon", k)));
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

/// The 16 bytes of four row words.
fn bytes(w: &[u32; 4]) -> [i32; 16] {
    let mut b = [0; 16];
    for j in 0..4 {
        for i in 0..4 {
            b[4 * j + i] = ((w[j] >> (8 * i)) & 255) as i32;
        }
    }
    b
}

fn pack(b: &[i32; 16]) -> [u32; 4] {
    let mut w = [0u32; 4];
    for j in 0..4 {
        w[j] =
            (b[4 * j] | (b[4 * j + 1] << 8) | (b[4 * j + 2] << 16) | (b[4 * j + 3] << 24)) as u32;
    }
    w
}

/// The fdct of d (raster), each coefficient cut to i16 as `transform::residual` stores it.
fn fdct(d: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for j in 0..4 {
        let (d0, d1, d2, d3) = (d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
        let (a0, a1, a2, a3) = (d0 + d3, d1 + d2, d1 - d2, d0 - d3);
        t[4 * j] = (a0 + a1) * 8;
        t[4 * j + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        t[4 * j + 2] = (a0 - a1) * 8;
        t[4 * j + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }
    let mut o = [0i32; 16];
    for i in 0..4 {
        let (t0, t1, t2, t3) = (t[i], t[4 + i], t[8 + i], t[12 + i]);
        let (a0, a1, a2, a3) = (t0 + t3, t1 + t2, t1 - t2, t0 - t3);
        let mut v = (a2 * 2217 + a3 * 5352 + 12000) >> 16;
        if a3 != 0 {
            v += 1;
        }
        o[i] = i32::from(((a0 + a1 + 7) >> 4) as i16);
        o[4 + i] = i32::from(v as i16);
        o[8 + i] = i32::from(((a0 - a1 + 7) >> 4) as i16);
        o[12 + i] = i32::from(((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16);
    }
    o
}

/// `QuantMatrix::quantize_sse` on blk without the SSE: levels (zigzag) into lv, blk becomes the
/// dequantized coefficients. Returns the last nonzero zigzag index or -1, and the nonzero count
/// of the AC levels.
fn quantize(q: &QuantMatrix, blk: &mut [i32; 16], lv: &mut [i16; 16]) -> (i32, i32) {
    let level = |v: i32, sharpen: u32, iq: u32, bias: u32| -> i32 {
        let s = v >> 31;
        let a = ((v ^ s) - s) as u32 + sharpen;
        ((a.wrapping_mul(iq).wrapping_add(bias) >> 17) as i32 ^ s) - s
    };
    let (mut last, mut nz) = (-1, 0);
    for n in 0..16 {
        let j = (ZIGZAG[n] & 15) as usize;
        let c = usize::from(n > 0);
        let l = level(blk[j], q.sharpen[j] as u32, q.iq[c], q.bias[c]);
        lv[n] = l as i16;
        blk[j] = i32::from(l.wrapping_mul(q.q[c] as i32) as i16);
        if l != 0 {
            last = n as i32;
            nz += i32::from(n > 0);
        }
    }
    (last, nz)
}

/// The IDCT of the dequantized c added to pred and clipped; its SSE against src and its
/// TTransform.
fn recon(c: &[i32; 16], pred: &[i32; 16], src: &[i32; 16]) -> ([i32; 16], i32, i32) {
    const C1: i32 = 85627; // 65536 * cos(pi/8) * sqrt(2).
    const C2: i32 = 35468; // 65536 * sin(pi/8) * sqrt(2).
    let mut m = [[0i32; 4]; 4];
    for i in 0..4 {
        let a = c[i] + c[i + 8];
        let b = c[i] - c[i + 8];
        let cc = (c[i + 4].wrapping_mul(C2) >> 16) - (c[i + 12].wrapping_mul(C1) >> 16);
        let d = (c[i + 4].wrapping_mul(C1) >> 16) + (c[i + 12].wrapping_mul(C2) >> 16);
        m[i] = [a + d, b + cc, b - cc, a - d];
    }
    let mut rec = [0i32; 16];
    for j in 0..4 {
        let dc = m[0][j] + 4;
        let a = dc + m[2][j];
        let b = dc - m[2][j];
        let cc = (m[1][j].wrapping_mul(C2) >> 16) - (m[3][j].wrapping_mul(C1) >> 16);
        let d = (m[1][j].wrapping_mul(C1) >> 16) + (m[3][j].wrapping_mul(C2) >> 16);
        let o = [(a + d) >> 3, (b + cc) >> 3, (b - cc) >> 3, (a - d) >> 3];
        for i in 0..4 {
            rec[4 * j + i] = (pred[4 * j + i] + o[i]).clamp(0, 255);
        }
    }
    let mut sse = 0;
    for p in 0..16 {
        let e = src[p] - rec[p];
        sse += e * e;
    }
    (rec, sse, tt(&rec))
}

/// TTransform of 16 pixels.
fn tt(px: &[i32; 16]) -> i32 {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let r = &px[4 * i..4 * i + 4];
        let (a0, a1, a2, a3) = (r[0] + r[2], r[1] + r[3], r[1] - r[3], r[0] - r[2]);
        t[4 * i] = a0 + a1;
        t[4 * i + 1] = a3 + a2;
        t[4 * i + 2] = a3 - a2;
        t[4 * i + 3] = a0 - a1;
    }
    let mut sum = 0;
    for i in 0..4 {
        let (a0, a1) = (t[i] + t[8 + i], t[4 + i] + t[12 + i]);
        let (a2, a3) = (t[4 + i] - t[12 + i], t[i] - t[8 + i]);
        sum += WEIGHT_Y[i] * (a0 + a1).abs()
            + WEIGHT_Y[4 + i] * (a3 + a2).abs()
            + WEIGHT_Y[8 + i] * (a3 - a2).abs()
            + WEIGHT_Y[12 + i] * (a0 - a1).abs();
    }
    sum
}

/// One mode's trial, Go's `i4TrialGo`.
fn trial(
    q: &QuantMatrix,
    pr: &[u32; 4],
    src: &[u32; 4],
    lv: &mut [i16; 16],
    rec: &mut [u32; 4],
    res: &mut [i32; 4],
) {
    let (p, s) = (bytes(pr), bytes(src));
    let mut blk = [0i32; 16];
    for i in 0..16 {
        blk[i] = s[i] - p[i];
    }
    let mut blk = fdct(&blk);
    let (last, nz) = quantize(q, &mut blk, lv);
    let (r, sse, t) = recon(&blk, &p, &s);
    *rec = pack(&r);
    *res = [last, sse, t, nz];
}

fn i4_trials_scalar(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    for m in 0..10 {
        trial(q, &pr[m], src, &mut lv[m], &mut rec[m], &mut res[m]);
    }
}

fn rec_metrics16_scalar(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    let (mut sse, mut sd) = (0, 0);
    for n in 0..16 {
        let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
        let mut pw = [0u32; 4];
        let mut sw = [0u32; 4];
        for j in 0..4 {
            pw[j] = u32::from_le_bytes(z[y + j].as_chunks::<4>().0[x / 4]);
            sw[j] = u32::from_le_bytes(src.as_chunks::<4>().0[(y - 1 + j) * 4 + (x - 8) / 4]);
        }
        let mut cb = [0i32; 16];
        for p in 0..16 {
            cb[p] = i32::from(c[n][p]);
        }
        let (r, k, t) = recon(&cb, &bytes(&pw), &bytes(&sw));
        for (j, w) in pack(&r).iter().enumerate() {
            z[y + j][x..x + 4].copy_from_slice(&w.to_le_bytes());
        }
        sse += k;
        sd += (t - st[n]).abs() >> 5;
    }
    (sse, sd)
}

fn tt_mb_scalar(src: &[u8; 256]) -> [i32; 16] {
    let mut st = [0i32; 16];
    for n in 0..16 {
        let mut w = [0u32; 4];
        for j in 0..4 {
            w[j] = u32::from_le_bytes(src.as_chunks::<4>().0[(4 * (n / 4) + j) * 4 + n % 4]);
        }
        st[n] = tt(&bytes(&w));
    }
    st
}

/// Go's `i4TrialsGo`: the trials of all ten modes of the block `src`. `lv[m]` are the zigzag
/// levels, `rec[m]` the reconstruction, `res[m]` is [last nonzero zigzag index or -1, SSE of the
/// reconstruction, its TTransform, nonzero AC levels].
pub(crate) fn i4_trials(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    (selected().i4_trials)(q, pr, src, lv, rec, res);
}

/// One mode of `i4_trials`. A single trial gains nothing from the lanes, so this is scalar.
pub(crate) fn i4_trial1(
    q: &QuantMatrix,
    pr: &[u32; 4],
    src: &[u32; 4],
    lv: &mut [i16; 16],
    rec: &mut [u32; 4],
    res: &mut [i32; 4],
) {
    trial(q, pr, src, lv, rec, res);
}

/// Go's `recMetricsGo`: the reconstruction of the dequantized coefficients `c` on `pred`, its SSE
/// against `src` and its TTransform.
pub(crate) fn rec_metrics(c: &[i16; 16], pred: &[u32; 4], src: &[u32; 4]) -> ([u32; 4], i32, i32) {
    let mut cb = [0i32; 16];
    for p in 0..16 {
        cb[p] = i32::from(c[p]);
    }
    let (r, sse, t) = recon(&cb, &bytes(pred), &bytes(src));
    (pack(&r), sse, t)
}

/// Go's `recMetrics16Go`: the 16 luma blocks `c` added to the prediction in `z` (block n at
/// y = 1 + 4 * (n / 4), x = 8 + 4 * (n % 4)), written back to `z`. Returns the SSE against `src`
/// (stride 16) and the sum of |TTransform - st[n]| >> 5.
pub(crate) fn rec_metrics16(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    (selected().rec_metrics16)(c, z, src, st)
}

/// TTransform of the 16 luma blocks of `src` (stride 16).
pub(crate) fn tt_mb(src: &[u8; 256]) -> [i32; 16] {
    (selected().tt_mb)(src)
}

/// The fuzz target's check, one kernel picked by the next byte: every tier equals the scalar
/// reference on arbitrary bytes. Quantizers are the encoder's (`qi` 0..=127, the uv delta -4..=6).
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    use crate::vp8enc::quant::new_quant;
    let tiers: Vec<Kernels> = tiers().into_iter().map(|(_, k)| k).collect();
    match r.u8() % 2 {
        0 => {
            let q = new_quant(usize::from(r.u8() % 128), i32::from(r.u8() % 11) - 4);
            let m = if r.u8() & 1 == 0 { &q.y1 } else { &q.uv };
            let near = r.u8() & 1 == 1;
            let src = r.rows();
            let mut pr = [[0u32; 4]; 10];
            for p in pr.iter_mut() {
                // Near the source, so many levels are zero.
                *p = if near {
                    std::array::from_fn(|j| src[j] ^ (r.u32() & 0x0303_0303))
                } else {
                    r.rows()
                };
            }
            let mut want = ([[0i16; 16]; 10], [[0u32; 4]; 10], [[0i32; 4]; 10]);
            i4_trials_scalar(m, &pr, &src, &mut want.0, &mut want.1, &mut want.2);
            for k in &tiers {
                let mut got = ([[0i16; 16]; 10], [[0u32; 4]; 10], [[0i32; 4]; 10]);
                (k.i4_trials)(m, &pr, &src, &mut got.0, &mut got.1, &mut got.2);
                assert!(got == want, "i4_trials");
            }
        }
        _ => {
            let z = r.z();
            let mut src = [0u8; 256];
            r.fill(&mut src);
            let st = tt_mb_scalar(&src);
            let c: [[i16; 16]; 16] = std::array::from_fn(|_| r.coeffs());
            let mut want_z = z;
            let want = rec_metrics16_scalar(&c, &mut want_z, &src, &st);
            for k in &tiers {
                let mut got_z = z;
                assert_eq!(
                    (k.rec_metrics16)(&c, &mut got_z, &src, &st),
                    want,
                    "rec_metrics16"
                );
                assert!(got_z == want_z, "rec_metrics16 z");
                assert_eq!((k.tt_mb)(&src), st, "tt_mb");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp8enc::quant::new_quant;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
        /// A row word of 4 bytes of the given kind: 0 random, 1 near `base`, 2 extreme.
        fn word(&mut self, kind: u32, base: u32) -> u32 {
            match kind {
                0 => self.next(),
                1 => {
                    let mut w = 0u32;
                    for i in 0..4 {
                        let b = ((base >> (8 * i)) & 255) as i32 + (self.next() % 7) as i32 - 3;
                        w |= (b.clamp(0, 255) as u32) << (8 * i);
                    }
                    w
                }
                _ => {
                    let mut w = 0u32;
                    for i in 0..4 {
                        if self.next() & 1 == 1 {
                            w |= 255 << (8 * i);
                        }
                    }
                    w
                }
            }
        }
    }

    /// Blocks of src and ten predictions: random, close to src (so many levels are zero), and
    /// 0/255 extremes.
    fn inputs(rng: &mut Rng, kind: u32) -> ([u32; 4], [[u32; 4]; 10]) {
        let mut src = [0u32; 4];
        let mut pr = [[0u32; 4]; 10];
        for j in 0..4 {
            src[j] = match kind {
                2 => rng.word(2, 0),
                // Checkerboard against its inverse: the largest coefficients, the 32-bit IDCT path.
                3 => 0xff00_ff00 >> (8 * (j as u32 & 1)),
                _ => rng.word(0, 0),
            };
            for m in 0..10 {
                pr[m][j] = match kind {
                    0 => rng.word(0, 0),
                    1 => rng.word(1, src[j]),
                    2 => rng.word(2, 0),
                    _ => !src[j],
                };
            }
        }
        (src, pr)
    }

    #[test]
    fn i4_trials_match() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let tiers = tiers();
        for qi in 0..=127 {
            for d in -4..=6 {
                let q = new_quant(qi, d);
                for m in [&q.y1, &q.uv] {
                    for kind in [0, 1, 2, 1, 3] {
                        let (src, pr) = inputs(&mut rng, kind);
                        let mut want = ([[0i16; 16]; 10], [[0u32; 4]; 10], [[0i32; 4]; 10]);
                        i4_trials_scalar(m, &pr, &src, &mut want.0, &mut want.1, &mut want.2);
                        for (name, k) in &tiers {
                            // Garbage the kernel has to overwrite.
                            let mut got = (
                                [[0x5a5ai16; 16]; 10],
                                [[0x5a5a_5a5au32; 4]; 10],
                                [[0x5a5a_5a5ai32; 4]; 10],
                            );
                            (k.i4_trials)(m, &pr, &src, &mut got.0, &mut got.1, &mut got.2);
                            assert!(got == want, "{name} qi {qi} d {d} kind {kind}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rec_metrics_16_and_tt_match() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let tiers = tiers();
        for round in 0..300 {
            let range = [1, 8, 200, 2300][round % 4];
            let mut c = [[0i16; 16]; 16];
            for b in c.iter_mut() {
                for v in b.iter_mut() {
                    let r = (rng.next() % (2 * range + 1)) as i32 - range as i32;
                    // Many zero blocks and DC-only blocks.
                    *v = if rng.next().is_multiple_of(3) {
                        0
                    } else {
                        r as i16
                    };
                }
                if rng.next().is_multiple_of(4) {
                    b[1..].fill(0);
                }
                if rng.next().is_multiple_of(8) {
                    *b = [0; 16];
                }
            }
            let mut z = [[0u8; 32]; 26];
            for row in z.iter_mut() {
                for v in row.iter_mut() {
                    *v = match round % 3 {
                        0 => rng.next() as u8,
                        1 => 255 * (rng.next() as u8 & 1),
                        _ => 128 + (rng.next() % 5) as u8,
                    };
                }
            }
            let mut src = [0u8; 256];
            for v in src.iter_mut() {
                *v = if round % 3 == 1 {
                    255 * (rng.next() as u8 & 1)
                } else {
                    rng.next() as u8
                };
            }
            let st = tt_mb_scalar(&src);
            let (mut want_z, mut got_z) = (z, z);
            let want = rec_metrics16_scalar(&c, &mut want_z, &src, &st);
            // The scalar reference against the blockwise one.
            for n in 0..16 {
                let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
                let w: [u32; 4] = std::array::from_fn(|j| {
                    u32::from_le_bytes(z[y + j][x..x + 4].try_into().unwrap())
                });
                let s: [u32; 4] = std::array::from_fn(|j| {
                    let o = (y - 1 + j) * 16 + x - 8;
                    u32::from_le_bytes(src[o..o + 4].try_into().unwrap())
                });
                let (r, _, _) = rec_metrics(&c[n], &w, &s);
                for j in 0..4 {
                    assert_eq!(want_z[y + j][x..x + 4], r[j].to_le_bytes(), "rec_metrics");
                }
            }
            for (name, k) in &tiers {
                got_z.clone_from(&z);
                assert_eq!((k.rec_metrics16)(&c, &mut got_z, &src, &st), want, "{name}");
                assert!(got_z == want_z, "{name} z round {round}");
                assert_eq!((k.tt_mb)(&src), st, "{name} tt_mb");
            }
        }
    }
}
