//! Encoder kernels of the block transform and quantizer (Go's `quantAsm`, `residualAsm`,
//! `idctAddAsm` and their batch forms): the scalar reference, tier dispatch, safe wrappers and
//! cross-tier tests.
//!
//! A block's four rows travel as four little-endian `u32`s (`Rows`), as in `pred4_all`. The batch
//! forms take the prediction workspace `Ybr`: luma block `n` is at `y = 1 + 4 * (n / 4)`,
//! `x = 8 + 4 * (n % 4)`; chroma block `c` at `y = 18 + 4 * ((c & 3) >> 1)`,
//! `x = 8 + 16 * (c >> 2) + 4 * (c & 1)`.

use crate::tables::ZIGZAG;
use crate::vp8enc::predict::{Ybr, clip8};
use crate::vp8enc::quant::{QuantMatrix, level_of};
use std::sync::OnceLock;

pub(crate) type Rows = [u32; 4];

type QuantSseFn = fn(&QuantMatrix, &mut [i16; 16], &mut [i16; 16]) -> (i32, i64);
type QuantNFn = fn(&QuantMatrix, &mut [[i16; 16]], &mut [[i16; 16]]) -> u32;
type ResidualsUvFn = fn(&Ybr, &[u8; 64], &[u8; 64], &mut [[i16; 16]; 8]);

/// One tier's kernels.
#[derive(Clone, Copy)]
pub(crate) struct Block {
    pub(super) fdct4: fn(&Rows, &Rows, &mut [i16; 16]),
    pub(super) quantize_sse: QuantSseFn,
    pub(super) quantize_n: QuantNFn,
    pub(super) idct_add4: fn(&[i16; 16], &mut Rows),
    pub(super) residuals_y: fn(&Ybr, &[u8; 256], &mut [[i16; 16]; 16]),
    pub(super) residuals_uv: ResidualsUvFn,
    pub(super) idct_adds_y: fn(&mut Ybr, &[[i16; 16]; 16]),
    pub(super) idct_adds_uv: fn(&mut Ybr, &[[i16; 16]; 8]),
}

pub(super) const SCALAR: Block = Block {
    fdct4: ref_fdct4,
    quantize_sse: ref_quantize_sse,
    quantize_n: ref_quantize_n,
    idct_add4: ref_idct_add4,
    residuals_y: ref_residuals_y,
    residuals_uv: ref_residuals_uv,
    idct_adds_y: ref_idct_adds_y,
    idct_adds_uv: ref_idct_adds_uv,
};

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Block> {
    super::avx512_wanted()
        .then(super::avx512::enc_block::detect)
        .flatten()
        .or_else(super::avx2::enc_block::detect)
}
#[cfg(target_arch = "aarch64")]
fn detect() -> Option<Block> {
    super::neon::enc_block::detect()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Block> {
    None
}

/// Every SIMD tier this CPU can run, whichever `selected` picks.
pub(crate) fn tiers() -> Vec<(&'static str, Block)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.extend(super::avx2::enc_block::detect().map(|k| ("avx2", k)));
        v.extend(super::avx512::enc_block::detect().map(|k| ("avx512", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_block::detect().map(|k| ("neon", k)));
    v
}

fn selected() -> Block {
    static K: OnceLock<Block> = OnceLock::new();
    *K.get_or_init(|| {
        if super::scalar_forced() {
            SCALAR
        } else {
            detect().unwrap_or(SCALAR)
        }
    })
}

/// The four rows of a 4x4 block at `off` of a plane with `stride`.
pub(crate) fn rows_at(p: &[u8], stride: usize, off: usize) -> Rows {
    [0, 1, 2, 3].map(|j| {
        let r = &p[off + j * stride..off + j * stride + 4];
        u32::from_le_bytes([r[0], r[1], r[2], r[3]])
    })
}

/// The 4x4 block of `z` at (y, x).
pub(crate) fn get_rows(z: &Ybr, y: usize, x: usize) -> Rows {
    [0, 1, 2, 3].map(|j| {
        let r = &z[y + j][x..x + 4];
        u32::from_le_bytes([r[0], r[1], r[2], r[3]])
    })
}

pub(super) fn put_rows(z: &mut Ybr, p: &Rows, y: usize, x: usize) {
    for (j, v) in p.iter().enumerate() {
        z[y + j][x..x + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// The fdct of `src` minus `pred`, raster order out.
pub(crate) fn fdct4(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    (selected().fdct4)(src, pred, out);
}

/// QuantizeBlock_C minus zthresh and MAX_LEVEL: `in_` raster in, dequantized in place, zigzag
/// levels out. Returns the last nonzero zigzag index (or -1) and the coefficient SSE.
pub(crate) fn quantize_sse(
    m: &QuantMatrix,
    in_: &mut [i16; 16],
    out: &mut [i16; 16],
) -> (i32, i64) {
    (selected().quantize_sse)(m, in_, out)
}

/// `quantize_sse` over the blocks of `in_`; bit n of the result is set when block n has a nonzero
/// level.
pub(crate) fn quantize_n(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    let n = in_.len().min(out.len()).min(32);
    (selected().quantize_n)(m, &mut in_[..n], &mut out[..n])
}

/// Adds the inverse DCT of `c` to the block of `z` at (y, x).
pub(crate) fn idct_add(z: &mut Ybr, c: &[i16; 16], y: usize, x: usize) {
    if *c == [0; 16] {
        return;
    }
    let mut r = get_rows(z, y, x);
    (selected().idct_add4)(c, &mut r);
    put_rows(z, &r, y, x);
}

/// The fdct of the 16 luma blocks of `src` (stride 16) against the prediction in `z`.
pub(crate) fn residuals_y(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    (selected().residuals_y)(z, src, dq);
}

/// The fdct of the 8 chroma blocks of `u` then `v` (stride 8) against the prediction in `z`.
pub(crate) fn residuals_uv(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    (selected().residuals_uv)(z, u, v, dq);
}

/// Adds the 16 luma blocks to `z`.
pub(crate) fn idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    (selected().idct_adds_y)(z, dq);
}

/// Adds the 8 chroma blocks to `z`.
pub(crate) fn idct_adds_uv(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    (selected().idct_adds_uv)(z, dq);
}

/// Luma block `n`: its (y, x) in `z` and its offset in the 16-stride source.
pub(super) fn luma_pos(n: usize) -> (usize, usize, usize) {
    (1 + 4 * (n / 4), 8 + 4 * (n % 4), 64 * (n / 4) + 4 * (n % 4))
}

/// Chroma block `c`: its (y, x) in `z` and its offset in the 8-stride source of its plane.
pub(super) fn chroma_pos(c: usize) -> (usize, usize, usize) {
    let (oy, ox) = (4 * ((c & 3) >> 1), 4 * (c & 1));
    (18 + oy, 8 + 16 * (c >> 2) + ox, oy * 8 + ox)
}

// The scalar reference, bit for bit what the encoder did before the kernels.

fn ref_fdct4(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    let mut t = [[0i32; 4]; 4];
    for r in 0..4 {
        let (s, p) = (src[r].to_le_bytes(), pred[r].to_le_bytes());
        let d = |i: usize| i32::from(s[i]) - i32::from(p[i]);
        let (a0, a1, a2, a3) = (d(0) + d(3), d(1) + d(2), d(1) - d(2), d(0) - d(3));
        t[r] = [
            (a0 + a1) * 8,
            (a2 * 2217 + a3 * 5352 + 1812) >> 9,
            (a0 - a1) * 8,
            (a3 * 2217 - a2 * 5352 + 937) >> 9,
        ];
    }
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
        out[c] = ((a0 + a1 + 7) >> 4) as i16;
        out[4 + c] = v as i16;
        out[8 + c] = ((a0 - a1 + 7) >> 4) as i16;
        out[12 + c] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16;
    }
}

fn ref_quantize_sse(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    let v = i32::from(in_[0]);
    let nz = level_of(v, m.sharpen[0] as u32, m.iq[0], m.bias[0]);
    out[0] = nz as i16;
    let dq = nz.wrapping_mul(m.q[0] as i32);
    in_[0] = dq as i16;
    let mut d = i64::from(v - dq) * i64::from(v - dq);
    let mut last = if nz != 0 { 0 } else { -1 };
    let q = m.q[1] as i32;
    for n in 1..16 {
        let j = (ZIGZAG[n] & 15) as usize;
        let v = i32::from(in_[j]);
        let level = level_of(v, m.sharpen[j] as u32, m.iq[1], m.bias[1]);
        out[n] = level as i16;
        let dq = level.wrapping_mul(q);
        in_[j] = dq as i16;
        d += i64::from(v - dq) * i64::from(v - dq);
        if level != 0 {
            last = n as i32;
        }
    }
    (last, d)
}

fn ref_quantize_n(m: &QuantMatrix, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
    let mut nz = 0;
    for (n, (i, o)) in in_.iter_mut().zip(out.iter_mut()).enumerate() {
        if ref_quantize_sse(m, i, o).0 >= 0 {
            nz |= 1 << n;
        }
    }
    nz
}

fn ref_idct_add4(c: &[i16; 16], rows: &mut Rows) {
    const C1: i32 = 85627; // 65536 * cos(pi/8) * sqrt(2).
    const C2: i32 = 35468; // 65536 * sin(pi/8) * sqrt(2).
    if *c == [0; 16] {
        return;
    }
    // Go multiplies in int32, which wraps for full-range input; so do these.
    let mh = |v: i32, k: i32| v.wrapping_mul(k) >> 16;
    if c[1..].iter().all(|&v| v == 0) {
        let dc = (i32::from(c[0]) + 4) >> 3;
        for r in rows.iter_mut() {
            *r = u32::from_le_bytes(r.to_le_bytes().map(|b| clip8(i32::from(b) + dc)));
        }
        return;
    }
    let mut m = [[0i32; 4]; 4];
    for i in 0..4 {
        let a = i32::from(c[i]) + i32::from(c[i + 8]);
        let b = i32::from(c[i]) - i32::from(c[i + 8]);
        let cc = mh(i32::from(c[i + 4]), C2) - mh(i32::from(c[i + 12]), C1);
        let d = mh(i32::from(c[i + 4]), C1) + mh(i32::from(c[i + 12]), C2);
        m[i] = [a + d, b + cc, b - cc, a - d];
    }
    for j in 0..4 {
        let dc = m[0][j] + 4;
        let a = dc + m[2][j];
        let b = dc - m[2][j];
        let cc = mh(m[1][j], C2) - mh(m[3][j], C1);
        let d = mh(m[1][j], C1) + mh(m[3][j], C2);
        let o = [(a + d) >> 3, (b + cc) >> 3, (b - cc) >> 3, (a - d) >> 3];
        let p = rows[j].to_le_bytes();
        rows[j] = u32::from_le_bytes([0, 1, 2, 3].map(|i| clip8(i32::from(p[i]) + o[i])));
    }
}

fn ref_residuals_y(z: &Ybr, src: &[u8; 256], dq: &mut [[i16; 16]; 16]) {
    for (n, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = luma_pos(n);
        ref_fdct4(&rows_at(src, 16, off), &get_rows(z, y, x), blk);
    }
}

fn ref_residuals_uv(z: &Ybr, u: &[u8; 64], v: &[u8; 64], dq: &mut [[i16; 16]; 8]) {
    for (c, blk) in dq.iter_mut().enumerate() {
        let (y, x, off) = chroma_pos(c);
        let src = if c >= 4 { v } else { u };
        ref_fdct4(&rows_at(src, 8, off), &get_rows(z, y, x), blk);
    }
}

fn ref_idct_adds_y(z: &mut Ybr, dq: &[[i16; 16]; 16]) {
    for (n, c) in dq.iter().enumerate() {
        let (y, x, _) = luma_pos(n);
        let mut r = get_rows(z, y, x);
        ref_idct_add4(c, &mut r);
        put_rows(z, &r, y, x);
    }
}

fn ref_idct_adds_uv(z: &mut Ybr, dq: &[[i16; 16]; 8]) {
    for (c, b) in dq.iter().enumerate() {
        let (y, x, _) = chroma_pos(c);
        let mut r = get_rows(z, y, x);
        ref_idct_add4(b, &mut r);
        put_rows(z, &r, y, x);
    }
}

/// The fuzz target's check, one kernel picked by the next byte: every tier equals the scalar
/// reference on arbitrary bytes. Quantizers are the encoder's (`qi` 0..=127, the uv delta -4..=6).
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    use crate::vp8enc::quant::new_quant;
    let tiers: Vec<Block> = tiers().into_iter().map(|(_, k)| k).collect();
    match r.u8() % 8 {
        0 => {
            let (s, p) = (r.rows(), r.rows());
            let mut want = [0; 16];
            ref_fdct4(&s, &p, &mut want);
            for k in &tiers {
                let mut got = [1; 16];
                (k.fdct4)(&s, &p, &mut got);
                assert_eq!(got, want, "fdct4");
            }
        }
        1 | 2 => {
            let one = r.u8() & 1 == 0;
            let q = new_quant(usize::from(r.u8() % 128), i32::from(r.u8() % 11) - 4);
            let m = [&q.y1, &q.y2, &q.uv][usize::from(r.u8() % 3)];
            // `quantize_n` takes up to 32 blocks, `quantize_sse` one.
            let n = if one { 1 } else { 1 + usize::from(r.u8() % 32) };
            let blocks: Vec<[i16; 16]> = (0..n).map(|_| r.coeffs()).collect();
            let (mut wi, mut wo) = (blocks.clone(), vec![[7i16; 16]; n]);
            let want = if one {
                let (l, d) = ref_quantize_sse(m, &mut wi[0], &mut wo[0]);
                (l, d, 0)
            } else {
                (0, 0, ref_quantize_n(m, &mut wi, &mut wo))
            };
            for k in &tiers {
                let (mut gi, mut go) = (blocks.clone(), vec![[7i16; 16]; n]);
                let got = if one {
                    let (l, d) = (k.quantize_sse)(m, &mut gi[0], &mut go[0]);
                    (l, d, 0)
                } else {
                    (0, 0, (k.quantize_n)(m, &mut gi, &mut go))
                };
                assert_eq!(
                    (got, gi, go),
                    (want, wi.clone(), wo.clone()),
                    "quantize n {n}"
                );
            }
        }
        3 => {
            let (c, p) = (r.coeffs(), r.rows());
            let mut want = p;
            ref_idct_add4(&c, &mut want);
            for k in &tiers {
                let mut got = p;
                (k.idct_add4)(&c, &mut got);
                assert_eq!(got, want, "idct_add4 {c:?}");
            }
        }
        4 => {
            let z = r.z();
            let mut src = [0u8; 256];
            r.fill(&mut src);
            let mut want = [[1i16; 16]; 16];
            ref_residuals_y(&z, &src, &mut want);
            for k in &tiers {
                let mut got = [[1i16; 16]; 16];
                (k.residuals_y)(&z, &src, &mut got);
                assert_eq!(got, want, "residuals_y");
            }
        }
        5 => {
            let z = r.z();
            let (mut u, mut v) = ([0u8; 64], [0u8; 64]);
            r.fill(&mut u);
            r.fill(&mut v);
            let mut want = [[1i16; 16]; 8];
            ref_residuals_uv(&z, &u, &v, &mut want);
            for k in &tiers {
                let mut got = [[1i16; 16]; 8];
                (k.residuals_uv)(&z, &u, &v, &mut got);
                assert_eq!(got, want, "residuals_uv");
            }
        }
        6 => {
            let z = r.z();
            let dq: [[i16; 16]; 16] = std::array::from_fn(|_| r.coeffs());
            let mut want = z;
            ref_idct_adds_y(&mut want, &dq);
            for k in &tiers {
                let mut got = z;
                (k.idct_adds_y)(&mut got, &dq);
                assert!(got == want, "idct_adds_y");
            }
        }
        _ => {
            let z = r.z();
            let dq: [[i16; 16]; 8] = std::array::from_fn(|_| r.coeffs());
            let mut want = z;
            ref_idct_adds_uv(&mut want, &dq);
            for k in &tiers {
                let mut got = z;
                (k.idct_adds_uv)(&mut got, &dq);
                assert!(got == want, "idct_adds_uv");
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
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 >> 16
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
        /// An i16 in [-r, r].
        fn small(&mut self, r: i32) -> i16 {
            ((self.next() % (2 * r as u64 + 1)) as i32 - r) as i16
        }
    }

    const EDGE: [i16; 12] = [
        0, 1, -1, 2047, 2048, -2047, -2048, 32767, -32768, -32767, 255, -256,
    ];

    /// A block of coefficients: random of magnitude `r`, or with edge values mixed in.
    fn coeffs(rng: &mut Rng, r: i32) -> [i16; 16] {
        let mut c = [0i16; 16];
        for v in c.iter_mut() {
            *v = if r == 0 {
                EDGE[(rng.next() % 12) as usize]
            } else if rng.next().is_multiple_of(4) {
                0
            } else {
                rng.small(r)
            };
        }
        c
    }

    fn rows(rng: &mut Rng, mode: u32) -> Rows {
        [0; 4].map(|_| match mode {
            0 => rng.next() as u32,
            1 => 0,
            _ => u32::MAX,
        })
    }

    #[test]
    fn dispatch_is_consistent() {
        // Whatever tier is selected, the public wrappers match the scalar reference.
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..50 {
            let (s, p) = (rows(&mut rng, 0), rows(&mut rng, 0));
            let (mut a, mut b) = ([0; 16], [0; 16]);
            fdct4(&s, &p, &mut a);
            ref_fdct4(&s, &p, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn fdct4_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for (name, k) in tiers() {
            for i in 0..2000 {
                let mode = if i < 4 { i % 3 } else { 0 };
                let (s, p) = if i < 4 {
                    (rows(&mut rng, (i % 2) * 2), rows(&mut rng, (i / 2) * 2))
                } else {
                    (rows(&mut rng, mode), rows(&mut rng, mode))
                };
                let (mut a, mut b) = ([0; 16], [0; 16]);
                (k.fdct4)(&s, &p, &mut a);
                ref_fdct4(&s, &p, &mut b);
                assert_eq!(a, b, "{name} fdct4 {s:?} {p:?}");
            }
        }
    }

    #[test]
    fn quantize_matches_scalar() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for (name, k) in tiers() {
            for qi in 0..=127usize {
                for d in -4..=6 {
                    let q = new_quant(qi, d);
                    for (mn, m) in [("y1", &q.y1), ("y2", &q.y2), ("uv", &q.uv)] {
                        // Only the uv matrix depends on d.
                        if d != 0 && mn != "uv" {
                            continue;
                        }
                        let mut blocks: Vec<[i16; 16]> = Vec::new();
                        for r in [0, 40, 600, 2047, 32767] {
                            for _ in 0..4 {
                                blocks.push(coeffs(&mut rng, r));
                            }
                        }
                        for blk in &blocks {
                            let (mut ia, mut ib) = (*blk, *blk);
                            let (mut oa, mut ob) = ([7i16; 16], [7i16; 16]);
                            let ra = (k.quantize_sse)(m, &mut ia, &mut oa);
                            let rb = ref_quantize_sse(m, &mut ib, &mut ob);
                            assert_eq!(ra, rb, "{name} qi {qi} {mn} d {d} {blk:?}");
                            assert_eq!((ia, oa), (ib, ob), "{name} qi {qi} {mn} d {d} {blk:?}");
                        }
                        for n in 1..=16 {
                            let mut ia: Vec<[i16; 16]> = blocks[..n].to_vec();
                            // Every third block all zero so the nz bits vary.
                            for (i, b) in ia.iter_mut().enumerate() {
                                if i % 3 == 2 {
                                    *b = [0; 16];
                                }
                            }
                            let mut ib = ia.clone();
                            let (mut oa, mut ob) = (vec![[9i16; 16]; n], vec![[9i16; 16]; n]);
                            let na = (k.quantize_n)(m, &mut ia, &mut oa);
                            let nb = ref_quantize_n(m, &mut ib, &mut ob);
                            assert_eq!(na, nb, "{name} qi {qi} {mn} n {n}");
                            assert_eq!((ia, oa), (ib, ob), "{name} qi {qi} {mn} n {n}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn idct_add4_matches_scalar() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for (name, k) in tiers() {
            for i in 0..6000 {
                let r = [0, 20, 300, 2047, 2048, 32767][i % 6];
                let mut c = coeffs(&mut rng, r);
                if i % 7 == 0 {
                    c[1..].fill(0); // DC only.
                }
                if i % 11 == 0 {
                    c = [0; 16];
                }
                let p = rows(&mut rng, if i % 5 == 0 { (i as u32 / 5) % 3 } else { 0 });
                let (mut a, mut b) = (p, p);
                (k.idct_add4)(&c, &mut a);
                ref_idct_add4(&c, &mut b);
                assert_eq!(a, b, "{name} idct {c:?} {p:?}");
            }
        }
    }

    fn rand_z(rng: &mut Rng) -> Ybr {
        let mut z = [[0u8; 32]; 26];
        for row in z.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.byte();
            }
        }
        z
    }

    #[test]
    fn batches_match_scalar() {
        let mut rng = Rng(0x5151_5151_a1a1_a1a1);
        for (name, k) in tiers() {
            for it in 0..300 {
                let z = rand_z(&mut rng);
                let mut src = [0u8; 256];
                let (mut u, mut v) = ([0u8; 64], [0u8; 64]);
                for b in src.iter_mut().chain(u.iter_mut()).chain(v.iter_mut()) {
                    *b = if it % 10 == 0 {
                        255 * (it as u8 / 10 % 2)
                    } else {
                        rng.byte()
                    };
                }
                let (mut a, mut b) = ([[1i16; 16]; 16], [[1i16; 16]; 16]);
                (k.residuals_y)(&z, &src, &mut a);
                ref_residuals_y(&z, &src, &mut b);
                assert_eq!(a, b, "{name} residuals_y");
                let (mut a, mut b) = ([[1i16; 16]; 8], [[1i16; 16]; 8]);
                (k.residuals_uv)(&z, &u, &v, &mut a);
                ref_residuals_uv(&z, &u, &v, &mut b);
                assert_eq!(a, b, "{name} residuals_uv");

                let r = [20, 300, 2047, 2048, 32767][it % 5];
                let mut dy = [[0i16; 16]; 16];
                for (n, c) in dy.iter_mut().enumerate() {
                    *c = if n % 4 == 3 {
                        [0; 16]
                    } else {
                        coeffs(&mut rng, r)
                    };
                }
                let (mut za, mut zb) = (z, z);
                (k.idct_adds_y)(&mut za, &dy);
                ref_idct_adds_y(&mut zb, &dy);
                assert!(za == zb, "{name} idct_adds_y");
                let mut du = [[0i16; 16]; 8];
                for (n, c) in du.iter_mut().enumerate() {
                    *c = if n % 4 == 3 {
                        [0; 16]
                    } else {
                        coeffs(&mut rng, r)
                    };
                }
                let (mut za, mut zb) = (z, z);
                (k.idct_adds_uv)(&mut za, &du);
                ref_idct_adds_uv(&mut zb, &du);
                assert!(za == zb, "{name} idct_adds_uv");
            }
        }
    }
}
