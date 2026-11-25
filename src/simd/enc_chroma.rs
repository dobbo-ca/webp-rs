//! Encoder kernels of package E, the opaque chroma row conversion (Go's `chromaRow`): scalar
//! reference, tier dispatch and cross-tier tests.

use std::sync::OnceLock;

/// One tier's kernel: converts exactly `cb.len()` 2x2 blocks (a multiple of 8) of the row pair
/// `p0`/`p1` (`8 * cb.len()` bytes each) into `cb`/`cr`, and returns the AND of their alpha bytes.
/// `to_gamma` is libwebp's table, whose steps fit a byte.
pub(crate) type ChromaFn = fn(&mut [u8], &mut [u8], &[u8], &[u8], &[i32; 256], &[i32; 33]) -> u8;

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) chroma: ChromaFn,
}

pub(crate) const SCALAR: Kernels = Kernels { chroma: scalar };

/// Interpolate plus LinearToGamma, libwebp yuv.c:257-276.
fn lin_to_gamma(g: &[i32; 33], s: i32) -> i32 {
    let (p, x) = ((s >> 9) as usize, s & 511);
    (g[p + 1] * x + g[p] * (512 - x) + 64) >> 7
}

/// The reference: the opaque chroma loop of `rgba_to_ycbcr`, over whole blocks.
fn scalar(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> u8 {
    let mut and = 255u8;
    for (((cb, cr), r0), r1) in cb
        .iter_mut()
        .zip(cr.iter_mut())
        .zip(p0.as_chunks::<8>().0)
        .zip(p1.as_chunks::<8>().0)
    {
        let px = [&r0[..4], &r0[4..], &r1[..4], &r1[4..]];
        and &= px[0][3] & px[1][3] & px[2][3] & px[3][3];
        let s = |i: usize| {
            lin_to_gamma(
                to_gamma,
                px.iter().map(|p| to_linear[p[i] as usize]).sum::<i32>(),
            )
        };
        let (sr, sg, sb) = (s(0), s(1), s(2));
        *cb = ((-9719 * sr - 19081 * sg + 28800 * sb + (1 << 17) + (128 << 18)) >> 18) as u8;
        *cr = ((28800 * sr - 24116 * sg - 4684 * sb + (1 << 17) + (128 << 18)) >> 18) as u8;
    }
    and
}

#[cfg(target_arch = "x86_64")]
fn detect() -> Option<Kernels> {
    super::avx512_wanted()
        .then(super::avx512::enc_chroma::detect)
        .flatten()
        .or_else(super::avx2::enc_chroma::detect)
}
#[cfg(target_arch = "aarch64")]
use super::neon::enc_chroma::detect;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Kernels> {
    None
}

/// Every SIMD tier this CPU can run, whichever `selected` picks.
pub(crate) fn tiers() -> Vec<(&'static str, Kernels)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.extend(super::avx2::enc_chroma::detect().map(|k| ("avx2", k)));
        v.extend(super::avx512::enc_chroma::detect().map(|k| ("avx512", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_chroma::detect().map(|k| ("neon", k)));
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

/// Go's `chromaRow`: converts the row pair's whole 8-block groups of `p0`/`p1` into `cb`/`cr` and
/// returns how many blocks it did and their alpha AND. `(0, 255)` means no fast path, and the
/// caller converts every block itself.
pub(crate) fn chroma_row(
    cb: &mut [u8],
    cr: &mut [u8],
    p0: &[u8],
    p1: &[u8],
    to_linear: &[i32; 256],
    to_gamma: &[i32; 33],
) -> (usize, u8) {
    let n = (p0.len() / 8).min(p1.len() / 8).min(cb.len()).min(cr.len()) & !7;
    if n == 0 {
        return (0, 255);
    }
    let and = (selected().chroma)(
        &mut cb[..n],
        &mut cr[..n],
        &p0[..8 * n],
        &p1[..8 * n],
        to_linear,
        to_gamma,
    );
    (n, and)
}

/// The fuzz target's check: every tier equals the scalar loop on arbitrary pixels, `n` blocks
/// (a multiple of 8, up to 64) of a row pair, with libwebp's gamma tables.
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    use crate::fmath::pow;
    let n = 8 * (1 + usize::from(r.u8() % 8));
    let (mut lin, mut gam) = ([0i32; 256], [0i32; 33]);
    for (v, e) in lin.iter_mut().enumerate() {
        *e = (pow(v as f64 * (1.0 / 255.0), 0.8) * 4095.0 + 0.5) as i32;
    }
    for (v, e) in gam.iter_mut().enumerate() {
        *e = (255.0 * pow(v as f64 * (128.0 / 4095.0), 1.25) + 0.5) as i32;
    }
    let (p0, p1) = (r.bytes(8 * n), r.bytes(8 * n));
    let (mut wcb, mut wcr) = (vec![7u8; n], vec![9u8; n]);
    let want = scalar(&mut wcb, &mut wcr, &p0, &p1, &lin, &gam);
    for (_, k) in tiers() {
        let (mut cb, mut cr) = (vec![7u8; n], vec![9u8; n]);
        let and = (k.chroma)(&mut cb, &mut cr, &p0, &p1, &lin, &gam);
        assert_eq!(
            (cb, cr, and),
            (wcb.clone(), wcr.clone(), want),
            "chroma n {n}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fmath::pow;

    /// gammaToLinear and linearToGamma, libwebp yuv.c:237-251.
    fn gamma() -> ([i32; 256], [i32; 33]) {
        let (mut l, mut g) = ([0; 256], [0; 33]);
        for (v, e) in l.iter_mut().enumerate() {
            *e = (pow(v as f64 * (1.0 / 255.0), 0.8) * 4095.0 + 0.5) as i32;
        }
        for (v, e) in g.iter_mut().enumerate() {
            *e = (255.0 * pow(v as f64 * (128.0 / 4095.0), 1.25) + 0.5) as i32;
        }
        (l, g)
    }

    fn xorshift(s: &mut u64) -> u8 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 24) as u8
    }

    /// Every tier equals the scalar loop, bit for bit, on `n` blocks of the row pair.
    fn check(name: &str, p0: &[u8], p1: &[u8], n: usize) {
        let (lin, gam) = gamma();
        let (mut wcb, mut wcr) = (vec![7u8; n], vec![9u8; n]);
        let want = scalar(&mut wcb, &mut wcr, &p0[..8 * n], &p1[..8 * n], &lin, &gam);
        for (tier, k) in tiers() {
            let (mut cb, mut cr) = (vec![7u8; n], vec![9u8; n]);
            let and = (k.chroma)(&mut cb, &mut cr, &p0[..8 * n], &p1[..8 * n], &lin, &gam);
            assert_eq!((&cb, &cr, and), (&wcb, &wcr, want), "{tier} {name} n {n}");
        }
    }

    /// Every `to_linear` index in every lane of every channel, and random pixels.
    #[test]
    fn tiers_match_scalar() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for shift in 0..256usize {
            for n in [8usize, 16] {
                let px = |lane: usize, row: usize| {
                    let v = shift + 17 * lane + 31 * row;
                    [v as u8, (3 * v + 1) as u8, (5 * v + 2) as u8, 255]
                };
                let p0: Vec<u8> = (0..2 * n).flat_map(|i| px(i, 0)).collect();
                let p1: Vec<u8> = (0..2 * n).flat_map(|i| px(i, 1)).collect();
                check("index sweep", &p0, &p1, n);
            }
        }
        for n in (8..=64).step_by(8) {
            for kind in 0..4 {
                let rand = |st: &mut u64| -> Vec<u8> {
                    (0..16 * n)
                        .map(|i| match kind {
                            0 => xorshift(st),
                            1 => 0,
                            2 => 255,
                            _ if i % 4 == 3 => 255,
                            _ => xorshift(st),
                        })
                        .collect()
                };
                let (p0, p1) = (rand(&mut st), rand(&mut st));
                check("random", &p0, &p1, n);
            }
        }
    }

    /// `chroma_row` over widths 1..=64: the blocks it converts equal the reference's, and an
    /// opaque alpha stays 255 while one transparent pixel clears it.
    #[test]
    fn chroma_row_widths() {
        let (lin, gam) = gamma();
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for w in 1..=64usize {
            let cw = w.div_ceil(2);
            let mut p0: Vec<u8> = (0..4 * w).map(|_| xorshift(&mut st)).collect();
            let mut p1: Vec<u8> = (0..4 * w).map(|_| xorshift(&mut st)).collect();
            for p in [&mut p0, &mut p1] {
                for a in p.iter_mut().skip(3).step_by(4) {
                    *a = 255;
                }
            }
            let (mut cb, mut cr) = (vec![0u8; cw], vec![0u8; cw]);
            let (n, and) = chroma_row(&mut cb, &mut cr, &p0, &p1, &lin, &gam);
            assert_eq!(n, (w / 2) & !7, "w {w}");
            assert_eq!(and, 255);
            let (mut wcb, mut wcr) = (vec![0u8; n], vec![0u8; n]);
            scalar(&mut wcb, &mut wcr, &p0[..8 * n], &p1[..8 * n], &lin, &gam);
            assert_eq!((&cb[..n], &cr[..n]), (&wcb[..], &wcr[..]), "w {w}");
            if n > 0 {
                p1[8 * n - 1] = 0;
                let (_, and) = chroma_row(&mut cb, &mut cr, &p0, &p1, &lin, &gam);
                assert_eq!(and, 0, "w {w}");
            }
        }
    }
}
