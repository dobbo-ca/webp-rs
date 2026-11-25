//! The kernel micro-bench behind `kbench_api` (and `examples/kbench.rs`): times one kernel of one
//! tier on deterministic inputs of a realistic size, so the AVX-512 tier can be compared with the
//! AVX2 one and the kernels that lose dropped (docs/safety.md).

use super::{enc_block, enc_chroma, enc_collect, enc_cost, enc_i4, enc_pred};
use crate::vp8enc::quant::new_quant;
use std::hint::black_box;
use std::time::Instant;

/// The kernels `run` knows, grouped as the modules are.
pub(crate) const KERNELS: &[&str] = &[
    "luma_row",
    "sse_row",
    "i4_trials",
    "rec_metrics16",
    "tt_mb",
    "sse4x10",
    "sse_block",
    "quantize_n",
    "residuals_y",
    "idct_adds_y",
    "collect16",
    "bin_stats",
    "chroma",
    "res_cost",
    "cost_grid",
];

/// The tiers this CPU can run, scalar first.
pub(crate) fn tier_names() -> Vec<&'static str> {
    let mut v = vec!["scalar"];
    v.extend(super::tiers().into_iter().map(|(n, _)| n));
    v
}

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
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        std::array::from_fn(|_| self.byte())
    }
    fn vec(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }
    /// A coefficient in -r..=r.
    fn coeff(&mut self, r: u64) -> i16 {
        (self.next() % (2 * r + 1)) as i16 - r as i16
    }
    fn rows(&mut self) -> [u32; 4] {
        std::array::from_fn(|_| self.next() as u32)
    }
}

/// `f` once to warm up, then `iters` times: the mean ns a call.
fn time(iters: u32, mut f: impl FnMut()) -> f64 {
    f();
    let n = iters.max(1);
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_nanos() as f64 / f64::from(n)
}

/// `scalar`, or the tier of that name among `tiers`.
fn pick<T: Copy>(tier: &str, scalar: T, tiers: Vec<(&'static str, T)>) -> Option<T> {
    if tier == "scalar" {
        return Some(scalar);
    }
    tiers.into_iter().find(|(n, _)| *n == tier).map(|(_, k)| k)
}

/// The mean ns a call of `name` on `tier`, over `iters` calls, or `None` when the kernel or the
/// tier is unknown (or this CPU does not run it).
pub(crate) fn run(name: &str, tier: &str, iters: u32) -> Option<f64> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let q = new_quant(40, 0).y1;
    let z: [[u8; 32]; 26] = std::array::from_fn(|_| rng.bytes());
    let src256: [u8; 256] = rng.bytes();
    let dq: [[i16; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| rng.coeff(300)));
    Some(match name {
        "luma_row" => {
            let k = pick(tier, super::SCALAR, super::tiers())?;
            let (px, mut out) = (rng.vec(4 * 768), vec![0u8; 768]);
            time(iters, || {
                (k.luma_row)(black_box(&mut out), black_box(&px));
            })
        }
        "sse_row" => {
            let k = pick(tier, super::SCALAR, super::tiers())?;
            let (a, b) = (rng.vec(768), rng.vec(768));
            time(iters, || {
                black_box((k.sse_row)(black_box(&a), black_box(&b)));
            })
        }
        "i4_trials" => {
            let k = pick(tier, enc_i4::SCALAR, enc_i4::tiers())?;
            let pr: [[u32; 4]; 10] = std::array::from_fn(|_| rng.rows());
            let src = rng.rows();
            let (mut lv, mut rec, mut res) = ([[0i16; 16]; 10], [[0u32; 4]; 10], [[0i32; 4]; 10]);
            time(iters, || {
                (k.i4_trials)(
                    &q,
                    black_box(&pr),
                    black_box(&src),
                    &mut lv,
                    &mut rec,
                    &mut res,
                );
                black_box((&lv, &rec, &res));
            })
        }
        "rec_metrics16" => {
            let k = pick(tier, enc_i4::SCALAR, enc_i4::tiers())?;
            let st: [i32; 16] = std::array::from_fn(|_| rng.next() as i32 % 4096);
            let mut zz = z;
            time(iters, || {
                black_box((k.rec_metrics16)(
                    black_box(&dq),
                    &mut zz,
                    black_box(&src256),
                    &st,
                ));
            })
        }
        "tt_mb" => {
            let k = pick(tier, enc_i4::SCALAR, enc_i4::tiers())?;
            time(iters, || {
                black_box((k.tt_mb)(black_box(&src256)));
            })
        }
        "sse4x10" => {
            let k = pick(tier, enc_pred::SCALAR, enc_pred::tiers())?;
            let pr: [[u32; 4]; 10] = std::array::from_fn(|_| rng.rows());
            let src = rng.rows();
            time(iters, || {
                black_box((k.sse4x10)(black_box(&pr), black_box(&src)));
            })
        }
        "sse_block" => {
            let k = pick(tier, enc_pred::SCALAR, enc_pred::tiers())?;
            time(iters, || {
                black_box((k.sse_block)(
                    black_box(&src256),
                    16,
                    black_box(&z),
                    1,
                    8,
                    16,
                    16,
                ));
            })
        }
        "quantize_n" => {
            let k = pick(tier, enc_block::SCALAR, enc_block::tiers())?;
            let base: [[i16; 16]; 16] =
                std::array::from_fn(|_| std::array::from_fn(|_| rng.coeff(1000)));
            let mut out = [[0i16; 16]; 16];
            time(iters, || {
                let mut blocks = base;
                black_box((k.quantize_n)(&q, &mut blocks, &mut out));
                black_box((&blocks, &out));
            })
        }
        "residuals_y" => {
            let k = pick(tier, enc_block::SCALAR, enc_block::tiers())?;
            let mut out = [[0i16; 16]; 16];
            time(iters, || {
                (k.residuals_y)(black_box(&z), black_box(&src256), &mut out);
                black_box(&out);
            })
        }
        "idct_adds_y" => {
            let k = pick(tier, enc_block::SCALAR, enc_block::tiers())?;
            let mut zz = z;
            time(iters, || {
                (k.idct_adds_y)(&mut zz, black_box(&dq));
                black_box(&zz);
            })
        }
        "collect16" => {
            let k = pick(tier, enc_collect::SCALAR, enc_collect::tiers())?;
            time(iters, || {
                black_box((k.collect16)(black_box(&src256), 16, black_box(&z)));
            })
        }
        "bin_stats" => {
            let k = pick(tier, enc_collect::SCALAR, enc_collect::tiers())?;
            let bins: Vec<u8> = rng.vec(1024).into_iter().map(|v| v & 31).collect();
            time(iters, || {
                black_box((k.bin_stats)(black_box(&bins)));
            })
        }
        "chroma" => {
            let k = pick(tier, enc_chroma::SCALAR, enc_chroma::tiers())?;
            // libwebp's tables, yuv.c:237-251.
            let (mut lin, mut gam) = ([0i32; 256], [0i32; 33]);
            for (v, e) in lin.iter_mut().enumerate() {
                *e = (crate::fmath::pow(v as f64 * (1.0 / 255.0), 0.8) * 4095.0 + 0.5) as i32;
            }
            for (v, e) in gam.iter_mut().enumerate() {
                *e = (255.0 * crate::fmath::pow(v as f64 * (128.0 / 4095.0), 1.25) + 0.5) as i32;
            }
            // A 768 pixel row pair: 384 blocks.
            let (p0, p1) = (rng.vec(8 * 384), rng.vec(8 * 384));
            let (mut cb, mut cr) = (vec![0u8; 384], vec![0u8; 384]);
            time(iters, || {
                black_box((k.chroma)(
                    &mut cb,
                    &mut cr,
                    black_box(&p0),
                    black_box(&p1),
                    &lin,
                    &gam,
                ));
            })
        }
        "res_cost" | "cost_grid" => {
            let k = pick(tier, enc_cost::SCALAR, enc_cost::tiers())?;
            let mut cost = Box::new([[[0u16; 128]; 4]; 16]);
            for row in cost.iter_mut().flatten() {
                for v in &mut row[..127] {
                    *v = (rng.next() % 4096) as u16;
                }
            }
            let eob: [[u16; 4]; 16] =
                std::array::from_fn(|_| std::array::from_fn(|_| (rng.next() % 4096) as u16));
            let fixed: Box<[i32; 2048]> =
                Box::new(std::array::from_fn(|_| (rng.next() % 8192) as i32));
            let t = enc_cost::Tab {
                cost: &cost,
                eob: &eob,
                fixed: &fixed,
                empty: [100, 200, 300],
                nz0: 50,
            };
            let mut lv: [[i16; 16]; 16] =
                std::array::from_fn(|_| std::array::from_fn(|_| rng.coeff(3)));
            for b in &mut lv {
                b[15] = 1;
            }
            if name == "res_cost" {
                time(iters, || {
                    black_box((k.res_cost)(&t, 1, 0, 15, black_box(&lv[0])));
                })
            } else {
                time(iters, || {
                    black_box((k.cost_grid)(&t, 0, black_box(&lv), 4, 5, 10));
                })
            }
        }
        _ => return None,
    })
}
