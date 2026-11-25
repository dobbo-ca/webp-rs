//! Encoder kernels of package P3, the rate costs (Go's `resCostAsm`, `costGridAsm` and
//! `cost4Asm`): scalar reference, tier dispatch and cross-tier tests. libwebp's GetResidualCost
//! (src/dsp/cost.c) in table form, BSD-3-Clause.

use std::sync::OnceLock;

/// The tables one residual type costs a block with, as plain arrays (`vp8enc::rdo::CostTabs`
/// folds them). Every row of `cost` has entry 127 at 0, which the vector tiers read for the
/// positions they mask out.
#[derive(Clone, Copy)]
pub(crate) struct Tab<'a> {
    /// [position][ctx][level]: levels 0..=67 and the level class costs folded in.
    pub cost: &'a [[[u16; 128]; 4]; 16],
    /// The cost of ending the block after position n, at the context of the last level.
    pub eob: &'a [[u16; 4]; 16],
    /// The fixed cost of each level, for the levels past 67.
    pub fixed: &'a [i32; 2048],
    /// The cost of an empty block at ctx 0, 1 and 2.
    pub empty: [i32; 3],
    /// The cost of the not-EOB bit at ctx 0, which `cost` carries for ctx > 0 only.
    pub nz0: i32,
}

/// `cost_grid`'s kernel: the table, `first`, the blocks, the columns, `lbits` and `tbits`.
pub(crate) type CostGrid = fn(&Tab, usize, &[[i16; 16]], usize, u32, u32) -> i32;

/// One tier's kernels, called with `ctx <= 2`, `first <= 1` and `-1 <= last <= 15`.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) res_cost: fn(&Tab, usize, usize, i32, &[i16; 16]) -> i32,
    pub(crate) cost_grid: CostGrid,
}

pub(crate) const SCALAR: Kernels = Kernels {
    res_cost: scalar_res_cost,
    cost_grid: scalar_cost_grid,
};

/// The last nonzero index at or after `first`, or -1.
fn scalar_last(lv: &[i16; 16], first: usize) -> i32 {
    (first..16)
        .rev()
        .find(|&n| lv[n] != 0)
        .map_or(-1, |n| n as i32)
}

pub(in crate::simd) fn scalar_res_cost(
    t: &Tab,
    ctx: usize,
    first: usize,
    last: i32,
    lv: &[i16; 16],
) -> i32 {
    if last < 0 {
        return t.empty[ctx];
    }
    let last = last as usize;
    let mut r = if ctx == 0 { t.nz0 } else { 0 };
    let mut c = ctx;
    for n in first..=last {
        let v = usize::from(lv[n].unsigned_abs());
        r += i32::from(t.cost[n][c][v.min(67)]);
        if v > 67 {
            r += t.fixed[v & 2047] - t.fixed[67];
        }
        c = v.min(2);
    }
    if last < 15 {
        r += i32::from(t.eob[last][usize::from(lv[last].unsigned_abs()).min(2)]);
    }
    r
}

pub(in crate::simd) fn scalar_cost_grid(
    t: &Tab,
    first: usize,
    lv: &[[i16; 16]],
    cols: usize,
    mut lbits: u32,
    mut tbits: u32,
) -> i32 {
    let mut r = 0;
    for (i, b) in lv.iter().enumerate() {
        let (y, x) = (i / cols, i % cols);
        let ctx = ((lbits >> y) & 1) + ((tbits >> x) & 1);
        let last = scalar_last(b, first);
        r += scalar_res_cost(t, ctx as usize, first, last, b);
        let nz = u32::from(last >= 0);
        lbits = lbits & !(1 << y) | nz << y;
        tbits = tbits & !(1 << x) | nz << x;
    }
    r
}

#[cfg(target_arch = "x86_64")]
use super::avx2::enc_cost::detect;
#[cfg(target_arch = "aarch64")]
use super::neon::enc_cost::detect;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> Option<Kernels> {
    None
}

/// Every SIMD tier this CPU can run, whichever `selected` picks.
pub(crate) fn tiers() -> Vec<(&'static str, Kernels)> {
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.extend(super::avx2::enc_cost::detect().map(|k| ("avx2", k)));
    }
    #[cfg(target_arch = "aarch64")]
    v.extend(super::neon::enc_cost::detect().map(|k| ("neon", k)));
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

/// GetResidualCost_C over `lv[first..=last]` at context `ctx`; `last` is -1 (an empty block) or
/// at least `first`, usually `last_nz`.
pub(crate) fn res_cost(t: &Tab, ctx: usize, first: usize, last: i32, lv: &[i16; 16]) -> i32 {
    (selected().res_cost)(t, ctx.min(2), first.min(1), last.clamp(-1, 15), lv)
}

/// The blocks `lv` in raster order on a grid of `cols` columns (4 blocks on 2, or 16 on 4), each
/// at the context of the bits `lbits` (its row) and `tbits` (its column), which then become
/// whether the block has a level: the summed `res_cost` at `last_nz`.
pub(crate) fn cost_grid(
    t: &Tab,
    first: usize,
    lv: &[[i16; 16]],
    cols: usize,
    lbits: u32,
    tbits: u32,
) -> i32 {
    assert!(matches!((lv.len(), cols), (4, 2) | (16, 4)));
    (selected().cost_grid)(t, first.min(1), lv, cols, lbits, tbits)
}

/// The fuzz target's check: every tier equals the scalar reference on random tables and levels.
#[cfg(fuzzing)]
pub(crate) fn fuzz(r: &mut super::Input) {
    let tiers: Vec<Kernels> = tiers().into_iter().map(|(_, k)| k).collect();
    let tabs = tests::RandTab::new(u64::from(r.u32()) << 32 | u64::from(r.u32()) | 1);
    let t = tabs.tab();
    let (ctx, first) = (usize::from(r.u8() % 3), usize::from(r.u8() & 1));
    if r.u8() & 1 == 0 {
        let lv = r.coeffs();
        let last = if r.u8() & 3 == 0 {
            i32::from(r.u8() % 17) - 1
        } else {
            scalar_last(&lv, first)
        };
        let last = if last >= 0 && (last as usize) < first {
            -1
        } else {
            last
        };
        let want = scalar_res_cost(&t, ctx, first, last, &lv);
        for k in &tiers {
            assert_eq!((k.res_cost)(&t, ctx, first, last, &lv), want, "res_cost");
        }
    } else {
        let (n, cols) = if r.u8() & 1 == 0 { (16, 4) } else { (4, 2) };
        let lv: Vec<[i16; 16]> = (0..n).map(|_| r.coeffs()).collect();
        let (lb, tb) = (u32::from(r.u8() & 15), u32::from(r.u8() & 15));
        let want = scalar_cost_grid(&t, first, &lv, cols, lb, tb);
        for k in &tiers {
            assert_eq!(
                (k.cost_grid)(&t, first, &lv, cols, lb, tb),
                want,
                "cost_grid"
            );
        }
    }
}

#[cfg(any(test, fuzzing))]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn xorshift64(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Random tables: rows below 4096 with entry 127 at 0, as `CostTabs::build` leaves them.
    pub(crate) struct RandTab {
        cost: Box<[[[u16; 128]; 4]; 16]>,
        eob: [[u16; 4]; 16],
        fixed: Box<[i32; 2048]>,
        empty: [i32; 3],
        nz0: i32,
    }

    impl RandTab {
        pub(crate) fn new(mut s: u64) -> RandTab {
            let mut cost = Box::new([[[0u16; 128]; 4]; 16]);
            for row in cost.iter_mut().flatten() {
                for v in &mut row[..127] {
                    *v = (xorshift64(&mut s) >> 20) as u16 % 4096;
                }
            }
            let mut eob = [[0u16; 4]; 16];
            eob.iter_mut()
                .flatten()
                .for_each(|v| *v = (xorshift64(&mut s) >> 20) as u16 % 4096);
            let mut fixed = Box::new([0i32; 2048]);
            fixed
                .iter_mut()
                .for_each(|v| *v = (xorshift64(&mut s) >> 20) as i32 % 8192);
            RandTab {
                cost,
                eob,
                fixed,
                empty: std::array::from_fn(|_| (xorshift64(&mut s) >> 20) as i32 % 4096),
                nz0: (xorshift64(&mut s) >> 20) as i32 % 4096,
            }
        }

        pub(crate) fn tab(&self) -> Tab<'_> {
            Tab {
                cost: &self.cost,
                eob: &self.eob,
                fixed: &self.fixed,
                empty: self.empty,
                nz0: self.nz0,
            }
        }
    }
}

#[cfg(test)]
mod cases {
    use super::tests::*;
    use super::*;

    /// A block of one of eight kinds: zeros, +-1, +-2, around 67, up to +-2047, the i16 extremes,
    /// sparse, and uniform random.
    fn block(s: &mut u64, kind: usize) -> [i16; 16] {
        std::array::from_fn(|_| {
            let r = xorshift64(s);
            let m = (r >> 16) as i16;
            let sign = |v: i16| if r & 1 == 0 { v } else { -v };
            match kind {
                0 => 0,
                1 => sign(1),
                2 => sign(1 + (r >> 8) as i16 % 2),
                3 => sign(66 + (r >> 8) as i16 % 3),
                4 => sign((r >> 8) as i16 & 2047),
                5 => [i16::MIN, i16::MAX, 0, 1, -1][(r >> 8) as usize % 5],
                6 if !(r >> 8).is_multiple_of(4) => 0,
                6 => sign((r >> 12) as i16 % 5),
                _ => m,
            }
        })
    }

    #[test]
    fn res_cost_matches_scalar() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for (name, k) in tiers() {
            for seed in 1..4 {
                let rt = RandTab::new(seed * 77);
                let t = rt.tab();
                for kind in 0..8 {
                    for _ in 0..300 {
                        let lv = block(&mut s, kind);
                        for ctx in 0..3 {
                            for first in 0..2 {
                                let mut lasts = vec![scalar_last(&lv, first), -1];
                                lasts.push(
                                    first as i32
                                        + (xorshift64(&mut s) % (16 - first as u64)) as i32,
                                );
                                for last in lasts {
                                    assert_eq!(
                                        (k.res_cost)(&t, ctx, first, last, &lv),
                                        scalar_res_cost(&t, ctx, first, last, &lv),
                                        "{name} {lv:?} ctx {ctx} first {first} last {last}"
                                    );
                                    assert_eq!(
                                        res_cost(&t, ctx, first, last, &lv),
                                        scalar_res_cost(&t, ctx, first, last, &lv)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cost_grid_matches_scalar() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for (name, k) in tiers() {
            let rt = RandTab::new(5);
            let t = rt.tab();
            for (n, cols, bits) in [(16usize, 4usize, 16u32), (4, 2, 4)] {
                for first in 0..2 {
                    for _ in 0..200 {
                        let kind = (xorshift64(&mut s) % 8) as usize;
                        let lv: Vec<[i16; 16]> = (0..n)
                            .map(|_| {
                                let k = if xorshift64(&mut s).is_multiple_of(3) {
                                    0
                                } else {
                                    kind
                                };
                                block(&mut s, k)
                            })
                            .collect();
                        for lb in 0..bits {
                            for tb in 0..bits {
                                let want = scalar_cost_grid(&t, first, &lv, cols, lb, tb);
                                assert_eq!(
                                    (k.cost_grid)(&t, first, &lv, cols, lb, tb),
                                    want,
                                    "{name} n {n} first {first} {lb:#x} {tb:#x}"
                                );
                                assert_eq!(cost_grid(&t, first, &lv, cols, lb, tb), want);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The grid is the blocks costed one by one, with the contexts carried along.
    #[test]
    fn scalar_grid_is_blockwise() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let rt = RandTab::new(9);
        let t = rt.tab();
        for _ in 0..200 {
            let lv: Vec<[i16; 16]> = (0..4).map(|_| block(&mut s, 6)).collect();
            let (mut l, mut tp, mut want) = ([0u32; 2], [0u32; 2], 0);
            for (i, b) in lv.iter().enumerate() {
                let (y, x) = (i / 2, i % 2);
                let last = scalar_last(b, 0);
                want += scalar_res_cost(&t, (l[y] + tp[x]) as usize, 0, last, b);
                l[y] = u32::from(last >= 0);
                tp[x] = l[y];
            }
            assert_eq!(scalar_cost_grid(&t, 0, &lv, 2, 0, 0), want);
        }
    }
}
