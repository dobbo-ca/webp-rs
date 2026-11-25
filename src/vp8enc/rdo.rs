//! Rate-distortion costs: libwebp VP8EntropyCost (dsp/cost.c), the level cost tables
//! (cost_enc.c), the lambdas (quant_enc.c) and TrellisQuantizeBlock (quant_enc.c:569-751),
//! BSD-3-Clause.

use super::header::{put_i4_mode, put_i16_mode, put_uv_mode};
use super::quant::{Quant, QuantMatrix};
use super::tokens::{DEFAULT_PROBS, FIXED_TOK, NzCtx, record_level};
use crate::simd::enc_cost;
use crate::tables::{BANDS, PRED_PROB, ZIGZAG};
use std::sync::OnceLock;

/// Cost of a 0 bit at prob p, in 1/256 bit (dsp/cost.c:25-47).
const ENTROPY_COST: [i32; 256] = [
    1792, 1792, 1792, 1536, 1536, 1408, 1366, 1280, 1280, 1216, 1178, 1152, 1110, 1076, 1061, 1024,
    1024, 992, 968, 951, 939, 911, 896, 878, 871, 854, 838, 820, 811, 794, 786, 768, 768, 752, 740,
    732, 720, 709, 704, 690, 683, 672, 666, 655, 647, 640, 631, 622, 615, 607, 598, 592, 586, 576,
    572, 564, 559, 555, 547, 541, 534, 528, 522, 512, 512, 504, 500, 494, 488, 483, 477, 473, 467,
    461, 458, 452, 448, 443, 438, 434, 427, 424, 419, 415, 410, 406, 403, 399, 394, 390, 384, 384,
    377, 374, 370, 366, 362, 359, 355, 351, 347, 342, 342, 336, 333, 330, 326, 323, 320, 316, 312,
    308, 305, 302, 299, 296, 293, 288, 287, 283, 280, 277, 274, 272, 268, 266, 262, 256, 256, 256,
    251, 248, 245, 242, 240, 237, 234, 232, 228, 226, 223, 221, 218, 216, 214, 211, 208, 205, 203,
    201, 198, 196, 192, 191, 188, 187, 183, 181, 179, 176, 175, 171, 171, 168, 165, 163, 160, 159,
    156, 154, 152, 150, 148, 146, 144, 142, 139, 138, 135, 133, 131, 128, 128, 125, 123, 121, 119,
    117, 115, 113, 111, 110, 107, 105, 103, 102, 100, 98, 96, 94, 92, 91, 89, 86, 86, 83, 82, 80,
    77, 76, 74, 73, 71, 69, 67, 66, 64, 63, 61, 59, 57, 55, 54, 52, 51, 49, 47, 46, 44, 43, 41, 40,
    38, 36, 35, 33, 32, 30, 29, 27, 25, 24, 22, 21, 19, 18, 16, 15, 13, 12, 10, 9, 7, 6, 4, 3,
];

/// Mode costs and the level fixed costs, built once.
pub(crate) struct Costs {
    /// VP8BitCost(bit, p).
    pub bit: [[i32; 256]; 2],
    /// Mode costs in 1/256 bit; `i16` includes the is_i16 bit.
    pub i16: [i32; 4],
    pub uv: [i32; 4],
    pub i4: [[[i32; 10]; 10]; 10],
    /// VP8LevelFixedCosts: the sign bit and the extra bits of level v.
    pub fixed: [i32; 2048],
}

pub(crate) fn costs() -> &'static Costs {
    static C: OnceLock<Box<Costs>> = OnceLock::new();
    C.get_or_init(|| {
        let mut c = Box::new(Costs {
            bit: [[0; 256]; 2],
            i16: [0; 4],
            uv: [0; 4],
            i4: [[[0; 10]; 10]; 10],
            fixed: [0; 2048],
        });
        for p in 0..256 {
            c.bit[0][p] = ENTROPY_COST[p];
            c.bit[1][p] = ENTROPY_COST[255 - p];
        }
        for m in 0..4u8 {
            c.i16[m as usize] = tree_cost(&c.bit, &mut |put: &mut dyn FnMut(bool, u8) -> bool| {
                put(true, 145);
                put_i16_mode(put, m);
            });
            c.uv[m as usize] = tree_cost(&c.bit, &mut |put: &mut dyn FnMut(bool, u8) -> bool| {
                put_uv_mode(put, m)
            });
        }
        for a in 0..10 {
            for l in 0..10 {
                for m in 0..10u8 {
                    c.i4[a][l][m as usize] =
                        tree_cost(&c.bit, &mut |put: &mut dyn FnMut(bool, u8) -> bool| {
                            put_i4_mode(put, m, &PRED_PROB[a][l])
                        });
                }
            }
        }
        let mut toks = Vec::new();
        for v in 1..2048 {
            toks.clear();
            record_level(&mut toks, v, 0);
            c.fixed[v] = 256;
            for &t in &toks {
                if t & FIXED_TOK != 0 {
                    c.fixed[v] += c.bit[(t >> 15) as usize][(t & 0xff) as usize];
                }
            }
        }
        c
    })
}

/// Sums the bit costs over a mode writer's bits.
#[allow(clippy::type_complexity)]
fn tree_cost(bit: &[[i32; 256]; 2], f: &mut dyn FnMut(&mut dyn FnMut(bool, u8) -> bool)) -> i32 {
    let mut c = 0;
    f(&mut |b, p| {
        c += bit[usize::from(b)][p as usize];
        b
    });
    c
}

/// libwebp quant_enc.c:184-212, 239-247.
#[derive(Clone, Copy, Default)]
pub(crate) struct Lambdas {
    pub i4: i64,
    pub i16: i64,
    pub uv: i64,
    pub mode: i64,
    pub ti4: i64,
    pub ti16: i64,
}

pub(crate) fn qa(m: &QuantMatrix) -> i64 {
    i64::from(m.q[0] + 15 * m.q[1] + 8) >> 4
}

pub(crate) fn new_lambdas(q: &Quant) -> Lambdas {
    let (y1, y2, uv) = (qa(&q.y1), qa(&q.y2), qa(&q.uv));
    Lambdas {
        i4: ((3 * y1 * y1) >> 7).max(1),
        i16: (3 * y2 * y2).max(1),
        uv: ((3 * uv * uv) >> 6).max(1),
        mode: ((y1 * y1) >> 7).max(1),
        ti4: ((7 * y1 * y1) >> 3).max(1),
        ti16: ((y2 * y2) >> 2).max(1),
    }
}

/// The trellis cost tables: the probabilities, level costs folded with the fixed costs, and the
/// cost of ending the block after position n at context ctx.
pub(crate) struct CostTabs {
    pub probs: [u8; 1056],
    /// [type][position][ctx][level]; ctx 3 and level > 67 unused.
    pub cost: [[[[u16; 128]; 4]; 16]; 4],
    pub eob: [[[u16; 4]; 16]; 4],
}

/// The level classes' tokens at prob offset 0: each level v's prob-coded tokens, and the distinct
/// ones with a class of each level.
struct Classes {
    toks: Vec<Vec<u16>>,
    of: [u8; 68],
}

fn classes() -> &'static Classes {
    static C: OnceLock<Classes> = OnceLock::new();
    C.get_or_init(|| {
        let mut cl = Classes {
            toks: Vec::new(),
            of: [0; 68],
        };
        let mut all = Vec::new();
        for v in 1..68 {
            all.clear();
            record_level(&mut all, v, 0);
            let t: Vec<u16> = all.iter().copied().filter(|t| t & FIXED_TOK == 0).collect();
            let k = match cl.toks.iter().position(|x| *x == t) {
                Some(k) => k,
                None => {
                    cl.toks.push(t);
                    cl.toks.len() - 1
                }
            };
            cl.of[v] = k as u8;
        }
        cl
    })
}

impl CostTabs {
    /// A table set at the default probabilities, not yet built.
    pub(crate) fn new_boxed() -> Box<CostTabs> {
        Box::new(CostTabs {
            probs: *DEFAULT_PROBS,
            cost: [[[[0; 128]; 4]; 16]; 4],
            eob: [[[0; 4]; 16]; 4],
        })
    }

    /// The tables at the default probabilities with the fixed costs of the non-libwebp path.
    pub(crate) fn defaults() -> &'static CostTabs {
        static T: OnceLock<Box<CostTabs>> = OnceLock::new();
        T.get_or_init(|| {
            let mut t = CostTabs::new_boxed();
            t.build(false);
            t
        })
    }

    /// The tables over the current probs, VP8CalculateLevelCosts. `lw` gives levels 9 and 10 the
    /// 640 libwebp's VP8LevelFixedCosts has there.
    pub(crate) fn build(&mut self, lw: bool) {
        let c = costs();
        let cl = classes();
        let mut sum = [0i32; 68];
        for typ in 0..4 {
            for n in 0..16 {
                // The first position of n's band has the same row.
                let m = (0..n).find(|&m| BANDS[m] == BANDS[n]).unwrap_or(n);
                for ctx in 0..3 {
                    if m < n {
                        let row = self.cost[typ][m][ctx];
                        self.cost[typ][n][ctx][..68].copy_from_slice(&row[..68]);
                        if n < 15 {
                            self.eob[typ][n][ctx] = self.eob_cost(typ, n, ctx);
                        }
                        continue;
                    }
                    let p = typ * 264 + (BANDS[n] as usize * 3 + ctx) * 11;
                    let c0 = if ctx > 0 {
                        c.bit[1][self.probs[p] as usize]
                    } else {
                        0
                    };
                    let t = &mut self.cost[typ][n][ctx];
                    t[0] = (c.bit[0][self.probs[p + 1] as usize] + c0) as u16;
                    for (k, toks) in cl.toks.iter().enumerate() {
                        let mut s = c.bit[1][self.probs[p + 1] as usize] + c0;
                        for &tok in toks {
                            s += c.bit[(tok >> 15) as usize]
                                [self.probs[p + (tok & 0x7ff) as usize] as usize];
                        }
                        sum[k] = s;
                    }
                    for v in 1..68 {
                        let f = if lw && (v == 9 || v == 10) {
                            640
                        } else {
                            c.fixed[v]
                        };
                        t[v] = (sum[cl.of[v] as usize] + f) as u16;
                    }
                    if n < 15 {
                        self.eob[typ][n][ctx] = self.eob_cost(typ, n, ctx);
                    }
                }
            }
        }
    }

    /// The cost of ending the block after position n at context ctx.
    fn eob_cost(&self, typ: usize, n: usize, ctx: usize) -> u16 {
        let p = self.probs[typ * 264 + (BANDS[n + 1] as usize * 3 + ctx) * 11];
        costs().bit[0][p as usize] as u16
    }

    /// The tables of one residual type for the cost kernels, with the empty-block costs of the
    /// first position's band.
    fn tab(&self, typ: usize, first: usize) -> enc_cost::Tab<'_> {
        let c = costs();
        let p0 =
            |ctx: usize| self.probs[typ * 264 + (BANDS[first] as usize * 3 + ctx) * 11] as usize;
        enc_cost::Tab {
            cost: &self.cost[typ & 3],
            eob: &self.eob[typ & 3],
            fixed: &c.fixed,
            empty: [c.bit[0][p0(0)], c.bit[0][p0(1)], c.bit[0][p0(2)]],
            nz0: c.bit[1][p0(0)],
        }
    }

    /// GetResidualCost_C over `lv[first..=last]` (last known) at context ctx, and whether any
    /// level is nonzero.
    pub(crate) fn res_cost(
        &self,
        typ: usize,
        ctx: usize,
        first: usize,
        last: i32,
        lv: &[i16; 16],
    ) -> (i32, u8) {
        (
            enc_cost::res_cost(&self.tab(typ, first), ctx, first, last, lv),
            u8::from(last >= 0),
        )
    }
}

impl CostTabs {
    /// GetResidualCost_C: the cost of `lv[first..]` at ctx, and whether any level is nonzero.
    pub(crate) fn cost_coeffs(
        &self,
        typ: usize,
        ctx: usize,
        first: usize,
        lv: &[i16; 16],
    ) -> (i32, u8) {
        self.res_cost(typ, ctx, first, last_nz(lv, first), lv)
    }

    /// VP8GetCostLuma16 over the 16 luma blocks and the DC block of an i16 MB.
    pub(crate) fn cost_luma16(&self, lv: &[[i16; 16]; 25], l: NzCtx, t: NzCtx) -> i32 {
        let bits = |a: &NzCtx| u32::from(a[0] | a[1] << 1 | a[2] << 2 | a[3] << 3);
        self.cost_coeffs(1, usize::from(l[8] + t[8]), 0, &lv[24]).0
            + enc_cost::cost_grid(&self.tab(0, 1), 1, &lv[..16], 4, bits(&l), bits(&t))
    }

    /// VP8GetCostUV over the 8 chroma blocks.
    pub(crate) fn cost_uv(&self, lv: &[[i16; 16]; 25], l: NzCtx, t: NzCtx) -> i32 {
        let bits = |a: &NzCtx, i: usize| u32::from(a[i] | a[i + 1] << 1);
        let tab = self.tab(2, 0);
        enc_cost::cost_grid(&tab, 0, &lv[16..20], 2, bits(&l, 4), bits(&t, 4))
            + enc_cost::cost_grid(&tab, 0, &lv[20..24], 2, bits(&l, 6), bits(&t, 6))
    }
}

/// The last nonzero index at or after `first`, or -1, from a SWAR nonzero mask.
pub(crate) fn last_nz(lv: &[i16; 16], first: usize) -> i32 {
    let m = nz4(lv, 0) | (nz4(lv, 4) << 4) | (nz4(lv, 8) << 8) | (nz4(lv, 12) << 12);
    15 - ((m >> first) << first).leading_zeros() as i32
}

/// Bits 0-3 are set where lv[i..i + 4] is nonzero.
fn nz4(lv: &[i16; 16], i: usize) -> u16 {
    let l = |k: usize| u64::from(lv[i + k] as u16) << (16 * k);
    let w = l(0) | l(1) | l(2) | l(3);
    // Bit 15 of a lane is set iff it is nonzero.
    let h = (((w & 0x7fff_7fff_7fff_7fff) + 0x7fff_7fff_7fff_7fff) | w) & 0x8000_8000_8000_8000;
    // The lanes land on bits 60-63.
    (h.wrapping_mul(0x0000_2000_4000_8001) >> 60) as u16
}

/// kWeightTrellis, quant_enc.c:492-499.
const TRELLIS_WEIGHT: [i64; 16] = [30, 27, 19, 11, 27, 24, 17, 10, 19, 17, 12, 8, 11, 10, 8, 6];

const MAX_COST: i64 = (1 << 55) - 1;

/// A trellis node with a signed level.
#[derive(Clone, Copy, Default)]
struct Node {
    prev: i8,
    level: i16,
}

/// VP8LevelCost on a fixed-cost-folded row.
fn level_cost(t: &[u16; 128], level: usize, f: &[i32; 2048]) -> i64 {
    let mut c = i64::from(t[level.min(67) & 127]);
    if level > 67 {
        c += i64::from(f[level & 2047] - f[67]);
    }
    c
}

impl CostTabs {
    /// TrellisQuantizeBlock (quant_enc.c:597-768) on these probs and level costs. Same
    /// conventions as `quantize`: raster in, dequantized in place, zigzag out. Returns whether
    /// the block has a nonzero level.
    pub(crate) fn trellis(
        &self,
        m: &QuantMatrix,
        in_: &mut [i16; 16],
        out: &mut [i16; 16],
        ctx0: usize,
        typ: usize,
        lambda: i64,
    ) -> bool {
        let c = costs();
        let first = usize::from(typ == 0);
        let pb = typ * 264;
        let p0 = self.probs[pb + (BANDS[first] as usize * 3 + ctx0) * 11];
        let thresh = i64::from(m.q[1] * m.q[1] / 4);
        let mut last = first as i32 - 1;
        for n in (first..16).rev() {
            let v = i64::from(in_[ZIGZAG[n] as usize]);
            if v * v > thresh {
                last = n as i32;
                break;
            }
        }
        if last < 15 {
            last += 1;
        }
        let tc = &self.cost[typ & 3];
        // Previous position's node scores and cost rows.
        let (mut ps0, mut ps1) = (0i64, 0i64);
        if ctx0 == 0 {
            ps0 = i64::from(c.bit[1][p0 as usize]) * lambda;
            ps1 = ps0;
        }
        let mut pc0 = &tc[first & 15][ctx0 & 3];
        let mut pc1 = pc0;
        let mut best = i64::from(c.bit[0][p0 as usize]) * lambda;
        let (mut path_n, mut path_d) = (-1i32, 0usize);
        let mut nodes = [[Node::default(); 2]; 16];
        for n in first..=(last as usize) {
            let j = (ZIGZAG[n & 15] & 15) as usize;
            let k = j.min(1);
            let neg = in_[j] < 0;
            let c0 = u32::from(in_[j].unsigned_abs()) + m.sharpen[j] as u32;
            let l0 = ((c0.wrapping_mul(m.iq[k]) >> 17) as usize).min(2047);
            let tl = ((c0.wrapping_mul(m.iq[k]).wrapping_add(0x80 << 9) >> 17) as usize).min(2047);
            let row = &tc[(n + 1).min(15)];
            if tl == 0 {
                // Only a level 0 node, which cannot end the block.
                let mut cs0 = ps0 + i64::from(pc0[0]) * lambda;
                let mut bp = 0i8;
                let s1 = ps1 + i64::from(pc1[0]) * lambda;
                if s1 < cs0 {
                    cs0 = s1;
                    bp = 1;
                }
                nodes[n & 15][0] = Node { prev: bp, level: 0 };
                ps0 = cs0;
                ps1 = MAX_COST;
                pc0 = &row[0];
                pc1 = &row[0];
                continue;
            }
            let nc0 = &row[l0.min(2)];
            let nc1 = &row[(l0 + 1).min(2)];
            let q = i64::from(m.q[k]);
            let w = 256 * TRELLIS_WEIGHT[j];
            let cc = i64::from(c0) * i64::from(c0);
            let eob = &self.eob[typ & 3][n & 15];
            let sign = |l: usize| if neg { -(l as i16) } else { l as i16 };

            // Node 0 always lives: tl >= l0.
            let e = i64::from(c0) - l0 as i64 * q;
            let mut cs0 = ps0 + level_cost(pc0, l0, &c.fixed) * lambda;
            let mut bp = 0i8;
            let s1 = ps1 + level_cost(pc1, l0, &c.fixed) * lambda;
            if s1 < cs0 {
                cs0 = s1;
                bp = 1;
            }
            cs0 += w * (e * e - cc);
            nodes[n & 15][0] = Node {
                prev: bp,
                level: sign(l0),
            };
            if l0 != 0 && cs0 < best {
                let t = cs0 + i64::from(eob[l0.min(2)]) * lambda;
                if t < best {
                    best = t;
                    path_n = n as i32;
                    path_d = 0;
                }
            }

            let mut cs1 = MAX_COST;
            let l1 = l0 + 1;
            if l1 <= tl {
                let e = i64::from(c0) - l1 as i64 * q;
                cs1 = ps0 + level_cost(pc0, l1, &c.fixed) * lambda;
                let mut bp = 0i8;
                let s1 = ps1 + level_cost(pc1, l1, &c.fixed) * lambda;
                if s1 < cs1 {
                    cs1 = s1;
                    bp = 1;
                }
                cs1 += w * (e * e - cc);
                nodes[n & 15][1] = Node {
                    prev: bp,
                    level: sign(l1),
                };
                if cs1 < best {
                    let t = cs1 + i64::from(eob[l1.min(2)]) * lambda;
                    if t < best {
                        best = t;
                        path_n = n as i32;
                        path_d = 1;
                    }
                }
            }
            ps0 = cs0;
            ps1 = cs1;
            pc0 = nc0;
            pc1 = nc1;
        }
        let (dc, lv0) = (in_[0], out[0]);
        *in_ = [0; 16];
        *out = [0; 16];
        if first == 1 {
            in_[0] = dc;
            out[0] = lv0;
        }
        if path_n < 0 {
            return false;
        }
        let mut d = path_d;
        for n in (first..=path_n as usize).rev() {
            let nd = nodes[n & 15][d & 1];
            let j = (ZIGZAG[n & 15] & 15) as usize;
            out[n & 15] = nd.level;
            in_[j] = nd.level.wrapping_mul(m.q[j.min(1)] as i16);
            d = nd.prev as usize;
        }
        true // the end node's level is nonzero
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    /// A random block: sparse, with levels past 67 and up to 2047 now and then.
    fn block(s: &mut u32) -> [i16; 16] {
        let mut lv = [0i16; 16];
        let density = rng(s) % 4;
        for l in &mut lv {
            let r = rng(s);
            if r % 4 > density {
                continue;
            }
            let m = match (r >> 4) % 8 {
                0..=3 => 1,
                4 => 2 + (r >> 8) % 4,
                5 => 6 + (r >> 8) % 62,
                6 => 68 + (r >> 8) % 1980,
                _ => (r >> 8) % 2048,
            } as i16;
            *l = if r & 0x10_0000 != 0 { -m } else { m };
        }
        lv
    }

    /// Tables over random probs, which include 0 and 255.
    fn random_tabs(s: &mut u32, lw: bool) -> Box<CostTabs> {
        let mut t = CostTabs::new_boxed();
        for p in &mut t.probs {
            *p = match rng(s) % 8 {
                0 => 0,
                1 => 255,
                _ => (rng(s) >> 8) as u8,
            };
        }
        t.build(lw);
        t
    }

    /// The tables before the band-row copy.
    fn build_ref(t: &mut CostTabs, lw: bool) {
        let c = costs();
        let cl = classes();
        let mut sum = [0i32; 68];
        for typ in 0..4 {
            for n in 0..16 {
                for ctx in 0..3 {
                    let p = typ * 264 + (BANDS[n] as usize * 3 + ctx) * 11;
                    let c0 = if ctx > 0 {
                        c.bit[1][t.probs[p] as usize]
                    } else {
                        0
                    };
                    let r = &mut t.cost[typ][n][ctx];
                    r[0] = (c.bit[0][t.probs[p + 1] as usize] + c0) as u16;
                    for (k, toks) in cl.toks.iter().enumerate() {
                        let mut s = c.bit[1][t.probs[p + 1] as usize] + c0;
                        for &tok in toks {
                            s += c.bit[(tok >> 15) as usize]
                                [t.probs[p + (tok & 0x7ff) as usize] as usize];
                        }
                        sum[k] = s;
                    }
                    for v in 1..68 {
                        let f = if lw && (v == 9 || v == 10) {
                            640
                        } else {
                            c.fixed[v]
                        };
                        r[v] = (sum[cl.of[v] as usize] + f) as u16;
                    }
                    if n < 15 {
                        t.eob[typ][n][ctx] = c.bit[0]
                            [t.probs[typ * 264 + (BANDS[n + 1] as usize * 3 + ctx) * 11] as usize]
                            as u16;
                    }
                }
            }
        }
    }

    #[test]
    fn build_matches_ref() {
        let mut s = 0x1357_9bdf;
        for round in 0..8 {
            let t = random_tabs(&mut s, round & 1 == 1);
            let mut r = CostTabs::new_boxed();
            r.probs = t.probs;
            build_ref(&mut r, round & 1 == 1);
            assert!(t.cost == r.cost && t.eob == r.eob, "round {round}");
        }
        let (mut d, mut r) = (CostTabs::new_boxed(), CostTabs::new_boxed());
        d.build(false);
        build_ref(&mut r, false);
        assert!(d.cost == r.cost && d.eob == r.eob);
    }

    #[test]
    fn last_nz_matches_scan() {
        let mut s = 0x2468_ace1;
        for _ in 0..20_000 {
            let lv = block(&mut s);
            for first in 0..16 {
                let want = (first..16)
                    .rev()
                    .find(|&n| lv[n] != 0)
                    .map_or(-1, |n| n as i32);
                assert_eq!(last_nz(&lv, first), want, "{lv:?} first {first}");
            }
        }
        // Every lane pattern of one word, including the sign bit and the extremes.
        for k in 0..16 {
            for v in [1, -1, 0x7fff, i16::MIN, 0x100, 0x4000] {
                let mut lv = [0i16; 16];
                lv[k] = v;
                assert_eq!(last_nz(&lv, 0), k as i32);
            }
        }
    }

    /// GetResidualCost_C as this file had it before the branch-free rewrite.
    fn res_cost_ref(
        t: &CostTabs,
        typ: usize,
        ctx: usize,
        first: usize,
        last: i32,
        lv: &[i16; 16],
    ) -> (i32, u8) {
        let c = costs();
        let base = typ * 264;
        let p0 = t.probs[base + (BANDS[first] as usize * 3 + ctx) * 11];
        if last < 0 {
            return (c.bit[0][p0 as usize], 0);
        }
        let last = last as usize;
        let mut r = 0;
        if ctx == 0 {
            r = c.bit[1][p0 as usize];
        }
        let tc = &t.cost[typ & 3];
        let mut ctx = ctx;
        let mut v = 0usize;
        for n in first..=last {
            v = usize::from(lv[n].unsigned_abs());
            if v > 67 {
                r += c.fixed[v & 2047] - c.fixed[67];
            }
            r += i32::from(tc[n][ctx & 3][v.min(67)]);
            ctx = v.min(2);
        }
        if last < 15 {
            r += i32::from(t.eob[typ & 3][last][v.min(2)]);
        }
        (r, 1)
    }

    #[test]
    fn res_cost_matches_ref() {
        let mut s = 0x0bad_cafe;
        for round in 0..6 {
            let t = if round == 0 {
                CostTabs::defaults()
            } else {
                &*Box::leak(random_tabs(&mut s, round & 1 == 1))
            };
            for _ in 0..4000 {
                let lv = block(&mut s);
                let typ = (rng(&mut s) % 4) as usize;
                let ctx = (rng(&mut s) % 3) as usize;
                let first = (rng(&mut s) % 2) as usize;
                let last = last_nz(&lv, first);
                assert_eq!(
                    t.res_cost(typ, ctx, first, last, &lv),
                    res_cost_ref(t, typ, ctx, first, last, &lv)
                );
                // A `last` that is not the last nonzero level is allowed.
                let any = (first + (rng(&mut s) as usize) % (16 - first)) as i32;
                assert_eq!(
                    t.res_cost(typ, ctx, first, any, &lv),
                    res_cost_ref(t, typ, ctx, first, any, &lv)
                );
            }
        }
    }

    /// VP8GetCostLuma16 as this file had it before the grid kernel.
    fn cost_luma16_ref(t: &CostTabs, lv: &[[i16; 16]; 25], mut l: NzCtx, mut tp: NzCtx) -> i32 {
        let cc = |typ, ctx, first, lv: &[i16; 16]| {
            res_cost_ref(t, typ, ctx, first, last_nz(lv, first), lv)
        };
        let mut r = cc(1, usize::from(l[8] + tp[8]), 0, &lv[24]).0;
        for y in 0..4 {
            for x in 0..4 {
                let (c, nz) = cc(0, usize::from(l[y] + tp[x]), 1, &lv[4 * y + x]);
                l[y] = nz;
                tp[x] = nz;
                r += c;
            }
        }
        r
    }

    /// VP8GetCostUV as this file had it before the grid kernel.
    fn cost_uv_ref(t: &CostTabs, lv: &[[i16; 16]; 25], mut l: NzCtx, mut tp: NzCtx) -> i32 {
        let mut r = 0;
        for c in (0..4).step_by(2) {
            for y in 0..2 {
                for x in 0..2 {
                    let b = &lv[16 + 2 * c + 2 * y + x];
                    let (k, nz) = res_cost_ref(
                        t,
                        2,
                        usize::from(l[4 + c + y] + tp[4 + c + x]),
                        0,
                        last_nz(b, 0),
                        b,
                    );
                    l[4 + c + y] = nz;
                    tp[4 + c + x] = nz;
                    r += k;
                }
            }
        }
        r
    }

    #[test]
    fn cost_luma16_and_uv_match_ref() {
        let mut s = 0x5eed_1234;
        for round in 0..6 {
            let t = if round == 0 {
                CostTabs::defaults()
            } else {
                &*Box::leak(random_tabs(&mut s, round & 1 == 1))
            };
            for _ in 0..1500 {
                let mut lv = [[0i16; 16]; 25];
                for b in &mut lv {
                    *b = block(&mut s);
                }
                let nz = |s: &mut u32| -> NzCtx { std::array::from_fn(|_| (rng(s) & 1) as u8) };
                let (l, tp) = (nz(&mut s), nz(&mut s));
                assert_eq!(t.cost_luma16(&lv, l, tp), cost_luma16_ref(t, &lv, l, tp));
                assert_eq!(t.cost_uv(&lv, l, tp), cost_uv_ref(t, &lv, l, tp));
            }
        }
    }

    /// The cost kernels read entry 127 of every row for the positions they skip.
    #[test]
    fn cost_zero_entry_is_zero() {
        let mut s = 0x7777;
        let (a, b) = (random_tabs(&mut s, false), random_tabs(&mut s, true));
        for t in [CostTabs::defaults(), &*a, &*b] {
            for typ in t.cost.iter() {
                for n in typ {
                    for ctx in n {
                        assert_eq!(ctx[127], 0);
                    }
                }
            }
        }
    }
}
