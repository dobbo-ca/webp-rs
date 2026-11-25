//! The macroblock encoder of the non-libwebp path: libwebp RD_OPT_BASIC with SSE only,
//! quant_enc.c:982-1196, BSD-3-Clause.

use super::libwebp::MbSrc;
use super::predict::{PRED_DC, edge_sse, idct_add, pred_mb, pred4_all, put4};
use super::rdo::{CostTabs, costs, last_nz};
use super::tokens::{NzCtx, record_chroma, record_luma};
use super::transform::{
    fwht, idct_adds_uv, idct_adds_y, iwht, residual, residuals_uv, residuals_y,
};
use super::wave::Worker;
use super::{Enc, MbInfo, flat};
use crate::simd::enc_block::rows_at;
use crate::simd::enc_pred::{residual_quant3, sse4x10};

/// Coefficient SSE between a block and its dequantized quantization.
fn coeff_sse(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&v, &w)| {
            let k = i64::from(v) - i64::from(w);
            k * k
        })
        .sum()
}

/// A bit mask of the lowest score.
fn best1(s: &[i64; 4]) -> u8 {
    let mut b = 0;
    for m in 1..4 {
        if s[m] < s[b] {
            b = m;
        }
    }
    1 << b
}

impl Enc<'_> {
    /// The neighbouring i4 modes of block n.
    pub(super) fn ctx_modes(&self, mbx: usize, mby: usize, n: usize) -> (u8, u8) {
        let mb = &self.mbs[mby * self.mbw + mbx];
        let (mut above, mut left) = (PRED_DC, PRED_DC);
        if n >= 4 {
            above = mb.modes[n - 4];
        } else if mby > 0 {
            above = self.mbs[(mby - 1) * self.mbw + mbx].modes[12 + n];
        }
        if n & 3 != 0 {
            left = mb.modes[n - 1];
        } else if mbx > 0 {
            left = self.mbs[mby * self.mbw + mbx - 1].modes[n + 3];
        }
        (above, left)
    }
}

impl Worker<'_> {
    /// The MB's source luma (stride 16) and chroma (stride 8), from the row's lines.
    pub(crate) fn mb_src(&self, mbx: usize) -> MbSrc {
        let mut s = MbSrc {
            y: [0; 256],
            u: [0; 64],
            v: [0; 64],
        };
        let (ys, cs) = (self.ys, self.cs);
        for j in 0..16 {
            let o = j * ys + mbx * 16;
            s.y[j * 16..j * 16 + 16].copy_from_slice(&self.y[o..o + 16]);
        }
        for j in 0..8 {
            let o = j * cs + mbx * 8;
            s.u[j * 8..j * 8 + 8].copy_from_slice(&self.cb[o..o + 8]);
            s.v[j * 8..j * 8 + 8].copy_from_slice(&self.cr[o..o + 8]);
        }
        s
    }

    /// Copies the reconstruction in z over the MB's source, as later MBs predict from it.
    pub(crate) fn store_mb(&mut self, mbx: usize, chroma: bool) {
        let (ys, cs) = (self.ys, self.cs);
        for j in 0..16 {
            let o = j * ys + mbx * 16;
            self.y[o..o + 16].copy_from_slice(&self.z[1 + j][8..24]);
        }
        if chroma {
            for j in 0..8 {
                let o = j * cs + mbx * 8;
                self.cb[o..o + 8].copy_from_slice(&self.z[18 + j][8..16]);
                self.cr[o..o + 8].copy_from_slice(&self.z[18 + j][24..32]);
            }
        }
    }

    /// The neighbouring i4 modes of block n, the ones above the MB from the row above.
    fn ctx_modes(&self, mbx: usize, n: usize) -> (u8, u8) {
        let mb = &self.mbs[mbx];
        let above = if n >= 4 {
            mb.modes[n - 4]
        } else {
            self.above[n]
        };
        let mut left = PRED_DC;
        if n & 3 != 0 {
            left = mb.modes[n - 1];
        } else if mbx > 0 {
            left = self.mbs[mbx - 1].modes[n + 3];
        }
        (above, left)
    }

    /// Codes the luma residual after pred_mb into dq and returns the transform-domain distortion.
    /// Trellis contexts follow libwebp quant_enc.c:772-781.
    fn i16_quant(
        &mut self,
        dq: &mut [[i16; 16]; 16],
        src: &[u8; 256],
        mut l: NzCtx,
        mut t: NzCtx,
        tr: bool,
    ) -> i64 {
        let mut dc = [0i16; 16];
        let mut d = 0i64;
        residuals_y(&self.z, dq, src);
        let ct = CostTabs::defaults();
        for n in 0..16 {
            let b = &mut dq[n];
            dc[n] = b[0];
            b[0] = 0;
            if !tr {
                d += self.q.y1.quantize_sse(b, &mut self.lv[n]).1;
            } else {
                let raw = *b;
                let nz = ct.trellis(
                    &self.q.y1,
                    b,
                    &mut self.lv[n],
                    usize::from(l[n / 4] + t[n % 4]),
                    0,
                    self.lam.ti16,
                );
                self.lv[n][0] = 0;
                l[n / 4] = u8::from(nz);
                t[n % 4] = u8::from(nz);
                d += coeff_sse(&raw, b);
            }
        }
        let raw = dc;
        fwht(&mut dc);
        self.q.y2.quantize(&mut dc, &mut self.lv[24]);
        let q = iwht(&dc);
        for n in 0..16 {
            dq[n][0] = q[n];
            let k = i64::from(raw[n]) - i64::from(q[n]);
            d += k * k;
        }
        // fdct gain is 2, so coefficient SSE is 4x pixel SSE.
        (d + 2) >> 2
    }

    /// Ranks the i4 modes by prediction SSE plus mode rate and returns a bit mask of the 3 best.
    /// It saves every prediction in pr so trials skip recomputing them. Methods 5 and 6 try every
    /// mode.
    fn i4_shortlist(
        &self,
        pr: &mut [[u32; 4]; 10],
        rows: &[u32; 4],
        y: usize,
        x: usize,
        above: u8,
        lft: u8,
    ) -> u16 {
        pred4_all(&self.z, pr, y, x);
        if self.sh.method >= 5 {
            return (1 << 10) - 1;
        }
        let c = &costs().i4[above as usize][lft as usize];
        let li = self.lam.i4;
        let d = sse4x10(pr, rows);
        let (mut a, mut b, mut k) = (i64::MAX, i64::MAX, i64::MAX);
        for m in 0..10 {
            // The mode in the low bits keeps ties stable.
            let mut v = ((256 * i64::from(d[m]) + i64::from(c[m]) * li) << 4) | m as i64;
            (v, a) = (a.max(v), a.min(v));
            (v, b) = (b.max(v), b.min(v));
            k = k.min(v);
        }
        (1 << (a & 15)) | (1 << (b & 15)) | (1 << (k & 15))
    }

    /// Quantizes one i4 block, with trellis when tr. Returns the last nonzero index, or -1.
    fn quant4(&mut self, blk: &mut [i16; 16], lv: &mut [i16; 16], ctx: usize, tr: bool) -> i32 {
        if tr {
            CostTabs::defaults().trellis(&self.q.y1, blk, lv, ctx, 3, self.lam.ti4);
            return last_nz(lv, 0);
        }
        self.q.y1.quantize(blk, lv)
    }

    /// Codes the chroma residual after pred_mb into dq and returns the transform-domain
    /// distortion.
    fn uv_quant(&mut self, dq: &mut [[i16; 16]; 8], u: &[u8; 64], v: &[u8; 64]) -> i64 {
        let mut d = 0;
        residuals_uv(&self.z, dq, u, v);
        for c in 0..8 {
            d += self.q.uv.quantize_sse(&mut dq[c], &mut self.lv[16 + c]).1;
        }
        (d + 2) >> 2
    }

    /// Whether DC predicts the MB exactly.
    fn dc_exact(&mut self, src: &MbSrc, mbx: usize, mby: usize) -> bool {
        if !flat(&src.y, 0, 16, 16, 16, 1)
            || !flat(&src.u, 0, 8, 8, 8, 1)
            || !flat(&src.v, 0, 8, 8, 8, 1)
        {
            return false;
        }
        pred_mb(&mut self.z, 16, 1, 8, PRED_DC, mbx, mby);
        pred_mb(&mut self.z, 8, 18, 8, PRED_DC, mbx, mby);
        pred_mb(&mut self.z, 8, 18, 24, PRED_DC, mbx, mby);
        self.z[1][8] == src.y[0] && self.z[18][8] == src.u[0] && self.z[18][24] == src.v[0]
    }

    /// Encodes one MB and returns whether it is skipped.
    pub(crate) fn encode_mb(&mut self, mbx: usize, mby: usize, left: &mut NzCtx) -> bool {
        let c = costs();
        let ct = CostTabs::defaults();
        self.prepare(mbx, mby);
        let src = self.mb_src(mbx);
        let method = self.sh.method;
        let idx = mbx;
        let seg = self.mbs[idx].seg;
        self.q = self.sh.seg_q[seg as usize];
        self.lam = self.sh.seg_lam[seg as usize];
        let lam = self.lam;
        // Trials match the final pass except at method 5.
        let reuse = method != 5;

        if self.dc_exact(&src, mbx, mby) {
            let mb = &mut self.mbs[idx];
            *mb = MbInfo {
                skip: true,
                i16: true,
                modes: [PRED_DC; 16],
                uv: 0,
                seg,
            };
            *left = [0; 9];
            self.top = [0; 9];
            return true;
        }

        let (mut best16, mut bs, mut s16) = (0u8, -1i64, 0i64);
        // Trials alternate buffers so the best is never copied.
        let mut dq16 = [[[0i16; 16]; 16]; 2];
        let mut b16 = 0usize;
        let mut lv16 = [[0i16; 16]; 17];
        let mut try16 = 15u8;
        if method < 5 {
            let mut s = edge_sse(&self.z, &src.y, 16, 16, 1, 8, mbx, mby);
            for m in 0..4 {
                s[m] = 256 * s[m] + i64::from(c.i16[m]) * lam.i16;
            }
            try16 = best1(&s);
        }
        for m in 0..4u8 {
            if try16 & (1 << m) == 0 {
                continue;
            }
            pred_mb(&mut self.z, 16, 1, 8, m, mbx, mby);
            let d = self.i16_quant(&mut dq16[b16 ^ 1], &src.y, *left, self.top, method >= 6);
            let r =
                i64::from(ct.cost_luma16(&self.lv, *left, self.top)) + i64::from(c.i16[m as usize]);
            let s = r * lam.i16 + 256 * d;
            if bs < 0 || s < bs {
                best16 = m;
                bs = s;
                s16 = r * lam.mode + 256 * d;
                b16 ^= 1;
                if reuse {
                    lv16[..16].copy_from_slice(&self.lv[..16]);
                    lv16[16] = self.lv[24];
                }
            }
        }

        let mut i16m = false;
        let mut score = i64::from(c.bit[0][145]) * lam.mode;
        let (mut l, mut t) = (*left, self.top);
        // pred4_all rewrites all of pr.
        let mut pr = [[0u32; 4]; 10];
        for n in 0..16 {
            let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
            let s4 = &src.y[(y - 1) * 16 + x - 8..];
            let (above, lft) = self.ctx_modes(mbx, n);
            let ctx = usize::from(l[n / 4] + t[n % 4]);
            let (mut best, mut bs, mut b_score, mut b_last) = (0u8, -1i64, 0i64, -1);
            let (mut dq4, mut lv4) = ([0i16; 16], [0i16; 16]);
            let (mut blk, mut lv) = ([0i16; 16], [0i16; 16]);
            let rows = rows_at(s4, 16, 0);
            let try4 = self.i4_shortlist(&mut pr, &rows, y, x, above, lft);
            let mut tm = try4;
            if method < 5 {
                // Go's batched residualQuant3: the three shortlisted modes in one kernel call.
                debug_assert_eq!(try4.count_ones(), 3);
                let modes: [usize; 3] = std::array::from_fn(|_| {
                    let m = tm.trailing_zeros() as usize;
                    tm &= tm.wrapping_sub(1);
                    m
                });
                let (mut b3, mut l3) = ([[0i16; 16]; 3], [[0i16; 16]; 3]);
                let r3 = residual_quant3(&self.q.y1, &pr, modes, &rows, &mut b3, &mut l3);
                let mut kb = 0;
                for (k, &m) in modes.iter().enumerate() {
                    // fdct gain is 2, so coefficient SSE is 4x pixel SSE.
                    let d = (r3[k].1 + 2) >> 2;
                    let cost = i64::from(ct.res_cost(3, ctx, 0, r3[k].0, &l3[k]).0);
                    let r = cost + i64::from(c.i4[above as usize][lft as usize][m]);
                    let s = r * lam.i4 + 256 * d;
                    if bs < 0 || s < bs {
                        best = m as u8;
                        bs = s;
                        b_score = r * lam.mode + 256 * d;
                        kb = k;
                    }
                }
                dq4 = b3[kb];
                lv4 = l3[kb];
                b_last = r3[kb].0;
            }
            while tm != 0 {
                let m = tm.trailing_zeros() as u8;
                tm &= tm - 1;
                let (last, d);
                if method >= 6 {
                    put4(&mut self.z, &pr[m as usize], y, x);
                    residual(&self.z, &mut blk, s4, 16, y, x);
                    let raw = blk;
                    last = self.quant4(&mut blk, &mut lv, ctx, true);
                    d = coeff_sse(&raw, &blk);
                } else {
                    put4(&mut self.z, &pr[m as usize], y, x);
                    residual(&self.z, &mut blk, s4, 16, y, x);
                    let (la, dd) = self.q.y1.quantize_sse(&mut blk, &mut lv);
                    last = la;
                    d = dd;
                }
                // fdct gain is 2, so coefficient SSE is 4x pixel SSE.
                let d = (d + 2) >> 2;
                let cost = i64::from(ct.res_cost(3, ctx, 0, last, &lv).0);
                let r = cost + i64::from(c.i4[above as usize][lft as usize][m as usize]);
                let s = r * lam.i4 + 256 * d;
                if bs < 0 || s < bs {
                    best = m;
                    bs = s;
                    b_score = r * lam.mode + 256 * d;
                    b_last = last;
                    dq4 = blk;
                    lv4 = lv;
                }
            }
            score += b_score;
            if score >= s16 {
                i16m = true;
                break;
            }
            put4(&mut self.z, &pr[best as usize], y, x);
            let last;
            if reuse {
                idct_add(&mut self.z, &dq4, y, x);
                self.lv[n] = lv4;
                last = b_last;
            } else {
                residual(&self.z, &mut blk, s4, 16, y, x);
                let mut lvn = [0i16; 16];
                last = self.quant4(&mut blk, &mut lvn, ctx, method >= 5);
                self.lv[n] = lvn;
                idct_add(&mut self.z, &blk, y, x);
            }
            self.mbs[idx].modes[n] = best;
            // `last` is -1 exactly when all levels are zero.
            let nz = u8::from(last >= 0);
            l[n / 4] = nz;
            t[n % 4] = nz;
        }

        if i16m {
            pred_mb(&mut self.z, 16, 1, 8, best16, mbx, mby);
            if reuse {
                self.lv[..16].copy_from_slice(&lv16[..16]);
                self.lv[24] = lv16[16];
            } else {
                self.i16_quant(&mut dq16[b16], &src.y, *left, self.top, method >= 5);
            }
            idct_adds_y(&mut self.z, &dq16[b16]);
            self.mbs[idx].modes = [best16; 16];
        }

        let mut best_uv = 0u8;
        if self.sh.gray && method < 5 {
            // Gray chroma is flat 128; DC prediction needs no trial or residual.
            pred_mb(&mut self.z, 8, 18, 8, PRED_DC, mbx, mby);
            pred_mb(&mut self.z, 8, 18, 24, PRED_DC, mbx, mby);
            self.lv[16..24].fill([0; 16]);
        } else if method < 5 {
            let s = edge_sse(&self.z, &src.u, 8, 8, 18, 8, mbx, mby);
            let sv = edge_sse(&self.z, &src.v, 8, 8, 18, 24, mbx, mby);
            let mut sc = [0i64; 4];
            for m in 0..4 {
                sc[m] = 256 * (s[m] + sv[m]) + i64::from(c.uv[m]) * lam.uv;
            }
            // Single trial: RD score unused.
            best_uv = best1(&sc).trailing_zeros() as u8;
            pred_mb(&mut self.z, 8, 18, 8, best_uv, mbx, mby);
            pred_mb(&mut self.z, 8, 18, 24, best_uv, mbx, mby);
            let mut dq = [[0i16; 16]; 8];
            residuals_uv(&self.z, &mut dq, &src.u, &src.v);
            self.q.uv.quantize_n(&mut dq, &mut self.lv[16..24]);
            idct_adds_uv(&mut self.z, &dq);
        } else {
            let mut s_uv = -1i64;
            let mut dq_uv = [[[0i16; 16]; 8]; 2];
            let mut lv_uv = [[0i16; 16]; 8];
            let mut b_uv = 0usize;
            for m in 0..4u8 {
                pred_mb(&mut self.z, 8, 18, 8, m, mbx, mby);
                pred_mb(&mut self.z, 8, 18, 24, m, mbx, mby);
                let d = self.uv_quant(&mut dq_uv[b_uv ^ 1], &src.u, &src.v);
                let r =
                    i64::from(ct.cost_uv(&self.lv, *left, self.top)) + i64::from(c.uv[m as usize]);
                let s = r * lam.uv + 256 * d;
                if s_uv < 0 || s < s_uv {
                    best_uv = m;
                    s_uv = s;
                    b_uv ^= 1;
                    lv_uv.copy_from_slice(&self.lv[16..24]);
                }
            }
            pred_mb(&mut self.z, 8, 18, 8, best_uv, mbx, mby);
            pred_mb(&mut self.z, 8, 18, 24, best_uv, mbx, mby);
            idct_adds_uv(&mut self.z, &dq_uv[b_uv]);
            self.lv[16..24].copy_from_slice(&lv_uv);
        }

        let mut skip = !i16m || self.lv[24].iter().all(|&v| v == 0);
        for n in 0..24 {
            if !skip {
                break;
            }
            skip = self.lv[n].iter().all(|&v| v == 0);
        }
        if skip {
            if i16m {
                *left = [0; 9];
                self.top = [0; 9];
            } else {
                for i in 0..8 {
                    left[i] = 0;
                    self.top[i] = 0;
                }
            }
        } else {
            record_luma(self.tok, i16m, &self.lv, left, &mut self.top);
            record_chroma(self.tok, &self.lv, left, &mut self.top);
        }

        // Gray chroma is already flat 128; no copy-back needed.
        self.store_mb(mbx, !self.sh.gray || method >= 5);
        let modes = self.mbs[idx].modes;
        self.mbs[idx] = MbInfo {
            skip,
            i16: i16m,
            modes,
            uv: best_uv,
            seg,
        };
        skip
    }
}
