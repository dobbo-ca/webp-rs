//! `match_libwebp` mode: libwebp 1.6.0's RD_OPT_BASIC path (methods 3 to 6) ported as is, so the
//! output bytes equal cwebp's. Sources are libwebp quant_enc.c (VP8Decimate), frame_enc.c
//! (VP8EncTokenLoop), cost_enc.c and filter_enc.c, BSD-3-Clause.

use super::predict::{PRED_DC, Ybr, idct_add, pred_mb, pred_sse, pred4_all, put4};
use super::quant::QuantMatrix;
use super::rdo::{CostTabs, costs, last_nz, qa};
use super::segment::Segs;
use super::tokens::{DEFAULT_PROBS, NzCtx, UPDATE_PROBS, record_chroma, record_luma};
use super::transform::{
    fwht_plain, idct_adds_uv, idct_adds_y, iwht, residual, residuals_uv, residuals_y,
};
use super::wave::Worker;
use super::{Enc, MbInfo};
use crate::simd::enc_i4::{i4_trial1, i4_trials, rec_metrics, rec_metrics16, tt_mb};

/// libwebp's running state for one encode.
pub(crate) struct LwState {
    /// proba->coeffs and its costs, with VP8LevelFixedCosts.
    pub tabs: Box<CostTabs>,
    /// proba->stats: total << 16 | ones; slot 1056 takes the fixed-probability bits.
    stats: Vec<u32>,
    cnt: i64,
    max: i64,
    pub max_edge: [i64; 4],
    diffuse: bool,
    /// Equal predictions share a trial (Go's lwDedup); tests turn it off to compare.
    dedup: bool,
}

#[cfg(test)]
thread_local! {
    /// Makes `LwState::new` of this thread's encodes start with the dedup off.
    pub(super) static NO_DEDUP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// VP8FixedCostsUV, cost_enc.c:103.
const UV_COST_LW: [i64; 4] = [302, 984, 439, 642];
/// FLATNESS_PENALTY.
const FLAT_PENALTY: i64 = 140;
/// libwebp's stats slot of a token's prob index, with FIXED_TOK at bit 11 after the shift: prob 10
/// counts in slot 9 (cat5/cat6 bit, token_enc.c:172,178), the fixed-prob bits in the unused 1056.
static STAT_SLOT: [u16; 4096] = {
    let mut t = [0u16; 4096];
    let mut i = 0;
    while i < 4096 {
        t[i] = if i >= 1056 {
            1056
        } else if i % 11 == 10 {
            i as u16 - 1
        } else {
            i as u16
        };
        i += 1;
    }
    t
};

fn mult8b(a: i64, b: i64) -> i64 {
    (a * b + 128) >> 8
}

/// SetRDScore.
fn rd_score(lambda: i64, r: i64, h: i64, d: i64, sd: i64) -> i64 {
    (r + h) * lambda + 256 * (d + sd)
}

/// QuantizeSingle: quantizes *v and returns err >> 1.
fn quant_single(v: &mut i16, m: &QuantMatrix) -> i32 {
    let mut vv = i32::from(*v);
    let neg = vv < 0;
    if neg {
        vv = -vv;
    }
    let q_v = (((vv as u32).wrapping_mul(m.iq[0]).wrapping_add(m.bias[0]) >> 17) as i32)
        .wrapping_mul(m.q[0] as i32);
    let mut err = vv - q_v;
    if neg {
        *v = (-q_v) as i16;
        err = -err;
    } else {
        *v = q_v as i16;
    }
    err >> 1
}

/// CorrectDCValues on the 8 chroma blocks, with the errors diffused from above (`top`, [ch][top0,
/// top1]) and from the left; the corrected DCs go to dc.
fn correct_dc(
    m: &QuantMatrix,
    dq: &[[i16; 16]; 8],
    dc: &mut [i16; 8],
    top: &[[i8; 2]; 2],
    left: &[[i8; 2]; 2],
) -> [[i8; 3]; 2] {
    let mut derr = [[0i8; 3]; 2];
    for ch in 0..2 {
        let (top, left) = (&top[ch], &left[ch]);
        let c = &dq[4 * ch..4 * ch + 4];
        let d = &mut dc[4 * ch..4 * ch + 4];
        d[0] = c[0][0].wrapping_add(((7 * i32::from(top[0]) + 8 * i32::from(left[0])) >> 3) as i16);
        let e0 = quant_single(&mut d[0], m);
        d[1] = c[1][0].wrapping_add(((7 * i32::from(top[1]) + 8 * e0) >> 3) as i16);
        let e1 = quant_single(&mut d[1], m);
        d[2] = c[2][0].wrapping_add(((7 * e0 + 8 * i32::from(left[1])) >> 3) as i16);
        let e2 = quant_single(&mut d[2], m);
        d[3] = c[3][0].wrapping_add(((7 * e1 + 8 * e2) >> 3) as i16);
        let e3 = quant_single(&mut d[3], m);
        derr[ch] = [e1 as i8, e2 as i8, e3 as i8];
    }
    derr
}

/// StoreDiffusionErrors into the errors passed down (`top`) and to the right (`left`).
fn store_derr(derr: &[[i8; 3]; 2], top: &mut [[i8; 2]; 2], left: &mut [[i8; 2]; 2]) {
    for ch in 0..2 {
        let l = &mut left[ch];
        let t = &mut top[ch];
        l[0] = derr[ch][0];
        l[1] = ((3 * i32::from(derr[ch][2])) >> 2) as i8;
        t[0] = derr[ch][1];
        t[1] = (i32::from(derr[ch][2]) - i32::from(l[1])) as i8;
    }
}

/// IsFlat_C over the chroma blocks whose bit is set in nz: whether they have at most `thresh`
/// nonzero AC levels in all.
fn is_flat(lv: &[[i16; 16]], nz: u32, thresh: usize) -> bool {
    let mut k = 0;
    for (i, b) in lv.iter().enumerate() {
        if (nz >> i) & 1 == 0 {
            continue;
        }
        k += b[1..].iter().filter(|&&v| v != 0).count();
    }
    k <= thresh
}

/// One i16 or chroma trial's result. `r`, `flat` and the chroma `derr` are filled on first use.
#[derive(Clone, Copy, Default)]
struct Trial {
    nz: u32,
    d: i64,
    sd: i64,
    r: i64,
    derr: [[i8; 3]; 2],
    r_ok: bool,
    flat: bool,
    flat_ok: bool,
}

/// encodeMBLW's scratch, as Go's lwScratch, one per worker. Every element read is written first,
/// so it is never cleared. `p16` and `p_uv` hold the predictions of the modes tried, for the
/// dedup.
#[derive(Default)]
pub(crate) struct LwScratch {
    dq16: [[i16; 16]; 16],
    blv: [[i16; 16]; 25],
    brec: [[u8; 16]; 16],
    lv_uv: [[i16; 16]; 8],
    /// The best chroma reconstruction: per row, U then V.
    buv: [[u8; 8]; 16],
    pr: [[u32; 4]; 10],
    lv: [[i16; 16]; 10],
    rec: [[u32; 4]; 10],
    res: [[i32; 4]; 10],
    p16: [[[u8; 16]; 16]; 4],
    p_uv: [[[u8; 16]; 8]; 4],
}

#[cfg(test)]
impl LwScratch {
    /// Fills every element with garbage derived from `k`.
    pub(crate) fn poison(&mut self, k: u8) {
        let b = |i: usize| k.wrapping_add((i as u8).wrapping_mul(53));
        let w = |i: usize| i16::from(b(i)) * 97 - 9000;
        for (i, v) in self.dq16.as_flattened_mut().iter_mut().enumerate() {
            *v = w(i);
        }
        for (i, v) in self.blv.as_flattened_mut().iter_mut().enumerate() {
            *v = w(i + 1);
        }
        for (i, v) in self.brec.as_flattened_mut().iter_mut().enumerate() {
            *v = b(i);
        }
        for (i, v) in self.lv_uv.as_flattened_mut().iter_mut().enumerate() {
            *v = w(i + 2);
        }
        for (i, v) in self.buv.as_flattened_mut().iter_mut().enumerate() {
            *v = b(i + 3);
        }
        for (i, v) in self.pr.as_flattened_mut().iter_mut().enumerate() {
            *v = (u32::from(b(i)) * 0x0101_0101) ^ 0x5a5a;
        }
        for (i, v) in self.lv.as_flattened_mut().iter_mut().enumerate() {
            *v = w(i + 4);
        }
        for (i, v) in self.rec.as_flattened_mut().iter_mut().enumerate() {
            *v = u32::from(b(i + 5)).wrapping_mul(0x0103_0507);
        }
        for (i, v) in self.res.as_flattened_mut().iter_mut().enumerate() {
            *v = i32::from(w(i + 6));
        }
        for (i, v) in self
            .p16
            .as_flattened_mut()
            .as_flattened_mut()
            .iter_mut()
            .enumerate()
        {
            *v = b(i + 7);
        }
        for (i, v) in self
            .p_uv
            .as_flattened_mut()
            .as_flattened_mut()
            .iter_mut()
            .enumerate()
        {
            *v = b(i + 8);
        }
    }
}

/// The 16x16 luma prediction in z.
fn pred_y(z: &Ybr, p: &mut [[u8; 16]; 16]) {
    for j in 0..16 {
        p[j].copy_from_slice(&z[1 + j][8..24]);
    }
}

/// The 8x8 U and V predictions in z, one row of each per row.
fn pred_uv(z: &Ybr, p: &mut [[u8; 16]; 8]) {
    for j in 0..8 {
        p[j][..8].copy_from_slice(&z[18 + j][8..16]);
        p[j][8..].copy_from_slice(&z[18 + j][24..32]);
    }
}

/// The rows of luma block n of the source, as four little-endian words.
fn src4(src: &[u8; 256], n: usize) -> [u32; 4] {
    std::array::from_fn(|j| {
        u32::from_le_bytes(src.as_chunks::<4>().0[(4 * (n / 4) + j) * 4 + n % 4])
    })
}

/// The best of the ten i4 modes of one block: the mode, distortion, spectral distortion, rate and
/// header rate.
struct I4Pick {
    best: usize,
    d: i64,
    sd: i64,
    r: i64,
    h: i64,
}

impl LwState {
    pub(crate) fn new(mbw: usize, mbh: usize, quality: f32) -> Box<LwState> {
        let mut s = Box::new(LwState {
            tabs: CostTabs::new_boxed(),
            stats: vec![0; 1057],
            cnt: 0,
            max: 0,
            max_edge: [0; 4],
            diffuse: false,
            #[cfg(test)]
            dedup: !NO_DEDUP.get(),
            #[cfg(not(test))]
            dedup: true,
        });
        s.reset(mbw, mbh, quality);
        s
    }

    /// The bytes a state takes: `this`'s buffers at their capacity, or those of a new one. The
    /// workers' scratch is charged with the workers.
    pub(crate) fn bytes(this: Option<&LwState>) -> usize {
        let stats = this.map_or(1057, |l| l.stats.capacity());
        size_of::<LwState>() + size_of::<CostTabs>() + stats * size_of::<u32>()
    }

    /// Starts a new encode with this state's buffers (Go's lwState.reset).
    pub(crate) fn reset(&mut self, mbw: usize, mbh: usize, quality: f32) {
        self.tabs.probs = *DEFAULT_PROBS;
        self.tabs.build(true);
        self.stats.fill(0);
        self.max = ((mbw * mbh) >> 3).max(96) as i64; // MIN_COUNT
        self.cnt = self.max;
        self.max_edge = [0; 4];
        self.diffuse = quality <= 98.0; // ERROR_DIFFUSION_QUALITY
    }

    /// FinalizeTokenProbas; it reports a prob off its default.
    pub(crate) fn finalize(&mut self) -> bool {
        let c = costs();
        let mut changed = false;
        for i in 0..1056 {
            let (nb, total) = (
                i64::from(self.stats[i] & 0xffff),
                i64::from(self.stats[i] >> 16),
            );
            let np = if nb != 0 { 255 - nb * 255 / total } else { 255 };
            let bc = |q: usize| nb * i64::from(c.bit[1][q]) + (total - nb) * i64::from(c.bit[0][q]);
            let old = bc(DEFAULT_PROBS[i] as usize) + i64::from(c.bit[0][UPDATE_PROBS[i] as usize]);
            let nw = bc(np as usize) + i64::from(c.bit[1][UPDATE_PROBS[i] as usize]) + 8 * 256;
            self.tabs.probs[i] = DEFAULT_PROBS[i];
            if old > nw {
                self.tabs.probs[i] = np as u8;
                changed = changed || np as u8 != DEFAULT_PROBS[i];
            }
        }
        changed
    }

    /// Runs before each MB, frame_enc.c:832-837.
    pub(crate) fn refresh(&mut self) {
        self.cnt -= 1;
        if self.cnt < 0 {
            if self.finalize() {
                self.tabs.build(true);
            }
            self.cnt = self.max;
        }
    }

    /// `refresh` for the MB at a segment's start, then the MBs from there, at most `left`, that
    /// share the tables it leaves (the next refresh that fires starts the next segment). `cnt`
    /// is then as after the last of them, so the MBs between take no `refresh` of their own.
    pub(crate) fn segment(&mut self, left: usize) -> usize {
        self.refresh();
        // cnt is 0 to max here.
        let n = usize::try_from(self.cnt + 1).map_or(1, |n| n.clamp(1, left.max(1)));
        self.cnt -= n as i64 - 1;
        n
    }

    /// VP8RecordStats over one MB's tokens. libwebp counts the cat5/cat6 bit (prob 10) in slot 9,
    /// token_enc.c:172,178.
    pub(crate) fn record(&mut self, toks: &[u16]) {
        for &v in toks {
            let i = usize::from(STAT_SLOT[usize::from((v & 0x7ff) | ((v >> 3) & 0x800))]);
            let mut x = self.stats[i];
            if x >= 0xfffe_0000 {
                x = ((x + 1) >> 1) & 0x7fff_7fff;
            }
            self.stats[i] = x + 0x10000 + u32::from(v >> 15);
        }
    }

    /// The rate of one i4 block's levels with last nonzero index `last`.
    fn rate_i4(&self, last: i32, lv: &[i16; 16], ctx: usize) -> i64 {
        if last < 0 {
            // No levels: the EOB-only cost.
            i64::from(costs().bit[0][self.tabs.probs[3 * 264 + ctx * 11] as usize])
        } else {
            i64::from(self.tabs.res_cost(3, ctx, 0, last, lv).0)
        }
    }

    /// pickI4Same: `pick_i4` when every mode has mode 0's trial, read from index 0 only. The
    /// scores then differ only in header cost plus flatness penalty, times a lambda of at least
    /// 1, so the pick is their first argmin, costed once.
    fn pick_i4_same(
        &self,
        res: &[[i32; 4]; 10],
        lv: &[[i16; 16]; 10],
        hc: &[i32; 10],
        st: i64,
        ctx: usize,
        tlambda: i64,
    ) -> I4Pick {
        let pen = if res[0][3] <= 3 { FLAT_PENALTY } else { 0 };
        let (mut best, mut bh) = (0usize, i64::from(hc[0]));
        for m in 1..10 {
            let k = i64::from(hc[m]) + pen;
            if k < bh {
                (best, bh) = (m, k);
            }
        }
        let mut r = self.rate_i4(res[0][0], &lv[0], ctx);
        if best > 0 {
            r += pen;
        }
        I4Pick {
            best,
            d: i64::from(res[0][1]),
            sd: mult8b(tlambda, (i64::from(res[0][2]) - st).abs() >> 5),
            r,
            h: i64::from(hc[best]),
        }
    }

    /// PickBestIntra4's mode choice over the ten trials `res` ([last, SSE, TTransform, nonzero AC]).
    fn pick_i4(
        &self,
        res: &[[i32; 4]; 10],
        lv: &[[i16; 16]; 10],
        hc: &[i32; 10],
        st: i64,
        ctx: usize,
        lambda: i64,
        tlambda: i64,
    ) -> I4Pick {
        let (mut best, mut bs, mut bc) = (0usize, 0i64, 0i64);
        for m in 0..10 {
            let sd = mult8b(tlambda, (i64::from(res[m][2]) - st).abs() >> 5);
            let mut r = 0;
            if m > 0 && res[m][3] <= 3 {
                r = FLAT_PENALTY;
            }
            let mut k = rd_score(lambda, r, i64::from(hc[m]), i64::from(res[m][1]), sd);
            if m > 0 && k >= bs {
                continue;
            }
            let rate = self.rate_i4(res[m][0], &lv[m], ctx);
            k += rate * lambda;
            if m == 0 || k < bs {
                best = m;
                bs = k;
                bc = rate;
            }
        }
        let sd = mult8b(tlambda, (i64::from(res[best][2]) - st).abs() >> 5);
        let mut r = bc;
        if best > 0 && res[best][3] <= 3 {
            r += FLAT_PENALTY;
        }
        I4Pick {
            best,
            d: i64::from(res[best][1]),
            sd,
            r,
            h: i64::from(hc[best]),
        }
    }
}

/// A pointer-free copy of the MB's source: luma stride 16, chroma stride 8.
pub(crate) struct MbSrc {
    pub y: [u8; 256],
    pub u: [u8; 64],
    pub v: [u8; 64],
}

impl Worker<'_> {
    /// One i16 trial on the prediction in z: levels into self.lv, the dequantized coefficients
    /// into dq, which are added to z. Sets t.nz, t.d and t.sd.
    fn i16_trial(
        &mut self,
        t: &mut Trial,
        lw: &LwState,
        dq: &mut [[i16; 16]; 16],
        left: &NzCtx,
        src: &MbSrc,
        st: &[i32; 16],
        tlambda: i64,
    ) {
        t.nz = if self.sh.method >= 6 {
            self.lw_i16_t(lw, dq, *left, self.top, &src.y)
        } else {
            residuals_y(&self.z, dq, &src.y);
            self.quant_i16(dq)
        };
        let (sse, sd) = rec_metrics16(dq, &mut self.z, &src.y, st);
        t.d = i64::from(sse);
        t.sd = mult8b(tlambda, i64::from(sd));
    }

    /// One chroma trial on the prediction in z: levels into self.lv[16..24], the reconstruction
    /// into z. Sets t.nz, t.derr and t.d.
    fn uv_trial(&mut self, t: &mut Trial, lw: &LwState, src: &MbSrc) {
        (t.nz, t.derr) = self.lw_uv(lw, src);
        t.d =
            pred_sse(&self.z, &src.u, 8, 18, 8, 8, 8) + pred_sse(&self.z, &src.v, 8, 18, 24, 8, 8);
    }

    /// quantI16 after the residuals: the DC block's WHT, then the AC levels. Returns nz as
    /// libwebp packs it (bit 24 for the DC block).
    fn quant_i16(&mut self, dq: &mut [[i16; 16]; 16]) -> u32 {
        let mut dc = [0i16; 16];
        for n in 0..16 {
            dc[n] = dq[n][0];
            dq[n][0] = 0;
        }
        fwht_plain(&mut dc);
        let y2 = self.q.y2.quantize(&mut dc, &mut self.lv[24]) >= 0;
        dc = iwht(&dc);
        let mut nz = self.q.y1.quantize_n(dq, &mut self.lv[..16]);
        for n in 0..16 {
            dq[n][0] = dc[n];
        }
        if y2 {
            nz |= 1 << 24;
        }
        nz
    }

    /// lwI16 with trellis on the AC blocks, contexts from l and t as ReconstructIntra16 updates
    /// them.
    fn lw_i16_t(
        &mut self,
        lw: &LwState,
        dq: &mut [[i16; 16]; 16],
        mut l: NzCtx,
        mut t: NzCtx,
        src: &[u8; 256],
    ) -> u32 {
        let mut dc = [0i16; 16];
        let mut nz = 0u32;
        for n in 0..16 {
            let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
            residual(&self.z, &mut dq[n], &src[(y - 1) * 16 + x - 8..], 16, y, x);
            dc[n] = dq[n][0];
        }
        fwht_plain(&mut dc);
        if self.q.y2.quantize(&mut dc, &mut self.lv[24]) >= 0 {
            nz |= 1 << 24;
        }
        let lam = self.lam.ti16;
        for n in 0..16 {
            let (y, x) = (n / 4, n % 4);
            let k = lw.tabs.trellis(
                &self.q.y1,
                &mut dq[n],
                &mut self.lv[n],
                usize::from(l[y] + t[x]),
                0,
                lam,
            );
            l[y] = u8::from(k);
            t[x] = u8::from(k);
            self.lv[n][0] = 0;
            if k {
                nz |= 1 << n;
            }
        }
        let q = iwht(&dc);
        for n in 0..16 {
            dq[n][0] = q[n];
        }
        nz
    }

    /// One chroma trial: ReconstructUV into z.
    fn lw_uv(&mut self, lw: &LwState, src: &MbSrc) -> (u32, [[i8; 3]; 2]) {
        let mut dq = [[0i16; 16]; 8];
        residuals_uv(&self.z, &mut dq, &src.u, &src.v);
        let mut derr = [[0i8; 3]; 2];
        if lw.diffuse {
            let mut dc = [0i16; 8];
            derr = correct_dc(&self.q.uv, &dq, &mut dc, &self.top_derr, &self.left_derr);
            for n in 0..8 {
                dq[n][0] = dc[n];
            }
        }
        let nz = self.q.uv.quantize_n(&mut dq, &mut self.lv[16..24]);
        idct_adds_uv(&mut self.z, &dq);
        (nz, derr)
    }

    /// One method 6 i4 trial of the prediction `pr` at (y, x): the trellis levels into lv, the
    /// reconstruction into rec, [last, SSE, TTransform, nonzero AC count] into res. `s4` is the
    /// block's source (stride 16), `w4` the same as words.
    fn i4_trial_t(
        &mut self,
        lw: &LwState,
        pr: &[u32; 4],
        (s4, w4): (&[u8], &[u32; 4]),
        lv: &mut [i16; 16],
        rec: &mut [u32; 4],
        res: &mut [i32; 4],
        (y, x, ctx): (usize, usize, usize),
    ) {
        let mut blk = [0i16; 16];
        put4(&mut self.z, pr, y, x);
        residual(&self.z, &mut blk, s4, 16, y, x);
        lw.tabs
            .trellis(&self.q.y1, &mut blk, lv, ctx, 3, self.lam.ti4);
        let last = last_nz(lv, 0);
        let (r, sse, t) = rec_metrics(&blk, pr, w4);
        let nz = lv[1..].iter().filter(|&&v| v != 0).count() as i32;
        *rec = r;
        *res = [last, sse, t, nz];
    }

    /// VP8Decimate at RD_OPT_BASIC, then RecordTokens.
    pub(crate) fn encode_mb_lw(&mut self, mbx: usize, mby: usize, left: &mut NzCtx) {
        let (Some(lw), Some(tmp)) = (self.sh.lw, self.tmp.take()) else {
            return;
        };
        let c = costs();
        let method = self.sh.method;
        self.prepare(mbx, mby);
        let src = self.mb_src(mbx);
        let idx = mbx;
        let seg = self.mbs[idx].seg as usize;
        self.q = self.sh.seg_q[seg];
        self.lam = self.sh.seg_lam[seg];
        let lam = self.lam;
        let qi4 = qa(&self.q.y1);
        let mut tlambda = ((50 * qi4) >> 5).max(1);
        if method < 4 {
            tlambda = 1;
        }

        // TTransform of the source blocks, shared by all luma trials.
        let st = tt_mb(&src.y);

        // PickBestIntra16. A trial is a function of its prediction, so a mode predicting an
        // earlier mode's bytes reuses that trial (rep). A trial reruns only where its levels or
        // reconstruction are gone.
        let (mut best16, mut bs, mut bnz) = (0u8, 0i64, 0u32);
        let (mut b_d, mut b_sd, mut b_r, mut b_h) = (0i64, 0i64, 0i64, 0i64);
        let mut flat16 = super::flat(&src.y, 0, 16, 16, 16, 1);
        let mut c16 = [Trial::default(); 4];
        // The reps held by blv/brec and by self.lv.
        let (mut b_rep, mut lv_rep) = (usize::MAX, usize::MAX);
        for m in 0..4u8 {
            pred_mb(&mut self.z, 16, 1, 8, m, mbx, mby);
            let mu = usize::from(m);
            // z_rec: the rep whose reconstruction z holds.
            let (mut rep, mut z_rec) = (mu, usize::MAX);
            if lw.dedup {
                pred_y(&self.z, &mut tmp.p16[mu]);
                if let Some(j) = (0..mu).find(|&j| tmp.p16[j] == tmp.p16[mu]) {
                    rep = j;
                }
            }
            if rep == mu {
                self.i16_trial(&mut c16[rep], lw, &mut tmp.dq16, left, &src, &st, tlambda);
                (lv_rep, z_rec) = (rep, rep);
            }
            let (nz, mut d, mut sd) = (c16[rep].nz, c16[rep].d, c16[rep].sd);
            let h = i64::from(c.i16[mu]);
            if flat16 {
                flat16 = nz & 0xffff == 0;
                if flat16 {
                    d *= 2;
                    sd *= 2;
                }
            }
            // The rate is never negative.
            if lw.dedup && m > 0 && rd_score(lam.i16, 0, h, d, sd) >= bs {
                continue;
            }
            if !c16[rep].r_ok {
                if lv_rep != rep {
                    // z still holds the prediction.
                    self.i16_trial(&mut c16[rep], lw, &mut tmp.dq16, left, &src, &st, tlambda);
                    (lv_rep, z_rec) = (rep, rep);
                }
                c16[rep].r = i64::from(lw.tabs.cost_luma16(&self.lv, *left, self.top));
                c16[rep].r_ok = true;
            }
            let r = c16[rep].r;
            let sc = rd_score(lam.i16, r, h, d, sd);
            if m == 0 || sc < bs {
                if b_rep != rep {
                    if z_rec != rep {
                        self.i16_trial(&mut c16[rep], lw, &mut tmp.dq16, left, &src, &st, tlambda);
                        lv_rep = rep;
                    }
                    for j in 0..16 {
                        tmp.brec[j].copy_from_slice(&self.z[1 + j][8..24]);
                    }
                    tmp.blv[..16].copy_from_slice(&self.lv[..16]);
                    tmp.blv[24] = self.lv[24];
                    b_rep = rep;
                }
                best16 = m;
                bs = sc;
                bnz = nz;
                (b_d, b_sd, b_r, b_h) = (d, sd, r, h);
            }
        }
        let s16 = rd_score(lam.mode, b_r, b_h, b_d, b_sd);
        if bnz & 0x100_ffff == 0x100_0000 && b_d > 20 * i64::from(self.q.y1.q[0]) {
            let dc = &tmp.blv[24];
            let e = i64::from(dc[1])
                .abs()
                .max(i64::from(dc[2]).abs())
                .max(i64::from(dc[4]).abs());
            self.max_edge[seg] = self.max_edge[seg].max(e);
        }

        // PickBestIntra4.
        let mut i16m = false;
        let mut modes = [0u8; 16];
        {
            let mut sc = rd_score(lam.mode, 0, 211, 0, 0);
            let mut hdr = 0i64;
            let (mut l, mut t) = (*left, self.top);
            for n in 0..16 {
                let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
                let s4 = &src.y[(y - 1) * 16 + x - 8..];
                let w4 = src4(&src.y, n);
                let (above, lft) = self.lw_modes(&modes, mbx, n);
                let ctx = usize::from(l[n / 4] + t[n % 4]);
                pred4_all(&self.z, &mut tmp.pr, y, x);
                // All ten predictions equal: mode 0's trial stands for every mode.
                let same = lw.dedup && tmp.pr[1..].iter().all(|p| *p == tmp.pr[0]);
                if method >= 6 {
                    for m in 0..if same { 1 } else { 10 } {
                        self.i4_trial_t(
                            lw,
                            &tmp.pr[m],
                            (s4, &w4),
                            &mut tmp.lv[m],
                            &mut tmp.rec[m],
                            &mut tmp.res[m],
                            (y, x, ctx),
                        );
                    }
                } else if same {
                    i4_trial1(
                        &self.q.y1,
                        &tmp.pr[0],
                        &w4,
                        &mut tmp.lv[0],
                        &mut tmp.rec[0],
                        &mut tmp.res[0],
                    );
                } else {
                    i4_trials(
                        &self.q.y1,
                        &tmp.pr,
                        &w4,
                        &mut tmp.lv,
                        &mut tmp.rec,
                        &mut tmp.res,
                    );
                }
                let hc = &c.i4[above as usize][lft as usize];
                let sn = i64::from(st[n]);
                let p = if same {
                    lw.pick_i4_same(&tmp.res, &tmp.lv, hc, sn, ctx, tlambda)
                } else {
                    lw.pick_i4(&tmp.res, &tmp.lv, hc, sn, ctx, lam.i4, tlambda)
                };
                sc += rd_score(lam.mode, p.r, p.h, p.d, p.sd);
                if sc >= s16 {
                    i16m = true;
                    break;
                }
                hdr += p.h;
                if hdr > 65536 {
                    // max_i4_header_bits at method 4
                    i16m = true;
                    break;
                }
                // With `same` only mode 0 has a trial.
                let k = if same { 0 } else { p.best };
                put4(&mut self.z, &tmp.rec[k], y, x);
                self.lv[n] = tmp.lv[k];
                modes[n] = p.best as u8;
                let nz = u8::from(tmp.res[k][0] >= 0);
                l[n / 4] = nz;
                t[n % 4] = nz;
            }
        }
        if i16m {
            for j in 0..16 {
                self.z[1 + j][8..24].copy_from_slice(&tmp.brec[j]);
            }
            self.lv[..16].copy_from_slice(&tmp.blv[..16]);
            self.lv[24] = tmp.blv[24];
            modes = [best16; 16];
        }

        // PickBestUV. As PickBestIntra16: equal predictions share a trial.
        let mut bderr = [[0i8; 3]; 2];
        let (mut best_uv, mut bs_uv) = (0u8, 0i64);
        let mut c_uv = [Trial::default(); 4];
        let (mut b_rep_uv, mut lv_rep_uv, mut z_rec_uv) = (usize::MAX, usize::MAX, usize::MAX);
        for m in 0..4u8 {
            pred_mb(&mut self.z, 8, 18, 8, m, mbx, mby);
            pred_mb(&mut self.z, 8, 18, 24, m, mbx, mby);
            let mu = usize::from(m);
            let mut rep = mu;
            z_rec_uv = usize::MAX;
            if lw.dedup {
                pred_uv(&self.z, &mut tmp.p_uv[mu]);
                if let Some(j) = (0..mu).find(|&j| tmp.p_uv[j] == tmp.p_uv[mu]) {
                    rep = j;
                }
            }
            if rep == mu {
                self.uv_trial(&mut c_uv[rep], lw, &src);
                (lv_rep_uv, z_rec_uv) = (rep, rep);
            }
            let d = c_uv[rep].d;
            if lw.dedup && m > 0 && rd_score(lam.uv, 0, UV_COST_LW[mu], d, 0) >= bs_uv {
                continue;
            }
            if !c_uv[rep].r_ok || m > 0 && !c_uv[rep].flat_ok {
                if lv_rep_uv != rep {
                    // z still holds the prediction.
                    self.uv_trial(&mut c_uv[rep], lw, &src);
                    (lv_rep_uv, z_rec_uv) = (rep, rep);
                }
                if !c_uv[rep].r_ok {
                    c_uv[rep].r = i64::from(lw.tabs.cost_uv(&self.lv, *left, self.top));
                    c_uv[rep].r_ok = true;
                }
                if m > 0 && !c_uv[rep].flat_ok {
                    c_uv[rep].flat = is_flat(&self.lv[16..24], c_uv[rep].nz, 2);
                    c_uv[rep].flat_ok = true;
                }
            }
            let mut r = c_uv[rep].r;
            if m > 0 && c_uv[rep].flat {
                r += FLAT_PENALTY * 8;
            }
            let sc = rd_score(lam.uv, r, UV_COST_LW[mu], d, 0);
            if m == 0 || sc < bs_uv {
                if b_rep_uv != rep {
                    if z_rec_uv != rep {
                        self.uv_trial(&mut c_uv[rep], lw, &src);
                        (lv_rep_uv, z_rec_uv) = (rep, rep);
                    }
                    for j in 0..8 {
                        tmp.buv[j].copy_from_slice(&self.z[18 + j][8..16]);
                        tmp.buv[8 + j].copy_from_slice(&self.z[18 + j][24..32]);
                    }
                    tmp.lv_uv.copy_from_slice(&self.lv[16..24]);
                    b_rep_uv = rep;
                }
                best_uv = m;
                bs_uv = sc;
                bderr = c_uv[rep].derr;
            }
        }
        if z_rec_uv != b_rep_uv {
            // Otherwise z holds the best reconstruction.
            for j in 0..8 {
                self.z[18 + j][8..16].copy_from_slice(&tmp.buv[j]);
                self.z[18 + j][24..32].copy_from_slice(&tmp.buv[8 + j]);
            }
        }
        self.lv[16..24].copy_from_slice(&tmp.lv_uv);
        self.tmp = Some(tmp);
        if lw.diffuse {
            store_derr(&bderr, &mut self.top_derr, &mut self.left_derr);
        }
        if method == 5 {
            self.simple_quantize(lw, i16m, &modes, best_uv, mbx, mby, left, &src);
        }

        // RecordTokens: every MB, no skip flag. The statistics are recorded from the tokens
        // after the segment.
        record_luma(self.tok, i16m, &self.lv, left, &mut self.top);
        record_chroma(self.tok, &self.lv, left, &mut self.top);

        self.store_mb(mbx, true);
        self.mbs[idx] = MbInfo {
            skip: false,
            i16: i16m,
            modes,
            uv: best_uv,
            seg: seg as u8,
        };
    }

    /// lwModes: the context modes of block n with this MB's i4 modes still in progress.
    fn lw_modes(&self, modes: &[u8; 16], mbx: usize, n: usize) -> (u8, u8) {
        let above = if n >= 4 { modes[n - 4] } else { self.above[n] };
        let mut left = PRED_DC;
        if n & 3 != 0 {
            left = modes[n - 1];
        } else if mbx > 0 {
            left = self.mbs[mbx - 1].modes[n + 3];
        }
        (above, left)
    }

    /// SimpleQuantize for method 5: the chosen modes again with trellis. As in libwebp the i4
    /// contexts stay at the MB's (the iterator never updates them there) and UV reads the
    /// diffusion errors PickBestUV already stored.
    fn simple_quantize(
        &mut self,
        lw: &LwState,
        i16m: bool,
        modes: &[u8; 16],
        uv: u8,
        mbx: usize,
        mby: usize,
        left: &NzCtx,
        src: &MbSrc,
    ) {
        if i16m {
            let mut dq = [[0i16; 16]; 16];
            pred_mb(&mut self.z, 16, 1, 8, modes[0], mbx, mby);
            self.lw_i16_t(lw, &mut dq, *left, self.top, &src.y);
            idct_adds_y(&mut self.z, &dq);
        } else {
            let (l, t) = (*left, self.top);
            for n in 0..16 {
                let (y, x) = (1 + 4 * (n / 4), 8 + 4 * (n % 4));
                let mut pr = [[0u32; 4]; 10];
                pred4_all(&self.z, &mut pr, y, x);
                put4(&mut self.z, &pr[modes[n] as usize], y, x);
                let mut blk = [0i16; 16];
                residual(&self.z, &mut blk, &src.y[(y - 1) * 16 + x - 8..], 16, y, x);
                lw.tabs.trellis(
                    &self.q.y1,
                    &mut blk,
                    &mut self.lv[n],
                    usize::from(l[n / 4] + t[n % 4]),
                    3,
                    self.lam.ti4,
                );
                idct_add(&mut self.z, &blk, y, x);
            }
        }
        pred_mb(&mut self.z, 8, 18, 8, uv, mbx, mby);
        pred_mb(&mut self.z, 8, 18, 24, uv, mbx, mby);
        self.lw_uv(lw, src);
    }
}

impl Enc<'_> {
    /// VP8AdjustFilterStrength with sharpness 0.
    pub(crate) fn adjust_filter(&self, s: &mut Segs) {
        if let Some(lw) = self.lw.as_ref() {
            for i in 0..4 {
                let lv = ((lw.max_edge[i] * i64::from(self.seg_q[i].y2.q[1])) >> 3).min(63) as i32;
                if lv > s.level[i] {
                    s.level[i] = lv;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::{NO_DEDUP, Ybr, i4_trial1, i4_trials, idct_add, put4, rec_metrics, residual};
    use crate::testutil::{pcg::Pcg, png::decode_png, read};

    static NO_CANCEL: AtomicBool = AtomicBool::new(false);

    /// The top-left w x h of a PNG as opaque NRGBA.
    fn crop(name: &str, w: usize, h: usize) -> Vec<u8> {
        let p = decode_png(&read(name));
        assert!(
            p.channels == 1 || p.channels == 3,
            "{name}: {} channels",
            p.channels
        );
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let o = p.channels * (y * p.width + x);
                let g = |c: usize| p.pix[o + if p.channels == 1 { 0 } else { c }];
                v.extend_from_slice(&[g(0), g(1), g(2), 255]);
            }
        }
        v
    }

    fn run(dedup: bool, w: usize, h: usize, pix: &[u8], quality: f32, method: i32) -> Vec<u8> {
        NO_DEDUP.set(!dedup);
        let o = Options {
            quality,
            method,
            match_libwebp: true,
            ..Options::default()
        };
        let src = Source::Nrgba { pix, stride: 4 * w };
        let mut out = Vec::new();
        let r = encode(
            w,
            h,
            &src,
            &o,
            &Limits::default(),
            &NO_CANCEL,
            &mut Bufs::default(),
            &mut out,
        );
        NO_DEDUP.set(false);
        r.unwrap();
        out
    }

    /// `i4_trials` and `rec_metrics` against the trial through `transform::residual`,
    /// `quantize_sse`, `idct_add` and the SSE, the way `i4_trial` ran before the kernels.
    #[test]
    fn i4_kernels_match_old_trial() {
        use super::super::quant::new_quant;
        let mut rng = Pcg::new(3, 5);
        for qi in (0..=127).step_by(3) {
            let q = new_quant(qi, 1);
            for round in 0..6 {
                let src: [u32; 4] = std::array::from_fn(|_| rng.uint32());
                let pr: [[u32; 4]; 10] = std::array::from_fn(|_| {
                    std::array::from_fn(|j| match round % 3 {
                        0 => rng.uint32(),
                        1 => src[j] ^ (rng.uint32() & 0x0303_0303),
                        _ => (rng.uint32() & 0xff00_ff00) | 0x00ff_0000,
                    })
                });
                let (mut lv, mut rec, mut res) =
                    ([[0i16; 16]; 10], [[0u32; 4]; 10], [[0i32; 4]; 10]);
                i4_trials(&q.y1, &pr, &src, &mut lv, &mut rec, &mut res);
                let sb: Vec<u8> = src.iter().flat_map(|w| w.to_le_bytes()).collect();
                for m in 0..10 {
                    let mut z: Ybr = [[0; 32]; 26];
                    put4(&mut z, &pr[m], 1, 8);
                    let mut blk = [0i16; 16];
                    residual(&z, &mut blk, &sb, 4, 1, 8);
                    let mut want_lv = [0i16; 16];
                    let last = q.y1.quantize_sse(&mut blk, &mut want_lv).0;
                    idct_add(&mut z, &blk, 1, 8);
                    let mut sse = 0;
                    for j in 0..4 {
                        let r = &z[1 + j];
                        let w = u32::from_le_bytes([r[8], r[9], r[10], r[11]]);
                        assert_eq!(rec[m][j], w, "rec qi {qi} round {round} m {m}");
                        for i in 0..4 {
                            let k = i32::from(sb[4 * j + i]) - i32::from(r[8 + i]);
                            sse += k * k;
                        }
                    }
                    let nz = want_lv[1..].iter().filter(|&&v| v != 0).count() as i32;
                    assert_eq!(lv[m], want_lv, "lv qi {qi} round {round} m {m}");
                    assert_eq!(
                        res[m][..2],
                        [last, sse],
                        "last, sse qi {qi} round {round} m {m}"
                    );
                    assert_eq!(res[m][3], nz, "nz qi {qi} round {round} m {m}");
                    let (r1, s1, t1) = rec_metrics(&blk, &pr[m], &src);
                    assert_eq!((r1, s1, t1), (rec[m], sse, res[m][2]), "rec_metrics m {m}");
                    let (mut l1, mut c1, mut e1) = ([0i16; 16], [0u32; 4], [0i32; 4]);
                    i4_trial1(&q.y1, &pr[m], &src, &mut l1, &mut c1, &mut e1);
                    assert_eq!((l1, c1, e1), (lv[m], rec[m], res[m]), "trial1 m {m}");
                }
            }
        }
    }

    /// Go's TestLWDedup: sharing the trials of equal predictions changes no byte.
    #[test]
    fn dedup_matches_full_trials() {
        let mut rng = Pcg::new(7, 11);
        let noise: Vec<u8> = (0..4 * 64 * 48)
            .map(|i| if i % 4 == 3 { 255 } else { rng.uint32() as u8 })
            .collect();
        let mut gray = vec![255u8; 4 * 80 * 64];
        for (i, p) in gray.chunks_mut(4).enumerate() {
            // Flat gray with one block of texture.
            let v = if i % 80 > 40 && i / 80 > 30 {
                (i * 7 % 23) as u8 * 9
            } else {
                128
            };
            p[..3].fill(v);
        }
        let inputs = [
            ("fl_768", 160, 128, crop("fl_768.png", 160, 128)),
            ("kodim23", 192, 192, crop("kodim23_192.png", 192, 192)),
            ("kodim05", 96, 80, crop("kodim05_192.png", 96, 80)),
            ("noise", 64, 48, noise),
            ("gray", 80, 64, gray),
        ];
        for (name, w, h, pix) in &inputs {
            for method in 3..=6 {
                for quality in [0.0, 30.0, 75.0, 100.0] {
                    let on = run(true, *w, *h, pix, quality, method);
                    let off = run(false, *w, *h, pix, quality, method);
                    assert!(
                        on == off,
                        "{name} m{method} q{quality}: dedup changes the output"
                    );
                }
            }
        }
    }
}
