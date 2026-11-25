//! Segment analysis: libwebp analysis_enc.c (VP8EncAnalyze, AssignSegments) and the segment
//! parameters of quant_enc.c (VP8SetSegmentParams, SimplifySegments), BSD-3-Clause.

use super::predict::{PRED_DC, PRED_TM, Planes, Ybr, pred_mb, prepare};
use super::quant::{filter_level, quality_compression};
use super::wave::{lock, workers};
use super::{Enc, MbInfo, flat};
use crate::Error;
use crate::fmath::pow;
use crate::limits::check_cancel;
use crate::simd::enc_collect::{bins_alpha, collect_uv, collect16};
use std::sync::{Mutex, PoisonError};

/// Per frame segment parameters, libwebp VP8EncSegmentHeader plus dqm.
#[derive(Default)]
pub(crate) struct Segs {
    pub n: usize,
    pub quant: [i32; 4],
    pub level: [i32; 4],
    pub probs: [u8; 3],
    pub dq_uv_ac: i32,
    pub update_map: bool,
}

/// GetProba, frame_enc.c:189-193.
fn seg_prob(a: usize, b: usize) -> u8 {
    if a + b == 0 {
        return 255;
    }
    ((255 * a + (a + b) / 2) / (a + b)) as u8
}

impl Segs {
    /// SetSegmentProbas, frame_enc.c:202-230.
    pub(crate) fn set_probs(&mut self, mbs: &mut [MbInfo]) {
        let mut c = [0usize; 4];
        for m in mbs.iter() {
            c[m.seg as usize] += 1;
        }
        self.probs = [
            seg_prob(c[0] + c[1], c[2] + c[3]),
            seg_prob(c[0], c[1]),
            seg_prob(c[2], c[3]),
        ];
        self.update_map = self.n > 1 && self.probs != [255, 255, 255];
        if !self.update_map {
            for m in mbs.iter_mut() {
                m.seg = 0;
            }
        }
    }
}

/// AssignSegments, analysis_enc.c:135-222. `mbs[i].seg` holds alphas on entry.
fn k_means(hist: &[usize; 256], mbs: &mut [MbInfo]) -> ([i32; 4], i32) {
    let (mut min_a, mut max_a) = (0usize, 255usize);
    while min_a < 255 && hist[min_a] == 0 {
        min_a += 1;
    }
    while max_a > min_a && hist[max_a] == 0 {
        max_a -= 1;
    }
    let mut centers = [0i32; 4];
    for (k, c) in centers.iter_mut().enumerate() {
        *c = (min_a + (2 * k + 1) * (max_a - min_a) / 8) as i32;
    }
    let mut seg = [0u8; 256];
    let mut avg = 0;
    for _ in 0..6 {
        let (mut acc, mut dist) = ([0i32; 4], [0i32; 4]);
        let mut n = 0usize;
        for a in min_a..=max_a {
            if hist[a] == 0 {
                continue;
            }
            let ai = a as i32;
            while n + 1 < 4 && (ai - centers[n + 1]).abs() < (ai - centers[n]).abs() {
                n += 1;
            }
            seg[a] = n as u8;
            dist[n] += ai * hist[a] as i32;
            acc[n] += hist[a] as i32;
        }
        let (mut displaced, mut sum, mut tw) = (0, 0, 0);
        for n in 0..4 {
            if acc[n] == 0 {
                continue;
            }
            let c = (dist[n] + acc[n] / 2) / acc[n];
            displaced += (centers[n] - c).abs();
            centers[n] = c;
            sum += c * acc[n];
            tw += acc[n];
        }
        avg = (sum + tw / 2) / tw;
        if displaced < 5 {
            break;
        }
    }
    for m in mbs.iter_mut() {
        m.seg = seg[m.seg as usize];
    }
    (centers, avg)
}

/// Whether the block's top and left edges equal v.
fn edges_are(z: &Ybr, y: usize, x: usize, n: usize, v: u8) -> bool {
    (x - 1..x + n).all(|i| z[y - 1][i] == v) && (y..y + n).all(|j| z[j][x - 1] == v)
}

/// The analysis of one MB row `mby` of the source `img` (`mbs` its MBs, `z` a workspace): each
/// MB's alpha into its `seg`, counted in `hist`, and the chroma alphas summed into `sum_uv`. A row
/// reads only the source, so the rows can go in any order and on any thread.
fn analyze_row(
    img: &Planes,
    gray: bool,
    method: usize,
    mby: usize,
    mbs: &mut [MbInfo],
    z: &mut Ybr,
    hist: &mut [usize; 256],
    sum_uv: &mut i32,
) {
    let (ys, cs, mbw) = (img.ys, img.cs, mbs.len());
    for mbx in 0..mbw {
        prepare(z, img, mbx, mby, mbw);
        let src_y = &img.y[mby * 16 * ys + mbx * 16..];
        let src_u = &img.cb[mby * 8 * cs + mbx * 8..];
        let src_v = &img.cr[mby * 8 * cs + mbx * 8..];
        // prepare copied the left column from a prediction.
        if mbx > 0 {
            let (oy, oc) = (mby * 16 * ys + mbx * 16 - 1, mby * 8 * cs + mbx * 8 - 1);
            for j in 0..16 {
                z[1 + j][7] = img.y[oy + j * ys];
            }
            for j in 0..8 {
                z[18 + j][7] = img.cb[oc + j * cs];
                z[18 + j][23] = img.cr[oc + j * cs];
            }
        }
        // Every mode predicts a flat MB exactly: alpha 0.
        if flat(src_y, 0, ys, 16, 16, 1)
            && flat(src_u, 0, cs, 8, 8, 1)
            && flat(src_v, 0, cs, 8, 8, 1)
            && edges_are(z, 1, 8, 16, src_y[0])
            && edges_are(z, 18, 8, 8, src_u[0])
            && edges_are(z, 18, 24, 8, src_v[0])
        {
            hist[255] += 1;
            mbs[mbx].seg = 255;
            continue;
        }
        // FastMBAnalyze returns 0, analysis_enc.c:260-286. TM predicting DC's bytes has
        // DC's alpha, so its histogram is skipped.
        let mut a = 0;
        if method > 1 {
            let mut p0 = [[0u8; 16]; 16];
            for m in PRED_DC..=PRED_TM {
                pred_mb(z, 16, 1, 8, m, mbx, mby);
                if m == PRED_DC {
                    for (r, row) in p0.iter_mut().zip(&z[1..17]) {
                        r.copy_from_slice(&row[8..24]);
                    }
                } else if p0.iter().zip(&z[1..17]).all(|(r, row)| *r == row[8..24]) {
                    continue;
                }
                a = a.max(bins_alpha(&collect16(src_y, ys, z)));
            }
        }
        let mut uva = 0;
        // Gray has no chroma; skip UV mode search except at borders.
        if !gray || mbx == 0 || mby == 0 {
            let mut p0 = [[0u8; 16]; 8];
            for m in PRED_DC..=PRED_TM {
                pred_mb(z, 8, 18, 8, m, mbx, mby);
                pred_mb(z, 8, 18, 24, m, mbx, mby);
                let uv_rows =
                    |r: &[u8; 16], row: &[u8; 32]| r[..8] == row[8..16] && r[8..] == row[24..];
                if m == PRED_DC {
                    for (r, row) in p0.iter_mut().zip(&z[18..26]) {
                        r[..8].copy_from_slice(&row[8..16]);
                        r[8..].copy_from_slice(&row[24..]);
                    }
                } else if p0.iter().zip(&z[18..26]).all(|(r, row)| uv_rows(r, row)) {
                    continue;
                }
                uva = uva.max(bins_alpha(&collect_uv(src_u, src_v, cs, z)));
            }
        }
        let mix = (255 - ((3 * a + uva + 2) >> 2)).clamp(0, 255);
        hist[mix as usize] += 1;
        mbs[mbx].seg = mix as u8;
        *sum_uv += uva;
    }
}

impl Enc<'_> {
    /// VP8EncAnalyze over the padded source, analysis_enc.c:317-342, 431-485. The MB rows are
    /// analysed on as many threads as the MB pass takes, each with its own workspace and counts,
    /// which are summed after.
    pub(crate) fn analyze(&mut self, q: f32, method: usize) -> Result<Segs, Error> {
        let (img, gray, cancel) = (&self.img, self.gray, self.cancel);
        let n = workers(self.mbw, self.mbh, self.budget.max_threads());
        let total = Mutex::new(([0usize; 256], 0i32));
        {
            let rows = Mutex::new(self.mbs.chunks_mut(self.mbw).enumerate());
            self.budget.par_workers(n, || {
                let mut z: Ybr = [[0; 32]; 26];
                let (mut hist, mut sum_uv) = ([0usize; 256], 0i32);
                loop {
                    // The lock is held only to take the row.
                    let next = lock(&rows).next();
                    let Some((mby, mbs)) = next else {
                        break;
                    };
                    check_cancel(cancel)?;
                    analyze_row(img, gray, method, mby, mbs, &mut z, &mut hist, &mut sum_uv);
                }
                let mut t = lock(&total);
                for (a, b) in t.0.iter_mut().zip(hist) {
                    *a += b;
                }
                t.1 += sum_uv;
                Ok(())
            })?;
        }
        let (hist, sum_uv) = total.into_inner().unwrap_or_else(PoisonError::into_inner);
        let uv_alpha = sum_uv / self.mbs.len() as i32;
        let (centers, avg) = k_means(&hist, &mut self.mbs);

        // SetSegmentAlphas, analysis_enc.c:80-100.
        let lo = centers.iter().copied().min().unwrap_or(0);
        let mut hi = centers.iter().copied().max().unwrap_or(0);
        if hi == lo {
            hi = lo + 1;
        }
        // VP8SetSegmentParams, quant_enc.c:378-430, sns 50.
        let mut s = Segs::default();
        let amp = 9.0 / 2560.0; // 0.9 * 50 / 100 / 128
        let c0 = quality_compression(q);
        for (i, &c) in centers.iter().enumerate() {
            let alpha = (255 * (c - avg) / (hi - lo)).clamp(-127, 127);
            let beta = (255 * (c - lo) / (hi - lo)).clamp(0, 255);
            let v = 127.0 * (1.0 - pow(c0, 1.0 - amp * f64::from(alpha)));
            s.quant[i] = (v as i32).clamp(0, 127);
            s.level[i] = filter_level(s.quant[i] as usize, beta);
        }
        s.dq_uv_ac = ((uv_alpha - 64) * 10 / 70 * 50 / 100).clamp(-4, 6);

        // SimplifySegments, quant_enc.c:338-376.
        let mut remap = [0u8; 4];
        let mut n = 1;
        for s1 in 1..4 {
            let mut s2 = 0;
            while s2 < n && (s.quant[s2] != s.quant[s1] || s.level[s2] != s.level[s1]) {
                s2 += 1;
            }
            remap[s1] = s2 as u8;
            if s2 == n {
                s.quant[n] = s.quant[s1];
                s.level[n] = s.level[s1];
                n += 1;
            }
        }
        if n < 4 {
            for m in self.mbs.iter_mut() {
                m.seg = remap[m.seg as usize];
            }
            for i in n..4 {
                s.quant[i] = s.quant[n - 1];
                s.level[i] = s.level[n - 1];
            }
        }
        s.n = n;
        s.set_probs(&mut self.mbs);
        Ok(s)
    }

    /// VP8EncAnalyze with segments 1: no analysis, every alpha 0 (ResetAllMBInfo,
    /// analysis_enc.c:353-365, 470-472), then VP8SetSegmentParams as in `analyze`.
    pub(crate) fn one_segment(&mut self, q: f32) -> Segs {
        for m in self.mbs.iter_mut() {
            m.seg = 0;
        }
        let mut s = Segs::default();
        let qi = ((127.0 * (1.0 - quality_compression(q))) as i32).clamp(0, 127);
        for i in 0..4 {
            s.quant[i] = qi;
            s.level[i] = filter_level(qi as usize, 0);
        }
        s.dq_uv_ac = ((-64) * 10 / 70 * 50 / 100i32).clamp(-4, 6);
        s.n = 1;
        s.set_probs(&mut self.mbs);
        s
    }
}
