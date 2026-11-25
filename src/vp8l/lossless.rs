//! Port of libwebp 1.2.2 dsp/lossless_enc.c, dsp/lossless.c and dsp/lossless_common.h, with clog
//! from Go's math/log.go. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use super::hist::Histo;
use super::tables::{LOG2_TABLE, SLOG2_TABLE};
use crate::fmath::frexp;

// lossless_common.h:63-66.
const APPROX_LOG_WITH_CORRECTION_MAX: u32 = 65536;
const APPROX_LOG_MAX: u32 = 4096;
const LOG2_RECIP: f64 = 1.44269504088896338700465094007086;
const LOG_LOOKUP_IDX_MAX: u32 = 256;
pub(super) const NON_TRIVIAL_SYM: u32 = 0xffff_ffff;
pub(super) const ARGB_BLACK: u32 = 0xff00_0000;

/// A pure-Go copy of Go's math.Log, so every architecture agrees.
pub(super) fn clog(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const L1: f64 = 6.666666666666735130e-01;
    const L2: f64 = 3.999999999940941908e-01;
    const L3: f64 = 2.857142874366239149e-01;
    const L4: f64 = 2.222219843214978396e-01;
    const L5: f64 = 1.818357216161805012e-01;
    const L6: f64 = 1.531383769920937332e-01;
    const L7: f64 = 1.479819860511658591e-01;
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let (mut f1, mut ki) = frexp(x);
    if f1 < std::f64::consts::FRAC_1_SQRT_2 {
        f1 *= 2.0;
        ki -= 1;
    }
    let f = f1 - 1.0;
    let k = f64::from(ki);
    let s = f / (2.0 + f);
    let s2 = s * s;
    let s4 = s2 * s2;
    let t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)));
    let t2 = s4 * (L2 + s4 * (L4 + s4 * L6));
    let r = t1 + t2;
    let hfsq = (0.5 * f) * f;
    k * LN2_HI - ((hfsq - (s * (hfsq + r) + k * LN2_LO)) - f)
}

/// FastSLog2Slow_C, lossless_enc.c:329.
fn fast_slog2_slow(mut v: u32) -> f32 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let log_cnt = (32 - v.leading_zeros()) as i32 - 8; // BitsLog2Floor(v) - 7
        let y = 1u32 << log_cnt;
        let vf = v as f32;
        let orig_v = v;
        v >>= log_cnt;
        let correction = ((23 * (orig_v & (y - 1))) >> 4) as i32;
        return vf * (LOG2_TABLE[v as usize] + log_cnt as f32) + correction as f32;
    }
    ((LOG2_RECIP * f64::from(v)) * clog(f64::from(v))) as f32
}

/// FastLog2Slow_C, lossless_enc.c:365.
fn fast_log2_slow(mut v: u32) -> f32 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let log_cnt = (32 - v.leading_zeros()) as i32 - 8;
        let y = 1u32 << log_cnt;
        let orig_v = v;
        v >>= log_cnt;
        let mut log2 = f64::from(LOG2_TABLE[v as usize] + log_cnt as f32);
        if orig_v >= APPROX_LOG_MAX {
            let correction = ((23 * (orig_v & (y - 1))) >> 4) as i32;
            log2 += f64::from(correction) / f64::from(orig_v);
        }
        return log2 as f32;
    }
    (LOG2_RECIP * clog(f64::from(v))) as f32
}

/// VP8LFastSLog2, lossless_common.h:89.
pub(super) fn fast_slog2(v: u32) -> f32 {
    if v < LOG_LOOKUP_IDX_MAX {
        return SLOG2_TABLE[v as usize];
    }
    fast_slog2_slow(v)
}

/// VP8LFastLog2, lossless_common.h:85.
pub(super) fn fast_log2(v: u32) -> f32 {
    if v < LOG_LOOKUP_IDX_MAX {
        return LOG2_TABLE[v as usize];
    }
    fast_log2_slow(v)
}

/// VP8LSubSampleSize, lossless_common.h:51.
pub(super) fn sub_sample_size(size: usize, bits: usize) -> usize {
    (size + (1 << bits) - 1) >> bits
}

/// CombinedShannonEntropy_C, lossless_enc.c:403.
pub(super) fn combined_shannon_entropy(x: &[i32; 256], y: &[i32; 256]) -> f32 {
    let mut retval = 0.0f64;
    let (mut sum_x, mut sum_xy) = (0i32, 0i32);
    for i in 0..256 {
        let xi = x[i];
        if xi != 0 {
            let xy = xi + y[i];
            sum_x += xi;
            retval -= f64::from(fast_slog2(xi as u32));
            sum_xy += xy;
            retval -= f64::from(fast_slog2(xy as u32));
        } else if y[i] != 0 {
            sum_xy += y[i];
            retval -= f64::from(fast_slog2(y[i] as u32));
        }
    }
    retval += f64::from(fast_slog2(sum_x as u32) + fast_slog2(sum_xy as u32));
    retval as f32
}

/// VP8LBitEntropy, dsp/lossless.h:200.
#[derive(Clone, Copy, Default)]
pub(super) struct BitEntropy {
    pub entropy: f64,
    pub sum: u32,
    pub nonzeros: i32,
    pub max_val: u32,
    pub nonzero_code: u32,
}

/// VP8LStreaks, dsp/lossless.h:195.
#[derive(Clone, Copy, Default)]
pub(super) struct Streaks {
    /// Zero streak, non-zero streak.
    pub counts: [i32; 2],
    /// [zero/non-zero][streak <= 3 / streak > 3].
    pub streaks: [[i32; 2]; 2],
}

impl BitEntropy {
    /// VP8LBitEntropyInit, lossless_enc.c:424.
    pub(super) fn init(&mut self) {
        *self = BitEntropy {
            nonzero_code: NON_TRIVIAL_SYM,
            ..BitEntropy::default()
        };
    }
}

/// VP8LBitsEntropyUnrefined, lossless_enc.c:432.
pub(super) fn bits_entropy_unrefined(a: &[u32], b: &mut BitEntropy) {
    b.init();
    for (i, &v) in a.iter().enumerate() {
        if v != 0 {
            b.sum += v;
            b.nonzero_code = i as u32;
            b.nonzeros += 1;
            b.entropy -= f64::from(fast_slog2(v));
            if b.max_val < v {
                b.max_val = v;
            }
        }
    }
    b.entropy += f64::from(fast_slog2(b.sum));
}

/// GetEntropyUnrefinedHelper, lossless_enc.c:452.
fn entropy_unrefined_helper(
    val: u32,
    i: usize,
    val_prev: &mut u32,
    i_prev: &mut usize,
    b: &mut BitEntropy,
    s: &mut Streaks,
) {
    let streak = (i - *i_prev) as i32;

    // Gather info for the bit entropy.
    if *val_prev != 0 {
        b.sum += *val_prev * streak as u32;
        b.nonzeros += streak;
        b.nonzero_code = *i_prev as u32;
        b.entropy -= f64::from(fast_slog2(*val_prev) * streak as f32);
        if b.max_val < *val_prev {
            b.max_val = *val_prev;
        }
    }

    // Gather info for the Huffman cost.
    let nz = usize::from(*val_prev != 0);
    let long = usize::from(streak > 3);
    s.counts[nz] += long as i32;
    s.streaks[nz][long] += streak;

    *val_prev = val;
    *i_prev = i;
}

/// GetEntropyUnrefined_C, lossless_enc.c:476.
pub(super) fn get_entropy_unrefined(x: &[u32], b: &mut BitEntropy, s: &mut Streaks) {
    let mut i_prev = 0usize;
    let mut x_prev = x[0];
    *s = Streaks::default();
    b.init();
    let mut i = 1;
    while i < x.len() {
        if x[i] != x_prev {
            entropy_unrefined_helper(x[i], i, &mut x_prev, &mut i_prev, b, s);
        }
        i += 1;
    }
    entropy_unrefined_helper(0, i, &mut x_prev, &mut i_prev, b, s);
    b.entropy += f64::from(fast_slog2(b.sum));
}

/// GetCombinedEntropyUnrefined_C, lossless_enc.c:497.
pub(super) fn get_combined_entropy_unrefined(
    x: &[u32],
    y: &[u32],
    b: &mut BitEntropy,
    s: &mut Streaks,
) {
    let y = &y[..x.len()];
    let mut i_prev = 0usize;
    let mut xy_prev = x[0].wrapping_add(y[0]);
    *s = Streaks::default();
    b.init();
    let mut i = 1;
    while i < x.len() {
        let xy = x[i].wrapping_add(y[i]);
        if xy != xy_prev {
            entropy_unrefined_helper(xy, i, &mut xy_prev, &mut i_prev, b, s);
        }
        i += 1;
    }
    entropy_unrefined_helper(0, i, &mut xy_prev, &mut i_prev, b, s);
    b.entropy += f64::from(fast_slog2(b.sum));
}

/// VectorMismatch_C, lossless_enc.c:606: the first index below `length` where a and b differ.
pub(super) fn vector_mismatch(a: &[u32], b: &[u32], length: usize) -> usize {
    let mut i = 0;
    while i < length && a[i] == b[i] {
        i += 1;
    }
    i
}

/// VP8LBundleColorMap_C, lossless_enc.c:617.
pub(super) fn bundle_color_map(row: &[u8], width: usize, xbits: usize, dst: &mut [u32]) {
    if xbits > 0 {
        let bit_depth = 1usize << (3 - xbits);
        let mask = (1usize << xbits) - 1;
        let mut code = 0xff00_0000u32;
        for x in 0..width {
            let xsub = x & mask;
            if xsub == 0 {
                code = 0xff00_0000;
            }
            code |= u32::from(row[x]) << (8 + bit_depth * xsub);
            dst[x >> xbits] = code;
        }
    } else {
        for x in 0..width {
            dst[x] = 0xff00_0000 | (u32::from(row[x]) << 8);
        }
    }
}

/// ExtraCost_C, lossless_enc.c:639: the uint32 products.
pub(super) fn extra_cost(pop: &[u32]) -> f64 {
    let mut cost = 0.0;
    for i in 2..pop.len() - 2 {
        cost += f64::from(((i >> 1) as u32).wrapping_mul(pop[i + 2]));
    }
    cost
}

/// ExtraCostCombined_C, lossless_enc.c:646: the int32 products.
pub(super) fn extra_cost_combined(x: &[u32], y: &[u32]) -> f64 {
    let mut cost = 0.0;
    for i in 2..x.len() - 2 {
        let xy = x[i + 2].wrapping_add(y[i + 2]) as i32;
        cost += f64::from(((i >> 1) as i32).wrapping_mul(xy));
    }
    cost
}

/// VP8LHistogramAdd, lossless_enc.c:694, for distinct a, b and out.
pub(super) fn histogram_add3(a: &Histo, b: &Histo, out: &mut Histo) {
    for k in 0..5 {
        let (ap, bp) = (a.part(k), b.part(k));
        let op = out.part_mut(k);
        if a.is_used[k] {
            if b.is_used[k] {
                for i in 0..op.len() {
                    op[i] = ap[i].wrapping_add(bp[i]);
                }
            } else {
                op.copy_from_slice(ap);
            }
        } else if b.is_used[k] {
            op.copy_from_slice(bp);
        } else {
            op.fill(0);
        }
    }
    for k in 0..5 {
        out.is_used[k] = a.is_used[k] || b.is_used[k];
    }
}

/// VP8LHistogramAdd with out == b: `out += a`.
pub(super) fn histogram_add_eq(a: &Histo, out: &mut Histo) {
    for k in 0..5 {
        if a.is_used[k] {
            let ap = a.part(k);
            let used = out.is_used[k];
            let op = out.part_mut(k);
            if used {
                for i in 0..op.len() {
                    op[i] = op[i].wrapping_add(ap[i]);
                }
            } else {
                op.copy_from_slice(ap);
            }
        }
    }
    for k in 0..5 {
        out.is_used[k] = out.is_used[k] || a.is_used[k];
    }
}

/// VP8LSubPixels, lossless_common.h:160.
pub(super) fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha_and_green = 0x00ff_00ffu32
        .wrapping_add(a & 0xff00_ff00)
        .wrapping_sub(b & 0xff00_ff00);
    let red_and_blue = 0xff00_ff00u32
        .wrapping_add(a & 0x00ff_00ff)
        .wrapping_sub(b & 0x00ff_00ff);
    (alpha_and_green & 0xff00_ff00) | (red_and_blue & 0x00ff_00ff)
}

/// Average2, lossless.c:29.
fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefe) >> 1) + (a & b)
}

/// Average3, lossless.c:33.
fn average3(a: u32, b: u32, c: u32) -> u32 {
    average2(average2(a, c), b)
}

/// Average4, lossless.c:37.
fn average4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    average2(average2(a, b), average2(c, d))
}

/// Clip255, lossless.c:42.
fn clip255(a: u32) -> u32 {
    if a < 256 { a } else { !a >> 24 }
}

/// AddSubtractComponentFull, lossless.c:51.
fn add_subtract_component_full(a: i32, b: i32, c: i32) -> i32 {
    clip255((a + b - c) as u32) as i32
}

/// ClampedAddSubtractFull, lossless.c:55.
fn clamped_add_subtract_full(c0: u32, c1: u32, c2: u32) -> u32 {
    let a = add_subtract_component_full((c0 >> 24) as i32, (c1 >> 24) as i32, (c2 >> 24) as i32);
    let r = add_subtract_component_full(
        ((c0 >> 16) & 0xff) as i32,
        ((c1 >> 16) & 0xff) as i32,
        ((c2 >> 16) & 0xff) as i32,
    );
    let g = add_subtract_component_full(
        ((c0 >> 8) & 0xff) as i32,
        ((c1 >> 8) & 0xff) as i32,
        ((c2 >> 8) & 0xff) as i32,
    );
    let b = add_subtract_component_full((c0 & 0xff) as i32, (c1 & 0xff) as i32, (c2 & 0xff) as i32);
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

/// AddSubtractComponentHalf, lossless.c:68.
fn add_subtract_component_half(a: i32, b: i32) -> i32 {
    clip255((a + (a - b) / 2) as u32) as i32
}

/// ClampedAddSubtractHalf, lossless.c:72.
fn clamped_add_subtract_half(c0: u32, c1: u32, c2: u32) -> u32 {
    let ave = average2(c0, c1);
    let a = add_subtract_component_half((ave >> 24) as i32, (c2 >> 24) as i32);
    let r = add_subtract_component_half(((ave >> 16) & 0xff) as i32, ((c2 >> 16) & 0xff) as i32);
    let g = add_subtract_component_half(((ave >> 8) & 0xff) as i32, ((c2 >> 8) & 0xff) as i32);
    let b = add_subtract_component_half((ave & 0xff) as i32, (c2 & 0xff) as i32);
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

/// Sub3, lossless.c:90.
fn sub3(a: i32, b: i32, c: i32) -> i32 {
    let pb = b - c;
    let pa = a - c;
    pb.abs() - pa.abs()
}

/// Select, lossless.c:98.
fn select_pred(a: u32, b: u32, c: u32) -> u32 {
    let pa_minus_pb = sub3((a >> 24) as i32, (b >> 24) as i32, (c >> 24) as i32)
        + sub3(
            ((a >> 16) & 0xff) as i32,
            ((b >> 16) & 0xff) as i32,
            ((c >> 16) & 0xff) as i32,
        )
        + sub3(
            ((a >> 8) & 0xff) as i32,
            ((b >> 8) & 0xff) as i32,
            ((c >> 8) & 0xff) as i32,
        )
        + sub3((a & 0xff) as i32, (b & 0xff) as i32, (c & 0xff) as i32);
    if pa_minus_pb <= 0 { a } else { b }
}

/// VP8LPredictor2_C..13_C, lossless.c:121-182, for mode 2..13. `top` is upper[x-1..], so
/// top[0] = TL, top[1] = T, top[2] = TR.
fn predictor(mode: usize, left: u32, top: &[u32]) -> u32 {
    match mode {
        2 => top[1],
        3 => top[2],
        4 => top[0],
        5 => average3(left, top[1], top[2]),
        6 => average2(left, top[0]),
        7 => average2(left, top[1]),
        8 => average2(top[0], top[1]),
        9 => average2(top[1], top[2]),
        10 => average4(left, top[0], top[1], top[2]),
        11 => select_pred(top[1], left, top[0]),
        12 => clamped_add_subtract_full(left, top[1], top[0]),
        _ => clamped_add_subtract_half(left, top[1], top[0]),
    }
}

/// VP8LPredictorsSub[mode], lossless_enc.c:724-764: writes out[0..n] for in_[x..x+n], reading
/// in_[x-1] and upper[x-1..x+n+1]. `upper` is empty for modes 0 and 1.
pub(super) fn predictors_sub(
    mode: usize,
    in_: &[u32],
    upper: &[u32],
    x: usize,
    n: usize,
    out: &mut [u32],
) {
    match mode {
        0 => {
            for i in 0..n {
                out[i] = sub_pixels(in_[x + i], ARGB_BLACK);
            }
        }
        1 => {
            for i in 0..n {
                out[i] = sub_pixels(in_[x + i], in_[x + i - 1]);
            }
        }
        2 => {
            for i in 0..n {
                out[i] = sub_pixels(in_[x + i], upper[x + i]);
            }
        }
        _ => {
            for i in 0..n {
                out[i] = sub_pixels(
                    in_[x + i],
                    predictor(mode, in_[x + i - 1], &upper[x + i - 1..]),
                );
            }
        }
    }
}
