//! Port of libwebp 1.6.0 dsp/lossless_enc.c and dsp/lossless_common.h, fixed-point entropy.
//! Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.
#![allow(clippy::excessive_precision)]

use super::tables::{LOG2_TABLE, SLOG2_TABLE};
use crate::vp8l::lossless::{Streaks, clog};

pub(super) const LOG_BITS: u32 = 23; // LOG_2_PRECISION_BITS
const LOG2_RECIP_FIXED: u64 = 12102203;
const LOG2_RECIP_FIXED_D: f64 = 12102203.161561485379934310913085937500;
pub(super) const NON_TRIVIAL_SYM: u16 = 0xffff; // VP8L_NON_TRIVIAL_SYM
const APPROX_LOG_WITH_CORRECTION_MAX: u32 = 65536;
const APPROX_LOG_MAX: u32 = 4096;
const LOG_LOOKUP_IDX_MAX: u32 = 256;

/// DivRound, lossless_common.h:110. Go and C99 both truncate.
pub(super) fn div_round(a: i64, b: i64) -> i64 {
    if (a < 0) == (b < 0) {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

/// FastSLog2Slow_C, lossless_enc.c:252.
fn fast_slog2_slow(mut v: u32) -> u64 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let orig_v = u64::from(v);
        let log_cnt = u64::from(32 - v.leading_zeros() - 8); // BitsLog2Floor(v) - 7
        let y = 1u32 << log_cnt;
        v >>= log_cnt;
        let correction = LOG2_RECIP_FIXED * (orig_v & u64::from(y - 1));
        return orig_v * (u64::from(LOG2_TABLE[v as usize]) + (log_cnt << LOG_BITS)) + correction;
    }
    // The conversions keep the products unfused.
    ((LOG2_RECIP_FIXED_D * f64::from(v)) * clog(f64::from(v)) + 0.5) as u64
}

/// FastLog2Slow_C, lossless_enc.c:284.
fn fast_log2_slow(mut v: u32) -> u32 {
    if v < APPROX_LOG_WITH_CORRECTION_MAX {
        let orig_v = v;
        let log_cnt = 32 - v.leading_zeros() - 8; // BitsLog2Floor(v) - 7
        let y = 1u32 << log_cnt;
        v >>= log_cnt;
        let mut log2 = LOG2_TABLE[v as usize] + (log_cnt << LOG_BITS);
        if orig_v >= APPROX_LOG_MAX {
            let correction = LOG2_RECIP_FIXED * u64::from(orig_v & (y - 1));
            log2 = log2.wrapping_add(div_round(correction as i64, i64::from(orig_v)) as u32);
        }
        return log2;
    }
    (LOG2_RECIP_FIXED_D * clog(f64::from(v)) + 0.5) as u32
}

/// VP8LFastLog2, lossless_common.h:98.
pub(super) fn fast_log2(v: u32) -> u32 {
    if v < LOG_LOOKUP_IDX_MAX {
        return LOG2_TABLE[v as usize];
    }
    fast_log2_slow(v)
}

/// VP8LFastSLog2, lossless_common.h:102.
pub(super) fn fast_slog2(v: u32) -> u64 {
    if v < LOG_LOOKUP_IDX_MAX {
        return SLOG2_TABLE[v as usize];
    }
    fast_slog2_slow(v)
}

/// CombinedShannonEntropy_C, lossless_enc.c:320; x and y hold 256. The SSE2 and AVX2 versions add
/// the same terms mod 2^64.
pub(super) fn combined_shannon_entropy(x: &[u32], y: &[u32]) -> u64 {
    let mut retval = 0u64;
    let (mut sum_x, mut sum_xy) = (0u32, 0u32);
    for i in 0..256 {
        let xi = x[i];
        if xi != 0 {
            let xy = xi.wrapping_add(y[i]);
            sum_x = sum_x.wrapping_add(xi);
            retval = retval.wrapping_add(fast_slog2(xi));
            sum_xy = sum_xy.wrapping_add(xy);
            retval = retval.wrapping_add(fast_slog2(xy));
        } else if y[i] != 0 {
            sum_xy = sum_xy.wrapping_add(y[i]);
            retval = retval.wrapping_add(fast_slog2(y[i]));
        }
    }
    fast_slog2(sum_x)
        .wrapping_add(fast_slog2(sum_xy))
        .wrapping_sub(retval)
}

/// ShannonEntropy_C, lossless_enc.c:342.
pub(super) fn shannon_entropy(x: &[u32]) -> u64 {
    let mut retval = 0u64;
    let mut sum_x = 0u32;
    for &xi in x {
        if xi != 0 {
            sum_x = sum_x.wrapping_add(xi);
            retval = retval.wrapping_add(fast_slog2(xi));
        }
    }
    fast_slog2(sum_x).wrapping_sub(retval)
}

/// VP8LBitEntropy, dsp/lossless.h:209.
#[derive(Clone, Copy, Default)]
pub(super) struct BitEntropy {
    pub entropy: u64,
    pub sum: u32,
    pub nonzeros: i32,
    pub max_val: u32,
    pub nonzero_code: u32,
}

impl BitEntropy {
    /// VP8LBitEntropyInit, lossless_enc.c:357.
    pub(super) fn init(&mut self) {
        *self = BitEntropy {
            nonzero_code: u32::from(NON_TRIVIAL_SYM),
            ..BitEntropy::default()
        };
    }
}

/// VP8LBitsEntropyUnrefined, lossless_enc.c:365.
pub(super) fn bits_entropy_unrefined(a: &[u32], b: &mut BitEntropy) {
    b.init();
    for (i, &v) in a.iter().enumerate() {
        if v != 0 {
            b.sum = b.sum.wrapping_add(v);
            b.nonzero_code = i as u32;
            b.nonzeros += 1;
            b.entropy = b.entropy.wrapping_add(fast_slog2(v));
            if b.max_val < v {
                b.max_val = v;
            }
        }
    }
    b.entropy = fast_slog2(b.sum).wrapping_sub(b.entropy);
}

/// GetEntropyUnrefinedHelper, lossless_enc.c:385.
fn entropy_helper(
    val: u32,
    i: usize,
    val_prev: &mut u32,
    i_prev: &mut usize,
    b: &mut BitEntropy,
    s: &mut Streaks,
) {
    let streak = (i - *i_prev) as i32;
    if *val_prev != 0 {
        b.sum = b.sum.wrapping_add(val_prev.wrapping_mul(streak as u32));
        b.nonzeros += streak;
        b.nonzero_code = *i_prev as u32;
        b.entropy = b
            .entropy
            .wrapping_add(fast_slog2(*val_prev).wrapping_mul(streak as u64));
        if b.max_val < *val_prev {
            b.max_val = *val_prev;
        }
    }
    let nz = usize::from(*val_prev != 0);
    let long = usize::from(streak > 3);
    s.counts[nz] += long as i32;
    s.streaks[nz][long] += streak;
    *val_prev = val;
    *i_prev = i;
}

/// GetEntropyUnrefined_C, lossless_enc.c:411.
pub(super) fn entropy_unrefined(x: &[u32], b: &mut BitEntropy, s: &mut Streaks) {
    let mut i_prev = 0usize;
    let mut x_prev = x[0];
    *s = Streaks::default();
    b.init();
    let mut i = 1;
    while i < x.len() {
        if x[i] != x_prev {
            entropy_helper(x[i], i, &mut x_prev, &mut i_prev, b, s);
        }
        i += 1;
    }
    entropy_helper(0, i, &mut x_prev, &mut i_prev, b, s);
    b.entropy = fast_slog2(b.sum).wrapping_sub(b.entropy);
}

/// GetCombinedEntropyUnrefined_C, lossless_enc.c:433.
pub(super) fn combined_entropy_unrefined(
    x: &[u32],
    y: &[u32],
    b: &mut BitEntropy,
    s: &mut Streaks,
) {
    let mut i_prev = 0usize;
    let mut xy_prev = x[0].wrapping_add(y[0]);
    *s = Streaks::default();
    b.init();
    let mut i = 1;
    while i < x.len() {
        let xy = x[i].wrapping_add(y[i]);
        if xy != xy_prev {
            entropy_helper(xy, i, &mut xy_prev, &mut i_prev, b, s);
        }
        i += 1;
    }
    entropy_helper(0, i, &mut xy_prev, &mut i_prev, b, s);
    b.entropy = fast_slog2(b.sum).wrapping_sub(b.entropy);
}

/// ExtraCost_C, lossless_enc.c:576: the uint32 sum, equal to ExtraCost_SSE41 at lengths 24 and 40.
pub(super) fn extra_cost(pop: &[u32]) -> u32 {
    let mut cost = pop[4].wrapping_add(pop[5]);
    for i in 2..pop.len() / 2 - 1 {
        cost =
            cost.wrapping_add((i as u32).wrapping_mul(pop[2 * i + 2].wrapping_add(pop[2 * i + 3])));
    }
    cost
}
