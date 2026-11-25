//! Port of libwebp 1.6.0 enc/histogram_enc.c. Copyright 2012 Google Inc. BSD-3, see
//! LICENSE.libwebp.

use super::log::{
    BitEntropy, LOG_BITS, NON_TRIVIAL_SYM, bits_entropy_unrefined, combined_entropy_unrefined,
    div_round, entropy_unrefined, extra_cost,
};
use crate::Error;
use crate::limits::Ledger;
use crate::vp8l::lossless::{Streaks, sub_sample_size};
use crate::vp8l::refs::{CHECK_EVERY, PixOrCopy, distance_to_plane_code};
use crate::vp8l::{LITERAL, NUM_DISTANCE, NUM_LITERAL, prefix_bits};

pub(super) const NUM_CODES: usize = NUM_LITERAL + 24;
const CODE_LEN_CODES: usize = 19;
const NUM_PARTITIONS: usize = 4;
pub(super) const BIN_SIZE: usize = NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS;
const BIN_ID_EPS: f64 = 1e-6;
const BIN_ID_FACTOR: f64 = NUM_PARTITIONS as f64 - BIN_ID_EPS;
const MAX_HISTO_GREEDY: i64 = 100;

// HistogramIndex, histogram_enc.c:39.
pub(super) const LITERAL_I: usize = 0;
pub(super) const RED_I: usize = 1;
pub(super) const BLUE_I: usize = 2;
pub(super) const ALPHA_I: usize = 3;
pub(super) const DISTANCE_I: usize = 4;

/// kCacheIdx, backward_references_enc.h.
pub(super) const CACHE_IDX: u8 = 1;

/// VP8LHistogramNumCodes, histogram_enc.h.
pub(super) fn num_codes(cache_bits: usize) -> usize {
    if cache_bits > 0 {
        NUM_CODES + (1 << cache_bits)
    } else {
        NUM_CODES
    }
}

/// VP8LHistogram, histogram_enc.h:29. `literal` has `num_codes(palette_code_bits)` entries.
#[derive(Clone)]
pub(super) struct Histo {
    pub literal: Vec<u32>,
    pub red: [u32; NUM_LITERAL],
    pub blue: [u32; NUM_LITERAL],
    pub alpha: [u32; NUM_LITERAL],
    pub distance: [u32; NUM_DISTANCE],
    pub palette_code_bits: usize,
    pub trivial_symbol: [u16; 5],
    pub bit_cost: u64,
    pub costs: [u64; 5],
    pub is_used: [bool; 5],
    pub bin_id: u16,
}

impl Histo {
    pub(super) fn new(cache_bits: usize) -> Histo {
        let mut h = Histo {
            literal: vec![0; num_codes(cache_bits)],
            red: [0; NUM_LITERAL],
            blue: [0; NUM_LITERAL],
            alpha: [0; NUM_LITERAL],
            distance: [0; NUM_DISTANCE],
            palette_code_bits: cache_bits,
            trivial_symbol: [0; 5],
            bit_cost: 0,
            costs: [0; 5],
            is_used: [false; 5],
            bin_id: 0,
        };
        h.stats_clear();
        h
    }

    /// Sizes `literal` for `cache_bits`.
    fn set_bits(&mut self, cache_bits: usize) {
        self.palette_code_bits = cache_bits;
        self.literal.resize(num_codes(cache_bits), 0);
    }

    /// HistogramStatsClear, histogram_enc.c:55.
    fn stats_clear(&mut self) {
        self.trivial_symbol = [NON_TRIVIAL_SYM; 5];
        self.is_used = [true; 5];
        self.bit_cost = 0;
        self.costs = [0; 5];
    }

    /// HistogramClear, histogram_enc.c:66.
    pub(super) fn clear(&mut self) {
        self.literal.fill(0);
        self.red = [0; NUM_LITERAL];
        self.blue = [0; NUM_LITERAL];
        self.alpha = [0; NUM_LITERAL];
        self.distance = [0; NUM_DISTANCE];
        self.bin_id = 0;
        self.stats_clear();
    }

    /// The memset of VP8LHistogramSetClear, histogram_enc.c:176.
    fn zero(&mut self) {
        let bits = self.palette_code_bits;
        let mut lit = std::mem::take(&mut self.literal);
        lit.fill(0);
        *self = Histo {
            literal: lit,
            red: [0; NUM_LITERAL],
            blue: [0; NUM_LITERAL],
            alpha: [0; NUM_LITERAL],
            distance: [0; NUM_DISTANCE],
            palette_code_bits: bits,
            trivial_symbol: [0; 5],
            bit_cost: 0,
            costs: [0; 5],
            is_used: [false; 5],
            bin_id: 0,
        };
    }

    /// VP8LHistogramInit.
    pub(super) fn init(&mut self, cache_bits: usize, arrays: bool) {
        self.set_bits(cache_bits);
        if arrays {
            self.clear();
        } else {
            self.stats_clear();
        }
    }

    /// VP8LHistogramCreate; a negative `cache_bits` keeps the current size.
    pub(super) fn create(&mut self, refs: &[PixOrCopy], cache_bits: Option<usize>) {
        if let Some(b) = cache_bits {
            self.set_bits(b);
        }
        self.clear();
        self.store_refs(refs, 0);
    }

    /// VP8LHistogramStoreRefs: maps distances to plane codes when `plane_xsize` > 0.
    pub(super) fn store_refs(&mut self, refs: &[PixOrCopy], plane_xsize: usize) {
        for r in refs {
            self.add_single_pix_or_copy(r, plane_xsize);
        }
    }

    /// HistogramAddSinglePixOrCopy, histogram_enc.c:203.
    pub(super) fn add_single_pix_or_copy(&mut self, v: &PixOrCopy, plane_xsize: usize) {
        match v.mode {
            LITERAL => {
                let p = v.arg;
                self.alpha[(p >> 24) as usize] += 1;
                self.red[((p >> 16) & 0xff) as usize] += 1;
                self.literal[((p >> 8) & 0xff) as usize] += 1;
                self.blue[(p & 0xff) as usize] += 1;
            }
            CACHE_IDX => self.literal[NUM_CODES + v.arg as usize] += 1,
            _ => {
                let (code, _, _) = prefix_bits(usize::from(v.len));
                self.literal[NUM_LITERAL + code] += 1;
                let mut dist = v.arg as usize;
                if plane_xsize > 0 {
                    dist = distance_to_plane_code(plane_xsize, dist);
                }
                let (code, _, _) = prefix_bits(dist);
                self.distance[code] += 1;
            }
        }
    }

    /// GetPopulationInfo, histogram_enc.c:337.
    pub(super) fn pop(&self, i: usize) -> &[u32] {
        match i {
            LITERAL_I => &self.literal,
            RED_I => &self.red,
            BLUE_I => &self.blue,
            ALPHA_I => &self.alpha,
            _ => &self.distance,
        }
    }

    fn pop_mut(&mut self, i: usize) -> &mut [u32] {
        match i {
            LITERAL_I => &mut self.literal,
            RED_I => &mut self.red,
            BLUE_I => &mut self.blue,
            ALPHA_I => &mut self.alpha,
            _ => &mut self.distance,
        }
    }

    /// VP8LHistogramEstimateBits, histogram_enc.c:395.
    pub(super) fn estimate_bits(&self) -> u64 {
        let mut cost = 0u64;
        for i in 0..5 {
            cost = cost.wrapping_add(population_cost(self.pop(i), None, None));
        }
        let extra = extra_cost(&self.literal[NUM_LITERAL..NUM_CODES])
            .wrapping_add(extra_cost(&self.distance));
        cost.wrapping_add(u64::from(extra) << LOG_BITS)
    }

    /// UpdateHistogramCost, histogram_enc.c:491.
    fn update_cost(&mut self, bit_cost: u64, costs: &[u64; 5]) {
        self.bit_cost = bit_cost;
        self.costs = *costs;
    }

    /// ComputeHistogramCost, histogram_enc.c:576.
    fn compute_cost(&mut self) {
        for i in 0..5 {
            let mut t = self.trivial_symbol[i];
            let mut used = self.is_used[i];
            let c = population_cost(self.pop(i), Some(&mut t), Some(&mut used));
            self.trivial_symbol[i] = t;
            self.is_used[i] = used;
            self.costs[i] = c;
        }
        self.bit_cost = self.costs[LITERAL_I]
            .wrapping_add(self.costs[RED_I])
            .wrapping_add(self.costs[BLUE_I])
            .wrapping_add(self.costs[ALPHA_I])
            .wrapping_add(self.costs[DISTANCE_I]);
    }
}

/// BitsEntropyRefine, histogram_enc.c:246.
fn bits_entropy_refine(b: &BitEntropy) -> u64 {
    let mix: u64 = if b.nonzeros < 5 {
        if b.nonzeros <= 1 {
            return 0;
        }
        if b.nonzeros == 2 {
            return div_round(
                (99u64
                    .wrapping_mul(u64::from(b.sum) << LOG_BITS)
                    .wrapping_add(b.entropy)) as i64,
                100,
            ) as u64;
        }
        if b.nonzeros == 3 { 950 } else { 700 }
    } else {
        627
    };
    // The uint32 wrap is C's.
    let min_limit = u64::from(b.sum.wrapping_mul(2).wrapping_sub(b.max_val)) << LOG_BITS;
    let min_limit = div_round(
        (mix.wrapping_mul(min_limit)
            .wrapping_add((1000 - mix).wrapping_mul(b.entropy))) as i64,
        1000,
    ) as u64;
    if b.entropy < min_limit {
        min_limit
    } else {
        b.entropy
    }
}

/// VP8LBitsEntropy, histogram_enc.c:282.
pub(super) fn bits_entropy(a: &[u32]) -> u64 {
    let mut b = BitEntropy::default();
    bits_entropy_unrefined(a, &mut b);
    bits_entropy_refine(&b)
}

/// FinalHuffmanCost, histogram_enc.c:299; uint32 extra as in C.
fn final_huffman_cost(s: &Streaks) -> u64 {
    let mut retval = initial_huffman_cost();
    let mut extra = (s.counts[0]
        .wrapping_mul(1600)
        .wrapping_add(240i32.wrapping_mul(s.streaks[0][1]))) as u32;
    extra = extra.wrapping_add(
        (s.counts[1]
            .wrapping_mul(2640)
            .wrapping_add(720i32.wrapping_mul(s.streaks[1][1]))) as u32,
    );
    extra = extra.wrapping_add(1840i32.wrapping_mul(s.streaks[0][0]) as u32);
    extra = extra.wrapping_add(3360i32.wrapping_mul(s.streaks[1][0]) as u32);
    retval = retval.wrapping_add(u64::from(extra) << (LOG_BITS - 10));
    retval
}

/// InitialHuffmanCost, histogram_enc.c:289: 19 * 3 << 23 minus DivRound(91 << 23, 10).
fn initial_huffman_cost() -> u64 {
    (((CODE_LEN_CODES * 3) as u64) << LOG_BITS).wrapping_sub(div_round(91 << LOG_BITS, 10) as u64)
}

/// PopulationCost, histogram_enc.c:319: `trivial_sym` and `is_used` may be None.
fn population_cost(pop: &[u32], trivial_sym: Option<&mut u16>, is_used: Option<&mut bool>) -> u64 {
    let mut b = BitEntropy::default();
    let mut s = Streaks::default();
    entropy_unrefined(pop, &mut b, &mut s);
    if let Some(t) = trivial_sym {
        *t = if b.nonzeros == 1 {
            b.nonzero_code as u16
        } else {
            NON_TRIVIAL_SYM
        };
    }
    if let Some(u) = is_used {
        *u = s.streaks[1][0] != 0 || s.streaks[1][1] != 0;
    }
    bits_entropy_refine(&b).wrapping_add(final_huffman_cost(&s))
}

/// GetCombinedEntropy, histogram_enc.c:369.
fn get_combined_entropy(h1: &Histo, h2: &Histo, i: usize) -> u64 {
    let (used1, used2) = (h1.is_used[i], h2.is_used[i]);
    let trivial =
        h1.trivial_symbol[i] != NON_TRIVIAL_SYM && h1.trivial_symbol[i] == h2.trivial_symbol[i];
    if trivial || !used1 || !used2 {
        if used1 {
            return h1.costs[i];
        }
        return h2.costs[i];
    }
    let mut b = BitEntropy::default();
    let mut s = Streaks::default();
    combined_entropy_unrefined(h1.pop(i), h2.pop(i), &mut b, &mut s);
    bits_entropy_refine(&b).wrapping_add(final_huffman_cost(&s))
}

/// SaturateAdd, histogram_enc.c:416.
fn saturate_add(a: u64, b: &mut i64) {
    if *b < 0 || a as i64 <= i64::MAX - *b {
        *b = b.wrapping_add(a as i64);
    } else {
        *b = i64::MAX;
    }
}

/// GetCombinedHistogramEntropy, histogram_enc.c:426.
fn get_combined_histogram_entropy(
    a: &Histo,
    b: &Histo,
    threshold_in: i64,
    cost: &mut u64,
    costs: &mut [u64; 5],
) -> bool {
    let threshold = threshold_in as u64;
    if threshold_in <= 0 {
        return false;
    }
    *cost = 0;
    for i in 0..5 {
        costs[i] = get_combined_entropy(a, b, i);
        *cost = cost.wrapping_add(costs[i]);
        if *cost >= threshold {
            return false;
        }
    }
    true
}

/// HistogramAdd, histogram_enc.c:446, for distinct h1, h2 and out.
fn histogram_add3(h1: &Histo, h2: &Histo, out: &mut Histo) {
    for i in 0..5 {
        let (p1, p2) = (h1.pop(i), h2.pop(i));
        let po = out.pop_mut(i);
        if h1.is_used[i] {
            if h2.is_used[i] {
                for k in 0..po.len() {
                    po[k] = p1[k].wrapping_add(p2[k]);
                }
            } else {
                po.copy_from_slice(p1);
            }
        } else if h2.is_used[i] {
            po.copy_from_slice(p2);
        } else {
            po.fill(0);
        }
    }
    merge_stats(h1, h2, out);
}

/// HistogramAdd with h2 == out: `out += h1`.
fn histogram_add_into(h1: &Histo, out: &mut Histo) {
    let stats = (out.trivial_symbol, out.is_used);
    for i in 0..5 {
        if h1.is_used[i] {
            let p1 = h1.pop(i);
            let used = out.is_used[i];
            let po = out.pop_mut(i);
            if used {
                for k in 0..po.len() {
                    po[k] = po[k].wrapping_add(p1[k]);
                }
            } else {
                po.copy_from_slice(p1);
            }
        }
    }
    for i in 0..5 {
        out.trivial_symbol[i] = if h1.trivial_symbol[i] == stats.0[i] {
            h1.trivial_symbol[i]
        } else {
            NON_TRIVIAL_SYM
        };
        out.is_used[i] = h1.is_used[i] || stats.1[i];
    }
}

fn merge_stats(h1: &Histo, h2: &Histo, out: &mut Histo) {
    for i in 0..5 {
        out.trivial_symbol[i] = if h1.trivial_symbol[i] == h2.trivial_symbol[i] {
            h1.trivial_symbol[i]
        } else {
            NON_TRIVIAL_SYM
        };
        out.is_used[i] = h1.is_used[i] || h2.is_used[i];
    }
}

/// HistogramAddEval, histogram_enc.c:508.
fn histogram_add_eval(a: &Histo, b: &Histo, out: &mut Histo, mut threshold: i64) -> bool {
    let sum_cost = a.bit_cost.wrapping_add(b.bit_cost);
    let mut bit_cost = 0u64;
    let mut costs = [0u64; 5];
    saturate_add(sum_cost, &mut threshold);
    if !get_combined_histogram_entropy(a, b, threshold, &mut bit_cost, &mut costs) {
        return false;
    }
    histogram_add3(a, b, out);
    out.update_cost(bit_cost, &costs);
    true
}

/// HistogramAddThresh, histogram_enc.c:529.
fn histogram_add_thresh(a: &Histo, b: &Histo, mut threshold: i64, cost_out: &mut i64) -> bool {
    let mut cost = 0u64;
    let mut costs = [0u64; 5];
    saturate_add(a.bit_cost, &mut threshold);
    if !get_combined_histogram_entropy(a, b, threshold, &mut cost, &mut costs) {
        return false;
    }
    *cost_out = (cost as i64).wrapping_sub(a.bit_cost as i64);
    true
}

/// DominantCostRange, histogram_enc.c:548.
struct CostRange {
    literal_max: u64,
    literal_min: u64,
    red_max: u64,
    red_min: u64,
    blue_max: u64,
    blue_min: u64,
}

impl CostRange {
    fn new() -> CostRange {
        CostRange {
            literal_max: 0,
            literal_min: u64::MAX,
            red_max: 0,
            red_min: u64::MAX,
            blue_max: 0,
            blue_min: u64::MAX,
        }
    }

    /// UpdateDominantCostRange, histogram_enc.c:566.
    fn update(&mut self, h: &Histo) {
        self.literal_max = self.literal_max.max(h.costs[LITERAL_I]);
        self.literal_min = self.literal_min.min(h.costs[LITERAL_I]);
        self.red_max = self.red_max.max(h.costs[RED_I]);
        self.red_min = self.red_min.min(h.costs[RED_I]);
        self.blue_max = self.blue_max.max(h.costs[BLUE_I]);
        self.blue_min = self.blue_min.min(h.costs[BLUE_I]);
    }
}

/// GetBinIdForEntropy, histogram_enc.c:591; double as in C.
fn get_bin_id_for_entropy(lo: u64, hi: u64, val: u64) -> usize {
    let rng = hi.wrapping_sub(lo);
    if rng > 0 {
        let delta = val.wrapping_sub(lo);
        return ((BIN_ID_FACTOR * delta as f64) / rng as f64) as i64 as usize;
    }
    0
}

/// GetHistoBinIndex, histogram_enc.c:601, low_effort 0.
fn get_histo_bin_index(h: &Histo, c: &CostRange) -> usize {
    let mut bin_id = get_bin_id_for_entropy(c.literal_min, c.literal_max, h.costs[LITERAL_I]);
    bin_id = bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(c.red_min, c.red_max, h.costs[RED_I]);
    bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(c.blue_min, c.blue_max, h.costs[BLUE_I])
}

/// VP8LHistogramSet: `pool` holds the histograms, `h[..size]` the live ones by position (an index
/// into the pool, or None once an `orig` entry is emptied).
pub(super) struct HistoSet {
    pub pool: Vec<Histo>,
    pub h: Vec<Option<usize>>,
    pub size: usize,
}

impl HistoSet {
    pub(super) fn new() -> HistoSet {
        HistoSet {
            pool: Vec::new(),
            h: Vec::new(),
            size: 0,
        }
    }

    /// VP8LAllocateHistogramSet: n histograms for `cache_bits`, stats cleared.
    pub(super) fn alloc(&mut self, cx: &Ledger, n: usize, cache_bits: usize) -> Result<(), Error> {
        // Each histogram owns a heap `literal`: give back the old ones, charge the new ones.
        let freed: usize = self.pool.iter().map(|p| 4 * p.literal.capacity()).sum();
        cx.give(freed as u64);
        self.pool.clear();
        self.h.clear();
        cx.take((n * 4 * num_codes(cache_bits)) as u64)?;
        cx.reserve(&mut self.pool, n)?;
        cx.reserve(&mut self.h, n)?;
        for i in 0..n {
            let mut p = Histo::new(cache_bits);
            p.init(cache_bits, false);
            self.pool.push(p);
            self.h.push(Some(i));
        }
        self.size = n;
        Ok(())
    }

    /// VP8LHistogramSetClear: zeroes every histogram and resets the pointers.
    fn set_clear(&mut self) {
        for i in 0..self.pool.len() {
            self.pool[i].zero();
            self.h[i] = Some(i);
        }
        self.size = self.h.len();
    }

    pub(super) fn get(&self, i: usize) -> &Histo {
        &self.pool[self.h[i].unwrap_or(0)]
    }

    pub(super) fn get_mut(&mut self, i: usize) -> &mut Histo {
        let k = self.h[i].unwrap_or(0);
        &mut self.pool[k]
    }

    /// HistogramSetRemoveHistogram, histogram_enc.c:195.
    fn remove(&mut self, i: usize) {
        self.h[i] = self.h[self.size - 1];
        self.size -= 1;
    }

    /// `out += a` for the histograms at positions a and out (a != out).
    fn add_into(&mut self, a: usize, out: usize) {
        let (ia, io) = (self.h[a].unwrap_or(0), self.h[out].unwrap_or(0));
        let (x, y) = pair_mut(&mut self.pool, ia, io);
        histogram_add_into(x, y);
    }
}

/// Two distinct elements of the pool, the first shared and the second exclusive.
fn pair_mut(pool: &mut [Histo], i: usize, j: usize) -> (&Histo, &mut Histo) {
    if i < j {
        let (l, r) = pool.split_at_mut(j);
        (&l[i], &mut r[0])
    } else {
        let (l, r) = pool.split_at_mut(i);
        (&r[0], &mut l[j])
    }
}

/// HistogramBuild, histogram_enc.c:617.
fn histogram_build(
    cx: &Ledger,
    xsize: usize,
    histo_bits: usize,
    refs: &[PixOrCopy],
    set: &mut HistoSet,
) -> Result<(), Error> {
    let (mut x, mut y) = (0usize, 0usize);
    let histo_xsize = sub_sample_size(xsize, histo_bits);
    set.set_clear();
    for (k, v) in refs.iter().enumerate() {
        if k % CHECK_EVERY == 0 {
            cx.check()?;
        }
        let ix = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
        set.get_mut(ix).add_single_pix_or_copy(v, 0);
        x += usize::from(v.len);
        while x >= xsize {
            x -= xsize;
            y += 1;
        }
    }
    Ok(())
}

/// HistogramCopyAndAnalyze, histogram_enc.c:640. Empty `orig` entries become None.
fn histogram_copy_and_analyze(orig: &mut HistoSet, image: &mut HistoSet) {
    image.size = 0;
    for i in 0..orig.h.len() {
        let k = orig.h[i].unwrap_or(0);
        orig.pool[k].compute_cost();
        if !orig.pool[k].is_used.iter().any(|&u| u) {
            orig.h[i] = None;
        } else {
            let k2 = image.h[image.size].unwrap_or(0);
            image.pool[k2] = orig.pool[k].clone();
            image.size += 1;
        }
    }
}

/// HistogramAnalyzeEntropyBin, histogram_enc.c:669, low_effort 0.
fn histogram_analyze_entropy_bin(set: &mut HistoSet) {
    let mut c = CostRange::new();
    for i in 0..set.size {
        c.update(set.get(i));
    }
    for i in 0..set.size {
        let b = get_histo_bin_index(set.get(i), &c) as u16;
        set.get_mut(i).bin_id = b;
    }
}

/// HistogramCombineEntropyBin, histogram_enc.c:693, low_effort 0. `cur_combo` is swapped with the
/// first histogram of a bin when they combine.
fn histogram_combine_entropy_bin(
    set: &mut HistoSet,
    cur_combo: &mut Histo,
    num_bins: usize,
    factor: i32,
) {
    const MAX_COMBINE_FAILURES: u16 = 32;
    const NT: u16 = NON_TRIVIAL_SYM;
    let mut bin_info = [(-1i32, 0u16); BIN_SIZE]; // first, failures
    for b in bin_info.iter_mut().take(num_bins) {
        *b = (-1, 0);
    }
    let mut idx = 0;
    while idx < set.size {
        let bin_id = set.get(idx).bin_id as usize;
        let first = bin_info[bin_id].0;
        if first == -1 {
            bin_info[bin_id].0 = idx as i32;
            idx += 1;
            continue;
        }
        let first = first as usize;
        let bit_cost_thresh = -div_round(
            (set.get(idx).bit_cost as i64).wrapping_mul(i64::from(factor)),
            100,
        );
        if !histogram_add_eval(set.get(first), set.get(idx), cur_combo, bit_cost_thresh) {
            idx += 1;
            continue;
        }
        let c = &cur_combo.trivial_symbol;
        let mut try_combine = c[RED_I] != NT && c[BLUE_I] != NT && c[ALPHA_I] != NT;
        if !try_combine {
            let (a, b) = (&set.get(idx).trivial_symbol, &set.get(first).trivial_symbol);
            try_combine = (a[RED_I] == NT || a[BLUE_I] == NT || a[ALPHA_I] == NT)
                && (b[RED_I] == NT || b[BLUE_I] == NT || b[ALPHA_I] == NT);
        }
        if try_combine || bin_info[bin_id].1 >= MAX_COMBINE_FAILURES {
            let k = set.h[first].unwrap_or(0);
            std::mem::swap(cur_combo, &mut set.pool[k]);
            set.remove(idx);
        } else {
            bin_info[bin_id].1 += 1;
            idx += 1;
        }
    }
}

/// MyRand, histogram_enc.c:721.
fn my_rand(seed: &mut u32) -> u32 {
    *seed = (u64::from(*seed) * 48271 % 2147483647) as u32;
    *seed
}

/// HistogramPair, histogram_enc.c:781.
#[derive(Clone, Copy, Default)]
struct Pair {
    idx1: usize,
    idx2: usize,
    cost_diff: i64,
    cost_combo: u64,
    costs: [u64; 5],
}

/// HistoQueue, histogram_enc.c:789.
#[derive(Default)]
pub(super) struct Queue {
    q: Vec<Pair>,
    size: usize,
    max: usize,
}

impl Queue {
    /// HistoQueueInit, histogram_enc.c:795.
    fn init(&mut self, cx: &Ledger, n: usize) -> Result<(), Error> {
        self.size = 0;
        self.max = n;
        cx.resize(&mut self.q, n + 1, Pair::default())?;
        Ok(())
    }

    /// HistoQueuePopPair, histogram_enc.c:814.
    fn pop_pair(&mut self, i: usize) {
        self.q[i] = self.q[self.size - 1];
        self.size -= 1;
    }

    /// HistoQueueUpdateHead, histogram_enc.c:824.
    fn update_head(&mut self, i: usize) {
        if self.q[i].cost_diff < self.q[0].cost_diff {
            self.q.swap(0, i);
        }
    }

    /// HistoQueuePush, histogram_enc.c:871.
    fn push(&mut self, set: &HistoSet, idx1: usize, idx2: usize, threshold: i64) -> i64 {
        if self.size == self.max {
            return 0;
        }
        let (idx1, idx2) = if idx1 > idx2 {
            (idx2, idx1)
        } else {
            (idx1, idx2)
        };
        let mut pair = Pair {
            idx1,
            idx2,
            ..Pair::default()
        };
        if !update_pair(set.get(idx1), set.get(idx2), threshold, &mut pair) {
            return 0;
        }
        self.q[self.size] = pair;
        self.size += 1;
        self.update_head(self.size - 1);
        pair.cost_diff
    }
}

/// HistoQueueFixPair, histogram_enc.c:839.
fn fix_pair(bad_id: usize, good_id: usize, p: &mut Pair) {
    if p.idx1 == bad_id {
        p.idx1 = good_id;
    }
    if p.idx2 == bad_id {
        p.idx2 = good_id;
    }
    if p.idx1 > p.idx2 {
        std::mem::swap(&mut p.idx1, &mut p.idx2);
    }
}

/// HistoQueueUpdatePair, histogram_enc.c:854.
fn update_pair(h1: &Histo, h2: &Histo, mut threshold: i64, p: &mut Pair) -> bool {
    let sum_cost = h1.bit_cost.wrapping_add(h2.bit_cost) as i64;
    saturate_add(sum_cost as u64, &mut threshold);
    if !get_combined_histogram_entropy(h1, h2, threshold, &mut p.cost_combo, &mut p.costs) {
        return false;
    }
    p.cost_diff = (p.cost_combo as i64).wrapping_sub(sum_cost);
    true
}

/// HistogramCombineGreedy, histogram_enc.c:904.
fn histogram_combine_greedy(cx: &Ledger, q: &mut Queue, set: &mut HistoSet) -> Result<(), Error> {
    let size = set.size;
    q.init(cx, size * size)?;
    for i in 0..size {
        cx.check()?;
        for j in i + 1..size {
            q.push(set, i, j, 0);
        }
    }
    while q.size > 0 {
        cx.check()?;
        let (idx1, idx2) = (q.q[0].idx1, q.q[0].idx2);
        set.add_into(idx2, idx1);
        let (cc, costs) = (q.q[0].cost_combo, q.q[0].costs);
        set.get_mut(idx1).update_cost(cc, &costs);
        set.remove(idx2);

        // Remove pairs intersecting the merged pair.
        let mut i = 0;
        while i < q.size {
            let mut p = q.q[i];
            if p.idx1 == idx1 || p.idx2 == idx1 || p.idx1 == idx2 || p.idx2 == idx2 {
                q.pop_pair(i);
            } else {
                fix_pair(set.size, idx2, &mut p);
                q.q[i] = p;
                q.update_head(i);
                i += 1;
            }
        }

        for i in 0..set.size {
            if i == idx1 {
                continue;
            }
            q.push(set, idx1, i, 0);
        }
    }
    Ok(())
}

/// HistogramCombineStochastic, histogram_enc.c:970. Returns whether to run the greedy pass.
fn histogram_combine_stochastic(
    cx: &Ledger,
    q: &mut Queue,
    set: &mut HistoSet,
    min_cluster_size: i64,
) -> Result<bool, Error> {
    const HISTO_QUEUE_SIZE: usize = 9;
    let mut seed = 1u32;
    let mut tries_with_no_success = 0usize;
    let outer_iters = set.size;
    let num_tries_no_success = outer_iters / 2;

    if (set.size as i64) < min_cluster_size {
        return Ok(true);
    }
    q.init(cx, HISTO_QUEUE_SIZE)?;

    let mut iter = 0usize;
    loop {
        cx.check()?;
        // C's for condition, ++tries only if the rest holds.
        if !(iter < outer_iters && set.size as i64 >= min_cluster_size) {
            break;
        }
        tries_with_no_success += 1;
        if tries_with_no_success >= num_tries_no_success {
            break;
        }

        let mut best_cost = 0i64;
        if q.size != 0 {
            best_cost = q.q[0].cost_diff;
        }
        let rand_range = ((set.size - 1) * set.size) as u32;
        let num_tries = set.size / 2;

        let mut j = 0;
        while set.size >= 2 && j < num_tries {
            let tmp = my_rand(&mut seed) % rand_range;
            let idx1 = tmp / (set.size - 1) as u32;
            let mut idx2 = tmp % (set.size - 1) as u32;
            if idx2 >= idx1 {
                idx2 += 1;
            }
            let curr_cost = q.push(set, idx1 as usize, idx2 as usize, best_cost);
            if curr_cost < 0 {
                best_cost = curr_cost;
                if q.size == q.max {
                    break;
                }
            }
            j += 1;
        }
        if q.size == 0 {
            iter += 1;
            continue;
        }

        let (best_idx1, best_idx2) = (q.q[0].idx1, q.q[0].idx2);
        set.add_into(best_idx2, best_idx1);
        let (cc, costs) = (q.q[0].cost_combo, q.q[0].costs);
        set.get_mut(best_idx1).update_cost(cc, &costs);
        set.remove(best_idx2);

        let mut j = 0;
        while j < q.size {
            let mut p = q.q[j];
            let is_idx1_best = p.idx1 == best_idx1 || p.idx1 == best_idx2;
            let is_idx2_best = p.idx2 == best_idx1 || p.idx2 == best_idx2;
            // The front pair may be a duplicate from a random pick.
            if is_idx1_best && is_idx2_best {
                q.pop_pair(j);
                continue;
            }
            if is_idx1_best || is_idx2_best {
                fix_pair(best_idx2, best_idx1, &mut p);
                if !update_pair(set.get(p.idx1), set.get(p.idx2), 0, &mut p) {
                    q.pop_pair(j);
                    continue;
                }
            }
            fix_pair(set.size, best_idx2, &mut p);
            q.q[j] = p;
            q.update_head(j);
            j += 1;
        }
        tries_with_no_success = 0;
        iter += 1;
    }
    Ok(set.size as i64 <= min_cluster_size)
}

/// HistogramRemap, histogram_enc.c:1078.
fn histogram_remap(
    cx: &Ledger,
    in_: &HistoSet,
    out: &mut HistoSet,
    symbols: &mut [u32],
) -> Result<(), Error> {
    let in_size = out.h.len();
    let out_size = out.size;
    if out_size > 1 {
        for i in 0..in_size {
            cx.check()?;
            if in_.h[i].is_none() {
                // The previous id helps later LZ77.
                symbols[i] = symbols[i - 1];
                continue;
            }
            let mut best_out = 0usize;
            let mut best_bits = i64::MAX;
            for k in 0..out_size {
                let mut cur_bits = 0i64;
                if histogram_add_thresh(out.get(k), in_.get(i), best_bits, &mut cur_bits) {
                    best_bits = cur_bits;
                    best_out = k;
                }
            }
            symbols[i] = best_out as u32;
        }
    } else {
        symbols[..in_size].fill(0);
    }

    // Recompute each out from in and symbols.
    out.set_clear();
    out.size = out_size;
    for i in 0..in_size {
        if in_.h[i].is_none() {
            continue;
        }
        let idx = symbols[i] as usize;
        let k = out.h[idx].unwrap_or(0);
        histogram_add_into(in_.get(i), &mut out.pool[k]);
    }
    Ok(())
}

/// GetCombineCostFactor, histogram_enc.c:1125.
fn get_combine_cost_factor(histo_size: usize, quality: i32) -> i32 {
    let mut f = 16;
    if quality < 90 {
        if histo_size > 256 {
            f /= 2;
        }
        if histo_size > 512 {
            f /= 2;
        }
        if histo_size > 1024 {
            f /= 2;
        }
        if quality <= 50 {
            f /= 2;
        }
    }
    f
}

/// Scratch of the clustering.
#[derive(Default)]
pub(super) struct Scratch {
    orig: Option<HistoSet>,
    queue: Queue,
}

/// VP8LGetHistoImageSymbols, histogram_enc.c:1136, low_effort 0. `orig` is orig_histo.
#[allow(clippy::too_many_arguments)]
pub(super) fn get_histo_image_symbols(
    cx: &Ledger,
    sc: &mut Scratch,
    xsize: usize,
    ysize: usize,
    refs: &[PixOrCopy],
    quality: i32,
    histo_bits: usize,
    cache_bits: usize,
    image: &mut HistoSet,
    tmp: &mut Histo,
    symbols: &mut [u32],
) -> Result<(), Error> {
    let raw_size = sub_sample_size(xsize, histo_bits) * sub_sample_size(ysize, histo_bits);
    let mut orig = sc.orig.take().unwrap_or_else(HistoSet::new);
    orig.alloc(cx, raw_size, cache_bits)?;

    histogram_build(cx, xsize, histo_bits, refs, &mut orig)?;
    histogram_copy_and_analyze(&mut orig, image);
    if image.size > BIN_SIZE * 2 && quality < 100 {
        let factor = get_combine_cost_factor(raw_size, quality);
        histogram_analyze_entropy_bin(image);
        histogram_combine_entropy_bin(image, tmp, BIN_SIZE, factor);
    }

    // Cubic ramp between 1 and MAX_HISTO_GREEDY.
    let q = i64::from(quality);
    let threshold_size = 1 + div_round(q * q * q * (MAX_HISTO_GREEDY - 1), 100 * 100 * 100);
    if histogram_combine_stochastic(cx, &mut sc.queue, image, threshold_size)? {
        histogram_combine_greedy(cx, &mut sc.queue, image)?;
    }
    histogram_remap(cx, &orig, image, symbols)?;
    sc.orig = Some(orig);
    Ok(())
}
