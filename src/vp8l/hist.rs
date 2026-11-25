//! Port of libwebp 1.2.2 enc/histogram_enc.c. Copyright 2012 Google Inc. BSD-3, see
//! LICENSE.libwebp.

use super::lossless::{
    BitEntropy, NON_TRIVIAL_SYM, Streaks, bits_entropy_unrefined, extra_cost, extra_cost_combined,
    get_combined_entropy_unrefined, get_entropy_unrefined, histogram_add_eq, histogram_add3,
    sub_sample_size,
};
use super::refs::{CHECK_EVERY, PixOrCopy, distance_to_plane_code};
use super::{LITERAL, NUM_DISTANCE, NUM_LITERAL, cap_bytes, prefix_bits};
use crate::Error;
use crate::limits::Ledger;

pub(super) const NUM_CODES: usize = NUM_LITERAL + 24; // VP8LHistogramNumCodes(0)
const CODE_LEN_CODES: usize = 19;

const MAX_COST: f64 = 1.0e38;
const NUM_PARTITIONS: usize = 4;
pub(super) const BIN_SIZE: usize = NUM_PARTITIONS * NUM_PARTITIONS * NUM_PARTITIONS;
const MAX_HISTO_GREEDY: i32 = 100;
const INVALID_HISTO_SYM: u16 = 0xffff;
const SMALL_BIAS: f64 = 9.1;
const BIN_ID_EPS: f64 = 1e-6;
const BIN_ID_FACTOR: f64 = NUM_PARTITIONS as f64 - BIN_ID_EPS;

/// VP8LHistogram with cache_bits 0.
#[derive(Clone)]
pub(super) struct Histo {
    /// Green, then length prefixes.
    pub literal: [u32; NUM_CODES],
    pub red: [u32; NUM_LITERAL],
    pub blue: [u32; NUM_LITERAL],
    pub alpha: [u32; NUM_LITERAL],
    pub distance: [u32; NUM_DISTANCE],
    pub trivial_symbol: u32,
    pub bit_cost: f64,
    pub literal_cost: f64,
    pub red_cost: f64,
    pub blue_cost: f64,
    /// Literal, red, blue, alpha, distance.
    pub is_used: [bool; 5],
}

impl Histo {
    pub(super) fn new() -> Histo {
        Histo {
            literal: [0; NUM_CODES],
            red: [0; NUM_LITERAL],
            blue: [0; NUM_LITERAL],
            alpha: [0; NUM_LITERAL],
            distance: [0; NUM_DISTANCE],
            trivial_symbol: 0,
            bit_cost: 0.0,
            literal_cost: 0.0,
            red_cost: 0.0,
            blue_cost: 0.0,
            is_used: [false; 5],
        }
    }

    /// The five histogram arrays in `is_used` order.
    pub(super) fn part(&self, k: usize) -> &[u32] {
        match k {
            0 => &self.literal,
            1 => &self.red,
            2 => &self.blue,
            3 => &self.alpha,
            _ => &self.distance,
        }
    }

    pub(super) fn part_mut(&mut self, k: usize) -> &mut [u32] {
        match k {
            0 => &mut self.literal,
            1 => &mut self.red,
            2 => &mut self.blue,
            3 => &mut self.alpha,
            _ => &mut self.distance,
        }
    }

    /// VP8LHistogramInit with `arrays` clears the arrays too.
    pub(super) fn init(&mut self, arrays: bool) {
        if arrays {
            *self = Histo::new();
            return;
        }
        self.trivial_symbol = 0;
        self.bit_cost = 0.0;
        self.literal_cost = 0.0;
        self.red_cost = 0.0;
        self.blue_cost = 0.0;
        self.is_used = [false; 5];
    }

    /// VP8LHistogramStoreRefs.
    pub(super) fn store_refs(&mut self, refs: &[PixOrCopy]) {
        for r in refs {
            self.add_single_pix_or_copy(r, 0);
        }
    }

    /// VP8LHistogramCreate.
    pub(super) fn create(&mut self, refs: &[PixOrCopy]) {
        *self = Histo::new();
        self.store_refs(refs);
    }

    /// VP8LHistogramAddSinglePixOrCopy: maps distances through `distance_to_plane_code` when
    /// `plane_xsize` > 0 (CostModelBuild), else counts them raw.
    pub(super) fn add_single_pix_or_copy(&mut self, v: &PixOrCopy, plane_xsize: usize) {
        if v.mode == LITERAL {
            let p = v.arg;
            self.alpha[(p >> 24) as usize] += 1;
            self.red[((p >> 16) & 0xff) as usize] += 1;
            self.literal[((p >> 8) & 0xff) as usize] += 1;
            self.blue[(p & 0xff) as usize] += 1;
            return;
        }
        let (code, _, _) = prefix_bits(usize::from(v.len));
        self.literal[NUM_LITERAL + code] += 1;
        let mut dist = v.arg as usize;
        if plane_xsize > 0 {
            dist = distance_to_plane_code(plane_xsize, dist);
        }
        let (code, _, _) = prefix_bits(dist);
        self.distance[code] += 1;
    }

    /// VP8LHistogramEstimateBits.
    pub(super) fn estimate_bits(&mut self) -> f64 {
        let a = population_cost(&self.literal, None, &mut self.is_used[0]);
        let b = population_cost(&self.red, None, &mut self.is_used[1]);
        let c = population_cost(&self.blue, None, &mut self.is_used[2]);
        let d = population_cost(&self.alpha, None, &mut self.is_used[3]);
        let e = population_cost(&self.distance, None, &mut self.is_used[4]);
        a + b + c + d + e + extra_cost(&self.literal[NUM_LITERAL..]) + extra_cost(&self.distance)
    }
}

/// BitsEntropyRefine, histogram_enc.c:231.
fn bits_entropy_refine(b: &BitEntropy) -> f64 {
    let mix = if b.nonzeros < 5 {
        if b.nonzeros <= 1 {
            return 0.0;
        }
        if b.nonzeros == 2 {
            return 0.99 * f64::from(b.sum) + 0.01 * b.entropy;
        }
        if b.nonzeros == 3 { 0.95 } else { 0.7 }
    } else {
        0.627
    };
    // The uint32 wrap is C's.
    let min_limit = f64::from(b.sum.wrapping_mul(2).wrapping_sub(b.max_val));
    let min_limit = mix * min_limit + (1.0 - mix) * b.entropy;
    if b.entropy < min_limit {
        min_limit
    } else {
        b.entropy
    }
}

/// VP8LBitsEntropy, histogram_enc.c:263.
pub(super) fn bits_entropy(a: &[u32]) -> f64 {
    let mut b = BitEntropy::default();
    bits_entropy_unrefined(a, &mut b);
    bits_entropy_refine(&b)
}

/// InitialHuffmanCost, histogram_enc.c:270.
fn initial_huffman_cost() -> f64 {
    (CODE_LEN_CODES * 3) as f64 - SMALL_BIAS
}

/// FinalHuffmanCost, histogram_enc.c:279.
fn final_huffman_cost(s: &Streaks) -> f64 {
    let mut retval = initial_huffman_cost();
    retval += f64::from(s.counts[0]) * 1.5625 + 0.234375 * f64::from(s.streaks[0][1]);
    retval += f64::from(s.counts[1]) * 2.578125 + 0.703125 * f64::from(s.streaks[1][1]);
    retval += 1.796875 * f64::from(s.streaks[0][0]);
    retval += 3.28125 * f64::from(s.streaks[1][0]);
    retval
}

/// PopulationCost, histogram_enc.c:299.
fn population_cost(pop: &[u32], trivial_sym: Option<&mut u32>, is_used: &mut bool) -> f64 {
    let mut b = BitEntropy::default();
    let mut s = Streaks::default();
    get_entropy_unrefined(pop, &mut b, &mut s);
    if let Some(t) = trivial_sym {
        *t = if b.nonzeros == 1 {
            b.nonzero_code
        } else {
            NON_TRIVIAL_SYM
        };
    }
    *is_used = s.streaks[1][0] != 0 || s.streaks[1][1] != 0;
    bits_entropy_refine(&b) + final_huffman_cost(&s)
}

/// GetCombinedEntropy, histogram_enc.c:317.
fn get_combined_entropy(
    x: &[u32],
    y: &[u32],
    is_x_used: bool,
    is_y_used: bool,
    trivial_at_end: bool,
) -> f64 {
    let mut s = Streaks::default();
    let length = x.len() as i32;
    if trivial_at_end {
        s.streaks[1][0] = 1;
        s.counts[0] = 1;
        s.streaks[0][1] = length - 1;
        return final_huffman_cost(&s);
    }
    let mut b = BitEntropy::default();
    if is_x_used {
        if is_y_used {
            get_combined_entropy_unrefined(x, y, &mut b, &mut s);
        } else {
            get_entropy_unrefined(x, &mut b, &mut s);
        }
    } else if is_y_used {
        get_entropy_unrefined(y, &mut b, &mut s);
    } else {
        s.counts[0] = 1;
        if length > 3 {
            s.streaks[0][1] = length;
        } else {
            s.streaks[0][0] = length;
        }
        b.init();
    }
    bits_entropy_refine(&b) + final_huffman_cost(&s)
}

/// GetCombinedHistogramEntropy, histogram_enc.c:374. Keeps the early exits.
fn get_combined_histogram_entropy(a: &Histo, b: &Histo, threshold: f64, cost: &mut f64) -> bool {
    let mut trivial_at_end = false;
    *cost += get_combined_entropy(&a.literal, &b.literal, a.is_used[0], b.is_used[0], false);
    *cost += extra_cost_combined(&a.literal[NUM_LITERAL..], &b.literal[NUM_LITERAL..]);
    if *cost > threshold {
        return false;
    }

    if a.trivial_symbol != NON_TRIVIAL_SYM && a.trivial_symbol == b.trivial_symbol {
        // A, R and B are all 0 or 0xff.
        let ca = (a.trivial_symbol >> 24) & 0xff;
        let cr = (a.trivial_symbol >> 16) & 0xff;
        let cb = a.trivial_symbol & 0xff;
        if (ca == 0 || ca == 0xff) && (cr == 0 || cr == 0xff) && (cb == 0 || cb == 0xff) {
            trivial_at_end = true;
        }
    }

    *cost += get_combined_entropy(&a.red, &b.red, a.is_used[1], b.is_used[1], trivial_at_end);
    if *cost > threshold {
        return false;
    }
    *cost += get_combined_entropy(&a.blue, &b.blue, a.is_used[2], b.is_used[2], trivial_at_end);
    if *cost > threshold {
        return false;
    }
    *cost += get_combined_entropy(
        &a.alpha,
        &b.alpha,
        a.is_used[3],
        b.is_used[3],
        trivial_at_end,
    );
    if *cost > threshold {
        return false;
    }
    *cost += get_combined_entropy(&a.distance, &b.distance, a.is_used[4], b.is_used[4], false);
    *cost += extra_cost_combined(&a.distance, &b.distance);
    *cost <= threshold
}

/// The trivial symbol of a merge.
fn merged_trivial(a: &Histo, b: &Histo) -> u32 {
    if a.trivial_symbol == b.trivial_symbol {
        a.trivial_symbol
    } else {
        NON_TRIVIAL_SYM
    }
}

/// HistogramAdd, histogram_enc.c:427, for distinct a, b and out.
fn histogram_add(a: &Histo, b: &Histo, out: &mut Histo) {
    histogram_add3(a, b, out);
    out.trivial_symbol = merged_trivial(a, b);
}

/// HistogramAdd with out == b: `out += a`.
fn histogram_add_into(a: &Histo, out: &mut Histo) {
    let t = merged_trivial(a, out);
    histogram_add_eq(a, out);
    out.trivial_symbol = t;
}

/// HistogramAddEval, histogram_enc.c:442.
fn histogram_add_eval(a: &Histo, b: &Histo, out: &mut Histo, mut threshold: f64) -> f64 {
    let mut cost = 0.0;
    let sum_cost = a.bit_cost + b.bit_cost;
    threshold += sum_cost;
    if get_combined_histogram_entropy(a, b, threshold, &mut cost) {
        histogram_add(a, b, out);
        out.bit_cost = cost;
    }
    cost - sum_cost
}

/// HistogramAddThresh, histogram_enc.c:462.
fn histogram_add_thresh(a: &Histo, b: &Histo, threshold: f64) -> f64 {
    let mut cost = -a.bit_cost;
    get_combined_histogram_entropy(a, b, threshold, &mut cost);
    cost
}

/// DominantCostRange, histogram_enc.c:478.
struct DominantCostRange {
    literal_max: f64,
    literal_min: f64,
    red_max: f64,
    red_min: f64,
    blue_max: f64,
    blue_min: f64,
}

impl DominantCostRange {
    fn new() -> DominantCostRange {
        DominantCostRange {
            literal_max: 0.0,
            literal_min: MAX_COST,
            red_max: 0.0,
            red_min: MAX_COST,
            blue_max: 0.0,
            blue_min: MAX_COST,
        }
    }

    /// UpdateDominantCostRange, histogram_enc.c:496.
    fn update(&mut self, h: &Histo) {
        if self.literal_max < h.literal_cost {
            self.literal_max = h.literal_cost;
        }
        if self.literal_min > h.literal_cost {
            self.literal_min = h.literal_cost;
        }
        if self.red_max < h.red_cost {
            self.red_max = h.red_cost;
        }
        if self.red_min > h.red_cost {
            self.red_min = h.red_cost;
        }
        if self.blue_max < h.blue_cost {
            self.blue_max = h.blue_cost;
        }
        if self.blue_min > h.blue_cost {
            self.blue_min = h.blue_cost;
        }
    }
}

/// UpdateHistogramCost, histogram_enc.c:506.
fn update_histogram_cost(h: &mut Histo) {
    let (mut alpha_sym, mut red_sym, mut blue_sym) = (0u32, 0u32, 0u32);
    let alpha_cost = population_cost(&h.alpha, Some(&mut alpha_sym), &mut h.is_used[3]);
    let distance_cost =
        population_cost(&h.distance, None, &mut h.is_used[4]) + extra_cost(&h.distance);
    h.literal_cost = population_cost(&h.literal, None, &mut h.is_used[0])
        + extra_cost(&h.literal[NUM_LITERAL..]);
    h.red_cost = population_cost(&h.red, Some(&mut red_sym), &mut h.is_used[1]);
    h.blue_cost = population_cost(&h.blue, Some(&mut blue_sym), &mut h.is_used[2]);
    h.bit_cost = h.literal_cost + h.red_cost + h.blue_cost + alpha_cost + distance_cost;
    if alpha_sym | red_sym | blue_sym == NON_TRIVIAL_SYM {
        h.trivial_symbol = NON_TRIVIAL_SYM;
    } else {
        h.trivial_symbol = (alpha_sym << 24) | (red_sym << 16) | blue_sym;
    }
}

/// GetBinIdForEntropy, histogram_enc.c:532.
fn get_bin_id_for_entropy(lo: f64, hi: f64, val: f64) -> usize {
    let rng = hi - lo;
    if rng > 0.0 {
        let delta = val - lo;
        return (BIN_ID_FACTOR * delta / rng) as usize;
    }
    0
}

/// GetHistoBinIndex, histogram_enc.c:542, low_effort 0.
fn get_histo_bin_index(h: &Histo, c: &DominantCostRange) -> usize {
    let mut bin_id = get_bin_id_for_entropy(c.literal_min, c.literal_max, h.literal_cost);
    bin_id = bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(c.red_min, c.red_max, h.red_cost);
    bin_id * NUM_PARTITIONS + get_bin_id_for_entropy(c.blue_min, c.blue_max, h.blue_cost)
}

/// VP8LHistogramSet: `pool` holds the histograms, `h` the live ones by position (None once
/// removed); `max` is max_size.
pub(super) struct HistoSet {
    pub pool: Vec<Histo>,
    pub h: Vec<Option<usize>>,
    pub max: usize,
}

impl HistoSet {
    pub(super) fn new() -> HistoSet {
        HistoSet {
            pool: Vec::new(),
            h: Vec::new(),
            max: 0,
        }
    }

    /// The bytes held, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        cap_bytes(&self.pool) + cap_bytes(&self.h)
    }

    /// VP8LAllocateHistogramSet: n cleared histograms.
    pub(super) fn alloc(&mut self, cx: &Ledger, n: usize) -> Result<(), Error> {
        cx.reserve(&mut self.pool, n)?;
        cx.reserve(&mut self.h, n)?;
        self.pool.resize_with(n, Histo::new);
        self.pool.truncate(n);
        self.max = n;
        self.clear();
        Ok(())
    }

    /// VP8LHistogramSetClear: also undoes pointer moves, as C resets the pointers.
    pub(super) fn clear(&mut self) {
        self.h.clear();
        for i in 0..self.max {
            self.pool[i] = Histo::new();
            self.h.push(Some(i));
        }
    }

    pub(super) fn len(&self) -> usize {
        self.h.len()
    }

    /// The histogram at position i, which must be live.
    pub(super) fn get(&self, i: usize) -> &Histo {
        &self.pool[self.h[i].unwrap_or(0)]
    }

    pub(super) fn get_mut(&mut self, i: usize) -> &mut Histo {
        let k = self.h[i].unwrap_or(0);
        &mut self.pool[k]
    }

    fn live(&self, i: usize) -> bool {
        self.h[i].is_some()
    }

    /// Whether position i is removed, including positions cut off the end of `h`.
    fn dead(&self, i: usize) -> bool {
        self.h.get(i).is_none_or(|h| h.is_none())
    }

    /// HistogramSetRemoveHistogram, histogram_enc.c:184.
    fn remove(&mut self, i: usize, num_used: &mut i32) {
        self.h[i] = None;
        *num_used -= 1;
        if i + 1 == self.h.len() {
            while self.h.last() == Some(&None) {
                self.h.pop();
            }
        }
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

/// HistogramBuild, histogram_enc.c:558. The caller's alloc has cleared the set.
pub(super) fn histogram_build(
    cx: &Ledger,
    xsize: usize,
    histo_bits: usize,
    refs: &[PixOrCopy],
    set: &mut HistoSet,
) -> Result<(), Error> {
    let (mut x, mut y) = (0usize, 0usize);
    let histo_xsize = sub_sample_size(xsize, histo_bits);
    for (k, v) in refs.iter().enumerate() {
        if k % CHECK_EVERY == 0 {
            cx.check()?;
        }
        let ix = (y >> histo_bits) * histo_xsize + (x >> histo_bits);
        set.pool[ix].add_single_pix_or_copy(v, 0);
        x += usize::from(v.len);
        while x >= xsize {
            x -= xsize;
            y += 1;
        }
    }
    Ok(())
}

/// HistogramCopyAndAnalyze, histogram_enc.c:582.
fn histogram_copy_and_analyze(
    orig: &mut HistoSet,
    image: &mut HistoSet,
    num_used: &mut i32,
    symbols: &mut [u16],
) {
    let mut num_used_orig = *num_used;
    let mut cluster_id = 0u16;
    for i in 0..orig.max {
        let k = orig.h[i].unwrap_or(0);
        update_histogram_cost(&mut orig.pool[k]);
        let histo = &orig.pool[k];
        // Empty tiles take the previous id, see HistogramRemap.
        if !histo.is_used.iter().any(|&u| u) {
            image.remove(i, num_used);
            orig.remove(i, &mut num_used_orig);
            symbols[i] = INVALID_HISTO_SYM;
        } else {
            let k2 = image.h[i].unwrap_or(0);
            image.pool[k2] = orig.pool[k].clone();
            symbols[i] = cluster_id;
            cluster_id += 1;
        }
    }
}

/// HistogramAnalyzeEntropyBin, histogram_enc.c:617, low_effort 0.
fn histogram_analyze_entropy_bin(set: &HistoSet, bin_map: &mut [u16]) {
    let mut c = DominantCostRange::new();
    for i in 0..set.len() {
        if set.live(i) {
            c.update(set.get(i));
        }
    }
    for i in 0..set.len() {
        if set.live(i) {
            bin_map[i] = get_histo_bin_index(set.get(i), &c) as u16;
        }
    }
}

/// HistogramCombineEntropyBin, histogram_enc.c:644, low_effort 0. `cur_combo` is swapped with the
/// first histogram of a bin when they combine.
#[allow(clippy::too_many_arguments)]
fn histogram_combine_entropy_bin(
    set: &mut HistoSet,
    num_used: &mut i32,
    clusters: &[u16],
    cluster_mappings: &mut [u16],
    cur_combo: &mut Histo,
    bin_map: &[u16],
    num_bins: usize,
    factor: f64,
) {
    const MAX_COMBINE_FAILURES: u16 = 32;
    let mut bin_info = [(-1i32, 0u16); BIN_SIZE]; // first, failures
    for b in bin_info.iter_mut().take(num_bins) {
        *b = (-1, 0);
    }
    for idx in 0..*num_used as usize {
        cluster_mappings[idx] = idx as u16;
    }
    let mut idx = 0;
    while idx < set.len() {
        if !set.live(idx) {
            idx += 1;
            continue;
        }
        let bin_id = bin_map[idx] as usize;
        let first = bin_info[bin_id].0;
        if first == -1 {
            bin_info[bin_id].0 = idx as i32;
            idx += 1;
            continue;
        }
        let first = first as usize;
        let bit_cost = set.get(idx).bit_cost;
        let bit_cost_thresh = -bit_cost * factor;
        let curr_cost_diff =
            histogram_add_eval(set.get(first), set.get(idx), cur_combo, bit_cost_thresh);
        if curr_cost_diff < bit_cost_thresh {
            let try_combine = cur_combo.trivial_symbol != NON_TRIVIAL_SYM
                || (set.get(idx).trivial_symbol == NON_TRIVIAL_SYM
                    && set.get(first).trivial_symbol == NON_TRIVIAL_SYM);
            if try_combine || bin_info[bin_id].1 >= MAX_COMBINE_FAILURES {
                let k = set.h[first].unwrap_or(0);
                std::mem::swap(cur_combo, &mut set.pool[k]);
                set.remove(idx, num_used);
                cluster_mappings[usize::from(clusters[idx])] = clusters[first];
            } else {
                bin_info[bin_id].1 += 1;
            }
        }
        idx += 1;
    }
}

/// MyRand, histogram_enc.c:721.
fn my_rand(seed: &mut u32) -> u32 {
    *seed = (u64::from(*seed) * 48271 % 2147483647) as u32;
    *seed
}

/// HistogramPair; idx1 < 0 means out of date.
#[derive(Clone, Copy, Default)]
pub(super) struct Pair {
    idx1: usize,
    idx2: usize,
    cost_diff: f64,
    cost_combo: f64,
}

/// HistoQueue; `q` has len max.
#[derive(Default)]
pub(super) struct Queue {
    q: Vec<Pair>,
    size: usize,
    max: usize,
}

impl Queue {
    /// The bytes held, at capacity.
    fn bytes(&self) -> u64 {
        cap_bytes(&self.q)
    }

    /// HistoQueueInit.
    fn init(&mut self, cx: &Ledger, size: usize) -> Result<(), Error> {
        self.size = 0;
        self.max = size;
        cx.resize(&mut self.q, size, Pair::default())?;
        Ok(())
    }

    /// HistoQueuePopPair.
    fn pop_pair(&mut self, i: usize) {
        self.q[i] = self.q[self.size - 1];
        self.size -= 1;
    }

    /// HistoQueueUpdateHead.
    fn update_head(&mut self, i: usize) {
        if self.q[i].cost_diff < self.q[0].cost_diff {
            self.q.swap(0, i);
        }
    }

    /// HistoQueuePush, histogram_enc.c:802.
    fn push(&mut self, set: &HistoSet, idx1: usize, idx2: usize, threshold: f64) -> f64 {
        if self.size == self.max {
            return 0.0;
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
        update_pair(set.get(idx1), set.get(idx2), threshold, &mut pair);
        if pair.cost_diff >= threshold {
            return 0.0;
        }
        self.q[self.size] = pair;
        self.size += 1;
        self.update_head(self.size - 1);
        pair.cost_diff
    }
}

/// HistoQueueUpdatePair, histogram_enc.c:789.
fn update_pair(h1: &Histo, h2: &Histo, threshold: f64, p: &mut Pair) {
    let sum_cost = h1.bit_cost + h2.bit_cost;
    p.cost_combo = 0.0;
    get_combined_histogram_entropy(h1, h2, sum_cost + threshold, &mut p.cost_combo);
    p.cost_diff = p.cost_combo - sum_cost;
}

/// HistogramCombineGreedy, histogram_enc.c:837.
fn histogram_combine_greedy(
    cx: &Ledger,
    q: &mut Queue,
    set: &mut HistoSet,
    num_used: &mut i32,
) -> Result<(), Error> {
    let size = set.len();
    q.init(cx, size * size)?;
    for i in 0..size {
        cx.check()?;
        if !set.live(i) {
            continue;
        }
        for j in i + 1..size {
            if !set.live(j) {
                continue;
            }
            q.push(set, i, j, 0.0);
        }
    }

    while q.size > 0 {
        cx.check()?;
        let (idx1, idx2) = (q.q[0].idx1, q.q[0].idx2);
        set.add_into(idx2, idx1);
        set.get_mut(idx1).bit_cost = q.q[0].cost_combo;
        set.remove(idx2, num_used);

        // Remove pairs intersecting the merged pair.
        let mut i = 0;
        while i < q.size {
            let p = q.q[i];
            if p.idx1 == idx1 || p.idx2 == idx1 || p.idx1 == idx2 || p.idx2 == idx2 {
                q.pop_pair(i);
            } else {
                q.update_head(i);
                i += 1;
            }
        }

        for i in 0..set.len() {
            if i == idx1 || !set.live(i) {
                continue;
            }
            q.push(set, idx1, i, 0.0);
        }
    }
    Ok(())
}

/// HistogramCombineStochastic, histogram_enc.c:908. Returns whether to run the greedy pass.
fn histogram_combine_stochastic(
    cx: &Ledger,
    q: &mut Queue,
    mappings: &mut Vec<usize>,
    set: &mut HistoSet,
    num_used: &mut i32,
    min_cluster_size: i32,
) -> Result<bool, Error> {
    const HISTO_QUEUE_SIZE: usize = 9;
    let mut seed = 1u32;
    let mut tries_with_no_success = 0;
    let outer_iters = *num_used;
    let num_tries_no_success = outer_iters / 2;

    if *num_used < min_cluster_size {
        return Ok(true);
    }

    mappings.clear();
    q.init(cx, HISTO_QUEUE_SIZE)?;
    for iter in 0..set.len() {
        if set.live(iter) {
            cx.push(mappings, iter)?;
        }
    }

    let mut iter = 0;
    loop {
        cx.check()?;
        // C's for condition, ++tries only if the rest holds.
        if !(iter < outer_iters && *num_used >= min_cluster_size) {
            break;
        }
        tries_with_no_success += 1;
        if tries_with_no_success >= num_tries_no_success {
            break;
        }

        let mut best_cost = 0.0;
        if q.size != 0 {
            best_cost = q.q[0].cost_diff;
        }
        let rand_range = ((*num_used - 1) * *num_used) as u32;
        let num_tries = *num_used / 2;

        let mut j = 0;
        while *num_used >= 2 && j < num_tries {
            let tmp = my_rand(&mut seed) % rand_range;
            let mut idx1 = tmp / (*num_used - 1) as u32;
            let mut idx2 = tmp % (*num_used - 1) as u32;
            if idx2 >= idx1 {
                idx2 += 1;
            }
            idx1 = mappings[idx1 as usize] as u32;
            idx2 = mappings[idx2 as usize] as u32;

            let curr_cost = q.push(set, idx1 as usize, idx2 as usize, best_cost);
            if curr_cost < 0.0 {
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
        let used = *num_used as usize;
        let k = mappings[..used]
            .binary_search(&best_idx2)
            .unwrap_or_else(|k| k);
        mappings.copy_within(k + 1..used, k);
        set.add_into(best_idx2, best_idx1);
        set.get_mut(best_idx1).bit_cost = q.q[0].cost_combo;
        set.remove(best_idx2, num_used);

        let mut j = 0;
        while j < q.size {
            let mut p = q.q[j];
            let is_idx1_best = p.idx1 == best_idx1 || p.idx1 == best_idx2;
            let is_idx2_best = p.idx2 == best_idx1 || p.idx2 == best_idx2;
            let mut do_eval = false;
            // The front pair may be a duplicate from a random pick.
            if is_idx1_best && is_idx2_best {
                q.pop_pair(j);
                continue;
            }
            if is_idx1_best {
                p.idx1 = best_idx1;
                do_eval = true;
            } else if is_idx2_best {
                p.idx2 = best_idx1;
                do_eval = true;
            }
            if p.idx1 > p.idx2 {
                std::mem::swap(&mut p.idx1, &mut p.idx2);
            }
            if do_eval {
                update_pair(set.get(p.idx1), set.get(p.idx2), 0.0, &mut p);
                if p.cost_diff >= 0.0 {
                    q.q[j] = p;
                    q.pop_pair(j);
                    continue;
                }
            }
            q.q[j] = p;
            q.update_head(j);
            j += 1;
        }
        tries_with_no_success = 0;
        iter += 1;
    }
    Ok(*num_used <= min_cluster_size)
}

/// HistogramRemap, histogram_enc.c:1049.
fn histogram_remap(
    cx: &Ledger,
    in_: &HistoSet,
    out: &mut HistoSet,
    symbols: &mut [u16],
) -> Result<(), Error> {
    let in_size = out.max;
    let out_size = out.len();
    if out_size > 1 {
        for i in 0..in_size {
            cx.check()?;
            let (mut best_out, mut best_bits) = (0usize, MAX_COST);
            if in_.dead(i) {
                // The previous id helps later LZ77.
                symbols[i] = symbols[i - 1];
                continue;
            }
            for k in 0..out_size {
                let cur_bits = histogram_add_thresh(out.get(k), in_.get(i), best_bits);
                if k == 0 || cur_bits < best_bits {
                    best_bits = cur_bits;
                    best_out = k;
                }
            }
            symbols[i] = best_out as u16;
        }
    } else {
        for s in symbols[..in_size].iter_mut() {
            *s = 0;
        }
    }

    // Recompute each out from in and symbols.
    out.clear();
    out.h.truncate(out_size);
    for i in 0..in_size {
        if in_.dead(i) {
            continue;
        }
        let idx = usize::from(symbols[i]);
        let (src, dst) = (in_.get(i), &mut out.pool[idx]);
        histogram_add_into(src, dst);
    }
    Ok(())
}

/// GetCombineCostFactor, histogram_enc.c:1096.
fn get_combine_cost_factor(histo_size: usize, quality: i32) -> f64 {
    let mut f = 0.16;
    if quality < 90 {
        if histo_size > 256 {
            f /= 2.0;
        }
        if histo_size > 512 {
            f /= 2.0;
        }
        if histo_size > 1024 {
            f /= 2.0;
        }
        if quality <= 50 {
            f /= 2.0;
        }
    }
    f
}

/// OptimizeHistogramSymbols, histogram_enc.c:1110.
fn optimize_histogram_symbols(
    set_max: usize,
    cluster_mappings: &mut [u16],
    num_clusters: usize,
    tmp: &mut [u16],
    symbols: &mut [u16],
) {
    let mut do_continue = true;
    // Assign the lowest cluster to each pixel.
    while do_continue {
        do_continue = false;
        for i in 0..num_clusters {
            let mut k = cluster_mappings[i];
            while k != cluster_mappings[usize::from(k)] {
                cluster_mappings[usize::from(k)] =
                    cluster_mappings[usize::from(cluster_mappings[usize::from(k)])];
                k = cluster_mappings[usize::from(k)];
            }
            if k != cluster_mappings[i] {
                do_continue = true;
                cluster_mappings[i] = k;
            }
        }
    }
    let mut cluster_max = 0u16;
    tmp[..set_max].fill(0);
    for i in 0..set_max {
        if symbols[i] == INVALID_HISTO_SYM {
            continue;
        }
        let cluster = usize::from(cluster_mappings[usize::from(symbols[i])]);
        if cluster > 0 && tmp[cluster] == 0 {
            cluster_max += 1;
            tmp[cluster] = cluster_max;
        }
        symbols[i] = tmp[cluster];
    }
}

/// RemoveEmptyHistograms, histogram_enc.c:1161.
fn remove_empty_histograms(set: &mut HistoSet) {
    let mut size = 0;
    for i in 0..set.h.len() {
        if set.h[i].is_none() {
            continue;
        }
        set.h[size] = set.h[i];
        size += 1;
    }
    set.h.truncate(size);
}

/// Scratch the histogram clustering keeps between calls.
#[derive(Default)]
pub(super) struct HistoScratch {
    pub orig: Option<HistoSet>,
    pub map_tmp: Vec<u16>,
    pub mappings: Vec<usize>,
    pub queue: Queue,
}

impl HistoScratch {
    /// The bytes held, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        self.orig.as_ref().map_or(0, HistoSet::bytes)
            + cap_bytes(&self.map_tmp)
            + cap_bytes(&self.mappings)
            + self.queue.bytes()
    }
}

/// VP8LGetHistoImageSymbols, histogram_enc.c:1171: low_effort 0, no colour cache.
#[allow(clippy::too_many_arguments)]
pub(super) fn get_histo_image_symbols(
    cx: &Ledger,
    sc: &mut HistoScratch,
    xsize: usize,
    ysize: usize,
    refs: &[PixOrCopy],
    quality: i32,
    histo_bits: usize,
    image: &mut HistoSet,
    tmp: &mut Histo,
    symbols: &mut [u16],
) -> Result<(), Error> {
    let (mut histo_xsize, mut histo_ysize) = (1, 1);
    if histo_bits != 0 {
        histo_xsize = sub_sample_size(xsize, histo_bits);
        histo_ysize = sub_sample_size(ysize, histo_bits);
    }
    let raw_size = histo_xsize * histo_ysize;
    let mut orig = sc.orig.take().unwrap_or_else(HistoSet::new);
    orig.alloc(cx, raw_size)?;
    const ENTROPY_COMBINE_NUM_BINS: usize = BIN_SIZE;
    cx.resize(&mut sc.map_tmp, 2 * raw_size, 0)?;
    let (bin_map, cluster_mappings) = sc.map_tmp.split_at_mut(raw_size);
    let mut num_used = raw_size as i32;

    histogram_build(cx, xsize, histo_bits, refs, &mut orig)?;
    histogram_copy_and_analyze(&mut orig, image, &mut num_used, symbols);

    let entropy_combine = num_used > ENTROPY_COMBINE_NUM_BINS as i32 * 2 && quality < 100;
    if entropy_combine {
        let factor = get_combine_cost_factor(raw_size, quality);
        let num_clusters = num_used as usize;
        histogram_analyze_entropy_bin(image, bin_map);
        histogram_combine_entropy_bin(
            image,
            &mut num_used,
            symbols,
            cluster_mappings,
            tmp,
            bin_map,
            ENTROPY_COMBINE_NUM_BINS,
            factor,
        );
        optimize_histogram_symbols(image.max, cluster_mappings, num_clusters, bin_map, symbols);
    }

    // float32 cubic ramp, as C's quality / 100.f.
    let x = quality as f32 / 100.0f32;
    let threshold_size = (1.0f32 + (x * x * x) * (MAX_HISTO_GREEDY - 1) as f32) as i32;
    if histogram_combine_stochastic(
        cx,
        &mut sc.queue,
        &mut sc.mappings,
        image,
        &mut num_used,
        threshold_size,
    )? {
        remove_empty_histograms(image);
        histogram_combine_greedy(cx, &mut sc.queue, image, &mut num_used)?;
    }

    remove_empty_histograms(image);
    histogram_remap(cx, &orig, image, symbols)?;
    sc.orig = Some(orig);
    Ok(())
}
