//! Port of libwebp 1.6.0 enc/backward_references_cost_enc.c (integer costs, colour cache) and the
//! colour cache selection of enc/backward_references_enc.c. LZ77, RLE, Box, the hash chain and 2D
//! locality are the 1.2.2 port. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

use super::hist::{CACHE_IDX, Histo, NUM_CODES, num_codes};
use super::log::{LOG_BITS, div_round, fast_log2};
use crate::Error;
use crate::limits::Ledger;
use crate::vp8l::cost::{prefix_code_bits, trace_backwards};
use crate::vp8l::refs::{
    CHECK_EVERY, HashChain, PixOrCopy, Refs, backward_references_2d_locality,
    backward_references_lz77, backward_references_lz77_box, backward_references_rle,
    distance_to_plane_code,
};
use crate::vp8l::{
    COPY, LITERAL, LZ77_BOX, LZ77_RLE, LZ77_STANDARD, MAX_LENGTH, NUM_DISTANCE, NUM_LITERAL,
};

pub(super) const MAX_CACHE_BITS: usize = 10; // MAX_COLOR_CACHE_BITS
const COST_INTERVAL_MAX: usize = 500; // COST_CACHE_INTERVAL_SIZE_MAX
const SKIP_DISTANCE: usize = 10; // kSkipDistance
const NIL: i32 = -1;

/// VP8LColorCache, utils/color_cache_utils.h.
#[derive(Default)]
pub(super) struct ColorCache {
    colors: Vec<u32>,
    shift: u32,
}

/// VP8LHashPix, color_cache_utils.h:38.
fn hash_pix(argb: u32, shift: u32) -> usize {
    (argb.wrapping_mul(0x1e35_a7bd) >> shift) as usize
}

impl ColorCache {
    /// VP8LColorCacheInit, color_cache_utils.c:25; colors start at 0.
    pub(super) fn init(&mut self, hash_bits: usize) {
        self.colors.clear();
        self.colors.resize(1 << hash_bits, 0);
        self.shift = 32 - hash_bits as u32;
    }

    /// VP8LColorCacheContains, color_cache_utils.h:66.
    fn contains(&self, argb: u32) -> Option<usize> {
        let k = hash_pix(argb, self.shift);
        (self.colors[k] == argb).then_some(k)
    }

    /// VP8LColorCacheInsert, color_cache_utils.h:54.
    fn insert(&mut self, argb: u32) {
        self.colors[hash_pix(argb, self.shift)] = argb;
    }
}

/// CostModel, backward_references_cost_enc.c:35.
struct CostModel {
    alpha: [u32; 256],
    red: [u32; 256],
    blue: [u32; 256],
    distance: [u32; NUM_DISTANCE],
    literal: Vec<u32>,
}

/// ConvertPopulationCountTableToBitEstimates, backward_references_cost_enc.c:43.
fn convert_population(counts: &[u32], out: &mut [u32]) {
    let mut sum = 0u32;
    let mut nonzeros = 0;
    for &c in counts {
        sum = sum.wrapping_add(c);
        if c > 0 {
            nonzeros += 1;
        }
    }
    if nonzeros <= 1 {
        out[..counts.len()].fill(0);
        return;
    }
    let logsum = fast_log2(sum);
    for (i, &c) in counts.iter().enumerate() {
        out[i] = logsum.wrapping_sub(fast_log2(c));
    }
}

impl CostModel {
    /// CostModelBuild, backward_references_cost_enc.c:64.
    fn build(&mut self, h: &mut Histo, xsize: usize, cache_bits: usize, refs: &[PixOrCopy]) {
        h.init(cache_bits, true);
        h.store_refs(refs, xsize);
        self.literal.clear();
        self.literal.resize(num_codes(cache_bits), 0);
        convert_population(&h.literal, &mut self.literal);
        convert_population(&h.red, &mut self.red);
        convert_population(&h.blue, &mut self.blue);
        convert_population(&h.alpha, &mut self.alpha);
        convert_population(&h.distance, &mut self.distance);
    }

    /// GetLiteralCost, backward_references_cost_enc.c:93.
    fn literal_cost(&self, v: u32) -> i64 {
        i64::from(self.alpha[(v >> 24) as usize])
            + i64::from(self.red[((v >> 16) & 0xff) as usize])
            + i64::from(self.literal[((v >> 8) & 0xff) as usize])
            + i64::from(self.blue[(v & 0xff) as usize])
    }

    /// GetCacheCost, backward_references_cost_enc.c:99.
    fn cache_cost(&self, idx: usize) -> i64 {
        i64::from(self.literal[NUM_CODES + idx])
    }

    /// GetLengthCost, backward_references_cost_enc.c:105.
    fn length_cost(&self, n: u32) -> i64 {
        let (code, extra) = prefix_code_bits(n as usize);
        i64::from(self.literal[NUM_LITERAL + code]) + ((extra as i64) << LOG_BITS)
    }

    /// GetDistanceCost, backward_references_cost_enc.c:113.
    fn distance_cost(&self, d: u32) -> i64 {
        let (code, extra) = prefix_code_bits(d as usize);
        i64::from(self.distance[code]) + ((extra as i64) << LOG_BITS)
    }
}

/// CostInterval, backward_references_cost_enc.c:161.
#[derive(Clone, Copy)]
struct CostInterval {
    cost: i64,
    start: usize,
    end: usize,
    index: usize,
    prev: i32,
    next: i32,
}

/// CostCacheInterval; end is exclusive.
#[derive(Clone, Copy)]
struct CostCacheInterval {
    cost: i64,
    start: usize,
    end: usize,
}

/// CostManager.
struct CostManager {
    head: i32,
    count: usize,
    cache_intervals: Vec<CostCacheInterval>,
    cost_cache: Vec<i64>,
    costs: Vec<i64>,
    dist_array: Vec<u16>,
    ivs: Vec<CostInterval>,
    free: i32,
}

impl CostManager {
    fn new() -> CostManager {
        CostManager {
            head: NIL,
            count: 0,
            cache_intervals: Vec::new(),
            cost_cache: Vec::new(),
            costs: Vec::new(),
            dist_array: Vec::new(),
            ivs: Vec::new(),
            free: NIL,
        }
    }

    /// AddSingleLiteralWithCostModel, backward_references_cost_enc.c:121: `cc` is None without a
    /// colour cache.
    fn add_single_literal(
        &mut self,
        argb: &[u32],
        cc: &mut Option<&mut ColorCache>,
        m: &CostModel,
        idx: usize,
        prev_cost: i64,
    ) {
        let mut cost_val = prev_cost;
        let color = argb[idx];
        let ix = cc.as_ref().and_then(|c| c.contains(color));
        if let Some(ix) = ix {
            cost_val += div_round(m.cache_cost(ix) * 68, 100);
        } else {
            if let Some(c) = cc.as_mut() {
                c.insert(color);
            }
            cost_val += div_round(m.literal_cost(color) * 82, 100);
        }
        if self.costs[idx] > cost_val {
            self.costs[idx] = cost_val;
            self.dist_array[idx] = 1;
        }
    }

    fn add_to_free_list(&mut self, iv: usize) {
        self.ivs[iv].next = self.free;
        self.free = iv as i32;
    }

    fn new_interval(&mut self) -> usize {
        let iv = self.free as usize;
        self.free = self.ivs[iv].next;
        iv
    }

    /// CostManagerInit, backward_references_cost_enc.c:249.
    fn init(&mut self, cx: &Ledger, pix_count: usize, m: &CostModel) -> Result<(), Error> {
        let cache_size = pix_count.min(MAX_LENGTH);
        self.head = NIL;
        self.count = 0;
        self.free = NIL;
        cx.resize(
            &mut self.ivs,
            COST_INTERVAL_MAX,
            CostInterval {
                cost: 0,
                start: 0,
                end: 0,
                index: 0,
                prev: NIL,
                next: NIL,
            },
        )?;
        for i in 0..COST_INTERVAL_MAX {
            self.add_to_free_list(i);
        }

        cx.resize(&mut self.cost_cache, MAX_LENGTH, 0)?;
        for i in 0..cache_size {
            self.cost_cache[i] = m.length_cost(i as u32);
        }

        // Equal neighbours share one interval.
        self.cache_intervals.clear();
        cx.push(
            &mut self.cache_intervals,
            CostCacheInterval {
                cost: self.cost_cache[0],
                start: 0,
                end: 1,
            },
        )?;
        for i in 1..cache_size {
            let cost_val = self.cost_cache[i];
            if cost_val != self.cache_intervals[self.cache_intervals.len() - 1].cost {
                cx.push(
                    &mut self.cache_intervals,
                    CostCacheInterval {
                        cost: cost_val,
                        start: i,
                        end: 0,
                    },
                )?;
            }
            let last = self.cache_intervals.len() - 1;
            self.cache_intervals[last].end = i + 1;
        }

        cx.resize(&mut self.costs, pix_count, i64::MAX)?;
        Ok(())
    }

    /// UpdateCost, backward_references_cost_enc.c:327.
    fn update_cost(&mut self, i: usize, position: usize, cost: i64) {
        let k = i - position;
        if self.costs[i] > cost {
            self.costs[i] = cost;
            self.dist_array[i] = (k + 1) as u16;
        }
    }

    fn update_cost_per_interval(&mut self, start: usize, end: usize, position: usize, cost: i64) {
        for i in start..end {
            self.update_cost(i, position, cost);
        }
    }

    fn connect_intervals(&mut self, prev: i32, next: i32) {
        if prev != NIL {
            self.ivs[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.ivs[next as usize].prev = prev;
        }
    }

    fn pop_interval(&mut self, iv: i32) {
        if iv == NIL {
            return;
        }
        let (p, n) = (self.ivs[iv as usize].prev, self.ivs[iv as usize].next);
        self.connect_intervals(p, n);
        self.add_to_free_list(iv as usize);
        self.count -= 1;
    }

    fn update_cost_at_index(&mut self, i: usize, clean: bool) {
        let mut cur = self.head;
        while cur != NIL && self.ivs[cur as usize].start <= i {
            let c = self.ivs[cur as usize];
            let next = c.next;
            if c.end <= i {
                if clean {
                    self.pop_interval(cur);
                }
            } else {
                self.update_cost(i, c.index, c.cost);
            }
            cur = next;
        }
    }

    fn position_orphan_interval(&mut self, cur: i32, mut prev: i32) {
        if prev == NIL {
            prev = self.head;
        }
        let cur_start = self.ivs[cur as usize].start;
        while prev != NIL && cur_start < self.ivs[prev as usize].start {
            prev = self.ivs[prev as usize].prev;
        }
        while prev != NIL {
            let n = self.ivs[prev as usize].next;
            if n != NIL && self.ivs[n as usize].start < cur_start {
                prev = n;
            } else {
                break;
            }
        }
        if prev != NIL {
            let n = self.ivs[prev as usize].next;
            self.connect_intervals(cur, n);
        } else {
            let h = self.head;
            self.connect_intervals(cur, h);
        }
        self.connect_intervals(prev, cur);
    }

    fn insert_interval(&mut self, in_: i32, cost: i64, position: usize, start: usize, end: usize) {
        if start >= end {
            return;
        }
        if self.count >= COST_INTERVAL_MAX {
            self.update_cost_per_interval(start, end, position, cost);
            return;
        }
        let iv = self.new_interval();
        let (p, n) = (self.ivs[iv].prev, self.ivs[iv].next);
        self.ivs[iv] = CostInterval {
            cost,
            start,
            end,
            index: position,
            prev: p,
            next: n,
        };
        self.position_orphan_interval(iv as i32, in_);
        self.count += 1;
    }

    fn push_interval(&mut self, dist_cost: i64, position: usize, length: usize) {
        let mut iv = self.head;

        if length < SKIP_DISTANCE {
            for j in position..position + length {
                let k = j - position;
                let cost_tmp = dist_cost + self.cost_cache[k];
                if self.costs[j] > cost_tmp {
                    self.costs[j] = cost_tmp;
                    self.dist_array[j] = (k + 1) as u16;
                }
            }
            return;
        }

        let mut i = 0;
        while i < self.cache_intervals.len() && self.cache_intervals[i].start < length {
            let ci = self.cache_intervals[i];
            let mut start = position + ci.start;
            let mut end = position + ci.end;
            if ci.end > length {
                end = position + length;
            }
            let cost = dist_cost + ci.cost;

            while iv != NIL && self.ivs[iv as usize].start < end {
                let iv_next = self.ivs[iv as usize].next;
                let cur = self.ivs[iv as usize];

                if start >= cur.end {
                    iv = iv_next;
                    continue;
                }

                if cost >= cur.cost {
                    let start_new = cur.end;
                    self.insert_interval(iv, cost, position, start, cur.start);
                    start = start_new;
                    if start >= end {
                        break;
                    }
                    iv = iv_next;
                    continue;
                }

                if start <= cur.start {
                    if cur.end <= end {
                        self.pop_interval(iv);
                    } else {
                        self.ivs[iv as usize].start = end;
                        break;
                    }
                } else if end < cur.end {
                    let end_original = cur.end;
                    self.ivs[iv as usize].end = start;
                    self.insert_interval(iv, cur.cost, cur.index, end, end_original);
                    iv = self.ivs[iv as usize].next;
                    break;
                } else {
                    self.ivs[iv as usize].end = start;
                }
                iv = iv_next;
            }
            self.insert_interval(iv, cost, position, start, end);
            i += 1;
        }
    }
}

impl CostScratch {
    /// BackwardReferencesHashChainDistanceOnly, backward_references_cost_enc.c:569: fills
    /// `cost_mgr.dist_array`.
    fn distance_only(
        &mut self,
        cx: &Ledger,
        xsize: usize,
        ysize: usize,
        argb: &[u32],
        cache_bits: usize,
        hc: &HashChain,
        refs: &[PixOrCopy],
    ) -> Result<(), Error> {
        let pix_count = xsize * ysize;
        let mut cc = if cache_bits > 0 {
            self.cc.init(cache_bits);
            Some(&mut self.cc)
        } else {
            None
        };
        let (mut offset_prev, mut len_prev) = (-1i64, -1i64);
        let mut offset_cost = -1i64;
        let mut first_offset_is_constant = -1i32;
        let mut reach = 0i64;

        self.cost_model
            .build(&mut self.cost_histo, xsize, cache_bits, refs);
        let m = &self.cost_model;
        let cm = &mut self.cost_mgr;
        cm.init(cx, pix_count, m)?;
        cx.resize(&mut cm.dist_array, pix_count, 0)?;

        cm.dist_array[0] = 0;
        cm.add_single_literal(argb, &mut cc, m, 0, 0);

        for i in 1..pix_count {
            if i % CHECK_EVERY == 0 {
                cx.check()?;
            }
            let prev_cost = cm.costs[i - 1];
            let (offset, length) = (hc.find_offset(i), hc.find_length(i));

            cm.add_single_literal(argb, &mut cc, m, i, prev_cost);

            if length >= 2 {
                if offset as i64 != offset_prev {
                    let code = distance_to_plane_code(xsize, offset);
                    offset_cost = m.distance_cost(code as u32);
                    first_offset_is_constant = 1;
                    cm.push_interval(prev_cost + offset_cost, i, length);
                } else {
                    if first_offset_is_constant != 0 {
                        reach = i as i64 - 1 + len_prev - 1;
                        first_offset_is_constant = 0;
                    }

                    if i as i64 + length as i64 - 1 > reach {
                        let mut len_j = 0usize;
                        let mut j = i;
                        while (j as i64) <= reach {
                            let (offset_j, l) = (hc.find_offset(j + 1), hc.find_length(j + 1));
                            len_j = l;
                            if offset_j != offset {
                                len_j = hc.find_length(j);
                                break;
                            }
                            j += 1;
                        }
                        cm.update_cost_at_index(j - 1, false);
                        cm.update_cost_at_index(j, false);

                        cm.push_interval(cm.costs[j - 1] + offset_cost, j, len_j);
                        reach = j as i64 + len_j as i64 - 1;
                    }
                }
            }

            cm.update_cost_at_index(i, true);
            offset_prev = offset as i64;
            len_prev = length as i64;
        }
        Ok(())
    }

    /// BackwardReferencesHashChainFollowChosenPath, backward_references_cost_enc.c:711.
    fn follow_chosen_path(
        &mut self,
        cx: &Ledger,
        argb: &[u32],
        cache_bits: usize,
        start: usize,
        hc: &HashChain,
        refs: &mut Refs,
    ) -> Result<(), Error> {
        let mut cc = if cache_bits > 0 {
            self.cc.init(cache_bits);
            Some(&mut self.cc)
        } else {
            None
        };
        let path = &self.cost_mgr.dist_array[start..];
        let mut i = 0;
        refs.clear();
        for (k, &l) in path.iter().enumerate() {
            if k % CHECK_EVERY == 0 {
                cx.check()?;
            }
            let n = usize::from(l);
            if n != 1 {
                let offset = hc.find_offset(i);
                cx.push(
                    refs,
                    PixOrCopy {
                        mode: COPY,
                        len: n as u16,
                        arg: offset as u32,
                    },
                )?;
                if let Some(c) = cc.as_mut() {
                    for k in 0..n {
                        c.insert(argb[i + k]);
                    }
                }
                i += n;
                continue;
            }
            let idx = cc.as_ref().and_then(|c| c.contains(argb[i]));
            if let Some(idx) = idx {
                cx.push(
                    refs,
                    PixOrCopy {
                        mode: CACHE_IDX,
                        len: 1,
                        arg: idx as u32,
                    },
                )?;
            } else {
                if let Some(c) = cc.as_mut() {
                    c.insert(argb[i]);
                }
                cx.push(
                    refs,
                    PixOrCopy {
                        mode: LITERAL,
                        len: 1,
                        arg: argb[i],
                    },
                )?;
            }
            i += 1;
        }
        Ok(())
    }

    /// VP8LBackwardReferencesTraceBackwards, backward_references_cost_enc.c:767.
    #[allow(clippy::too_many_arguments)]
    fn trace_backwards_refs(
        &mut self,
        cx: &Ledger,
        xsize: usize,
        ysize: usize,
        argb: &[u32],
        cache_bits: usize,
        hc: &HashChain,
        src: &[PixOrCopy],
        dst: &mut Refs,
    ) -> Result<(), Error> {
        self.distance_only(cx, xsize, ysize, argb, cache_bits, hc, src)?;
        let p = trace_backwards(&mut self.cost_mgr.dist_array);
        self.follow_chosen_path(cx, argb, cache_bits, p, hc, dst)?;
        Ok(())
    }
}

/// The scratch of the cost-based references.
struct CostScratch {
    cc: ColorCache,
    cost_model: CostModel,
    cost_histo: Histo,
    cost_mgr: CostManager,
}

/// The scratch of the 1.6.0 reference search.
pub(super) struct RefsScratch {
    est: Histo,
    cache_histos: Vec<Histo>,
    caches: Vec<ColorCache>,
    cost: CostScratch,
    pub(super) hash_first: Vec<i32>,
    pub(super) hc_box: HashChain,
    box_counts: Vec<u16>,
}

impl RefsScratch {
    pub(super) fn new() -> RefsScratch {
        RefsScratch {
            est: Histo::new(MAX_CACHE_BITS),
            cache_histos: Vec::new(),
            caches: Vec::new(),
            cost: CostScratch {
                cc: ColorCache::default(),
                cost_model: CostModel {
                    alpha: [0; 256],
                    red: [0; 256],
                    blue: [0; 256],
                    distance: [0; NUM_DISTANCE],
                    literal: Vec::new(),
                },
                cost_histo: Histo::new(0),
                cost_mgr: CostManager::new(),
            },
            hash_first: Vec::new(),
            hc_box: HashChain::default(),
            box_counts: Vec::new(),
        }
    }

    /// CalculateBestCacheSize, backward_references_enc.c:755: the cache bits in [0, bits_max] with
    /// the lowest estimate.
    fn calculate_best_cache_size(
        &mut self,
        cx: &Ledger,
        argb: &[u32],
        quality: i32,
        refs: &[PixOrCopy],
        mut bits_max: usize,
    ) -> Result<usize, Error> {
        if quality <= 25 {
            bits_max = 0;
        }
        if bits_max == 0 {
            return Ok(0);
        }
        self.cache_histos.clear();
        self.caches.clear();
        for i in 0..=bits_max {
            let mut h = Histo::new(i);
            h.init(i, true);
            self.cache_histos.push(h);
            let mut c = ColorCache::default();
            if i > 0 {
                c.init(i);
            }
            self.caches.push(c);
        }
        let histos = &mut self.cache_histos;
        let caches = &mut self.caches;

        let mut pos = 0usize;
        for (k, v) in refs.iter().enumerate() {
            if k % CHECK_EVERY == 0 {
                cx.check()?;
            }
            if v.mode == LITERAL {
                let pix = argb[pos];
                pos += 1;
                let (a, r, g, b) = (
                    (pix >> 24) as usize,
                    ((pix >> 16) & 0xff) as usize,
                    ((pix >> 8) & 0xff) as usize,
                    (pix & 0xff) as usize,
                );
                // Shorter keys are prefixes of the longest.
                let mut key = hash_pix(pix, 32 - bits_max as u32);
                histos[0].blue[b] += 1;
                histos[0].literal[g] += 1;
                histos[0].red[r] += 1;
                histos[0].alpha[a] += 1;
                for i in (1..=bits_max).rev() {
                    let h = &mut histos[i];
                    if caches[i].colors[key] == pix {
                        h.literal[NUM_CODES + key] += 1;
                    } else {
                        caches[i].colors[key] = pix;
                        h.blue[b] += 1;
                        h.literal[g] += 1;
                        h.red[r] += 1;
                        h.alpha[a] += 1;
                    }
                    key >>= 1;
                }
                continue;
            }
            // Only the length prefix differs by cache size.
            let mut n = usize::from(v.len);
            let mut argb_prev = argb[pos] ^ 0xffff_ffff;
            let (code, _, _) = crate::vp8l::prefix_bits(n);
            for h in histos.iter_mut() {
                h.literal[NUM_LITERAL + code] += 1;
            }
            while n != 0 {
                if argb[pos] != argb_prev {
                    let mut key = hash_pix(argb[pos], 32 - bits_max as u32);
                    for i in (1..=bits_max).rev() {
                        caches[i].colors[key] = argb[pos];
                        key >>= 1;
                    }
                    argb_prev = argb[pos];
                }
                pos += 1;
                n -= 1;
            }
        }

        let mut best = 0;
        let mut entropy_min = u64::MAX;
        for (i, h) in histos.iter().enumerate() {
            let entropy = h.estimate_bits();
            if i == 0 || entropy < entropy_min {
                entropy_min = entropy;
                best = i;
            }
        }
        Ok(best)
    }

    /// BackwardRefsWithLocalCache, backward_references_enc.c:861: rewrites literals as cache hits.
    fn backward_refs_with_local_cache(
        &mut self,
        cx: &Ledger,
        argb: &[u32],
        cache_bits: usize,
        refs: &mut Refs,
    ) -> Result<(), Error> {
        let cc = &mut self.cost.cc;
        cc.init(cache_bits);
        let mut pos = 0usize;
        for (k, v) in refs.iter_mut().enumerate() {
            if k % CHECK_EVERY == 0 {
                cx.check()?;
            }
            if v.mode == LITERAL {
                let lit = v.arg;
                if let Some(ix) = cc.contains(lit) {
                    *v = PixOrCopy {
                        mode: CACHE_IDX,
                        len: 1,
                        arg: ix as u32,
                    };
                } else {
                    cc.insert(lit);
                }
                pos += 1;
            } else {
                for _ in 0..v.len {
                    cc.insert(argb[pos]);
                    pos += 1;
                }
            }
        }
        Ok(())
    }

    /// VP8LGetBackwardReferences (backward_references_enc.c:1039, GetBackwardReferences :911), low
    /// effort 0. Leaves the best refs, as plane codes, in refs[0] and returns their cache bits;
    /// refs[1] is refs_tmp.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_backward_references(
        &mut self,
        cx: &Ledger,
        w: usize,
        h: usize,
        argb: &[u32],
        quality: i32,
        lz77_types: u32,
        cache_bits_max: usize,
        hc: &HashChain,
        refs: &mut [Refs],
    ) -> Result<usize, Error> {
        let mut lz77_type_best = 0;
        let mut cache_bits_best = 0;
        let mut bit_cost_best = u64::MAX;
        let mut types = lz77_types;
        let mut lz77_type = 1u32;
        while types != 0 {
            if types & lz77_type != 0 {
                let tmp = &mut refs[1];
                match lz77_type {
                    LZ77_RLE => backward_references_rle(cx, w, h, argb, tmp)?,
                    LZ77_STANDARD => backward_references_lz77(cx, w, h, argb, hc, tmp)?,
                    LZ77_BOX => {
                        self.hc_box.init(cx, w * h)?;
                        backward_references_lz77_box(
                            cx,
                            &mut self.box_counts,
                            w,
                            h,
                            argb,
                            hc,
                            &mut self.hc_box,
                            tmp,
                        )?;
                    }
                    _ => {}
                }
                // i == 0 only: no do_no_cache pass.
                let cache_bits =
                    self.calculate_best_cache_size(cx, argb, quality, &refs[1], cache_bits_max)?;
                if cache_bits > 0 {
                    self.backward_refs_with_local_cache(cx, argb, cache_bits, &mut refs[1])?;
                }
                self.est.create(&refs[1], Some(cache_bits));
                let bit_cost = self.est.estimate_bits();
                if bit_cost < bit_cost_best {
                    refs.swap(0, 1);
                    bit_cost_best = bit_cost;
                    lz77_type_best = lz77_type;
                    cache_bits_best = cache_bits;
                }
            }
            types &= !lz77_type;
            lz77_type <<= 1;
        }

        // TraceBackwards only for quality >= 25.
        if (lz77_type_best == LZ77_STANDARD || lz77_type_best == LZ77_BOX) && quality >= 25 {
            let chain = if lz77_type_best == LZ77_STANDARD {
                hc
            } else {
                &self.hc_box
            };
            let (r0, r1) = refs.split_at_mut(1);
            let (src, dst) = (&r0[0], &mut r1[0]);
            self.cost
                .trace_backwards_refs(cx, w, h, argb, cache_bits_best, chain, src, dst)?;
            self.est.create(&refs[1], Some(cache_bits_best));
            if self.est.estimate_bits() < bit_cost_best {
                refs.swap(0, 1);
            }
        }
        backward_references_2d_locality(w, &mut refs[0]);
        Ok(cache_bits_best)
    }
}
