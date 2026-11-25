//! Port of libwebp 1.2.2 enc/backward_references_cost_enc.c, without the colour cache. Copyright
//! 2012 Google Inc. BSD-3, see LICENSE.libwebp.

use super::hist::{Histo, NUM_CODES};
use super::lossless::fast_log2;
use super::refs::{CHECK_EVERY, HashChain, PixOrCopy, Refs, distance_to_plane_code};
use super::{COPY, LITERAL, MAX_LENGTH, NUM_DISTANCE, NUM_LITERAL, cap_bytes, prefix_bits};
use crate::Error;
use crate::limits::Ledger;

const COST_MUL1: f64 = 0.82; // mul1, AddSingleLiteralWithCostModel
const COST_INIT: f32 = 1e38; // 1e38f, CostManagerInit
const SKIP_DISTANCE: usize = 10; // kSkipDistance, PushInterval
const COST_INTERVAL_MAX: usize = 500; // COST_CACHE_INTERVAL_SIZE_MAX
const NIL: i32 = -1;

/// ConvertPopulationCountTableToBitEstimates, backward_references_cost_enc.c:40.
fn convert_population_count_table_to_bit_estimates(counts: &[u32], out: &mut [f64]) {
    let mut sum = 0u32;
    let mut nonzeros = 0;
    for &c in counts {
        sum += c;
        if c > 0 {
            nonzeros += 1;
        }
    }
    if nonzeros <= 1 {
        out[..counts.len()].fill(0.0);
    } else {
        let logsum = f64::from(fast_log2(sum));
        for (i, &c) in counts.iter().enumerate() {
            out[i] = logsum - f64::from(fast_log2(c));
        }
    }
}

/// CostModel, backward_references_cost_enc.c:32.
pub(super) struct CostModel {
    alpha: [f64; 256],
    red: [f64; 256],
    blue: [f64; 256],
    distance: [f64; NUM_DISTANCE],
    literal: [f64; NUM_CODES],
}

/// VP8LPrefixEncodeBits; kPrefixEncodeCode[0] is {0, 0}.
pub(super) fn prefix_code_bits(v: usize) -> (usize, usize) {
    if v == 0 {
        return (0, 0);
    }
    let (code, extra, _) = prefix_bits(v);
    (code, extra)
}

impl CostModel {
    /// CostModelBuild, backward_references_cost_enc.c:61, with plane-code distances.
    fn build(&mut self, h: &mut Histo, xsize: usize, refs: &[PixOrCopy]) {
        h.init(true);
        for r in refs {
            h.add_single_pix_or_copy(r, xsize);
        }
        convert_population_count_table_to_bit_estimates(&h.literal, &mut self.literal);
        convert_population_count_table_to_bit_estimates(&h.red, &mut self.red);
        convert_population_count_table_to_bit_estimates(&h.blue, &mut self.blue);
        convert_population_count_table_to_bit_estimates(&h.alpha, &mut self.alpha);
        convert_population_count_table_to_bit_estimates(&h.distance, &mut self.distance);
    }

    /// GetLiteralCost, backward_references_cost_enc.c:95.
    fn literal_cost(&self, v: u32) -> f64 {
        self.alpha[(v >> 24) as usize]
            + self.red[((v >> 16) & 0xff) as usize]
            + self.literal[((v >> 8) & 0xff) as usize]
            + self.blue[(v & 0xff) as usize]
    }

    /// GetLengthCost, backward_references_cost_enc.c:107.
    fn length_cost(&self, n: u32) -> f64 {
        let (code, extra) = prefix_code_bits(n as usize);
        self.literal[NUM_LITERAL + code] + extra as f64
    }

    /// GetDistanceCost, backward_references_cost_enc.c:114.
    fn distance_cost(&self, d: u32) -> f64 {
        let (code, extra) = prefix_code_bits(d as usize);
        self.distance[code] + extra as f64
    }
}

/// CostInterval: cost for [start, end) from index.
#[derive(Clone, Copy)]
struct CostInterval {
    cost: f32,
    start: usize,
    end: usize,
    index: usize,
    prev: i32,
    next: i32,
}

/// CostCacheInterval; end is exclusive.
#[derive(Clone, Copy)]
struct CostCacheInterval {
    cost: f64,
    start: usize,
    end: usize,
}

/// CostManager. `ivs` is the interval store; the live list never exceeds COST_INTERVAL_MAX.
pub(super) struct CostManager {
    head: i32,
    count: usize,
    cache_intervals: Vec<CostCacheInterval>,
    cost_cache: Vec<f64>, // GetLengthCost(k)
    costs: Vec<f32>,
    pub dist_array: Vec<u16>,
    ivs: Vec<CostInterval>,
    free: i32,
}

/// Scratch of the cost-based backward references.
pub(super) struct CostScratch {
    model: CostModel,
    histo: Histo,
    mgr: CostManager,
}

impl Default for CostScratch {
    fn default() -> CostScratch {
        CostScratch {
            model: CostModel {
                alpha: [0.0; 256],
                red: [0.0; 256],
                blue: [0.0; 256],
                distance: [0.0; NUM_DISTANCE],
                literal: [0.0; NUM_CODES],
            },
            histo: Histo::new(),
            mgr: CostManager {
                head: NIL,
                count: 0,
                cache_intervals: Vec::new(),
                cost_cache: Vec::new(),
                costs: Vec::new(),
                dist_array: Vec::new(),
                ivs: Vec::new(),
                free: NIL,
            },
        }
    }
}

impl CostScratch {
    /// The bytes held, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        let m = &self.mgr;
        cap_bytes(&m.cache_intervals)
            + cap_bytes(&m.cost_cache)
            + cap_bytes(&m.costs)
            + cap_bytes(&m.dist_array)
            + cap_bytes(&m.ivs)
    }
}

impl CostManager {
    /// AddSingleLiteralWithCostModel, backward_references_cost_enc.c:121, no cache.
    fn add_single_literal(&mut self, argb: &[u32], m: &CostModel, idx: usize, prev_cost: f32) {
        let mut cost_val = f64::from(prev_cost);
        cost_val += m.literal_cost(argb[idx]) * COST_MUL1;
        if f64::from(self.costs[idx]) > cost_val {
            self.costs[idx] = cost_val as f32;
            self.dist_array[idx] = 1;
        }
    }

    /// CostIntervalAddToFreeList, backward_references_cost_enc.c:202.
    fn add_to_free_list(&mut self, iv: usize) {
        self.ivs[iv].next = self.free;
        self.free = iv as i32;
    }

    /// Pops the free list; the store never runs out (count <= 500). InsertInterval,
    /// backward_references_cost_enc.c:431-444.
    fn new_interval(&mut self) -> usize {
        let iv = self.free as usize;
        self.free = self.ivs[iv].next;
        iv
    }

    /// CostManagerInit, backward_references_cost_enc.c:250 (and InitFreeList :214).
    fn init(&mut self, cx: &Ledger, pix_count: usize, m: &CostModel) -> Result<(), Error> {
        let cache_size = pix_count.min(MAX_LENGTH);
        self.head = NIL;
        self.count = 0;
        self.free = NIL;
        cx.resize(
            &mut self.ivs,
            COST_INTERVAL_MAX,
            CostInterval {
                cost: 0.0,
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

        cx.resize(&mut self.cost_cache, MAX_LENGTH, 0.0)?;
        self.cost_cache[0] = m.length_cost(0);
        for i in 1..cache_size {
            self.cost_cache[i] = m.length_cost(i as u32);
        }

        // Equal neighbours share one interval. One interval per length prefix code at most.
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

        cx.resize(&mut self.costs, pix_count, COST_INIT)?;
        Ok(())
    }

    /// UpdateCost, backward_references_cost_enc.c:321.
    fn update_cost(&mut self, i: usize, position: usize, cost: f32) {
        let k = i - position;
        if self.costs[i] > cost {
            self.costs[i] = cost;
            self.dist_array[i] = (k + 1) as u16;
        }
    }

    /// UpdateCostPerInterval, backward_references_cost_enc.c:334.
    fn update_cost_per_interval(&mut self, start: usize, end: usize, position: usize, cost: f32) {
        for i in start..end {
            self.update_cost(i, position, cost);
        }
    }

    /// ConnectIntervals, backward_references_cost_enc.c:342.
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

    /// PopInterval, backward_references_cost_enc.c:355.
    fn pop_interval(&mut self, iv: i32) {
        if iv == NIL {
            return;
        }
        let (p, n) = (self.ivs[iv as usize].prev, self.ivs[iv as usize].next);
        self.connect_intervals(p, n);
        self.add_to_free_list(iv as usize);
        self.count -= 1;
    }

    /// UpdateCostAtIndex, backward_references_cost_enc.c:374.
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

    /// PositionOrphanInterval, backward_references_cost_enc.c:395.
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

    /// InsertInterval, backward_references_cost_enc.c:419.
    fn insert_interval(&mut self, in_: i32, cost: f32, position: usize, start: usize, end: usize) {
        if start >= end {
            return;
        }
        if self.count >= COST_INTERVAL_MAX {
            self.update_cost_per_interval(start, end, position, cost);
            return;
        }
        let iv = self.new_interval();
        self.ivs[iv] = CostInterval {
            cost,
            start,
            end,
            index: position,
            prev: self.ivs[iv].prev,
            next: self.ivs[iv].next,
        };
        self.position_orphan_interval(iv as i32, in_);
        self.count += 1;
    }

    /// PushInterval, backward_references_cost_enc.c:459.
    fn push_interval(&mut self, dist_cost: f64, position: usize, length: usize) {
        let mut iv = self.head;

        if length < SKIP_DISTANCE {
            for j in position..position + length {
                let k = j - position;
                let cost_tmp = (dist_cost + self.cost_cache[k]) as f32;
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
            let cost = (dist_cost + ci.cost) as f32;

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

/// BackwardReferencesHashChainDistanceOnly, backward_references_cost_enc.c:563: fills
/// `sc.mgr.dist_array`.
fn distance_only(
    cx: &Ledger,
    sc: &mut CostScratch,
    xsize: usize,
    ysize: usize,
    argb: &[u32],
    hc: &HashChain,
    refs: &[PixOrCopy],
) -> Result<(), Error> {
    let pix_count = xsize * ysize;
    let (mut offset_prev, mut len_prev) = (-1i64, -1i64);
    let mut offset_cost = -1.0f64;
    let mut first_offset_is_constant = -1i32;
    let mut reach = 0i64;

    sc.model.build(&mut sc.histo, xsize, refs);
    let m = &sc.model;
    let cm = &mut sc.mgr;
    cm.init(cx, pix_count, m)?;
    cx.resize(&mut cm.dist_array, pix_count, 0)?;

    cm.dist_array[0] = 0;
    cm.add_single_literal(argb, m, 0, 0.0);

    for i in 1..pix_count {
        if i % CHECK_EVERY == 0 {
            cx.check()?;
        }
        let prev_cost = cm.costs[i - 1];
        let (offset, length) = (hc.find_offset(i), hc.find_length(i));

        cm.add_single_literal(argb, m, i, prev_cost);

        if length >= 2 {
            if offset as i64 != offset_prev {
                let code = distance_to_plane_code(xsize, offset);
                offset_cost = m.distance_cost(code as u32);
                first_offset_is_constant = 1;
                cm.push_interval(f64::from(prev_cost) + offset_cost, i, length);
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

                    cm.push_interval(f64::from(cm.costs[j - 1]) + offset_cost, j, len_j);
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

/// TraceBackwards, backward_references_cost_enc.c:689: packs the path at the end of dist_array
/// and returns where it starts.
pub(super) fn trace_backwards(dist_array: &mut [u16]) -> usize {
    let mut p = dist_array.len();
    let mut cur = dist_array.len() as i64 - 1;
    while cur >= 0 {
        let k = dist_array[cur as usize];
        p -= 1;
        dist_array[p] = k;
        cur -= i64::from(k);
    }
    p
}

/// BackwardReferencesHashChainFollowChosenPath, backward_references_cost_enc.c:705.
fn follow_chosen_path(
    cx: &Ledger,
    argb: &[u32],
    path: &[u16],
    hc: &HashChain,
    refs: &mut Refs,
) -> Result<(), Error> {
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
            i += n;
        } else {
            cx.push(
                refs,
                PixOrCopy {
                    mode: LITERAL,
                    len: 1,
                    arg: argb[i],
                },
            )?;
            i += 1;
        }
    }
    Ok(())
}

/// VP8LBackwardReferencesTraceBackwards, backward_references_cost_enc.c:761.
pub(super) fn trace_backwards_refs(
    cx: &Ledger,
    sc: &mut CostScratch,
    xsize: usize,
    ysize: usize,
    argb: &[u32],
    hc: &HashChain,
    src: &[PixOrCopy],
    dst: &mut Refs,
) -> Result<(), Error> {
    distance_only(cx, sc, xsize, ysize, argb, hc, src)?;
    let p = trace_backwards(&mut sc.mgr.dist_array);
    follow_chosen_path(cx, argb, &sc.mgr.dist_array[p..], hc, dst)?;
    Ok(())
}
