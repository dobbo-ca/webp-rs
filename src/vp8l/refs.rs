//! Port of libwebp 1.2.2 enc/backward_references_enc.c, without the colour cache. Copyright 2012
//! Google Inc. BSD-3, see LICENSE.libwebp.

use super::cost::{CostScratch, trace_backwards_refs};
use super::hist::Histo;
use super::lossless::vector_mismatch;
use super::{
    COPY, LITERAL, LZ77_BOX, LZ77_RLE, LZ77_STANDARD, MAX_LENGTH, MAX_LENGTH_BITS, cap_bytes,
};
use crate::Error;
use crate::limits::Ledger;

const HASH_BITS: u32 = 18;
pub(super) const HASH_SIZE: usize = 1 << HASH_BITS;
const WINDOW_SIZE: usize = (1 << 20) - 120;
const MIN_LENGTH: usize = 4;
const WINDOW_OFFSETS_MAX: usize = 32;
/// The passes over the pixels look at the cancel flag once this many pixels.
pub(super) const CHECK_EVERY: usize = 1 << 10;

/// PixOrCopy: a literal argb or a copy (len, distance).
#[derive(Clone, Copy, Default)]
pub(super) struct PixOrCopy {
    pub mode: u8,
    pub len: u16,
    /// argb, raw distance, or plane code after 2D locality.
    pub arg: u32,
}

/// VP8LBackwardRefs as one flat vector, same iteration order.
pub(super) type Refs = Vec<PixOrCopy>;

/// VP8LHashChain: offset << 12 | length per pixel. `hash_chain_fill` uses it as an int32 chain
/// first, cast through uint32.
#[derive(Default)]
pub(super) struct HashChain {
    pub off_len: Vec<u32>,
}

impl HashChain {
    pub(super) fn find_offset(&self, i: usize) -> usize {
        (self.off_len[i] >> MAX_LENGTH_BITS) as usize
    }

    pub(super) fn find_length(&self, i: usize) -> usize {
        (self.off_len[i] & ((1 << MAX_LENGTH_BITS) - 1)) as usize
    }

    /// The bytes held, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        cap_bytes(&self.off_len)
    }

    /// VP8LHashChainInit.
    pub(super) fn init(&mut self, cx: &Ledger, size: usize) -> Result<(), Error> {
        cx.resize(&mut self.off_len, size, 0)?;
        Ok(())
    }
}

/// plane_to_code_lut, backward_references_enc.c:38.
const PLANE_TO_CODE_LUT: [u8; 128] = [
    96, 73, 55, 39, 23, 13, 5, 1, 255, 255, 255, 255, 255, 255, 255, 255, 101, 78, 58, 42, 26, 16,
    8, 2, 0, 3, 9, 17, 27, 43, 59, 79, 102, 86, 62, 46, 32, 20, 10, 6, 4, 7, 11, 21, 33, 47, 63,
    87, 105, 90, 70, 52, 37, 28, 18, 14, 12, 15, 19, 29, 38, 53, 71, 91, 110, 99, 82, 66, 48, 35,
    30, 24, 22, 25, 31, 36, 49, 67, 83, 100, 115, 108, 94, 76, 64, 50, 44, 40, 34, 41, 45, 51, 65,
    77, 95, 109, 118, 113, 103, 92, 80, 68, 60, 56, 54, 57, 61, 69, 81, 93, 104, 114, 119, 116,
    111, 106, 97, 88, 84, 74, 72, 75, 85, 89, 98, 107, 112, 117,
];

/// VP8LDistanceToPlaneCode, backward_references_enc.c:50.
pub(super) fn distance_to_plane_code(xsize: usize, dist: usize) -> usize {
    let yoffset = dist / xsize;
    let xoffset = dist - yoffset * xsize;
    if xoffset <= 8 && yoffset < 8 {
        usize::from(PLANE_TO_CODE_LUT[yoffset * 16 + 8 - xoffset]) + 1
    } else if xoffset + 8 > xsize && yoffset < 7 {
        usize::from(PLANE_TO_CODE_LUT[(yoffset + 1) * 16 + 8 + (xsize - xoffset)]) + 1
    } else {
        dist + 120
    }
}

/// FindMatchLength, backward_references_enc.c:66.
pub(super) fn find_match_length(
    a1: &[u32],
    a2: &[u32],
    best_len: usize,
    max_limit: usize,
) -> usize {
    if a1[best_len] != a2[best_len] {
        return 0;
    }
    vector_mismatch(a1, a2, max_limit)
}

/// GetPixPairHash64, backward_references_enc.c:229.
fn get_pix_pair_hash64(a0: u32, a1: u32) -> usize {
    let mut key = a1.wrapping_mul(0xc6a4_a793);
    key = key.wrapping_add(a0.wrapping_mul(0x5bd1_e996));
    (key >> (32 - HASH_BITS)) as usize
}

/// GetMaxItersForQuality, backward_references_enc.c:239.
fn get_max_iters_for_quality(quality: i32) -> i32 {
    8 + (quality * quality) / 128
}

/// GetWindowSizeForHashChain, backward_references_enc.c:243.
fn get_window_size_for_hash_chain(quality: i32, xsize: usize) -> usize {
    let m = if quality > 75 {
        WINDOW_SIZE
    } else if quality > 50 {
        xsize << 8
    } else if quality > 25 {
        xsize << 6
    } else {
        xsize << 4
    };
    m.min(WINDOW_SIZE)
}

/// MaxFindCopyLength, backward_references_enc.c:252.
fn max_find_copy_length(n: usize) -> usize {
    n.min(MAX_LENGTH)
}

/// VP8LHashChainFill, backward_references_enc.c:256, low_effort 0. `hc` must be init'ed to
/// xsize * ysize.
pub(super) fn hash_chain_fill(
    cx: &Ledger,
    hash_first: &mut Vec<i32>,
    hc: &mut HashChain,
    quality: i32,
    argb: &[u32],
    xsize: usize,
    ysize: usize,
) -> Result<(), Error> {
    let size = xsize * ysize;
    let iter_max = get_max_iters_for_quality(quality);
    let window_size = get_window_size_for_hash_chain(quality, xsize);
    let ol = &mut hc.off_len;
    if size <= 2 {
        ol[size - 1] = 0;
        ol[0] = 0;
        return Ok(());
    }
    // first holds pos + 1; 0 is empty.
    cx.resize(hash_first, HASH_SIZE, 0)?;
    let first = hash_first;
    // ol doubles as an int32 chain until the match pass.
    let mut pos = 0usize;
    let mut next_check = 0usize;
    let mut argb_comp = argb[0] == argb[1];
    while pos < size - 2 {
        if pos >= next_check {
            cx.check()?;
            next_check = pos + CHECK_EVERY;
        }
        let argb_comp_next = argb[pos + 1] == argb[pos + 2];
        if argb_comp && argb_comp_next {
            // Runs hash on (color, run length).
            let mut ln = 1usize;
            let c = argb[pos];
            while pos + ln + 2 < size && argb[pos + ln + 2] == argb[pos] {
                ln += 1;
            }
            if ln > MAX_LENGTH {
                for k in pos..pos + ln - MAX_LENGTH {
                    ol[k] = 0xffff_ffff;
                }
                pos += ln - MAX_LENGTH;
                ln = MAX_LENGTH;
            }
            while ln != 0 {
                let h = get_pix_pair_hash64(c, ln as u32);
                ln -= 1;
                ol[pos] = (first[h] as u32).wrapping_sub(1);
                first[h] = pos as i32 + 1;
                pos += 1;
            }
            argb_comp = false;
        } else {
            let h = get_pix_pair_hash64(argb[pos], argb[pos + 1]);
            ol[pos] = (first[h] as u32).wrapping_sub(1);
            first[h] = pos as i32 + 1;
            pos += 1;
            argb_comp = argb_comp_next;
        }
    }
    // Penultimate pixel.
    ol[pos] = (first[get_pix_pair_hash64(argb[pos], argb[pos + 1])] as u32).wrapping_sub(1);

    ol[size - 1] = 0;
    ol[0] = 0;
    let mut base = size - 2;
    let mut next_check = base;
    while base > 0 {
        if base <= next_check {
            cx.check()?;
            next_check = base.saturating_sub(CHECK_EVERY);
        }
        let max_len = max_find_copy_length(size - 1 - base);
        let start = &argb[base..];
        let mut iter = iter_max;
        let mut best_length = 0usize;
        let mut best_distance = 0usize;
        let min_pos = base.saturating_sub(window_size);
        let length_max = max_len.min(256);

        let mut pos = i64::from(ol[base] as i32);
        // Seed with the pixel above, then the one to the left.
        if base >= xsize {
            let curr = find_match_length(&argb[base - xsize..], start, best_length, max_len);
            if curr > best_length {
                best_length = curr;
                best_distance = xsize;
            }
            iter -= 1;
        }
        let curr = find_match_length(&argb[base - 1..], start, best_length, max_len);
        if curr > best_length {
            best_length = curr;
            best_distance = 1;
        }
        iter -= 1;
        if best_length == MAX_LENGTH {
            pos = min_pos as i64 - 1;
        }
        let mut best_argb = start[best_length];

        while pos >= min_pos as i64 {
            iter -= 1;
            if iter == 0 {
                break;
            }
            let p = pos as usize;
            if argb[p + best_length] == best_argb {
                let curr = vector_mismatch(&argb[p..], start, max_len);
                if best_length < curr {
                    best_length = curr;
                    best_distance = base - p;
                    best_argb = start[best_length];
                    if best_length >= length_max {
                        break;
                    }
                }
            }
            pos = i64::from(ol[p] as i32);
        }
        // Extend the match to the left while it holds.
        let mut max_base = base;
        loop {
            ol[base] = ((best_distance as u32) << MAX_LENGTH_BITS) | best_length as u32;
            base -= 1;
            if best_distance == 0 || base == 0 {
                break;
            }
            if base < best_distance || argb[base - best_distance] != argb[base] {
                break;
            }
            if best_length == MAX_LENGTH && best_distance != 1 && base + MAX_LENGTH < max_base {
                break;
            }
            if best_length < MAX_LENGTH {
                best_length += 1;
                max_base = base;
            }
        }
    }
    Ok(())
}

/// AddSingleLiteral, backward_references_enc.c:424, no cache.
fn add_single_literal(cx: &Ledger, pixel: u32, refs: &mut Refs) -> Result<(), Error> {
    cx.push(
        refs,
        PixOrCopy {
            mode: LITERAL,
            len: 1,
            arg: pixel,
        },
    )?;
    Ok(())
}

/// BackwardReferencesRle, backward_references_enc.c:442.
pub(super) fn backward_references_rle(
    cx: &Ledger,
    xsize: usize,
    ysize: usize,
    argb: &[u32],
    refs: &mut Refs,
) -> Result<(), Error> {
    let pix_count = xsize * ysize;
    refs.clear();
    add_single_literal(cx, argb[0], refs)?;
    let mut i = 1;
    let mut next_check = 0usize;
    while i < pix_count {
        if i >= next_check {
            cx.check()?;
            next_check = i + CHECK_EVERY;
        }
        let max_len = max_find_copy_length(pix_count - i);
        let rle_len = find_match_length(&argb[i..], &argb[i - 1..], 0, max_len);
        let mut prev_row_len = 0;
        if i >= xsize {
            prev_row_len = find_match_length(&argb[i..], &argb[i - xsize..], 0, max_len);
        }
        if rle_len >= prev_row_len && rle_len >= MIN_LENGTH {
            cx.push(
                refs,
                PixOrCopy {
                    mode: COPY,
                    len: rle_len as u16,
                    arg: 1,
                },
            )?;
            i += rle_len;
        } else if prev_row_len >= MIN_LENGTH {
            cx.push(
                refs,
                PixOrCopy {
                    mode: COPY,
                    len: prev_row_len as u16,
                    arg: xsize as u32,
                },
            )?;
            i += prev_row_len;
        } else {
            add_single_literal(cx, argb[i], refs)?;
            i += 1;
        }
    }
    Ok(())
}

/// BackwardReferencesLz77, backward_references_enc.c:485.
pub(super) fn backward_references_lz77(
    cx: &Ledger,
    xsize: usize,
    ysize: usize,
    argb: &[u32],
    hc: &HashChain,
    refs: &mut Refs,
) -> Result<(), Error> {
    let mut i_last_check: i64 = -1;
    let pix_count = xsize * ysize;
    refs.clear();
    let mut i = 0;
    let mut next_check = 0usize;
    while i < pix_count {
        if i >= next_check {
            cx.check()?;
            next_check = i + CHECK_EVERY;
        }
        let offset = hc.find_offset(i);
        let mut ln = hc.find_length(i);
        if ln >= MIN_LENGTH {
            let len_ini = ln;
            let mut max_reach = 0;
            let mut j_max = i + len_ini;
            if i + len_ini >= pix_count {
                j_max = pix_count - 1;
            }
            if i as i64 > i_last_check {
                i_last_check = i as i64;
            }
            // Best split of [i, j) plus match at j.
            let mut j = (i_last_check + 1) as usize;
            while j <= j_max {
                let len_j = hc.find_length(j);
                let mut reach = j + 1;
                if len_j >= MIN_LENGTH {
                    reach = j + len_j;
                }
                if reach > max_reach {
                    ln = j - i;
                    max_reach = reach;
                    if max_reach >= pix_count {
                        break;
                    }
                }
                j += 1;
            }
        } else {
            ln = 1;
        }
        if ln == 1 {
            add_single_literal(cx, argb[i], refs)?;
        } else {
            cx.push(
                refs,
                PixOrCopy {
                    mode: COPY,
                    len: ln as u16,
                    arg: offset as u32,
                },
            )?;
        }
        i += ln;
    }
    Ok(())
}

/// BackwardReferencesLz77Box, backward_references_enc.c:557: fills `box_hc` from `best`.
pub(super) fn backward_references_lz77_box(
    cx: &Ledger,
    box_counts: &mut Vec<u16>,
    xsize: usize,
    ysize: usize,
    argb: &[u32],
    best: &HashChain,
    box_hc: &mut HashChain,
    refs: &mut Refs,
) -> Result<(), Error> {
    let pix_count = xsize * ysize;
    let mut window_offsets = [0usize; WINDOW_OFFSETS_MAX];
    let mut window_offsets_new = [0usize; WINDOW_OFFSETS_MAX];
    let mut off_ind = [0usize; WINDOW_OFFSETS_MAX]; // into window_offsets
    let mut off_ind_new = [0usize; WINDOW_OFFSETS_MAX];
    let mut match_end = [0usize; WINDOW_OFFSETS_MAX]; // exact match end, or 0
    let mut max_wo = 0usize;
    let mut window_offsets_size = 0usize;
    let mut window_offsets_new_size = 0usize;
    let (mut best_offset_prev, mut best_length_prev) = (-1i64, -1i64);
    cx.resize(box_counts, pix_count, 0)?;
    let counts = box_counts;

    // Run length at i, capped at MAX_LENGTH.
    counts[pix_count - 1] = 1;
    for i in (0..pix_count - 1).rev() {
        if argb[i] == argb[i + 1] {
            let mut c = counts[i + 1];
            if usize::from(c) != MAX_LENGTH {
                c += 1;
            }
            counts[i] = c;
        } else {
            counts[i] = 1;
        }
    }

    // Window offsets in plane code order.
    for y in 0..=6i64 {
        for x in -6..=6i64 {
            let offset = y * xsize as i64 + x;
            if offset <= 0 {
                continue;
            }
            let plane_code = distance_to_plane_code(xsize, offset as usize) - 1;
            if plane_code >= WINDOW_OFFSETS_MAX {
                continue;
            }
            window_offsets[plane_code] = offset as usize;
        }
    }
    // Narrow images do not reach every plane code.
    for i in 0..WINDOW_OFFSETS_MAX {
        if window_offsets[i] == 0 {
            continue;
        }
        window_offsets[window_offsets_size] = window_offsets[i];
        off_ind[window_offsets_size] = window_offsets_size;
        max_wo = max_wo.max(window_offsets[i]);
        window_offsets_size += 1;
    }
    // Offsets reaching pixels that P-1 cannot reach.
    for i in 0..window_offsets_size {
        let mut is_reachable = false;
        let mut j = 0;
        while j < window_offsets_size && !is_reachable {
            is_reachable = is_reachable || window_offsets[i] == window_offsets[j] + 1;
            j += 1;
        }
        if !is_reachable {
            window_offsets_new[window_offsets_new_size] = window_offsets[i];
            off_ind_new[window_offsets_new_size] = i;
            window_offsets_new_size += 1;
        }
    }

    let ol = &mut box_hc.off_len;
    ol[0] = 0;
    for i in 1..pix_count {
        if i % CHECK_EVERY == 0 {
            cx.check()?;
        }
        let mut best_length = best.find_length(i);
        let mut best_offset = 0usize;
        let mut do_compute = true;
        if best_length >= MAX_LENGTH {
            // Keep a maximal match already inside the window.
            best_offset = best.find_offset(i);
            for ind in 0..window_offsets_size {
                if best_offset == window_offsets[ind] {
                    do_compute = false;
                    break;
                }
            }
        }
        if do_compute {
            let use_prev = best_length_prev > 1 && best_length_prev < MAX_LENGTH as i64;
            let (wos, inds) = if use_prev {
                (
                    &window_offsets_new[..window_offsets_new_size],
                    &off_ind_new[..window_offsets_new_size],
                )
            } else {
                (
                    &window_offsets[..window_offsets_size],
                    &off_ind[..window_offsets_size],
                )
            };
            best_length = 0;
            best_offset = 0;
            if use_prev {
                best_length = (best_length_prev - 1) as usize;
                best_offset = best_offset_prev as usize;
            }
            // Bit ind set when argb[i - wos[ind]] == argb[i].
            let p = argb[i];
            let mut mask = 0u32;
            for (ind, &wo) in wos.iter().enumerate() {
                if wo <= i && argb[i - wo] == p {
                    mask |= 1 << ind;
                }
            }
            while mask != 0 {
                let ind = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let wo = wos[ind];
                let k = inds[ind];
                // Inside a known match, no walk.
                let mut curr_length = match_end[k] as i64 - i as i64;
                if curr_length <= 0 {
                    curr_length = 0;
                    let mut j = i;
                    let mut j_offset = i - wo;
                    // Sum the matching run lengths.
                    loop {
                        let counts_j_offset = i64::from(counts[j_offset]);
                        let counts_j = i64::from(counts[j]);
                        if counts_j_offset != counts_j {
                            if counts_j_offset < counts_j {
                                curr_length += counts_j_offset;
                            } else {
                                curr_length += counts_j;
                            }
                            break;
                        }
                        curr_length += counts_j_offset;
                        j_offset += counts_j_offset as usize;
                        j += counts_j_offset as usize;
                        if !(curr_length <= MAX_LENGTH as i64
                            && j < pix_count
                            && argb[j_offset] == argb[j])
                        {
                            break;
                        }
                    }
                    // Only an uncapped walk is exact.
                    if curr_length <= MAX_LENGTH as i64 {
                        match_end[k] = i + curr_length as usize;
                    }
                }
                if (best_length as i64) < curr_length {
                    best_offset = wo;
                    if curr_length >= MAX_LENGTH as i64 {
                        best_length = MAX_LENGTH;
                        break;
                    }
                    best_length = curr_length as usize;
                }
            }
        }

        if best_length <= MIN_LENGTH {
            ol[i] = 0;
            best_offset_prev = 0;
            best_length_prev = 0;
        } else {
            ol[i] = ((best_offset as u32) << MAX_LENGTH_BITS) | best_length as u32;
            best_offset_prev = best_offset as i64;
            best_length_prev = best_length as i64;
        }
    }
    ol[0] = 0;
    backward_references_lz77(cx, xsize, ysize, argb, box_hc, refs)?;
    Ok(())
}

/// BackwardReferences2DLocality, backward_references_enc.c:708; in place.
pub(super) fn backward_references_2d_locality(xsize: usize, refs: &mut Refs) {
    for r in refs.iter_mut() {
        if r.mode == COPY {
            r.arg = distance_to_plane_code(xsize, r.arg as usize) as u32;
        }
    }
}

/// Scratch of the backward references and their costs.
#[derive(Default)]
pub(super) struct RefsScratch {
    pub hash_first: Vec<i32>,
    pub hc_box: HashChain,
    pub box_counts: Vec<u16>,
    pub est_histo: Option<Box<Histo>>,
    pub cost: CostScratch,
}

impl RefsScratch {
    /// The bytes held, at capacity, without the estimate histogram the ledger never charged.
    pub(super) fn bytes(&self) -> u64 {
        cap_bytes(&self.hash_first)
            + self.hc_box.bytes()
            + cap_bytes(&self.box_counts)
            + self.cost.bytes()
    }
}

/// VP8LGetBackwardReferences (backward_references_enc.c:1009, GetBackwardReferences :882), low
/// effort 0, no colour cache. Leaves the best refs, as plane codes, in refs[0]; refs[1] is
/// refs_tmp.
pub(super) fn get_backward_references(
    cx: &Ledger,
    sc: &mut RefsScratch,
    w: usize,
    h: usize,
    argb: &[u32],
    quality: i32,
    lz77_types: u32,
    hc: &HashChain,
    refs: &mut [Refs],
) -> Result<(), Error> {
    let mut lz77_type_best = 0;
    let mut bit_cost_best = f64::MAX;
    let mut histo = sc
        .est_histo
        .take()
        .unwrap_or_else(|| Box::new(Histo::new()));
    let mut types = lz77_types;
    let mut lz77_type = 1u32;
    while types != 0 {
        if types & lz77_type != 0 {
            let tmp = &mut refs[1];
            match lz77_type {
                LZ77_RLE => backward_references_rle(cx, w, h, argb, tmp)?,
                LZ77_STANDARD => backward_references_lz77(cx, w, h, argb, hc, tmp)?,
                LZ77_BOX => {
                    sc.hc_box.init(cx, w * h)?;
                    backward_references_lz77_box(
                        cx,
                        &mut sc.box_counts,
                        w,
                        h,
                        argb,
                        hc,
                        &mut sc.hc_box,
                        tmp,
                    )?;
                }
                _ => {}
            }
            // Cache bits 0, no do_no_cache pass.
            histo.create(&refs[1]);
            let bit_cost = histo.estimate_bits();
            if bit_cost < bit_cost_best {
                refs.swap(0, 1);
                bit_cost_best = bit_cost;
                lz77_type_best = lz77_type;
            }
        }
        types &= !lz77_type;
        lz77_type <<= 1;
    }

    // TraceBackwards only for quality >= 25.
    if (lz77_type_best == LZ77_STANDARD || lz77_type_best == LZ77_BOX) && quality >= 25 {
        let (r0, r1) = refs.split_at_mut(1);
        let chain = if lz77_type_best == LZ77_STANDARD {
            hc
        } else {
            &sc.hc_box
        };
        trace_backwards_refs(cx, &mut sc.cost, w, h, argb, chain, &r0[0], &mut r1[0])?;
        histo.create(&refs[1]);
        if histo.estimate_bits() < bit_cost_best {
            refs.swap(0, 1);
        }
    }
    backward_references_2d_locality(w, &mut refs[0]);
    sc.est_histo = Some(histo);
    Ok(())
}
