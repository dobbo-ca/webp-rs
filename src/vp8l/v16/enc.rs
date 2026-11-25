//! Port of libwebp 1.6.0 enc/vp8l_enc.c (analysis, palette, stream and Huffman storage) and
//! enc/predictor_enc.c (exact = 1, no near-lossless or cross colour). The Huffman tree code, bit
//! writer and Huffman storage helpers are the 1.2.2 port. Copyright 2012 Google Inc. BSD-3, see
//! LICENSE.libwebp.

use super::cost::{MAX_CACHE_BITS, RefsScratch};
use super::hist::{
    CACHE_IDX, Histo, HistoSet, NUM_CODES, Scratch as HistoScratch, bits_entropy,
    get_histo_image_symbols,
};
use super::log::{LOG_BITS, combined_shannon_entropy, div_round, fast_log2, shannon_entropy};
use crate::Error;
use crate::limits::Ledger;
use crate::vp8l::analyze::{
    HISTO_ALPHA, HISTO_ALPHA_PRED, HISTO_BLUE, HISTO_BLUE_PRED, HISTO_BLUE_PRED_SUB_GREEN,
    HISTO_BLUE_SUB_GREEN, HISTO_GREEN, HISTO_GREEN_PRED, HISTO_PALETTE, HISTO_RED, HISTO_RED_PRED,
    HISTO_RED_PRED_SUB_GREEN, HISTO_RED_SUB_GREEN, HISTO_TOTAL, MAX_HISTO_BITS, MAX_HUFF_IMAGE,
    MIN_HISTO_BITS, add_single, add_single_sub_green, copy_image_with_prediction,
    get_transform_bits, hash_pix, palette_color_distance, palette_has_non_monotonous_deltas,
    predict_batch,
};
use crate::vp8l::encode::{HuffCodes, clear_huffman_tree_if_only_one_symbol, store_huffman_code};
use crate::vp8l::huff::{Code, create_huffman_tree};
use crate::vp8l::lossless::{ARGB_BLACK, sub_pixels, sub_sample_size};
use crate::vp8l::refs::{HashChain, Refs, hash_chain_fill};
use crate::vp8l::{
    CrunchConfig, K_DIRECT, K_PALETTE, K_SPATIAL, K_SPATIAL_SUB_GREEN, K_SUB_GREEN, LITERAL,
    LZ77_BOX, LZ77_RLE, LZ77_STANDARD, Lw, NUM_LITERAL, NUM_PRED_MODES, prefix_bits,
};

const MAX_TRANS_BITS: usize = 9; // MAX_TRANSFORM_BITS, vp8li_enc.h:41
const MAX_PRED_IMAGE: usize = 1 << 14; // MAX_PREDICTOR_IMAGE_SIZE, vp8l_enc.c:39
const HISTO_SIZE: usize = 4 * 256; // HISTO_SIZE
const SPATIAL_BIAS: i64 = 15 << 23; // kSpatialPredictorBias
const MODE_HISTO_SIZE: usize = NUM_PRED_MODES * HISTO_SIZE;

/// The scratch of the 1.6.0 path, made on first use.
pub(in crate::vp8l) struct V16 {
    hs: HistoScratch,
    image: HistoSet,
    tmp: Histo,
    histo_symbol: Vec<u32>,
    rs: RefsScratch,
    huff: HuffCodes,
}

impl V16 {
    pub(in crate::vp8l) fn new() -> V16 {
        V16 {
            hs: HistoScratch::default(),
            image: HistoSet::new(),
            tmp: Histo::new(0),
            histo_symbol: Vec::new(),
            rs: RefsScratch::new(),
            huff: HuffCodes::default(),
        }
    }
}

// Analysis ---------------------------------------------------------------------

/// ClampBits, vp8l_enc.c:254.
fn clamp_bits(
    w: usize,
    h: usize,
    bits: i64,
    min_bits: i64,
    max_bits: i64,
    image_size_max: usize,
) -> usize {
    let mut bits = min_bits.max(bits.min(max_bits));
    let size = |b: i64| sub_sample_size(w, b as usize) * sub_sample_size(h, b as usize);
    let mut image_size = size(bits);
    while bits < max_bits && image_size > image_size_max {
        bits += 1;
        image_size = size(bits);
    }
    // Keep the smallest bits that still give one tile.
    while bits > min_bits && image_size == 1 {
        image_size = size(bits - 1);
        if image_size != 1 {
            break;
        }
        bits -= 1;
    }
    bits as usize
}

/// GetHistoBits, vp8l_enc.c:275.
fn get_histo_bits(method: usize, use_palette: bool, w: usize, h: usize) -> usize {
    let histo_bits = if use_palette { 9 - method } else { 7 - method };
    clamp_bits(
        w,
        h,
        histo_bits as i64,
        MIN_HISTO_BITS as i64,
        MAX_HISTO_BITS as i64,
        MAX_HUFF_IMAGE,
    )
}

/// AnalyzeEntropy, vp8l_enc.c:106.
#[allow(clippy::too_many_arguments)]
fn analyze_entropy(
    cx: &Ledger,
    ae_histo: &mut Vec<u32>,
    argb: &[u32],
    w: usize,
    h: usize,
    stride: usize,
    use_palette: bool,
    palette_size: usize,
    transform_bits: usize,
) -> Result<usize, Error> {
    if use_palette && palette_size <= 16 {
        return Ok(K_PALETTE);
    }
    cx.resize(ae_histo, HISTO_TOTAL * 256, 0)?;
    let mut prev_row: Option<usize> = None;
    let mut off = 0usize;
    let mut pix_prev = argb[0]; // skip the first pixel
    for _ in 0..h {
        cx.check()?;
        for x in 0..w {
            let pix = argb[off + x];
            let pix_diff = sub_pixels(pix, pix_prev);
            pix_prev = pix;
            if pix_diff == 0 || prev_row.is_some_and(|p| pix == argb[p + x]) {
                continue;
            }
            add_single(
                ae_histo,
                pix,
                HISTO_ALPHA,
                HISTO_RED,
                HISTO_GREEN,
                HISTO_BLUE,
            );
            add_single(
                ae_histo,
                pix_diff,
                HISTO_ALPHA_PRED,
                HISTO_RED_PRED,
                HISTO_GREEN_PRED,
                HISTO_BLUE_PRED,
            );
            add_single_sub_green(ae_histo, pix, HISTO_RED_SUB_GREEN, HISTO_BLUE_SUB_GREEN);
            add_single_sub_green(
                ae_histo,
                pix_diff,
                HISTO_RED_PRED_SUB_GREEN,
                HISTO_BLUE_PRED_SUB_GREEN,
            );
            ae_histo[HISTO_PALETTE * 256 + hash_pix(pix) as usize] += 1;
        }
        prev_row = Some(off);
        off += stride;
    }

    let mut entropy_comp = [0u64; HISTO_TOTAL];
    let mut entropy = [0u64; 5];
    let last_mode = if use_palette {
        K_PALETTE
    } else {
        K_SPATIAL_SUB_GREEN
    };
    // One zero for each predicted histogram.
    ae_histo[HISTO_RED_PRED_SUB_GREEN * 256] += 1;
    ae_histo[HISTO_BLUE_PRED_SUB_GREEN * 256] += 1;
    ae_histo[HISTO_RED_PRED * 256] += 1;
    ae_histo[HISTO_GREEN_PRED * 256] += 1;
    ae_histo[HISTO_BLUE_PRED * 256] += 1;
    ae_histo[HISTO_ALPHA_PRED * 256] += 1;

    for j in 0..HISTO_TOTAL {
        entropy_comp[j] = bits_entropy(&ae_histo[j * 256..(j + 1) * 256]);
    }
    let sum4 = |a: usize, b: usize, c: usize, d: usize| {
        entropy_comp[a]
            .wrapping_add(entropy_comp[b])
            .wrapping_add(entropy_comp[c])
            .wrapping_add(entropy_comp[d])
    };
    entropy[K_DIRECT] = sum4(HISTO_ALPHA, HISTO_RED, HISTO_GREEN, HISTO_BLUE);
    entropy[K_SPATIAL] = sum4(
        HISTO_ALPHA_PRED,
        HISTO_RED_PRED,
        HISTO_GREEN_PRED,
        HISTO_BLUE_PRED,
    );
    entropy[K_SUB_GREEN] = sum4(
        HISTO_ALPHA,
        HISTO_RED_SUB_GREEN,
        HISTO_GREEN,
        HISTO_BLUE_SUB_GREEN,
    );
    entropy[K_SPATIAL_SUB_GREEN] = sum4(
        HISTO_ALPHA_PRED,
        HISTO_RED_PRED_SUB_GREEN,
        HISTO_GREEN_PRED,
        HISTO_BLUE_PRED_SUB_GREEN,
    );
    entropy[K_PALETTE] = entropy_comp[HISTO_PALETTE];

    let tiles =
        (sub_sample_size(w, transform_bits) as u64) * (sub_sample_size(h, transform_bits) as u64);
    entropy[K_SPATIAL] =
        entropy[K_SPATIAL].wrapping_add(tiles.wrapping_mul(u64::from(fast_log2(14))));
    entropy[K_SPATIAL_SUB_GREEN] =
        entropy[K_SPATIAL_SUB_GREEN].wrapping_add(tiles.wrapping_mul(u64::from(fast_log2(24))));
    entropy[K_PALETTE] = entropy[K_PALETTE].wrapping_add(((palette_size * 8) as u64) << LOG_BITS);

    let mut min_ix = K_DIRECT;
    for k in K_DIRECT + 1..=last_mode {
        if entropy[min_ix] > entropy[k] {
            min_ix = k;
        }
    }
    Ok(min_ix)
}

/// PaletteSortMinimizeDeltas, palette.c:187; palette.len() >= sorted.len().
fn palette_sort_minimize_deltas(sorted: &[u32], palette: &mut [u32]) {
    let mut n = sorted.len();
    palette[..n].copy_from_slice(sorted);
    if !palette_has_non_monotonous_deltas(sorted) {
        return;
    }
    // A leading 0 moves last, so EncodePalette can drop it.
    if n > 17 && palette[0] == 0 {
        n -= 1;
        palette.swap(n, 0);
    }
    let mut predict = 0u32;
    for i in 0..n {
        let mut best_ix = i;
        let mut best_score = u32::MAX;
        for k in i..n {
            let cur = palette_color_distance(palette[k], predict);
            if best_score > cur {
                best_score = cur;
                best_ix = k;
            }
        }
        palette.swap(best_ix, i);
        predict = palette[i];
    }
}

// Predictors -------------------------------------------------------------------

/// PredictionCostBias, predictor_enc.c:46; `exp_val` is in 1/100.
fn prediction_cost_bias(counts: &[u32], weight0: u64, mut exp_val: u64) -> i64 {
    const SIGNIFICANT_SYMBOLS: usize = 256 >> 4;
    const EXP_DECAY_FACTOR: u64 = 6; // in 1/10
    let mut bits = (weight0 * u64::from(counts[0])) << LOG_BITS;
    exp_val <<= LOG_BITS;
    for i in 1..SIGNIFICANT_SYMBOLS {
        bits = bits.wrapping_add(div_round(
            exp_val.wrapping_mul(u64::from(counts[i].wrapping_add(counts[256 - i]))) as i64,
            100,
        ) as u64);
        exp_val = div_round(EXP_DECAY_FACTOR.wrapping_mul(exp_val) as i64, 10) as u64;
    }
    -div_round(bits as i64, 10)
}

/// PredictionCostSpatialHistogram, predictor_enc.c:60.
fn prediction_cost_spatial_histogram(
    acc: &[u32],
    tile: &[u32],
    mode: usize,
    left_mode: usize,
    above_mode: usize,
) -> i64 {
    let mut retval = 0i64;
    for i in 0..4 {
        const EXP_VALUE: u64 = 94;
        retval = retval.wrapping_add(prediction_cost_bias(&tile[i * 256..], 1, EXP_VALUE));
        retval =
            retval.wrapping_add(combined_shannon_entropy(&tile[i * 256..], &acc[i * 256..]) as i64);
    }
    // Favor keeping the areas locally similar.
    if mode == left_mode {
        retval -= SPATIAL_BIAS;
    }
    if mode == above_mode {
        retval -= SPATIAL_BIAS;
    }
    retval
}

/// UpdateHisto, predictor_enc.c:80.
fn update_histo(h: &mut [u32], argb: u32) {
    h[(argb >> 24) as usize] += 1;
    h[256 + ((argb >> 16) & 0xff) as usize] += 1;
    h[2 * 256 + ((argb >> 8) & 0xff) as usize] += 1;
    h[3 * 256 + (argb & 0xff) as usize] += 1;
}

/// GetBestPredictorForTile, predictor_enc.c:334: `all_argb` holds the mode histograms of each
/// level, `all_acc` one histogram per level.
#[allow(clippy::too_many_arguments)]
fn get_best_predictor_for_tile(
    all_argb: &[u32],
    sub: usize,
    tile_x: usize,
    tile_y: usize,
    tiles_per_row: usize,
    all_acc: &mut [u32],
    all_modes: &mut [Vec<u32>],
    all_pred: &mut [u32],
) {
    let acc = &mut all_acc[sub * HISTO_SIZE..(sub + 1) * HISTO_SIZE];
    let modes = &mut all_modes[sub];
    let pred = &mut all_pred[sub * NUM_PRED_MODES..];
    let (mut left_mode, mut above_mode) = (0xffusize, 0xffusize);
    if tile_x > 0 {
        left_mode = ((modes[tile_y * tiles_per_row + tile_x - 1] >> 8) & 0xff) as usize;
    }
    if tile_y > 0 {
        above_mode = ((modes[(tile_y - 1) * tiles_per_row + tile_x] >> 8) & 0xff) as usize;
    }
    let mut best_diff = i64::MAX;
    let mut best_mode = 0usize;
    let mut best_off = 0usize;
    for mode in 0..NUM_PRED_MODES {
        let o = sub * MODE_HISTO_SIZE + mode * HISTO_SIZE;
        let h = &all_argb[o..o + HISTO_SIZE];
        let cur_diff = prediction_cost_spatial_histogram(acc, h, mode, left_mode, above_mode);
        if cur_diff < best_diff {
            best_off = o;
            best_diff = cur_diff;
            best_mode = mode;
        }
    }
    let best = &all_argb[best_off..best_off + HISTO_SIZE];
    for (a, &b) in acc.iter_mut().zip(best) {
        *a = a.wrapping_add(b);
    }
    modes[tile_y * tiles_per_row + tile_x] = ARGB_BLACK | ((best_mode as u32) << 8);
    pred[best_mode] += 1;
}

/// ComputeResidualsForTile, predictor_enc.c:380: `scratch` holds two rows of w + 1.
#[allow(clippy::too_many_arguments)]
fn compute_residuals_for_tile(
    w: usize,
    h: usize,
    tile_x: usize,
    tile_y: usize,
    min_bits: usize,
    update_up_to: usize,
    all_argb: &mut [u32],
    scratch: &mut [u32],
    argb: &[u32],
) {
    let start_x = tile_x << min_bits;
    let start_y = tile_y << min_bits;
    let tile_size = 1usize << min_bits;
    let max_y = tile_size.min(h - start_y);
    let max_x = tile_size.min(w - start_x);
    let have_left = usize::from(start_x > 0);
    let ctx_x = start_x - have_left;
    // Rows are w + 1 wide so the top right wraps to the next row.
    let (mut upper_off, mut cur_off) = (0usize, w + 1);
    let mut residuals = [0u32; 1 << MAX_TRANS_BITS];
    for mode in 0..NUM_PRED_MODES {
        if start_y > 0 {
            let o = (start_y - 1) * w + ctx_x;
            let n = max_x + have_left + 1;
            scratch[cur_off + ctx_x..cur_off + ctx_x + n].copy_from_slice(&argb[o..o + n]);
        }
        for ry in 0..max_y {
            let y = start_y + ry;
            std::mem::swap(&mut upper_off, &mut cur_off);
            let mut n = max_x + have_left;
            if y + 1 < h {
                n += 1;
            }
            let o = y * w + ctx_x;
            scratch[cur_off + ctx_x..cur_off + ctx_x + n].copy_from_slice(&argb[o..o + n]);
            {
                let (upper, cur) = if upper_off < cur_off {
                    let (a, b) = scratch.split_at(cur_off);
                    (&a[upper_off..upper_off + w + 1], &b[..w + 1])
                } else {
                    let (a, b) = scratch.split_at(upper_off);
                    (&b[..w + 1], &a[cur_off..cur_off + w + 1])
                };
                predict_batch(mode, start_x, y, max_x, cur, upper, &mut residuals);
            }
            let histo = &mut all_argb[mode * HISTO_SIZE..(mode + 1) * HISTO_SIZE];
            for rx in 0..max_x {
                update_histo(histo, residuals[rx]);
            }
            for s in 1..=update_up_to {
                let o = s * MODE_HISTO_SIZE + mode * HISTO_SIZE;
                let sup = &mut all_argb[o..o + HISTO_SIZE];
                for rx in 0..max_x {
                    update_histo(sup, residuals[rx]);
                }
            }
        }
    }
}

/// VP8LOptimizeSampling, predictor_enc.c:526: the bits of the largest uniform squares of `image`,
/// which is subsampled in place to them.
fn optimize_sampling(
    image: &mut [u32],
    full_w: usize,
    full_h: usize,
    bits: usize,
    max_bits: usize,
) -> usize {
    let mut width = sub_sample_size(full_w, bits);
    let mut height = sub_sample_size(full_h, bits);
    let mut best_bits = bits;
    // Rows first.
    while best_bits < max_bits {
        let new_square = 1usize << (best_bits + 1 - bits);
        let square = 1usize << (best_bits - bits);
        let mut is_good = true;
        let mut y = 0;
        while y + square < height {
            for i in 0..width {
                if image[y * width + i] != image[(y + square) * width + i] {
                    is_good = false;
                    break;
                }
            }
            if !is_good {
                break;
            }
            y += new_square;
        }
        if !is_good {
            break;
        }
        best_bits += 1;
    }
    if best_bits == bits {
        return bits;
    }
    // Then columns.
    while best_bits > bits {
        let mut is_good = true;
        let square = 1usize << (best_bits - bits);
        let mut y = 0;
        while is_good && y < height {
            let mut x = 0;
            while is_good && x < width {
                for i in x + 1..(x + square).min(width) {
                    if image[y * width + i] != image[y * width + x] {
                        is_good = false;
                        break;
                    }
                }
                x += square;
            }
            y += 1;
        }
        if is_good {
            break;
        }
        best_bits -= 1;
    }
    if best_bits == bits {
        return bits;
    }
    let old_width = width;
    let square = 1usize << (best_bits - bits);
    width = sub_sample_size(full_w, best_bits);
    height = sub_sample_size(full_h, best_bits);
    for y in 0..height {
        for x in 0..width {
            image[y * width + x] = image[square * (y * old_width + x)];
        }
    }
    best_bits
}

/// GetBestPredictorsAndSubSampling, predictor_enc.c:615: walks tiles in Z order over the levels
/// min_bits..max_bits; `all_modes[i]` is the mode image at min_bits + i. Returns the chosen bits and
/// the index of its mode image, resampled.
fn get_best_predictors_and_sub_sampling(
    cx: &Ledger,
    w: usize,
    h: usize,
    min_bits: usize,
    max_bits: usize,
    scratch: &mut [u32],
    argb: &[u32],
    all_modes: &mut [Vec<u32>],
) -> Result<(usize, usize), Error> {
    let tiles_per_row = sub_sample_size(w, min_bits) as u32;
    let tiles_per_col = sub_sample_size(h, min_bits) as u32;
    let max_sub = (max_bits - min_bits) as u32;
    let num_argb = (max_sub as usize + 1) * MODE_HISTO_SIZE;
    let num_acc = (max_sub as usize + 1) * HISTO_SIZE;
    let num_pred = (max_sub as usize + 1) * NUM_PRED_MODES;
    let mut all_argb = cx.alloc(num_argb, 0u32)?;
    let mut all_acc = cx.alloc(num_acc, 0u32)?;
    let mut all_pred = cx.alloc(num_pred, 0u32)?;
    let max_tile_size = 1u32 << max_sub;
    // Levels up to this one take residuals directly.
    let update_up_to = (4usize.min(max_bits).max(min_bits) - min_bits) as u32;
    let (mut local_x, mut local_y, mut max_tile_x, mut max_tile_y, mut tile_x, mut tile_y) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);

    while tile_y < tiles_per_col {
        cx.check()?;
        compute_residuals_for_tile(
            w,
            h,
            tile_x as usize,
            tile_y as usize,
            min_bits,
            update_up_to as usize,
            &mut all_argb,
            scratch,
            argb,
        );

        // Update every completed super-tile.
        let mut sub = 0u32;
        loop {
            let super_x = tile_x >> sub;
            let super_y = tile_y >> sub;
            let super_per_row = sub_sample_size(w, min_bits + sub as usize);
            get_best_predictor_for_tile(
                &all_argb,
                sub as usize,
                super_x as usize,
                super_y as usize,
                super_per_row,
                &mut all_acc,
                all_modes,
                &mut all_pred,
            );
            if sub == max_sub {
                break;
            }
            sub += 1;
            if sub > update_up_to && sub <= max_sub {
                let (lo, hi) = all_argb.split_at_mut(sub as usize * MODE_HISTO_SIZE);
                let src = &lo[(sub as usize - 1) * MODE_HISTO_SIZE..];
                let dst = &mut hi[..MODE_HISTO_SIZE];
                for i in 0..MODE_HISTO_SIZE {
                    dst[i] = dst[i].wrapping_add(src[i]);
                }
            }
            if !((tile_x == tiles_per_row - 1 || (local_x + 1).is_multiple_of(1 << sub))
                && (tile_y == tiles_per_col - 1 || (local_y + 1).is_multiple_of(1 << sub)))
            {
                sub -= 1; // last finished level
                break;
            }
        }
        all_argb[..(sub as usize + 1) * MODE_HISTO_SIZE].fill(0);

        if sub == max_sub {
            // Start a new max-tile.
            if tile_x == tiles_per_row - 1 {
                max_tile_x = 0;
                max_tile_y += 1;
            } else {
                max_tile_x += 1;
            }
            local_x = 0;
            local_y = 0;
        } else {
            // Z traversal.
            let mut cx = local_x >> sub;
            let mut cy = local_y >> sub;
            if tile_x == tiles_per_row - 1 && cx.is_multiple_of(2) {
                cy += 1;
            } else if cx.is_multiple_of(2) {
                cx += 1;
            } else {
                cy += 1;
                cx -= 1;
            }
            local_x = cx << sub;
            local_y = cy << sub;
        }
        tile_x = max_tile_x * max_tile_size + local_x;
        tile_y = max_tile_y * max_tile_size + local_y;
    }

    let mut best_cost = i64::MAX;
    let mut best_bits = 0;
    let mut best_sub = 0;
    for sub in 0..=max_sub as usize {
        let acc = &all_acc[sub * HISTO_SIZE..];
        let mut cost =
            shannon_entropy(&all_pred[sub * NUM_PRED_MODES..(sub + 1) * NUM_PRED_MODES]) as i64;
        for plane in 0..4 {
            cost = cost.wrapping_add(shannon_entropy(&acc[plane * 256..(plane + 1) * 256]) as i64);
        }
        if cost < best_cost {
            best_cost = cost;
            best_bits = min_bits + sub;
            best_sub = sub;
        }
    }
    best_bits = optimize_sampling(&mut all_modes[best_sub], w, h, best_bits, MAX_TRANS_BITS);
    cx.give(4 * (all_argb.len() + all_acc.len() + all_pred.len()) as u64);
    drop((all_argb, all_acc, all_pred));
    Ok((best_bits, best_sub))
}

/// VP8LResidualImage, predictor_enc.c:776, low_effort 0: writes the modes to `image` and the
/// residuals over `argb`, and returns the predictor bits.
fn residual_image(
    cx: &Ledger,
    w: usize,
    h: usize,
    min_bits: usize,
    max_bits: usize,
    argb: &mut [u32],
    scratch: &mut [u32],
    image: &mut [u32],
) -> Result<usize, Error> {
    let mut modes: Vec<Vec<u32>> = (min_bits..=max_bits)
        .map(|bits| cx.alloc(sub_sample_size(w, bits) * sub_sample_size(h, bits), 0u32))
        .collect::<Result<_, _>>()?;
    let (best_bits, best_sub) = get_best_predictors_and_sub_sampling(
        cx, w, h, min_bits, max_bits, scratch, argb, &mut modes,
    )?;
    let n = sub_sample_size(w, best_bits) * sub_sample_size(h, best_bits);
    image[..n].copy_from_slice(&modes[best_sub][..n]);
    cx.give(4 * modes.iter().map(Vec::len).sum::<usize>() as u64);
    drop(modes);
    copy_image_with_prediction(cx, w, h, best_bits, image, scratch, argb)?;
    Ok(best_bits)
}

// Stream -----------------------------------------------------------------------

/// GetHuffBitLengthsAndCodes, vp8l_enc.c:439: sizes `hc` to 5 codes per live histogram of `set` and
/// builds them. It mutates the histograms, as C does.
fn get_huff_bit_lengths_and_codes(
    cx: &Ledger,
    set: &mut HistoSet,
    hc: &mut HuffCodes,
    buf_rle: &mut Vec<u8>,
    huff_tree: &mut Vec<crate::vp8l::huff::Tree>,
) -> Result<(), Error> {
    let n = set.size;
    let mut total = 0;
    for i in 0..n {
        total += set.get(i).literal.len() + 3 * NUM_LITERAL + crate::vp8l::NUM_DISTANCE;
    }
    hc.offs.clear();
    hc.ns.clear();
    cx.resize(&mut hc.lens, total, 0)?; // calloc
    cx.resize(&mut hc.bits, total, 0)?;
    cx.reserve(&mut hc.offs, 5 * n)?;
    cx.reserve(&mut hc.ns, 5 * n)?;
    let mut off = 0;
    let mut max_symbols = 0;
    for i in 0..5 * n {
        let ns = match i % 5 {
            0 => set.get(i / 5).literal.len(),
            4 => crate::vp8l::NUM_DISTANCE,
            _ => NUM_LITERAL,
        };
        hc.offs.push(off);
        hc.ns.push(ns);
        off += ns;
        max_symbols = max_symbols.max(ns);
    }

    buf_rle.clear();
    buf_rle.resize(max_symbols, 0);
    huff_tree.clear();
    huff_tree.resize(3 * max_symbols, crate::vp8l::huff::Tree::default());

    for i in 0..n {
        let histo = set.get_mut(i);
        let parts: [&mut [u32]; 5] = [
            &mut histo.literal,
            &mut histo.red,
            &mut histo.blue,
            &mut histo.alpha,
            &mut histo.distance,
        ];
        for (k, hist) in parts.into_iter().enumerate() {
            let idx = 5 * i + k;
            let (o, ns) = (hc.offs[idx], hc.ns[idx]);
            let mut c = Code {
                num_symbols: ns,
                code_lengths: &mut hc.lens[o..o + ns],
                codes: &mut hc.bits[o..o + ns],
            };
            create_huffman_tree(hist, 15, buf_rle, huff_tree, &mut c);
        }
    }
    Ok(())
}

/// StoreImageToBitMask, vp8l_enc.c:701: `symbols` hold ids << 8; `codes` 5 per id.
fn store_image_to_bit_mask(
    cx: &Ledger,
    bw: &mut crate::vp8l::BitWriter,
    w: usize,
    histo_bits: usize,
    refs: &[crate::vp8l::refs::PixOrCopy],
    symbols: &[u32],
    codes: &HuffCodes,
    skip: u64,
) -> Result<(), Error> {
    let mut histo_xsize = 1;
    let mut tile_mask: i64 = 0;
    if histo_bits != 0 {
        histo_xsize = sub_sample_size(w, histo_bits);
        tile_mask = -(1i64 << histo_bits);
    }
    let (mut x, mut y) = (0usize, 0usize);
    let (mut tile_x, mut tile_y) = (x as i64 & tile_mask, y as i64 & tile_mask);
    let mut g = (symbols[0] >> 8) & 0xffff;
    let mut c = 5 * g as usize;
    let mut rest = skip.checked_shr(g).unwrap_or(0) & 1 == 0;
    for v in refs {
        if tile_x != x as i64 & tile_mask || tile_y != y as i64 & tile_mask {
            tile_x = x as i64 & tile_mask;
            tile_y = y as i64 & tile_mask;
            g = (symbols[(y >> histo_bits) * histo_xsize + (x >> histo_bits)] >> 8) & 0xffff;
            c = 5 * g as usize;
            rest = skip.checked_shr(g).unwrap_or(0) & 1 == 0;
        }
        match v.mode {
            LITERAL => {
                codes.write(bw, c, ((v.arg >> 8) & 0xff) as usize)?;
                if rest {
                    codes.write(bw, c + 1, ((v.arg >> 16) & 0xff) as usize)?;
                    codes.write(bw, c + 2, (v.arg & 0xff) as usize)?;
                    codes.write(bw, c + 3, (v.arg >> 24) as usize)?;
                }
            }
            CACHE_IDX => codes.write(bw, c, NUM_CODES + v.arg as usize)?,
            _ => {
                let (code, n, extra) = prefix_bits(usize::from(v.len));
                codes.write_with_extra_bits(bw, c, 256 + code, extra as u64, n as i32)?;
                // Distance extra bits go separately, up to 18 bits.
                let (code, n, extra) = prefix_bits(v.arg as usize);
                codes.write(bw, c + 4, code)?;
                bw.put(extra as u64, n as i32)?;
            }
        }
        x += usize::from(v.len);
        while x >= w {
            x -= w;
            y += 1;
            cx.check()?;
        }
    }
    Ok(())
}

impl Lw<'_> {
    /// VP8LEncodeStream, vp8l_enc.c:1690, and EncodeStreamHook :1512: writes the VP8L stream of `pic`
    /// (no header) to `bw`. One crunch config and no side thread.
    pub(in crate::vp8l) fn v16_encode_stream(&mut self, w: usize, h: usize) -> Result<(), Error> {
        let cx = self.cx;
        let mut s = self.v16.take().unwrap_or_else(|| Box::new(V16::new()));
        let method = self.method;
        // EncoderAnalyze, vp8l_enc.c:307.
        let use_palette_hint = true;
        self.histo_bits = get_histo_bits(method, use_palette_hint, w, h);
        self.transform_bits = get_transform_bits(method, self.histo_bits);
        let n_lz77s = if self.palette_size > 0 && self.palette_size <= 16 {
            2
        } else {
            1
        };
        let min_ix = analyze_entropy(
            cx,
            &mut self.ae_histo,
            &self.pic,
            w,
            h,
            w,
            use_palette_hint,
            self.palette_size,
            self.transform_bits,
        )?;
        let cc = CrunchConfig {
            entropy_idx: if min_ix == K_SUB_GREEN || min_ix == K_SPATIAL_SUB_GREEN {
                K_SPATIAL
            } else {
                min_ix
            },
            sub: [LZ77_STANDARD | LZ77_RLE, LZ77_BOX],
            sub_size: n_lz77s,
        };
        self.hc.init(cx, w * h)?; // EncoderInit :422

        self.use_palette = cc.entropy_idx == K_PALETTE;
        self.use_predict = cc.entropy_idx == K_SPATIAL;
        let mut cache_bits = 0usize;
        for r in self.refs.iter_mut().take(2) {
            r.clear();
        }

        // AllocateTransformBuffer :1226; w*h covers the packed palette image.
        cx.resize(&mut self.argb, w * h, 0)?;
        if self.use_palette {
            let (sorted, pal) = (self.palette_sorted, &mut self.palette);
            palette_sort_minimize_deltas(&sorted[..self.palette_size], pal);
            self.v16_encode_palette(&mut s)?;
            self.map_image_from_palette(w, h)?;
            // Cache no bigger than the palette.
            if self.palette_size < 1 << MAX_CACHE_BITS {
                cache_bits = (u32::BITS - (self.palette_size as u32).leading_zeros()) as usize;
            }
        } else {
            // MakeInputImageCopy :1267
            self.argb.copy_from_slice(&self.pic[..w * h]);
            self.current_width = w;
            if self.use_predict {
                cx.resize(&mut self.argb_scratch, 2 * (w + 1), 0)?;
                let n = sub_sample_size(w, MIN_HISTO_BITS) * sub_sample_size(h, MIN_HISTO_BITS); // MIN_TRANSFORM_BITS
                cx.resize(&mut self.transform_data, n, 0)?;
                self.v16_apply_predict_filter(&mut s, self.current_width, h)?;
            }
        }

        self.bw.put(0, 1)?; // no more transforms
        let argb = std::mem::take(&mut self.argb);
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        let (cw, hb) = (self.current_width, self.histo_bits);
        self.v16_encode_image_internal(
            &mut s, &argb, &mut hc, &mut refs, cw, h, &cc, cache_bits, hb,
        )?;
        self.argb = argb;
        self.hc = hc;
        self.refs = refs;
        self.v16 = Some(s);
        Ok(())
    }

    /// EncodePalette, vp8l_enc.c:1442: writes the colour indexing transform.
    fn v16_encode_palette(&mut self, s: &mut V16) -> Result<(), Error> {
        let mut tmp = [0u32; 256];
        let mut n = self.palette_size;
        // A trailing 0 is implied when there is no pixel packing.
        if self.palette[n - 1] == 0 && n > 17 {
            n -= 1;
        }
        self.bw.put(1, 1)?; // TRANSFORM_PRESENT
        self.bw.put(3, 2)?; // COLOR_INDEXING_TRANSFORM
        self.bw.put((n - 1) as u64, 8)?;
        for i in (1..n).rev() {
            tmp[i] = sub_pixels(self.palette[i], self.palette[i - 1]);
        }
        tmp[0] = self.palette[0];
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        self.v16_encode_image_no_huffman(&mut s.rs, &tmp[..n], &mut hc, &mut refs, n, 1, 20)?;
        self.hc = hc;
        self.refs = refs;
        Ok(())
    }

    /// ApplyPredictFilter, vp8l_enc.c:1100: runs the residual image and writes the predictor
    /// transform.
    fn v16_apply_predict_filter(&mut self, s: &mut V16, w: usize, h: usize) -> Result<(), Error> {
        const MIN_BITS: i64 = MIN_HISTO_BITS as i64; // MIN_TRANSFORM_BITS
        let max_bits = clamp_bits(
            w,
            h,
            self.transform_bits as i64,
            MIN_BITS,
            MAX_TRANS_BITS as i64,
            MAX_PRED_IMAGE,
        );
        let extra = if self.method > 4 {
            self.method as i64 - 4
        } else {
            0
        };
        let low_bits = clamp_bits(
            w,
            h,
            max_bits as i64 - 2 * extra,
            MIN_BITS,
            MAX_TRANS_BITS as i64,
            MAX_PRED_IMAGE,
        );
        let best_bits = residual_image(
            self.cx,
            w,
            h,
            low_bits,
            max_bits,
            &mut self.argb,
            &mut self.argb_scratch,
            &mut self.transform_data,
        )?;
        self.bw.put(1, 1)?; // TRANSFORM_PRESENT
        self.bw.put(0, 2)?; // PREDICTOR_TRANSFORM
        self.bw.put((best_bits - MIN_HISTO_BITS) as u64, 3)?;
        let td = std::mem::take(&mut self.transform_data);
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        let q = self.quality;
        self.v16_encode_image_no_huffman(
            &mut s.rs,
            &td,
            &mut hc,
            &mut refs,
            sub_sample_size(w, best_bits),
            sub_sample_size(h, best_bits),
            q,
        )?;
        self.transform_data = td;
        self.hc = hc;
        self.refs = refs;
        Ok(())
    }

    /// EncodeImageNoHuffman, vp8l_enc.c:769: writes a sub-image with one Huffman group. Its refs
    /// never use the cache, so the 1.2.2 histogram and Huffman storage apply.
    #[allow(clippy::too_many_arguments)]
    fn v16_encode_image_no_huffman(
        &mut self,
        rs: &mut RefsScratch,
        argb: &[u32],
        hc: &mut HashChain,
        refs: &mut [Refs],
        w: usize,
        h: usize,
        quality: i32,
    ) -> Result<(), Error> {
        let cx = self.cx;
        let symbols = [0u16; 1]; // one tree, one symbol

        hash_chain_fill(cx, &mut rs.hash_first, hc, quality, argb, w, h)?;
        rs.get_backward_references(
            cx,
            w,
            h,
            argb,
            quality,
            LZ77_STANDARD | LZ77_RLE,
            0,
            hc,
            refs,
        )?;
        self.histo_nh.alloc(cx, 1)?;
        self.histo_nh.get_mut(0).store_refs(&refs[0]);
        crate::vp8l::encode::get_huff_bit_lengths_and_codes(
            cx,
            &mut self.histo_nh,
            &mut self.huff_nh,
            &mut self.buf_rle,
            &mut self.huff_tree,
        )?;

        self.bw.put(0, 1)?; // no color cache, no Huffman image

        for k in 0..5 {
            let (o, ns) = (self.huff_nh.offs[k], self.huff_nh.ns[k]);
            store_huffman_code(
                &mut self.bw,
                &mut self.huff_tree,
                &mut self.tokens,
                &self.huff_nh.lens[o..o + ns],
            )?;
            let (l, b) = (
                &mut self.huff_nh.lens[o..o + ns],
                &mut self.huff_nh.bits[o..o + ns],
            );
            clear_huffman_tree_if_only_one_symbol(l, b);
        }
        crate::vp8l::encode::store_image_to_bit_mask(
            cx,
            &mut self.bw,
            w,
            0,
            &refs[0],
            &symbols,
            &self.huff_nh,
            0,
        )?;
        Ok(())
    }

    /// EncodeImageInternal, vp8l_enc.c:862: writes the main image, keeping the smaller of the
    /// sub-configs (strict <).
    #[allow(clippy::too_many_arguments)]
    fn v16_encode_image_internal(
        &mut self,
        s: &mut V16,
        argb: &[u32],
        hc: &mut HashChain,
        refs: &mut [Refs; 4],
        w: usize,
        h: usize,
        cc: &CrunchConfig,
        cache_bits: usize,
        histo_bits: usize,
    ) -> Result<(), Error> {
        let cx = self.cx;
        let quality = self.quality;
        let xy = sub_sample_size(w, histo_bits) * sub_sample_size(h, histo_bits);
        cx.resize(&mut s.histo_symbol, xy, 0)?;
        self.hc_histo.init(cx, xy)?;
        hash_chain_fill(cx, &mut s.rs.hash_first, hc, quality, argb, w, h)?;

        // Nonzero cache_bits came from the palette.
        let cache_bits_init = if cache_bits == 0 {
            MAX_CACHE_BITS
        } else {
            cache_bits
        };
        let (init_len, init_acc, init_nacc) = (self.bw.buf.len(), self.bw.acc, self.bw.nacc);
        let mut best_size = usize::MAX;
        let mut best_in_bw = false;
        for i in 0..cc.sub_size {
            let cache_bits_tmp = s.rs.get_backward_references(
                cx,
                w,
                h,
                argb,
                quality,
                cc.sub[i],
                cache_bits_init,
                hc,
                refs,
            )?;
            let mut histogram_bits = histo_bits;

            // VP8LBitWriterReset
            self.bw.buf.truncate(init_len);
            self.bw.acc = init_acc;
            self.bw.nacc = init_nacc;

            s.image.alloc(cx, xy, cache_bits_tmp)?;
            s.tmp = Histo::new(cache_bits_tmp);
            get_histo_image_symbols(
                cx,
                &mut s.hs,
                w,
                h,
                &refs[0],
                quality,
                histogram_bits,
                cache_bits_tmp,
                &mut s.image,
                &mut s.tmp,
                &mut s.histo_symbol,
            )?;
            get_huff_bit_lengths_and_codes(
                cx,
                &mut s.image,
                &mut s.huff,
                &mut self.buf_rle,
                &mut self.huff_tree,
            )?;

            if cache_bits_tmp > 0 {
                self.bw.put(1, 1)?;
                self.bw.put(cache_bits_tmp as u64, 4)?;
            } else {
                self.bw.put(0, 1)?;
            }

            let mut size = 0usize;
            for sym in s.histo_symbol.iter_mut() {
                if *sym as usize >= size {
                    size = *sym as usize + 1;
                }
                *sym <<= 8;
            }
            if size > 1 {
                self.bw.put(1, 1)?;
                histogram_bits =
                    optimize_sampling(&mut s.histo_symbol, w, h, histo_bits, MAX_HISTO_BITS);
                self.bw.put((histogram_bits - 2) as u64, 3)?;
                let mut hc_histo = std::mem::take(&mut self.hc_histo);
                self.v16_encode_image_no_huffman(
                    &mut s.rs,
                    &s.histo_symbol,
                    &mut hc_histo,
                    &mut refs[2..],
                    sub_sample_size(w, histogram_bits),
                    sub_sample_size(h, histogram_bits),
                    quality,
                )?;
                self.hc_histo = hc_histo;
            } else {
                self.bw.put(0, 1)?;
            }

            let ncodes = 5 * size;
            let mut skip = !0u64; // groups with empty red, blue, alpha
            for k in 0..ncodes {
                let (o, ns) = (s.huff.offs[k], s.huff.ns[k]);
                store_huffman_code(
                    &mut self.bw,
                    &mut self.huff_tree,
                    &mut self.tokens,
                    &s.huff.lens[o..o + ns],
                )?;
                let (l, b) = (&mut s.huff.lens[o..o + ns], &mut s.huff.bits[o..o + ns]);
                if !clear_huffman_tree_if_only_one_symbol(l, b)
                    && k % 5 != 0
                    && k % 5 != 4
                    && k / 5 < 64
                {
                    skip &= !(1u64 << (k / 5));
                }
            }
            store_image_to_bit_mask(
                cx,
                &mut self.bw,
                w,
                histogram_bits,
                &refs[0],
                &s.histo_symbol,
                &s.huff,
                skip,
            )?;

            best_in_bw = false;
            let n = self.bw.num_bytes();
            if n < best_size {
                best_size = n;
                best_in_bw = true;
                if i + 1 < cc.sub_size {
                    self.bw_best.buf.clear();
                    cx.extend_from_slice(&mut self.bw_best.buf, &self.bw.buf[init_len..])?;
                    self.bw_best.acc = self.bw.acc;
                    self.bw_best.nacc = self.bw.nacc;
                }
            }
        }
        if !best_in_bw {
            self.bw.buf.truncate(init_len);
            cx.extend_from_slice(&mut self.bw.buf, &self.bw_best.buf)?;
            self.bw.acc = self.bw_best.acc;
            self.bw.nacc = self.bw_best.nacc;
        }
        Ok(())
    }
}
