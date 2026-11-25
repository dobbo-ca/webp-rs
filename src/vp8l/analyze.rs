//! Port of libwebp 1.2.2 enc/vp8l_enc.c (analysis, palette), utils/utils.c WebPGetColorPalette and
//! enc/predictor_enc.c (exact = 1, no near-lossless or cross colour). Copyright 2012 Google Inc.
//! BSD-3, see LICENSE.libwebp.

use super::hist::bits_entropy;
use super::lossless::{
    ARGB_BLACK, bundle_color_map, combined_shannon_entropy, fast_log2, predictors_sub, sub_pixels,
    sub_sample_size,
};
use super::{
    CrunchConfig, K_DIRECT, K_PALETTE, K_SPATIAL, K_SPATIAL_SUB_GREEN, K_SUB_GREEN, LZ77_BOX,
    LZ77_RLE, LZ77_STANDARD, NUM_PRED_MODES,
};
use crate::Error;
use crate::limits::Ledger;

pub(super) const MAX_HISTO_BITS: usize = 9; // MAX_HUFFMAN_BITS
pub(super) const MIN_HISTO_BITS: usize = 2; // MIN_HUFFMAN_BITS
pub(super) const MAX_HUFF_IMAGE: usize = 2600; // MAX_HUFF_IMAGE_SIZE

// HistoIx, vp8l_enc.c:343.
pub(super) const HISTO_ALPHA: usize = 0;
pub(super) const HISTO_ALPHA_PRED: usize = 1;
pub(super) const HISTO_GREEN: usize = 2;
pub(super) const HISTO_GREEN_PRED: usize = 3;
pub(super) const HISTO_RED: usize = 4;
pub(super) const HISTO_RED_PRED: usize = 5;
pub(super) const HISTO_BLUE: usize = 6;
pub(super) const HISTO_BLUE_PRED: usize = 7;
pub(super) const HISTO_RED_SUB_GREEN: usize = 8;
pub(super) const HISTO_RED_PRED_SUB_GREEN: usize = 9;
pub(super) const HISTO_BLUE_SUB_GREEN: usize = 10;
pub(super) const HISTO_BLUE_PRED_SUB_GREEN: usize = 11;
pub(super) const HISTO_PALETTE: usize = 12;
pub(super) const HISTO_TOTAL: usize = 13;

/// kHistoPairs, vp8l_enc.c:502.
pub(super) const HISTO_PAIRS: [[usize; 2]; 5] = [
    [HISTO_RED, HISTO_BLUE],
    [HISTO_RED_PRED, HISTO_BLUE_PRED],
    [HISTO_RED_SUB_GREEN, HISTO_BLUE_SUB_GREEN],
    [HISTO_RED_PRED_SUB_GREEN, HISTO_BLUE_PRED_SUB_GREEN],
    [HISTO_RED, HISTO_BLUE],
];

/// The analysis of one plane: where the encoder goes next.
pub(super) struct Analysis {
    pub cc: CrunchConfig,
    pub histo_bits: usize,
    pub transform_bits: usize,
}

/// EncoderAnalyze, vp8l_enc.c:566, for the methods 3 to 5 the ALPH encoder runs at.
pub(super) fn encoder_analyze(
    cx: &Ledger,
    ae_histo: &mut Vec<u32>,
    pic: &[u32],
    w: usize,
    h: usize,
    method: usize,
    palette_size: usize,
) -> Result<Analysis, Error> {
    // encodeLossless set the sorted palette.
    let use_palette = true;
    let histo_bits = get_histo_bits(method, use_palette, w, h);
    let transform_bits = get_transform_bits(method, histo_bits);

    let n_lz77s = if palette_size > 0 && palette_size <= 16 {
        2
    } else {
        1
    };
    let (min_ix, _red_and_blue_always_zero) = analyze_entropy(
        cx,
        ae_histo,
        pic,
        w,
        h,
        w,
        use_palette,
        palette_size,
        transform_bits,
    )?;
    Ok(Analysis {
        cc: CrunchConfig {
            entropy_idx: min_ix,
            sub: [LZ77_STANDARD | LZ77_RLE, LZ77_BOX],
            sub_size: n_lz77s,
        },
        histo_bits,
        transform_bits,
    })
}

/// AnalyzeEntropy, vp8l_enc.c:381.
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
) -> Result<(usize, bool), Error> {
    if use_palette && palette_size <= 16 {
        return Ok((K_PALETTE, true));
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

    let mut entropy_comp = [0f64; HISTO_TOTAL];
    let mut entropy = [0f64; 5];
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
    entropy[K_DIRECT] = entropy_comp[HISTO_ALPHA]
        + entropy_comp[HISTO_RED]
        + entropy_comp[HISTO_GREEN]
        + entropy_comp[HISTO_BLUE];
    entropy[K_SPATIAL] = entropy_comp[HISTO_ALPHA_PRED]
        + entropy_comp[HISTO_RED_PRED]
        + entropy_comp[HISTO_GREEN_PRED]
        + entropy_comp[HISTO_BLUE_PRED];
    entropy[K_SUB_GREEN] = entropy_comp[HISTO_ALPHA]
        + entropy_comp[HISTO_RED_SUB_GREEN]
        + entropy_comp[HISTO_GREEN]
        + entropy_comp[HISTO_BLUE_SUB_GREEN];
    entropy[K_SPATIAL_SUB_GREEN] = entropy_comp[HISTO_ALPHA_PRED]
        + entropy_comp[HISTO_RED_PRED_SUB_GREEN]
        + entropy_comp[HISTO_GREEN_PRED]
        + entropy_comp[HISTO_BLUE_PRED_SUB_GREEN];
    entropy[K_PALETTE] = entropy_comp[HISTO_PALETTE];

    // uint32 tile count times float, as in C.
    let tiles = (sub_sample_size(w, transform_bits) as u32)
        .wrapping_mul(sub_sample_size(h, transform_bits) as u32) as f32;
    entropy[K_SPATIAL] += f64::from(tiles * fast_log2(14));
    entropy[K_SPATIAL_SUB_GREEN] += f64::from(tiles * fast_log2(24));
    entropy[K_PALETTE] += (palette_size * 8) as f64;

    let mut min_ix = K_DIRECT;
    for k in K_DIRECT + 1..=last_mode {
        if entropy[min_ix] > entropy[k] {
            min_ix = k;
        }
    }
    let mut red_and_blue_always_zero = true;
    let red = HISTO_PAIRS[min_ix][0] * 256;
    let blue = HISTO_PAIRS[min_ix][1] * 256;
    for i in 1..256 {
        if ae_histo[red + i] | ae_histo[blue + i] != 0 {
            red_and_blue_always_zero = false;
            break;
        }
    }
    Ok((min_ix, red_and_blue_always_zero))
}

/// AddSingleSubGreen, vp8l_enc.c:360. Only the low 8 bits matter, so C's signed shifts give the
/// same index.
pub(super) fn add_single_sub_green(h: &mut [u32], p: u32, r: usize, b: usize) {
    let green = p >> 8;
    h[r * 256 + (((p >> 16).wrapping_sub(green)) & 0xff) as usize] += 1;
    h[b * 256 + (p.wrapping_sub(green) & 0xff) as usize] += 1;
}

/// AddSingle, vp8l_enc.c:366.
pub(super) fn add_single(h: &mut [u32], p: u32, a: usize, r: usize, g: usize, b: usize) {
    h[a * 256 + ((p >> 24) & 0xff) as usize] += 1;
    h[r * 256 + ((p >> 16) & 0xff) as usize] += 1;
    h[g * 256 + ((p >> 8) & 0xff) as usize] += 1;
    h[b * 256 + (p & 0xff) as usize] += 1;
}

/// HashPix, vp8l_enc.c:375.
pub(super) fn hash_pix(p: u32) -> u32 {
    ((u64::from(p) + u64::from(p >> 19)).wrapping_mul(0x39c5_fba7) & 0xffff_ffff) as u32 >> 24
}

/// GetHistoBits, vp8l_enc.c:528.
fn get_histo_bits(method: usize, use_palette: bool, w: usize, h: usize) -> usize {
    let mut histo_bits = if use_palette { 9 - method } else { 7 - method };
    loop {
        let huff_image_size = sub_sample_size(w, histo_bits) * sub_sample_size(h, histo_bits);
        if huff_image_size <= MAX_HUFF_IMAGE {
            break;
        }
        histo_bits += 1;
    }
    histo_bits.clamp(MIN_HISTO_BITS, MAX_HISTO_BITS)
}

/// GetTransformBits, vp8l_enc.c:541.
pub(super) fn get_transform_bits(method: usize, histo_bits: usize) -> usize {
    let max_bits = if method < 4 {
        6
    } else if method > 4 {
        4
    } else {
        5
    };
    histo_bits.min(max_bits)
}

/// WebPGetColorPalette, sorted, utils.c:258: the alpha values in use as palette colours.
pub(super) fn alpha_palette(used: &[bool; 256], palette: &mut [u32; 256]) -> usize {
    let mut n = 0;
    for (v, &u) in used.iter().enumerate() {
        if u {
            palette[n] = (v as u32) << 8;
            n += 1;
        }
    }
    n
}

/// PaletteComponentDistance, vp8l_enc.c:41.
fn palette_component_distance(v: u32) -> u32 {
    if v <= 128 { v } else { 256 - v }
}

/// PaletteColorDistance, vp8l_enc.c:50.
pub(super) fn palette_color_distance(c1: u32, c2: u32) -> u32 {
    let diff = sub_pixels(c1, c2);
    const MORE_WEIGHT_FOR_RGB: u32 = 9;
    let mut score = palette_component_distance(diff & 0xff);
    score += palette_component_distance((diff >> 8) & 0xff);
    score += palette_component_distance((diff >> 16) & 0xff);
    score *= MORE_WEIGHT_FOR_RGB;
    score += palette_component_distance((diff >> 24) & 0xff);
    score
}

/// PaletteHasNonMonotonousDeltas, vp8l_enc.c:92.
pub(super) fn palette_has_non_monotonous_deltas(p: &[u32]) -> bool {
    let mut predict = 0u32;
    let mut sign_found = 0u8;
    for &c in p {
        let diff = sub_pixels(c, predict);
        let (rd, gd, bd) = ((diff >> 16) as u8, (diff >> 8) as u8, diff as u8);
        if rd != 0 {
            sign_found |= if rd < 0x80 { 1 } else { 2 };
        }
        if gd != 0 {
            sign_found |= if gd < 0x80 { 8 } else { 16 };
        }
        if bd != 0 {
            sign_found |= if bd < 0x80 { 64 } else { 128 };
        }
        predict = c;
    }
    // Bit 8 of the C int shift is masked by the & anyway.
    sign_found & (sign_found << 1) != 0
}

/// PaletteSortMinimizeDeltas, vp8l_enc.c:116; palette.len() >= sorted.len().
pub(super) fn palette_sort_minimize_deltas(sorted: &[u32], palette: &mut [u32]) {
    let n = sorted.len();
    palette[..n].copy_from_slice(sorted);
    if !palette_has_non_monotonous_deltas(sorted) {
        return;
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

/// ApplyPalette, vp8l_enc.c:1562: maps `src` to packed palette indices; `row` is tmp_row. A
/// green-byte table stands in for C's greedy/hash search: every lookup returns the colour's palette
/// index, so the bytes match. It needs distinct greens, true for ALPH (a << 8).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_palette(
    cx: &Ledger,
    src: &[u32],
    src_stride: usize,
    dst: &mut [u32],
    dst_stride: usize,
    palette: &[u32],
    w: usize,
    h: usize,
    xbits: usize,
    row: &mut [u8],
) -> Result<(), Error> {
    let mut lut = [0u8; 256];
    for (j, &c) in palette.iter().enumerate() {
        lut[((c >> 8) & 0xff) as usize] = j as u8;
    }
    for y in 0..h {
        cx.check()?;
        let s = &src[y * src_stride..y * src_stride + w];
        for (x, &pix) in s.iter().enumerate() {
            row[x] = lut[((pix >> 8) & 0xff) as usize];
        }
        bundle_color_map(row, w, xbits, &mut dst[y * dst_stride..]);
    }
    Ok(())
}

// predictor_enc.c ------------------------------------------------------------

const MAX_DIFF_COST: f32 = 1e30; // MAX_DIFF_COST
const SPATIAL_PREDICTOR_BIAS: f32 = 15.0; // kSpatialPredictorBias
const EXP_DECAY_FACTOR: f64 = 0.6;
const SPATIAL_EXP_VALUE: f64 = 0.94;
const SPATIAL_COST_SCALE: f64 = -0.1;

/// PredictionCostSpatial, predictor_enc.c:33.
fn prediction_cost_spatial(counts: &[i32; 256], weight0: i32, mut exp_val: f64) -> f32 {
    const SIGNIFICANT_SYMBOLS: usize = 256 >> 4;
    let mut bits = f64::from(weight0.wrapping_mul(counts[0]));
    for i in 1..SIGNIFICANT_SYMBOLS {
        bits += exp_val * f64::from(counts[i] + counts[256 - i]);
        exp_val *= EXP_DECAY_FACTOR;
    }
    (SPATIAL_COST_SCALE * bits) as f32
}

/// PredictBatch, predictor_enc.c:68: writes out[0..n] for cur[x_start..x_start + n]; cur and upper
/// are whole rows.
pub(super) fn predict_batch(
    mode: usize,
    mut x_start: usize,
    y: usize,
    mut n: usize,
    cur: &[u32],
    upper: &[u32],
    out: &mut [u32],
) {
    let mut out = out;
    if x_start == 0 {
        if y == 0 {
            predictors_sub(0, cur, &[], 0, 1, out); // ARGB_BLACK
        } else {
            predictors_sub(2, cur, upper, 0, 1, out); // top
        }
        x_start += 1;
        out = &mut out[1..];
        n -= 1;
    }
    if y == 0 {
        predictors_sub(1, cur, &[], x_start, n, out); // left
    } else {
        predictors_sub(mode, cur, upper, x_start, n, out);
    }
}

/// GetBestPredictorForTile, predictor_enc.c:298; needs green-only argb (v << 8).
#[allow(clippy::too_many_arguments)]
fn get_best_predictor_for_tile(
    w: usize,
    h: usize,
    tile_x: usize,
    tile_y: usize,
    bits: usize,
    acc: &mut [[i32; 256]; 4],
    argb: &[u32],
    modes: &[u32],
) -> usize {
    let start_x = tile_x << bits;
    let start_y = tile_y << bits;
    let tile_size = 1usize << bits;
    let max_y = tile_size.min(h - start_y);
    let max_x = tile_size.min(w - start_x);
    let tiles_per_row = sub_sample_size(w, bits);
    let (mut left_mode, mut above_mode) = (0xffusize, 0xffusize);
    if tile_x > 0 {
        left_mode = ((modes[tile_y * tiles_per_row + tile_x - 1] >> 8) & 0xff) as usize;
    }
    if tile_y > 0 {
        above_mode = ((modes[(tile_y - 1) * tiles_per_row + tile_x] >> 8) & 0xff) as usize;
    }
    let mut green = [[0i32; 256]; NUM_PRED_MODES];
    let (mut black, mut inner) = (0i32, 0i32); // black: all modes; inner: mode 0
    for y in start_y..start_y + max_y {
        let row = &argb[y * w..y * w + w];
        for x in start_x..start_x + max_x {
            let g = i32::from((row[x] >> 8) as u8);
            if y == 0 || x == 0 {
                // PredictBatch: black, left or top.
                let mut p = 0;
                if x > 0 {
                    p = i32::from((row[x - 1] >> 8) as u8);
                } else if y > 0 {
                    p = i32::from((argb[(y - 1) * w] >> 8) as u8);
                } else {
                    black += 1;
                }
                for gm in green.iter_mut() {
                    gm[((g - p) as u8) as usize] += 1;
                }
                continue;
            }
            // TR wraps at x = w - 1.
            let up = &argb[(y - 1) * w + x - 1..];
            let l = i32::from((row[x - 1] >> 8) as u8);
            let (tl, t, tr) = (
                i32::from((up[0] >> 8) as u8),
                i32::from((up[1] >> 8) as u8),
                i32::from((up[2] >> 8) as u8),
            );
            // Select(T, L, TL)
            let sel = if (l - tl).abs() - (t - tl).abs() <= 0 {
                t
            } else {
                l
            };
            let ave = (l + t) >> 1;
            let p: [i32; NUM_PRED_MODES] = [
                0,
                l,
                t,
                tr,
                tl,
                (((l + tr) >> 1) + t) >> 1,
                (l + tl) >> 1,
                ave,
                (tl + t) >> 1,
                (t + tr) >> 1,
                (((l + tl) >> 1) + ((t + tr) >> 1)) >> 1,
                sel,
                (l + t - tl).clamp(0, 255),
                (ave + (ave - tl) / 2).clamp(0, 255),
            ];
            for (m, gm) in green.iter_mut().enumerate() {
                gm[((g - p[m]) as u8) as usize] += 1;
            }
            inner += 1;
        }
    }

    let n = (max_x * max_y) as i32;
    let mut alpha = [0i32; 256];
    let mut rb = [0i32; 256];
    rb[0] = n;
    let ps_rb = prediction_cost_spatial(&rb, 1, SPATIAL_EXP_VALUE);
    let ce_r = combined_shannon_entropy(&rb, &acc[1]);
    let ce_b = combined_shannon_entropy(&rb, &acc[3]);
    let n_alpha1 = [black + inner, black]; // mode 0, others
    let mut ps_a = [0f32; 2];
    let mut ce_a = [0f32; 2];
    for (i, &k) in n_alpha1.iter().enumerate() {
        alpha[0] = n - k;
        alpha[1] = k;
        ps_a[i] = prediction_cost_spatial(&alpha, 1, SPATIAL_EXP_VALUE);
        ce_a[i] = combined_shannon_entropy(&alpha, &acc[0]);
    }
    let mut best_diff = MAX_DIFF_COST;
    let mut best_mode = 0usize;
    for mode in 0..NUM_PRED_MODES {
        let a = mode.min(1);
        // PredictionCostSpatialHistogram, predictor_enc.c:46.
        let mut retval = 0f64;
        retval += f64::from(ps_a[a]);
        retval += f64::from(ce_a[a]);
        retval += f64::from(ps_rb);
        retval += f64::from(ce_r);
        retval += f64::from(prediction_cost_spatial(&green[mode], 1, SPATIAL_EXP_VALUE));
        retval += f64::from(combined_shannon_entropy(&green[mode], &acc[2]));
        retval += f64::from(ps_rb);
        retval += f64::from(ce_b);
        let mut cur_diff = retval as f32;
        // Favor keeping the areas locally similar.
        if mode == left_mode {
            cur_diff -= SPATIAL_PREDICTOR_BIAS;
        }
        if mode == above_mode {
            cur_diff -= SPATIAL_PREDICTOR_BIAS;
        }
        if cur_diff < best_diff {
            best_diff = cur_diff;
            best_mode = mode;
        }
    }

    let k = n_alpha1[best_mode.min(1)];
    acc[0][0] += n - k;
    acc[0][1] += k;
    acc[1][0] += n;
    for (j, &v) in green[best_mode].iter().enumerate() {
        acc[2][j] += v;
    }
    acc[3][0] += n;
    best_mode
}

/// CopyImageWithPrediction, predictor_enc.c:413.
pub(super) fn copy_image_with_prediction(
    cx: &Ledger,
    w: usize,
    h: usize,
    bits: usize,
    modes: &[u32],
    scratch: &mut [u32],
    argb: &mut [u32],
) -> Result<(), Error> {
    let tiles_per_row = sub_sample_size(w, bits);
    let (mut upper_off, mut cur_off) = (0usize, w + 1);
    for y in 0..h {
        cx.check()?;
        std::mem::swap(&mut upper_off, &mut cur_off);
        let n = if y + 1 < h { w + 1 } else { w };
        scratch[cur_off..cur_off + n].copy_from_slice(&argb[y * w..y * w + n]);
        let mut x = 0;
        while x < w {
            let mode = ((modes[(y >> bits) * tiles_per_row + (x >> bits)] >> 8) & 0xff) as usize;
            let x_end = (x + (1 << bits)).min(w);
            let (upper, cur) = if upper_off < cur_off {
                let (a, b) = scratch.split_at(cur_off);
                (&a[upper_off..upper_off + w + 1], &b[..w + 1])
            } else {
                let (a, b) = scratch.split_at(upper_off);
                (&b[..w + 1], &a[cur_off..cur_off + w + 1])
            };
            predict_batch(mode, x, y, x_end - x, cur, upper, &mut argb[y * w + x..]);
            x = x_end;
        }
    }
    Ok(())
}

/// VP8LResidualImage, predictor_enc.c:475: writes modes to `image` and residuals over `argb`.
pub(super) fn residual_image(
    cx: &Ledger,
    w: usize,
    h: usize,
    bits: usize,
    argb: &mut [u32],
    scratch: &mut [u32],
    image: &mut [u32],
) -> Result<(), Error> {
    let tiles_per_row = sub_sample_size(w, bits);
    let tiles_per_col = sub_sample_size(h, bits);
    let mut histo = [[0i32; 256]; 4];
    for tile_y in 0..tiles_per_col {
        cx.check()?;
        for tile_x in 0..tiles_per_row {
            let pred =
                get_best_predictor_for_tile(w, h, tile_x, tile_y, bits, &mut histo, argb, image);
            image[tile_y * tiles_per_row + tile_x] = ARGB_BLACK | ((pred as u32) << 8);
        }
    }
    copy_image_with_prediction(cx, w, h, bits, image, scratch, argb)?;
    Ok(())
}
