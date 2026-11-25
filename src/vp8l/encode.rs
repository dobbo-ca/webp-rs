//! Port of libwebp 1.2.2 enc/vp8l_enc.c (stream and Huffman storage), utils/bit_writer_utils.h,
//! enc/alpha_enc.c, utils/filters_utils.c and dsp/filters.c. Copyright 2012 Google Inc. BSD-3, see
//! LICENSE.libwebp.

use super::analyze::{
    alpha_palette, apply_palette, encoder_analyze, palette_sort_minimize_deltas, residual_image,
};
use super::hist::{Histo, HistoSet, NUM_CODES, get_histo_image_symbols};
use super::huff::{Code, Token, Tree, create_compressed_huffman_tree, create_huffman_tree};
use super::lossless::{sub_pixels, sub_sample_size};
use super::refs::{HashChain, Refs, get_backward_references, hash_chain_fill};
use super::{
    BitWriter, CL_ORDER, K_PALETTE, K_SPATIAL, K_SPATIAL_SUB_GREEN, K_SUB_GREEN, LITERAL, LZ77_RLE,
    LZ77_STANDARD, Lw, NUM_DISTANCE, NUM_LITERAL, cap_bytes, prefix_bits,
};
use crate::Error;
use crate::limits::Ledger;

const CODE_LEN_CODES: usize = 19;

// WEBP_FILTER_TYPE, filters_utils.h.
const FILTER_NONE: usize = 0;
const FILTER_HORIZONTAL: usize = 1;
const FILTER_VERTICAL: usize = 2;
const FILTER_GRADIENT: usize = 3;

/// Five codes per histogram and their backing.
#[derive(Default)]
pub(super) struct HuffCodes {
    /// Offset and symbol count of each code in `lens` and `bits`.
    pub(super) offs: Vec<usize>,
    pub(super) ns: Vec<usize>,
    pub(super) lens: Vec<u8>,
    pub(super) bits: Vec<u16>,
}

/// The number of symbols of code i of a group of five.
fn num_symbols(i: usize) -> usize {
    match i % 5 {
        0 => NUM_CODES,
        4 => NUM_DISTANCE,
        _ => NUM_LITERAL,
    }
}

impl HuffCodes {
    /// The bytes held, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        cap_bytes(&self.offs) + cap_bytes(&self.ns) + cap_bytes(&self.lens) + cap_bytes(&self.bits)
    }

    /// WriteHuffmanCode, vp8l_enc.c:933: symbol `s` of code `i`.
    pub(super) fn write(&self, bw: &mut BitWriter, i: usize, s: usize) -> Result<(), Error> {
        let o = self.offs[i] + s;
        bw.put(u64::from(self.bits[o]), i32::from(self.lens[o]))?;
        Ok(())
    }

    /// WriteHuffmanCodeWithExtraBits, vp8l_enc.c:941.
    pub(super) fn write_with_extra_bits(
        &self,
        bw: &mut BitWriter,
        i: usize,
        s: usize,
        bits: u64,
        n: i32,
    ) -> Result<(), Error> {
        let o = self.offs[i] + s;
        let depth = i32::from(self.lens[o]);
        bw.put((bits << depth) | u64::from(self.bits[o]), depth + n)?;
        Ok(())
    }
}

/// ClearHuffmanTreeIfOnlyOneSymbol, vp8l_enc.c:786: reports whether it cleared the code.
pub(super) fn clear_huffman_tree_if_only_one_symbol(lens: &mut [u8], codes: &mut [u16]) -> bool {
    if lens.iter().filter(|&&l| l != 0).count() > 1 {
        return false;
    }
    lens.fill(0);
    codes.fill(0);
    true
}

/// StoreHuffmanTreeOfHuffmanTreeToBitMask, vp8l_enc.c:764 (kStorageOrder is CL_ORDER).
fn store_huffman_tree_of_huffman_tree_to_bit_mask(
    bw: &mut BitWriter,
    depths: &[u8; CODE_LEN_CODES],
) -> Result<(), Error> {
    let mut n = CODE_LEN_CODES;
    while n > 4 {
        if depths[CL_ORDER[n - 1] as usize] != 0 {
            break;
        }
        n -= 1;
    }
    bw.put((n - 4) as u64, 4)?;
    for &c in CL_ORDER.iter().take(n) {
        bw.put(u64::from(depths[c as usize]), 3)?;
    }
    Ok(())
}

/// StoreHuffmanTreeToBitMask, vp8l_enc.c:802.
fn store_huffman_tree_to_bit_mask(
    bw: &mut BitWriter,
    tokens: &[Token],
    lens: &[u8],
    codes: &[u16],
) -> Result<(), Error> {
    for t in tokens {
        let ix = t.code as usize;
        bw.put(u64::from(codes[ix]), i32::from(lens[ix]))?;
        match t.code {
            16 => bw.put(u64::from(t.extra_bits), 2)?,
            17 => bw.put(u64::from(t.extra_bits), 3)?,
            18 => bw.put(u64::from(t.extra_bits), 7)?,
            _ => {}
        }
    }
    Ok(())
}

/// StoreFullHuffmanCode, vp8l_enc.c:826. `tree` and `tokens` are scratch.
fn store_full_huffman_code(
    bw: &mut BitWriter,
    tree: &mut [Tree],
    tokens: &mut Vec<Token>,
    c_lens: &[u8],
) -> Result<(), Error> {
    let mut depths = [0u8; CODE_LEN_CODES];
    let mut symbols = [0u16; CODE_LEN_CODES];

    bw.put(0, 1)?;
    create_compressed_huffman_tree(c_lens, tokens);
    let num_tokens = tokens.len();
    {
        let mut histogram = [0u32; CODE_LEN_CODES];
        let mut buf_rle = [0u8; CODE_LEN_CODES];
        for t in tokens.iter() {
            histogram[t.code as usize] += 1;
        }
        let mut code = Code {
            num_symbols: CODE_LEN_CODES,
            code_lengths: &mut depths,
            codes: &mut symbols,
        };
        create_huffman_tree(&mut histogram, 7, &mut buf_rle, tree, &mut code);
    }

    store_huffman_tree_of_huffman_tree_to_bit_mask(bw, &depths)?;
    clear_huffman_tree_if_only_one_symbol(&mut depths, &mut symbols);

    let mut trailing_zero_bits = 0;
    let mut trimmed_length = num_tokens;
    for i in (0..num_tokens).rev() {
        let ix = tokens[i].code as usize;
        if ix == 0 || ix == 17 || ix == 18 {
            trimmed_length -= 1; // discount trailing zeros
            trailing_zero_bits += i32::from(depths[ix]);
            if ix == 17 {
                trailing_zero_bits += 3;
            } else if ix == 18 {
                trailing_zero_bits += 7;
            }
        } else {
            break;
        }
    }
    let write_trimmed = trimmed_length > 1 && trailing_zero_bits > 12;
    let mut length = num_tokens;
    if write_trimmed {
        length = trimmed_length;
        bw.put(1, 1)?;
        if trimmed_length == 2 {
            bw.put(0, 3 + 2)?; // nbitpairs=1, trimmed_length=2
        } else {
            let nbits = (31 - ((trimmed_length - 2) as u32).leading_zeros()) as i32; // BitsLog2Floor
            let nbitpairs = nbits / 2 + 1;
            bw.put((nbitpairs - 1) as u64, 3)?;
            bw.put((trimmed_length - 2) as u64, nbitpairs * 2)?;
        }
    } else {
        bw.put(0, 1)?;
    }
    store_huffman_tree_to_bit_mask(bw, &tokens[..length], &depths, &symbols)?;
    Ok(())
}

/// StoreHuffmanCode, vp8l_enc.c:894.
pub(super) fn store_huffman_code(
    bw: &mut BitWriter,
    tree: &mut [Tree],
    tokens: &mut Vec<Token>,
    lens: &[u8],
) -> Result<(), Error> {
    const MAX_SYMBOL: usize = 1 << 8;
    let mut count = 0;
    let mut symbols = [0usize; 2];
    for (i, &l) in lens.iter().enumerate() {
        if count >= 3 {
            break;
        }
        if l != 0 {
            if count < 2 {
                symbols[count] = i;
            }
            count += 1;
        }
    }

    if count == 0 {
        // Minimal tree for empty cases.
        bw.put(0x01, 4)?;
    } else if count <= 2 && symbols[0] < MAX_SYMBOL && symbols[1] < MAX_SYMBOL {
        bw.put(1, 1)?; // small tree marker
        bw.put((count - 1) as u64, 1)?;
        if symbols[0] <= 1 {
            bw.put(0, 1)?;
            bw.put(symbols[0] as u64, 1)?;
        } else {
            bw.put(1, 1)?;
            bw.put(symbols[0] as u64, 8)?;
        }
        if count == 2 {
            bw.put(symbols[1] as u64, 8)?;
        }
    } else {
        store_full_huffman_code(bw, tree, tokens, lens)?;
    }
    Ok(())
}

/// StoreImageToBitMask, vp8l_enc.c:952; `codes` holds 5 per histogram. Skip bit g: group g is
/// green-only.
pub(super) fn store_image_to_bit_mask(
    cx: &Ledger,
    bw: &mut BitWriter,
    w: usize,
    histo_bits: usize,
    refs: &[super::refs::PixOrCopy],
    symbols: &[u16],
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
    let mut c = 5 * usize::from(symbols[0]);
    let mut rest = skip.checked_shr(u32::from(symbols[0])).unwrap_or(0) & 1 == 0;
    for v in refs {
        if tile_x != x as i64 & tile_mask || tile_y != y as i64 & tile_mask {
            tile_x = x as i64 & tile_mask;
            tile_y = y as i64 & tile_mask;
            let s = symbols[(y >> histo_bits) * histo_xsize + (x >> histo_bits)];
            c = 5 * usize::from(s);
            rest = skip.checked_shr(u32::from(s)).unwrap_or(0) & 1 == 0;
        }
        if v.mode == LITERAL {
            codes.write(bw, c, ((v.arg >> 8) & 0xff) as usize)?;
            if rest {
                codes.write(bw, c + 1, ((v.arg >> 16) & 0xff) as usize)?;
                codes.write(bw, c + 2, (v.arg & 0xff) as usize)?;
                codes.write(bw, c + 3, (v.arg >> 24) as usize)?;
            }
        } else {
            // kCacheIdx is unreachable with use_cache = 0.
            let (code, n, extra) = prefix_bits(usize::from(v.len));
            codes.write_with_extra_bits(bw, c, 256 + code, extra as u64, n as i32)?;
            // Distance extra bits go separately, up to 18 bits.
            let (code, n, extra) = prefix_bits(v.arg as usize);
            codes.write(bw, c + 4, code)?;
            bw.put(extra as u64, n as i32)?;
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

/// GetHuffBitLengthsAndCodes, vp8l_enc.c:690: sizes `hc` to 5 codes per histogram and builds them.
/// It mutates the histograms, as C does.
pub(super) fn get_huff_bit_lengths_and_codes(
    cx: &Ledger,
    set: &mut HistoSet,
    hc: &mut HuffCodes,
    buf_rle: &mut Vec<u8>,
    huff_tree: &mut Vec<Tree>,
) -> Result<(), Error> {
    let n = set.len();
    let total = n * (NUM_CODES + 3 * NUM_LITERAL + NUM_DISTANCE);
    hc.offs.clear();
    hc.ns.clear();
    cx.resize(&mut hc.lens, total, 0)?; // calloc
    cx.resize(&mut hc.bits, total, 0)?;
    cx.reserve(&mut hc.offs, 5 * n)?;
    cx.reserve(&mut hc.ns, 5 * n)?;
    let mut off = 0;
    for i in 0..5 * n {
        hc.offs.push(off);
        hc.ns.push(num_symbols(i));
        off += num_symbols(i);
    }

    buf_rle.clear();
    buf_rle.resize(NUM_CODES, 0);
    huff_tree.clear();
    huff_tree.resize(3 * NUM_CODES, Tree::default());

    for i in 0..n {
        let histo: &mut Histo = set.get_mut(i);
        let parts: [&mut [u32]; 5] = [
            &mut histo.literal,
            &mut histo.red,
            &mut histo.blue,
            &mut histo.alpha,
            &mut histo.distance,
        ];
        for (k, hist) in parts.into_iter().enumerate() {
            let idx = 5 * i + k;
            let (o, ns) = (hc.offs[idx], num_symbols(idx));
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

// Alpha filters ---------------------------------------------------------------

/// GradientPredictor, filters_utils.c:24 and filters.c:121.
fn gradient_predictor(a: u8, b: u8, c: u8) -> i32 {
    let g = i32::from(a) + i32::from(b) - i32::from(c);
    if g & !0xff == 0 {
        g
    } else if g < 0 {
        0
    } else {
        255
    }
}

/// HorizontalFilter_C, filters.c:173 (DoHorizontalFilter_C :45).
fn horizontal_filter(in_: &[u8], w: usize, h: usize, out: &mut [u8]) {
    out[0] = in_[0];
    for i in 1..w {
        out[i] = in_[i].wrapping_sub(in_[i - 1]);
    }
    for y in 1..h {
        let o = y * w;
        out[o] = in_[o].wrapping_sub(in_[o - w]);
        for i in 1..w {
            out[o + i] = in_[o + i].wrapping_sub(in_[o + i - 1]);
        }
    }
}

/// VerticalFilter_C, filters.c:179 (DoVerticalFilter_C :82).
fn vertical_filter(in_: &[u8], w: usize, h: usize, out: &mut [u8]) {
    out[0] = in_[0];
    for i in 1..w {
        out[i] = in_[i].wrapping_sub(in_[i - 1]);
    }
    for y in 1..h {
        let o = y * w;
        for i in 0..w {
            out[o + i] = in_[o + i].wrapping_sub(in_[o + i - w]);
        }
    }
}

/// GradientFilter_C, filters.c:184 (DoGradientFilter_C :127).
fn gradient_filter(in_: &[u8], w: usize, h: usize, out: &mut [u8]) {
    out[0] = in_[0];
    for i in 1..w {
        out[i] = in_[i].wrapping_sub(in_[i - 1]);
    }
    for y in 1..h {
        let o = y * w;
        out[o] = in_[o].wrapping_sub(in_[o - w]);
        for x in 1..w {
            let pred = gradient_predictor(in_[o + x - 1], in_[o + x - w], in_[o + x - w - 1]);
            out[o + x] = (i32::from(in_[o + x]) - pred) as u8;
        }
    }
}

/// GetNumColors, alpha_enc.c:179.
fn get_num_colors(a: &[u8]) -> usize {
    let mut color = [false; 256];
    for &v in a {
        color[v as usize] = true;
    }
    color.iter().filter(|&&c| c).count()
}

/// GetFilterMap, alpha_enc.c:202, the WEBP_FILTER_FAST case only.
fn get_filter_map(a: &[u8], w: usize, h: usize, effort: usize) -> u32 {
    const MIN_COLORS_FOR_FILTER_NONE: usize = 16;
    const MAX_COLORS_FOR_FILTER_NONE: usize = 192;
    let try_filter_none = effort > 3;
    let num_colors = get_num_colors(a);
    let mut filter = FILTER_NONE;
    if num_colors > MIN_COLORS_FOR_FILTER_NONE {
        filter = estimate_best_filter(a, w, h);
    }
    let mut bit_map = 1u32 << filter;
    if try_filter_none || num_colors > MAX_COLORS_FOR_FILTER_NONE {
        bit_map |= 1 << FILTER_NONE;
    }
    bit_map
}

/// WebPEstimateBestFilter, filters_utils.c:29.
fn estimate_best_filter(a: &[u8], w: usize, h: usize) -> usize {
    const SMAX: usize = 16;
    let sdiff = |x: i32, y: i32| ((x - y).max(y - x) >> 4) as usize;
    let mut bins = [[0i32; SMAX]; 4];
    // Every other pixel is sampled.
    let mut j = 2;
    while j + 1 < h {
        let p = j * w;
        let mut mean = i32::from(a[p]);
        let mut i = 2;
        while i + 1 < w {
            let x = i32::from(a[p + i]);
            let diff0 = sdiff(x, mean);
            let diff1 = sdiff(x, i32::from(a[p + i - 1]));
            let diff2 = sdiff(x, i32::from(a[p + i - w]));
            let grad_pred = gradient_predictor(a[p + i - 1], a[p + i - w], a[p + i - w - 1]);
            let diff3 = sdiff(x, grad_pred);
            bins[FILTER_NONE][diff0] = 1;
            bins[FILTER_HORIZONTAL][diff1] = 1;
            bins[FILTER_VERTICAL][diff2] = 1;
            bins[FILTER_GRADIENT][diff3] = 1;
            mean = (3 * mean + x + 2) >> 2;
            i += 2;
        }
        j += 2;
    }
    let mut best_filter = FILTER_NONE;
    let mut best_score = 0x7fff_ffffi32;
    for filter in FILTER_NONE..4 {
        let mut score = 0;
        for i in 0..SMAX {
            if bins[filter][i] > 0 {
                score += i as i32;
            }
        }
        if score < best_score {
            best_score = score;
            best_filter = filter;
        }
    }
    best_filter
}

impl Lw<'_> {
    /// Appends the ALPH chunk data (header byte, then payload) of `a` (w x h, stride w) to `dst`, as
    /// libwebp 1.2.2 (1.6.0 with `libwebp16`) EncodeAlpha with alpha_quality 100,
    /// alpha_compression 1 and alpha_filtering 1. `method` is the WebP method, 3 to 5.
    /// EncodeAlpha, alpha_enc.c:292.
    pub(crate) fn alpha(
        &mut self,
        dst: Vec<u8>,
        a: &[u8],
        w: usize,
        h: usize,
        method: usize,
    ) -> Result<Vec<u8>, Error> {
        self.method = method;
        self.quality = 8 * method as i32;
        self.apply_filters_and_encode(dst, a, w, h)
    }

    /// ApplyFiltersAndEncode, alpha_enc.c:234: the smallest filter trial.
    fn apply_filters_and_encode(
        &mut self,
        mut dst: Vec<u8>,
        a: &[u8],
        w: usize,
        h: usize,
    ) -> Result<Vec<u8>, Error> {
        let cx = self.cx;
        cx.check()?;
        let mut try_map = get_filter_map(a, w, h, self.method);
        if try_map == 1 << FILTER_NONE {
            return self.encode_alpha_internal(dst, a, w, h, FILTER_NONE);
        }
        let base = dst.len();
        let mut best = usize::MAX; // C uses 2^32-1, never reached
        let mut filter = FILTER_NONE;
        while try_map != 0 {
            cx.check()?;
            if try_map & 1 != 0 {
                let trial = std::mem::take(&mut self.trial);
                let trial = self.encode_alpha_internal(trial, a, w, h, filter)?;
                if trial.len() < best {
                    dst.truncate(base);
                    cx.extend_from_slice(&mut dst, &trial)?;
                    best = trial.len();
                }
                self.trial = trial;
                self.trial.clear();
            }
            filter += 1;
            try_map >>= 1;
        }
        Ok(dst)
    }

    /// EncodeAlphaInternal, alpha_enc.c:109: appends header + payload for one filter to `dst`.
    fn encode_alpha_internal(
        &mut self,
        mut dst: Vec<u8>,
        a: &[u8],
        w: usize,
        h: usize,
        filter: usize,
    ) -> Result<Vec<u8>, Error> {
        let cx = self.cx;
        let mut filt = std::mem::take(&mut self.filt);
        cx.resize(&mut filt, w * h, 0)?;
        let src: &[u8] = match filter {
            FILTER_HORIZONTAL => {
                horizontal_filter(a, w, h, &mut filt);
                &filt
            }
            FILTER_VERTICAL => {
                vertical_filter(a, w, h, &mut filt);
                &filt
            }
            FILTER_GRADIENT => {
                gradient_filter(a, w, h, &mut filt);
                &filt
            }
            _ => a,
        };
        let start = dst.len();
        let mut method = 1u8; // ALPHA_LOSSLESS_COMPRESSION
        cx.push(&mut dst, 0)?;
        let mut dst = self.encode_lossless(dst, src, w, h)?;
        if dst.len() - start - 1 > w * h {
            method = 0; // ALPHA_NO_COMPRESSION
            dst.truncate(start + 1);
            cx.extend_from_slice(&mut dst, src)?;
        }
        dst[start] = method | ((filter as u8) << 2);
        self.filt = filt;
        Ok(dst)
    }

    /// EncodeLossless, alpha_enc.c:49: appends the VP8L stream of plane `a` to `dst`.
    fn encode_lossless(
        &mut self,
        mut dst: Vec<u8>,
        a: &[u8],
        w: usize,
        h: usize,
    ) -> Result<Vec<u8>, Error> {
        let cx = self.cx;
        self.pic.clear();
        cx.reserve(&mut self.pic, a.len())?;
        self.pic.extend(a.iter().map(|&v| u32::from(v) << 8)); // AlphaToGreen_C
        let mut used = [false; 256];
        for &v in a {
            used[v as usize] = true;
        }
        self.palette_size = alpha_palette(&used, &mut self.palette_sorted);
        // Raw wins past w*h; slack covers headers.
        let cap = dst.len() + w * h + ((w * h) >> 3) + 1024;
        cx.reserve(&mut dst, cap)?;
        self.bw = BitWriter {
            buf: dst,
            acc: 0,
            nacc: 0,
            cx: Some(cx),
        };
        if self.libwebp16 {
            self.v16_encode_stream(w, h)?;
        } else {
            self.encode_stream(w, h)?;
        }
        self.bw.flush()
    }

    /// VP8LEncodeStream, vp8l_enc.c:1896, and EncodeStreamHook :1719: writes the VP8L stream of
    /// `pic` (no header) to `bw`. One crunch config and no side thread, so the hook's best-of swap
    /// is the identity.
    fn encode_stream(&mut self, w: usize, h: usize) -> Result<(), Error> {
        let cx = self.cx;
        let an = encoder_analyze(
            cx,
            &mut self.ae_histo,
            &self.pic,
            w,
            h,
            self.method,
            self.palette_size,
        )?;
        let mut cc = an.cc;
        self.histo_bits = an.histo_bits;
        self.transform_bits = an.transform_bits;
        // Subtract green and cross colour are not reachable at the methods ALPH runs at: the
        // entropy mode is palette, spatial or direct.
        if cc.entropy_idx == K_SUB_GREEN || cc.entropy_idx == K_SPATIAL_SUB_GREEN {
            cc.entropy_idx = K_SPATIAL;
        }
        self.hc.init(cx, w * h)?; // EncoderInit :673

        self.use_palette = cc.entropy_idx == K_PALETTE;
        self.use_predict = cc.entropy_idx == K_SPATIAL;
        for r in self.refs.iter_mut().take(2) {
            r.clear();
        }

        // AllocateTransformBuffer :1432; w*h covers the packed palette image.
        cx.resize(&mut self.argb, w * h, 0)?;
        if self.use_palette {
            let (sorted, pal) = (self.palette_sorted, &mut self.palette);
            palette_sort_minimize_deltas(&sorted[..self.palette_size], pal);
            self.encode_palette()?;
            self.map_image_from_palette(w, h)?;
        } else {
            // MakeInputImageCopy :1478
            self.argb.copy_from_slice(&self.pic[..w * h]);
            self.current_width = w;
            if self.use_predict {
                cx.resize(&mut self.argb_scratch, 2 * (w + 1), 0)?;
                let n = sub_sample_size(w, self.transform_bits)
                    * sub_sample_size(h, self.transform_bits);
                cx.resize(&mut self.transform_data, n, 0)?;
                self.apply_predict_filter(self.current_width, h)?;
            }
        }

        self.bw.put(0, 1)?; // no more transforms
        let argb = std::mem::take(&mut self.argb);
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        let (cw, hb) = (self.current_width, self.histo_bits);
        self.encode_image_internal(&argb, &mut hc, &mut refs, cw, h, &cc, hb)?;
        self.argb = argb;
        self.hc = hc;
        self.refs = refs;
        Ok(())
    }

    /// EncodeImageInternal, vp8l_enc.c:1097: writes the main image, keeping the smaller of the
    /// sub-configs (strict <). C swaps writers; here the best trial is copied to `bw_best` and
    /// copied back if a later trial loses. The bytes are the same.
    #[allow(clippy::too_many_arguments)]
    fn encode_image_internal(
        &mut self,
        argb: &[u32],
        hc: &mut HashChain,
        refs: &mut [Refs; 4],
        w: usize,
        h: usize,
        cc: &super::CrunchConfig,
        histo_bits: usize,
    ) -> Result<(), Error> {
        let cx = self.cx;
        let quality = self.quality;
        let (hw, hh) = (
            sub_sample_size(w, histo_bits),
            sub_sample_size(h, histo_bits),
        );
        let xy = hw * hh;
        cx.resize(&mut self.histo_symbols, xy, 0)?;
        self.hc_histo.init(cx, xy)?;
        hash_chain_fill(cx, &mut self.rs.hash_first, hc, quality, argb, w, h)?;

        let (init_len, init_acc, init_nacc) = (self.bw.buf.len(), self.bw.acc, self.bw.nacc);
        let mut best_size = usize::MAX;
        let mut best_in_bw = false;
        for i in 0..cc.sub_size {
            get_backward_references(cx, &mut self.rs, w, h, argb, quality, cc.sub[i], hc, refs)?;

            // VP8LBitWriterReset
            self.bw.buf.truncate(init_len);
            self.bw.acc = init_acc;
            self.bw.nacc = init_nacc;

            self.histo_image.alloc(cx, xy)?;
            get_histo_image_symbols(
                cx,
                &mut self.hs,
                w,
                h,
                &refs[0],
                quality,
                histo_bits,
                &mut self.histo_image,
                &mut self.tmp_histo,
                &mut self.histo_symbols,
            )?;
            let mut size = self.histo_image.len();
            get_huff_bit_lengths_and_codes(
                cx,
                &mut self.histo_image,
                &mut self.huff,
                &mut self.buf_rle,
                &mut self.huff_tree,
            )?;

            self.bw.put(0, 1)?; // no color cache

            let write = size > 1;
            if write {
                self.bw.put(1, 1)?;
                cx.resize(&mut self.histo_argb, xy, 0)?;
                let mut max_index = 0usize;
                for k in 0..xy {
                    let s = usize::from(self.histo_symbols[k]);
                    self.histo_argb[k] = (s as u32) << 8;
                    if s >= max_index {
                        max_index = s + 1;
                    }
                }
                size = max_index;
                self.bw.put((histo_bits - 2) as u64, 3)?;
                let histo_argb = std::mem::take(&mut self.histo_argb);
                let mut hc_histo = std::mem::take(&mut self.hc_histo);
                self.encode_image_no_huffman(
                    &histo_argb,
                    &mut hc_histo,
                    &mut refs[2..],
                    hw,
                    hh,
                    quality,
                )?;
                self.histo_argb = histo_argb;
                self.hc_histo = hc_histo;
            } else {
                self.bw.put(0, 1)?;
            }

            let ncodes = 5 * size;
            let mut skip = !0u64; // groups with empty red, blue, alpha
            for k in 0..ncodes {
                let (o, ns) = (self.huff.offs[k], self.huff.ns[k]);
                store_huffman_code(
                    &mut self.bw,
                    &mut self.huff_tree,
                    &mut self.tokens,
                    &self.huff.lens[o..o + ns],
                )?;
                let (l, b) = (
                    &mut self.huff.lens[o..o + ns],
                    &mut self.huff.bits[o..o + ns],
                );
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
                histo_bits,
                &refs[0],
                &self.histo_symbols,
                &self.huff,
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

    /// EncodeImageNoHuffman, vp8l_enc.c:1013: writes a sub-image with one Huffman group. `refs` is
    /// refs_array; it uses refs[0] and refs[1].
    fn encode_image_no_huffman(
        &mut self,
        argb: &[u32],
        hc: &mut HashChain,
        refs: &mut [Refs],
        w: usize,
        h: usize,
        quality: i32,
    ) -> Result<(), Error> {
        let cx = self.cx;
        let symbols = [0u16; 1]; // one tree, one symbol

        hash_chain_fill(cx, &mut self.rs.hash_first, hc, quality, argb, w, h)?;
        get_backward_references(
            cx,
            &mut self.rs,
            w,
            h,
            argb,
            quality,
            LZ77_STANDARD | LZ77_RLE,
            hc,
            refs,
        )?;
        self.histo_nh.alloc(cx, 1)?;
        self.histo_nh.get_mut(0).store_refs(&refs[0]);
        get_huff_bit_lengths_and_codes(
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
        store_image_to_bit_mask(cx, &mut self.bw, w, 0, &refs[0], &symbols, &self.huff_nh, 0)?;
        Ok(())
    }

    /// ApplyPredictFilter, vp8l_enc.c:1308: runs the residual image and writes the predictor
    /// transform.
    fn apply_predict_filter(&mut self, w: usize, h: usize) -> Result<(), Error> {
        let pred_bits = self.transform_bits;
        let tw = sub_sample_size(w, pred_bits);
        let th = sub_sample_size(h, pred_bits);
        residual_image(
            self.cx,
            w,
            h,
            pred_bits,
            &mut self.argb,
            &mut self.argb_scratch,
            &mut self.transform_data,
        )?;
        self.bw.put(1, 1)?; // TRANSFORM_PRESENT
        self.bw.put(0, 2)?; // PREDICTOR_TRANSFORM
        self.bw.put((pred_bits - 2) as u64, 3)?;
        let td = std::mem::take(&mut self.transform_data);
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        let q = self.quality;
        self.encode_image_no_huffman(&td, &mut hc, &mut refs, tw, th, q)?;
        self.transform_data = td;
        self.hc = hc;
        self.refs = refs;
        Ok(())
    }

    /// EncodePalette, vp8l_enc.c:1655: writes the colour indexing transform.
    fn encode_palette(&mut self) -> Result<(), Error> {
        let mut tmp = [0u32; 256];
        let n = self.palette_size;
        self.bw.put(1, 1)?; // TRANSFORM_PRESENT
        self.bw.put(3, 2)?; // COLOR_INDEXING_TRANSFORM
        self.bw.put((n - 1) as u64, 8)?;
        for i in (1..n).rev() {
            tmp[i] = sub_pixels(self.palette[i], self.palette[i - 1]);
        }
        tmp[0] = self.palette[0];
        let mut hc = std::mem::take(&mut self.hc);
        let mut refs = std::mem::take(&mut self.refs);
        self.encode_image_no_huffman(&tmp[..n], &mut hc, &mut refs, n, 1, 20)?;
        self.hc = hc;
        self.refs = refs;
        Ok(())
    }

    /// MapImageFromPalette, vp8l_enc.c:1624 (in_place = 0): packs `pic` into `argb` and sets
    /// `current_width`.
    pub(super) fn map_image_from_palette(&mut self, w: usize, h: usize) -> Result<(), Error> {
        let n = self.palette_size;
        let xbits = if n <= 4 {
            if n <= 2 { 3 } else { 2 }
        } else if n <= 16 {
            1
        } else {
            0
        };
        // AllocateTransformBuffer, argb part only (no predict).
        let cx = self.cx;
        self.current_width = sub_sample_size(w, xbits);
        cx.resize(&mut self.argb, self.current_width * h, 0)?;
        cx.resize(&mut self.pal_row, w, 0)?;
        let cw = self.current_width;
        apply_palette(
            cx,
            &self.pic,
            w,
            &mut self.argb,
            cw,
            &self.palette[..n],
            w,
            h,
            xbits,
            &mut self.pal_row,
        )?;
        Ok(())
    }
}
