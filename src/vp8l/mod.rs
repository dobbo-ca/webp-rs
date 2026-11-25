//! VP8L encoders for ALPH chunks. `legacy` is the simple writer of webp-go's internal/vp8l/vp8l.go;
//! `Lw` ports libwebp's lossless encoder as the ALPH writer of 1.2.2 (src/vp8l/*.rs, the `lw*.go`
//! files of webp-go), whose bytes equal libwebp's. See LICENSE.libwebp.

mod analyze;
mod cost;
mod encode;
mod hist;
mod huff;
pub(crate) mod legacy;
mod lossless;
mod refs;
mod tables;
#[cfg(test)]
mod tests;
mod v16;

use crate::Error;
use crate::limits::{Ledger, cap_bytes};
use hist::{Histo, HistoScratch, HistoSet};
use huff::{Token, Tree};
use refs::{HashChain, Refs, RefsScratch};

const NUM_LITERAL: usize = 256; // NUM_LITERAL_CODES
const NUM_DISTANCE: usize = 40; // NUM_DISTANCE_CODES
const MAX_LENGTH: usize = 4095; // MAX_LENGTH, backward_references_enc.h
const MAX_LENGTH_BITS: u32 = 12; // MAX_LENGTH_BITS
const NUM_PRED_MODES: usize = 14; // kNumPredModes

// EntropyIx, vp8l_enc.c:326.
const K_DIRECT: usize = 0;
const K_SPATIAL: usize = 1;
const K_SUB_GREEN: usize = 2;
const K_SPATIAL_SUB_GREEN: usize = 3;
const K_PALETTE: usize = 4;

// VP8LLZ77Type, backward_references_enc.h:214.
const LZ77_STANDARD: u32 = 1;
const LZ77_RLE: u32 = 2;
const LZ77_BOX: u32 = 4;

// pixOrCopy modes; kCacheIdx is unreachable with use_cache = 0.
const LITERAL: u8 = 0;
const COPY: u8 = 2;

/// kCodeLengthCodeOrder, spec section 3.7.2.1.2.
const CL_ORDER: [u8; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// VP8LPrefixEncode: the prefix code, the number of extra bits and their value, of v >= 1.
pub(crate) fn prefix_bits(v: usize) -> (usize, usize, usize) {
    let v = v - 1;
    if v < 4 {
        return (v, 0, 0);
    }
    let hb = (usize::BITS - v.leading_zeros()) as usize - 1;
    let second = (v >> (hb - 1)) & 1;
    let n = hb - 1;
    (2 * hb + second, n, v & ((1 << n) - 1))
}

/// Packs bits LSB first, as VP8L reads them. With a `cx` its buffer grows through the ledger.
#[derive(Default)]
pub(crate) struct BitWriter<'a> {
    pub buf: Vec<u8>,
    pub acc: u64,
    pub nacc: i32,
    pub cx: Option<&'a Ledger<'a>>,
}

impl BitWriter<'_> {
    pub(crate) fn put(&mut self, v: u64, n: i32) -> Result<(), Error> {
        self.acc |= v << self.nacc;
        self.nacc += n;
        while self.nacc >= 8 {
            match self.cx {
                Some(cx) => cx.push(&mut self.buf, self.acc as u8)?,
                None => self.buf.push(self.acc as u8),
            }
            self.acc >>= 8;
            self.nacc -= 8;
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<u8>, Error> {
        if self.nacc > 0 {
            match self.cx {
                Some(cx) => cx.push(&mut self.buf, self.acc as u8)?,
                None => self.buf.push(self.acc as u8),
            }
        }
        self.acc = 0;
        self.nacc = 0;
        Ok(std::mem::take(&mut self.buf))
    }

    /// VP8LBitWriterNumBytes, bit_writer_utils.h:104.
    fn num_bytes(&self) -> usize {
        self.buf.len() + ((self.nacc as usize + 7) >> 3)
    }
}

/// CrunchConfig; the entropy mode and the LZ77 types of each sub-encode.
pub(super) struct CrunchConfig {
    pub entropy_idx: usize,
    pub sub: [u32; 2],
    pub sub_size: usize,
}

/// The libwebp ALPH encoder. Its scratch is charged through `cx` and kept between calls in an
/// `LwScratch`.
pub(crate) struct Lw<'a> {
    cx: &'a Ledger<'a>,
    /// Writes ALPH as libwebp 1.6.0 does.
    libwebp16: bool,
    method: usize,
    quality: i32,

    // alpha_enc.c
    filt: Vec<u8>,
    trial: Vec<u8>,

    // VP8LEncoder, vp8li_enc.h
    pic: Vec<u32>,
    argb: Vec<u32>,
    argb_scratch: Vec<u32>,
    transform_data: Vec<u32>,
    current_width: usize,
    histo_bits: usize,
    transform_bits: usize,
    use_predict: bool,
    use_palette: bool,
    palette_size: usize,
    palette: [u32; 256],
    palette_sorted: [u32; 256],
    refs: [Refs; 4],
    hc: HashChain,
    bw: BitWriter<'a>,
    bw_best: BitWriter<'a>,
    rs: RefsScratch,

    // histogram_enc.c
    hs: HistoScratch,
    histo_image: HistoSet,
    histo_nh: HistoSet,
    tmp_histo: Histo,
    histo_symbols: Vec<u16>,
    histo_argb: Vec<u32>,
    hc_histo: HashChain,
    ae_histo: Vec<u32>,

    // huffman_encode_utils.c and code storage
    huff: encode::HuffCodes,
    huff_nh: encode::HuffCodes,
    buf_rle: Vec<u8>,
    huff_tree: Vec<Tree>,
    tokens: Vec<Token>,

    pal_row: Vec<u8>,

    /// The 1.6.0 scratch, made on first use.
    v16: Option<Box<v16::V16>>,
}

/// The buffers of an `Lw` encode, kept between calls (the `Lw` takes them at its start and
/// `into_scratch` gives them back). The 1.6.0 path's own scratch is not kept.
pub(crate) struct LwScratch {
    filt: Vec<u8>,
    trial: Vec<u8>,
    pic: Vec<u32>,
    argb: Vec<u32>,
    argb_scratch: Vec<u32>,
    transform_data: Vec<u32>,
    refs: [Refs; 4],
    hc: HashChain,
    bw_best: Vec<u8>,
    rs: RefsScratch,
    hs: HistoScratch,
    histo_image: HistoSet,
    histo_nh: HistoSet,
    tmp_histo: Histo,
    histo_symbols: Vec<u16>,
    histo_argb: Vec<u32>,
    hc_histo: HashChain,
    ae_histo: Vec<u32>,
    huff: encode::HuffCodes,
    huff_nh: encode::HuffCodes,
    buf_rle: Vec<u8>,
    huff_tree: Vec<Tree>,
    tokens: Vec<Token>,
    pal_row: Vec<u8>,
}

impl Default for LwScratch {
    fn default() -> LwScratch {
        LwScratch {
            filt: Vec::new(),
            trial: Vec::new(),
            pic: Vec::new(),
            argb: Vec::new(),
            argb_scratch: Vec::new(),
            transform_data: Vec::new(),
            refs: Default::default(),
            hc: HashChain::default(),
            bw_best: Vec::new(),
            rs: RefsScratch::default(),
            hs: HistoScratch::default(),
            histo_image: HistoSet::new(),
            histo_nh: HistoSet::new(),
            tmp_histo: Histo::new(),
            histo_symbols: Vec::new(),
            histo_argb: Vec::new(),
            hc_histo: HashChain::default(),
            ae_histo: Vec::new(),
            huff: encode::HuffCodes::default(),
            huff_nh: encode::HuffCodes::default(),
            buf_rle: Vec::new(),
            huff_tree: Vec::new(),
            tokens: Vec::new(),
            pal_row: Vec::new(),
        }
    }
}

impl LwScratch {
    /// The bytes the kept buffers hold that the ledger charges as they grow: the ones every
    /// `Ledger` call sizes. The few small ones it does not charge (the Huffman tree, the token
    /// list and the run buffer) are left out here too.
    fn bytes(&self) -> u64 {
        cap_bytes(&self.filt)
            + cap_bytes(&self.trial)
            + cap_bytes(&self.pic)
            + cap_bytes(&self.argb)
            + cap_bytes(&self.argb_scratch)
            + cap_bytes(&self.transform_data)
            + self.refs.iter().map(cap_bytes).sum::<u64>()
            + self.hc.bytes()
            + cap_bytes(&self.bw_best)
            + self.rs.bytes()
            + self.hs.bytes()
            + self.histo_image.bytes()
            + self.histo_nh.bytes()
            + cap_bytes(&self.histo_symbols)
            + cap_bytes(&self.histo_argb)
            + self.hc_histo.bytes()
            + cap_bytes(&self.ae_histo)
            + self.huff.bytes()
            + self.huff_nh.bytes()
            + cap_bytes(&self.pal_row)
    }
}

/// What the ALPH encoders keep between calls: the output buffer, the `Lw` scratch and the
/// legacy writer's code-length list. With `max_threads` 1 an ALPH chunk then allocates nothing
/// once the buffers have grown to the image (the 1.6.0 writer, `alpha_libwebp16`, still makes
/// its own scratch in each call).
#[derive(Default)]
pub(crate) struct AlphaScratch {
    /// The chunk data of the last call, given back by the caller to keep its capacity.
    pub(crate) out: Vec<u8>,
    lw: LwScratch,
    cl_syms: Vec<u16>,
}

impl AlphaScratch {
    /// Whether any buffer is held from an earlier call that a later call charges.
    pub(crate) fn warm(&self) -> bool {
        self.out.capacity() > 0 || self.cl_syms.capacity() > 0 || self.lw.bytes() > 0
    }

    /// Makes the ALPH chunk data of the w by h plane `a`: libwebp's writer if `lw`, else the
    /// legacy one. The kept buffers are charged through `cx` at their capacity before use, like
    /// the ones that grow; the `Lw` scratch stays charged to the call after `cx` drops, as it
    /// stays allocated.
    pub(crate) fn encode(
        &mut self,
        cx: &Ledger<'_>,
        lw: bool,
        libwebp16: bool,
        a: &[u8],
        w: usize,
        h: usize,
        method: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut out = std::mem::take(&mut self.out);
        out.clear();
        if lw {
            cx.take(self.lw.bytes() + cap_bytes(&out))?;
            let mut e = Lw::with_scratch(libwebp16, cx, std::mem::take(&mut self.lw));
            let r = e.alpha(out, a, w, h, method);
            self.lw = e.into_scratch();
            // The scratch is kept after the ledger drops, until the call ends: its charge stays
            // with the call instead of being given back while it is still allocated.
            cx.keep(self.lw.bytes());
            r
        } else {
            cx.take(cap_bytes(&out))?;
            out.push(1);
            let mut e = legacy::Legacy::with_syms(cx, std::mem::take(&mut self.cl_syms));
            let r = e.alpha(out, a, w);
            self.cl_syms = e.into_syms();
            r
        }
    }
}

impl<'a> Lw<'a> {
    /// An encoder with empty buffers.
    #[cfg(test)]
    pub(crate) fn new(libwebp16: bool, cx: &'a Ledger<'a>) -> Lw<'a> {
        Lw::with_scratch(libwebp16, cx, LwScratch::default())
    }

    /// An encoder that works in the buffers of `s`.
    fn with_scratch(libwebp16: bool, cx: &'a Ledger<'a>, s: LwScratch) -> Lw<'a> {
        Lw {
            cx,
            libwebp16,
            method: 0,
            quality: 0,
            filt: s.filt,
            trial: s.trial,
            pic: s.pic,
            argb: s.argb,
            argb_scratch: s.argb_scratch,
            transform_data: s.transform_data,
            current_width: 0,
            histo_bits: 0,
            transform_bits: 0,
            use_predict: false,
            use_palette: false,
            palette_size: 0,
            palette: [0; 256],
            palette_sorted: [0; 256],
            refs: s.refs,
            hc: s.hc,
            bw: BitWriter::default(),
            bw_best: BitWriter {
                buf: s.bw_best,
                ..BitWriter::default()
            },
            rs: s.rs,
            hs: s.hs,
            histo_image: s.histo_image,
            histo_nh: s.histo_nh,
            tmp_histo: s.tmp_histo,
            histo_symbols: s.histo_symbols,
            histo_argb: s.histo_argb,
            hc_histo: s.hc_histo,
            ae_histo: s.ae_histo,
            huff: s.huff,
            huff_nh: s.huff_nh,
            buf_rle: s.buf_rle,
            huff_tree: s.huff_tree,
            tokens: s.tokens,
            pal_row: s.pal_row,
            v16: None,
        }
    }

    /// The buffers, to keep for the next encode.
    fn into_scratch(self) -> LwScratch {
        LwScratch {
            filt: self.filt,
            trial: self.trial,
            pic: self.pic,
            argb: self.argb,
            argb_scratch: self.argb_scratch,
            transform_data: self.transform_data,
            refs: self.refs,
            hc: self.hc,
            bw_best: self.bw_best.buf,
            rs: self.rs,
            hs: self.hs,
            histo_image: self.histo_image,
            histo_nh: self.histo_nh,
            tmp_histo: self.tmp_histo,
            histo_symbols: self.histo_symbols,
            histo_argb: self.histo_argb,
            hc_histo: self.hc_histo,
            ae_histo: self.ae_histo,
            huff: self.huff,
            huff_nh: self.huff_nh,
            buf_rle: self.buf_rle,
            huff_tree: self.huff_tree,
            tokens: self.tokens,
            pal_row: self.pal_row,
        }
    }
}
