//! The frame tag, the compressed frame header (RFC 6386 9.1, 19.2) and the mode writers
//! (libwebp tree_enc.c, BSD-3-Clause).

use super::boolenc::BoolEncoder;
use super::predict::{PRED_DC, PRED_HD, PRED_HE, PRED_LD, PRED_RD, PRED_TM, PRED_VE, PRED_VL};
use super::quant::DQ_UV_DC;
use super::segment::Segs;
use super::tokens::{DEFAULT_PROBS, UPDATE_PROBS};

/// Key frame tag, RFC 6386 9.1.
pub(crate) fn frame_tag(width: usize, height: usize, size0: usize) -> [u8; 10] {
    let v = ((size0 as u32) << 5) | (1 << 4);
    let mut t = [0u8; 10];
    t[..3].copy_from_slice(&v.to_le_bytes()[..3]);
    t[3] = 0x9d;
    t[4] = 0x01;
    t[5] = 0x2a;
    t[6..8].copy_from_slice(&(width as u16).to_le_bytes());
    t[8..10].copy_from_slice(&(height as u16).to_le_bytes());
    t
}

/// VP8PutSignedBits, bit_writer_utils.c:151-158.
fn put_signed(e: &mut BoolEncoder, v: i32, n: i32) {
    if !e.put_bit(v != 0, 128) {
        return;
    }
    let neg = v < 0;
    e.put_literal(v.unsigned_abs(), n);
    e.put_bit(neg, 128);
}

/// Compressed frame header, RFC 6386 19.2. `lw` writes one DCT partition and no skip flags, as
/// libwebp's token loop.
pub(crate) fn put_frame_header(
    e: &mut BoolEncoder,
    s: &Segs,
    skip_prob: u8,
    probs: &[u8; 1056],
    lw: bool,
) {
    e.put_literal(0, 1); // color_space
    e.put_literal(0, 1); // clamping_type

    // PutSegmentHeader, syntax_enc.c:185-212.
    if e.put_bit(s.n > 1, 128) {
        e.put_bit(s.update_map, 128);
        e.put_literal(1, 1); // update_segment_feature_data
        e.put_literal(1, 1); // absolute values
        for &q in &s.quant {
            put_signed(e, q, 7);
        }
        for &l in &s.level {
            put_signed(e, l, 6);
        }
        if s.update_map {
            for &p in &s.probs {
                if e.put_bit(p != 255, 128) {
                    e.put_literal(u32::from(p), 8);
                }
            }
        }
    }

    // Decoders skip filtering when the frame level is 0.
    let level = s.level.iter().copied().max().unwrap_or(0);
    e.put_literal(0, 1); // filter_type
    e.put_literal(level as u32, 6); // loop_filter_level
    e.put_literal(0, 3); // sharpness_level

    e.put_literal(0, 1); // loop_filter_adj_enable

    if lw {
        e.put_literal(0, 2); // log2_nbr_of_dct_partitions
    } else {
        e.put_literal(1, 2);
    }

    e.put_literal(s.quant[0] as u32, 7); // y_ac_qi
    put_signed(e, 0, 4); // y_dc
    put_signed(e, 0, 4); // y2_dc
    put_signed(e, 0, 4); // y2_ac
    put_signed(e, DQ_UV_DC, 4);
    put_signed(e, s.dq_uv_ac, 4);

    e.put_literal(0, 1); // refresh_entropy_probs

    // VP8WriteProbas (tree_enc.c:417-432): update iff changed.
    for i in 0..1056 {
        if e.put_bit(probs[i] != DEFAULT_PROBS[i], UPDATE_PROBS[i]) {
            e.put_literal(u32::from(probs[i]), 8);
        }
    }

    if lw {
        e.put_literal(0, 1); // mb_no_skip_coeff
        return;
    }
    e.put_literal(1, 1); // mb_no_skip_coeff
    e.put_literal(u32::from(skip_prob), 8); // prob_skip_false
}

/// Mode writers mirror libwebp tree_enc.c Put*Mode. `put` writes a bit with a probability and
/// returns the bit.
pub(crate) fn put_i16_mode(put: &mut dyn FnMut(bool, u8) -> bool, m: u8) {
    if put(m == PRED_TM || m == PRED_HE, 156) {
        put(m == PRED_TM, 128);
    } else {
        put(m == PRED_VE, 163);
    }
}

pub(crate) fn put_uv_mode(put: &mut dyn FnMut(bool, u8) -> bool, m: u8) {
    if put(m != PRED_DC, 142) && put(m != PRED_VE, 114) {
        put(m != PRED_HE, 183);
    }
}

pub(crate) fn put_i4_mode(put: &mut dyn FnMut(bool, u8) -> bool, m: u8, p: &[u8; 9]) {
    if !put(m != PRED_DC, p[0]) || !put(m != PRED_TM, p[1]) || !put(m != PRED_VE, p[2]) {
        return;
    }
    if !put(m >= PRED_LD, p[3]) {
        if put(m != PRED_HE, p[4]) {
            put(m != PRED_RD, p[5]);
        }
    } else if put(m != PRED_LD, p[6]) && put(m != PRED_VL, p[7]) {
        put(m != PRED_HD, p[8]);
    }
}
