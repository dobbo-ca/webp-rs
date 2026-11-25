//! The libwebp 1.6.0 ALPH encoder: fixed-point entropy, colour cache selection, predictor
//! subsampling. The LZ77, Huffman and bit writer parts are the 1.2.2 port (`super`).

mod cost;
mod enc;
mod hist;
mod log;
mod tables;

pub(super) use enc::V16;
