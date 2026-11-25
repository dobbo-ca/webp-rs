//! Quantizer setup and quantization, from libwebp quant_enc.c (BSD-3-Clause).

use crate::fmath::pow;
use crate::simd::enc_block;
use crate::tables::{AC_TABLE, DC_TABLE};

/// libwebp kFreqSharpening, quant_enc.c:178.
const FREQ_SHARPEN: [u32; 16] = [
    0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90,
];

/// libwebp VP8Matrix, indexed [dc, ac].
#[derive(Clone, Copy, Default)]
pub(crate) struct QuantMatrix {
    pub q: [u32; 2],
    pub iq: [u32; 2],
    pub bias: [u32; 2],
    pub sharpen: [i32; 16],
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Quant {
    pub y1: QuantMatrix,
    pub y2: QuantMatrix,
    pub uv: QuantMatrix,
}

/// One quantized level: (|v| + sharpen) * iq + bias >> 17, with v's sign.
pub(crate) fn level_of(v: i32, sharpen: u32, iq: u32, bias: u32) -> i32 {
    let s = v >> 31;
    let a = ((v ^ s) - s) as u32 + sharpen;
    ((a.wrapping_mul(iq).wrapping_add(bias) >> 17) as i32 ^ s) - s
}

impl QuantMatrix {
    /// libwebp quant_enc.c ExpandMatrix.
    fn expand(&mut self, bias_dc: u8, bias_ac: u8, sharpen_ac: bool) {
        let b = [u32::from(bias_dc), u32::from(bias_ac)];
        for k in 0..2 {
            self.iq[k] = (1 << 17) / self.q[k];
            self.bias[k] = b[k] << 9;
        }
        if sharpen_ac {
            for j in 0..16 {
                self.sharpen[j] = ((FREQ_SHARPEN[j] * self.q[j.min(1)]) >> 11) as i32;
            }
        }
    }

    /// QuantizeBlock_C minus zthresh and MAX_LEVEL: `in_` raster in, dequantized in place, zigzag
    /// levels out. Returns the last nonzero zigzag index, or -1.
    pub(crate) fn quantize(&self, in_: &mut [i16; 16], out: &mut [i16; 16]) -> i32 {
        self.quantize_sse(in_, out).0
    }

    /// `quantize` that also returns the coefficient SSE.
    pub(crate) fn quantize_sse(&self, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
        enc_block::quantize_sse(self, in_, out)
    }

    /// `quantize` over n blocks; bit n of the result is set when block n has a nonzero level.
    pub(crate) fn quantize_n(&self, in_: &mut [[i16; 16]], out: &mut [[i16; 16]]) -> u32 {
        enc_block::quantize_n(self, in_, out)
    }
}

/// Fixed by libwebp, quant_enc.c:408-418.
pub(crate) const DQ_UV_DC: i32 = -2;

/// Y1 and Y2 deltas are 0.
pub(crate) fn new_quant(qi: usize, dq_uv_ac: i32) -> Quant {
    let mut q = Quant::default();
    q.y1.q = [u32::from(DC_TABLE[qi]), u32::from(AC_TABLE[qi])];
    q.y1.expand(96, 110, true);

    let y2_ac = (u32::from(AC_TABLE[qi]) * 155 / 100).max(8);
    q.y2.q = [u32::from(DC_TABLE[qi]) * 2, y2_ac];
    q.y2.expand(96, 108, false);

    let uvdc = (qi as i32 + DQ_UV_DC).clamp(0, 117) as usize;
    let uvac = (qi as i32 + dq_uv_ac).clamp(0, 127) as usize;
    q.uv.q = [u32::from(DC_TABLE[uvdc]), u32::from(AC_TABLE[uvac])];
    q.uv.expand(110, 115, false);
    q
}

/// QualityToCompression, quant_enc.c:302-313.
pub(crate) fn quality_compression(q: f32) -> f64 {
    let c = f64::from(q) / 100.0;
    let lin = if c < 0.75 {
        c * (2.0 / 3.0)
    } else {
        2.0 * c - 1.0
    };
    pow(lin, 1.0 / 3.0)
}

/// libwebp SetupFilterStrength, strength 60, sharpness 0 (quant_enc.c:272-290).
pub(crate) fn filter_level(qi: usize, beta: i32) -> i32 {
    let s = i32::from(AC_TABLE[qi] >> 2).min(63);
    let f = s * 300 / (256 + beta);
    if f < 2 { 0 } else { f.min(63) }
}
