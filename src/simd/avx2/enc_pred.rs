//! AVX2 form of the encoder kernels of package B, the intra predictors and SSE.

use super::mem::*;
use crate::simd::enc_block::Rows;
use crate::simd::enc_pred::{DC, HD, HE, HU, Kernels, LD, RD, TM, VE, VL, VR, row};
use crate::tables::ZIGZAG;
use crate::vp8enc::quant::QuantMatrix;
use std::arch::x86_64::*;

type Z = [[u8; 32]; 26];

pub(in crate::simd) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels {
        sse4x10,
        edge_sse,
        sse_block,
        residual_quant3,
        pred4_all,
    })
}

/// Dispatch: the kernels are safe `#[target_feature(enable = "avx2")]` fns, so calling them from
/// here needs `unsafe`. Only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn sse4x10(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { sse4x10_kernel(pr, src) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn edge_sse(src: &[u8], stride: usize, n: usize, z: &Z, y: usize, x: usize, dc: i32) -> [i32; 4] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { edge_sse_kernel(src, stride, n, z, y, x, dc) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn sse_block(src: &[u8], stride: usize, z: &Z, y: usize, x: usize, w: usize, h: usize) -> i32 {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { sse_block_kernel(src, stride, z, y, x, w, h) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn residual_quant3(
    m: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    modes: [usize; 3],
    src: &[u32; 4],
    blk: &mut [[i16; 16]; 3],
    lv: &mut [[i16; 16]; 3],
) -> [(i32, i64); 3] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { residual_quant3_kernel(m, pr, modes, src, blk, lv) }
}

/// Dispatch, as `sse4x10`.
#[allow(unsafe_code)]
fn pred4_all(e: &[u8; 16], pr: &mut [[u32; 4]; 10]) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { pred4_all_kernel(e, pr) }
}

/// Two modes a register: the SSE of the source block against each prediction.
#[target_feature(enable = "avx2")]
fn sse4x10_kernel(pr: &[[u32; 4]; 10], src: &[u32; 4]) -> [i32; 10] {
    let s = _mm256_broadcastsi128_si256(load128_u32(src));
    let zero = _mm256_setzero_si256();
    let mut v = [zero; 5];
    for (v, p) in v.iter_mut().zip(pr.as_chunks::<2>().0) {
        let m = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(load128_u32(&p[0])),
            load128_u32(&p[1]),
        );
        let d = _mm256_or_si256(_mm256_subs_epu8(s, m), _mm256_subs_epu8(m, s));
        let lo = _mm256_unpacklo_epi8(d, zero);
        let hi = _mm256_unpackhi_epi8(d, zero);
        *v = _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
    }
    // Lane 0..3 of `a` are modes 0, 2, 4, 6 and lanes 4..7 modes 1, 3, 5, 7; `b` has 8 and 9.
    let a = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
    let t = _mm256_hadd_epi32(v[4], v[4]);
    let b = _mm256_hadd_epi32(t, t);
    let (mut a8, mut b8) = ([0i32; 8], [0i32; 8]);
    store256_i32(&mut a8, a);
    store256_i32(&mut b8, b);
    [
        a8[0], a8[4], a8[1], a8[5], a8[2], a8[6], a8[3], a8[7], b8[0], b8[4],
    ]
}

/// Adds the squared differences of the DC, TM, VE and HE predictions against `s` to `acc`. Each
/// register holds the bytes of one 16-pixel row, or two 8-pixel rows.
#[target_feature(enable = "avx2")]
#[inline]
fn edge_step(acc: &mut [__m256i; 4], s: __m128i, t: __m128i, l: __m128i, a: __m128i, dc: __m128i) {
    // clip8(l - a + t): add the positive part of l - a, subtract the negative one, both saturating.
    let tm = _mm_subs_epu8(_mm_adds_epu8(t, _mm_subs_epu8(l, a)), _mm_subs_epu8(a, l));
    for (acc, p) in acc.iter_mut().zip([dc, tm, t, l]) {
        let d = _mm256_cvtepu8_epi16(_mm_or_si128(_mm_subs_epu8(s, p), _mm_subs_epu8(p, s)));
        *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(d, d));
    }
}

#[target_feature(enable = "avx2")]
fn edge_sse_kernel(
    src: &[u8],
    stride: usize,
    n: usize,
    z: &Z,
    y: usize,
    x: usize,
    dc: i32,
) -> [i32; 4] {
    let mut acc = [_mm256_setzero_si256(); 4];
    let a = _mm_set1_epi8(z[y - 1][x - 1] as i8);
    let dc = _mm_set1_epi8(dc as i8);
    if n == 16 {
        let t = load128(row(&z[y - 1][x..]));
        for j in 0..16 {
            let s = load128(row(&src[j * stride..]));
            let l = _mm_set1_epi8(z[y + j][x - 1] as i8);
            edge_step(&mut acc, s, t, l, a, dc);
        }
    } else {
        let t8 = loadl64(row(&z[y - 1][x..]));
        let t = _mm_unpacklo_epi64(t8, t8);
        for j in (0..8).step_by(2) {
            let s0 = loadl64(row(&src[j * stride..]));
            let s1 = loadl64(row(&src[(j + 1) * stride..]));
            let s = _mm_unpacklo_epi64(s0, s1);
            let l0 = _mm_set1_epi8(z[y + j][x - 1] as i8);
            let l1 = _mm_set1_epi8(z[y + j + 1][x - 1] as i8);
            edge_step(&mut acc, s, t, _mm_unpacklo_epi64(l0, l1), a, dc);
        }
    }
    let mut d = [0i32; 4];
    for (d, acc) in d.iter_mut().zip(acc) {
        let mut v = [0i32; 8];
        store256_i32(&mut v, acc);
        *d = v.iter().sum();
    }
    d
}

#[target_feature(enable = "avx2")]
fn sse_block_kernel(
    src: &[u8],
    stride: usize,
    z: &Z,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
) -> i32 {
    let (mut acc, mut acc8) = (_mm256_setzero_si256(), _mm_setzero_si128());
    for j in 0..h {
        let (r, p) = (&src[j * stride..], &z[y + j][x..]);
        if w == 16 {
            let (s, p) = (load128(row(r)), load128(row(p)));
            let d = _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
        } else {
            let (s, p) = (loadl64(row(r)), loadl64(row(p)));
            let d = _mm_sub_epi16(_mm_cvtepu8_epi16(s), _mm_cvtepu8_epi16(p));
            acc8 = _mm_add_epi32(acc8, _mm_madd_epi16(d, d));
        }
    }
    let (mut v, mut v8) = ([0i32; 8], [0i32; 4]);
    store256_i32(&mut v, acc);
    store128_i32(&mut v8, acc8);
    v.iter().chain(&v8).sum()
}

// The fdct and quantizer of `avx2::enc_block`, copied so that they inline into the batch kernel
// below: the same 32-bit steps as the scalar reference, so bit-exact for every input.

/// Transposes four vectors of four i32 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(a, b);
    let t1 = _mm_unpacklo_epi32(c, d);
    let t2 = _mm_unpackhi_epi32(a, b);
    let t3 = _mm_unpackhi_epi32(c, d);
    [
        _mm_unpacklo_epi64(t0, t1),
        _mm_unpackhi_epi64(t0, t1),
        _mm_unpacklo_epi64(t2, t3),
        _mm_unpackhi_epi64(t2, t3),
    ]
}

/// The fdct of `src` minus `pred`. Lanes are the block's rows in the row pass and its columns in
/// the column pass, with a transpose between them.
#[target_feature(enable = "avx2")]
fn fdct(src: &Rows, pred: &Rows, out: &mut [i16; 16]) {
    let (s, p) = (load128_u32(src), load128_u32(pred));
    let d0 = _mm_sub_epi32(_mm_cvtepu8_epi32(s), _mm_cvtepu8_epi32(p));
    let d1 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<4>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<4>(p)),
    );
    let d2 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<8>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<8>(p)),
    );
    let d3 = _mm_sub_epi32(
        _mm_cvtepu8_epi32(_mm_srli_si128::<12>(s)),
        _mm_cvtepu8_epi32(_mm_srli_si128::<12>(p)),
    );
    let (k2217, k5352) = (_mm_set1_epi32(2217), _mm_set1_epi32(5352));
    // Row pass: e_i holds column i of every row.
    let [e0, e1, e2, e3] = transpose(d0, d1, d2, d3);
    let a0 = _mm_add_epi32(e0, e3);
    let a1 = _mm_add_epi32(e1, e2);
    let a2 = _mm_sub_epi32(e1, e2);
    let a3 = _mm_sub_epi32(e0, e3);
    let t0 = _mm_slli_epi32::<3>(_mm_add_epi32(a0, a1));
    let t1 = _mm_srai_epi32::<9>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a2, k2217), _mm_mullo_epi32(a3, k5352)),
        _mm_set1_epi32(1812),
    ));
    let t2 = _mm_slli_epi32::<3>(_mm_sub_epi32(a0, a1));
    let t3 = _mm_srai_epi32::<9>(_mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(a3, k2217), _mm_mullo_epi32(a2, k5352)),
        _mm_set1_epi32(937),
    ));
    // Column pass: y_r holds row r's four transformed values.
    let [y0, y1, y2, y3] = transpose(t0, t1, t2, t3);
    let a0 = _mm_add_epi32(y0, y3);
    let a1 = _mm_add_epi32(y1, y2);
    let a2 = _mm_sub_epi32(y1, y2);
    let a3 = _mm_sub_epi32(y0, y3);
    let o0 = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_add_epi32(a0, a1), _mm_set1_epi32(7)));
    let v = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a2, k2217), _mm_mullo_epi32(a3, k5352)),
        _mm_set1_epi32(12000),
    ));
    // Plus one where a3 != 0 (the compare is -1 where a3 == 0).
    let o1 = _mm_add_epi32(
        _mm_add_epi32(v, _mm_set1_epi32(1)),
        _mm_cmpeq_epi32(a3, _mm_setzero_si128()),
    );
    let o2 = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_sub_epi32(a0, a1), _mm_set1_epi32(7)));
    let o3 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(a3, k2217), _mm_mullo_epi32(a2, k5352)),
        _mm_set1_epi32(51000),
    ));
    // The low 16 bits of each lane, like `as i16`.
    let m = _mm_set1_epi32(0xFFFF);
    let lo = _mm_packus_epi32(_mm_and_si128(o0, m), _mm_and_si128(o1, m));
    let hi = _mm_packus_epi32(_mm_and_si128(o2, m), _mm_and_si128(o3, m));
    if let [a, b] = out.as_chunks_mut::<8>().0 {
        store128_i16(a, lo);
        store128_i16(b, hi);
    }
}

/// One half of the quantizer: levels and dequantized values of eight raster coefficients.
#[target_feature(enable = "avx2")]
#[inline]
fn quant8(
    v: __m256i,
    sharp: __m256i,
    iq: __m256i,
    bias: __m256i,
    q: __m256i,
) -> (__m256i, __m256i) {
    let s = _mm256_srai_epi32::<31>(v);
    let a = _mm256_add_epi32(_mm256_abs_epi32(v), sharp);
    let l = _mm256_srli_epi32::<17>(_mm256_add_epi32(_mm256_mullo_epi32(a, iq), bias));
    let l = _mm256_sub_epi32(_mm256_xor_si256(l, s), s);
    (l, _mm256_mullo_epi32(l, q))
}

/// The low 16 bits of 16 i32 lanes (like `as i16`) as 16 i16 lanes, in order.
#[target_feature(enable = "avx2")]
#[inline]
fn pack16(lo: __m256i, hi: __m256i) -> __m256i {
    let m = _mm256_set1_epi32(0xFFFF);
    let p = _mm256_packus_epi32(_mm256_and_si256(lo, m), _mm256_and_si256(hi, m));
    _mm256_permute4x64_epi64::<0xD8>(p)
}

/// Lane 0 from `dc`, the rest from `ac`.
#[target_feature(enable = "avx2")]
#[inline]
fn dc_ac(dc: u32, ac: u32) -> __m256i {
    _mm256_blend_epi32::<1>(_mm256_set1_epi32(ac as i32), _mm256_set1_epi32(dc as i32))
}

/// `quantize_sse` (with the SSE when `D`, else 0).
#[target_feature(enable = "avx2")]
fn quant<const D: bool>(m: &QuantMatrix, in_: &mut [i16; 16], out: &mut [i16; 16]) -> (i32, i64) {
    let x = load256_i16(in_);
    let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(x));
    let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(x));
    let [sh0, sh1] = m.sharpen.as_chunks::<8>().0 else {
        return (0, 0);
    };
    let (iq1, bias1, q1) = (
        _mm256_set1_epi32(m.iq[1] as i32),
        _mm256_set1_epi32(m.bias[1] as i32),
        _mm256_set1_epi32(m.q[1] as i32),
    );
    let (l0, d0) = quant8(
        lo,
        load256_i32(sh0),
        dc_ac(m.iq[0], m.iq[1]),
        dc_ac(m.bias[0], m.bias[1]),
        dc_ac(m.q[0], m.q[1]),
    );
    let (l1, d1) = quant8(hi, load256_i32(sh1), iq1, bias1, q1);
    store256_i16(in_, pack16(d0, d1));
    let mut r = [0i16; 16];
    store256_i16(&mut r, pack16(l0, l1));
    let mut last = -1;
    for n in 0..16 {
        let l = r[(ZIGZAG[n] & 15) as usize];
        out[n] = l;
        if l != 0 {
            last = n as i32;
        }
    }
    if !D {
        return (last, 0);
    }
    // Each square is below 2^32: |v - dq| is below 2^16 (a level is within one step of |v|).
    let (e0, e1) = (_mm256_sub_epi32(lo, d0), _mm256_sub_epi32(hi, d1));
    let (mut s0, mut s1) = ([0i32; 8], [0i32; 8]);
    store256_i32(&mut s0, _mm256_mullo_epi32(e0, e0));
    store256_i32(&mut s1, _mm256_mullo_epi32(e1, e1));
    let d = s0.iter().chain(&s1).map(|&v| i64::from(v as u32)).sum();
    (last, d)
}

/// `residual_quant3` without indirect calls and without the prediction leaving registers: three
/// fdct plus quantizer runs on the source against the predictions of `modes`.
#[target_feature(enable = "avx2")]
fn residual_quant3_kernel(
    m: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    modes: [usize; 3],
    src: &[u32; 4],
    blk: &mut [[i16; 16]; 3],
    lv: &mut [[i16; 16]; 3],
) -> [(i32, i64); 3] {
    let mut out = [(0, 0); 3];
    for (((o, b), l), &md) in out.iter_mut().zip(blk).zip(lv).zip(&modes) {
        fdct(src, &pr[md], b);
        *o = quant::<true>(m, b, l);
    }
    out
}

/// A byte of an output row: the source (0 average of two, 1 average of three, 2 the edge vector)
/// and the byte index in it.
type Sel = (u8, u8);
const fn h2(i: u8) -> Sel {
    (0, i)
}
const fn h3(i: u8) -> Sel {
    (1, i)
}
const fn hv(i: u8) -> Sel {
    (2, i)
}

/// The PSHUFB mask taking the bytes of `rows` that come from source `which`, zero elsewhere.
const fn tab(rows: [[Sel; 4]; 4], which: u8) -> [u8; 16] {
    let mut o = [0x80u8; 16];
    let mut j = 0;
    while j < 4 {
        let mut i = 0;
        while i < 4 {
            if rows[j][i].0 == which {
                o[4 * j + i] = rows[j][i].1;
            }
            i += 1;
        }
        j += 1;
    }
    o
}

// With V = L3 L3 L2 L1 L0 A T0..T7 T7 (16 bytes), R2[i] = avg2(V[i], V[i+1]) and R3[i] =
// avg3(V[i], V[i+1], V[i+2]). Each row of a mode below is four bytes of those.
const RD_ROWS: [[Sel; 4]; 4] = [
    [h3(4), h3(5), h3(6), h3(7)],
    [h3(3), h3(4), h3(5), h3(6)],
    [h3(2), h3(3), h3(4), h3(5)],
    [h3(1), h3(2), h3(3), h3(4)],
];
const VR_ROWS: [[Sel; 4]; 4] = [
    [h2(5), h2(6), h2(7), h2(8)],
    [h3(4), h3(5), h3(6), h3(7)],
    [h3(3), h2(5), h2(6), h2(7)],
    [h3(2), h3(4), h3(5), h3(6)],
];
const LD_ROWS: [[Sel; 4]; 4] = [
    [h3(6), h3(7), h3(8), h3(9)],
    [h3(7), h3(8), h3(9), h3(10)],
    [h3(8), h3(9), h3(10), h3(11)],
    [h3(9), h3(10), h3(11), h3(12)],
];
const VL_ROWS: [[Sel; 4]; 4] = [
    [h2(6), h2(7), h2(8), h2(9)],
    [h3(6), h3(7), h3(8), h3(9)],
    [h2(7), h2(8), h2(9), h3(10)],
    [h3(7), h3(8), h3(9), h3(11)],
];
const HD_ROWS: [[Sel; 4]; 4] = [
    [h2(4), h3(4), h3(5), h3(6)],
    [h2(3), h3(3), h2(4), h3(4)],
    [h2(2), h3(2), h2(3), h3(3)],
    [h2(1), h3(1), h2(2), h3(2)],
];
const HU_ROWS: [[Sel; 4]; 4] = [
    [h2(3), h3(2), h2(2), h3(1)],
    [h2(2), h3(1), h2(1), h3(0)],
    [h2(1), h3(0), hv(0), hv(0)],
    [hv(0), hv(0), hv(0), hv(0)],
];
static RD_T: [u8; 16] = tab(RD_ROWS, 1);
static VR_T: [[u8; 16]; 2] = [tab(VR_ROWS, 0), tab(VR_ROWS, 1)];
static LD_T: [u8; 16] = tab(LD_ROWS, 1);
static VL_T: [[u8; 16]; 2] = [tab(VL_ROWS, 0), tab(VL_ROWS, 1)];
static HD_T: [[u8; 16]; 2] = [tab(HD_ROWS, 0), tab(HD_ROWS, 1)];
static HU_T: [[u8; 16]; 3] = [tab(HU_ROWS, 0), tab(HU_ROWS, 1), tab(HU_ROWS, 2)];
static VE_T: [u8; 16] = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8];
static HE_T: [u8; 16] = [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0];
/// The edge vector V from the edge bytes.
static V_T: [u8; 16] = [12, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8];
/// The four pixels above, four times, and each left pixel four times.
static TM_T: [u8; 16] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
static TM_L: [u8; 16] = [9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12];
/// The four above and four left pixels, to sum for DC.
static DC_T: [u8; 16] = [
    1, 2, 3, 4, 9, 10, 11, 12, 128, 128, 128, 128, 128, 128, 128, 128,
];

/// Byte i is avg3(v[i], v[i+1], v[i+2]) exactly: the rounded average of v[i] and v[i+2] minus
/// the carry of their sum, averaged with v[i+1]. The last two bytes are not meaningful.
#[target_feature(enable = "avx2")]
#[inline]
fn avg3_bytes(v: __m128i) -> __m128i {
    let (n1, n2) = (_mm_srli_si128::<1>(v), _mm_srli_si128::<2>(v));
    let odd = _mm_and_si128(_mm_xor_si128(v, n2), _mm_set1_epi8(1));
    _mm_avg_epu8(_mm_subs_epu8(_mm_avg_epu8(v, n2), odd), n1)
}

/// Go's `pred4AllAsm` in SSSE3 form.
#[target_feature(enable = "avx2")]
fn pred4_all_kernel(e: &[u8; 16], pr: &mut [[u32; 4]; 10]) {
    let ev = load128(e);
    let v = _mm_shuffle_epi8(ev, load128(&V_T));
    let r2 = _mm_avg_epu8(v, _mm_srli_si128::<1>(v));
    let r3 = avg3_bytes(v);

    let sum = _mm_sad_epu8(_mm_shuffle_epi8(ev, load128(&DC_T)), _mm_setzero_si128());
    let dc = ((_mm_cvtsi128_si32(sum) + 4) >> 3) as u32 * 0x0101_0101;
    store128_u32(&mut pr[DC], _mm_set1_epi32(dc as i32));

    // clip8(l - a + t) as in `edge_step`.
    let a = _mm_shuffle_epi8(ev, _mm_setzero_si128());
    let t = _mm_shuffle_epi8(ev, load128(&TM_T));
    let l = _mm_shuffle_epi8(ev, load128(&TM_L));
    let tm = _mm_subs_epu8(_mm_adds_epu8(t, _mm_subs_epu8(l, a)), _mm_subs_epu8(a, l));
    store128_u32(&mut pr[TM], tm);

    store128_u32(&mut pr[VE], _mm_shuffle_epi8(r3, load128(&VE_T)));
    store128_u32(&mut pr[HE], _mm_shuffle_epi8(r3, load128(&HE_T)));
    store128_u32(&mut pr[RD], _mm_shuffle_epi8(r3, load128(&RD_T)));
    store128_u32(&mut pr[LD], _mm_shuffle_epi8(r3, load128(&LD_T)));
    for (mode, t) in [(VR, &VR_T), (VL, &VL_T), (HD, &HD_T)] {
        let (x, y) = (
            _mm_shuffle_epi8(r2, load128(&t[0])),
            _mm_shuffle_epi8(r3, load128(&t[1])),
        );
        store128_u32(&mut pr[mode], _mm_or_si128(x, y));
    }
    let hu = _mm_or_si128(
        _mm_or_si128(
            _mm_shuffle_epi8(r2, load128(&HU_T[0])),
            _mm_shuffle_epi8(r3, load128(&HU_T[1])),
        ),
        _mm_shuffle_epi8(v, load128(&HU_T[2])),
    );
    store128_u32(&mut pr[HU], hu);
}
