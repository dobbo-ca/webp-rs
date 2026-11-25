//! AVX2 form of the i4 mode trials, in the 16-bit layout of Go's `i4TrialsAVX2Asm` and
//! `recMetrics16Asm`. Same arithmetic as the scalar reference in `simd::enc_i4`, which the tests
//! compare against bit for bit. `tt_mb` still runs one block per 32-bit lane.

use super::mem::{
    load128, load128_i16, load128_i32, load128_u32, load256, load256_i16, load256_i32, store128,
    store128_u32, store256_i16, store256_i32,
};
use crate::simd::enc_i4::{Kernels, WEIGHT_Y, i4_trial1, rec_metrics};
use crate::vp8enc::quant::QuantMatrix;
use std::arch::x86_64::*;

type V = __m256i;

pub(in crate::simd) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels {
        i4_trials,
        rec_metrics16,
        tt_mb,
    })
}

/// Dispatch, as `luma_row` in the parent: only reachable through `detect`, which checked AVX2.
#[allow(unsafe_code)]
fn i4_trials(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { i4_trials_kernel(q, pr, src, lv, rec, res) }
}

/// Dispatch, as `i4_trials`.
#[allow(unsafe_code)]
fn rec_metrics16(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { rec_metrics16_kernel(c, z, src, st) }
}

/// Dispatch, as `i4_trials`.
#[allow(unsafe_code)]
fn tt_mb(src: &[u8; 256]) -> [i32; 16] {
    // SAFETY: `detect` returned this fn only after `is_x86_feature_detected!("avx2")`.
    unsafe { tt_mb_kernel(src) }
}

#[target_feature(enable = "avx2")]
#[inline]
fn k(x: i32) -> V {
    _mm256_set1_epi32(x)
}

#[target_feature(enable = "avx2")]
#[inline]
fn add(a: V, b: V) -> V {
    _mm256_add_epi32(a, b)
}

#[target_feature(enable = "avx2")]
#[inline]
fn sub(a: V, b: V) -> V {
    _mm256_sub_epi32(a, b)
}

/// `a * x` per lane, low 32 bits.
#[target_feature(enable = "avx2")]
#[inline]
fn mulk(a: V, x: i32) -> V {
    _mm256_mullo_epi32(a, k(x))
}
/// The 16 byte values of four row words per lane.
#[target_feature(enable = "avx2")]
#[inline]
fn unpack(w: &[V; 4]) -> [V; 16] {
    let m = k(255);
    let mut o = [k(0); 16];
    for j in 0..4 {
        o[4 * j] = _mm256_and_si256(w[j], m);
        o[4 * j + 1] = _mm256_and_si256(_mm256_srli_epi32::<8>(w[j]), m);
        o[4 * j + 2] = _mm256_and_si256(_mm256_srli_epi32::<16>(w[j]), m);
        o[4 * j + 3] = _mm256_srli_epi32::<24>(w[j]);
    }
    o
}

/// TTransform of 16 pixel lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn tt(px: &[V; 16]) -> V {
    let mut t = [k(0); 16];
    for i in 0..4 {
        let (r0, r1, r2, r3) = (px[4 * i], px[4 * i + 1], px[4 * i + 2], px[4 * i + 3]);
        let (a0, a1, a2, a3) = (add(r0, r2), add(r1, r3), sub(r1, r3), sub(r0, r2));
        t[4 * i] = add(a0, a1);
        t[4 * i + 1] = add(a3, a2);
        t[4 * i + 2] = sub(a3, a2);
        t[4 * i + 3] = sub(a0, a1);
    }
    let mut sum = k(0);
    for i in 0..4 {
        let (a0, a1) = (add(t[i], t[8 + i]), add(t[4 + i], t[12 + i]));
        let (a2, a3) = (sub(t[4 + i], t[12 + i]), sub(t[i], t[8 + i]));
        sum = add(sum, mulk(_mm256_abs_epi32(add(a0, a1)), WEIGHT_Y[i]));
        sum = add(sum, mulk(_mm256_abs_epi32(add(a3, a2)), WEIGHT_Y[4 + i]));
        sum = add(sum, mulk(_mm256_abs_epi32(sub(a3, a2)), WEIGHT_Y[8 + i]));
        sum = add(sum, mulk(_mm256_abs_epi32(sub(a0, a1)), WEIGHT_Y[12 + i]));
    }
    sum
}

// The 16-bit kernels below are Go's i4TrialsAVX2Asm and recMetrics16Asm (gen/i4sse.py, gen/i4avx2.py,
// gen/recm.py): every step is one SSE instruction on a 128-bit lane, and a lane works on its own mode
// or block. A pair of modes (m, n) shares a register: the words of m and n alternate, so one
// instruction advances both.

const ZZ: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// The raster coefficients held by the eight words of each quantizer vector.
const POS: [[usize; 8]; 2] = [[0, 2, 1, 3, 8, 10, 9, 11], [4, 6, 5, 7, 12, 14, 13, 15]];

const fn twice(a: [u8; 16]) -> [u8; 32] {
    let mut o = [0u8; 32];
    let mut i = 0;
    while i < 16 {
        o[i] = a[i];
        o[i + 16] = a[i];
        i += 1;
    }
    o
}

/// The rows of a block in the order 0, 3, 1, 2, as the pairs (x0, x3) and (x1, x2) of each row.
const fn psh() -> [u8; 32] {
    const RO: [u8; 4] = [0, 3, 1, 2];
    let mut o = [0u8; 16];
    let mut s = 0;
    while s < 4 {
        let b = 4 * RO[s];
        o[4 * s] = b;
        o[4 * s + 1] = b + 3;
        o[4 * s + 2] = b + 1;
        o[4 * s + 3] = b + 2;
        s += 1;
    }
    twice(o)
}

/// 4x4 byte transpose.
const fn tr() -> [u8; 32] {
    let mut o = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        o[i] = (4 * (i % 4) + i / 4) as u8;
        i += 1;
    }
    twice(o)
}

const fn eo() -> [u8; 32] {
    const W: [u8; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
    let mut o = [0u8; 16];
    let mut i = 0;
    while i < 8 {
        o[2 * i] = 2 * W[i];
        o[2 * i + 1] = 2 * W[i] + 1;
        i += 1;
    }
    twice(o)
}

/// Raster words of each half of a coefficient block to quantizer lane order: c0 c2 c1 c3 c4 c6 c5 c7.
const fn evn() -> [u8; 32] {
    const W: [u8; 8] = [0, 2, 1, 3, 4, 6, 5, 7];
    let mut o = [0u8; 16];
    let mut i = 0;
    while i < 8 {
        o[2 * i] = 2 * W[i];
        o[2 * i + 1] = 2 * W[i] + 1;
        i += 1;
    }
    twice(o)
}

/// Bytes of the zigzag levels `half * 8..half * 8 + 8` that quantizer vector `reg` holds.
const fn zzt(half: usize, reg: usize) -> [u8; 32] {
    let mut o = [0x80u8; 16];
    let mut i = 0;
    while i < 8 {
        let j = ZZ[8 * half + i];
        let mut k = 0;
        while k < 8 {
            if POS[reg][k] == j {
                o[2 * i] = (2 * k) as u8;
                o[2 * i + 1] = (2 * k) as u8 + 1;
            }
            k += 1;
        }
        i += 1;
    }
    twice(o)
}

static PSH: [u8; 32] = psh();
static TR: [u8; 32] = tr();
static EO: [u8; 32] = eo();
static EVN: [u8; 32] = evn();
static Z00: [u8; 32] = zzt(0, 0);
static Z01: [u8; 32] = zzt(0, 1);
static Z10: [u8; 32] = zzt(1, 0);
static Z11: [u8; 32] = zzt(1, 1);

const fn pk(a: i16, b: i16) -> i32 {
    (((b as u16 as u32) << 16) | (a as u16 as u32)) as i32
}

/// The word pair (a, b) in every dword.
#[target_feature(enable = "avx2")]
#[inline]
fn w2(a: i16, b: i16) -> V {
    _mm256_set1_epi32(pk(a, b))
}

#[target_feature(enable = "avx2")]
#[inline]
fn add16(a: V, b: V) -> V {
    _mm256_add_epi16(a, b)
}

#[target_feature(enable = "avx2")]
#[inline]
fn sub16(a: V, b: V) -> V {
    _mm256_sub_epi16(a, b)
}

#[target_feature(enable = "avx2")]
#[inline]
fn abs16(a: V) -> V {
    _mm256_abs_epi16(a)
}

#[target_feature(enable = "avx2")]
#[inline]
fn shuf(a: V, t: &[u8; 32]) -> V {
    _mm256_shuffle_epi8(a, load256(t))
}

/// SHUFPS on integer vectors.
#[target_feature(enable = "avx2")]
#[inline]
fn shufps<const I: i32>(a: V, b: V) -> V {
    _mm256_castps_si256(_mm256_shuffle_ps::<I>(
        _mm256_castsi256_ps(a),
        _mm256_castsi256_ps(b),
    ))
}

/// The quantizer constants: word 0 of the first vector is the DC, every other word the AC.
struct Qc {
    sh: [V; 2],
    iq: [V; 2],
    cb: [V; 2],
    bh: [V; 2],
    q: [V; 2],
}

#[target_feature(enable = "avx2")]
#[inline]
fn dcac(dc: u32, ac: u32) -> [V; 2] {
    let w = _mm256_set1_epi16(ac as i16);
    [_mm256_blend_epi16::<1>(w, _mm256_set1_epi16(dc as i16)), w]
}

#[target_feature(enable = "avx2")]
#[inline]
fn qconsts(q: &QuantMatrix) -> Qc {
    let s = |i: usize| q.sharpen[i] as i16;
    let a = [s(0), s(2), s(1), s(3), s(8), s(10), s(9), s(11)];
    let b = [s(4), s(6), s(5), s(7), s(12), s(14), s(13), s(15)];
    Qc {
        sh: [
            _mm256_broadcastsi128_si256(load128_i16(&a)),
            _mm256_broadcastsi128_si256(load128_i16(&b)),
        ],
        iq: dcac(q.iq[0], q.iq[1]),
        // 65535 - (bias & 0xffff) and (bias >> 16) + 1: the carry of the product plus the bias.
        cb: dcac(!q.bias[0], !q.bias[1]),
        bh: dcac((q.bias[0] >> 16) + 1, (q.bias[1] >> 16) + 1),
        q: dcac(q.q[0], q.q[1]),
    }
}

/// The source block in the forms the trial needs, the same in both lanes.
struct Sc {
    ss: V,
    sd: V,
    /// Column c as words: the four rows twice, once for each mode of the pair.
    sc: [V; 4],
}

#[target_feature(enable = "avx2")]
#[inline]
fn src_consts(src: &[u32; 4]) -> Sc {
    let x = _mm256_broadcastsi128_si256(load128_u32(src));
    let zero = _mm256_setzero_si256();
    let b = shuf(x, &PSH);
    let c = shuf(x, &TR);
    Sc {
        ss: _mm256_maddubs_epi16(b, _mm256_set1_epi8(1)),
        sd: _mm256_maddubs_epi16(b, _mm256_set1_epi16(0xff01u16 as i16)),
        sc: [
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32::<0x00>(c), zero),
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32::<0x55>(c), zero),
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32::<0xaa>(c), zero),
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32::<0xff>(c), zero),
        ],
    }
}

/// One quantizer vector: levels and dequantized coefficients of the words of `w`.
#[target_feature(enable = "avx2")]
#[inline]
fn quant1(w: V, q: &Qc, i: usize) -> (V, V) {
    let a = add16(abs16(w), q.sh[i]);
    let lo = _mm256_mullo_epi16(a, q.iq[i]);
    let hi = _mm256_mulhi_epu16(a, q.iq[i]);
    // The carry out of the low 16 bits of a * iq + bias: no carry where lo <= 65535 - bias_lo.
    let nc = _mm256_cmpeq_epi16(_mm256_max_epu16(lo, q.cb[i]), q.cb[i]);
    let a = add16(add16(hi, nc), q.bh[i]);
    let a = _mm256_srli_epi16::<1>(a);
    let a = _mm256_sign_epi16(a, w);
    (a, _mm256_mullo_epi16(a, q.q[i]))
}

/// FDCT and quantizer of the (source - prediction) block of each lane. Returns the dequantized
/// coefficients (W0, W1) and the zigzag levels 0..8 and 8..16, all in words.
#[target_feature(enable = "avx2")]
#[inline]
fn fdct_quant(pred: V, s: &Sc, q: &Qc) -> (V, V, V, V) {
    let zero = _mm256_setzero_si256();
    let p = shuf(pred, &PSH);
    let qv = _mm256_maddubs_epi16(p, _mm256_set1_epi16(0xff01u16 as i16));
    let p = _mm256_maddubs_epi16(p, _mm256_set1_epi8(1));
    let p = sub16(p, s.ss);
    let qv = sub16(qv, s.sd);
    let e2 = _mm256_madd_epi16(p, w2(-8, 8));
    let p = _mm256_madd_epi16(p, _mm256_set1_epi16(-8));
    let e3 = _mm256_madd_epi16(qv, w2(-2217, 5352));
    let qv = _mm256_madd_epi16(qv, w2(-5352, -2217));
    let qv = _mm256_srai_epi32::<9>(add(qv, k(1812)));
    let e3 = _mm256_srai_epi32::<9>(add(e3, k(937)));
    let p = _mm256_packs_epi32(p, e2);
    let qv = _mm256_packs_epi32(qv, e3);
    // Second pass.
    let ones = _mm256_set1_epi16(1);
    let pm = w2(1, -1);
    let f = _mm256_madd_epi16(p, pm);
    let p = _mm256_madd_epi16(p, ones);
    let g = _mm256_madd_epi16(qv, pm);
    let qv = _mm256_madd_epi16(qv, ones);
    let p = _mm256_packs_epi32(p, qv);
    let f = _mm256_packs_epi32(f, g);
    let h = _mm256_madd_epi16(p, pm);
    let p = _mm256_madd_epi16(p, ones);
    let p = _mm256_packs_epi32(p, h);
    let p = _mm256_srai_epi16::<4>(add16(p, _mm256_set1_epi16(7)));
    // The +1 of the third row where a3 != 0 is in the 65536 of the constant; undo it where the
    // low word of the pair is zero.
    let z = _mm256_cmpeq_epi32(_mm256_slli_epi32::<16>(f), zero);
    let j = _mm256_madd_epi16(f, w2(2217, -5352));
    let f = _mm256_madd_epi16(f, w2(5352, 2217));
    let f = _mm256_srai_epi32::<16>(add(f, k(12000 + 65536)));
    let j = _mm256_srai_epi32::<16>(add(j, k(51000)));
    let f = _mm256_packs_epi32(add(f, z), j);
    // p = W0, f = W1.
    let (a0, d0) = quant1(p, q, 0);
    let (a1, d1) = quant1(f, q, 1);
    let z0 = _mm256_or_si256(shuf(a0, &Z00), shuf(a1, &Z01));
    let z1 = _mm256_or_si256(shuf(a0, &Z10), shuf(a1, &Z11));
    (d0, d1, z0, z1)
}

/// The index of the highest set bit, -1 if none.
fn last(m: u32) -> i32 {
    31 - m.leading_zeros() as i32
}

/// A `fdct_quant` of the predictions of modes m (lane 0) and m + 1 (lane 1): stores their levels,
/// last index and nonzero AC count. Returns the dequantized (W0, W1).
#[target_feature(enable = "avx2")]
#[inline]
fn stream(
    pred: V,
    m: usize,
    s: &Sc,
    q: &Qc,
    lv: &mut [[i16; 16]; 10],
    res: &mut [[i32; 4]; 10],
) -> (V, V) {
    let (d0, d1, z0, z1) = fdct_quant(pred, s, q);
    store256_i16(&mut lv[m], _mm256_permute2x128_si256::<0x20>(z0, z1));
    store256_i16(&mut lv[m + 1], _mm256_permute2x128_si256::<0x31>(z0, z1));
    let nzm = _mm256_cmpeq_epi8(_mm256_packs_epi16(z0, z1), _mm256_setzero_si256());
    // Bit n of each half is set where level n is nonzero.
    let mk = !(_mm256_movemask_epi8(nzm) as u32);
    for (i, h) in [mk & 0xffff, mk >> 16].into_iter().enumerate() {
        res[m + i][0] = last(h);
        res[m + i][3] = (h & 0xfffe).count_ones() as i32;
    }
    (d0, d1)
}

/// The 16-bit inverse DCT pass on rows r0..r3: a + d, b + c, b - c, a - d.
#[target_feature(enable = "avx2")]
#[inline]
fn ipass(r0: V, r1: V, r2: V, r3: V) -> [V; 4] {
    let km1 = _mm256_set1_epi16(20091);
    let km2 = _mm256_set1_epi16(-30068); // 35468 - 65536.
    let a = add16(r0, r2);
    let b = sub16(r0, r2);
    // x * 35468 >> 16 is x * (35468 - 65536) >> 16 plus x, and the same for 20091.
    let c = add16(
        sub16(_mm256_mulhi_epi16(r1, km2), _mm256_mulhi_epi16(r3, km1)),
        sub16(r1, r3),
    );
    let d = add16(
        add16(_mm256_mulhi_epi16(r1, km1), _mm256_mulhi_epi16(r3, km2)),
        add16(r1, r3),
    );
    [add16(a, d), add16(b, c), sub16(b, c), sub16(a, d)]
}

/// The TTransform weights of row `v` as word pairs: u0 u1 u0 u1 u2 u3 u2 u3 in each lane.
#[target_feature(enable = "avx2")]
#[inline]
fn wv(v: usize) -> V {
    let w = |u: usize| WEIGHT_Y[4 * v + u] as i16;
    let (a, b) = (pk(w(0), w(1)), pk(w(2), w(3)));
    _mm256_setr_epi32(a, a, b, b, a, a, b, b)
}

/// The reconstruction of a pair of modes (or blocks) in each lane: dequantized coefficients
/// d{k}{o} (W vector k of the pair member o) added to the predictions pm and pn (raster rows, one
/// 16-byte block per lane), cut to bytes. Returns the two reconstructions in raster order and, per
/// lane, the dwords [SSE m, TTransform m, SSE n, TTransform n]. None if a coefficient is
/// 2048 or more in size, where the 16-bit IDCT is no longer exact.
#[target_feature(enable = "avx2")]
#[inline]
fn recon_pair(d00: V, d10: V, d01: V, d11: V, pm: V, pn: V, sc: &[V; 4]) -> Option<(V, V, V)> {
    let zero = _mm256_setzero_si256();
    let rc = _mm256_or_si256(
        _mm256_or_si256(abs16(d00), abs16(d10)),
        _mm256_or_si256(abs16(d01), abs16(d11)),
    );
    if _mm256_testz_si256(rc, _mm256_set1_epi16(0xf800u16 as i16)) == 0 {
        return None;
    }
    // Rows with the two members in alternate words.
    let r0 = _mm256_unpacklo_epi16(d00, d01);
    let r2 = _mm256_unpackhi_epi16(d00, d01);
    let r1 = _mm256_unpacklo_epi16(d10, d11);
    let r3 = _mm256_unpackhi_epi16(d10, d11);
    let [m0, m1, m2, m3] = ipass(r0, r1, r2, r3);
    // Transpose to [m column | n column] per column.
    let u2 = _mm256_unpackhi_epi16(m0, m1);
    let m0 = _mm256_unpacklo_epi16(m0, m1);
    let u3 = _mm256_unpackhi_epi16(m2, m3);
    let m2 = _mm256_unpacklo_epi16(m2, m3);
    let t0 = _mm256_unpacklo_epi32(m0, m2);
    let m0 = _mm256_unpackhi_epi32(m0, m2);
    let t1 = _mm256_unpacklo_epi32(u2, u3);
    let u2 = _mm256_unpackhi_epi32(u2, u3);
    let t0 = add16(t0, _mm256_set1_epi16(4));
    let [o0, o1, o2, o3] = ipass(t0, t1, m0, u2);
    let o0 = _mm256_srai_epi16::<3>(o0);
    let o1 = _mm256_srai_epi16::<3>(o1);
    let o2 = _mm256_srai_epi16::<3>(o2);
    let o3 = _mm256_srai_epi16::<3>(o3);
    // Prediction as [m column c | n column c] words.
    let pm = shuf(pm, &TR);
    let pn = shuf(pn, &TR);
    let x = _mm256_unpacklo_epi32(pm, pn);
    let pm = _mm256_unpackhi_epi32(pm, pn);
    let o0 = _mm256_adds_epi16(o0, _mm256_unpacklo_epi8(x, zero));
    let o1 = _mm256_adds_epi16(o1, _mm256_unpackhi_epi8(x, zero));
    let o2 = _mm256_adds_epi16(o2, _mm256_unpacklo_epi8(pm, zero));
    let o3 = _mm256_adds_epi16(o3, _mm256_unpackhi_epi8(pm, zero));
    let o0 = _mm256_packus_epi16(o0, o1);
    let o2 = _mm256_packus_epi16(o2, o3);
    // Back to raster per member.
    let zm = shuf(shufps::<0x88>(o0, o2), &TR);
    let zn = shuf(shufps::<0xdd>(o0, o2), &TR);
    let c0 = _mm256_unpacklo_epi8(o0, zero);
    let c1 = _mm256_unpackhi_epi8(o0, zero);
    let c2 = _mm256_unpacklo_epi8(o2, zero);
    let c3 = _mm256_unpackhi_epi8(o2, zero);
    // SSE against the source columns.
    let g0 = _mm256_sub_epi16(c0, sc[0]);
    let g1 = _mm256_sub_epi16(c1, sc[1]);
    let g2 = _mm256_sub_epi16(c2, sc[2]);
    let g3 = _mm256_sub_epi16(c3, sc[3]);
    let g = add(
        add(_mm256_madd_epi16(g0, g0), _mm256_madd_epi16(g1, g1)),
        add(_mm256_madd_epi16(g2, g2), _mm256_madd_epi16(g3, g3)),
    );
    // TTransform: across columns, then within the rows of each half.
    let d02 = sub16(c0, c2);
    let c0 = add16(c0, c2);
    let d13 = sub16(c1, c3);
    let c1 = add16(c1, c3);
    let e3 = sub16(c0, c1);
    let e0 = add16(c0, c1);
    let e2 = sub16(d02, d13);
    let e1 = add16(d02, d13);
    let s01 = shufps::<0x88>(e0, e1);
    let x01 = shufps::<0xdd>(e0, e1);
    let s23 = shufps::<0x88>(e2, e3);
    let x23 = shufps::<0xdd>(e2, e3);
    let (d01, s01) = (shuf(sub16(s01, x01), &EO), shuf(add16(s01, x01), &EO));
    let (d23, s23) = (shuf(sub16(s23, x23), &EO), shuf(add16(s23, x23), &EO));
    let pa = _mm256_unpacklo_epi64(s01, s23);
    let s01 = _mm256_unpackhi_epi64(s01, s23);
    let f3 = sub16(pa, s01);
    let pa = add16(pa, s01);
    let pb = _mm256_unpacklo_epi64(d01, d23);
    let d01 = _mm256_unpackhi_epi64(d01, d23);
    let f2 = sub16(pb, d01);
    let pb = add16(pb, d01);
    let pa = _mm256_madd_epi16(abs16(pa), wv(0));
    let pb = _mm256_madd_epi16(abs16(pb), wv(1));
    let f2 = _mm256_madd_epi16(abs16(f2), wv(2));
    let f3 = _mm256_madd_epi16(abs16(f3), wv(3));
    let tt = add(add(pa, pb), add(f2, f3));
    // g = SSE [m m n n], tt = [m n m n].
    let g = add(g, _mm256_shuffle_epi32::<0xb1>(g));
    let tt = add(tt, _mm256_shuffle_epi32::<0x4e>(tt));
    let tt = _mm256_shuffle_epi32::<0x50>(tt);
    Some((zm, zn, _mm256_blend_epi16::<0xcc>(g, tt)))
}

/// Two predictions, `a` in lane 0 and `b` in lane 1.
#[target_feature(enable = "avx2")]
#[inline]
fn lanes(a: &[u32; 4], b: &[u32; 4]) -> V {
    _mm256_set_m128i(load128_u32(b), load128_u32(a))
}

#[target_feature(enable = "avx2")]
fn i4_trials_kernel(
    q: &QuantMatrix,
    pr: &[[u32; 4]; 10],
    src: &[u32; 4],
    lv: &mut [[i16; 16]; 10],
    rec: &mut [[u32; 4]; 10],
    res: &mut [[i32; 4]; 10],
) {
    let qc = qconsts(q);
    let s = src_consts(src);
    // Modes 4j to 4j + 3: two streams, one pair of modes in each lane of the reconstruction.
    for j in 0..2 {
        let m = 4 * j;
        let (x0, x1) = stream(lanes(&pr[m], &pr[m + 1]), m, &s, &qc, lv, res);
        let (y0, y1) = stream(lanes(&pr[m + 2], &pr[m + 3]), m + 2, &s, &qc, lv, res);
        let r = recon_pair(
            _mm256_permute2x128_si256::<0x20>(x0, y0),
            _mm256_permute2x128_si256::<0x20>(x1, y1),
            _mm256_permute2x128_si256::<0x31>(x0, y0),
            _mm256_permute2x128_si256::<0x31>(x1, y1),
            lanes(&pr[m], &pr[m + 2]),
            lanes(&pr[m + 1], &pr[m + 3]),
            &s.sc,
        );
        let Some((zm, zn, g)) = r else {
            for i in m..m + 4 {
                i4_trial1(q, &pr[i], src, &mut lv[i], &mut rec[i], &mut res[i]);
            }
            continue;
        };
        store128_u32(&mut rec[m], _mm256_castsi256_si128(zm));
        store128_u32(&mut rec[m + 1], _mm256_castsi256_si128(zn));
        store128_u32(&mut rec[m + 2], _mm256_extracti128_si256::<1>(zm));
        store128_u32(&mut rec[m + 3], _mm256_extracti128_si256::<1>(zn));
        let mut a = [0i32; 8];
        store256_i32(&mut a, g);
        for i in 0..4 {
            res[m + i][1] = a[2 * i];
            res[m + i][2] = a[2 * i + 1];
        }
    }
    // Modes 8 and 9 in lane 0 (and again in lane 1, which is ignored).
    let (x0, x1) = stream(lanes(&pr[8], &pr[9]), 8, &s, &qc, lv, res);
    let r = recon_pair(
        _mm256_permute2x128_si256::<0x00>(x0, x0),
        _mm256_permute2x128_si256::<0x00>(x1, x1),
        _mm256_permute2x128_si256::<0x11>(x0, x0),
        _mm256_permute2x128_si256::<0x11>(x1, x1),
        lanes(&pr[8], &pr[8]),
        lanes(&pr[9], &pr[9]),
        &s.sc,
    );
    let Some((zm, zn, g)) = r else {
        for i in 8..10 {
            i4_trial1(q, &pr[i], src, &mut lv[i], &mut rec[i], &mut res[i]);
        }
        return;
    };
    store128_u32(&mut rec[8], _mm256_castsi256_si128(zm));
    store128_u32(&mut rec[9], _mm256_castsi256_si128(zn));
    let mut a = [0i32; 8];
    store256_i32(&mut a, g);
    for i in 0..2 {
        res[8 + i][1] = a[2 * i];
        res[8 + i][2] = a[2 * i + 1];
    }
}

/// Rows of four blocks, [a_j b_j c_j d_j] dwords, to one block per vector, rows 0..4 as dwords.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose(r: [__m128i; 4]) -> [__m128i; 4] {
    let a = _mm_unpacklo_epi32(r[0], r[1]);
    let c = _mm_unpackhi_epi32(r[0], r[1]);
    let b = _mm_unpacklo_epi32(r[2], r[3]);
    let d = _mm_unpackhi_epi32(r[2], r[3]);
    [
        _mm_unpacklo_epi64(a, b),
        _mm_unpackhi_epi64(a, b),
        _mm_unpacklo_epi64(c, d),
        _mm_unpackhi_epi64(c, d),
    ]
}

/// Load the 16 bytes at columns 8..24 of row `y` of the prediction buffer.
#[target_feature(enable = "avx2")]
#[inline]
fn zload(z: &[[u8; 32]; 26], y: usize) -> __m128i {
    load128(z[y][8..].first_chunk().unwrap_or(&[0; 16]))
}

/// Store the 16 bytes at columns 8..24 of row `y` of the prediction buffer.
#[target_feature(enable = "avx2")]
#[inline]
fn zstore(z: &mut [[u8; 32]; 26], y: usize, v: __m128i) {
    if let Some(d) = z[y][8..].first_chunk_mut() {
        store128(d, v);
    }
}

#[target_feature(enable = "avx2")]
fn rec_metrics16_kernel(
    c: &[[i16; 16]; 16],
    z: &mut [[u8; 32]; 26],
    src: &[u8; 256],
    st: &[i32; 16],
) -> (i32, i32) {
    let zero = _mm256_setzero_si256();
    let tr = _mm256_castsi256_si128(load256(&TR));
    // SSE in the even dwords, the TTransform distance in the odd ones.
    let mut acc = zero;
    let (mut sse, mut sd) = (0, 0);
    // One row of four blocks per pass: lane 0 holds blocks a and b, lane 1 blocks c and d.
    for r in 0..4 {
        let y = 1 + 4 * r;
        let mut cv = [zero; 4];
        for k in 0..4 {
            cv[k] = shuf(load256_i16(&c[4 * r + k]), &EVN);
        }
        let lo = _mm256_permute2x128_si256::<0x20>(cv[0], cv[2]);
        let hi = _mm256_permute2x128_si256::<0x31>(cv[0], cv[2]);
        let (d00, d10) = (_mm256_unpacklo_epi64(lo, hi), _mm256_unpackhi_epi64(lo, hi));
        let lo = _mm256_permute2x128_si256::<0x20>(cv[1], cv[3]);
        let hi = _mm256_permute2x128_si256::<0x31>(cv[1], cv[3]);
        let (d01, d11) = (_mm256_unpacklo_epi64(lo, hi), _mm256_unpackhi_epi64(lo, hi));
        let mut sr = [_mm_setzero_si128(); 4];
        for j in 0..4 {
            sr[j] = load128(&src.as_chunks::<16>().0[4 * r + j]);
        }
        let mut sq = transpose(sr);
        for v in sq.iter_mut() {
            *v = _mm_shuffle_epi8(*v, tr);
        }
        let (ya, yb) = (
            _mm256_set_m128i(sq[2], sq[0]),
            _mm256_set_m128i(sq[3], sq[1]),
        );
        let (l, h) = (_mm256_unpacklo_epi32(ya, yb), _mm256_unpackhi_epi32(ya, yb));
        let scv = [
            _mm256_unpacklo_epi8(l, zero),
            _mm256_unpackhi_epi8(l, zero),
            _mm256_unpacklo_epi8(h, zero),
            _mm256_unpackhi_epi8(h, zero),
        ];
        let mut zr = [_mm_setzero_si128(); 4];
        for j in 0..4 {
            zr[j] = zload(z, y + j);
        }
        let [pa, pb, pc, pd] = transpose(zr);
        let out = recon_pair(
            d00,
            d10,
            d01,
            d11,
            _mm256_set_m128i(pc, pa),
            _mm256_set_m128i(pd, pb),
            &scv,
        );
        let Some((zm, zn, g)) = out else {
            for kk in 0..4 {
                let n = 4 * r + kk;
                let x = 8 + 4 * kk;
                let mut pw = [0u32; 4];
                let mut sw = [0u32; 4];
                for j in 0..4 {
                    pw[j] = u32::from_le_bytes(z[y + j].as_chunks::<4>().0[x / 4]);
                    sw[j] =
                        u32::from_le_bytes(src.as_chunks::<4>().0[(y - 1 + j) * 4 + (x - 8) / 4]);
                }
                let (rc, k_sse, t) = rec_metrics(&c[n], &pw, &sw);
                for (j, w) in rc.iter().enumerate() {
                    z[y + j][x..x + 4].copy_from_slice(&w.to_le_bytes());
                }
                sse += k_sse;
                sd += (t - st[n]).abs() >> 5;
            }
            continue;
        };
        let r01 = _mm256_permute4x64_epi64::<0xd8>(_mm256_unpacklo_epi32(zm, zn));
        let r23 = _mm256_permute4x64_epi64::<0xd8>(_mm256_unpackhi_epi32(zm, zn));
        zstore(z, y, _mm256_castsi256_si128(r01));
        zstore(z, y + 1, _mm256_extracti128_si256::<1>(r01));
        zstore(z, y + 2, _mm256_castsi256_si128(r23));
        zstore(z, y + 3, _mm256_extracti128_si256::<1>(r23));
        // [SSE a, TT a, SSE b, TT b | c, d]: keep |TT - st| >> 5 in the odd dwords.
        let stv = _mm256_slli_epi64::<32>(_mm256_cvtepu32_epi64(load128_i32(
            &st.as_chunks::<4>().0[r],
        )));
        let dt = _mm256_srli_epi32::<5>(_mm256_abs_epi32(sub(g, stv)));
        acc = add(acc, _mm256_blend_epi32::<0xaa>(g, dt));
    }
    let mut a = [0i32; 8];
    store256_i32(&mut a, acc);
    for i in 0..4 {
        sse += a[2 * i];
        sd += a[2 * i + 1];
    }
    (sse, sd)
}

#[target_feature(enable = "avx2")]
fn tt_mb_kernel(src: &[u8; 256]) -> [i32; 16] {
    let mut st = [0i32; 16];
    for g in 0..2 {
        let mut sw = [[0i32; 8]; 4];
        for l in 0..8 {
            let n = 8 * g + l;
            for j in 0..4 {
                sw[j][l] =
                    i32::from_le_bytes(src.as_chunks::<4>().0[(4 * (n / 4) + j) * 4 + n % 4]);
            }
        }
        let sb = unpack(&[
            load256_i32(&sw[0]),
            load256_i32(&sw[1]),
            load256_i32(&sw[2]),
            load256_i32(&sw[3]),
        ]);
        let mut a = [0i32; 8];
        store256_i32(&mut a, tt(&sb));
        st[8 * g..8 * g + 8].copy_from_slice(&a);
    }
    st
}
