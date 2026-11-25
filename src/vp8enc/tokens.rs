//! Coefficient tokens: recording, probability finalisation and coding, ported from libwebp
//! PutCoeffs, CodeResiduals (frame_enc.c) and FinalizeTokenProbas, BSD-3-Clause.
//!
//! A token is one boolean coding decision held until the probabilities are final: bit 15 is the
//! value, bit 14 marks a fixed probability in bits 0-7, otherwise bits 0-10 index the flat
//! probability table.

use super::boolenc::BoolEncoder;
use super::rdo::{costs, last_nz};
use crate::tables::{BANDS, CAT_PROBS, COEFF_PROBS, COEFF_UPDATE_PROBS};
use std::sync::LazyLock;

pub(crate) const FIXED_TOK: u16 = 1 << 14;

/// Tokens one block can record: per coefficient a zero flag, then record_level's worst case
/// (v>1, v>4, v>10, k>=2, k&1, 11 cat6 bits = 16), then a sign and EOB (19), plus a leading EOB.
pub(crate) const MAX_BLOCK_TOK: usize = 16 * 19 + 1;
/// Tokens one macroblock can record: 25 blocks.
pub(crate) const MAX_MB_TOK: usize = 25 * MAX_BLOCK_TOK;

/// Nonzero flags: Y 0..3, U 4..5, V 6..7, Y2 8.
pub(crate) type NzCtx = [u8; 9];

fn flatten(p: &[[[[u8; 11]; 3]; 8]; 4]) -> [u8; 1056] {
    let mut f = [0u8; 1056];
    let mut n = 0;
    for a in p {
        for b in a {
            for c in b {
                f[n..n + 11].copy_from_slice(c);
                n += 11;
            }
        }
    }
    f
}

/// The default and the update probabilities, flat.
pub(crate) static DEFAULT_PROBS: LazyLock<[u8; 1056]> = LazyLock::new(|| flatten(&COEFF_PROBS));
pub(crate) static UPDATE_PROBS: LazyLock<[u8; 1056]> =
    LazyLock::new(|| flatten(&COEFF_UPDATE_PROBS));

fn tok(bit: bool, v: usize) -> u16 {
    v as u16 | (u16::from(bit) << 15)
}

/// The recorded tokens: `buf` is zero filled to its length, `n` of them are used. Recording
/// writes by index into the spare room, so a block needs no capacity check per token.
#[derive(Default)]
pub(crate) struct TokBuf {
    buf: Vec<u16>,
    n: usize,
}

impl TokBuf {
    pub(crate) fn clear(&mut self) {
        self.n = 0;
    }

    /// The tokens the buffer holds room for, used or not.
    pub(crate) fn held(&self) -> usize {
        self.buf.len()
    }

    /// The tokens the allocation holds room for, which is at least `held`.
    pub(crate) fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// The tokens that still fit without growing.
    pub(crate) fn room(&self) -> usize {
        self.buf.len() - self.n
    }

    /// Adds `extra` tokens of room, or fails when the allocator does.
    pub(crate) fn grow(&mut self, extra: usize) -> Result<(), ()> {
        self.buf.try_reserve_exact(extra).map_err(|_| ())?;
        self.buf.resize(self.buf.len() + extra, 0);
        Ok(())
    }
}

impl std::ops::Deref for TokBuf {
    type Target = [u16];

    fn deref(&self) -> &[u16] {
        &self.buf[..self.n]
    }
}

/// Codes |v| >= 1 from the v>1 node.
pub(crate) fn record_level(c: &mut Vec<u16>, v: usize, p: usize) {
    let gt1 = v > 1;
    c.push(tok(gt1, p + 2));
    if gt1 {
        let i = c.len();
        c.resize(i + 16, 0); // the tail is at most 15 tokens
        let e = record_level_tail(c, i, v, p);
        c.truncate(e);
    }
}

/// Codes |v| > 1 from the gt4 node, writing at `c[i..]`, and returns the end.
fn record_level_tail(c: &mut [u16], mut i: usize, v: usize, p: usize) -> usize {
    let mut put = |bit: bool, t: usize| {
        c[i] = (u16::from(bit) << 15) | t as u16;
        i += 1;
    };
    let gt4 = v > 4;
    put(gt4, p + 3);
    if !gt4 {
        let ne2 = v != 2;
        put(ne2, p + 4);
        if ne2 {
            put(v == 4, p + 5);
        }
        return i;
    }
    let gt10 = v > 10;
    put(gt10, p + 6);
    if !gt10 {
        let gt6 = v > 6;
        put(gt6, p + 7);
        if !gt6 {
            put(v == 6, FIXED_TOK as usize | 159);
        } else {
            put(v >= 9, FIXED_TOK as usize | 165);
            put(v & 1 == 0, FIXED_TOK as usize | 145);
        }
        return i;
    }
    let k = match v {
        ..19 => 0,
        19..35 => 1,
        35..67 => 2,
        _ => 3,
    };
    put(k >= 2, p + 8);
    put(k & 1 == 1, p + 9 + (k >> 1));
    let v = v - (3 + (8 << k));
    let probs = CAT_PROBS[k];
    let nb = probs.len();
    for (j, &pb) in probs.iter().enumerate() {
        put(
            (v >> (nb - 1 - j)) & 1 == 1,
            FIXED_TOK as usize | usize::from(pb),
        );
    }
    i
}

/// PutCoeffs: records block `lv` (zigzag levels) of type `typ` from position `first` at context
/// `ctx`, and returns whether it has a nonzero level. Zero and one levels are written by index
/// without branches: the nonzero flag and the v>1 slot are set for every position, the v>1 slot
/// speculatively for a zero, and the write position only advances past what is real. The caller
/// keeps `MAX_BLOCK_TOK` tokens of room.
fn record_coeffs(t: &mut TokBuf, typ: usize, ctx: usize, first: usize, lv: &[i16; 16]) -> u8 {
    let last = last_nz(lv, first);
    let base = typ * 264;
    let mut p = base + (BANDS[first] as usize * 3 + ctx) * 11;
    let c = &mut t.buf[t.n..];
    c[0] = tok(last >= 0, p);
    if last < 0 {
        t.n += 1;
        return 0;
    }
    let last = last as usize;
    let mut i = 1;
    for n in first..=last {
        let v = i32::from(lv[n]);
        let s = v >> 31;
        let a = ((v ^ s) - s) as usize;
        let nz = a.min(1);
        c[i] = ((nz as u16) << 15) | (p + 1) as u16;
        c[i + 1] = (((a.min(2) >> 1) as u16) << 15) | (p + 2) as u16;
        i += 1 + nz;
        if a > 1 {
            i = record_level_tail(c, i, a, p);
        }
        let pn = base + BANDS[n + 1] as usize * 33 + 11 * a.min(2);
        c[i] = (((s & 1) as u16) << 15) | FIXED_TOK | 128;
        c[i + 1] = (u16::from(n < last) << 15) | pn as u16;
        i += nz * (1 + usize::from(n < 15));
        p = pn;
    }
    t.n += i;
    1
}

/// CodeResiduals for the luma blocks of one MB.
pub(crate) fn record_luma(
    c: &mut TokBuf,
    i16: bool,
    lv: &[[i16; 16]; 25],
    l: &mut NzCtx,
    t: &mut NzCtx,
) {
    let (mut typ, mut first) = (3, 0);
    if i16 {
        let nz = record_coeffs(c, 1, usize::from(l[8] + t[8]), 0, &lv[24]);
        l[8] = nz;
        t[8] = nz;
        typ = 0;
        first = 1;
    }
    for y in 0..4 {
        for x in 0..4 {
            let nz = record_coeffs(c, typ, usize::from(l[y] + t[x]), first, &lv[4 * y + x]);
            l[y] = nz;
            t[x] = nz;
        }
    }
}

pub(crate) fn record_chroma(c: &mut TokBuf, lv: &[[i16; 16]; 25], l: &mut NzCtx, t: &mut NzCtx) {
    for ch in (0..4).step_by(2) {
        for y in 0..2 {
            for x in 0..2 {
                let nz = record_coeffs(
                    c,
                    2,
                    usize::from(l[4 + ch + y] + t[4 + ch + x]),
                    0,
                    &lv[16 + 2 * ch + 2 * y + x],
                );
                l[4 + ch + y] = nz;
                t[4 + ch + x] = nz;
            }
        }
    }
}

/// FinalizeTokenProbas (frame_enc.c:136-181) over the tokens of all of `toks`, in any order.
/// Key frames reset to the defaults, so the old probability is always the default.
pub(crate) fn finalize_probs<'t>(toks: impl IntoIterator<Item = &'t [u16]>) -> [u8; 1056] {
    let c = costs();
    let mut cnt = [0u64; 2048]; // nb << 32 | total
    for &v in toks.into_iter().flatten() {
        // Fixed tokens count at 2047.
        let j = ((v | 0u16.wrapping_sub((v >> 14) & 1)) & 0x7ff) as usize;
        cnt[j] += 1 | (u64::from(v >> 15) << 32);
    }
    let mut p = *DEFAULT_PROBS;
    for i in 0..1056 {
        let (total, nb) = ((cnt[i] & 0xffff_ffff) as i64, (cnt[i] >> 32) as i64);
        if nb == 0 {
            continue;
        }
        let np = 255 - nb * 255 / total;
        let bc = |q: usize| nb * i64::from(c.bit[1][q]) + (total - nb) * i64::from(c.bit[0][q]);
        let old = bc(DEFAULT_PROBS[i] as usize) + i64::from(c.bit[0][UPDATE_PROBS[i] as usize]);
        let new = bc(np as usize) + i64::from(c.bit[1][UPDATE_PROBS[i] as usize]) + 8 * 256;
        if old > new {
            p[i] = np as u8;
        }
    }
    p
}

/// One token's probability.
#[inline(always)]
fn tok_prob(v: u16, p: &[u8; 1056]) -> u64 {
    // A fixed token's low bits are below 256, so the table read is in range either way.
    let pr = u64::from(p[(v & 0x7ff) as usize]);
    if v & FIXED_TOK != 0 {
        u64::from(v & 0xff)
    } else {
        pr
    }
}

/// putBit with the range kept as `n = range << 24`, so the renormalising shift comes from one
/// leading-zero count. Returns the new `n` and shift for a token.
#[inline(always)]
fn step(n: u64, v: u16, pr: u64) -> (u64, u32, i64) {
    let q = 256 - pr;
    // split+1 in the top half for a zero, range-split for a one.
    let (a, b) = if v & 0x8000 != 0 {
        (q, pr.wrapping_sub(1) << 24)
    } else {
        (pr, q << 24)
    };
    let add = (((n * pr + (q << 24)) >> 32) as i64) & -i64::from(v >> 15);
    let x = n.wrapping_mul(a).wrapping_add(b);
    let lz = x.leading_zeros();
    ((x >> 32) << (lz & 63), (lz - 24) & 7, add)
}

/// Codes tokens with the final probabilities: putBit inlined with the coder state in registers,
/// flushing four bytes at a time when no carry, 0xff byte or pending run is involved.
pub(crate) fn emit(e: &mut BoolEncoder, toks: &[u16], p: &[u8; 1056]) {
    let (mut n, mut value, mut nb) = (((e.rng + 1) as u64) << 24, e.value, e.nb_bits);
    for &v in toks {
        let (nn, sh, add) = step(n, v, tok_prob(v, p));
        n = nn;
        value = (value + add) << sh;
        nb += sh as i32;
        if nb > 24 {
            let s = (nb - 16) as u32;
            let w = (value as u64) >> s;
            let x = !(w as u32);
            if w >> 32 == 0 && e.run == 0 && x.wrapping_sub(0x0101_0101) & !x & 0x8080_8080 == 0 {
                e.buf.extend_from_slice(&(w as u32).to_be_bytes());
                value &= (1i64 << s) - 1;
                nb -= 32;
                continue;
            }
            e.value = value;
            e.nb_bits = nb;
            while e.nb_bits > 0 {
                e.flush();
            }
            value = e.value;
            nb = e.nb_bits;
        }
    }
    e.rng = (n >> 24) as i32 - 1;
    e.value = value;
    e.nb_bits = nb;
    while e.nb_bits > 0 {
        e.flush();
    }
}

/// `emit` on two streams at once, so the two dependency chains overlap. `tb` is `ta`'s length.
fn emit_pair(a: &mut BoolEncoder, b: &mut BoolEncoder, ta: &[u16], tb: &[u16], p: &[u8; 1056]) {
    let (mut na, mut va, mut nba) = (((a.rng + 1) as u64) << 24, a.value, a.nb_bits);
    let (mut nb, mut vb, mut nbb) = (((b.rng + 1) as u64) << 24, b.value, b.nb_bits);
    for (&v, &w) in ta.iter().zip(tb) {
        let (nn, sh, add) = step(na, v, tok_prob(v, p));
        na = nn;
        va = (va + add) << sh;
        nba += sh as i32;
        let (nn, sh, add) = step(nb, w, tok_prob(w, p));
        nb = nn;
        vb = (vb + add) << sh;
        nbb += sh as i32;
        if nba > 24 {
            a.value = va;
            a.nb_bits = nba;
            while a.nb_bits > 0 {
                a.flush();
            }
            va = a.value;
            nba = a.nb_bits;
        }
        if nbb > 24 {
            b.value = vb;
            b.nb_bits = nbb;
            while b.nb_bits > 0 {
                b.flush();
            }
            vb = b.value;
            nbb = b.nb_bits;
        }
    }
    a.rng = (na >> 24) as i32 - 1;
    a.value = va;
    a.nb_bits = nba;
    b.rng = (nb >> 24) as i32 - 1;
    b.value = vb;
    b.nb_bits = nbb;
    while a.nb_bits > 0 {
        a.flush();
    }
    while b.nb_bits > 0 {
        b.flush();
    }
}

/// Codes `nrows` MB rows, whose tokens `row(r)` gives: even rows into `a`, odd rows into `b`, two
/// rows at a time through `emit_pair`.
pub(crate) fn emit2<'t>(
    a: &mut BoolEncoder,
    b: &mut BoolEncoder,
    nrows: usize,
    row: impl Fn(usize) -> &'t [u16],
    p: &[u8; 1056],
) {
    for r in (0..nrows).step_by(2) {
        let sa = row(r);
        let sb: &[u16] = if r + 1 < nrows { row(r + 1) } else { &[] };
        let k = sa.len().min(sb.len());
        emit_pair(a, b, &sa[..k], &sb[..k], p);
        emit(a, &sa[k..], p);
        emit(b, &sb[k..], p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    /// A random level: mostly zero, some ones, and now and then past 67 and up to 2047.
    fn level(s: &mut u32) -> i16 {
        let r = rng(s);
        let m = match r % 16 {
            0..=8 => 0,
            9..=11 => 1,
            12 => 2 + (r >> 8) % 4,
            13 => 6 + (r >> 8) % 64,
            14 => 68 + (r >> 8) % 1980,
            _ => (r >> 8) % 2048,
        } as i16;
        if r & 0x10_0000 != 0 { -m } else { m }
    }

    fn fixed(bit: bool, p: u8) -> u16 {
        FIXED_TOK | u16::from(p) | (u16::from(bit) << 15)
    }

    /// The Vec-pushing record_level this file had before the indexed writer.
    fn record_level_ref(c: &mut Vec<u16>, v: usize, p: usize) {
        let gt1 = v > 1;
        c.push(tok(gt1, p + 2));
        if !gt1 {
            return;
        }
        let gt4 = v > 4;
        c.push(tok(gt4, p + 3));
        if !gt4 {
            let ne2 = v != 2;
            c.push(tok(ne2, p + 4));
            if ne2 {
                c.push(tok(v == 4, p + 5));
            }
            return;
        }
        let gt10 = v > 10;
        c.push(tok(gt10, p + 6));
        if !gt10 {
            let gt6 = v > 6;
            c.push(tok(gt6, p + 7));
            if !gt6 {
                c.push(fixed(v == 6, 159));
            } else {
                c.push(fixed(v >= 9, 165));
                c.push(fixed(v & 1 == 0, 145));
            }
            return;
        }
        let k = match v {
            ..19 => 0,
            19..35 => 1,
            35..67 => 2,
            _ => 3,
        };
        c.push(tok(k >= 2, p + 8));
        c.push(tok(k & 1 == 1, p + 9 + (k >> 1)));
        let v = v - (3 + (8 << k));
        let probs = CAT_PROBS[k];
        let nb = probs.len();
        for (j, &pb) in probs.iter().enumerate() {
            c.push(fixed((v >> (nb - 1 - j)) & 1 == 1, pb));
        }
    }

    /// The Vec-pushing record_coeffs this file had before.
    fn record_coeffs_ref(
        c: &mut Vec<u16>,
        typ: usize,
        ctx: usize,
        first: usize,
        lv: &[i16; 16],
    ) -> u8 {
        let last = (first..16)
            .rev()
            .find(|&n| lv[n] != 0)
            .map_or(-1, |n| n as i32);
        let base = typ * 264;
        let mut p = base + (BANDS[first] as usize * 3 + ctx) * 11;
        c.push(tok(last >= 0, p));
        if last < 0 {
            return 0;
        }
        let last = last as usize;
        for n in first..=last {
            let v = lv[n];
            let a = usize::from(v.unsigned_abs());
            c.push(tok(a > 0, p + 1));
            if a > 0 {
                record_level_ref(c, a, p);
                c.push(fixed(v < 0, 128));
                if n < 15 {
                    let pn = base + BANDS[n + 1] as usize * 33 + 11 * a.min(2);
                    c.push(tok(n < last, pn));
                }
            }
            p = base + BANDS[n + 1] as usize * 33 + 11 * a.min(2);
        }
        1
    }

    fn random_block(s: &mut u32) -> [i16; 16] {
        let mut lv = [0i16; 16];
        let density = rng(s) % 4;
        for l in &mut lv {
            if rng(s) % 4 <= density {
                *l = level(s);
            }
        }
        lv
    }

    #[test]
    fn record_coeffs_matches_ref() {
        let mut s = 0x1234_5678;
        // Exactly one block of room: the speculative writes must stay inside it.
        let mut t = TokBuf::default();
        t.grow(MAX_BLOCK_TOK).unwrap();
        let mut blocks = vec![[2047; 16], [-2047; 16], [-1; 16], [0; 16]];
        for _ in 0..3000 {
            blocks.push(random_block(&mut s));
        }
        for lv in &blocks {
            for typ in 0..4 {
                for first in 0..2 {
                    let ctx = (rng(&mut s) % 3) as usize;
                    t.clear();
                    let nz = record_coeffs(&mut t, typ, ctx, first, lv);
                    let mut r = Vec::new();
                    assert_eq!(nz, record_coeffs_ref(&mut r, typ, ctx, first, lv));
                    assert_eq!(&t[..], &r[..], "{lv:?} typ {typ} ctx {ctx} first {first}");
                }
            }
        }
    }

    /// The Vec-counting finalize_probs this file had before.
    fn finalize_probs_ref(toks: &[u16]) -> [u8; 1056] {
        let c = costs();
        let mut cnt = vec![0u64; 2048];
        for &v in toks {
            let j = if v & FIXED_TOK != 0 { 2047 } else { v & 0x7ff } as usize;
            cnt[j] += 1 | (u64::from(v >> 15) << 32);
        }
        let mut p = *DEFAULT_PROBS;
        for i in 0..1056 {
            let (total, nb) = ((cnt[i] & 0xffff_ffff) as i64, (cnt[i] >> 32) as i64);
            if nb == 0 {
                continue;
            }
            let np = 255 - nb * 255 / total;
            let bc = |q: usize| nb * i64::from(c.bit[1][q]) + (total - nb) * i64::from(c.bit[0][q]);
            let old = bc(DEFAULT_PROBS[i] as usize) + i64::from(c.bit[0][UPDATE_PROBS[i] as usize]);
            let new = bc(np as usize) + i64::from(c.bit[1][UPDATE_PROBS[i] as usize]) + 8 * 256;
            if old > new {
                p[i] = np as u8;
            }
        }
        p
    }

    fn random_tokens(s: &mut u32) -> TokBuf {
        let mut t = TokBuf::default();
        t.grow(MAX_BLOCK_TOK * 4000).unwrap();
        for _ in 0..3000 {
            let lv = random_block(s);
            let (typ, ctx) = ((rng(s) % 4) as usize, (rng(s) % 3) as usize);
            record_coeffs(&mut t, typ, ctx, usize::from(typ == 0), &lv);
        }
        t
    }

    #[test]
    fn finalize_probs_matches_ref() {
        let mut s = 0x9e37_79b9;
        let t = random_tokens(&mut s);
        assert_eq!(finalize_probs([&t[..]]), finalize_probs_ref(&t));
        // Split anywhere, the counts are the same.
        let (a, b) = t.split_at(t.len() / 3);
        assert_eq!(finalize_probs([b, a]), finalize_probs_ref(&t));
        assert_eq!(
            finalize_probs(std::iter::empty::<&[u16]>()),
            finalize_probs_ref(&[])
        );
    }

    /// Final probabilities with the edges the coder must handle: 0, 1, 255.
    fn random_probs(s: &mut u32) -> [u8; 1056] {
        let mut p = [0u8; 1056];
        for x in &mut p {
            *x = match rng(s) % 8 {
                0 => 0,
                1 => 1,
                2 => 255,
                3 => 254,
                _ => (rng(s) >> 8) as u8,
            };
        }
        p
    }

    /// Random tokens, fixed and table, with bits that follow, oppose or ignore the probability.
    fn random_stream(s: &mut u32, n: usize, p: &[u8; 1056], mode: u32) -> Vec<u16> {
        (0..n)
            .map(|_| {
                let r = rng(s);
                let fixed_tok = r & 3 == 0;
                let (idx, pr) = if fixed_tok {
                    let q = (r >> 8) as u8;
                    (u16::from(q), q)
                } else {
                    let i = ((r >> 8) % 1056) as u16;
                    (i, p[usize::from(i)])
                };
                let bit = match mode {
                    0 => (rng(s) & 0xff) >= u32::from(pr),
                    1 => (rng(s) & 0xff) < u32::from(pr),
                    2 => !rng(s).is_multiple_of(10),
                    _ => rng(s) & 1 == 1,
                };
                idx | (u16::from(bit) << 15) | if fixed_tok { FIXED_TOK } else { 0 }
            })
            .collect()
    }

    /// Codes with put_bit, the definition of the coder.
    fn put_ref(e: &mut BoolEncoder, toks: &[u16], p: &[u8; 1056]) {
        for &v in toks {
            let pr = if v & FIXED_TOK != 0 {
                (v & 0xff) as u8
            } else {
                p[(v & 0x7ff) as usize]
            };
            e.put_bit(v >> 15 != 0, pr);
        }
    }

    #[test]
    fn emit_matches_put_bit() {
        let mut s = 0x2545_f491;
        for round in 0..48 {
            let p = random_probs(&mut s);
            let n = (rng(&mut s) % 6000) as usize;
            let toks = random_stream(&mut s, n, &p, round % 4);
            let mut a = BoolEncoder::default();
            a.reset();
            put_ref(&mut a, &toks, &p);
            // The same tokens in pieces, so the state carries between calls.
            let mut b = BoolEncoder::default();
            b.reset();
            let cut = if n == 0 {
                0
            } else {
                (rng(&mut s) as usize) % n
            };
            emit(&mut b, &toks[..cut], &p);
            emit(&mut b, &toks[cut..], &p);
            assert_eq!(a.finish(), b.finish(), "round {round}");
        }
    }

    #[test]
    fn emit2_matches_put_bit() {
        let mut s = 0x7f4a_7c15;
        for round in 0..48 {
            let p = random_probs(&mut s);
            let nrows = 1 + (rng(&mut s) % 6) as usize;
            let mut toks = Vec::new();
            let mut rows = vec![0];
            for _ in 0..nrows {
                let n = (rng(&mut s) % 1500) as usize;
                toks.extend(random_stream(&mut s, n, &p, round % 4));
                rows.push(toks.len());
            }
            let (mut a, mut b) = (BoolEncoder::default(), BoolEncoder::default());
            let (mut ra, mut rb) = (BoolEncoder::default(), BoolEncoder::default());
            for e in [&mut a, &mut b, &mut ra, &mut rb] {
                e.reset();
            }
            for r in 0..nrows {
                let e = if r & 1 == 0 { &mut ra } else { &mut rb };
                put_ref(e, &toks[rows[r]..rows[r + 1]], &p);
            }
            emit2(&mut a, &mut b, nrows, |r| &toks[rows[r]..rows[r + 1]], &p);
            assert_eq!(a.finish(), ra.finish(), "even rows, round {round}");
            assert_eq!(b.finish(), rb.finish(), "odd rows, round {round}");
        }
    }
}
