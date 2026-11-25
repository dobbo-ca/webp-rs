//! VP8 key frame decoder (RFC 6386), the way libwebp does it, so the pixels are bit exact with
//! dwebp. Follows libwebp's src/dec (vp8_dec.c, tree_dec.c, quant_dec.c, frame_dec.c, io_dec.c)
//! and src/dsp (dec.c, upsampling.c, yuv.h), BSD-3-Clause; see NOTICE.

mod filter;
mod pred;
mod yuv;

use crate::Error;
use crate::limits::{Budget, cap_bytes, check_cancel, fit_zeroed};
use crate::tables::{
    AC_TABLE, BANDS, CAT_PROBS, COEFF_PROBS, COEFF_UPDATE_PROBS, DC_TABLE, PRED_PROB, ZIGZAG,
};
use pred::BPS;
use std::sync::atomic::AtomicBool;

/// A decoded key frame: 4:2:0 planes of the macroblock-padded size.
#[derive(Default)]
pub(crate) struct Frame {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub c_stride: usize,
}

/// Checks the key frame tag and returns the frame size.
pub(crate) fn header(data: &[u8]) -> Result<(usize, usize), Error> {
    if data.len() < 10
        || data[0] & 1 != 0
        || (data[0] >> 1) & 7 > 3
        || (data[0] >> 4) & 1 == 0
        || data[3] != 0x9d
        || data[4] != 0x01
        || data[5] != 0x2a
    {
        return Err(Error::Vp8Header);
    }
    let tag = u32::from_le_bytes([data[0], data[1], data[2], 0]);
    if (tag >> 5) as usize > data.len() - 10 {
        return Err(Error::Vp8Header);
    }
    let w = usize::from(u16::from_le_bytes([data[6], data[7]]) & 0x3fff);
    let h = usize::from(u16::from_le_bytes([data[8], data[9]]) & 0x3fff);
    if w == 0 || h == 0 {
        return Err(Error::Vp8Header);
    }
    Ok((w, h))
}

/// The planes and the per-macroblock arrays of a decode, kept between calls. The caller charges
/// what they hold (`bytes`) to the budget before a call; one that has to grow is charged at its
/// new size and the old size is given back.
#[derive(Default)]
pub(crate) struct Scratch {
    /// The planes of the last frame decoded.
    pub frame: Frame,
    intra_t: Vec<u8>,
    nz_t: Vec<[u8; 9]>,
    top_y: Vec<u8>,
    top_u: Vec<u8>,
    top_v: Vec<u8>,
    mbs: Vec<Mb>,
    finfos: Vec<FInfo>,
}

impl Scratch {
    /// The bytes held, at capacity.
    pub(crate) fn bytes(&self) -> u64 {
        cap_bytes(&self.frame.y)
            + cap_bytes(&self.frame.u)
            + cap_bytes(&self.frame.v)
            + cap_bytes(&self.intra_t)
            + cap_bytes(&self.nz_t)
            + cap_bytes(&self.top_y)
            + cap_bytes(&self.top_u)
            + cap_bytes(&self.top_v)
            + cap_bytes(&self.mbs)
            + cap_bytes(&self.finfos)
    }
}

/// The boolean entropy decoder of RFC 6386 7, in libwebp's form (src/utils/bit_reader_utils.c,
/// BSD-3-Clause): `rng` is the range minus one and `eof` is set when a bit is needed past the end.
#[derive(Default)]
struct BoolDec<'a> {
    b: &'a [u8],
    pos: usize,
    val: u64,
    rng: u32,
    bits: i32,
    eof: bool,
}

impl<'a> BoolDec<'a> {
    fn new(b: &'a [u8]) -> BoolDec<'a> {
        let mut d = BoolDec {
            b,
            pos: 0,
            val: 0,
            rng: 254,
            bits: -8,
            eof: false,
        };
        d.load();
        d
    }

    fn load(&mut self) {
        if self.pos + 8 <= self.b.len() {
            let mut a = [0u8; 8];
            a.copy_from_slice(&self.b[self.pos..self.pos + 8]);
            self.val = (u64::from_be_bytes(a) >> 8) | (self.val << 56);
            self.pos += 7;
            self.bits += 56;
        } else if self.pos < self.b.len() {
            self.val = u64::from(self.b[self.pos]) | (self.val << 8);
            self.pos += 1;
            self.bits += 8;
        } else if !self.eof {
            self.val <<= 8;
            self.bits += 8;
            self.eof = true;
        } else {
            self.bits = 0; // avoid undefined shifts on garbage input
        }
    }

    /// One bit with probability prob/256 of being 0.
    fn bit(&mut self, prob: u8) -> u32 {
        let mut rng = self.rng;
        if self.bits < 0 {
            self.load();
        }
        let pos = self.bits as u32;
        let split = (rng * u32::from(prob)) >> 8;
        let b;
        if (self.val >> pos) as u32 > split {
            rng -= split;
            self.val -= u64::from(split + 1) << pos;
            b = 1;
        } else {
            rng = split + 1;
            b = 0;
        }
        let shift = 7 ^ (31 - rng.leading_zeros() as i32);
        self.bits -= shift;
        self.rng = (rng << shift) - 1;
        b
    }

    fn flag(&mut self) -> bool {
        self.bit(0x80) != 0
    }

    /// n bits, most significant first.
    fn value(&mut self, n: u32) -> i32 {
        let mut v = 0;
        for _ in 0..n {
            v = v << 1 | self.bit(0x80) as i32;
        }
        v
    }

    /// n bits and a sign.
    fn signed(&mut self, n: u32) -> i32 {
        let v = self.value(n);
        if self.flag() { -v } else { v }
    }
}

#[derive(Default)]
struct SegHeader {
    used: bool,
    update_map: bool,
    abs_delta: bool,
    quant: [i32; 4],
    strength: [i32; 4],
    probs: [u8; 3],
}

#[derive(Default)]
struct FilterHeader {
    simple: bool,
    use_delta: bool,
    level: i32,
    sharpness: i32,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],
}

/// dc and ac dequantizers.
#[derive(Clone, Copy, Default)]
struct Quant {
    y1: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
}

#[derive(Clone, Copy, Default)]
struct FInfo {
    limit: u8,
    ilevel: u8,
    hev: u8,
    inner: bool,
}

#[derive(Clone, Copy, Default)]
struct Mb {
    segment: u8,
    skip: bool,
    i4: bool,
    uvmode: u8,
    modes: [u8; 16],
}

/// Prediction modes, libwebp's order. The 16x16 modes share the first four.
pub(crate) const B_DC: u8 = 0;
pub(crate) const B_TM: u8 = 1;
pub(crate) const B_VE: u8 = 2;
pub(crate) const B_HE: u8 = 3;
pub(crate) const B_RD: u8 = 4;
pub(crate) const B_VR: u8 = 5;
pub(crate) const B_LD: u8 = 6;
pub(crate) const B_VL: u8 = 7;
pub(crate) const B_HD: u8 = 8;
pub(crate) const B_HU: u8 = 9;

/// libwebp's tree for sub-block modes (negated leaves).
const K_Y_MODES_INTRA4: [i8; 18] = [
    -(B_DC as i8),
    1,
    -(B_TM as i8),
    2,
    -(B_VE as i8),
    3,
    4,
    6,
    -(B_HE as i8),
    5,
    -(B_RD as i8),
    -(B_VR as i8),
    -(B_LD as i8),
    7,
    -(B_VL as i8),
    8,
    -(B_HD as i8),
    -(B_HU as i8),
];

type Probs = [[[[u8; 11]; 3]; 8]; 4];

struct Dec<'a> {
    cancel: &'a AtomicBool,
    br: BoolDec<'a>,
    parts: [BoolDec<'a>; 8],
    nparts: usize,
    mbw: usize,
    mbh: usize,
    seg: SegHeader,
    flt: FilterHeader,
    f_type: u8, // 0 none, 1 simple, 2 complex
    q: [Quant; 4],
    probs: Probs,
    use_skip: bool,
    skip_p: u8,
    fs: [[FInfo; 2]; 4],

    intra_t: Vec<u8>,
    intra_l: [u8; 4],
    nz_t: Vec<[u8; 9]>,
    nz_l: [u8; 9],
    top_y: Vec<u8>,
    top_u: Vec<u8>,
    top_v: Vec<u8>,
    mbs: Vec<Mb>,
    finfos: Vec<FInfo>,

    f: Frame,
    yuv: [u8; BPS * 17 + BPS * 9],
    coeffs: [i16; 384],
    nonzero: [bool; 24], // per 4x4 block: needs the inverse transform
}

/// Decodes the key frame in `data` into `sc.frame`, in the buffers of `sc`. What it allocates
/// beyond them is charged to `budget` first. `cancel` is checked once a macroblock row. After an
/// error the frame and the buffers hold whatever was done.
pub(crate) fn decode(
    data: &[u8],
    budget: &Budget,
    cancel: &AtomicBool,
    sc: &mut Scratch,
) -> Result<(), Error> {
    let (w, h) = header(data)?;
    let mbw = (w + 15) >> 4;
    let mbh = (h + 15) >> 4;
    let pw = mbw * 16;
    let f = Frame {
        w,
        h,
        y_stride: pw,
        c_stride: pw / 2,
        ..std::mem::take(&mut sc.frame)
    };
    let part_len = (u32::from_le_bytes([data[0], data[1], data[2], 0]) >> 5) as usize;
    let mut d = Dec {
        cancel,
        br: BoolDec::new(&data[10..10 + part_len]),
        parts: Default::default(),
        nparts: 0,
        mbw,
        mbh,
        seg: SegHeader::default(),
        flt: FilterHeader::default(),
        f_type: 0,
        q: [Quant::default(); 4],
        probs: [[[[0; 11]; 3]; 8]; 4],
        use_skip: false,
        skip_p: 0,
        fs: [[FInfo::default(); 2]; 4],
        intra_t: std::mem::take(&mut sc.intra_t),
        intra_l: [0; 4],
        nz_t: std::mem::take(&mut sc.nz_t),
        nz_l: [0; 9],
        top_y: std::mem::take(&mut sc.top_y),
        top_u: std::mem::take(&mut sc.top_u),
        top_v: std::mem::take(&mut sc.top_v),
        mbs: std::mem::take(&mut sc.mbs),
        finfos: std::mem::take(&mut sc.finfos),
        f,
        yuv: [0; BPS * 17 + BPS * 9],
        coeffs: [0; 384],
        nonzero: [false; 24],
    };
    let r = d.run(&data[10 + part_len..], budget);
    sc.intra_t = d.intra_t;
    sc.nz_t = d.nz_t;
    sc.top_y = d.top_y;
    sc.top_u = d.top_u;
    sc.top_v = d.top_v;
    sc.mbs = d.mbs;
    sc.finfos = d.finfos;
    sc.frame = d.f;
    r
}

fn clip(v: i32, m: i32) -> i32 {
    v.clamp(0, m)
}

impl<'a> Dec<'a> {
    /// Zeroes the planes and the per-macroblock arrays at the frame's size, charging any that
    /// grow, then decodes the headers and the rows.
    fn run(&mut self, rest: &'a [u8], budget: &Budget) -> Result<(), Error> {
        let mbw = self.mbw;
        let pn = mbw * 16 * self.mbh * 16;
        fit_zeroed(budget, &mut self.f.y, pn)?;
        fit_zeroed(budget, &mut self.f.u, pn / 4)?;
        fit_zeroed(budget, &mut self.f.v, pn / 4)?;
        fit_zeroed(budget, &mut self.intra_t, 4 * mbw)?;
        fit_zeroed(budget, &mut self.nz_t, mbw)?;
        fit_zeroed(budget, &mut self.top_y, 16 * mbw)?;
        fit_zeroed(budget, &mut self.top_u, 8 * mbw)?;
        fit_zeroed(budget, &mut self.top_v, 8 * mbw)?;
        fit_zeroed(budget, &mut self.mbs, mbw)?;
        fit_zeroed(budget, &mut self.finfos, mbw)?;
        self.parse_headers(rest)?;
        for my in 0..self.mbh {
            self.row(my)?;
        }
        Ok(())
    }

    fn parse_headers(&mut self, rest: &'a [u8]) -> Result<(), Error> {
        let br = &mut self.br;
        br.value(2); // color space and clamping type
        let s = &mut self.seg;
        s.probs = [255, 255, 255];
        s.abs_delta = true;
        s.used = br.flag();
        if s.used {
            s.update_map = br.flag();
            if br.flag() {
                // update data
                s.abs_delta = br.flag();
                for i in 0..4 {
                    s.quant[i] = 0;
                    if br.flag() {
                        s.quant[i] = br.signed(7);
                    }
                }
                for i in 0..4 {
                    s.strength[i] = 0;
                    if br.flag() {
                        s.strength[i] = br.signed(6);
                    }
                }
            }
            if s.update_map {
                for i in 0..3 {
                    if br.flag() {
                        s.probs[i] = br.value(8) as u8;
                    }
                }
            }
        }
        if br.eof {
            return Err(Error::Vp8Truncated);
        }
        let f = &mut self.flt;
        f.simple = br.flag();
        f.level = br.value(6);
        f.sharpness = br.value(3);
        f.use_delta = br.flag();
        if f.use_delta && br.flag() {
            for i in 0..4 {
                if br.flag() {
                    f.ref_delta[i] = br.signed(6);
                }
            }
            for i in 0..4 {
                if br.flag() {
                    f.mode_delta[i] = br.signed(6);
                }
            }
        }
        self.f_type = if f.level == 0 {
            0
        } else if f.simple {
            1
        } else {
            2
        };
        if br.eof {
            return Err(Error::Vp8Truncated);
        }
        self.parse_partitions(rest)?;
        self.parse_quant();
        let br = &mut self.br;
        br.flag(); // update_proba: ignored on key frames
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for p in 0..11 {
                        self.probs[t][b][c][p] = if br.bit(COEFF_UPDATE_PROBS[t][b][c][p]) != 0 {
                            br.value(8) as u8
                        } else {
                            COEFF_PROBS[t][b][c][p]
                        };
                    }
                }
            }
        }
        self.use_skip = br.flag();
        if self.use_skip {
            self.skip_p = br.value(8) as u8;
        }
        if br.eof {
            return Err(Error::Vp8Truncated);
        }
        self.filter_strengths();
        Ok(())
    }

    fn parse_partitions(&mut self, buf: &'a [u8]) -> Result<(), Error> {
        self.nparts = 1 << self.br.value(2);
        let last = self.nparts - 1;
        if buf.len() < 3 * last {
            return Err(Error::Vp8Truncated);
        }
        let (mut sz, mut start) = (buf, &buf[3 * last..]);
        for p in 0..last {
            let n = (usize::from(sz[0]) | usize::from(sz[1]) << 8 | usize::from(sz[2]) << 16)
                .min(start.len());
            self.parts[p] = BoolDec::new(&start[..n]);
            start = &start[n..];
            sz = &sz[3..];
        }
        self.parts[last] = BoolDec::new(start);
        if start.is_empty() {
            return Err(Error::Vp8Truncated); // libwebp: the last partition must not be empty
        }
        Ok(())
    }

    fn parse_quant(&mut self) {
        let br = &mut self.br;
        let base = br.value(7);
        let mut dq = [0i32; 5];
        for v in &mut dq {
            if br.flag() {
                *v = br.signed(4);
            }
        }
        let [dy1dc, dy2dc, dy2ac, duvdc, duvac] = dq;
        for i in 0..4 {
            let q = if self.seg.used {
                let mut v = self.seg.quant[i];
                if !self.seg.abs_delta {
                    v += base;
                }
                v
            } else if i > 0 {
                self.q[i] = self.q[0];
                continue;
            } else {
                base
            };
            let m = &mut self.q[i];
            m.y1[0] = i32::from(DC_TABLE[clip(q + dy1dc, 127) as usize]);
            m.y1[1] = i32::from(AC_TABLE[clip(q, 127) as usize]);
            m.y2[0] = i32::from(DC_TABLE[clip(q + dy2dc, 127) as usize]) * 2;
            m.y2[1] = ((i32::from(AC_TABLE[clip(q + dy2ac, 127) as usize]) * 101581) >> 16).max(8);
            m.uv[0] = i32::from(DC_TABLE[clip(q + duvdc, 117) as usize]);
            m.uv[1] = i32::from(AC_TABLE[clip(q + duvac, 127) as usize]);
        }
    }

    fn filter_strengths(&mut self) {
        if self.f_type == 0 {
            return;
        }
        let h = &self.flt;
        for s in 0..4 {
            let mut base = h.level;
            if self.seg.used {
                base = self.seg.strength[s];
                if !self.seg.abs_delta {
                    base += h.level;
                }
            }
            for i4 in 0..2 {
                let mut level = base;
                if h.use_delta {
                    level += h.ref_delta[0];
                    if i4 == 1 {
                        level += h.mode_delta[0];
                    }
                }
                level = clip(level, 63);
                let fi = &mut self.fs[s][i4];
                *fi = FInfo {
                    inner: i4 == 1,
                    ..FInfo::default()
                };
                if level == 0 {
                    continue;
                }
                let mut il = level;
                if h.sharpness > 0 {
                    if h.sharpness > 4 {
                        il >>= 2;
                    } else {
                        il >>= 1;
                    }
                    il = il.min(9 - h.sharpness);
                }
                il = il.max(1);
                fi.ilevel = il as u8;
                fi.limit = (2 * level + il) as u8;
                if level >= 40 {
                    fi.hev = 2;
                } else if level >= 15 {
                    fi.hev = 1;
                }
            }
        }
    }

    fn row(&mut self, my: usize) -> Result<(), Error> {
        check_cancel(self.cancel)?;
        self.intra_l = [0; 4];
        self.nz_l = [0; 9];
        for x in 0..self.mbw {
            self.parse_intra_mode(x);
        }
        if self.br.eof {
            return Err(Error::Vp8Truncated);
        }
        let pi = my & (self.nparts - 1);
        let mut tb = std::mem::take(&mut self.parts[pi]);
        let r = self.row_mbs(my, &mut tb);
        self.parts[pi] = tb;
        r?;
        if self.f_type > 0 {
            self.filter_row(my);
        }
        Ok(())
    }

    fn row_mbs(&mut self, my: usize, tb: &mut BoolDec<'a>) -> Result<(), Error> {
        for x in 0..self.mbw {
            let m = self.mbs[x];
            let mut skip = self.use_skip && m.skip;
            if !skip {
                skip = self.parse_residuals(x, &m, tb);
            } else {
                let t = &mut self.nz_t[x];
                for i in 0..8 {
                    t[i] = 0;
                    self.nz_l[i] = 0;
                }
                if !m.i4 {
                    t[8] = 0;
                    self.nz_l[8] = 0;
                }
                self.nonzero = [false; 24];
            }
            if self.f_type > 0 {
                let mut fi = self.fs[usize::from(m.segment)][usize::from(m.i4)];
                fi.inner = fi.inner || !skip;
                self.finfos[x] = fi;
            }
            if tb.eof {
                return Err(Error::Vp8Truncated);
            }
            self.recon(x, my, &m);
        }
        Ok(())
    }

    fn parse_intra_mode(&mut self, x: usize) {
        let br = &mut self.br;
        let mut m = Mb::default();
        if self.seg.update_map {
            if br.bit(self.seg.probs[0]) == 0 {
                m.segment = br.bit(self.seg.probs[1]) as u8;
            } else {
                m.segment = br.bit(self.seg.probs[2]) as u8 + 2;
            }
        }
        if self.use_skip {
            m.skip = br.bit(self.skip_p) != 0;
        }
        let top = &mut self.intra_t[4 * x..4 * x + 4];
        let left = &mut self.intra_l;
        m.i4 = br.bit(145) == 0;
        if !m.i4 {
            let y = if br.bit(156) != 0 {
                if br.bit(128) != 0 { B_TM } else { B_HE }
            } else if br.bit(163) != 0 {
                B_VE
            } else {
                B_DC
            };
            m.modes = [y; 16];
            for i in 0..4 {
                top[i] = y;
                left[i] = y;
            }
        } else {
            for y in 0..4 {
                let mut ym = left[y];
                for x in 0..4 {
                    let p = &PRED_PROB[usize::from(top[x])][usize::from(ym)];
                    let mut i = i32::from(K_Y_MODES_INTRA4[br.bit(p[0]) as usize]);
                    while i > 0 {
                        i = i32::from(
                            K_Y_MODES_INTRA4[(2 * i) as usize + br.bit(p[i as usize]) as usize],
                        );
                    }
                    ym = (-i) as u8;
                    top[x] = ym;
                    m.modes[4 * y + x] = ym;
                }
                left[y] = ym;
            }
        }
        m.uvmode = if br.bit(142) == 0 {
            B_DC
        } else if br.bit(114) == 0 {
            B_VE
        } else if br.bit(183) != 0 {
            B_TM
        } else {
            B_HE
        };
        self.mbs[x] = m;
    }

    /// Reads one macroblock's coefficients into `coeffs` and reports whether it has none
    /// (libwebp's "skip").
    fn parse_residuals(&mut self, x: usize, m: &Mb, br: &mut BoolDec<'a>) -> bool {
        self.coeffs = [0; 384];
        let q = self.q[usize::from(m.segment)];
        let top = &mut self.nz_t[x];
        let left = &mut self.nz_l;
        let mut any = false;
        let (mut first, mut t) = (0usize, 3usize);
        if !m.i4 {
            let mut dc = [0i16; 16];
            let ctx = usize::from(top[8] + left[8]);
            let nz = get_coeffs(br, &self.probs[1], ctx, &q.y2, 0, &mut dc);
            top[8] = u8::from(nz > 0);
            left[8] = u8::from(nz > 0);
            if nz > 1 {
                pred::wht(&dc, &mut self.coeffs);
            } else {
                let dc0 = ((i32::from(dc[0]) + 3) >> 3) as i16;
                for i in (0..256).step_by(16) {
                    self.coeffs[i] = dc0;
                }
            }
            first = 1;
            t = 0;
        }
        for y in 0..4 {
            for x in 0..4 {
                let n = 4 * y + x;
                let ctx = usize::from(left[y] + top[x]);
                let out = &mut self.coeffs[16 * n..16 * n + 16];
                let nz = get_coeffs(br, &self.probs[t], ctx, &q.y1, first, out);
                let l = u8::from(nz > first);
                left[y] = l;
                top[x] = l;
                self.nonzero[n] = nz > 1 || out[0] != 0;
                any = any || self.nonzero[n];
            }
        }
        for ch in 0..2 {
            // U then V
            for y in 0..2 {
                for x in 0..2 {
                    let n = 16 + 4 * ch + 2 * y + x;
                    let ctx = usize::from(left[4 + 2 * ch + y] + top[4 + 2 * ch + x]);
                    let out = &mut self.coeffs[16 * n..16 * n + 16];
                    let nz = get_coeffs(br, &self.probs[2], ctx, &q.uv, 0, out);
                    let l = u8::from(nz > 0);
                    left[4 + 2 * ch + y] = l;
                    top[4 + 2 * ch + x] = l;
                    self.nonzero[n] = nz > 1 || out[0] != 0;
                    any = any || self.nonzero[n];
                }
            }
        }
        !any
    }
}

/// Reads one block's tokens from position n and returns the position after the last nonzero
/// coefficient (n if there is none).
fn get_coeffs(
    br: &mut BoolDec<'_>,
    pr: &[[[u8; 11]; 3]; 8],
    ctx: usize,
    dq: &[i32; 2],
    mut n: usize,
    out: &mut [i16],
) -> usize {
    let mut p = &pr[usize::from(BANDS[n])][ctx];
    while n < 16 {
        if br.bit(p[0]) == 0 {
            return n;
        }
        while br.bit(p[1]) == 0 {
            n += 1;
            p = &pr[usize::from(BANDS[n])][0];
            if n == 16 {
                return 16;
            }
        }
        let pn = &pr[usize::from(BANDS[n + 1])];
        let mut v: i32;
        if br.bit(p[2]) == 0 {
            v = 1;
            p = &pn[1];
        } else {
            v = large_value(br, p);
            p = &pn[2];
        }
        if br.bit(0x80) != 0 {
            v = -v;
        }
        out[usize::from(ZIGZAG[n])] = (v * dq[usize::from(n > 0)]) as i16;
        n += 1;
    }
    16
}

fn large_value(br: &mut BoolDec<'_>, p: &[u8; 11]) -> i32 {
    if br.bit(p[3]) == 0 {
        if br.bit(p[4]) == 0 {
            return 2;
        }
        return 3 + br.bit(p[5]) as i32;
    }
    if br.bit(p[6]) == 0 {
        if br.bit(p[7]) == 0 {
            return 5 + br.bit(159) as i32;
        }
        let v = 7 + 2 * br.bit(165) as i32;
        return v + br.bit(145) as i32;
    }
    let bit1 = br.bit(p[8]);
    let bit0 = br.bit(p[9 + bit1 as usize]);
    let cat = (2 * bit1 + bit0) as usize;
    let mut v = 0i32;
    for &pr in CAT_PROBS[cat] {
        v = 2 * v + br.bit(pr) as i32;
    }
    v + 3 + (8 << cat)
}
