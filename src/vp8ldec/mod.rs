//! VP8L lossless image streams (RFC 9649), bit exact with libwebp. The transforms and
//! prefix-code reading follow the spec and libwebp's src/dec/vp8l_dec.c and src/dsp/lossless.c
//! (BSD-3-Clause; see NOTICE).

mod bits;
mod huff;
mod transform;

use crate::Error;
use crate::limits::{Budget, cap_bytes, check_cancel, fit_zeroed};
use bits::BitReader;
use huff::{CODE_BYTES, Code};
use std::sync::atomic::AtomicBool;
use transform::inverse_row;

/// Reads the image size and the alpha-is-used flag.
pub(crate) fn header(data: &[u8]) -> Result<(usize, usize, bool), Error> {
    if data.len() < 5 || data[0] != 0x2f {
        return Err(Error::Vp8lHeader);
    }
    let v = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    if v >> 29 != 0 {
        return Err(Error::Vp8lHeader); // version
    }
    Ok((
        (v & 0x3fff) as usize + 1,
        ((v >> 14) & 0x3fff) as usize + 1,
        (v >> 28) & 1 != 0,
    ))
}

/// The buffers of a decode, kept between calls. The caller charges what they hold (`bytes`) to
/// the budget before a call; a buffer that has to grow is charged at its new size and the old
/// size is given back. After an error some of them may have been dropped.
#[derive(Default)]
pub(crate) struct Scratch {
    /// The transform list, and the data of each kind of transform (a kind occurs once).
    ts: Vec<Transform>,
    tdata: [Vec<u8>; 4],
    meta: Vec<u8>,
    slot_of: Vec<i32>,
    groups: Vec<Code>,
    /// The code of the code lengths.
    clc: Code,
    lens: Vec<u8>,
    cache: Vec<u32>,
    ring: Vec<u8>,
    rows: Vec<u8>,
}

impl Scratch {
    /// The bytes the buffers hold, at capacity.
    pub(crate) fn bytes(&self) -> u64 {
        cap_bytes(&self.ts)
            + self.tdata.iter().map(cap_bytes).sum::<u64>()
            + cap_bytes(&self.meta)
            + cap_bytes(&self.slot_of)
            + cap_bytes(&self.groups)
            + self.groups.iter().map(Code::bytes).sum::<u64>()
            + self.clc.bytes()
            + cap_bytes(&self.lens)
            + cap_bytes(&self.cache)
            + cap_bytes(&self.ring)
            + cap_bytes(&self.rows)
    }
}

/// Decodes `data` into `dst`, 4*w*h bytes of R, G, B, A (not premultiplied), w and h from
/// `header`, in the buffers of `sc`. What it allocates beyond them is charged to `budget` first;
/// its error ends the decode. `cancel` is checked once a row.
pub(crate) fn decode(
    data: &[u8],
    dst: &mut [u8],
    budget: &Budget,
    cancel: &AtomicBool,
    sc: &mut Scratch,
) -> Result<(), Error> {
    let (w, h, _) = header(data)?;
    let mut d = Dec {
        br: BitReader::new(&data[5..]),
        cancel,
        budget,
        sc,
    };
    d.decode(&mut dst[..4 * w * h], w, h)?;
    if d.br.over {
        return Err(Error::Vp8lTruncated);
    }
    for px in dst.as_chunks_mut::<4>().0 {
        px.swap(0, 2); // BGRA to RGBA
    }
    Ok(())
}

/// Decodes the headerless VP8L stream of an ALPH chunk, w x h pixels, into `alpha` (w*h bytes, the
/// green channel of each pixel). It holds the last `RING_PX` pixels the stream can still copy
/// from, not the image, and undoes the transforms a row at a time as the rows complete.
pub(crate) fn decode_alpha(
    data: &[u8],
    w: usize,
    h: usize,
    alpha: &mut [u8],
    budget: &Budget,
    cancel: &AtomicBool,
    sc: &mut Scratch,
) -> Result<(), Error> {
    let mut d = Dec {
        br: BitReader::new(data),
        cancel,
        budget,
        sc,
    };
    let mut ts = std::mem::take(&mut d.sc.ts);
    let r = d.alpha_in(&mut ts, w, h, alpha);
    d.recycle(ts);
    r
}

/// The farthest a copy reaches back (distance code 40 with all its extra bits is 1<<20 pixels)
/// plus the pixels a run can write before its rows are converted (a run of 4096 and a row of
/// 16383).
pub(crate) const RING_PX: usize = (1 << 20) + (1 << 15);

type Sink<'s> = Option<&'s mut dyn FnMut(&[u8], usize) -> Result<(), Error>>;

struct Dec<'a> {
    br: BitReader<'a>,
    cancel: &'a AtomicBool,
    budget: &'a Budget,
    sc: &'a mut Scratch,
}

/// Where pixel `pos` lives in an array of `n` pixels used as a ring.
fn slot(n: usize, pos: usize) -> usize {
    if pos >= n { pos % n } else { pos }
}

fn get(a: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]])
}

fn put(a: &mut [u8], i: usize, v: u32) {
    a[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}

fn ceil(n: usize, bits: u32) -> usize {
    (n + (1 << bits) - 1) >> bits
}

pub(super) struct Transform {
    kind: u8,
    bits: u32,
    /// Image width this transform sees.
    xs: usize,
    data: Vec<u8>,
}

const T_PREDICTOR: u8 = 0;
const T_CROSS: u8 = 1;
const T_SUB_GREEN: u8 = 2;
const T_INDEX: u8 = 3;

impl Dec<'_> {
    fn decode(&mut self, a: &mut [u8], w: usize, h: usize) -> Result<(), Error> {
        let mut ts = std::mem::take(&mut self.sc.ts);
        let r = self.decode_ts(&mut ts, a, w, h);
        self.recycle(ts);
        r
    }

    fn decode_ts(
        &mut self,
        ts: &mut Vec<Transform>,
        a: &mut [u8],
        w: usize,
        h: usize,
    ) -> Result<(), Error> {
        let xs = self.read_transforms(ts, w, h)?;
        self.decode_image(&mut a[..4 * xs * h], xs, h, true, &mut None)?;
        for t in ts.iter().rev() {
            self.inverse(a, t, h, w)?;
        }
        Ok(())
    }

    /// The alpha plane of an ALPH chunk: the main image goes through a ring and the rows are
    /// converted as they complete.
    fn alpha_in(
        &mut self,
        ts: &mut Vec<Transform>,
        w: usize,
        h: usize,
        alpha: &mut [u8],
    ) -> Result<(), Error> {
        let xs = self.read_transforms(ts, w, h)?;
        let mut ring = std::mem::take(&mut self.sc.ring);
        fit_zeroed(self.budget, &mut ring, 4 * (xs * h).min(RING_PX))?;
        let mut rows = std::mem::take(&mut self.sc.rows);
        // The row being converted and the last one the predictor made.
        fit_zeroed(self.budget, &mut rows, 8 * w)?;
        let (cur, prev) = rows.split_at_mut(4 * w);
        let ts: &[Transform] = ts;
        let mut sink = |ring: &[u8], y: usize| -> Result<(), Error> {
            let n = ring.len() / 4;
            for x in 0..xs {
                put(cur, x, get(ring, slot(n, y * xs + x)));
            }
            inverse_row(cur, prev, ts, y, w);
            for x in 0..w {
                alpha[y * w + x] = cur[4 * x + 1]; // green, in BGRA order
            }
            Ok(())
        };
        self.decode_image(&mut ring, xs, h, true, &mut Some(&mut sink))?;
        if self.br.over {
            return Err(Error::Vp8lTruncated);
        }
        self.sc.ring = ring;
        self.sc.rows = rows;
        Ok(())
    }

    /// Takes the transforms back into the scratch, each kind's data to its slot.
    fn recycle(&mut self, mut ts: Vec<Transform>) {
        for t in ts.drain(..) {
            self.sc.tdata[usize::from(t.kind)] = t.data;
        }
        self.sc.ts = ts;
    }

    /// Reads the transforms and their images into `ts`, and returns the width the main image has
    /// once they are applied.
    fn read_transforms(
        &mut self,
        ts: &mut Vec<Transform>,
        w: usize,
        h: usize,
    ) -> Result<usize, Error> {
        let mut seen = [false; 4];
        let mut xs = w;
        while self.br.bit() {
            let kind = self.br.bits(2) as u8;
            let mut t = Transform {
                kind,
                bits: 0,
                xs,
                data: std::mem::take(&mut self.sc.tdata[usize::from(kind)]),
            };
            if seen[usize::from(kind)] {
                return Err(Error::Vp8lBitstream);
            }
            seen[usize::from(kind)] = true;
            match kind {
                T_PREDICTOR | T_CROSS => {
                    t.bits = self.br.bits(3) + 2;
                    let (tw, th) = (ceil(xs, t.bits), ceil(h, t.bits));
                    fit_zeroed(self.budget, &mut t.data, 4 * tw * th)?;
                    self.decode_image(&mut t.data, tw, th, false, &mut None)?;
                }
                T_INDEX => {
                    let n = self.br.bits(8) as usize + 1;
                    if n <= 2 {
                        t.bits = 3;
                    } else if n <= 4 {
                        t.bits = 2;
                    } else if n <= 16 {
                        t.bits = 1;
                    }
                    // The table pads to every index its packed pixels can hold.
                    fit_zeroed(self.budget, &mut t.data, 4 << (8 >> t.bits))?;
                    self.decode_image(&mut t.data[..4 * n], n, 1, false, &mut None)?;
                    for i in 4..4 * n {
                        t.data[i] = t.data[i].wrapping_add(t.data[i - 4]);
                    }
                    xs = ceil(xs, t.bits);
                }
                _ => {}
            }
            if self.br.over {
                return Err(Error::Vp8lTruncated);
            }
            ts.push(t);
        }
        Ok(xs)
    }

    /// Reads one prefix code over an alphabet of `lens.len()` symbols.
    fn read_code(&mut self, c: &mut Code, lens: &mut [u8], keep: bool) -> Result<(), Error> {
        const CL_ORDER: [usize; 19] = [
            17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let br = &mut self.br;
        lens.fill(0);
        if br.bit() {
            // simple
            let n = br.bits(1) + 1;
            let first = if br.bit() { 8 } else { 1 };
            let s = br.bits(first) as usize;
            if s < lens.len() {
                lens[s] = 1;
            }
            if n == 2 {
                let s = br.bits(8) as usize;
                if s < lens.len() {
                    lens[s] = 1;
                }
            }
        } else {
            let mut cl = [0u8; 19];
            let n = br.bits(4) as usize + 4;
            for &o in &CL_ORDER[..n] {
                cl[o] = br.bits(3) as u8;
            }
            let clc = &mut self.sc.clc;
            clc.build(&cl, None, true)?;
            let mut max_sym = lens.len();
            if br.bit() {
                let nb = 2 + 2 * br.bits(3);
                max_sym = 2 + br.bits(nb) as usize;
                if max_sym > lens.len() {
                    return Err(Error::Vp8lBitstream);
                }
            }
            let mut prev = 8u8;
            let mut s = 0;
            while s < lens.len() && max_sym > 0 {
                max_sym -= 1;
                if br.over {
                    return Err(Error::Vp8lTruncated);
                }
                let v = clc.read(br);
                if v < 16 {
                    lens[s] = v as u8;
                    s += 1;
                    if v != 0 {
                        prev = v as u8;
                    }
                    continue;
                }
                let (rep, val) = match v {
                    16 => (3 + br.bits(2) as usize, prev),
                    17 => (3 + br.bits(3) as usize, 0),
                    _ => (11 + br.bits(7) as usize, 0),
                };
                if s + rep > lens.len() {
                    return Err(Error::Vp8lBitstream);
                }
                lens[s..s + rep].fill(val);
                s += rep;
            }
        }
        if br.over {
            return Err(Error::Vp8lTruncated);
        }
        c.build(lens, Some(self.budget), keep)
    }

    /// Reads one entropy-coded image of xs x ys pixels into `a`. Only the main image has a meta
    /// prefix image. The meta image, the prefix codes and the colour cache are in the scratch's
    /// buffers; the meta image is read before the codes are taken, so its own image can use them.
    fn decode_image(
        &mut self,
        a: &mut [u8],
        xs: usize,
        ys: usize,
        main: bool,
        sink: &mut Sink<'_>,
    ) -> Result<(), Error> {
        // Only the main image feeds the sink.
        let mut sink = if main { sink.as_deref_mut() } else { None };
        let mut cache_bits = 0u32;
        if self.br.bit() {
            cache_bits = self.br.bits(4);
            if !(1..=11).contains(&cache_bits) {
                return Err(Error::Vp8lBitstream);
            }
        }
        let mut meta: Vec<u8> = Vec::new();
        let (mut mw, mut pbits) = (0usize, 0u32);
        let mut slot_of: Vec<i32> = Vec::new(); // group number to slot in groups, -1 for a group no pixel uses
        let (mut n_groups, mut n_slots) = (1usize, 1usize);
        let has_meta = main && self.br.bit();
        if has_meta {
            pbits = self.br.bits(3) + 2;
            mw = ceil(xs, pbits);
            let mh = ceil(ys, pbits);
            meta = std::mem::take(&mut self.sc.meta);
            fit_zeroed(self.budget, &mut meta, 4 * mw * mh)?;
            self.decode_image(&mut meta, mw, mh, false, &mut None)?;
            for i in 0..mw * mh {
                n_groups = n_groups.max(((get(&meta, i) >> 8) & 0xffff) as usize + 1);
            }
            slot_of = std::mem::take(&mut self.sc.slot_of);
            fit_zeroed(self.budget, &mut slot_of, n_groups)?;
            slot_of.fill(-1);
            n_slots = 0;
            for i in 0..mw * mh {
                let g = ((get(&meta, i) >> 8) & 0xffff) as usize;
                if slot_of[g] < 0 {
                    slot_of[g] = n_slots as i32;
                    n_slots += 1;
                }
                put(&mut meta, i, slot_of[g] as u32);
            }
        }
        let mut groups = std::mem::take(&mut self.sc.groups);
        let need = 5 * n_slots;
        // Codes past `need` from an earlier image keep their tables for later ones.
        if groups.len() < need {
            if groups.capacity() < need {
                let old = cap_bytes(&groups);
                self.budget.grow(CODE_BYTES * need as u64)?;
                groups
                    .try_reserve_exact(need - groups.len())
                    .map_err(|_| Error::OutOfMemory)?;
                self.budget.release(old);
            }
            groups.resize_with(need, Code::new);
        }
        let g0 = 256
            + 24
            + if cache_bits > 0 {
                1usize << cache_bits
            } else {
                0
            };
        // The alphabet size of each of a group's five codes, in stream order: green with
        // lengths and cache, red, blue, alpha, distance.
        let alpha = [g0, 256, 256, 256, 40];
        let mut lens = std::mem::take(&mut self.sc.lens);
        fit_zeroed(self.budget, &mut lens, g0)?;
        let mut skip = Code::new();
        for g in 0..n_groups {
            check_cancel(self.cancel)?;
            let slot = if has_meta { slot_of[g] } else { 0 };
            for k in 0..5 {
                let (c, keep) = if slot >= 0 {
                    (&mut groups[5 * slot as usize + k], true)
                } else {
                    (&mut skip, false)
                };
                self.read_code(c, &mut lens[..alpha[k]], keep)?;
            }
        }
        let mut cache = std::mem::take(&mut self.sc.cache);
        cache.clear();
        if cache_bits > 0 {
            fit_zeroed(self.budget, &mut cache, 1 << cache_bits)?;
        }
        let cshift = 32 - cache_bits;

        let total = xs * ys;
        let mask = (1usize << pbits) - 1;
        let (mut x, mut y, mut cur_y) = (0usize, 0usize, 0usize);
        let ring_n = a.len() / 4; // pixels a holds: all of them but for decode_alpha's ring
        let group_at = |x: usize, y: usize| -> usize {
            if has_meta {
                5 * get(&meta, (y >> pbits) * mw + (x >> pbits)) as usize
            } else {
                0
            }
        };
        let mut gb = group_at(x, y);
        let mut pos = 0usize;
        while pos < total {
            if self.br.over {
                return Err(Error::Vp8lTruncated);
            }
            while cur_y < y {
                check_cancel(self.cancel)?;
                if let Some(s) = sink.as_deref_mut() {
                    s(a, cur_y)?;
                }
                cur_y += 1;
            }
            let br = &mut self.br;
            let s = groups[gb].read(br);
            if s < 256 {
                let r = groups[gb + 1].read(br) as u32;
                let b = groups[gb + 2].read(br) as u32;
                let al = groups[gb + 3].read(br) as u32;
                let px = al << 24 | r << 16 | (s as u32) << 8 | b;
                put(a, slot(ring_n, pos), px);
                if !cache.is_empty() {
                    cache[(px.wrapping_mul(0x1e35a7bd) >> cshift) as usize] = px;
                }
                pos += 1;
                x += 1;
                if x == xs {
                    x = 0;
                    y += 1;
                }
                if x & mask == 0 && has_meta && pos < total {
                    gb = group_at(x, y);
                }
            } else if s < 256 + 24 {
                let n = prefix_value(br, s - 256);
                let dsym = groups[gb + 4].read(br);
                let dist = plane_dist(prefix_value(br, dsym), xs);
                if pos < dist || total - pos < n {
                    return Err(Error::Vp8lBitstream);
                }
                for _ in 0..n {
                    let px = get(a, slot(ring_n, pos - dist));
                    put(a, slot(ring_n, pos), px);
                    if !cache.is_empty() {
                        cache[(px.wrapping_mul(0x1e35a7bd) >> cshift) as usize] = px;
                    }
                    pos += 1;
                }
                x += n;
                while x >= xs {
                    x -= xs;
                    y += 1;
                }
                if pos < total && has_meta {
                    gb = group_at(x, y);
                }
            } else if s < 256 + 24 + cache.len() {
                put(a, slot(ring_n, pos), cache[s - 256 - 24]);
                pos += 1;
                x += 1;
                if x == xs {
                    x = 0;
                    y += 1;
                }
                if x & mask == 0 && has_meta && pos < total {
                    gb = group_at(x, y);
                }
            } else {
                return Err(Error::Vp8lBitstream);
            }
        }
        while sink.is_some() && cur_y < ys {
            if let Some(s) = sink.as_deref_mut() {
                s(a, cur_y)?;
            }
            cur_y += 1;
        }
        if has_meta {
            self.sc.meta = meta;
            self.sc.slot_of = slot_of;
        }
        self.sc.groups = groups;
        self.sc.lens = lens;
        self.sc.cache = cache;
        Ok(())
    }
}

/// A length or distance from its prefix symbol, plus its extra bits.
fn prefix_value(br: &mut BitReader<'_>, sym: usize) -> usize {
    if sym < 4 {
        return sym + 1;
    }
    let nb = ((sym - 2) >> 1) as u32;
    let off = (2 + (sym & 1)) << nb;
    off + br.bits(nb) as usize + 1
}

/// The pixel distance of distance code `c` in an image `xs` wide.
fn plane_dist(c: usize, xs: usize) -> usize {
    if c > 120 {
        return c - 120;
    }
    let e = DIST_MAP[c - 1];
    (i64::from(e[0]) + i64::from(e[1]) * xs as i64).max(1) as usize
}

/// The (x, y) offsets of distance codes 1 to 120, spec 5.2.2.
#[rustfmt::skip]
const DIST_MAP: [[i8; 2]; 120] = [
    [0, 1], [1, 0], [1, 1], [-1, 1], [0, 2], [2, 0], [1, 2], [-1, 2],
    [2, 1], [-2, 1], [2, 2], [-2, 2], [0, 3], [3, 0], [1, 3], [-1, 3],
    [3, 1], [-3, 1], [2, 3], [-2, 3], [3, 2], [-3, 2], [0, 4], [4, 0],
    [1, 4], [-1, 4], [4, 1], [-4, 1], [3, 3], [-3, 3], [2, 4], [-2, 4],
    [4, 2], [-4, 2], [0, 5], [3, 4], [-3, 4], [4, 3], [-4, 3], [5, 0],
    [1, 5], [-1, 5], [5, 1], [-5, 1], [2, 5], [-2, 5], [5, 2], [-5, 2],
    [4, 4], [-4, 4], [3, 5], [-3, 5], [5, 3], [-5, 3], [0, 6], [6, 0],
    [1, 6], [-1, 6], [6, 1], [-6, 1], [2, 6], [-2, 6], [6, 2], [-6, 2],
    [4, 5], [-4, 5], [5, 4], [-5, 4], [3, 6], [-3, 6], [6, 3], [-6, 3],
    [0, 7], [7, 0], [1, 7], [-1, 7], [5, 5], [-5, 5], [7, 1], [-7, 1],
    [4, 6], [-4, 6], [6, 4], [-6, 4], [2, 7], [-2, 7], [7, 2], [-7, 2],
    [3, 7], [-3, 7], [7, 3], [-7, 3], [5, 6], [-5, 6], [6, 5], [-6, 5],
    [8, 0], [4, 7], [-4, 7], [7, 4], [-7, 4], [8, 1], [8, 2], [6, 6],
    [-6, 6], [8, 3], [5, 7], [-5, 7], [7, 5], [-7, 5], [8, 4], [6, 7],
    [-6, 7], [7, 6], [-7, 6], [8, 5], [7, 7], [-7, 7], [8, 6], [8, 7],
];
