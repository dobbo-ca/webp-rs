//! The simple VP8L writer for ALPH chunks (webp-go's internal/vp8l/vp8l.go): no transforms, no
//! colour cache, one prefix-code group and LZ77 copies from the left or the row above. The output
//! is lossless, so it decodes to the same alpha as libwebp's, though the bytes differ.

use super::refs::CHECK_EVERY;
use super::{BitWriter, prefix_bits};
use crate::Error;
use crate::gosort::sort_func;
use crate::limits::Ledger;

const NUM_LEN: usize = 24; // length prefix codes
const NUM_GREEN: usize = 256 + NUM_LEN;
const NUM_DIST: usize = 40;
const MAX_LEN: usize = 4096;
const MIN_COPY: usize = 3;
const DIST_ABOVE: usize = 1; // plane code (0, 1)
const DIST_LEFT: usize = 2; // plane code (1, 0)

/// kCodeLengthCodeOrder, spec section 3.7.2.1.2.
const CL_ORDER: [u8; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// A canonical prefix code; bits are stored reversed.
struct Code {
    len: [u8; NUM_GREEN],
    bits: [u16; NUM_GREEN],
    n: usize,
    /// Symbols with a nonzero length.
    used: usize,
}

impl Code {
    fn new() -> Code {
        Code {
            len: [0; NUM_GREEN],
            bits: [0; NUM_GREEN],
            n: 0,
            used: 0,
        }
    }

    /// Makes lengths up to `limit` from counts, then the codes.
    fn build(&mut self, counts: &[u32], limit: usize) {
        self.n = counts.len();
        self.len.fill(0);
        huff_lens(counts, &mut self.len[..self.n], limit);
        self.used = self.len[..self.n].iter().filter(|&&l| l != 0).count();
        let mut next = [0u16; 16];
        let mut cnt = [0u16; 16];
        for &l in &self.len[..self.n] {
            cnt[l as usize] += 1;
        }
        cnt[0] = 0;
        let mut v = 0u16;
        for l in 1..16 {
            v = (v + cnt[l - 1]) << 1;
            next[l] = v;
        }
        for s in 0..self.n {
            let l = self.len[s];
            if l == 0 {
                continue;
            }
            let mut v = next[l as usize];
            next[l as usize] += 1;
            let mut r = 0u16;
            for _ in 0..l {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            self.bits[s] = r;
        }
    }

    /// Writes symbol s; a one-symbol code takes no bits.
    fn put(&self, b: &mut BitWriter, s: usize) -> Result<(), Error> {
        if self.used > 1 {
            b.put(u64::from(self.bits[s]), i32::from(self.len[s]))?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Node {
    w: u32,
    /// Children, -1 for a leaf.
    left: i16,
    right: i16,
    sym: i16,
}

/// Sets Huffman code lengths of at most `limit` bits into `lens`. Zero counts get length 0; a lone
/// symbol gets 1, which decoders read as a zero-bit code. Halves counts until the tree fits.
fn huff_lens(counts: &[u32], lens: &mut [u8], limit: usize) {
    let mut c = [0u32; NUM_GREEN];
    c[..counts.len()].copy_from_slice(counts);
    loop {
        let mut nodes = [Node {
            w: 0,
            left: -1,
            right: -1,
            sym: -1,
        }; 2 * NUM_GREEN];
        let mut n = 0;
        for (s, &w) in c[..counts.len()].iter().enumerate() {
            if w > 0 {
                nodes[n] = Node {
                    w,
                    left: -1,
                    right: -1,
                    sym: s as i16,
                };
                n += 1;
            }
        }
        match n {
            0 => return,
            1 => {
                lens[nodes[0].sym as usize] = 1;
                return;
            }
            _ => {}
        }
        sort_func(&mut nodes[..n], |a, b| {
            (i64::from(a.w) - i64::from(b.w)).cmp(&0)
        });
        // Two-queue build: leaves sorted, internal nodes made in order.
        let leaves = n;
        let (mut li, mut qi) = (0usize, leaves);
        while n < 2 * leaves - 1 {
            let mut pick = |nodes: &[Node], n: usize| {
                if li < leaves && (qi >= n || nodes[li].w <= nodes[qi].w) {
                    li += 1;
                    li - 1
                } else {
                    qi += 1;
                    qi - 1
                }
            };
            let x = pick(&nodes, n);
            let y = pick(&nodes, n);
            let w = nodes[x].w + nodes[y].w;
            nodes[n] = Node {
                w,
                left: x as i16,
                right: y as i16,
                sym: -1,
            };
            n += 1;
        }
        let mut depth = [0u8; 2 * NUM_GREEN];
        let mut deep = 0usize;
        for i in (0..n).rev() {
            if nodes[i].left >= 0 {
                depth[nodes[i].left as usize] = depth[i] + 1;
                depth[nodes[i].right as usize] = depth[i] + 1;
            } else {
                deep = deep.max(depth[i] as usize);
            }
        }
        if deep <= limit {
            for i in 0..leaves {
                lens[nodes[i].sym as usize] = depth[i];
            }
            return;
        }
        for s in c[..counts.len()].iter_mut() {
            if *s > 0 {
                *s = (*s >> 1).max(1);
            }
        }
    }
}

/// A literal or a copy (n pixels, d back).
enum Op {
    Lit(u8),
    Copy(usize, usize),
}

/// Greedy left or above copies.
fn walk(
    cx: &Ledger,
    a: &[u8],
    w: usize,
    f: &mut impl FnMut(Op) -> Result<(), Error>,
) -> Result<(), Error> {
    let n = a.len();
    let mut i = 0;
    let mut next_check = 0;
    while i < n {
        if i >= next_check {
            cx.check()?;
            next_check = i + CHECK_EVERY;
        }
        let (mut best, mut dist) = (0, 0);
        if i >= 1 {
            best = run(a, i, 1, n);
            dist = DIST_LEFT;
        }
        if i >= w {
            let k = run(a, i, w, n);
            if k > best {
                best = k;
                dist = DIST_ABOVE;
            }
        }
        if best >= MIN_COPY {
            f(Op::Copy(best, dist))?;
            i += best;
            continue;
        }
        f(Op::Lit(a[i]))?;
        i += 1;
    }
    Ok(())
}

/// How many pixels from i repeat those d back, up to MAX_LEN.
fn run(a: &[u8], i: usize, d: usize, n: usize) -> usize {
    let mut k = 0;
    while k < MAX_LEN && i + k < n && a[i + k] == a[i + k - d] {
        k += 1;
    }
    k
}

/// Writes ALPH image streams.
pub(crate) struct Legacy<'a> {
    cx: &'a Ledger<'a>,
    bw: BitWriter<'a>,
    green: Code,
    dist: Code,
    cl: Code,
    cl_syms: Vec<u16>,
}

impl<'a> Legacy<'a> {
    #[cfg(test)]
    pub(crate) fn new(cx: &'a Ledger<'a>) -> Legacy<'a> {
        Legacy::with_syms(cx, Vec::new())
    }

    /// A writer that uses `cl_syms` for the code-length symbols, so a kept list is reused.
    pub(crate) fn with_syms(cx: &'a Ledger<'a>, cl_syms: Vec<u16>) -> Legacy<'a> {
        Legacy {
            cx,
            bw: BitWriter::default(),
            green: Code::new(),
            dist: Code::new(),
            cl: Code::new(),
            cl_syms,
        }
    }

    /// The code-length symbol list, to keep for the next writer.
    pub(crate) fn into_syms(self) -> Vec<u16> {
        self.cl_syms
    }

    /// Appends the image stream of `a` (w wide, stride w) to `dst`.
    pub(crate) fn alpha(&mut self, dst: Vec<u8>, a: &[u8], w: usize) -> Result<Vec<u8>, Error> {
        self.alpha_in(dst, a, w)
    }

    fn alpha_in(&mut self, dst: Vec<u8>, a: &[u8], w: usize) -> Result<Vec<u8>, Error> {
        let cx = self.cx;
        let mut gh = [0u32; NUM_GREEN];
        let mut dh = [0u32; NUM_DIST];
        walk(cx, a, w, &mut |op| {
            match op {
                Op::Lit(v) => gh[v as usize] += 1,
                Op::Copy(n, d) => {
                    gh[256 + prefix_bits(n).0] += 1;
                    dh[prefix_bits(d).0] += 1;
                }
            }
            Ok(())
        })?;
        self.green.build(&gh, 15);
        self.dist.build(&dh, 15);

        self.bw = BitWriter {
            buf: dst,
            acc: 0,
            nacc: 0,
            cx: Some(cx),
        };
        self.bw.put(0, 1)?; // no transform
        self.bw.put(0, 1)?; // no color cache
        self.bw.put(0, 1)?; // no meta prefix codes
        self.write_code(true)?;
        for _ in 0..3 {
            // red, blue, alpha: always 0
            self.bw.put(1, 1)?; // simple code
            self.bw.put(0, 1)?; // one symbol
            self.bw.put(0, 1)?; // 1-bit symbol
            self.bw.put(0, 1)?; // symbol 0
        }
        self.write_code(false)?;
        let (b, green, dist) = (&mut self.bw, &self.green, &self.dist);
        walk(cx, a, w, &mut |op| match op {
            Op::Lit(v) => green.put(b, v as usize),
            Op::Copy(n, d) => {
                let (p, xb, xv) = prefix_bits(n);
                green.put(b, 256 + p)?;
                b.put(xv as u64, xb as i32)?;
                let (p, xb, xv) = prefix_bits(d);
                dist.put(b, p)?;
                b.put(xv as u64, xb as i32)
            }
        })?;
        self.bw.flush()
    }

    /// Writes the green code (`green`) or the distance code, as a simple code when it has at most
    /// one symbol, else as a normal code with run-length coded lengths.
    fn write_code(&mut self, green: bool) -> Result<(), Error> {
        let c = if green { &self.green } else { &self.dist };
        let b = &mut self.bw;
        if c.used <= 1 {
            let s = c.len[..c.n].iter().position(|&l| l != 0).unwrap_or(0);
            b.put(1, 1)?; // simple code
            b.put(0, 1)?; // one symbol
            if s < 2 {
                b.put(0, 1)?;
                b.put(s as u64, 1)?;
            } else {
                b.put(1, 1)?;
                b.put(s as u64, 8)?;
            }
            return Ok(());
        }
        let syms = &mut self.cl_syms;
        syms.clear();
        let mut prev = 8u8;
        let lens = &c.len[..c.n];
        let mut i = 0;
        while i < lens.len() {
            let v = lens[i];
            let mut r = 1;
            while i + r < lens.len() && lens[i + r] == v {
                r += 1;
            }
            i += r;
            if v == 0 {
                while r > 0 {
                    if r >= 11 {
                        let k = r.min(138);
                        syms.push(18 | (((k - 11) as u16) << 8));
                        r -= k;
                    } else if r >= 3 {
                        let k = r.min(10);
                        syms.push(17 | (((k - 3) as u16) << 8));
                        r -= k;
                    } else {
                        syms.push(0);
                        r -= 1;
                    }
                }
                continue;
            }
            if v != prev {
                syms.push(u16::from(v));
                prev = v;
                r -= 1;
            }
            while r >= 3 {
                let k = r.min(6);
                syms.push(16 | (((k - 3) as u16) << 8));
                r -= k;
            }
            for _ in 0..r {
                syms.push(u16::from(v));
            }
        }

        let mut ch = [0u32; 19];
        for &s in syms.iter() {
            ch[(s & 0xff) as usize] += 1;
        }
        self.cl.build(&ch, 7);
        let mut k = 4;
        for (i, &s) in CL_ORDER.iter().enumerate() {
            if self.cl.len[s as usize] != 0 {
                k = k.max(i + 1);
            }
        }
        b.put(0, 1)?; // normal code
        b.put((k - 4) as u64, 4)?;
        for &s in &CL_ORDER[..k] {
            b.put(u64::from(self.cl.len[s as usize]), 3)?;
        }
        b.put(0, 1)?; // max_symbol is the alphabet size
        for &s in syms.iter() {
            self.cl.put(b, (s & 0xff) as usize)?;
            match s & 0xff {
                16 => b.put(u64::from(s >> 8), 2)?,
                17 => b.put(u64::from(s >> 8), 3)?,
                18 => b.put(u64::from(s >> 8), 7)?,
                _ => {}
            }
        }
        Ok(())
    }
}
