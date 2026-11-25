//! A minimal PNG decoder for the test fixtures: 8-bit gray, RGB and RGBA, no interlacing.
#![allow(dead_code)]

pub struct Png {
    pub width: usize,
    pub height: usize,
    /// 1 (gray), 3 (RGB) or 4 (RGBA).
    pub channels: usize,
    pub pix: Vec<u8>,
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bits(&mut self, n: usize) -> u32 {
        let mut v = 0;
        for i in 0..n {
            let b = (self.d[self.pos >> 3] >> (self.pos & 7)) & 1;
            v |= u32::from(b) << i;
            self.pos += 1;
        }
        v
    }
}

struct Huff {
    count: [u16; 16],
    symbol: Vec<u16>,
}

fn huff(lengths: &[u8]) -> Huff {
    let mut count = [0u16; 16];
    for &l in lengths {
        count[l as usize] += 1;
    }
    let mut offs = [0u16; 16];
    for i in 1..15 {
        offs[i + 1] = offs[i] + count[i];
    }
    let mut symbol = vec![0u16; lengths.len()];
    for (s, &l) in lengths.iter().enumerate() {
        if l != 0 {
            symbol[offs[l as usize] as usize] = s as u16;
            offs[l as usize] += 1;
        }
    }
    Huff { count, symbol }
}

fn decode(b: &mut Bits, h: &Huff) -> usize {
    let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
    for len in 1..16 {
        code |= b.bits(1) as i32;
        let count = i32::from(h.count[len]);
        if code - count < first {
            return h.symbol[(index + (code - first)) as usize] as usize;
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    panic!("bad huffman code");
}

const LBASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEXT: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DEXT: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn inflate(d: &[u8]) -> Vec<u8> {
    let mut b = Bits { d, pos: 0 };
    let mut out = Vec::new();
    loop {
        let last = b.bits(1);
        match b.bits(2) {
            0 => {
                b.pos = (b.pos + 7) & !7;
                let n = b.bits(16) as usize;
                b.bits(16);
                let s = b.pos >> 3;
                out.extend_from_slice(&d[s..s + n]);
                b.pos += 8 * n;
            }
            t => {
                let (lit, dist) = if t == 1 {
                    let mut l = [8u8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (huff(&l), huff(&[5u8; 30]))
                } else {
                    let (nlen, ndist, ncode) = (
                        b.bits(5) as usize + 257,
                        b.bits(5) as usize + 1,
                        b.bits(4) as usize + 4,
                    );
                    const ORDER: [usize; 19] = [
                        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                    ];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(ncode) {
                        cl[o] = b.bits(3) as u8;
                    }
                    let ch = huff(&cl);
                    let mut lens = vec![0u8; nlen + ndist];
                    let mut i = 0;
                    while i < nlen + ndist {
                        let s = decode(&mut b, &ch);
                        if s < 16 {
                            lens[i] = s as u8;
                            i += 1;
                        } else {
                            let (prev, rep) = match s {
                                16 => (lens[i - 1], 3 + b.bits(2)),
                                17 => (0, 3 + b.bits(3)),
                                _ => (0, 11 + b.bits(7)),
                            };
                            for _ in 0..rep {
                                lens[i] = prev;
                                i += 1;
                            }
                        }
                    }
                    (huff(&lens[..nlen]), huff(&lens[nlen..]))
                };
                loop {
                    let s = decode(&mut b, &lit);
                    if s < 256 {
                        out.push(s as u8);
                    } else if s == 256 {
                        break;
                    } else {
                        let s = s - 257;
                        let len = LBASE[s] as usize + b.bits(LEXT[s] as usize) as usize;
                        let ds = decode(&mut b, &dist);
                        let dd = DBASE[ds] as usize + b.bits(DEXT[ds] as usize) as usize;
                        for _ in 0..len {
                            out.push(out[out.len() - dd]);
                        }
                    }
                }
            }
        }
        if last == 1 {
            return out;
        }
    }
}

pub fn decode_png(data: &[u8]) -> Png {
    assert_eq!(&data[..8], b"\x89PNG\r\n\x1a\n");
    let (mut pos, mut idat) = (8, Vec::new());
    let (mut w, mut h, mut ct) = (0, 0, 0);
    while pos < data.len() {
        let n = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let tag = &data[pos + 4..pos + 8];
        let body = &data[pos + 8..pos + 8 + n];
        match tag {
            b"IHDR" => {
                w = u32::from_be_bytes(body[0..4].try_into().unwrap()) as usize;
                h = u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize;
                assert_eq!(body[8], 8, "bit depth");
                assert_eq!(body[12], 0, "interlaced");
                ct = body[9];
            }
            b"IDAT" => idat.extend_from_slice(body),
            _ => {}
        }
        pos += 12 + n;
    }
    let channels = match ct {
        0 => 1,
        2 => 3,
        6 => 4,
        _ => panic!("colour type {ct}"),
    };
    let raw = inflate(&idat[2..]);
    let stride = w * channels;
    let mut pix = vec![0u8; stride * h];
    for y in 0..h {
        let f = raw[y * (stride + 1)];
        let line = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        for x in 0..stride {
            let a = if x >= channels {
                pix[y * stride + x - channels]
            } else {
                0
            };
            let b = if y > 0 { pix[(y - 1) * stride + x] } else { 0 };
            let c = if x >= channels && y > 0 {
                pix[(y - 1) * stride + x - channels]
            } else {
                0
            };
            let pred = match f {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                4 => {
                    let p = i32::from(a) + i32::from(b) - i32::from(c);
                    let (pa, pb, pc) = (
                        (p - i32::from(a)).abs(),
                        (p - i32::from(b)).abs(),
                        (p - i32::from(c)).abs(),
                    );
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                _ => panic!("filter {f}"),
            };
            pix[y * stride + x] = line[x].wrapping_add(pred);
        }
    }
    Png {
        width: w,
        height: h,
        channels,
        pix,
    }
}
