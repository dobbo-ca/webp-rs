//! The prediction workspace and intra predictors, from golang.org/x/image/vp8 predfunc.go and
//! reconstruct.go (BSD-3-Clause, NOTICE).

use crate::simd::enc_pred;

/// Y at (1, 8), U at (18, 8), V at (18, 24).
pub(crate) type Ybr = [[u8; 32]; 26];

/// The source planes of one frame, padded to the macroblock grid.
#[derive(Default)]
pub(crate) struct Planes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub ys: usize,
    pub cs: usize,
    pub w: usize,
    pub h: usize,
}

/// Mode numbers double as implied 4x4 submodes.
pub(crate) const PRED_DC: u8 = 0;
pub(crate) const PRED_TM: u8 = 1;
pub(crate) const PRED_VE: u8 = 2;
pub(crate) const PRED_HE: u8 = 3;
pub(crate) const PRED_RD: u8 = 4;
pub(crate) const PRED_VR: u8 = 5;
pub(crate) const PRED_LD: u8 = 6;
pub(crate) const PRED_VL: u8 = 7;
pub(crate) const PRED_HD: u8 = 8;
pub(crate) const PRED_HU: u8 = 9;

pub(crate) fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The pixels above a macroblock that its predictions read: the luma row and the four pixels to
/// its top right (the row's last pixel repeated in the last column), and the chroma rows.
#[derive(Clone, Copy)]
pub(crate) struct Above {
    pub y: [u8; 20],
    pub u: [u8; 8],
    pub v: [u8; 8],
}

/// Fills the edges from the planes `f`, x/image prepareYBR.
pub(crate) fn prepare(z: &mut Ybr, f: &Planes, mbx: usize, mby: usize, mbw: usize) {
    let above = (mby > 0).then(|| {
        let yoff = (16 * mby - 1) * f.ys + 16 * mbx;
        let coff = (8 * mby - 1) * f.cs + 8 * mbx;
        let mut a = Above {
            y: [0; 20],
            u: [0; 8],
            v: [0; 8],
        };
        a.y[..16].copy_from_slice(&f.y[yoff..yoff + 16]);
        if mbx == mbw - 1 {
            a.y[16..].fill(f.y[yoff + 15]);
        } else {
            a.y[16..].copy_from_slice(&f.y[yoff + 16..yoff + 20]);
        }
        a.u.copy_from_slice(&f.cb[coff..coff + 8]);
        a.v.copy_from_slice(&f.cr[coff..coff + 8]);
        a
    });
    prepare_above(z, mbx, above.as_ref());
}

/// Fills the edges: the left ones from the macroblock before in z, the top ones from `above`
/// (`None` in the first row).
pub(crate) fn prepare_above(z: &mut Ybr, mbx: usize, above: Option<&Above>) {
    if mbx == 0 {
        for row in z.iter_mut().take(17) {
            row[7] = 0x81;
        }
        for row in z.iter_mut().skip(17) {
            row[7] = 0x81;
            row[23] = 0x81;
        }
    } else {
        for row in z.iter_mut().take(17) {
            row[7] = row[7 + 16];
        }
        for row in z.iter_mut().skip(17) {
            row[7] = row[15];
            row[23] = row[31];
        }
    }
    match above {
        None => {
            for x in 7..28 {
                z[0][x] = 0x7f;
            }
            for x in 7..16 {
                z[17][x] = 0x7f;
            }
            for x in 23..32 {
                z[17][x] = 0x7f;
            }
        }
        Some(a) => {
            z[0][8..28].copy_from_slice(&a.y);
            z[17][8..16].copy_from_slice(&a.u);
            z[17][24..32].copy_from_slice(&a.v);
        }
    }
    let rt: [u8; 4] = [z[0][24], z[0][25], z[0][26], z[0][27]];
    for y in [4, 8, 12] {
        z[y][24..28].copy_from_slice(&rt);
    }
}

/// The DC prediction value of an n by n block.
fn dc_avg<const N: usize>(z: &Ybr, y: usize, x: usize, mbx: usize, mby: usize) -> u8 {
    let log2n = if N == 16 { 4 } else { 3 };
    let top = || -> usize { z[y - 1][x..x + N].iter().map(|&v| usize::from(v)).sum() };
    let left = || -> usize { (0..N).map(|j| usize::from(z[y + j][x - 1])).sum() };
    let avg = if mbx == 0 && mby == 0 {
        128
    } else if mby == 0 {
        (left() + N / 2) >> log2n
    } else if mbx == 0 {
        (top() + N / 2) >> log2n
    } else {
        (top() + left() + N) >> (log2n + 1)
    };
    avg as u8
}

/// Predicts an n by n block, n 16 or 8, with DC, TM, VE or HE.
pub(crate) fn pred_mb(z: &mut Ybr, n: usize, y: usize, x: usize, mode: u8, mbx: usize, mby: usize) {
    if n == 16 {
        pred_mb_n::<16>(z, y, x, mode, mbx, mby);
    } else {
        pred_mb_n::<8>(z, y, x, mode, mbx, mby);
    }
}

fn pred_mb_n<const N: usize>(z: &mut Ybr, y: usize, x: usize, mode: u8, mbx: usize, mby: usize) {
    match mode {
        PRED_DC => {
            let avg = dc_avg::<N>(z, y, x, mbx, mby);
            for row in &mut z[y..y + N] {
                row[x..x + N].fill(avg);
            }
        }
        PRED_TM => pred_tm_n::<N>(z, y, x),
        PRED_VE => {
            let top: [u8; N] = std::array::from_fn(|i| z[y - 1][x + i]);
            for row in &mut z[y..y + N] {
                row[x..x + N].copy_from_slice(&top);
            }
        }
        PRED_HE => {
            for row in &mut z[y..y + N] {
                let v = row[x - 1];
                row[x..x + N].fill(v);
            }
        }
        _ => {}
    }
}

/// The SSE of the DC, TM, VE and HE predictions of src against an n by n block, leaving z unchanged.
pub(crate) fn edge_sse(
    z: &Ybr,
    src: &[u8],
    stride: usize,
    n: usize,
    y: usize,
    x: usize,
    mbx: usize,
    mby: usize,
) -> [i64; 4] {
    let edges = u8::from(mbx != 0) | (u8::from(mby != 0) << 1);
    enc_pred::edge_sse(src, stride, n, z, y, x, edges).map(i64::from)
}

/// TrueMotion prediction of an n by n block.
fn pred_tm_n<const N: usize>(z: &mut Ybr, y: usize, x: usize) {
    let a = i32::from(z[y - 1][x - 1]);
    let top: [u8; N] = std::array::from_fn(|i| z[y - 1][x + i]);
    for row in &mut z[y..y + N] {
        let d = i32::from(row[x - 1]) - a;
        for (p, &t) in row[x..x + N].iter_mut().zip(&top) {
            *p = clip8(d + i32::from(t));
        }
    }
}

/// All ten 4x4 predictions at (y, x) as little-endian rows, gathering the edges once.
pub(crate) fn pred4_all(z: &Ybr, pr: &mut [[u32; 4]; 10], y: usize, x: usize) {
    // The corner and the eight pixels above, then the four to the left.
    let mut e = [0u8; 16];
    e[..9].copy_from_slice(&z[y - 1][x - 1..x + 8]);
    for (j, v) in e[9..13].iter_mut().enumerate() {
        *v = z[y + j][x - 1];
    }
    enc_pred::pred4_all(&e, pr);
}

/// Writes a saved 4x4 prediction at (y, x).
pub(crate) fn put4(z: &mut Ybr, p: &[u32; 4], y: usize, x: usize) {
    for (j, v) in p.iter().enumerate() {
        z[y + j][x..x + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// The SSE between src and the w by h prediction at (y, x).
pub(crate) fn pred_sse(
    z: &Ybr,
    src: &[u8],
    stride: usize,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
) -> i64 {
    i64::from(enc_pred::sse_block(src, stride, z, y, x, w, h))
}

pub(crate) use super::transform::idct_add;
