//! Intra prediction, inverse transforms and reconstruction of one macroblock.

use super::{B_DC, B_HD, B_HE, B_HU, B_LD, B_RD, B_TM, B_VE, B_VL, B_VR, Dec, Mb};

// The work buffer holds one macroblock with its neighbours, as libwebp's yuv_b: 32 bytes a row,
// the row above and the column to the left at -1.
pub(super) const BPS: usize = 32;
pub(super) const Y_OFF: usize = BPS + 8;
pub(super) const U_OFF: usize = Y_OFF + BPS * 16 + BPS;
pub(super) const V_OFF: usize = U_OFF + 16;

/// Offset of 4x4 block n of the luma macroblock in the work buffer.
fn scan(n: usize) -> usize {
    (n & 3) * 4 + (n >> 2) * 4 * BPS
}

pub(super) fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn avg2(a: u8, b: u8) -> u8 {
    ((i32::from(a) + i32::from(b) + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((i32::from(a) + 2 * i32::from(b) + i32::from(c) + 2) >> 2) as u8
}

fn true_motion(b: &mut [u8], o: usize, size: usize) {
    let tl = i32::from(b[o - BPS - 1]);
    for y in 0..size {
        let l = i32::from(b[o + y * BPS - 1]) - tl;
        for x in 0..size {
            b[o + y * BPS + x] = clip8(i32::from(b[o - BPS + x]) + l);
        }
    }
}

fn fill(b: &mut [u8], o: usize, size: usize, v: u8) {
    for y in 0..size {
        b[o + y * BPS..o + y * BPS + size].fill(v);
    }
}

/// Predicts a size x size block (16 luma, 8 chroma) at o; `no_top` and `no_left` say that edge is
/// the frame's, which only DC cares about.
fn pred(b: &mut [u8], o: usize, size: usize, mode: u8, no_top: bool, no_left: bool) {
    match mode {
        B_DC => {
            let (mut sum, mut n) = (0usize, 0usize);
            if !no_top {
                for i in 0..size {
                    sum += usize::from(b[o - BPS + i]);
                }
                n += size;
            }
            if !no_left {
                for i in 0..size {
                    sum += usize::from(b[o + i * BPS - 1]);
                }
                n += size;
            }
            let dc = (sum + n / 2).checked_div(n).map_or(0x80, |v| v as u8);
            fill(b, o, size, dc);
        }
        B_TM => true_motion(b, o, size),
        B_VE => {
            for y in 0..size {
                b.copy_within(o - BPS..o - BPS + size, o + y * BPS);
            }
        }
        B_HE => {
            for y in 0..size {
                let v = b[o + y * BPS - 1];
                b[o + y * BPS..o + y * BPS + size].fill(v);
            }
        }
        _ => {}
    }
}

/// Predicts the 4x4 block at o from the 13 pixels around it.
fn pred4(b: &mut [u8], o: usize, mode: u8) {
    let x_ = b[o - BPS - 1];
    let (i_, j, k, l) = (
        b[o - 1],
        b[o - 1 + BPS],
        b[o - 1 + 2 * BPS],
        b[o - 1 + 3 * BPS],
    );
    let t = &b[o - BPS..o - BPS + 8];
    let (a, bb, c, d, e, f, g, h) = (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    let mut dst = |x: usize, y: usize, v: u8| b[o + x + y * BPS] = v;
    match mode {
        B_DC => {
            let dc: i32 = [a, bb, c, d, i_, j, k, l]
                .iter()
                .map(|&v| i32::from(v))
                .sum::<i32>()
                + 4;
            fill(b, o, 4, (dc >> 3) as u8);
        }
        B_TM => true_motion(b, o, 4),
        B_VE => {
            let v = [
                avg3(x_, a, bb),
                avg3(a, bb, c),
                avg3(bb, c, d),
                avg3(c, d, e),
            ];
            for y in 0..4 {
                b[o + y * BPS..o + y * BPS + 4].copy_from_slice(&v);
            }
        }
        B_HE => {
            let r = [
                avg3(x_, i_, j),
                avg3(i_, j, k),
                avg3(j, k, l),
                avg3(k, l, l),
            ];
            for y in 0..4 {
                b[o + y * BPS..o + y * BPS + 4].fill(r[y]);
            }
        }
        B_RD => {
            dst(0, 3, avg3(j, k, l));
            let v = avg3(i_, j, k);
            dst(1, 3, v);
            dst(0, 2, v);
            let v = avg3(x_, i_, j);
            dst(2, 3, v);
            dst(1, 2, v);
            dst(0, 1, v);
            let v = avg3(a, x_, i_);
            dst(3, 3, v);
            dst(2, 2, v);
            dst(1, 1, v);
            dst(0, 0, v);
            let v = avg3(bb, a, x_);
            dst(3, 2, v);
            dst(2, 1, v);
            dst(1, 0, v);
            let v = avg3(c, bb, a);
            dst(3, 1, v);
            dst(2, 0, v);
            dst(3, 0, avg3(d, c, bb));
        }
        B_VR => {
            let v = avg2(x_, a);
            dst(0, 0, v);
            dst(1, 2, v);
            let v = avg2(a, bb);
            dst(1, 0, v);
            dst(2, 2, v);
            let v = avg2(bb, c);
            dst(2, 0, v);
            dst(3, 2, v);
            dst(3, 0, avg2(c, d));
            dst(0, 3, avg3(k, j, i_));
            dst(0, 2, avg3(j, i_, x_));
            let v = avg3(i_, x_, a);
            dst(0, 1, v);
            dst(1, 3, v);
            let v = avg3(x_, a, bb);
            dst(1, 1, v);
            dst(2, 3, v);
            let v = avg3(a, bb, c);
            dst(2, 1, v);
            dst(3, 3, v);
            dst(3, 1, avg3(bb, c, d));
        }
        B_LD => {
            dst(0, 0, avg3(a, bb, c));
            let v = avg3(bb, c, d);
            dst(1, 0, v);
            dst(0, 1, v);
            let v = avg3(c, d, e);
            dst(2, 0, v);
            dst(1, 1, v);
            dst(0, 2, v);
            let v = avg3(d, e, f);
            dst(3, 0, v);
            dst(2, 1, v);
            dst(1, 2, v);
            dst(0, 3, v);
            let v = avg3(e, f, g);
            dst(3, 1, v);
            dst(2, 2, v);
            dst(1, 3, v);
            let v = avg3(f, g, h);
            dst(3, 2, v);
            dst(2, 3, v);
            dst(3, 3, avg3(g, h, h));
        }
        B_VL => {
            dst(0, 0, avg2(a, bb));
            let v = avg2(bb, c);
            dst(1, 0, v);
            dst(0, 2, v);
            let v = avg2(c, d);
            dst(2, 0, v);
            dst(1, 2, v);
            let v = avg2(d, e);
            dst(3, 0, v);
            dst(2, 2, v);
            dst(0, 1, avg3(a, bb, c));
            let v = avg3(bb, c, d);
            dst(1, 1, v);
            dst(0, 3, v);
            let v = avg3(c, d, e);
            dst(2, 1, v);
            dst(1, 3, v);
            let v = avg3(d, e, f);
            dst(3, 1, v);
            dst(2, 3, v);
            dst(3, 2, avg3(e, f, g));
            dst(3, 3, avg3(f, g, h));
        }
        B_HD => {
            let v = avg2(i_, x_);
            dst(0, 0, v);
            dst(2, 1, v);
            let v = avg2(j, i_);
            dst(0, 1, v);
            dst(2, 2, v);
            let v = avg2(k, j);
            dst(0, 2, v);
            dst(2, 3, v);
            dst(0, 3, avg2(l, k));
            dst(3, 0, avg3(a, bb, c));
            dst(2, 0, avg3(x_, a, bb));
            let v = avg3(i_, x_, a);
            dst(1, 0, v);
            dst(3, 1, v);
            let v = avg3(j, i_, x_);
            dst(1, 1, v);
            dst(3, 2, v);
            let v = avg3(k, j, i_);
            dst(1, 2, v);
            dst(3, 3, v);
            dst(1, 3, avg3(l, k, j));
        }
        B_HU => {
            dst(0, 0, avg2(i_, j));
            let v = avg2(j, k);
            dst(2, 0, v);
            dst(0, 1, v);
            let v = avg2(k, l);
            dst(2, 1, v);
            dst(0, 2, v);
            dst(1, 0, avg3(i_, j, k));
            let v = avg3(j, k, l);
            dst(3, 0, v);
            dst(1, 1, v);
            let v = avg3(k, l, l);
            dst(3, 1, v);
            dst(1, 2, v);
            for (x, y) in [(3, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)] {
                dst(x, y, l);
            }
        }
        _ => {}
    }
}

/// Inverts the luma DC transform into the DC of each of 16 blocks.
pub(super) fn wht(inp: &[i16; 16], out: &mut [i16; 384]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = i32::from(inp[i]) + i32::from(inp[12 + i]);
        let a1 = i32::from(inp[4 + i]) + i32::from(inp[8 + i]);
        let a2 = i32::from(inp[4 + i]) - i32::from(inp[8 + i]);
        let a3 = i32::from(inp[i]) - i32::from(inp[12 + i]);
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for i in 0..4 {
        let dc = tmp[i * 4] + 3;
        let a0 = dc + tmp[3 + i * 4];
        let a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        let a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        let a3 = dc - tmp[3 + i * 4];
        out[64 * i] = ((a0 + a1) >> 3) as i16;
        out[64 * i + 16] = ((a3 + a2) >> 3) as i16;
        out[64 * i + 32] = ((a0 - a1) >> 3) as i16;
        out[64 * i + 48] = ((a3 - a2) >> 3) as i16;
    }
}

// Go multiplies in int32, which wraps for the full-range input of a corrupt file; so do these.
fn mul1(a: i32) -> i32 {
    (a.wrapping_mul(20091) >> 16) + a
}

fn mul2(a: i32) -> i32 {
    a.wrapping_mul(35468) >> 16
}

/// Adds the inverse transform of `inp` to the 4x4 block at o.
fn idct_add(inp: &[i16], b: &mut [u8], o: usize) {
    let mut c = [0i32; 16];
    for i in 0..4 {
        let a = i32::from(inp[i]) + i32::from(inp[8 + i]);
        let bb = i32::from(inp[i]) - i32::from(inp[8 + i]);
        let cc = mul2(i32::from(inp[4 + i])) - mul1(i32::from(inp[12 + i]));
        let dd = mul1(i32::from(inp[4 + i])) + mul2(i32::from(inp[12 + i]));
        c[4 * i] = a + dd;
        c[4 * i + 1] = bb + cc;
        c[4 * i + 2] = bb - cc;
        c[4 * i + 3] = a - dd;
    }
    for i in 0..4 {
        let dc = c[i] + 4;
        let a = dc + c[8 + i];
        let bb = dc - c[8 + i];
        let cc = mul2(c[4 + i]) - mul1(c[12 + i]);
        let dd = mul1(c[4 + i]) + mul2(c[12 + i]);
        let r = o + i * BPS;
        b[r] = clip8(i32::from(b[r]) + ((a + dd) >> 3));
        b[r + 1] = clip8(i32::from(b[r + 1]) + ((bb + cc) >> 3));
        b[r + 2] = clip8(i32::from(b[r + 2]) + ((bb - cc) >> 3));
        b[r + 3] = clip8(i32::from(b[r + 3]) + ((a - dd) >> 3));
    }
}

impl Dec<'_> {
    /// Predicts macroblock (mx, my) in the work buffer, adds its residuals and stores it in the
    /// planes.
    pub(super) fn recon(&mut self, mx: usize, my: usize, m: &Mb) {
        let y = &mut self.yuv;
        if mx == 0 {
            for j in 0..16 {
                y[Y_OFF + j * BPS - 1] = 129;
            }
            for j in 0..8 {
                y[U_OFF + j * BPS - 1] = 129;
                y[V_OFF + j * BPS - 1] = 129;
            }
            if my > 0 {
                y[Y_OFF - BPS - 1] = 129;
                y[U_OFF - BPS - 1] = 129;
                y[V_OFF - BPS - 1] = 129;
            } else {
                y[Y_OFF - BPS - 1..Y_OFF - BPS + 20].fill(127);
                y[U_OFF - BPS - 1..U_OFF - BPS + 8].fill(127);
                y[V_OFF - BPS - 1..V_OFF - BPS + 8].fill(127);
            }
        } else {
            for j in 0..17 {
                let r = Y_OFF - BPS + j * BPS;
                y.copy_within(r + 12..r + 16, r - 4);
            }
            for j in 0..9 {
                let r = U_OFF - BPS + j * BPS;
                y.copy_within(r + 4..r + 8, r - 4);
                let r = V_OFF - BPS + j * BPS;
                y.copy_within(r + 4..r + 8, r - 4);
            }
        }
        if my > 0 {
            y[Y_OFF - BPS..Y_OFF - BPS + 16].copy_from_slice(&self.top_y[16 * mx..16 * mx + 16]);
            y[U_OFF - BPS..U_OFF - BPS + 8].copy_from_slice(&self.top_u[8 * mx..8 * mx + 8]);
            y[V_OFF - BPS..V_OFF - BPS + 8].copy_from_slice(&self.top_v[8 * mx..8 * mx + 8]);
        }
        if m.i4 {
            let tr = Y_OFF - BPS + 16;
            if my > 0 {
                if mx >= self.mbw - 1 {
                    let v = self.top_y[16 * mx + 15];
                    y[tr..tr + 4].fill(v);
                } else {
                    y[tr..tr + 4].copy_from_slice(&self.top_y[16 * (mx + 1)..16 * (mx + 1) + 4]);
                }
            }
            for r in 1..4 {
                // replicate the top-right pixels below
                y.copy_within(tr..tr + 4, tr + 4 * r * BPS);
            }
            for n in 0..16 {
                let o = Y_OFF + scan(n);
                pred4(y, o, m.modes[n]);
                if self.nonzero[n] {
                    idct_add(&self.coeffs[16 * n..], y, o);
                }
            }
        } else {
            pred(y, Y_OFF, 16, m.modes[0], my == 0, mx == 0);
            for n in 0..16 {
                if self.nonzero[n] {
                    idct_add(&self.coeffs[16 * n..], y, Y_OFF + scan(n));
                }
            }
        }
        pred(y, U_OFF, 8, m.uvmode, my == 0, mx == 0);
        pred(y, V_OFF, 8, m.uvmode, my == 0, mx == 0);
        for n in 0..4 {
            let o = BPS * (n >> 1) * 4 + (n & 1) * 4;
            if self.nonzero[16 + n] {
                idct_add(&self.coeffs[16 * (16 + n)..], y, U_OFF + o);
            }
            if self.nonzero[20 + n] {
                idct_add(&self.coeffs[16 * (20 + n)..], y, V_OFF + o);
            }
        }
        if my < self.mbh - 1 {
            self.top_y[16 * mx..16 * mx + 16]
                .copy_from_slice(&y[Y_OFF + 15 * BPS..Y_OFF + 15 * BPS + 16]);
            self.top_u[8 * mx..8 * mx + 8]
                .copy_from_slice(&y[U_OFF + 7 * BPS..U_OFF + 7 * BPS + 8]);
            self.top_v[8 * mx..8 * mx + 8]
                .copy_from_slice(&y[V_OFF + 7 * BPS..V_OFF + 7 * BPS + 8]);
        }
        let f = &mut self.f;
        for j in 0..16 {
            let d = (16 * my + j) * f.y_stride + 16 * mx;
            f.y[d..d + 16].copy_from_slice(&y[Y_OFF + j * BPS..Y_OFF + j * BPS + 16]);
        }
        for j in 0..8 {
            let d = (8 * my + j) * f.c_stride + 8 * mx;
            f.u[d..d + 8].copy_from_slice(&y[U_OFF + j * BPS..U_OFF + j * BPS + 8]);
            f.v[d..d + 8].copy_from_slice(&y[V_OFF + j * BPS..V_OFF + j * BPS + 8]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-range coefficients make the second pass multiply more than 2^31 / 35468 (fuzz crash):
    /// the multiplies must wrap, as Go's int32 does, not panic in a debug build.
    #[test]
    fn idct_full_range_wraps() {
        let mut b = [128u8; 4 * BPS];
        idct_add(&[i16::MAX; 16], &mut b, 0);
        let mut c = [128u8; 4 * BPS];
        idct_add(&[i16::MIN; 16], &mut c, 0);
    }
}
