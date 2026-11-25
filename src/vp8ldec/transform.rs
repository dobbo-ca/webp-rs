//! The inverse transforms: predictor, cross-colour, subtract-green and colour indexing
//! (libwebp's src/dsp/lossless.c).

use super::{Dec, T_CROSS, T_PREDICTOR, T_SUB_GREEN, Transform, ceil, get, put};
use crate::Error;
use crate::limits::check_cancel;

impl Dec<'_> {
    /// Undoes transform `t` on `a`, whose rows are `t.xs` pixels wide until a colour-indexing
    /// transform widens them to `w`.
    pub(super) fn inverse(
        &self,
        a: &mut [u8],
        t: &Transform,
        h: usize,
        w: usize,
    ) -> Result<(), Error> {
        match t.kind {
            T_PREDICTOR => {
                for y in 0..h {
                    check_cancel(self.cancel)?;
                    let (before, after) = a.split_at_mut(4 * y * t.xs);
                    let prev = if y > 0 {
                        Some(&before[4 * (y - 1) * t.xs..])
                    } else {
                        None
                    };
                    predictor_row(&mut after[..4 * t.xs], prev, y, t);
                }
            }
            T_CROSS => {
                for y in 0..h {
                    check_cancel(self.cancel)?;
                    cross_row(&mut a[4 * y * t.xs..], y, t);
                }
            }
            T_SUB_GREEN => sub_green(&mut a[..4 * t.xs * h]),
            _ => {
                let pw = ceil(w, t.bits);
                for y in (0..h).rev() {
                    check_cancel(self.cancel)?;
                    index_row(a, 4 * y * w, 4 * y * pw, w, t);
                }
            }
        }
        Ok(())
    }
}

/// Undoes the transforms `ts`, last first, on row `y`, which is in `cur` and `w` pixels wide when
/// done. `prev` holds the row the predictor made last, and `cur` needs room for `w` pixels.
pub(super) fn inverse_row(cur: &mut [u8], prev: &mut [u8], ts: &[Transform], y: usize, w: usize) {
    for t in ts.iter().rev() {
        match t.kind {
            T_PREDICTOR => {
                let p = if y > 0 { Some(&prev[..4 * t.xs]) } else { None };
                predictor_row(&mut cur[..4 * t.xs], p, y, t);
                prev[..4 * t.xs].copy_from_slice(&cur[..4 * t.xs]);
            }
            T_CROSS => cross_row(cur, y, t),
            T_SUB_GREEN => sub_green(&mut cur[..4 * t.xs]),
            _ => index_row(cur, 0, 0, w, t),
        }
    }
}

fn cross_row(row: &mut [u8], y: usize, t: &Transform) {
    let tw = ceil(t.xs, t.bits);
    for x in 0..t.xs {
        let m = get(&t.data, (y >> t.bits) * tw + (x >> t.bits));
        let (gr, gb, rb) = (
            i32::from(m as i8),
            i32::from((m >> 8) as i8),
            i32::from((m >> 16) as i8),
        );
        let px = get(row, x);
        let g = i32::from((px >> 8) as i8);
        let r = ((px >> 16) as u8).wrapping_add(((gr * g) >> 5) as u8);
        let b = (px as u8)
            .wrapping_add(((gb * g) >> 5) as u8)
            .wrapping_add(((rb * i32::from(r as i8)) >> 5) as u8);
        put(
            row,
            x,
            (px & 0xff00ff00) | u32::from(r) << 16 | u32::from(b),
        );
    }
}

fn sub_green(a: &mut [u8]) {
    for i in 0..a.len() / 4 {
        let px = get(a, i);
        let g = (px >> 8) & 0xff;
        put(
            a,
            i,
            (px & 0xff00ff00) | ((px & 0xff0000) + (g << 16)) & 0xff0000 | ((px & 0xff) + g) & 0xff,
        );
    }
}

/// Writes the `w` pixels of a row at byte offset `dst` from the packed indexes at byte offset
/// `src`, last first, so the two may overlap.
fn index_row(buf: &mut [u8], dst: usize, src: usize, w: usize, t: &Transform) {
    let per = 1u32 << t.bits;
    let bpp = 8 / per;
    for x in (0..w).rev() {
        let g = get(&buf[src..], x >> t.bits) >> 8;
        let i = (g >> (bpp * (x as u32 & (per - 1)))) & ((1 << bpp) - 1);
        put(&mut buf[dst..], x, get(&t.data, i as usize));
    }
}

/// Undoes the predictor transform on row `y` of `t.xs` pixels. `prev` is the row above, `None`
/// for the first row. The rightmost pixel's top right is the first pixel of its own row.
fn predictor_row(cur: &mut [u8], prev: Option<&[u8]>, y: usize, t: &Transform) {
    let xs = t.xs;
    let Some(prev) = prev else {
        put(cur, 0, add(get(cur, 0), 0xff000000));
        for x in 1..xs {
            put(cur, x, add(get(cur, x), get(cur, x - 1)));
        }
        return;
    };
    let tw = ceil(xs, t.bits);
    put(cur, 0, add(get(cur, 0), get(prev, 0)));
    let mut mode = 0;
    for x in 1..xs {
        if x & ((1 << t.bits) - 1) == 0 || x == 1 {
            mode = (get(&t.data, (y >> t.bits) * tw + (x >> t.bits)) >> 8) & 15;
        }
        let (l, tp, tl) = (get(cur, x - 1), get(prev, x), get(prev, x - 1));
        let tr = if x + 1 < xs {
            get(prev, x + 1)
        } else {
            get(cur, 0)
        };
        let p = match mode {
            1 => l,
            2 => tp,
            3 => tr,
            4 => tl,
            5 => avg2(avg2(l, tr), tp),
            6 => avg2(l, tl),
            7 => avg2(l, tp),
            8 => avg2(tl, tp),
            9 => avg2(tp, tr),
            10 => avg2(avg2(l, tl), avg2(tp, tr)),
            11 => select_px(l, tp, tl),
            12 => clamp_add_sub(l, tp, tl),
            13 => clamp_add_sub_half(avg2(l, tp), tl),
            _ => 0xff000000, // 0, 14, 15
        };
        put(cur, x, add(get(cur, x), p));
    }
}

/// Adds per channel, modulo 256.
fn add(a: u32, b: u32) -> u32 {
    ((a & 0xff00ff00).wrapping_add(b & 0xff00ff00) & 0xff00ff00)
        | ((a & 0x00ff00ff).wrapping_add(b & 0x00ff00ff) & 0x00ff00ff)
}

fn avg2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefefefe) >> 1) + (a & b)
}

fn ch(v: u32, s: u32) -> i32 {
    ((v >> s) & 0xff) as i32
}

/// The spec's Select(L, T, TL).
fn select_px(l: u32, t: u32, tl: u32) -> u32 {
    let (mut pl, mut pt) = (0, 0);
    for s in [0, 8, 16, 24] {
        pl += (ch(t, s) - ch(tl, s)).abs();
        pt += (ch(l, s) - ch(tl, s)).abs();
    }
    if pl < pt { l } else { t }
}

fn clamp(v: i32) -> u32 {
    v.clamp(0, 255) as u32
}

fn clamp_add_sub(a: u32, b: u32, c: u32) -> u32 {
    let mut r = 0;
    for s in [0, 8, 16, 24] {
        r |= clamp(ch(a, s) + ch(b, s) - ch(c, s)) << s;
    }
    r
}

fn clamp_add_sub_half(a: u32, b: u32) -> u32 {
    let mut r = 0;
    for s in [0, 8, 16, 24] {
        r |= clamp(ch(a, s) + (ch(a, s) - ch(b, s)) / 2) << s;
    }
    r
}
