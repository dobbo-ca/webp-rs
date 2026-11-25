//! The loop filters (RFC 6386 15), libwebp's src/dsp/dec.c.

use super::Dec;
use super::pred::clip8;

fn sclip1(v: i32) -> i32 {
    v.clamp(-128, 127)
}

fn sclip2(v: i32) -> i32 {
    v.clamp(-16, 15)
}

// In these, p is the offset of q0 and step is the distance across the edge.

fn filter2(b: &mut [u8], p: usize, step: usize) {
    let (p1, p0, q0, q1) = (
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
        i32::from(b[p]),
        i32::from(b[p + step]),
    );
    let a = 3 * (q0 - p0) + sclip1(p1 - q1);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    b[p - step] = clip8(p0 + a2);
    b[p] = clip8(q0 - a1);
}

fn filter4(b: &mut [u8], p: usize, step: usize) {
    let (p1, p0, q0, q1) = (
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
        i32::from(b[p]),
        i32::from(b[p + step]),
    );
    let a = 3 * (q0 - p0);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    let a3 = (a1 + 1) >> 1;
    b[p - 2 * step] = clip8(p1 + a3);
    b[p - step] = clip8(p0 + a2);
    b[p] = clip8(q0 - a1);
    b[p + step] = clip8(q1 - a3);
}

fn filter6(b: &mut [u8], p: usize, step: usize) {
    let (p2, p1, p0) = (
        i32::from(b[p - 3 * step]),
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
    );
    let (q0, q1, q2) = (
        i32::from(b[p]),
        i32::from(b[p + step]),
        i32::from(b[p + 2 * step]),
    );
    let a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    let a1 = (27 * a + 63) >> 7;
    let a2 = (18 * a + 63) >> 7;
    let a3 = (9 * a + 63) >> 7;
    b[p - 3 * step] = clip8(p2 + a3);
    b[p - 2 * step] = clip8(p1 + a2);
    b[p - step] = clip8(p0 + a1);
    b[p] = clip8(q0 - a1);
    b[p + step] = clip8(q1 - a2);
    b[p + 2 * step] = clip8(q2 - a3);
}

fn hev(b: &[u8], p: usize, step: usize, thresh: i32) -> bool {
    let (p1, p0, q0, q1) = (
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
        i32::from(b[p]),
        i32::from(b[p + step]),
    );
    (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh
}

fn needs(b: &[u8], p: usize, step: usize, t: i32) -> bool {
    let (p1, p0, q0, q1) = (
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
        i32::from(b[p]),
        i32::from(b[p + step]),
    );
    4 * (p0 - q0).abs() + (p1 - q1).abs() <= t
}

fn needs2(b: &[u8], p: usize, step: usize, t: i32, it: i32) -> bool {
    let (p3, p2, p1, p0) = (
        i32::from(b[p - 4 * step]),
        i32::from(b[p - 3 * step]),
        i32::from(b[p - 2 * step]),
        i32::from(b[p - step]),
    );
    let (q0, q1, q2, q3) = (
        i32::from(b[p]),
        i32::from(b[p + step]),
        i32::from(b[p + 2 * step]),
        i32::from(b[p + 3 * step]),
    );
    if 4 * (p0 - q0).abs() + (p1 - q1).abs() > t {
        return false;
    }
    (p3 - p2).abs() <= it
        && (p2 - p1).abs() <= it
        && (p1 - p0).abs() <= it
        && (q3 - q2).abs() <= it
        && (q2 - q1).abs() <= it
        && (q1 - q0).abs() <= it
}

/// Filters 16 pixels along an edge: `hstride` crosses the edge, `vstride` walks along it.
fn simple(b: &mut [u8], p: usize, hstride: usize, vstride: usize, thresh: i32) {
    let t = 2 * thresh + 1;
    for i in 0..16 {
        if needs(b, p + i * vstride, hstride, t) {
            filter2(b, p + i * vstride, hstride);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn loop_filter(
    b: &mut [u8],
    mut p: usize,
    hstride: usize,
    vstride: usize,
    size: usize,
    thresh: i32,
    ithresh: i32,
    hev_t: i32,
    edge: bool,
) {
    let t = 2 * thresh + 1;
    for _ in 0..size {
        if needs2(b, p, hstride, t, ithresh) {
            if hev(b, p, hstride, hev_t) {
                filter2(b, p, hstride);
            } else if edge {
                filter6(b, p, hstride);
            } else {
                filter4(b, p, hstride);
            }
        }
        p += vstride;
    }
}

impl Dec<'_> {
    /// Runs the loop filter over macroblock row `my`, in place.
    pub(super) fn filter_row(&mut self, my: usize) {
        let f = &mut self.f;
        for mx in 0..self.mbw {
            let fi = self.finfos[mx];
            let limit = i32::from(fi.limit);
            if limit == 0 {
                continue;
            }
            let y = 16 * my * f.y_stride + 16 * mx;
            let u = 8 * my * f.c_stride + 8 * mx;
            let (ys, cs) = (f.y_stride, f.c_stride);
            if self.f_type == 1 {
                if mx > 0 {
                    simple(&mut f.y, y, 1, ys, limit + 4);
                }
                if fi.inner {
                    for k in (4..16).step_by(4) {
                        simple(&mut f.y, y + k, 1, ys, limit);
                    }
                }
                if my > 0 {
                    simple(&mut f.y, y, ys, 1, limit + 4);
                }
                if fi.inner {
                    for k in (4..16).step_by(4) {
                        simple(&mut f.y, y + k * ys, ys, 1, limit);
                    }
                }
                continue;
            }
            let (il, ht) = (i32::from(fi.ilevel), i32::from(fi.hev));
            if mx > 0 {
                loop_filter(&mut f.y, y, 1, ys, 16, limit + 4, il, ht, true);
                loop_filter(&mut f.u, u, 1, cs, 8, limit + 4, il, ht, true);
                loop_filter(&mut f.v, u, 1, cs, 8, limit + 4, il, ht, true);
            }
            if fi.inner {
                for k in (4..16).step_by(4) {
                    loop_filter(&mut f.y, y + k, 1, ys, 16, limit, il, ht, false);
                }
                loop_filter(&mut f.u, u + 4, 1, cs, 8, limit, il, ht, false);
                loop_filter(&mut f.v, u + 4, 1, cs, 8, limit, il, ht, false);
            }
            if my > 0 {
                loop_filter(&mut f.y, y, ys, 1, 16, limit + 4, il, ht, true);
                loop_filter(&mut f.u, u, cs, 1, 8, limit + 4, il, ht, true);
                loop_filter(&mut f.v, u, cs, 1, 8, limit + 4, il, ht, true);
            }
            if fi.inner {
                for k in (4..16).step_by(4) {
                    loop_filter(&mut f.y, y + k * ys, ys, 1, 16, limit, il, ht, false);
                }
                loop_filter(&mut f.u, u + 4 * cs, cs, 1, 8, limit, il, ht, false);
                loop_filter(&mut f.v, u + 4 * cs, cs, 1, 8, limit, il, ht, false);
            }
        }
    }
}
