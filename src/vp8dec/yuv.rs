//! YUV to RGB with libwebp's "fancy" chroma upsampling (io_dec.c EmitFancyRGB, upsampling.c).

use super::Frame;
use crate::Error;
use crate::limits::check_cancel;
use std::sync::atomic::AtomicBool;

fn mult_hi(v: i32, c: i32) -> i32 {
    (v * c) >> 8
}

fn clip6(v: i32) -> u8 {
    if v & !((256 << 6) - 1) == 0 {
        (v >> 6) as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

/// libwebp's VP8YUVToR, G and B (yuv.h).
fn yuv_to_rgb(dst: &mut [u8], y: i32, u: i32, v: i32) {
    dst[0] = clip6(mult_hi(y, 19077) + mult_hi(v, 26149) - 14234);
    dst[1] = clip6(mult_hi(y, 19077) - mult_hi(u, 6419) - mult_hi(v, 13320) + 8708);
    dst[2] = clip6(mult_hi(y, 19077) + mult_hi(u, 33050) - 17685);
}

fn px(row: &mut [u8], step: usize, x: usize, y: u8, u: i32, v: i32) {
    yuv_to_rgb(&mut row[step * x..], i32::from(y), u, v);
    if step == 4 {
        row[4 * x + 3] = 255;
    }
}

impl Frame {
    /// Converts rows r0 to r1 (exclusive) into `dst`, which holds just those rows: 3 bytes a
    /// pixel, or 4 with alpha 255 if `step` is 4. `cancel` is checked once a row.
    pub(crate) fn rows(
        &self,
        dst: &mut [u8],
        r0: usize,
        r1: usize,
        step: usize,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let (w, cw, ch) = (self.w, self.w.div_ceil(2), self.h.div_ceil(2));
        for r in r0..r1 {
            check_cancel(cancel)?;
            // Chroma row nearest to r and the next nearest (the same at the edges).
            let n = r / 2;
            let o = if r & 1 == 1 {
                (n + 1).min(ch - 1)
            } else if r > 0 {
                n - 1
            } else {
                n
            };
            let near = [
                &self.u[n * self.c_stride..n * self.c_stride + cw],
                &self.v[n * self.c_stride..n * self.c_stride + cw],
            ];
            let far = [
                &self.u[o * self.c_stride..o * self.c_stride + cw],
                &self.v[o * self.c_stride..o * self.c_stride + cw],
            ];
            let y = &self.y[r * self.y_stride..r * self.y_stride + w];
            let row = &mut dst[step * w * (r - r0)..step * w * (r - r0 + 1)];
            let mut uv = [0i32; 2];
            for c in 0..2 {
                uv[c] = (3 * i32::from(near[c][0]) + i32::from(far[c][0]) + 2) >> 2;
            }
            px(row, step, 0, y[0], uv[0], uv[1]);
            for x in 1..cw {
                let (mut l, mut rr) = ([0i32; 2], [0i32; 2]); // pixels 2x-1 and 2x
                for c in 0..2 {
                    let (n0, n1) = (i32::from(near[c][x - 1]), i32::from(near[c][x]));
                    let (f0, f1) = (i32::from(far[c][x - 1]), i32::from(far[c][x]));
                    l[c] = (((n0 + 3 * n1 + 3 * f0 + f1 + 8) >> 3) + n0) >> 1;
                    rr[c] = (((n1 + 3 * n0 + 3 * f1 + f0 + 8) >> 3) + n1) >> 1;
                }
                px(row, step, 2 * x - 1, y[2 * x - 1], l[0], l[1]);
                if 2 * x < w {
                    px(row, step, 2 * x, y[2 * x], rr[0], rr[1]);
                }
            }
            if w & 1 == 0 {
                let last = cw - 1;
                for c in 0..2 {
                    uv[c] = (3 * i32::from(near[c][last]) + i32::from(far[c][last]) + 2) >> 2;
                }
                px(row, step, w - 1, y[w - 1], uv[0], uv[1]);
            }
        }
        Ok(())
    }
}
