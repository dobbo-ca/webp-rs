package vp8dec

import "context"

func multHi(v, c int) int { return (v * c) >> 8 }

func clip6(v int) uint8 {
	if v&^(256<<6-1) == 0 {
		return uint8(v >> 6)
	}
	if v < 0 {
		return 0
	}
	return 255
}

// yuvToRGB is libwebp's VP8YUVToR, G and B (yuv.h).
func yuvToRGB(dst []byte, y, u, v int) {
	dst[0] = clip6(multHi(y, 19077) + multHi(v, 26149) - 14234)
	dst[1] = clip6(multHi(y, 19077) - multHi(u, 6419) - multHi(v, 13320) + 8708)
	dst[2] = clip6(multHi(y, 19077) + multHi(u, 33050) - 17685)
}

// RGBA converts the frame to non-premultiplied RGBA in dst (4*W*H bytes)
// with libwebp's default "fancy" chroma upsampling (io_dec.c EmitFancyRGB,
// upsampling.c). ctx is checked once a row.
func (f *Frame) RGBA(ctx context.Context, dst []byte) error {
	return f.Rows(ctx, dst, 0, f.H, 4)
}

// Rows converts rows r0 to r1 (exclusive) like RGBA, into dst, which holds
// just those rows: 3 bytes a pixel, or 4 with alpha 255 if step is 4.
func (f *Frame) Rows(ctx context.Context, dst []byte, r0, r1, step int) error {
	cw, ch := (f.W+1)/2, (f.H+1)/2
	var near, far [2][]byte // u and v samples of the two chroma rows
	for r := r0; r < r1; r++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		// Chroma row nearest to r and the next nearest (the same at the edges).
		n, o := r/2, r/2
		if r&1 == 1 {
			o = min(n+1, ch-1)
		} else if r > 0 {
			o = n - 1
		}
		near[0], near[1] = f.U[n*f.CStrd:n*f.CStrd+cw], f.V[n*f.CStrd:n*f.CStrd+cw]
		far[0], far[1] = f.U[o*f.CStrd:o*f.CStrd+cw], f.V[o*f.CStrd:o*f.CStrd+cw]
		y := f.Y[r*f.YStride : r*f.YStride+f.W]
		row := dst[step*f.W*(r-r0) : step*f.W*(r-r0+1)]
		px := func(x int, u, v int) {
			yuvToRGB(row[step*x:], int(y[x]), u, v)
			if step == 4 {
				row[4*x+3] = 255
			}
		}
		var uv [2]int
		for c := 0; c < 2; c++ {
			uv[c] = (3*int(near[c][0]) + int(far[c][0]) + 2) >> 2
		}
		px(0, uv[0], uv[1])
		for x := 1; x < cw; x++ {
			var l, rr [2]int // pixels 2x-1 and 2x
			for c := 0; c < 2; c++ {
				n0, n1 := int(near[c][x-1]), int(near[c][x])
				f0, f1 := int(far[c][x-1]), int(far[c][x])
				l[c] = ((n0+3*n1+3*f0+f1+8)>>3 + n0) >> 1
				rr[c] = ((n1+3*n0+3*f1+f0+8)>>3 + n1) >> 1
			}
			px(2*x-1, l[0], l[1])
			if 2*x < f.W {
				px(2*x, rr[0], rr[1])
			}
		}
		if f.W&1 == 0 {
			last := cw - 1
			for c := 0; c < 2; c++ {
				uv[c] = (3*int(near[c][last]) + int(far[c][last]) + 2) >> 2
			}
			px(f.W-1, uv[0], uv[1])
		}
	}
	return nil
}
