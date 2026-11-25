// idct and iwht from golang.org/x/image/vp8 idct.go, BSD-3-Clause.

package vp8enc

import (
	"image"

	"github.com/dobbo-ca/webp-go/internal/fmath"
)

// fdct is libwebp FTransform_C, enc.c:161-189.
func fdct(b *[16]int16) {
	var t [16]int32
	for i := 0; i < 4; i++ {
		d0 := int32(b[i*4+0])
		d1 := int32(b[i*4+1])
		d2 := int32(b[i*4+2])
		d3 := int32(b[i*4+3])
		a0 := d0 + d3
		a1 := d1 + d2
		a2 := d1 - d2
		a3 := d0 - d3
		t[i*4+0] = (a0 + a1) * 8
		t[i*4+1] = (a2*2217 + a3*5352 + 1812) >> 9
		t[i*4+2] = (a0 - a1) * 8
		t[i*4+3] = (a3*2217 - a2*5352 + 937) >> 9
	}
	fdctCols(&t, b)
}

// fdctCols is the vertical pass of fdct.
func fdctCols(t *[16]int32, b *[16]int16) {
	for i := 0; i < 4; i++ {
		a0 := t[i] + t[12+i]
		a1 := t[4+i] + t[8+i]
		a2 := t[4+i] - t[8+i]
		a3 := t[i] - t[12+i]
		b[i] = int16((a0 + a1 + 7) >> 4)
		v := (a2*2217 + a3*5352 + 12000) >> 16
		if a3 != 0 {
			v++
		}
		b[4+i] = int16(v)
		b[8+i] = int16((a0 - a1 + 7) >> 4)
		b[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}

// whtRound makes x/image iWHT exact.
var whtRound = [16]int32{0: 1, 1: 1, 2: 1, 4: 1, 8: 1}

// libwebp FTransformWHT_C plus whtRound.
func fwht(dc *[16]int16) {
	var t [16]int32
	for i := 0; i < 4; i++ {
		p0 := int32(dc[i*4+0])
		p1 := int32(dc[i*4+1])
		p2 := int32(dc[i*4+2])
		p3 := int32(dc[i*4+3])
		a0 := p0 + p2
		a1 := p1 + p3
		a2 := p1 - p3
		a3 := p0 - p2
		t[i*4+0] = a0 + a1
		t[i*4+1] = a3 + a2
		t[i*4+2] = a3 - a2
		t[i*4+3] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0 := t[i] + t[8+i]
		a1 := t[4+i] + t[12+i]
		a2 := t[4+i] - t[12+i]
		a3 := t[i] - t[8+i]
		dc[i] = int16((a0 + a1 + whtRound[i]) >> 1)
		dc[4+i] = int16((a3 + a2 + whtRound[4+i]) >> 1)
		dc[8+i] = int16((a3 - a2 + whtRound[8+i]) >> 1)
		dc[12+i] = int16((a0 - a1 + whtRound[12+i]) >> 1)
	}
}

// gammaToLinear and linearToGamma are libwebp yuv.c:237-251.
var gammaToLinear [256]int32
var linearToGamma [33]int32

func init() {
	for v := 0; v < 256; v++ {
		gammaToLinear[v] = int32(float64(fmath.Pow(float64(v)*(1.0/255), 0.8)*4095) + .5)
	}
	for v := 0; v < 33; v++ {
		linearToGamma[v] = int32(float64(255*fmath.Pow(float64(v)*(128.0/4095), 1/0.8)) + .5)
	}
}

// linToGamma is Interpolate plus LinearToGamma, yuv.c:257-276.
func linToGamma(s int32) int32 {
	p, x := s>>9, s&511
	return (linearToGamma[p+1]*x + linearToGamma[p]*(512-x) + 64) >> 7
}

// libwebp ImportYUVAFromRGBA, default gamma. With nil a it assumes
// opaque pixels and reports whether they were; with a (stride w) it
// copies alpha there and weights the chroma of partly transparent 2x2
// blocks by alpha, picture_csp_enc.c:362-437.
func rgbaToYCbCr(dst *image.YCbCr, pix []uint8, stride int, a []uint8) (opaque bool) {
	w, h := dst.Rect.Dx(), dst.Rect.Dy()
	if len(pix) == (h-1)*stride+4*w-1 {
		// Row slices need the last alpha; count it opaque.
		pix = append(pix[:len(pix):len(pix)], 255)
	}
	for y := 0; y < h; y++ {
		lumaRow(dst.Y[y*dst.YStride:][:w], pix[y*stride:][:4*w])
	}
	if a != nil {
		for y := 0; y < h; y++ {
			row, p := a[y*w:][:w], pix[y*stride:][:4*w]
			for x := range row {
				row[x] = p[4*x+3]
			}
		}
		rgbaChromaAlpha(dst, pix, stride)
		return false
	}
	g := &gammaToLinear
	and := uint8(255)
	for cy := 0; cy < (h+1)/2; cy++ {
		r0 := 2 * cy
		r1 := min(r0+1, h-1)
		p0 := pix[r0*stride:][:4*w]
		p1 := pix[r1*stride:][:4*w]
		cb := dst.Cb[cy*dst.CStride:][:(w+1)/2]
		cr := dst.Cr[cy*dst.CStride:][:(w+1)/2]
		n, an := chromaRow(cb, cr, p0, p1)
		and &= an
		for cx := n; cx < len(cb); cx++ {
			o0, o1 := 8*cx, 8*cx+4
			if o1 >= len(p0) {
				o1 = o0
			}
			a, b := p0[o0:o0+4], p0[o1:o1+4]
			c, d := p1[o0:o0+4], p1[o1:o1+4]
			and &= a[3] & b[3] & c[3] & d[3]
			sR := linToGamma(g[a[0]] + g[b[0]] + g[c[0]] + g[d[0]])
			sG := linToGamma(g[a[1]] + g[b[1]] + g[c[1]] + g[d[1]])
			sB := linToGamma(g[a[2]] + g[b[2]] + g[c[2]] + g[d[2]])
			cb[cx] = uint8((-9719*sR - 19081*sG + 28800*sB + 1<<17 + 128<<18) >> 18)
			cr[cx] = uint8((28800*sR - 24116*sG - 4684*sB + 1<<17 + 128<<18) >> 18)
		}
	}
	return and == 255
}

// rgbaChromaAlpha is rgbaToYCbCr's chroma with alpha weighting.
func rgbaChromaAlpha(dst *image.YCbCr, pix []uint8, stride int) {
	w, h := dst.Rect.Dx(), dst.Rect.Dy()
	g := &gammaToLinear
	for cy := 0; cy < (h+1)/2; cy++ {
		r0 := 2 * cy
		r1 := min(r0+1, h-1)
		p0 := pix[r0*stride:][:4*w]
		p1 := pix[r1*stride:][:4*w]
		cb := dst.Cb[cy*dst.CStride:][:(w+1)/2]
		cr := dst.Cr[cy*dst.CStride:][:(w+1)/2]
		for cx := range cb {
			o0, o1 := 8*cx, 8*cx+4
			if o1 >= len(p0) {
				o1 = o0
			}
			a, b := p0[o0:o0+4], p0[o1:o1+4]
			c, d := p1[o0:o0+4], p1[o1:o1+4]
			var sR, sG, sB int32
			if t := uint32(a[3]) + uint32(b[3]) + uint32(c[3]) + uint32(d[3]); t == 4*255 || t == 0 {
				sR = linToGamma(g[a[0]] + g[b[0]] + g[c[0]] + g[d[0]])
				sG = linToGamma(g[a[1]] + g[b[1]] + g[c[1]] + g[d[1]])
				sB = linToGamma(g[a[2]] + g[b[2]] + g[c[2]] + g[d[2]])
			} else {
				sR = weighted(a, b, c, d, 0, t)
				sG = weighted(a, b, c, d, 1, t)
				sB = weighted(a, b, c, d, 2, t)
			}
			cb[cx] = uint8((-9719*sR - 19081*sG + 28800*sB + 1<<17 + 128<<18) >> 18)
			cr[cx] = uint8((28800*sR - 24116*sG - 4684*sB + 1<<17 + 128<<18) >> 18)
		}
	}
}

// weighted is LinearToGammaWeighted for channel i with total alpha t.
// uint32 wraps like the C; kInvAlpha[t] is (1<<19)/t.
func weighted(a, b, c, d []uint8, i int, t uint32) int32 {
	g := &gammaToLinear
	s := uint32(a[3])*uint32(g[a[i]]) + uint32(b[3])*uint32(g[b[i]]) +
		uint32(c[3])*uint32(g[c[i]]) + uint32(d[3])*uint32(g[d[i]])
	return linToGamma(int32(s * (1 << 19 / t) >> 17))
}

func lumaRowGo(dst, src []uint8) {
	for x := range dst {
		p := src[4*x : 4*x+3]
		r, g, b := int32(p[0]), int32(p[1]), int32(p[2])
		dst[x] = uint8((16839*r + 33059*g + 6420*b + 1<<15 + 16<<16) >> 16)
	}
}

func clip8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// idctAddGo adds the inverse DCT of c at (y, x).
func (z *ybrBuf) idctAddGo(c *[16]int16, y, x int) {
	const (
		c1 = 85627 // 65536 * cos(pi/8) * sqrt(2).
		c2 = 35468 // 65536 * sin(pi/8) * sqrt(2).
	)
	if *c == [16]int16{} {
		return
	}
	if [15]int16(c[1:]) == [15]int16{} {
		dc := (int32(c[0]) + 4) >> 3
		for j := 0; j < 4; j++ {
			r := z[y+j][x : x+4 : x+4]
			r[0] = clip8(int32(r[0]) + dc)
			r[1] = clip8(int32(r[1]) + dc)
			r[2] = clip8(int32(r[2]) + dc)
			r[3] = clip8(int32(r[3]) + dc)
		}
		return
	}
	var m [4][4]int32
	for i := 0; i < 4; i++ {
		a := int32(c[i+0]) + int32(c[i+8])
		b := int32(c[i+0]) - int32(c[i+8])
		cc := (int32(c[i+4])*c2)>>16 - (int32(c[i+12])*c1)>>16
		d := (int32(c[i+4])*c1)>>16 + (int32(c[i+12])*c2)>>16
		m[i][0] = a + d
		m[i][1] = b + cc
		m[i][2] = b - cc
		m[i][3] = a - d
	}
	for j := 0; j < 4; j++ {
		dc := m[0][j] + 4
		a := dc + m[2][j]
		b := dc - m[2][j]
		cc := (m[1][j]*c2)>>16 - (m[3][j]*c1)>>16
		d := (m[1][j]*c1)>>16 + (m[3][j]*c2)>>16
		r := z[y+j][x : x+4 : x+4]
		r[0] = clip8(int32(r[0]) + (a+d)>>3)
		r[1] = clip8(int32(r[1]) + (b+cc)>>3)
		r[2] = clip8(int32(r[2]) + (b-cc)>>3)
		r[3] = clip8(int32(r[3]) + (a-d)>>3)
	}
}

// iwht is x/image inverseWHT16, raster order out.
func iwht(c *[16]int16) [16]int16 {
	var m [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(c[0+i]) + int32(c[12+i])
		a1 := int32(c[4+i]) + int32(c[8+i])
		a2 := int32(c[4+i]) - int32(c[8+i])
		a3 := int32(c[0+i]) - int32(c[12+i])
		m[0+i] = a0 + a1
		m[8+i] = a0 - a1
		m[4+i] = a3 + a2
		m[12+i] = a3 - a2
	}
	var o [16]int16
	for i := 0; i < 4; i++ {
		dc := m[0+i*4] + 3
		a0 := dc + m[3+i*4]
		a1 := m[1+i*4] + m[2+i*4]
		a2 := m[1+i*4] - m[2+i*4]
		a3 := dc - m[3+i*4]
		o[i*4+0] = int16((a0 + a1) >> 3)
		o[i*4+1] = int16((a3 + a2) >> 3)
		o[i*4+2] = int16((a0 - a1) >> 3)
		o[i*4+3] = int16((a3 - a2) >> 3)
	}
	return o
}
