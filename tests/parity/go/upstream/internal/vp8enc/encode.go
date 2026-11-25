// Package vp8enc encodes VP8 key frames.
package vp8enc

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"io"
	"math"
	"math/bits"
	"unsafe"

	"github.com/dobbo-ca/webp-go/internal/riff"
	"github.com/dobbo-ca/webp-go/internal/vp8l"
)

type mbInfo struct {
	skip, i16 bool
	modes     [16]uint8
	uv, seg   uint8
}

// Encoder keeps its buffers between calls.
type Encoder struct {
	w        io.Writer
	img      image.YCbCr // source, then reconstruction
	rgba     *image.RGBA
	nrgba    *image.NRGBA // strips for sources that may hold alpha
	alpha    []uint8      // w*h alpha plane when hasAlpha
	hasAlpha bool
	gray     bool // true for *image.Gray: skip UV analysis and search
	vl       vp8l.Encoder
	alph     []byte // ALPH chunk data
	mbw, mbh int
	z        ybrBuf
	q        quant
	p0, p1   boolEncoder
	p2       boolEncoder
	rows     []tokPos // token start of each MB row, then end
	psz      [3]byte
	mbs      []mbInfo
	top      []nzCtx
	lv       [25][16]int16
	tok      tokenBuf // ponytail: grows with output; ~19 B/output byte
	probs    [1056]uint8
	lam      lambdas
	segQ     [4]quant
	segLam   [4]lambdas
	method   int
	done     <-chan struct{} // Cancel
	lwOn     bool            // MatchLibwebp
	lw       *lwState        // made on first MatchLibwebp encode
	lwa      *vp8l.LW        // made on first MatchLibwebp ALPH
	lwa16    bool            // AlphaLibwebp16
	lwaOld   bool            // LegacyAlpha
	segments int             // Segments
	heldA    int64           // ALPH and alpha coder bytes in tok.base
	lwSrc    []uint8         // this MB's source luma
	quality  float32
	riff     [20]byte // header arrays here so Encode allocates nothing
	riffA    [46]byte
	tag      [10]byte
}

// Limit sets the page source; nil means make pages.
func (e *Encoder) Limit(l Limiter) { e.tok.lim = l }

// Held is what the last encode kept.
func (e *Encoder) Held() int64 {
	if e.tok.base == 0 {
		return 0
	}
	h := e.tok.base - e.heldA + int64(e.tok.n+1)*pageBytes + int64(cap(e.p0.buf)+cap(e.p1.buf)+cap(e.p2.buf))
	if e.lw != nil {
		h += int64(unsafe.Sizeof(*e.lw)) + 4*int64(cap(e.lw.topDerr))
	}
	if e.lwa != nil {
		h += e.lwa.Held()
	}
	return h + int64(20*cap(e.mbs)+9*cap(e.top)+8*cap(e.rows)+cap(e.alph))
}

// Trim drops the partitions if they hold over n bytes. After an opaque
// encode it also drops alpha state when that and the partitions hold over
// n, so alpha workloads keep theirs. It returns the bytes dropped.
func (e *Encoder) Trim(n int) int64 {
	p := int64(cap(e.p0.buf) + cap(e.p1.buf) + cap(e.p2.buf))
	a := int64(cap(e.alpha) + cap(e.alph))
	if e.lwa != nil {
		a += e.lwa.Held()
	}
	c := int64(0)
	if !e.hasAlpha && a > 0 && p+a > int64(n) {
		e.alpha, e.alph, e.lwa = nil, nil, nil
		e.vl = vp8l.Encoder{} // its bit writer keeps the old ALPH array
		c += a
	}
	if p > int64(n) {
		e.p0.buf, e.p1.buf, e.p2.buf = nil, nil, nil
		c += p
	}
	return c
}

// Pages moves the token pages to dst.
func (e *Encoder) Pages(dst [][]uint16) [][]uint16 {
	dst = append(dst, e.tok.pages...)
	clear(e.tok.pages)
	e.tok.pages = e.tok.pages[:0]
	e.tok.cur = nil
	e.tok.n = 0
	return dst
}

// Swap trades buffers; each keeps its writer, limiter and Cancel.
func (e *Encoder) Swap(o *Encoder) {
	w, l, d := e.w, e.tok.lim, e.done
	*e, *o = *o, *e
	o.w, o.tok.lim, o.done = e.w, e.tok.lim, e.done
	e.w, e.tok.lim, e.done = w, l, d
}

// Reset sets the output writer.
func (e *Encoder) Reset(w io.Writer) { e.w = w }

// ErrCanceled means the Cancel channel closed mid-encode.
var ErrCanceled = errors.New("vp8enc: canceled")

// Cancel sets a channel whose close stops the encode between MB rows;
// nil never stops it.
func (e *Encoder) Cancel(done <-chan struct{}) { e.done = done }

func (e *Encoder) stopped() bool {
	if e.done == nil {
		return false
	}
	select {
	case <-e.done:
		return true
	default:
		return false
	}
}

// MatchLibwebp makes Methods 3 to 6 follow libwebp's decisions and
// bitstream layout, so output equals cwebp's; it is slower.
func (e *Encoder) MatchLibwebp(on bool) { e.lwOn = on }

// AlphaLibwebp16 makes MatchLibwebp ALPH follow libwebp 1.6.0, not 1.2.2.
func (e *Encoder) AlphaLibwebp16(on bool) { e.lwa16 = on }

// LegacyAlpha keeps MatchLibwebp ALPH on webp-go's original writer.
func (e *Encoder) LegacyAlpha(on bool) { e.lwaOld = on }

// Segments 1 puts every MB in one segment, as cwebp -segments 1;
// anything else is the default 4.
func (e *Encoder) Segments(n int) { e.segments = n }

// lwPath reports whether this encode takes the libwebp path.
func (e *Encoder) lwPath() bool { return e.lwOn && e.method >= 3 }

// Encode clamps quality and method.
func (e *Encoder) Encode(m image.Image, quality float32, method int) error {
	e.hasAlpha = false
	e.method = clip(method, 0, 6)
	e.quality = min(max(quality, 0), 100)
	if y, u, v, a, ok := solid(m); ok && !e.lwPath() {
		b := m.Bounds()
		if err := e.alloc(b.Dx(), b.Dy()); err != nil {
			return err
		}
		if a != 255 {
			e.hasAlpha = true
			e.alpha = grow(e.alpha, b.Dx()*b.Dy())
			for i := range e.alpha {
				e.alpha[i] = a
			}
		}
		return e.encodeSolid(y, u, v)
	}
	if err := e.load(m); err != nil {
		return err
	}
	var s segs
	if e.segments == 1 {
		s = e.oneSegment(e.quality)
	} else {
		s = e.analyze(e.quality, e.method)
	}
	if e.stopped() {
		return ErrCanceled
	}
	return e.encodeFrame(&s)
}

// flat reports whether the w x h, n-byte-pixel plane at off is one colour.
// Overlapping compare: row repeats one pixel.
func flat(p []uint8, off, stride, w, h, n int) bool {
	r0 := p[off : off+w*n]
	if !bytes.Equal(r0[n:], r0[:len(r0)-n]) {
		return false
	}
	for y := 1; y < h; y++ {
		if !bytes.Equal(p[off+y*stride:off+y*stride+w*n], r0) {
			return false
		}
	}
	return true
}

// solidRGBA is solid's RGBA and NRGBA case; pre unpremultiplies RGBA.
// A uniform alpha weights chroma the same in every 2x2 block.
func solidRGBA(pix []uint8, off, stride, w, h int, pre bool) (y, u, v, a uint8, ok bool) {
	if len(pix) < off+(h-1)*stride+4*w || !flat(pix, off, stride, w, h, 4) {
		return 0, 0, 0, 0, false
	}
	p := [4]uint8(pix[off : off+4])
	if pre && p[3] != 255 {
		n := color.NRGBAModel.Convert(color.RGBA{p[0], p[1], p[2], p[3]}).(color.NRGBA)
		p = [4]uint8{n.R, n.G, n.B, n.A}
	}
	var c [3]uint8
	d := image.YCbCr{Y: c[0:1], Cb: c[1:2], Cr: c[2:3], YStride: 1, CStride: 1, Rect: image.Rect(0, 0, 1, 1)}
	if !rgbaToYCbCr(&d, p[:], 4, nil) {
		var ab [1]uint8
		rgbaToYCbCr(&d, p[:], 4, ab[:])
	}
	return c[0], c[1], c[2], p[3], true
}

// solid reports the one colour and alpha of m, if it has one.
func solid(m image.Image) (y, u, v, a uint8, ok bool) {
	b := m.Bounds()
	if b.Empty() {
		return 0, 0, 0, 0, false
	}
	w, h := b.Dx(), b.Dy()
	switch m := m.(type) {
	case *image.RGBA:
		return solidRGBA(m.Pix, m.PixOffset(b.Min.X, b.Min.Y), m.Stride, w, h, true)
	case *image.NRGBA:
		return solidRGBA(m.Pix, m.PixOffset(b.Min.X, b.Min.Y), m.Stride, w, h, false)
	case *image.Gray:
		off := m.PixOffset(b.Min.X, b.Min.Y)
		if !flat(m.Pix, off, m.Stride, w, h, 1) {
			return 0, 0, 0, 0, false
		}
		return grayY(m.Pix[off]), 128, 128, 255, true
	case *image.YCbCr:
		if !aligned420(m) {
			return 0, 0, 0, 0, false
		}
		if m.YOffset(b.Max.X-1, b.Max.Y-1) >= len(m.Y) || m.COffset(b.Max.X-1, b.Max.Y-1) >= min(len(m.Cb), len(m.Cr)) {
			return 0, 0, 0, 0, false
		}
		yo := m.YOffset(b.Min.X, b.Min.Y)
		co := m.COffset(b.Min.X, b.Min.Y)
		cw, ch := (w+1)/2, (h+1)/2
		if !flat(m.Y, yo, m.YStride, w, h, 1) || !flat(m.Cb, co, m.CStride, cw, ch, 1) || !flat(m.Cr, co, m.CStride, cw, ch, 1) {
			return 0, 0, 0, 0, false
		}
		return m.Y[yo], m.Cb[co], m.Cr[co], 255, true
	default:
		return 0, 0, 0, 0, false
	}
}

func grow(b []uint8, n int) []uint8 {
	if cap(b) < n {
		return make([]uint8, n)
	}
	return b[:n]
}

func (e *Encoder) alloc(w, h int) error {
	if w < 1 || w > 16383 || h < 1 || h > 16383 {
		return errors.New("webp: dimensions out of range")
	}
	e.mbw, e.mbh = (w+15)/16, (h+15)/16
	e.img.Rect = image.Rect(0, 0, w, h)
	e.img.SubsampleRatio = image.YCbCrSubsampleRatio420
	e.img.YStride, e.img.CStride = e.mbw*16, e.mbw*8
	e.img.Y = grow(e.img.Y, e.mbw*16*e.mbh*16)
	e.img.Cb = grow(e.img.Cb, e.mbw*8*e.mbh*8)
	e.img.Cr = grow(e.img.Cr, e.mbw*8*e.mbh*8)
	if cap(e.mbs) < e.mbw*e.mbh {
		e.mbs = make([]mbInfo, e.mbw*e.mbh)
	}
	e.mbs = e.mbs[:e.mbw*e.mbh]
	if cap(e.top) < e.mbw {
		e.top = make([]nzCtx, e.mbw)
	}
	e.top = e.top[:e.mbw]
	if cap(e.rows) < e.mbh+1 {
		e.rows = make([]tokPos, e.mbh+1)
	}
	e.rows = e.rows[:e.mbh+1]
	return nil
}

func (e *Encoder) load(m image.Image) error {
	e.hasAlpha = false
	e.gray = false
	b := m.Bounds()
	if err := e.alloc(b.Dx(), b.Dy()); err != nil {
		return err
	}
	w, h := b.Dx(), b.Dy()
	switch m := m.(type) {
	case *image.RGBA:
		pix := m.Pix[m.PixOffset(b.Min.X, b.Min.Y):]
		if !rgbaToYCbCr(&e.img, pix, m.Stride, nil) {
			e.loadPremul(pix, m.Stride)
		}
	case *image.NRGBA:
		pix := m.Pix[m.PixOffset(b.Min.X, b.Min.Y):]
		if !rgbaToYCbCr(&e.img, pix, m.Stride, nil) {
			e.hasAlpha = true
			e.alpha = grow(e.alpha, w*h)
			rgbaToYCbCr(&e.img, pix, m.Stride, e.alpha)
		}
	case *image.Gray:
		e.gray = true
		e.loadGray(m)
	case *image.YCbCr:
		// image.NewYCbCr under-allocates for negative Max; At panics.
		if m.YOffset(b.Max.X-1, b.Max.Y-1) >= len(m.Y) || m.COffset(b.Max.X-1, b.Max.Y-1) >= min(len(m.Cb), len(m.Cr)) {
			return errors.New("webp: YCbCr planes shorter than bounds")
		}
		if aligned420(m) {
			e.loadYCbCr(m)
			break
		}
		e.loadGeneric(m, false)
	default:
		op, ok := m.(interface{ Opaque() bool })
		e.loadGeneric(m, !ok || !op.Opaque())
	}
	if e.hasAlpha {
		cleanup(&e.img, e.alpha, w, h)
	}
	e.pad()
	return nil
}

// 16-row strips keep chroma pairs whole. With alpha, strips are NRGBA.
func (e *Encoder) loadGeneric(m image.Image, alpha bool) {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	if alpha {
		e.loadGenericAlpha(m)
		return
	}
	if e.rgba == nil || cap(e.rgba.Pix) < 4*w*16 {
		e.rgba = image.NewRGBA(image.Rect(0, 0, w, 16))
	}
	o := m.Bounds().Min
	for y := 0; y < h; y += 16 {
		sh := min(16, h-y)
		e.rgba.Rect = image.Rect(0, 0, w, sh)
		e.rgba.Stride = 4 * w
		e.rgba.Pix = e.rgba.Pix[:4*w*sh]
		draw.Draw(e.rgba, e.rgba.Rect, m, o.Add(image.Pt(0, y)), draw.Src)
		strip := e.img
		strip.Rect = image.Rect(0, 0, w, sh)
		strip.Y = e.img.Y[y*e.img.YStride:]
		strip.Cb = e.img.Cb[y/2*e.img.CStride:]
		strip.Cr = e.img.Cr[y/2*e.img.CStride:]
		rgbaToYCbCr(&strip, e.rgba.Pix, e.rgba.Stride, nil)
	}
}

func (e *Encoder) loadGenericAlpha(m image.Image) {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	if e.nrgba == nil || cap(e.nrgba.Pix) < 4*w*16 {
		e.nrgba = image.NewNRGBA(image.Rect(0, 0, w, 16))
	}
	e.alpha = grow(e.alpha, w*h)
	o := m.Bounds().Min
	for y := 0; y < h; y += 16 {
		sh := min(16, h-y)
		e.nrgba.Rect = image.Rect(0, 0, w, sh)
		e.nrgba.Stride = 4 * w
		e.nrgba.Pix = e.nrgba.Pix[:4*w*sh]
		draw.Draw(e.nrgba, e.nrgba.Rect, m, o.Add(image.Pt(0, y)), draw.Src)
		strip := e.img
		strip.Rect = image.Rect(0, 0, w, sh)
		strip.Y = e.img.Y[y*e.img.YStride:]
		strip.Cb = e.img.Cb[y/2*e.img.CStride:]
		strip.Cr = e.img.Cr[y/2*e.img.CStride:]
		rgbaToYCbCr(&strip, e.nrgba.Pix, e.nrgba.Stride, e.alpha[y*w:(y+sh)*w])
	}
	e.hasAlpha = !opaque1(e.alpha)
}

// loadPremul converts premultiplied RGBA in 16-row NRGBA strips, as
// color.NRGBAModel does; a cut-off last alpha counts as opaque.
func (e *Encoder) loadPremul(pix []uint8, stride int) {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	if e.nrgba == nil || cap(e.nrgba.Pix) < 4*w*16 {
		e.nrgba = image.NewNRGBA(image.Rect(0, 0, w, 16))
	}
	e.alpha = grow(e.alpha, w*h)
	for y := 0; y < h; y += 16 {
		sh := min(16, h-y)
		d := e.nrgba.Pix[:4*w*sh]
		for j := 0; j < sh; j++ {
			src, dst := pix[(y+j)*stride:], d[4*w*j:][:4*w]
			for i := 0; i < 4*w; i += 4 {
				a := uint32(255)
				if i+3 < len(src) {
					a = uint32(src[i+3])
				}
				switch a {
				case 255:
					copy(dst[i:i+3], src[i:i+3])
				case 0:
					dst[i], dst[i+1], dst[i+2] = 0, 0, 0
				default:
					a16 := a * 0x101
					for c := 0; c < 3; c++ {
						dst[i+c] = uint8(uint32(src[i+c]) * 0x101 * 0xffff / a16 >> 8)
					}
				}
				dst[i+3] = uint8(a)
			}
		}
		strip := e.img
		strip.Rect = image.Rect(0, 0, w, sh)
		strip.Y = e.img.Y[y*e.img.YStride:]
		strip.Cb = e.img.Cb[y/2*e.img.CStride:]
		strip.Cr = e.img.Cr[y/2*e.img.CStride:]
		rgbaToYCbCr(&strip, d, 4*w, e.alpha[y*w:(y+sh)*w])
	}
	e.hasAlpha = true
}

func opaque1(a []uint8) bool {
	for _, v := range a {
		if v != 255 {
			return false
		}
	}
	return true
}

// cleanup is WebPCleanupTransparentArea on YUVA, picture_tools_enc.c:104-181:
// fully transparent 8x8 blocks take the first colour of their run, and
// transparent pixels of other blocks take the mean luma of the opaque ones.
func cleanup(img *image.YCbCr, a []uint8, w, h int) {
	const s = 8
	ys, cs := img.YStride, img.CStride
	y := 0
	for ; y+s <= h; y += s {
		reset := true
		var vy, vu, vv uint8
		x := 0
		for ; x+s <= w; x += s {
			if !smoothen(a[y*w+x:], w, img.Y[y*ys+x:], ys, s, s) {
				reset = true
				continue
			}
			co := y/2*cs + x/2
			if reset {
				vy, vu, vv = img.Y[y*ys+x], img.Cb[co], img.Cr[co]
				reset = false
			}
			for j := 0; j < s; j++ {
				fill8(img.Y[(y+j)*ys+x:][:s], vy)
			}
			for j := 0; j < s/2; j++ {
				for i := 0; i < s/2; i++ {
					img.Cb[co+j*cs+i], img.Cr[co+j*cs+i] = vu, vv
				}
			}
		}
		if x < w {
			smoothen(a[y*w+x:], w, img.Y[y*ys+x:], ys, w-x, s)
		}
	}
	if y < h {
		x := 0
		for ; x+s <= w; x += s {
			smoothen(a[y*w+x:], w, img.Y[y*ys+x:], ys, s, h-y)
		}
		if x < w {
			smoothen(a[y*w+x:], w, img.Y[y*ys+x:], ys, w-x, h-y)
		}
	}
}

// smoothen is SmoothenBlock: it reports a fully transparent block.
func smoothen(a []uint8, as int, l []uint8, ls, w, h int) bool {
	sum, n := 0, 0
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			if a[j*as+i] != 0 {
				n++
				sum += int(l[j*ls+i])
			}
		}
	}
	if n > 0 && n < w*h {
		avg := uint8(sum / n)
		for j := 0; j < h; j++ {
			for i := 0; i < w; i++ {
				if a[j*as+i] == 0 {
					l[j*ls+i] = avg
				}
			}
		}
	}
	return n == 0
}

// grayY is limited-range luma from an 8-bit gray sample.
func grayY(g uint8) uint8 { return uint8((56318*uint32(g) + 1<<15 + 16<<16) >> 16) }

// grayYLUT caches grayY per sample.
var grayYLUT [256]uint8

func init() {
	for i := range grayYLUT {
		grayYLUT[i] = grayY(uint8(i))
	}
}

// loadGray applies limited-range luma, chroma 128.
func (e *Encoder) loadGray(m *image.Gray) {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	for y := 0; y < h; y++ {
		src := m.Pix[m.PixOffset(m.Rect.Min.X, m.Rect.Min.Y+y):]
		dst := e.img.Y[y*e.img.YStride:][:w]
		for x, g := range src[:w] {
			dst[x] = grayYLUT[g]
		}
	}
	fill128(e.img.Cb)
	fill128(e.img.Cr)
}

// fill128 sets b to 128 by doubling copies.
func fill128(b []uint8) {
	n := 8
	if n > len(b) {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		b[i] = 128
	}
	for n < len(b) {
		n += copy(b[n:], b[:n])
	}
}

// aligned420 reports whether COffset aligns to whole chroma cells.
// COffset truncates, so cells align only at non-negative even origins.
func aligned420(m *image.YCbCr) bool {
	b := m.Bounds()
	return m.SubsampleRatio == image.YCbCrSubsampleRatio420 &&
		b.Min.X >= 0 && b.Min.Y >= 0 && b.Min.X&1 == 0 && b.Min.Y&1 == 0
}

func (e *Encoder) loadYCbCr(m *image.YCbCr) {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	for y := 0; y < h; y++ {
		o := m.YOffset(m.Rect.Min.X, m.Rect.Min.Y+y)
		copy(e.img.Y[y*e.img.YStride:], m.Y[o:o+w])
	}
	cw := (w + 1) / 2
	for cy := 0; cy < (h+1)/2; cy++ {
		o := m.COffset(m.Rect.Min.X, m.Rect.Min.Y+2*cy)
		copy(e.img.Cb[cy*e.img.CStride:], m.Cb[o:o+cw])
		copy(e.img.Cr[cy*e.img.CStride:], m.Cr[o:o+cw])
	}
}

func padPlane(p []uint8, stride, w, h int) {
	for y := 0; y < h; y++ {
		row := p[y*stride : (y+1)*stride]
		for x := w; x < stride; x++ {
			row[x] = row[w-1]
		}
	}
	last := p[(h-1)*stride : h*stride]
	for y := h * stride; y < len(p); y += stride {
		copy(p[y:y+stride], last)
	}
}

// Edge replication to the macroblock grid.
func (e *Encoder) pad() {
	w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
	padPlane(e.img.Y, e.img.YStride, w, h)
	padPlane(e.img.Cb, e.img.CStride, (w+1)/2, (h+1)/2)
	padPlane(e.img.Cr, e.img.CStride, (w+1)/2, (h+1)/2)
}

// startTokens readies the token buffer for a fresh frame.
func (e *Encoder) startTokens() {
	e.tok.base = int64(cap(e.img.Y) + cap(e.img.Cb) + cap(e.img.Cr) + cap(e.alpha))
	if e.rgba != nil {
		e.tok.base += int64(cap(e.rgba.Pix))
	}
	if e.nrgba != nil {
		e.tok.base += int64(cap(e.nrgba.Pix))
	}
	// Alpha state held from earlier encodes stays inside the charge.
	e.heldA = int64(cap(e.alph))
	if e.lwa != nil {
		e.heldA += e.lwa.Held()
	}
	e.tok.base += e.heldA
	e.tok.parts = parts0(e.mbw * e.mbh)
	e.tok.reset()
}

func (e *Encoder) encodeFrame(s *segs) error {
	for i := range e.segQ {
		e.segQ[i] = newQuant(s.quant[i], s.dqUVAC)
		e.segLam[i] = newLambdas(&e.segQ[i])
	}

	e.startTokens()
	for i := range e.top {
		e.top[i] = nzCtx{}
	}
	lw := e.lwPath()
	if lw {
		if e.lw == nil {
			e.lw = new(lwState)
		}
		e.lw.reset(e.mbw, e.mbh, e.quality)
	}
	nskip := 0
	for mby := 0; mby < e.mbh; mby++ {
		if e.stopped() {
			return ErrCanceled
		}
		e.rows[mby] = e.tok.pos()
		var left nzCtx
		if lw {
			e.lw.leftDerr = [2][2]int8{}
		}
		for mbx := 0; mbx < e.mbw; mbx++ {
			if t := &e.tok; t.lim != nil && t.n+1 >= len(t.pages) && !t.refill() {
				return ErrRetry
			}
			if lw {
				e.lw.refresh()
				e.encodeMBLW(mbx, mby, &left)
			} else if e.encodeMB(mbx, mby, &left) {
				nskip++
			}
		}
	}
	e.rows[e.mbh] = e.tok.pos()
	if lw {
		e.adjustFilter(s)
	}
	return e.writeFrame(s, nskip)
}

// Exact at qi 0; DC predicts the rest.
func (e *Encoder) encodeSolid(y, u, v uint8) error {
	e.startTokens()
	for i := range e.mbs {
		e.mbs[i] = mbInfo{skip: true, i16: true}
	}
	nskip := len(e.mbs)
	e.rows[0] = e.tok.pos()
	if y != 128 || u != 128 || v != 128 {
		if t := &e.tok; t.lim != nil && t.n+1 >= len(t.pages) && !t.refill() {
			return ErrRetry
		}
		e.lv = [25][16]int16{}
		e.lv[24][0] = 8 * (int16(y) - 128)
		for c := 16; c < 20; c++ {
			e.lv[c][0] = 2 * (int16(u) - 128)
			e.lv[c+4][0] = 2 * (int16(v) - 128)
		}
		var l, t nzCtx
		recordLuma(&e.tok, true, &e.lv, &l, &t)
		recordChroma(&e.tok, &e.lv, &l, &t)
		e.mbs[0].skip = false
		nskip--
	} else if !e.tok.more(0) { // charge parts
		return ErrRetry
	}
	for i := 1; i < len(e.rows); i++ {
		e.rows[i] = e.tok.pos()
	}
	return e.writeFrame(&segs{n: 1}, nskip)
}

// writeFrame writes the header, mode bits and tokens for nskip skipped MBs.
func (e *Encoder) writeFrame(s *segs, nskip int) error {
	n := e.mbw * e.mbh
	skipProb := uint8((n - nskip) * 255 / n)
	lw := e.lwPath()
	if lw {
		e.lw.finalize()
		e.probs = e.lw.probs
	} else {
		finalizeProbs(&e.tok, &e.probs)
	}

	e.p0.reset()
	putFrameHeader(&e.p0, s, skipProb, &e.probs, lw)
	for mby := 0; mby < e.mbh; mby++ {
		for mbx := 0; mbx < e.mbw; mbx++ {
			mb := &e.mbs[mby*e.mbw+mbx]
			// tree_enc.c:278-281.
			if s.updateMap {
				if e.p0.putBit(mb.seg >= 2, s.probs[0]) {
					e.p0.putBit(mb.seg&1 == 1, s.probs[2])
				} else {
					e.p0.putBit(mb.seg&1 == 1, s.probs[1])
				}
			}
			if !lw {
				e.p0.putBit(mb.skip, skipProb)
			}
			if e.p0.putBit(mb.i16, 145) {
				putI16Mode(e.p0.putBit, mb.modes[0])
			} else {
				for n := 0; n < 16; n++ {
					above, left := e.ctxModes(mbx, mby, n)
					putI4Mode(e.p0.putBit, mb.modes[n], &predProb[above][left])
				}
			}
			putUVMode(e.p0.putBit, mb.uv)
		}
	}
	a := e.p0.finish()
	if len(a) >= 1<<19 {
		return errors.New("webp: first partition too large")
	}
	e.p1.reset()
	e.p2.reset()
	e.tag = frameTag(e.img.Rect.Dx(), e.img.Rect.Dy(), len(a))
	if lw {
		// One partition, and the chunk size counts the pad, syntax_enc.c:333-341.
		e.tok.emit1(&e.p1, &e.probs)
		return e.writeRIFF(true, e.tag[:], a, e.p1.finish())
	}
	e.tok.emit2(&e.p1, &e.p2, e.rows, &e.probs)
	b, c := e.p1.finish(), e.p2.finish()
	if len(b) >= 1<<24 {
		return errors.New("webp: token partition too large")
	}
	e.psz = [3]byte{byte(len(b)), byte(len(b) >> 8), byte(len(b) >> 16)}
	return e.writeRIFF(false, e.tag[:], a, e.psz[:], b, c)
}

// writeRIFF wraps the frame, with VP8X and ALPH when the image has alpha.
// padSize counts the pad in the VP8 chunk size, as libwebp does.
func (e *Encoder) writeRIFF(padSize bool, chunk ...[]byte) error {
	if e.hasAlpha {
		w, h := e.img.Rect.Dx(), e.img.Rect.Dy()
		if e.lwPath() && e.method <= 5 && !e.lwaOld {
			if e.lwa == nil {
				e.lwa = new(vp8l.LW)
			}
			e.lwa.Libwebp16 = e.lwa16
			e.alph = e.lwa.Alpha(e.alph[:0], e.alpha[:w*h], w, h, e.method)
		} else {
			// ponytail: method 6 keeps webp-go's original ALPH; port libwebp's q100 brute force to match it.
			// Lossless, no filter, no level reduction.
			e.alph = e.vl.Alpha(append(e.alph[:0], 1), e.alpha[:w*h], w, h)
		}
		// No cheap bound on ALPH: charge it and its output copy once built,
		// still before anything is written.
		a := 2*int64(cap(e.alph)) - e.heldA // base has the start's
		if e.lwa != nil {
			a += e.lwa.Held()
		}
		if !e.tok.more(a) {
			return ErrRetry
		}
		return riff.WriteAlpha(e.w, &e.riffA, e.alph, w, h, padSize, chunk...)
	}
	return riff.Write(e.w, &e.riff, padSize, chunk...)
}

// residualGo fills blk with the fdct of src minus the prediction at (y, x).
// Fused so the residual never round-trips memory.
func (e *Encoder) residualGo(blk *[16]int16, src []uint8, stride, y, x int) {
	z := &e.z
	t00, t01, t02, t03 := fdctRow(diff4(src, z[y][x:]))
	t10, t11, t12, t13 := fdctRow(diff4(src[stride:], z[y+1][x:]))
	t20, t21, t22, t23 := fdctRow(diff4(src[2*stride:], z[y+2][x:]))
	t30, t31, t32, t33 := fdctRow(diff4(src[3*stride:], z[y+3][x:]))
	blk[0], blk[4], blk[8], blk[12] = fdctCol(t00, t10, t20, t30)
	blk[1], blk[5], blk[9], blk[13] = fdctCol(t01, t11, t21, t31)
	blk[2], blk[6], blk[10], blk[14] = fdctCol(t02, t12, t22, t32)
	blk[3], blk[7], blk[11], blk[15] = fdctCol(t03, t13, t23, t33)
}

// diff4 is one row of s minus p.
func diff4(s, p []uint8) (int32, int32, int32, int32) {
	s, p = s[:4], p[:4]
	return int32(s[0]) - int32(p[0]), int32(s[1]) - int32(p[1]),
		int32(s[2]) - int32(p[2]), int32(s[3]) - int32(p[3])
}

// fdctRow is one fdct row pass.
func fdctRow(d0, d1, d2, d3 int32) (int32, int32, int32, int32) {
	a0, a1, a2, a3 := d0+d3, d1+d2, d1-d2, d0-d3
	return (a0 + a1) * 8, (a2*2217 + a3*5352 + 1812) >> 9,
		(a0 - a1) * 8, (a3*2217 - a2*5352 + 937) >> 9
}

// fdctCol is one fdctCols column.
func fdctCol(t0, t1, t2, t3 int32) (int16, int16, int16, int16) {
	a0, a1, a2, a3 := t0+t3, t1+t2, t1-t2, t0-t3
	v := (a2*2217 + a3*5352 + 12000) >> 16
	if a3 != 0 {
		v++
	}
	return int16((a0 + a1 + 7) >> 4), int16(v), int16((a0 - a1 + 7) >> 4),
		int16((a3*2217 - a2*5352 + 51000) >> 16)
}

// ctxModes gives the neighbouring i4 modes of block n.
func (e *Encoder) ctxModes(mbx, mby, n int) (above, left uint8) {
	mb := &e.mbs[mby*e.mbw+mbx]
	above, left = predDC, predDC
	if n >= 4 {
		above = mb.modes[n-4]
	} else if mby > 0 {
		above = e.mbs[(mby-1)*e.mbw+mbx].modes[12+n]
	}
	if n&3 != 0 {
		left = mb.modes[n-1]
	} else if mbx > 0 {
		left = e.mbs[mby*e.mbw+mbx-1].modes[n+3]
	}
	return above, left
}

// i16Quant codes the luma residual after predMB into dq and
// returns the transform-domain distortion.
// Trellis contexts follow libwebp quant_enc.c:772-781.
func (e *Encoder) i16Quant(dq *[16][16]int16, srcY []uint8, ys int, l, t nzCtx, tr bool) int {
	var dc [16]int16
	d := 0
	e.residualsY(dq, srcY, ys)
	if !tr {
		d = e.quantY16(dq, &dc)
	} else {
		for n := range dq {
			b := &dq[n]
			dc[n] = b[0]
			b[0] = 0
			raw := *b
			nz := e.q.y1.trellis(b, &e.lv[n], int(l[n/4]+t[n%4]), 0, e.lam.ti16)
			e.lv[n][0] = 0
			l[n/4], t[n%4] = b2u(nz), b2u(nz)
			d += coeffSSE(&raw, b)
		}
	}
	raw := dc
	fwht(&dc)
	e.q.y2.quantize(&dc, &e.lv[24])
	q := iwhtY(&dc)
	for n := range dq {
		dq[n][0] = q[n]
		k := int(raw[n]) - int(q[n])
		d += k * k
	}
	// fdct gain is 2, so coefficient SSE is 4x pixel SSE.
	return (d + 2) >> 2
}

func coeffSSE(a, b *[16]int16) int {
	d := 0
	for i, v := range a {
		k := int(v) - int(b[i])
		d += k * k
	}
	return d
}

func b2u(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

// i4Shortlist ranks the i4 modes by prediction SSE plus mode
// rate and returns a bit mask of the 3 best. It saves every
// prediction in pr so trials skip recomputing them.
// Methods 5 and 6 try every mode.
func (e *Encoder) i4Shortlist(pr *[10][4]uint32, src []uint8, ys, y, x int, above, lft uint8) uint16 {
	e.z.pred4All(pr, y, x)
	if e.method >= 5 {
		return 1<<10 - 1
	}
	var d [10]int32
	sse4x10(&d, pr, src, ys)
	c := &i4Cost[above][lft]
	li := e.lam.i4
	a, b, k := math.MaxInt, math.MaxInt, math.MaxInt
	// Indexed: a range copy stalls forwarding.
	for m := range d {
		// Mode in low bits keeps ties stable.
		v := (256*int(d[m])+c[m]*li)<<4 | m
		v, a = max(a, v), min(a, v)
		v, b = max(b, v), min(b, v)
		k = min(k, v)
	}
	return 1<<(a&15) | 1<<(b&15) | 1<<(k&15)
}

// sse4x10Go sets d[m] to the SSE of the 4x4 src block against
// prediction m.
func sse4x10Go(d *[10]int32, pr *[10][4]uint32, src []uint8, stride int) {
	var r [4]uint32
	for j := range r {
		r[j] = binary.LittleEndian.Uint32(src[j*stride:])
	}
	for m := range pr {
		n := 0
		for j, p := range pr[m] {
			k0 := int(r[j]&255) - int(p&255)
			k1 := int(r[j]>>8&255) - int(p>>8&255)
			k2 := int(r[j]>>16&255) - int(p>>16&255)
			k3 := int(r[j]>>24) - int(p>>24)
			n += k0*k0 + k1*k1 + k2*k2 + k3*k3
		}
		d[m] = int32(n)
	}
}

// put4 writes a saved 4x4 prediction at (y, x).
func (z *ybrBuf) put4(p *[4]uint32, y, x int) {
	for j, v := range p {
		binary.LittleEndian.PutUint32(z[y+j][x:x+4], v)
	}
}

// best1 returns a bit mask of the lowest score.
func best1(s *[4]int) uint8 {
	b := 0
	for m := 1; m < 4; m++ {
		if s[m] < s[b] {
			b = m
		}
	}
	return 1 << b
}

// predSSEGo is the SSE between src and the w x h prediction at (y, x).
func (z *ybrBuf) predSSEGo(src []uint8, stride, y, x, w, h int) int {
	d := 0
	for j := 0; j < h; j++ {
		r := src[j*stride : j*stride+w]
		p := z[y+j][x : x+w]
		for i, v := range r {
			k := int(v) - int(p[i])
			d += k * k
		}
	}
	return d
}

// quant4 quantizes one i4 block, with trellis when tr.
// Returns the last nonzero index, or -1.
func (e *Encoder) quant4(blk, lv *[16]int16, ctx int, tr bool) int {
	if tr {
		e.q.y1.trellis(blk, lv, ctx, 3, e.lam.ti4)
		return lastNZ(lv, 0)
	}
	return e.q.y1.quantize(blk, lv)
}

// uvQuant codes the chroma residual after predMB into dq and
// returns the transform-domain distortion.
func (e *Encoder) uvQuant(dq *[8][16]int16, srcU, srcV []uint8, cs int) int {
	d := 0
	for c := range dq {
		src, bx := srcU, 8
		if c >= 4 {
			src, bx = srcV, 24
		}
		oy, ox := 4*((c&3)>>1), 4*(c&1)
		e.residual(&dq[c], src[oy*cs+ox:], cs, 18+oy, bx+ox)
		_, k := e.q.uv.quantSSE(&dq[c], &e.lv[16+c])
		d += k
	}
	return (d + 2) >> 2
}

// libwebp RD_OPT_BASIC with SSE only, quant_enc.c:982-1196.
func (e *Encoder) encodeMB(mbx, mby int, left *nzCtx) bool {
	z := &e.z
	z.prepare(&e.img, mbx, mby, e.mbw)
	ys := e.img.YStride
	srcY := e.img.Y[mby*16*ys+mbx*16:]
	mb := &e.mbs[mby*e.mbw+mbx]
	e.q, e.lam = e.segQ[mb.seg], e.segLam[mb.seg]
	lam := &e.lam
	// Trials match the final pass except at method 5.
	reuse := e.method != 5

	if e.dcExact(srcY, ys, mbx, mby) {
		for n := range mb.modes {
			mb.modes[n] = predDC
		}
		*left, e.top[mbx] = nzCtx{}, nzCtx{}
		*mb = mbInfo{skip: true, i16: true, modes: mb.modes, seg: mb.seg}
		return true
	}

	best16, bs, s16 := uint8(0), -1, 0
	// Trials alternate buffers so the best is never copied.
	var dq16 [2][16][16]int16
	b16 := 0
	var lv16 [17][16]int16
	try := uint8(15)
	if e.method < 5 {
		var s [4]int
		z.edgeSSE(&s, srcY, ys, 16, 1, 8, mbx, mby)
		for m := range s {
			s[m] = 256*s[m] + i16Cost[m]*lam.i16
		}
		try = best1(&s)
	}
	for m := uint8(0); m < 4; m++ {
		if try&(1<<m) == 0 {
			continue
		}
		z.predMB(16, 1, 8, m, mbx, mby)
		d := e.i16Quant(&dq16[b16^1], srcY, ys, *left, e.top[mbx], e.method >= 6)
		r := costLuma16Nat(&e.lv, *left, e.top[mbx]) + i16Cost[m]
		if s := r*lam.i16 + 256*d; bs < 0 || s < bs {
			best16, bs, s16 = m, s, r*lam.mode+256*d
			b16 ^= 1
			if reuse {
				copy(lv16[:16], e.lv[:16])
				lv16[16] = e.lv[24]
			}
		}
	}

	var blk, lv [16]int16
	i16 := false
	score := bitCost[0][145] * lam.mode
	l, t := *left, e.top[mbx]
	var b3, l3 [3][16]int16
	var r3 [3][2]int
	// pred4All rewrites all of pr.
	var pr [10][4]uint32
	for n := 0; n < 16; n++ {
		y, x := 1+4*(n/4), 8+4*(n%4)
		src := srcY[(y-1)*ys+x-8:]
		above, lft := e.ctxModes(mbx, mby, n)
		ctx := int(l[n/4] + t[n%4])
		best, bs, bScore := uint8(0), -1, 0
		var dq4, lv4 [16]int16
		try := e.i4Shortlist(&pr, src, ys, y, x, above, lft)
		batched := e.method < 5 && e.residualQuant3(&b3, &l3, &r3, &pr, try, src, ys)
		k, kb := 0, 0
		for tm := try; tm != 0; tm &= tm - 1 {
			m := uint8(bits.TrailingZeros16(tm))
			var last, d int
			lp := &lv
			if batched {
				lp = &l3[k]
				last, d = r3[k][0], r3[k][1]
			} else if e.method >= 6 {
				z.put4(&pr[m], y, x)
				e.residual(&blk, src, ys, y, x)
				raw := blk
				last = e.quant4(&blk, &lv, ctx, true)
				d = coeffSSE(&raw, &blk)
			} else {
				last, d = e.residualQuant(&blk, &lv, &pr[m], src, ys, y, x)
			}
			// fdct gain is 2, so coefficient SSE is 4x pixel SSE.
			d = (d + 2) >> 2
			c := cost4(ctx, last, lp)
			r := c + i4Cost[above][lft][m]
			if s := r*lam.i4 + 256*d; bs < 0 || s < bs {
				best, bs, bScore = m, s, r*lam.mode+256*d
				kb = k
				if !batched {
					dq4, lv4 = blk, lv
				}
			}
			k++
		}
		score += bScore
		if score >= s16 {
			i16 = true
			break
		}
		if reuse {
			z.put4(&pr[best], y, x)
			if batched {
				z.idctAdd(&b3[kb], y, x)
				e.lv[n] = l3[kb]
			} else {
				z.idctAdd(&dq4, y, x)
				e.lv[n] = lv4
			}
		} else {
			z.put4(&pr[best], y, x)
			e.residual(&blk, src, ys, y, x)
			e.quant4(&blk, &e.lv[n], ctx, e.method >= 5)
			z.idctAdd(&blk, y, x)
		}
		mb.modes[n] = best
		nz := uint8(0)
		if batched {
			// last is -1 exactly when all levels are zero.
			nz = b2u(r3[kb][0] >= 0)
		} else if e.lv[n] != [16]int16{} {
			nz = 1
		}
		l[n/4], t[n%4] = nz, nz
	}

	if i16 {
		z.predMB(16, 1, 8, best16, mbx, mby)
		dq := &dq16[b16]
		if reuse {
			copy(e.lv[:16], lv16[:16])
			e.lv[24] = lv16[16]
		} else {
			e.i16Quant(dq, srcY, ys, *left, e.top[mbx], e.method >= 5)
		}
		z.idctAddsY(dq)
		for n := range mb.modes {
			mb.modes[n] = best16
		}
	}

	cs := e.img.CStride
	srcU := e.img.Cb[mby*8*cs+mbx*8:]
	srcV := e.img.Cr[mby*8*cs+mbx*8:]
	var bestUV uint8
	if e.gray && e.method < 5 {
		// Gray chroma is flat 128; DC prediction needs no trial or residual.
		z.predMB(8, 18, 8, predDC, mbx, mby)
		z.predMB(8, 18, 24, predDC, mbx, mby)
		clear(e.lv[16:24])
	} else if e.method < 5 {
		var s, sv [4]int
		z.edgeSSE(&s, srcU, cs, 8, 18, 8, mbx, mby)
		z.edgeSSE(&sv, srcV, cs, 8, 18, 24, mbx, mby)
		for m := range s {
			s[m] = 256*(s[m]+sv[m]) + uvCost[m]*lam.uv
		}
		// Single trial: RD score unused.
		bestUV = uint8(bits.TrailingZeros8(best1(&s)))
		z.predMB(8, 18, 8, bestUV, mbx, mby)
		z.predMB(8, 18, 24, bestUV, mbx, mby)
		var dq [8][16]int16
		e.residualsUV(&dq, srcU, srcV, cs)
		e.q.uv.quantizeN(dq[:], e.lv[16:24])
		z.idctAddsUV(&dq)
	} else {
		sUV := -1
		var dqUV [2][8][16]int16
		var lvUV [8][16]int16
		bUV := 0
		for m := uint8(0); m < 4; m++ {
			z.predMB(8, 18, 8, m, mbx, mby)
			z.predMB(8, 18, 24, m, mbx, mby)
			d := e.uvQuant(&dqUV[bUV^1], srcU, srcV, cs)
			r := costUVNat(&e.lv, *left, e.top[mbx]) + uvCost[m]
			if s := r*lam.uv + 256*d; sUV < 0 || s < sUV {
				bestUV, sUV = m, s
				bUV ^= 1
				copy(lvUV[:], e.lv[16:24])
			}
		}
		z.predMB(8, 18, 8, bestUV, mbx, mby)
		z.predMB(8, 18, 24, bestUV, mbx, mby)
		for c := range dqUV[bUV] {
			oy, ox := 4*((c&3)>>1), 4*(c&1)
			z.idctAdd(&dqUV[bUV][c], 18+oy, 8+16*(c>>2)+ox)
		}
		copy(e.lv[16:24], lvUV[:])
	}

	skip := !i16 || e.lv[24] == [16]int16{}
	for n := 0; n < 24 && skip; n++ {
		skip = e.lv[n] == [16]int16{}
	}
	if skip {
		if i16 {
			*left, e.top[mbx] = nzCtx{}, nzCtx{}
		} else {
			for i := 0; i < 8; i++ {
				left[i], e.top[mbx][i] = 0, 0
			}
		}
	} else {
		recordLuma(&e.tok, i16, &e.lv, left, &e.top[mbx])
		recordChroma(&e.tok, &e.lv, left, &e.top[mbx])
	}

	for j := 0; j < 16; j++ {
		copy(srcY[j*ys:j*ys+16], z[1+j][8:24])
	}
	// Gray chroma is already flat 128; no copy-back needed.
	if !e.gray || e.method >= 5 {
		for j := 0; j < 8; j++ {
			copy(srcU[j*cs:j*cs+8], z[18+j][8:16])
			copy(srcV[j*cs:j*cs+8], z[18+j][24:32])
		}
	}
	*mb = mbInfo{skip: skip, i16: i16, modes: mb.modes, uv: bestUV, seg: mb.seg}
	return skip
}

// dcExact reports DC predicts the MB exactly.
func (e *Encoder) dcExact(srcY []uint8, ys, mbx, mby int) bool {
	z := &e.z
	if !flat(srcY, 0, ys, 16, 16, 1) {
		return false
	}
	cs := e.img.CStride
	srcU := e.img.Cb[mby*8*cs+mbx*8:]
	srcV := e.img.Cr[mby*8*cs+mbx*8:]
	if !flat(srcU, 0, cs, 8, 8, 1) || !flat(srcV, 0, cs, 8, 8, 1) {
		return false
	}
	z.predMB(16, 1, 8, predDC, mbx, mby)
	z.predMB(8, 18, 8, predDC, mbx, mby)
	z.predMB(8, 18, 24, predDC, mbx, mby)
	return z[1][8] == srcY[0] && z[18][8] == srcU[0] && z[18][24] == srcV[0]
}
