// Port of libwebp 1.2.2 enc/alpha_enc.c, utils/filters_utils.c and dsp/filters.c. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import (
	"slices"
	"unsafe"
)

// Alpha appends the ALPH chunk data (header byte, then payload) of a
// (w x h, stride w) to dst, as libwebp 1.2.2 EncodeAlpha with
// alpha_quality 100, alpha_compression 1 and alpha_filtering 1.
// method is the WebP method, 0 to 5.
//
// EncodeAlpha, alpha_enc.c:292.
func (e *LW) Alpha(dst []byte, a []uint8, w, h, method int) []byte {
	e.cfg = lwConfig{method: method, quality: 8 * method, effort: method}
	return e.applyFiltersAndEncode(dst, a[:w*h], w, h)
}

// Held is the scratch e keeps, in bytes.
// e.costMgr.distArray aliases e.distArray.
func (e *LW) Held() int64 {
	n := int(unsafe.Sizeof(*e))
	n += cap(e.filt) + cap(e.trial) + cap(e.bwBest.buf) + cap(e.bufRle) + cap(e.palRow)
	n += 4 * (cap(e.pic) + cap(e.argb) + cap(e.argbScratch) + cap(e.transformData) + cap(e.histoArgb))
	n += 4 * (cap(e.hc.offLen) + cap(e.hcBox.offLen) + cap(e.hcHisto.offLen) + cap(e.hashFirst) + cap(e.costMgr.costs))
	n += 2 * (cap(e.boxCounts) + cap(e.distArray) + cap(e.histoSymbols) + cap(e.mapTmp) + cap(e.tokens))
	n += 8 * cap(e.mappings)
	for _, r := range e.refs {
		n += int(unsafe.Sizeof(pixOrCopy{})) * cap(r)
	}
	n += int(unsafe.Sizeof(lwCostCacheInterval{})) * cap(e.costMgr.cacheIntervals)
	for _, s := range []*lwHistoSet{&e.histoImage, &e.histoOrig, &e.histoNH} {
		n += 8*cap(s.h) + int(unsafe.Sizeof(lwHisto{}))*cap(s.pool)
	}
	n += int(unsafe.Sizeof(lwHistoPair{})) * cap(e.queue.q)
	for _, c := range []*lwHuffCodes{&e.huff, &e.huffNH} {
		n += int(unsafe.Sizeof(huffTreeCode{}))*cap(c.codes) + cap(c.lens) + 2*cap(c.bits)
	}
	n += int(unsafe.Sizeof(huffTree{})) * cap(e.huffTree)
	if e.v16 != nil {
		n += e.v16.held()
	}
	return int64(n)
}

// applyFiltersAndEncode appends the smallest filter trial to dst.
// ApplyFiltersAndEncode, alpha_enc.c:234.
func (e *LW) applyFiltersAndEncode(dst []byte, a []uint8, w, h int) []byte {
	tryMap := getFilterMap(a, w, h, e.cfg.effort)
	if tryMap == 1<<filterNone {
		return e.encodeAlphaInternal(dst, a, w, h, filterNone)
	}
	if cap(e.filt) < w*h {
		e.filt = make([]uint8, w*h)
	}
	e.filt = e.filt[:w*h]
	base := len(dst)
	best := int(^uint(0) >> 1) // C uses 2^32-1, never reached
	for filter := filterNone; tryMap != 0; filter, tryMap = filter+1, tryMap>>1 {
		if tryMap&1 != 0 {
			e.trial = e.encodeAlphaInternal(e.trial[:0], a, w, h, filter)
			if len(e.trial) < best {
				dst = append(dst[:base], e.trial...)
				best = len(e.trial)
			}
		}
	}
	return dst
}

// encodeAlphaInternal appends header + payload for one filter to dst.
// EncodeAlphaInternal, alpha_enc.c:109.
func (e *LW) encodeAlphaInternal(dst []byte, a []uint8, w, h, filter int) []byte {
	src := a
	switch filter {
	case filterHorizontal:
		horizontalFilter(a, w, h, e.filt)
		src = e.filt
	case filterVertical:
		verticalFilter(a, w, h, e.filt)
		src = e.filt
	case filterGradient:
		gradientFilter(a, w, h, e.filt)
		src = e.filt
	}
	start := len(dst)
	method := 1 // ALPHA_LOSSLESS_COMPRESSION
	dst = e.encodeLossless(append(dst, 0), src, w, h)
	if len(dst)-start-1 > w*h {
		method = 0 // ALPHA_NO_COMPRESSION
		dst = append(dst[:start+1], src...)
	}
	dst[start] = byte(method | filter<<2)
	return dst
}

// encodeLossless appends the VP8L stream of plane a to dst.
// EncodeLossless, alpha_enc.c:49.
func (e *LW) encodeLossless(dst []byte, a []uint8, w, h int) []byte {
	if cap(e.pic) < w*h {
		e.pic = make([]uint32, w*h)
	}
	e.pic = e.pic[:w*h]
	var used [256]bool
	for i, v := range a {
		e.pic[i] = uint32(v) << 8 // AlphaToGreen_C
		used[v] = true
	}
	e.paletteSize = alphaPalette(&used, &e.paletteSorted)
	// Raw wins past w*h; slack covers headers.
	e.bw = bitWriter{buf: slices.Grow(dst, w*h+w*h>>3+1024)}
	if e.Libwebp16 {
		e.v16EncodeStream(&e.bw, w, h)
	} else {
		e.encodeStream(&e.bw, w, h)
	}
	dst = e.bw.flush()
	e.bw.buf = nil // don't pin dst
	return dst
}

// GetNumColors, alpha_enc.c:179.
func getNumColors(a []uint8, w, h int) int {
	var color [256]uint8
	for _, v := range a[:w*h] {
		color[v] = 1
	}
	colors := 0
	for _, c := range color {
		if c > 0 {
			colors++
		}
	}
	return colors
}

// GetFilterMap, alpha_enc.c:202, WEBP_FILTER_FAST case only.
func getFilterMap(a []uint8, w, h, effort int) uint32 {
	const minColorsForFilterNone = 16
	const maxColorsForFilterNone = 192
	tryFilterNone := effort > 3
	numColors := getNumColors(a, w, h)
	filter := filterNone
	if numColors > minColorsForFilterNone {
		filter = estimateBestFilter(a, w, h)
	}
	bitMap := uint32(1) << filter
	if tryFilterNone || numColors > maxColorsForFilterNone {
		bitMap |= 1 << filterNone
	}
	return bitMap
}

// WebPEstimateBestFilter, filters_utils.c:29.
func estimateBestFilter(a []uint8, w, h int) int {
	const smax = 16
	sdiff := func(x, y int) int { return max(x-y, y-x) >> 4 }
	var bins [4][smax]int
	// Every other pixel is sampled.
	for j := 2; j < h-1; j += 2 {
		p := j * w
		mean := int(a[p])
		for i := 2; i < w-1; i += 2 {
			x := int(a[p+i])
			diff0 := sdiff(x, mean)
			diff1 := sdiff(x, int(a[p+i-1]))
			diff2 := sdiff(x, int(a[p+i-w]))
			gradPred := gradientPredictor(a[p+i-1], a[p+i-w], a[p+i-w-1])
			diff3 := sdiff(x, gradPred)
			bins[filterNone][diff0] = 1
			bins[filterHorizontal][diff1] = 1
			bins[filterVertical][diff2] = 1
			bins[filterGradient][diff3] = 1
			mean = (3*mean + x + 2) >> 2
		}
	}
	bestFilter := filterNone
	bestScore := 0x7fffffff
	for filter := filterNone; filter < 4; filter++ {
		score := 0
		for i := 0; i < smax; i++ {
			if bins[filter][i] > 0 {
				score += i
			}
		}
		if score < bestScore {
			bestScore = score
			bestFilter = filter
		}
	}
	return bestFilter
}

// GradientPredictor, filters_utils.c:24 and filters.c:121.
func gradientPredictor(a, b, c uint8) int {
	g := int(a) + int(b) - int(c)
	if g&^0xff == 0 {
		return g
	}
	if g < 0 {
		return 0
	}
	return 255
}

// PredictLine_C, filters.c:32, inverse=0.
func predictLine(src, pred, dst []uint8, n int) {
	for i := 0; i < n; i++ {
		dst[i] = src[i] - pred[i]
	}
}

// HorizontalFilter_C, filters.c:173 (DoHorizontalFilter_C :45).
func horizontalFilter(in []uint8, w, h int, out []uint8) {
	out[0] = in[0]
	predictLine(in[1:], in, out[1:], w-1)
	for y := 1; y < h; y++ {
		o := y * w
		predictLine(in[o:], in[o-w:], out[o:], 1)
		predictLine(in[o+1:], in[o:], out[o+1:], w-1)
	}
}

// VerticalFilter_C, filters.c:179 (DoVerticalFilter_C :82).
func verticalFilter(in []uint8, w, h int, out []uint8) {
	out[0] = in[0]
	predictLine(in[1:], in, out[1:], w-1)
	for y := 1; y < h; y++ {
		o := y * w
		predictLine(in[o:], in[o-w:], out[o:], w)
	}
}

// GradientFilter_C, filters.c:184 (DoGradientFilter_C :127).
func gradientFilter(in []uint8, w, h int, out []uint8) {
	out[0] = in[0]
	predictLine(in[1:], in, out[1:], w-1)
	for y := 1; y < h; y++ {
		o := y * w
		predictLine(in[o:], in[o-w:], out[o:], 1)
		for x := 1; x < w; x++ {
			pred := gradientPredictor(in[o+x-1], in[o+x-w], in[o+x-w-1])
			out[o+x] = uint8(int(in[o+x]) - pred)
		}
	}
}
