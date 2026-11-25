// Port of libwebp 1.2.2 enc/vp8l_enc.c (analysis and palette) and utils/utils.c WebPGetColorPalette. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

// HistoIx, vp8l_enc.c:343.
const (
	kHistoAlpha = iota
	kHistoAlphaPred
	kHistoGreen
	kHistoGreenPred
	kHistoRed
	kHistoRedPred
	kHistoBlue
	kHistoBluePred
	kHistoRedSubGreen
	kHistoRedPredSubGreen
	kHistoBlueSubGreen
	kHistoBluePredSubGreen
	kHistoPalette
)

// kHistoPairs, vp8l_enc.c:502.
var kHistoPairs = [5][2]int{
	{kHistoRed, kHistoBlue},
	{kHistoRedPred, kHistoBluePred},
	{kHistoRedSubGreen, kHistoBlueSubGreen},
	{kHistoRedPredSubGreen, kHistoBluePredSubGreen},
	{kHistoRed, kHistoBlue},
}

// encoderAnalyze sets paletteSize, paletteSorted, histoBits and
// transformBits from e.pic, and returns the one crunch config.
// It panics on the method 0, m6/q100 and m5/q>=75 branches.
// EncoderAnalyze, vp8l_enc.c:566.
func (e *LW) encoderAnalyze(w, h int) (cc crunchConfig, redAndBlueAlwaysZero bool) {
	method := e.cfg.method
	if method == 0 {
		panic("vp8l: low_effort not ported")
	}

	// encodeLossless set the sorted palette.
	usePalette := true

	e.histoBits = getHistoBits(method, usePalette, w, h)
	e.transformBits = getTransformBits(method, e.histoBits)

	nLz77s := 1
	if e.paletteSize > 0 && e.paletteSize <= 16 {
		nLz77s = 2
	}
	minIx, rb := e.analyzeEntropy(e.pic, w, h, w, usePalette, e.paletteSize, e.transformBits)
	if method == 6 && e.cfg.quality == 100 {
		panic("vp8l: m6/q100 brute force not ported")
	}
	cc.entropyIdx = minIx
	if e.cfg.quality >= 75 && method == 5 {
		panic("vp8l: do_no_cache not ported")
	}

	for j := 0; j < nLz77s; j++ {
		if j == 0 {
			cc.sub[j].lz77 = lz77Standard | lz77RLE
		} else {
			cc.sub[j].lz77 = lz77Box
		}
	}
	cc.subSize = nLz77s
	return cc, rb
}

// analyzeEntropy uses e.aeHisto. AnalyzeEntropy, vp8l_enc.c:381.
func (e *LW) analyzeEntropy(argb []uint32, w, h, stride int, usePalette bool, paletteSize, transformBits int) (minIx int, redAndBlueAlwaysZero bool) {
	if usePalette && paletteSize <= 16 {
		return kPalette, true
	}
	histo := e.aeHisto[:]
	clear(histo)
	seg := func(i int) []uint32 { return histo[i*256 : (i+1)*256] }
	hA, hAP := seg(kHistoAlpha), seg(kHistoAlphaPred)
	hG, hGP := seg(kHistoGreen), seg(kHistoGreenPred)
	hR, hRP := seg(kHistoRed), seg(kHistoRedPred)
	hB, hBP := seg(kHistoBlue), seg(kHistoBluePred)
	hRS, hRPS := seg(kHistoRedSubGreen), seg(kHistoRedPredSubGreen)
	hBS, hBPS := seg(kHistoBlueSubGreen), seg(kHistoBluePredSubGreen)
	hPal := seg(kHistoPalette)

	var prevRow []uint32
	off := 0
	pixPrev := argb[0] // skip the first pixel
	for y := 0; y < h; y++ {
		cur := argb[off : off+w]
		for x := 0; x < w; x++ {
			pix := cur[x]
			pixDiff := subPixels(pix, pixPrev)
			pixPrev = pix
			if pixDiff == 0 || (prevRow != nil && pix == prevRow[x]) {
				continue
			}
			addSingle(pix, hA, hR, hG, hB)
			addSingle(pixDiff, hAP, hRP, hGP, hBP)
			addSingleSubGreen(pix, hRS, hBS)
			addSingleSubGreen(pixDiff, hRPS, hBPS)
			hPal[hashPix(pix)]++
		}
		prevRow = cur
		off += stride
	}

	var entropyComp [lwHistoTotal]float64
	var entropy [5]float64
	lastMode := kSpatialSubGreen
	if usePalette {
		lastMode = kPalette
	}
	// One zero for each predicted histogram.
	hRPS[0]++
	hBPS[0]++
	hRP[0]++
	hGP[0]++
	hBP[0]++
	hAP[0]++

	for j := 0; j < lwHistoTotal; j++ {
		entropyComp[j] = bitsEntropy(seg(j))
	}
	entropy[kDirect] = entropyComp[kHistoAlpha] +
		entropyComp[kHistoRed] +
		entropyComp[kHistoGreen] +
		entropyComp[kHistoBlue]
	entropy[kSpatial] = entropyComp[kHistoAlphaPred] +
		entropyComp[kHistoRedPred] +
		entropyComp[kHistoGreenPred] +
		entropyComp[kHistoBluePred]
	entropy[kSubGreen] = entropyComp[kHistoAlpha] +
		entropyComp[kHistoRedSubGreen] +
		entropyComp[kHistoGreen] +
		entropyComp[kHistoBlueSubGreen]
	entropy[kSpatialSubGreen] = entropyComp[kHistoAlphaPred] +
		entropyComp[kHistoRedPredSubGreen] +
		entropyComp[kHistoGreenPred] +
		entropyComp[kHistoBluePredSubGreen]
	entropy[kPalette] = entropyComp[kHistoPalette]

	// uint32 tile count times float, as in C.
	tiles := float32(uint32(subSampleSize(w, transformBits)) * uint32(subSampleSize(h, transformBits)))
	entropy[kSpatial] += float64(float32(tiles * fastLog2(14)))
	entropy[kSpatialSubGreen] += float64(float32(tiles * fastLog2(24)))
	entropy[kPalette] += float64(paletteSize * 8)

	minIx = kDirect
	for k := kDirect + 1; k <= lastMode; k++ {
		if entropy[minIx] > entropy[k] {
			minIx = k
		}
	}
	redAndBlueAlwaysZero = true
	red := seg(kHistoPairs[minIx][0])
	blue := seg(kHistoPairs[minIx][1])
	for i := 1; i < 256; i++ {
		if red[i]|blue[i] != 0 {
			redAndBlueAlwaysZero = false
			break
		}
	}
	return minIx, redAndBlueAlwaysZero
}

// AddSingleSubGreen, vp8l_enc.c:360. Only the low 8 bits matter, so
// C's signed shifts give the same index.
func addSingleSubGreen(p uint32, r, b []uint32) {
	green := p >> 8
	r[((p>>16)-green)&0xff]++
	b[((p>>0)-green)&0xff]++
}

// AddSingle, vp8l_enc.c:366.
func addSingle(p uint32, a, r, g, b []uint32) {
	a[(p>>24)&0xff]++
	r[(p>>16)&0xff]++
	g[(p>>8)&0xff]++
	b[(p>>0)&0xff]++
}

// HashPix, vp8l_enc.c:375.
func hashPix(p uint32) uint32 {
	return uint32(((uint64(p)+uint64(p>>19))*0x39c5fba7)&0xffffffff) >> 24
}

// GetHistoBits, vp8l_enc.c:528.
func getHistoBits(method int, usePalette bool, w, h int) int {
	histoBits := 7 - method
	if usePalette {
		histoBits = 9 - method
	}
	for {
		huffImageSize := subSampleSize(w, histoBits) * subSampleSize(h, histoBits)
		if huffImageSize <= lwMaxHuffImage {
			break
		}
		histoBits++
	}
	if histoBits < lwMinHuffBits {
		return lwMinHuffBits
	}
	if histoBits > lwMaxHuffBits {
		return lwMaxHuffBits
	}
	return histoBits
}

// GetTransformBits, vp8l_enc.c:541.
func getTransformBits(method, histoBits int) int {
	maxBits := 5
	if method < 4 {
		maxBits = 6
	} else if method > 4 {
		maxBits = 4
	}
	if histoBits > maxBits {
		return maxBits
	}
	return histoBits
}

// alphaPalette: sorted WebPGetColorPalette, utils.c:258.
func alphaPalette(used *[256]bool, palette *[lwMaxPalette]uint32) int {
	n := 0
	for v, u := range used {
		if u {
			palette[n] = uint32(v) << 8
			n++
		}
	}
	return n
}

// PaletteComponentDistance, vp8l_enc.c:41.
func paletteComponentDistance(v uint32) uint32 {
	if v <= 128 {
		return v
	}
	return 256 - v
}

// PaletteColorDistance, vp8l_enc.c:50.
func paletteColorDistance(c1, c2 uint32) uint32 {
	diff := subPixels(c1, c2)
	const moreWeightForRGB = 9
	score := paletteComponentDistance((diff >> 0) & 0xff)
	score += paletteComponentDistance((diff >> 8) & 0xff)
	score += paletteComponentDistance((diff >> 16) & 0xff)
	score *= moreWeightForRGB
	score += paletteComponentDistance((diff >> 24) & 0xff)
	return score
}

// PaletteHasNonMonotonousDeltas, vp8l_enc.c:92.
func paletteHasNonMonotonousDeltas(p []uint32) bool {
	var predict uint32
	var signFound uint8
	for i := range p {
		diff := subPixels(p[i], predict)
		rd := uint8(diff >> 16)
		gd := uint8(diff >> 8)
		bd := uint8(diff >> 0)
		if rd != 0 {
			if rd < 0x80 {
				signFound |= 1
			} else {
				signFound |= 2
			}
		}
		if gd != 0 {
			if gd < 0x80 {
				signFound |= 8
			} else {
				signFound |= 16
			}
		}
		if bd != 0 {
			if bd < 0x80 {
				signFound |= 64
			} else {
				signFound |= 128
			}
		}
		predict = p[i]
	}
	// Bit 8 of the C int shift is masked by the & anyway.
	return signFound&(signFound<<1) != 0
}

// PaletteSortMinimizeDeltas, vp8l_enc.c:116; len(palette) >= len(sorted).
func paletteSortMinimizeDeltas(sorted, palette []uint32) {
	n := len(sorted)
	copy(palette, sorted)
	if !paletteHasNonMonotonousDeltas(sorted) {
		return
	}
	var predict uint32
	for i := 0; i < n; i++ {
		bestIx := i
		bestScore := ^uint32(0)
		for k := i; k < n; k++ {
			cur := paletteColorDistance(palette[k], predict)
			if bestScore > cur {
				bestScore = cur
				bestIx = k
			}
		}
		palette[bestIx], palette[i] = palette[i], palette[bestIx]
		predict = palette[i]
	}
}

// encodePalette writes the color indexing transform.
// EncodePalette, vp8l_enc.c:1655.
func (e *LW) encodePalette(bw *bitWriter) {
	var tmp [lwMaxPalette]uint32
	n := e.paletteSize
	bw.put(1, 1) // TRANSFORM_PRESENT
	bw.put(3, 2) // COLOR_INDEXING_TRANSFORM
	bw.put(uint64(n-1), 8)
	for i := n - 1; i >= 1; i-- {
		tmp[i] = subPixels(e.palette[i], e.palette[i-1])
	}
	tmp[0] = e.palette[0]
	e.encodeImageNoHuffman(bw, tmp[:n], &e.hc, e.refs[:], n, 1, 20)
}

// mapImageFromPalette packs e.pic into e.argb and sets currentWidth.
// MapImageFromPalette, vp8l_enc.c:1624 (in_place=0).
func (e *LW) mapImageFromPalette(w, h int) {
	n := e.paletteSize
	var xbits int
	if n <= 4 {
		xbits = 2
		if n <= 2 {
			xbits = 3
		}
	} else if n <= 16 {
		xbits = 1
	}
	// AllocateTransformBuffer, argb part only (no predict).
	e.currentWidth = subSampleSize(w, xbits)
	if cap(e.argb) < e.currentWidth*h {
		e.argb = make([]uint32, e.currentWidth*h)
	}
	e.argb = e.argb[:e.currentWidth*h]
	if cap(e.palRow) < w {
		e.palRow = make([]uint8, w)
	}
	e.palRow = e.palRow[:w]
	applyPalette(e.pic, w, e.argb, e.currentWidth, e.palette[:n], w, h, xbits, e.palRow)
}

// applyPalette maps src to packed palette indices; row is tmp_row.
// ApplyPalette, vp8l_enc.c:1562.
// ponytail: green-byte LUT in place of C's greedy/hash search. Every
// lookup returns the color's palette index, so the bytes match. It
// needs distinct greens, true for ALPH (a<<8); port hash1/hash2 if
// other inputs ever come here.
func applyPalette(src []uint32, srcStride int, dst []uint32, dstStride int, palette []uint32, w, h, xbits int, row []uint8) {
	var lut [256]uint8
	var seen [256]bool
	for j, c := range palette {
		g := uint8(c >> 8)
		if seen[g] {
			panic("vp8l: palette greens not distinct")
		}
		seen[g] = true
		lut[g] = uint8(j)
	}
	for y := 0; y < h; y++ {
		s := src[y*srcStride : y*srcStride+w]
		for x, pix := range s {
			row[x] = lut[uint8(pix>>8)]
		}
		bundleColorMap(row, w, xbits, dst[y*dstStride:])
	}
}
