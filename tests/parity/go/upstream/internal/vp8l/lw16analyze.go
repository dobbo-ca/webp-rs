// Port of libwebp 1.6.0 enc/vp8l_enc.c (analysis and palette) and utils/palette.c. GetColorPalette, GetTransformBits and palette distance helpers are the 1.2.2 port. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

// v16EncoderAnalyze sets paletteSize, paletteSorted, histoBits and
// transformBits from e.pic, and returns the one crunch config.
// It panics on the method 0, m6/q100 and m5/q>=75 branches.
// EncoderAnalyze, vp8l_enc.c:307.
func (e *LW) v16EncoderAnalyze(w, h int) (cc crunchConfig, redAndBlueAlwaysZero bool) {
	method := e.cfg.method
	if method == 0 {
		panic("vp8l: low_effort not ported")
	}

	// GetColorPalette, palette.c:97: encodeLossless set it, sorted.
	usePalette := true

	e.histoBits = v16GetHistoBits(method, usePalette, w, h)
	e.transformBits = getTransformBits(method, e.histoBits)

	nLz77s := 1
	if e.paletteSize > 0 && e.paletteSize <= 16 {
		nLz77s = 2
	}
	minIx, rb := e.v16AnalyzeEntropy(e.pic, w, h, w, usePalette, e.paletteSize, e.transformBits)
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

// ClampBits, vp8l_enc.c:254.
func v16ClampBits(w, h, bits, minBits, maxBits, imageSizeMax int) int {
	bits = max(minBits, min(bits, maxBits))
	imageSize := subSampleSize(w, bits) * subSampleSize(h, bits)
	for bits < maxBits && imageSize > imageSizeMax {
		bits++
		imageSize = subSampleSize(w, bits) * subSampleSize(h, bits)
	}
	// Keep the smallest bits that still give one tile.
	for bits > minBits && imageSize == 1 {
		imageSize = subSampleSize(w, bits-1) * subSampleSize(h, bits-1)
		if imageSize != 1 {
			break
		}
		bits--
	}
	return bits
}

// GetHistoBits, vp8l_enc.c:275.
func v16GetHistoBits(method int, usePalette bool, w, h int) int {
	histoBits := 7 - method
	if usePalette {
		histoBits = 9 - method
	}
	return v16ClampBits(w, h, histoBits, lwMinHuffBits, lwMaxHuffBits, lwMaxHuffImage)
}

// v16AnalyzeEntropy uses e.aeHisto. AnalyzeEntropy, vp8l_enc.c:106.
func (e *LW) v16AnalyzeEntropy(argb []uint32, w, h, stride int, usePalette bool, paletteSize, transformBits int) (minIx int, redAndBlueAlwaysZero bool) {
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

	var entropyComp [lwHistoTotal]uint64
	var entropy [5]uint64
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
		entropyComp[j] = v16BitsEntropy(seg(j))
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

	tiles := uint64(subSampleSize(w, transformBits)) * uint64(subSampleSize(h, transformBits))
	entropy[kSpatial] += tiles * uint64(v16FastLog2(14))
	entropy[kSpatialSubGreen] += tiles * uint64(v16FastLog2(24))
	entropy[kPalette] += uint64(paletteSize*8) << v16LogBits

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

// PaletteSortMinimizeDeltas, palette.c:187; len(palette) >= len(sorted).
func v16PaletteSortMinimizeDeltas(sorted, palette []uint32) {
	n := len(sorted)
	copy(palette, sorted)
	if !paletteHasNonMonotonousDeltas(sorted) {
		return
	}
	// A leading 0 moves last, so EncodePalette can drop it.
	if n > 17 && palette[0] == 0 {
		n--
		palette[n], palette[0] = palette[0], palette[n]
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

// v16EncodePalette writes the color indexing transform.
// EncodePalette, vp8l_enc.c:1442.
func (e *LW) v16EncodePalette(bw *bitWriter) {
	var tmp [lwMaxPalette]uint32
	n := e.paletteSize
	// A trailing 0 is implied when there is no pixel packing.
	if e.palette[n-1] == 0 && n > 17 {
		n--
	}
	bw.put(1, 1) // TRANSFORM_PRESENT
	bw.put(3, 2) // COLOR_INDEXING_TRANSFORM
	bw.put(uint64(n-1), 8)
	for i := n - 1; i >= 1; i-- {
		tmp[i] = subPixels(e.palette[i], e.palette[i-1])
	}
	tmp[0] = e.palette[0]
	e.v16EncodeImageNoHuffman(bw, tmp[:n], &e.hc, e.refs[:], n, 1, 20)
}
