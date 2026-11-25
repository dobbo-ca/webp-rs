// Port of libwebp 1.2.2 enc/vp8l_enc.c (stream and Huffman storage) and utils/bit_writer_utils.h. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import (
	"math"
	"math/bits"
	"slices"
)

// numBytes is VP8LBitWriterNumBytes, bit_writer_utils.h:104.
func (b *bitWriter) numBytes() int {
	return len(b.buf) + (b.nacc+7)>>3
}

// encodeStream writes the VP8L stream of e.pic (no header) to bw.
// VP8LEncodeStream, vp8l_enc.c:1896, and EncodeStreamHook :1719.
// One crunch config and no side thread, so the hook's best-of swap
// is the identity.
func (e *LW) encodeStream(bw *bitWriter, w, h int) {
	cc, redAndBlueAlwaysZero := e.encoderAnalyze(w, h)
	if cc.entropyIdx == kSubGreen || cc.entropyIdx == kSpatialSubGreen {
		panic("vp8l: subtract green not ported")
	}
	e.hc.init(w * h) // EncoderInit :673

	e.usePalette = cc.entropyIdx == kPalette
	e.usePredict = cc.entropyIdx == kSpatial
	if !e.usePalette && !redAndBlueAlwaysZero && e.usePredict {
		panic("vp8l: cross color not ported")
	}
	e.refs[0] = e.refs[0][:0]
	e.refs[1] = e.refs[1][:0]

	// AllocateTransformBuffer :1432; w*h covers the packed palette image.
	e.argb = slices.Grow(e.argb[:0], w*h)[:w*h]
	if e.usePalette {
		paletteSortMinimizeDeltas(e.paletteSorted[:e.paletteSize], e.palette[:])
		e.encodePalette(bw)
		e.mapImageFromPalette(w, h)
	} else {
		// MakeInputImageCopy :1478
		copy(e.argb, e.pic[:w*h])
		e.currentWidth = w
		if e.usePredict {
			n := 2 * (w + 1)
			e.argbScratch = slices.Grow(e.argbScratch[:0], n)[:n]
			n = subSampleSize(w, e.transformBits) * subSampleSize(h, e.transformBits)
			e.transformData = slices.Grow(e.transformData[:0], n)[:n]
			e.applyPredictFilter(bw, e.currentWidth, h)
		}
	}

	bw.put(0, 1) // no more transforms
	e.encodeImageInternal(bw, e.argb, &e.hc, e.refs[:], e.currentWidth, h, &cc, e.histoBits)
}

// encodeImageInternal writes the main image, keeping the smaller of
// the sub-configs (strict <). EncodeImageInternal, vp8l_enc.c:1097.
// C swaps writers; here the best trial is copied to e.bwBest and
// copied back if a later trial loses. Bytes are the same.
func (e *LW) encodeImageInternal(bw *bitWriter, argb []uint32, hc *lwHashChain, refs []lwRefs, w, h int, cc *crunchConfig, histoBits int) {
	quality := e.cfg.quality
	hw, hh := subSampleSize(w, histoBits), subSampleSize(h, histoBits)
	xy := hw * hh
	e.histoSymbols = slices.Grow(e.histoSymbols[:0], xy)[:xy]
	e.hcHisto.init(xy)
	e.hashChainFill(hc, quality, argb, w, h)

	initLen, initAcc, initNacc := len(bw.buf), bw.acc, bw.nacc
	bestSize := math.MaxInt
	bestInBw := false
	for i := 0; i < cc.subSize; i++ {
		e.getBackwardReferences(w, h, argb, quality, cc.sub[i].lz77, hc, refs)

		// VP8LBitWriterReset
		bw.buf, bw.acc, bw.nacc = bw.buf[:initLen], initAcc, initNacc

		e.histoImage.alloc(xy)
		e.getHistoImageSymbols(w, h, refs[0], quality, histoBits, &e.histoImage, &e.tmpHisto, e.histoSymbols)
		size := len(e.histoImage.h)
		e.getHuffBitLengthsAndCodes(&e.histoImage, &e.huff)

		bw.put(0, 1) // no color cache

		write := size > 1
		if write {
			bw.put(1, 1)
			e.histoArgb = slices.Grow(e.histoArgb[:0], xy)[:xy]
			maxIndex := 0
			for k := 0; k < xy; k++ {
				s := int(e.histoSymbols[k])
				e.histoArgb[k] = uint32(s) << 8
				if s >= maxIndex {
					maxIndex = s + 1
				}
			}
			size = maxIndex
			bw.put(uint64(histoBits-2), 3)
			e.encodeImageNoHuffman(bw, e.histoArgb, &e.hcHisto, refs[2:], hw, hh, quality)
		} else {
			bw.put(0, 1)
		}

		codes := e.huff.codes[:5*size]
		skip := ^uint64(0) // groups with empty red, blue, alpha
		for k := range codes {
			storeHuffmanCode(bw, e.huffTree, e.tokens, &codes[k])
			if !clearHuffmanTreeIfOnlyOneSymbol(&codes[k]) && k%5 != 0 && k%5 != 4 {
				skip &^= 1 << (k / 5)
			}
		}
		storeImageToBitMask(bw, w, histoBits, refs[0], e.histoSymbols, codes, skip)

		bestInBw = false
		if n := bw.numBytes(); n < bestSize {
			bestSize = n
			bestInBw = true
			if i+1 < cc.subSize {
				e.bwBest.buf = append(e.bwBest.buf[:0], bw.buf[initLen:]...)
				e.bwBest.acc, e.bwBest.nacc = bw.acc, bw.nacc
			}
		}
	}
	if !bestInBw {
		bw.buf = append(bw.buf[:initLen], e.bwBest.buf...)
		bw.acc, bw.nacc = e.bwBest.acc, e.bwBest.nacc
	}
}

// encodeImageNoHuffman writes a sub-image with one Huffman group.
// refs is refs_array; it uses refs[0] and refs[1].
// EncodeImageNoHuffman, vp8l_enc.c:1013.
func (e *LW) encodeImageNoHuffman(bw *bitWriter, argb []uint32, hc *lwHashChain, refs []lwRefs, w, h, quality int) {
	var symbols [1]uint16 // one tree, one symbol

	e.hashChainFill(hc, quality, argb, w, h)
	e.getBackwardReferences(w, h, argb, quality, lz77Standard|lz77RLE, hc, refs)
	e.histoNH.alloc(1)
	e.histoNH.clear()
	e.histoNH.h[0].storeRefs(refs[0])
	e.getHuffBitLengthsAndCodes(&e.histoNH, &e.huffNH)

	bw.put(0, 1) // no color cache, no Huffman image

	codes := e.huffNH.codes[:5]
	for i := range codes {
		storeHuffmanCode(bw, e.huffTree, e.tokens, &codes[i])
		clearHuffmanTreeIfOnlyOneSymbol(&codes[i])
	}
	storeImageToBitMask(bw, w, 0, refs[0], symbols[:], codes, 0)
}

// applyPredictFilter runs the residual image and writes the predictor
// transform. ApplyPredictFilter, vp8l_enc.c:1308.
func (e *LW) applyPredictFilter(bw *bitWriter, w, h int) {
	predBits := e.transformBits
	tw := subSampleSize(w, predBits)
	th := subSampleSize(h, predBits)
	residualImage(w, h, predBits, e.argb, e.argbScratch, e.transformData)
	bw.put(1, 1) // TRANSFORM_PRESENT
	bw.put(0, 2) // PREDICTOR_TRANSFORM
	bw.put(uint64(predBits-2), 3)
	e.encodeImageNoHuffman(bw, e.transformData, &e.hc, e.refs[:], tw, th, e.cfg.quality)
}

// getHuffBitLengthsAndCodes sizes hc to 5 codes per histogram and
// builds them with e.bufRle and e.huffTree. It mutates the histograms,
// as C does. GetHuffBitLengthsAndCodes, vp8l_enc.c:690.
func (e *LW) getHuffBitLengthsAndCodes(set *lwHistoSet, hc *lwHuffCodes) {
	n := len(set.h)
	total := n * (lwNumCodes + 3*lwNumLiteral + lwNumDistance)
	hc.codes = slices.Grow(hc.codes[:0], 5*n)[:5*n]
	hc.lens = slices.Grow(hc.lens[:0], total)[:total]
	hc.bits = slices.Grow(hc.bits[:0], total)[:total]
	clear(hc.lens) // calloc
	clear(hc.bits)
	off := 0
	for i := range hc.codes {
		ns := lwNumLiteral
		switch i % 5 {
		case 0:
			ns = lwNumCodes
		case 4:
			ns = lwNumDistance
		}
		hc.codes[i] = huffTreeCode{
			numSymbols:  ns,
			codeLengths: hc.lens[off : off+ns : off+ns],
			codes:       hc.bits[off : off+ns : off+ns],
		}
		off += ns
	}

	// max_num_symbols is lwNumCodes; tokens sized for the store calls.
	e.bufRle = slices.Grow(e.bufRle[:0], lwNumCodes)[:lwNumCodes]
	e.huffTree = slices.Grow(e.huffTree[:0], 3*lwNumCodes)[:3*lwNumCodes]
	e.tokens = slices.Grow(e.tokens[:0], lwNumCodes)

	for i, histo := range set.h {
		c := hc.codes[5*i : 5*i+5]
		createHuffmanTree(histo.literal[:], 15, e.bufRle, e.huffTree, &c[0])
		createHuffmanTree(histo.red[:], 15, e.bufRle, e.huffTree, &c[1])
		createHuffmanTree(histo.blue[:], 15, e.bufRle, e.huffTree, &c[2])
		createHuffmanTree(histo.alpha[:], 15, e.bufRle, e.huffTree, &c[3])
		createHuffmanTree(histo.distance[:], 15, e.bufRle, e.huffTree, &c[4])
	}
}

// StoreHuffmanTreeOfHuffmanTreeToBitMask, vp8l_enc.c:764 (kStorageOrder is clOrder).
func storeHuffmanTreeOfHuffmanTreeToBitMask(bw *bitWriter, depths *[lwCodeLenCodes]uint8) {
	n := lwCodeLenCodes
	for ; n > 4; n-- {
		if depths[clOrder[n-1]] != 0 {
			break
		}
	}
	bw.put(uint64(n-4), 4)
	for i := 0; i < n; i++ {
		bw.put(uint64(depths[clOrder[i]]), 3)
	}
}

// clearHuffmanTreeIfOnlyOneSymbol reports whether it cleared c.
// ClearHuffmanTreeIfOnlyOneSymbol, vp8l_enc.c:786.
func clearHuffmanTreeIfOnlyOneSymbol(c *huffTreeCode) bool {
	count := 0
	for k := 0; k < c.numSymbols; k++ {
		if c.codeLengths[k] != 0 {
			count++
			if count > 1 {
				return false
			}
		}
	}
	for k := 0; k < c.numSymbols; k++ {
		c.codeLengths[k] = 0
		c.codes[k] = 0
	}
	return true
}

// StoreHuffmanTreeToBitMask, vp8l_enc.c:802.
func storeHuffmanTreeToBitMask(bw *bitWriter, tokens []huffTreeToken, c *huffTreeCode) {
	for _, t := range tokens {
		ix := t.code
		bw.put(uint64(c.codes[ix]), int(c.codeLengths[ix]))
		switch ix {
		case 16:
			bw.put(uint64(t.extraBits), 2)
		case 17:
			bw.put(uint64(t.extraBits), 3)
		case 18:
			bw.put(uint64(t.extraBits), 7)
		}
	}
}

// storeFullHuffmanCode: tree and tokens are scratch, cap(tokens) >= c.numSymbols.
// StoreFullHuffmanCode, vp8l_enc.c:826.
func storeFullHuffmanCode(bw *bitWriter, tree []huffTree, tokens []huffTreeToken, c *huffTreeCode) {
	var depths [lwCodeLenCodes]uint8
	var symbols [lwCodeLenCodes]uint16
	code := huffTreeCode{numSymbols: lwCodeLenCodes, codeLengths: depths[:], codes: symbols[:]}

	bw.put(0, 1)
	tokens = createCompressedHuffmanTree(c, tokens)
	numTokens := len(tokens)
	{
		var histogram [lwCodeLenCodes]uint32
		var bufRle [lwCodeLenCodes]uint8
		for _, t := range tokens {
			histogram[t.code]++
		}
		createHuffmanTree(histogram[:], 7, bufRle[:], tree, &code)
	}

	storeHuffmanTreeOfHuffmanTreeToBitMask(bw, &depths)
	clearHuffmanTreeIfOnlyOneSymbol(&code)

	trailingZeroBits := 0
	trimmedLength := numTokens
	for i := numTokens - 1; i >= 0; i-- {
		ix := tokens[i].code
		if ix == 0 || ix == 17 || ix == 18 {
			trimmedLength-- // discount trailing zeros
			trailingZeroBits += int(depths[ix])
			if ix == 17 { //nolint:staticcheck // as libwebp
				trailingZeroBits += 3
			} else if ix == 18 {
				trailingZeroBits += 7
			}
		} else {
			break
		}
	}
	writeTrimmed := trimmedLength > 1 && trailingZeroBits > 12
	length := numTokens
	if writeTrimmed {
		length = trimmedLength
		bw.put(1, 1)
		if trimmedLength == 2 {
			bw.put(0, 3+2) // nbitpairs=1, trimmed_length=2
		} else {
			nbits := bits.Len32(uint32(trimmedLength-2)) - 1 // BitsLog2Floor
			nbitpairs := nbits/2 + 1
			bw.put(uint64(nbitpairs-1), 3)
			bw.put(uint64(trimmedLength-2), nbitpairs*2)
		}
	} else {
		bw.put(0, 1)
	}
	storeHuffmanTreeToBitMask(bw, tokens[:length], &code)
}

// StoreHuffmanCode, vp8l_enc.c:894.
func storeHuffmanCode(bw *bitWriter, tree []huffTree, tokens []huffTreeToken, c *huffTreeCode) {
	const kMaxSymbol = 1 << 8
	count := 0
	var symbols [2]int
	for i := 0; i < c.numSymbols && count < 3; i++ {
		if c.codeLengths[i] != 0 {
			if count < 2 {
				symbols[count] = i
			}
			count++
		}
	}

	if count == 0 { // minimal tree for empty cases
		bw.put(0x01, 4)
	} else if count <= 2 && symbols[0] < kMaxSymbol && symbols[1] < kMaxSymbol {
		bw.put(1, 1) // small tree marker
		bw.put(uint64(count-1), 1)
		if symbols[0] <= 1 {
			bw.put(0, 1)
			bw.put(uint64(symbols[0]), 1)
		} else {
			bw.put(1, 1)
			bw.put(uint64(symbols[0]), 8)
		}
		if count == 2 {
			bw.put(uint64(symbols[1]), 8)
		}
	} else {
		storeFullHuffmanCode(bw, tree, tokens, c)
	}
}

// WriteHuffmanCode, vp8l_enc.c:933.
func writeHuffmanCode(bw *bitWriter, c *huffTreeCode, i int) {
	bw.put(uint64(c.codes[i]), int(c.codeLengths[i]))
}

// WriteHuffmanCodeWithExtraBits, vp8l_enc.c:941.
func writeHuffmanCodeWithExtraBits(bw *bitWriter, c *huffTreeCode, i, bits, n int) {
	depth := int(c.codeLengths[i])
	bw.put(uint64(bits)<<depth|uint64(c.codes[i]), depth+n)
}

// StoreImageToBitMask, vp8l_enc.c:952; codes holds 5 per histogram.
// Skip bit g: group g green-only.
func storeImageToBitMask(bw *bitWriter, w, histoBits int, refs lwRefs, symbols []uint16, codes []huffTreeCode, skip uint64) {
	histoXsize := 1
	tileMask := 0
	if histoBits != 0 {
		histoXsize = subSampleSize(w, histoBits)
		tileMask = -(1 << histoBits)
	}
	x, y := 0, 0
	tileX, tileY := x&tileMask, y&tileMask
	c := codes[5*int(symbols[0]):]
	rest := skip>>symbols[0]&1 == 0
	for i := range refs {
		v := &refs[i]
		if tileX != x&tileMask || tileY != y&tileMask {
			tileX, tileY = x&tileMask, y&tileMask
			s := symbols[(y>>histoBits)*histoXsize+(x>>histoBits)]
			c = codes[5*int(s):]
			rest = skip>>s&1 == 0
		}
		if v.mode == lwLiteral {
			writeHuffmanCode(bw, &c[0], int(v.argbOrDist>>8&0xff))
			if rest {
				writeHuffmanCode(bw, &c[1], int(v.argbOrDist>>16&0xff))
				writeHuffmanCode(bw, &c[2], int(v.argbOrDist&0xff))
				writeHuffmanCode(bw, &c[3], int(v.argbOrDist>>24))
			}
		} else { // kCacheIdx is unreachable with use_cache=0
			code, n, extra := prefixBits(int(v.len))
			writeHuffmanCodeWithExtraBits(bw, &c[0], 256+code, extra, n)
			// Distance extra bits go separately, up to 18 bits.
			code, n, extra = prefixBits(int(v.argbOrDist))
			writeHuffmanCode(bw, &c[4], code)
			bw.put(uint64(extra), n)
		}
		x += int(v.len)
		for x >= w {
			x -= w
			y++
		}
	}
}
