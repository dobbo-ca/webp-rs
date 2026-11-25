// Port of libwebp 1.6.0 enc/vp8l_enc.c (stream and Huffman storage). The Huffman tree code, bit writer and Huffman storage helpers are the 1.2.2 port, equal under diff -u. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import (
	"math"
	"math/bits"
	"slices"
	"unsafe"
)

// lw16State is the scratch of the 1.6.0 path, made on first use.
type lw16State struct {
	// histogram_enc.c
	orig        lw16HistoSet
	image       lw16HistoSet
	tmp         lw16Histo
	queue       lw16Queue
	histoSymbol []uint32 // histogram_argb

	// backward_references_enc.c
	est         lw16Histo
	cacheHistos [v16MaxCacheBits + 1]lw16Histo
	caches      [v16MaxCacheBits + 1]v16ColorCache
	cc          v16ColorCache

	// backward_references_cost_enc.c
	costModel lw16CostModel
	costHisto lw16Histo
	costMgr   lw16CostManager

	// predictor_enc.c
	predRaw  []uint32
	modesRaw []uint32

	huff lwHuffCodes
}

// held is the scratch s keeps, in bytes.
func (s *lw16State) held() int {
	n := int(unsafe.Sizeof(*s))
	for _, set := range []*lw16HistoSet{&s.orig, &s.image} {
		n += 8*cap(set.h) + int(unsafe.Sizeof(lw16Histo{}))*cap(set.pool) + 4*cap(set.lits)
	}
	n += 4 * (cap(s.tmp.literal) + cap(s.est.literal) + cap(s.costHisto.literal) + cap(s.costModel.literal))
	for i := range s.cacheHistos {
		n += 4 * (cap(s.cacheHistos[i].literal) + cap(s.caches[i].colors))
	}
	n += 4 * (cap(s.cc.colors) + cap(s.histoSymbol) + cap(s.predRaw) + cap(s.modesRaw))
	n += 8 * cap(s.costMgr.costs)
	n += int(unsafe.Sizeof(lw16CostCacheInterval{})) * cap(s.costMgr.cacheIntervals)
	n += int(unsafe.Sizeof(lw16Pair{})) * cap(s.queue.q)
	n += int(unsafe.Sizeof(huffTreeCode{}))*cap(s.huff.codes) + cap(s.huff.lens) + 2*cap(s.huff.bits)
	return n
}

// v16EncodeStream writes the VP8L stream of e.pic (no header) to bw.
// VP8LEncodeStream, vp8l_enc.c:1690, and EncodeStreamHook :1512.
// One crunch config and no side thread.
func (e *LW) v16EncodeStream(bw *bitWriter, w, h int) {
	if e.v16 == nil {
		e.v16 = new(lw16State)
	}
	cc, redAndBlueAlwaysZero := e.v16EncoderAnalyze(w, h)
	if cc.entropyIdx == kSubGreen || cc.entropyIdx == kSpatialSubGreen {
		panic("vp8l: subtract green not ported")
	}
	e.hc.init(w * h) // EncoderInit :422

	e.usePalette = cc.entropyIdx == kPalette
	e.usePredict = cc.entropyIdx == kSpatial
	if !e.usePalette && !redAndBlueAlwaysZero && e.usePredict {
		panic("vp8l: cross color not ported")
	}
	cacheBits := 0
	e.refs[0] = e.refs[0][:0]
	e.refs[1] = e.refs[1][:0]

	// AllocateTransformBuffer :1226; w*h covers the packed palette image.
	e.argb = slices.Grow(e.argb[:0], w*h)[:w*h]
	if e.usePalette {
		v16PaletteSortMinimizeDeltas(e.paletteSorted[:e.paletteSize], e.palette[:])
		e.v16EncodePalette(bw)
		e.mapImageFromPalette(w, h)
		// Cache no bigger than the palette.
		if e.paletteSize < 1<<v16MaxCacheBits {
			cacheBits = bits.Len32(uint32(e.paletteSize)) // BitsLog2Floor + 1
		}
	} else {
		// MakeInputImageCopy :1267
		copy(e.argb, e.pic[:w*h])
		e.currentWidth = w
		if e.usePredict {
			n := 2 * (w + 1)
			e.argbScratch = slices.Grow(e.argbScratch[:0], n)[:n]
			n = subSampleSize(w, lwMinHuffBits) * subSampleSize(h, lwMinHuffBits) // MIN_TRANSFORM_BITS
			e.transformData = slices.Grow(e.transformData[:0], n)[:n]
			e.v16ApplyPredictFilter(bw, e.currentWidth, h)
		}
	}

	bw.put(0, 1) // no more transforms
	e.v16EncodeImageInternal(bw, e.argb, &e.hc, e.refs[:], e.currentWidth, h, &cc, cacheBits, e.histoBits)
}

// v16EncodeImageInternal writes the main image, keeping the smaller
// of the sub-configs (strict <). EncodeImageInternal, vp8l_enc.c:862.
// C swaps writers; here the best trial is copied to e.bwBest and
// copied back if a later trial loses. Bytes are the same.
func (e *LW) v16EncodeImageInternal(bw *bitWriter, argb []uint32, hc *lwHashChain, refs []lwRefs, w, h int, cc *crunchConfig, cacheBits, histoBits int) {
	s := e.v16
	quality := e.cfg.quality
	xy := subSampleSize(w, histoBits) * subSampleSize(h, histoBits)
	s.histoSymbol = slices.Grow(s.histoSymbol[:0], xy)[:xy]
	e.hcHisto.init(xy)
	e.hashChainFill(hc, quality, argb, w, h)

	// Nonzero cacheBits came from the palette.
	cacheBitsInit := cacheBits
	if cacheBitsInit == 0 {
		cacheBitsInit = v16MaxCacheBits
	}
	initLen, initAcc, initNacc := len(bw.buf), bw.acc, bw.nacc
	bestSize := math.MaxInt
	bestInBw := false
	for i := 0; i < cc.subSize; i++ {
		cacheBitsTmp := e.v16GetBackwardReferences(w, h, argb, quality, cc.sub[i].lz77, cacheBitsInit, hc, refs)
		histogramBits := histoBits

		// VP8LBitWriterReset
		bw.buf, bw.acc, bw.nacc = bw.buf[:initLen], initAcc, initNacc

		s.image.alloc(xy, cacheBitsTmp)
		s.tmp.alloc(cacheBitsTmp)
		s.getHistoImageSymbols(w, h, refs[0], quality, histogramBits, cacheBitsTmp, &s.image, &s.tmp, s.histoSymbol)
		e.v16GetHuffBitLengthsAndCodes(&s.image, &s.huff)

		if cacheBitsTmp > 0 {
			bw.put(1, 1)
			bw.put(uint64(cacheBitsTmp), 4)
		} else {
			bw.put(0, 1)
		}

		size := 0
		for k := range s.histoSymbol {
			if int(s.histoSymbol[k]) >= size {
				size = int(s.histoSymbol[k]) + 1
			}
			s.histoSymbol[k] <<= 8
		}
		if size > 1 {
			bw.put(1, 1)
			histogramBits = v16OptimizeSampling(s.histoSymbol, w, h, histoBits, lwMaxHuffBits)
			bw.put(uint64(histogramBits-2), 3)
			e.v16EncodeImageNoHuffman(bw, s.histoSymbol, &e.hcHisto, refs[2:], subSampleSize(w, histogramBits), subSampleSize(h, histogramBits), quality)
		} else {
			bw.put(0, 1)
		}

		codes := s.huff.codes[:5*size]
		skip := ^uint64(0) // groups with empty red, blue, alpha
		for k := range codes {
			storeHuffmanCode(bw, e.huffTree, e.tokens, &codes[k])
			if !clearHuffmanTreeIfOnlyOneSymbol(&codes[k]) && k%5 != 0 && k%5 != 4 {
				skip &^= 1 << (k / 5)
			}
		}
		v16StoreImageToBitMask(bw, w, histogramBits, refs[0], s.histoSymbol, codes, skip)

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

// v16EncodeImageNoHuffman writes a sub-image with one Huffman group.
// Its refs never use the cache, so the 1.2.2 histogram and Huffman
// storage apply. refs is refs_array; it uses refs[0] and refs[1].
// EncodeImageNoHuffman, vp8l_enc.c:769.
func (e *LW) v16EncodeImageNoHuffman(bw *bitWriter, argb []uint32, hc *lwHashChain, refs []lwRefs, w, h, quality int) {
	var symbols [1]uint16 // one tree, one symbol

	e.hashChainFill(hc, quality, argb, w, h)
	e.v16GetBackwardReferences(w, h, argb, quality, lz77Standard|lz77RLE, 0, hc, refs)
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

// v16ApplyPredictFilter runs the residual image and writes the
// predictor transform. ApplyPredictFilter, vp8l_enc.c:1100.
func (e *LW) v16ApplyPredictFilter(bw *bitWriter, w, h int) {
	const minBits = lwMinHuffBits // MIN_TRANSFORM_BITS
	maxBits := v16ClampBits(w, h, e.transformBits, minBits, v16MaxTransBits, v16MaxPredImage)
	extra := 0
	if e.cfg.method > 4 {
		extra = e.cfg.method - 4
	}
	lowBits := v16ClampBits(w, h, maxBits-2*extra, minBits, v16MaxTransBits, v16MaxPredImage)
	bestBits := e.v16.residualImage(w, h, lowBits, maxBits, e.argb, e.argbScratch, e.transformData)
	bw.put(1, 1) // TRANSFORM_PRESENT
	bw.put(0, 2) // PREDICTOR_TRANSFORM
	bw.put(uint64(bestBits-minBits), 3)
	e.v16EncodeImageNoHuffman(bw, e.transformData, &e.hc, e.refs[:], subSampleSize(w, bestBits), subSampleSize(h, bestBits), e.cfg.quality)
}

// v16GetHuffBitLengthsAndCodes sizes hc to 5 codes per histogram of
// set.h[:set.size] and builds them. It mutates the histograms, as C
// does. GetHuffBitLengthsAndCodes, vp8l_enc.c:439.
func (e *LW) v16GetHuffBitLengthsAndCodes(set *lw16HistoSet, hc *lwHuffCodes) {
	hs := set.h[:set.size]
	n := len(hs)
	total := 0
	for _, h := range hs {
		total += len(h.literal) + 3*lwNumLiteral + lwNumDistance
	}
	hc.codes = slices.Grow(hc.codes[:0], 5*n)[:5*n]
	hc.lens = slices.Grow(hc.lens[:0], total)[:total]
	hc.bits = slices.Grow(hc.bits[:0], total)[:total]
	clear(hc.lens) // calloc
	clear(hc.bits)
	off := 0
	maxSymbols := 0
	for i := range hc.codes {
		ns := lwNumLiteral
		switch i % 5 {
		case 0:
			ns = len(hs[i/5].literal)
		case 4:
			ns = lwNumDistance
		}
		hc.codes[i] = huffTreeCode{
			numSymbols:  ns,
			codeLengths: hc.lens[off : off+ns : off+ns],
			codes:       hc.bits[off : off+ns : off+ns],
		}
		off += ns
		maxSymbols = max(maxSymbols, ns)
	}

	// Tokens sized for the store calls.
	e.bufRle = slices.Grow(e.bufRle[:0], maxSymbols)[:maxSymbols]
	e.huffTree = slices.Grow(e.huffTree[:0], 3*maxSymbols)[:3*maxSymbols]
	e.tokens = slices.Grow(e.tokens[:0], maxSymbols)

	for i, histo := range hs {
		c := hc.codes[5*i : 5*i+5]
		createHuffmanTree(histo.literal, 15, e.bufRle, e.huffTree, &c[0])
		createHuffmanTree(histo.red[:], 15, e.bufRle, e.huffTree, &c[1])
		createHuffmanTree(histo.blue[:], 15, e.bufRle, e.huffTree, &c[2])
		createHuffmanTree(histo.alpha[:], 15, e.bufRle, e.huffTree, &c[3])
		createHuffmanTree(histo.distance[:], 15, e.bufRle, e.huffTree, &c[4])
	}
}

// v16StoreImageToBitMask: symbols hold ids << 8; codes 5 per id.
// StoreImageToBitMask, vp8l_enc.c:701. skip as storeImageToBitMask.
func v16StoreImageToBitMask(bw *bitWriter, w, histoBits int, refs lwRefs, symbols []uint32, codes []huffTreeCode, skip uint64) {
	histoXsize := 1
	tileMask := 0
	if histoBits != 0 {
		histoXsize = subSampleSize(w, histoBits)
		tileMask = -(1 << histoBits)
	}
	x, y := 0, 0
	tileX, tileY := x&tileMask, y&tileMask
	g := (symbols[0] >> 8) & 0xffff
	c := codes[5*int(g):]
	rest := skip>>g&1 == 0
	for i := range refs {
		v := &refs[i]
		if tileX != x&tileMask || tileY != y&tileMask {
			tileX, tileY = x&tileMask, y&tileMask
			g = (symbols[(y>>histoBits)*histoXsize+(x>>histoBits)] >> 8) & 0xffff
			c = codes[5*int(g):]
			rest = skip>>g&1 == 0
		}
		switch v.mode {
		case lwLiteral:
			writeHuffmanCode(bw, &c[0], int(v.argbOrDist>>8&0xff))
			if rest {
				writeHuffmanCode(bw, &c[1], int(v.argbOrDist>>16&0xff))
				writeHuffmanCode(bw, &c[2], int(v.argbOrDist&0xff))
				writeHuffmanCode(bw, &c[3], int(v.argbOrDist>>24))
			}
		case lw16CacheIdx:
			writeHuffmanCode(bw, &c[0], lwNumCodes+int(v.argbOrDist))
		default:
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
