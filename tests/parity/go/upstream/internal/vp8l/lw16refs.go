// Port of libwebp 1.6.0 enc/backward_references_enc.c, color cache selection. LZ77, RLE, Box, hash chain and 2D locality are the 1.2.2 port, equal under diff -u. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "math"

const v16MaxCacheBits = 10 // MAX_COLOR_CACHE_BITS

// calculateBestCacheSize returns the cache bits in [0, bitsMax] with
// the lowest estimate. Uses s.cacheHistos and s.caches.
// CalculateBestCacheSize, backward_references_enc.c:755.
func (s *lw16State) calculateBestCacheSize(argb []uint32, quality int, refs lwRefs, bitsMax int) int {
	if quality <= 25 {
		bitsMax = 0
	}
	if bitsMax == 0 {
		return 0
	}
	histos := s.cacheHistos[:bitsMax+1]
	for i := range histos {
		histos[i].alloc(i)
		histos[i].init(i, true)
		if i > 0 {
			s.caches[i].init(i)
		}
	}

	pos := 0
	for k := range refs {
		v := &refs[k]
		if v.mode == lwLiteral {
			pix := argb[pos]
			pos++
			a, r, g, b := pix>>24, (pix>>16)&0xff, (pix>>8)&0xff, pix&0xff
			// Shorter keys are prefixes of the longest.
			key := v16HashPix(pix, 32-bitsMax)
			histos[0].blue[b]++
			histos[0].literal[g]++
			histos[0].red[r]++
			histos[0].alpha[a]++
			for i := bitsMax; i >= 1; i, key = i-1, key>>1 {
				h := &histos[i]
				if s.caches[i].colors[key] == pix {
					h.literal[lwNumCodes+key]++
				} else {
					s.caches[i].colors[key] = pix
					h.blue[b]++
					h.literal[g]++
					h.red[r]++
					h.alpha[a]++
				}
			}
			continue
		}
		// Only the length prefix differs by cache size.
		n := int(v.len)
		argbPrev := argb[pos] ^ 0xffffffff
		code, _, _ := prefixBits(n)
		for i := range histos {
			histos[i].literal[lwNumLiteral+code]++
		}
		for ; n != 0; n-- {
			if argb[pos] != argbPrev {
				key := v16HashPix(argb[pos], 32-bitsMax)
				for i := bitsMax; i >= 1; i, key = i-1, key>>1 {
					s.caches[i].colors[key] = argb[pos]
				}
				argbPrev = argb[pos]
			}
			pos++
		}
	}

	best := 0
	entropyMin := uint64(math.MaxUint64)
	for i := range histos {
		entropy := histos[i].estimateBits()
		if i == 0 || entropy < entropyMin {
			entropyMin = entropy
			best = i
		}
	}
	return best
}

// backwardRefsWithLocalCache rewrites literals in refs as cache hits.
// BackwardRefsWithLocalCache, backward_references_enc.c:861.
func (s *lw16State) backwardRefsWithLocalCache(argb []uint32, cacheBits int, refs lwRefs) {
	cc := &s.cc
	cc.init(cacheBits)
	pos := 0
	for k := range refs {
		v := &refs[k]
		if v.mode == lwLiteral {
			lit := v.argbOrDist
			if ix := cc.contains(lit); ix >= 0 {
				*v = pixOrCopy{mode: lw16CacheIdx, len: 1, argbOrDist: uint32(ix)}
			} else {
				cc.insert(lit)
			}
			pos++
		} else {
			for j := 0; j < int(v.len); j++ {
				cc.insert(argb[pos])
				pos++
			}
		}
	}
}

// v16GetBackwardReferences leaves the best refs, as plane codes, in
// refs[0] and returns their cache bits; refs[1] is refs_tmp. Uses
// e.hcBox and s.est.
// GetBackwardReferences, backward_references_enc.c:911, via
// VP8LGetBackwardReferences :1039 (low_effort 0, do_no_cache 0).
func (e *LW) v16GetBackwardReferences(w, h int, argb []uint32, quality, lz77Types, cacheBitsMax int, hc *lwHashChain, refs []lwRefs) int {
	s := e.v16
	lz77TypeBest := 0
	cacheBitsBest := 0
	bitCostBest := uint64(math.MaxUint64)
	histo := &s.est
	histo.alloc(v16MaxCacheBits)
	tmp := &refs[1] // refs_tmp
	for lz77Type := 1; lz77Types != 0; lz77Types, lz77Type = lz77Types&^lz77Type, lz77Type<<1 {
		if lz77Types&lz77Type == 0 {
			continue
		}
		switch lz77Type {
		case lz77RLE:
			backwardReferencesRle(w, h, argb, tmp)
		case lz77Standard:
			backwardReferencesLz77(w, h, argb, hc, tmp)
		case lz77Box:
			e.hcBox.init(w * h)
			e.backwardReferencesLz77Box(w, h, argb, hc, &e.hcBox, tmp)
		}
		// i == 0 only: no do_no_cache pass.
		cacheBits := s.calculateBestCacheSize(argb, quality, *tmp, cacheBitsMax)
		if cacheBits > 0 {
			s.backwardRefsWithLocalCache(argb, cacheBits, *tmp)
		}
		histo.create(*tmp, cacheBits)
		bitCost := histo.estimateBits()
		if bitCost < bitCostBest {
			refs[1], refs[0] = refs[0], refs[1]
			bitCostBest = bitCost
			lz77TypeBest = lz77Type
			cacheBitsBest = cacheBits
		}
	}

	// TraceBackwards only for quality >= 25.
	if (lz77TypeBest == lz77Standard || lz77TypeBest == lz77Box) && quality >= 25 {
		chain := hc
		if lz77TypeBest != lz77Standard {
			chain = &e.hcBox
		}
		e.v16TraceBackwards(w, h, argb, cacheBitsBest, chain, refs[0], &refs[1])
		histo.create(refs[1], cacheBitsBest)
		if histo.estimateBits() < bitCostBest {
			refs[1], refs[0] = refs[0], refs[1]
		}
	}
	backwardReferences2DLocality(w, refs[0])
	return cacheBitsBest
}
