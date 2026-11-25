// Port of libwebp 1.2.2 enc/backward_references_enc.c, without the color cache. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import (
	"math"
	"math/bits"
	"slices"
)

const (
	lwMinLength        = 4  // MIN_LENGTH
	lwWindowOffsetsMax = 32 // WINDOW_OFFSETS_SIZE_MAX
)

// plane_to_code_lut, backward_references_enc.c:38.
var planeToCodeLUT = [128]uint8{
	96, 73, 55, 39, 23, 13, 5, 1, 255, 255, 255, 255, 255, 255, 255, 255,
	101, 78, 58, 42, 26, 16, 8, 2, 0, 3, 9, 17, 27, 43, 59, 79,
	102, 86, 62, 46, 32, 20, 10, 6, 4, 7, 11, 21, 33, 47, 63, 87,
	105, 90, 70, 52, 37, 28, 18, 14, 12, 15, 19, 29, 38, 53, 71, 91,
	110, 99, 82, 66, 48, 35, 30, 24, 22, 25, 31, 36, 49, 67, 83, 100,
	115, 108, 94, 76, 64, 50, 44, 40, 34, 41, 45, 51, 65, 77, 95, 109,
	118, 113, 103, 92, 80, 68, 60, 56, 54, 57, 61, 69, 81, 93, 104, 114,
	119, 116, 111, 106, 97, 88, 84, 74, 72, 75, 85, 89, 98, 107, 112, 117,
}

// VP8LHashChainFindOffset, backward_references_enc.h.
func (hc *lwHashChain) findOffset(i int) int {
	return int(hc.offLen[i] >> lwMaxLengthBits)
}

// VP8LHashChainFindLength, backward_references_enc.h.
func (hc *lwHashChain) findLength(i int) int {
	return int(hc.offLen[i] & (1<<lwMaxLengthBits - 1))
}

// VP8LHashChainInit, backward_references_enc.c:203; reuses capacity.
func (hc *lwHashChain) init(size int) {
	if cap(hc.offLen) < size {
		hc.offLen = make([]uint32, size)
	}
	hc.offLen = hc.offLen[:size]
}

// VP8LBackwardRefsCursorAdd, backward_references_enc.c:190.
func (r *lwRefs) add(v pixOrCopy) {
	*r = append(*r, v)
}

// VP8LDistanceToPlaneCode, backward_references_enc.c:50.
func distanceToPlaneCode(xsize, dist int) int {
	yoffset := dist / xsize
	xoffset := dist - yoffset*xsize
	if xoffset <= 8 && yoffset < 8 {
		return int(planeToCodeLUT[yoffset*16+8-xoffset]) + 1
	} else if xoffset > xsize-8 && yoffset < 7 {
		return int(planeToCodeLUT[(yoffset+1)*16+8+(xsize-xoffset)]) + 1
	}
	return dist + 120
}

// FindMatchLength, backward_references_enc.c:66.
func findMatchLength(a1, a2 []uint32, bestLen, maxLimit int) int {
	if a1[bestLen] != a2[bestLen] {
		return 0
	}
	return vectorMismatch(a1, a2, maxLimit)
}

// GetPixPairHash64, backward_references_enc.c:229; reads argb[0:2].
func getPixPairHash64(argb []uint32) uint32 {
	key := argb[1] * 0xc6a4a793
	key += argb[0] * 0x5bd1e996
	return key >> (32 - lwHashBits)
}

// GetMaxItersForQuality, backward_references_enc.c:239.
func getMaxItersForQuality(quality int) int {
	return 8 + (quality*quality)/128
}

// GetWindowSizeForHashChain, backward_references_enc.c:243.
func getWindowSizeForHashChain(quality, xsize int) int {
	var m int
	switch {
	case quality > 75:
		m = lwWindowSize
	case quality > 50:
		m = xsize << 8
	case quality > 25:
		m = xsize << 6
	default:
		m = xsize << 4
	}
	if m > lwWindowSize {
		return lwWindowSize
	}
	return m
}

// MaxFindCopyLength, backward_references_enc.c:252.
func maxFindCopyLength(n int) int {
	if n < lwMaxLength {
		return n
	}
	return lwMaxLength
}

// hashChainFill uses e.hashFirst. hc must be init'ed to >= xsize*ysize.
// VP8LHashChainFill, backward_references_enc.c:256, low_effort=0.
func (e *LW) hashChainFill(hc *lwHashChain, quality int, argb []uint32, xsize, ysize int) {
	size := xsize * ysize
	iterMax := getMaxItersForQuality(quality)
	windowSize := getWindowSizeForHashChain(quality, xsize)
	ol := hc.offLen
	if size <= 2 {
		ol[size-1] = 0
		ol[0] = 0
		return
	}
	if e.hashFirst == nil {
		e.hashFirst = make([]int32, lwHashSize)
	}
	// first holds pos+1; 0 is empty.
	first := e.hashFirst
	clear(first)
	// ol doubles as an int32 chain until the match pass.
	pos := 0
	argbComp := argb[0] == argb[1]
	for pos < size-2 {
		argbCompNext := argb[pos+1] == argb[pos+2]
		if argbComp && argbCompNext {
			// Runs hash on (color, run length).
			var tmp [2]uint32
			ln := 1
			tmp[0] = argb[pos]
			for pos+ln+2 < size && argb[pos+ln+2] == argb[pos] {
				ln++
			}
			if ln > lwMaxLength {
				for k := pos; k < pos+ln-lwMaxLength; k++ {
					ol[k] = 0xffffffff
				}
				pos += ln - lwMaxLength
				ln = lwMaxLength
			}
			for ln != 0 {
				tmp[1] = uint32(ln)
				ln--
				h := getPixPairHash64(tmp[:])
				ol[pos] = uint32(first[h]) - 1
				first[h] = int32(pos + 1)
				pos++
			}
			argbComp = false
		} else {
			h := getPixPairHash64(argb[pos:])
			ol[pos] = uint32(first[h]) - 1
			first[h] = int32(pos + 1)
			pos++
			argbComp = argbCompNext
		}
	}
	// Penultimate pixel.
	ol[pos] = uint32(first[getPixPairHash64(argb[pos:])]) - 1

	ol[size-1] = 0
	ol[0] = 0
	for base := size - 2; base > 0; {
		maxLen := maxFindCopyLength(size - 1 - base)
		start := argb[base:]
		iter := iterMax
		bestLength := 0
		bestDistance := 0
		minPos := 0
		if base > windowSize {
			minPos = base - windowSize
		}
		lengthMax := maxLen
		if lengthMax > 256 {
			lengthMax = 256
		}

		pos = int(int32(ol[base]))
		// Seed with the pixel above, then the one to the left.
		if base >= xsize {
			curr := findMatchLength(argb[base-xsize:], start, bestLength, maxLen)
			if curr > bestLength {
				bestLength = curr
				bestDistance = xsize
			}
			iter--
		}
		curr := findMatchLength(argb[base-1:], start, bestLength, maxLen)
		if curr > bestLength {
			bestLength = curr
			bestDistance = 1
		}
		iter--
		if bestLength == lwMaxLength {
			pos = minPos - 1
		}
		bestArgb := start[bestLength]

		for ; pos >= minPos; pos = int(int32(ol[pos])) {
			iter--
			if iter == 0 {
				break
			}
			if argb[pos+bestLength] != bestArgb {
				continue
			}
			curr := vectorMismatch(argb[pos:], start, maxLen)
			if bestLength < curr {
				bestLength = curr
				bestDistance = base - pos
				bestArgb = start[bestLength]
				if bestLength >= lengthMax {
					break
				}
			}
		}
		// Extend the match to the left while it holds.
		maxBase := base
		for {
			ol[base] = uint32(bestDistance)<<lwMaxLengthBits | uint32(bestLength)
			base--
			if bestDistance == 0 || base == 0 {
				break
			}
			if base < bestDistance || argb[base-bestDistance] != argb[base] {
				break
			}
			if bestLength == lwMaxLength && bestDistance != 1 && base+lwMaxLength < maxBase {
				break
			}
			if bestLength < lwMaxLength {
				bestLength++
				maxBase = base
			}
		}
	}
}

// AddSingleLiteral, backward_references_enc.c:424, no cache.
func addSingleLiteral(pixel uint32, refs *lwRefs) {
	refs.add(pixOrCopy{mode: lwLiteral, len: 1, argbOrDist: pixel})
}

// BackwardReferencesRle, backward_references_enc.c:442.
func backwardReferencesRle(xsize, ysize int, argb []uint32, refs *lwRefs) {
	pixCount := xsize * ysize
	*refs = slices.Grow((*refs)[:0], pixCount) // at most one ref per pixel
	addSingleLiteral(argb[0], refs)
	i := 1
	for i < pixCount {
		maxLen := maxFindCopyLength(pixCount - i)
		rleLen := findMatchLength(argb[i:], argb[i-1:], 0, maxLen)
		prevRowLen := 0
		if i >= xsize {
			prevRowLen = findMatchLength(argb[i:], argb[i-xsize:], 0, maxLen)
		}
		if rleLen >= prevRowLen && rleLen >= lwMinLength {
			refs.add(pixOrCopy{mode: lwCopy, len: uint16(rleLen), argbOrDist: 1})
			i += rleLen
		} else if prevRowLen >= lwMinLength {
			refs.add(pixOrCopy{mode: lwCopy, len: uint16(prevRowLen), argbOrDist: uint32(xsize)})
			i += prevRowLen
		} else {
			addSingleLiteral(argb[i], refs)
			i++
		}
	}
}

// BackwardReferencesLz77, backward_references_enc.c:485.
func backwardReferencesLz77(xsize, ysize int, argb []uint32, hc *lwHashChain, refs *lwRefs) {
	iLastCheck := -1
	pixCount := xsize * ysize
	*refs = slices.Grow((*refs)[:0], pixCount)
	for i := 0; i < pixCount; {
		offset := hc.findOffset(i)
		ln := hc.findLength(i)
		if ln >= lwMinLength {
			lenIni := ln
			maxReach := 0
			jMax := i + lenIni
			if i+lenIni >= pixCount {
				jMax = pixCount - 1
			}
			if i > iLastCheck {
				iLastCheck = i
			}
			// Best split of [i,j) plus match at j.
			for j := iLastCheck + 1; j <= jMax; j++ {
				lenJ := hc.findLength(j)
				reach := j + 1
				if lenJ >= lwMinLength {
					reach = j + lenJ
				}
				if reach > maxReach {
					ln = j - i
					maxReach = reach
					if maxReach >= pixCount {
						break
					}
				}
			}
		} else {
			ln = 1
		}
		if ln == 1 {
			addSingleLiteral(argb[i], refs)
		} else {
			refs.add(pixOrCopy{mode: lwCopy, len: uint16(ln), argbOrDist: uint32(offset)})
		}
		i += ln
	}
}

// backwardReferencesLz77Box fills box from best and uses e.boxCounts.
// BackwardReferencesLz77Box, backward_references_enc.c:557.
func (e *LW) backwardReferencesLz77Box(xsize, ysize int, argb []uint32, best, box *lwHashChain, refs *lwRefs) {
	pixCount := xsize * ysize
	var windowOffsets, windowOffsetsNew [lwWindowOffsetsMax]int
	var offInd, offIndNew [lwWindowOffsetsMax]int // into windowOffsets
	var matchEnd [lwWindowOffsetsMax]int          // exact match end, or 0
	maxWo := 0
	windowOffsetsSize := 0
	windowOffsetsNewSize := 0
	bestOffsetPrev, bestLengthPrev := -1, -1
	if cap(e.boxCounts) < pixCount {
		e.boxCounts = make([]uint16, pixCount)
	}
	counts := e.boxCounts[:pixCount]

	// Run length at i, capped at lwMaxLength.
	counts[pixCount-1] = 1
	for i := pixCount - 2; i >= 0; i-- {
		if argb[i] == argb[i+1] {
			c := counts[i+1]
			if c != lwMaxLength {
				c++
			}
			counts[i] = c
		} else {
			counts[i] = 1
		}
	}

	// Window offsets in plane code order.
	for y := 0; y <= 6; y++ {
		for x := -6; x <= 6; x++ {
			offset := y*xsize + x
			if offset <= 0 {
				continue
			}
			planeCode := distanceToPlaneCode(xsize, offset) - 1
			if planeCode >= lwWindowOffsetsMax {
				continue
			}
			windowOffsets[planeCode] = offset
		}
	}
	// Narrow images do not reach every plane code.
	for i := 0; i < lwWindowOffsetsMax; i++ {
		if windowOffsets[i] == 0 {
			continue
		}
		windowOffsets[windowOffsetsSize] = windowOffsets[i]
		offInd[windowOffsetsSize] = windowOffsetsSize
		maxWo = max(maxWo, windowOffsets[i])
		windowOffsetsSize++
	}
	// Offsets reaching pixels that P-1 cannot reach.
	for i := 0; i < windowOffsetsSize; i++ {
		isReachable := false
		for j := 0; j < windowOffsetsSize && !isReachable; j++ {
			isReachable = isReachable || windowOffsets[i] == windowOffsets[j]+1
		}
		if !isReachable {
			windowOffsetsNew[windowOffsetsNewSize] = windowOffsets[i]
			offIndNew[windowOffsetsNewSize] = i
			windowOffsetsNewSize++
		}
	}

	ol := box.offLen
	ol[0] = 0
	for i := 1; i < pixCount; i++ {
		bestLength := best.findLength(i)
		bestOffset := 0
		doCompute := true
		if bestLength >= lwMaxLength {
			// Keep a maximal match already inside the window.
			bestOffset = best.findOffset(i)
			for ind := 0; ind < windowOffsetsSize; ind++ {
				if bestOffset == windowOffsets[ind] {
					doCompute = false
					break
				}
			}
		}
		if doCompute {
			usePrev := bestLengthPrev > 1 && bestLengthPrev < lwMaxLength
			wos, inds := windowOffsets[:windowOffsetsSize], offInd[:windowOffsetsSize]
			bestLength = 0
			bestOffset = 0
			if usePrev {
				wos, inds = windowOffsetsNew[:windowOffsetsNewSize], offIndNew[:windowOffsetsNewSize]
				bestLength = bestLengthPrev - 1
				bestOffset = bestOffsetPrev
			}
			// Bit ind set when argb[i-wos[ind]] == argb[i].
			p := argb[i]
			var mask uint32
			if i >= maxWo {
				for ind, wo := range wos {
					mask |= uint32((uint64(argb[i-wo]^p)-1)>>63) << ind
				}
			} else {
				for ind, wo := range wos {
					if wo <= i && argb[i-wo] == p {
						mask |= 1 << ind
					}
				}
			}
			for ; mask != 0; mask &= mask - 1 {
				ind := bits.TrailingZeros32(mask)
				wo := wos[ind]
				k := inds[ind]
				// Inside a known match, no walk.
				currLength := matchEnd[k] - i
				if currLength <= 0 {
					currLength = 0
					j := i
					jOffset := i - wo
					// Sum the matching run lengths.
					for {
						countsJOffset := int(counts[jOffset])
						countsJ := int(counts[j])
						if countsJOffset != countsJ {
							if countsJOffset < countsJ {
								currLength += countsJOffset
							} else {
								currLength += countsJ
							}
							break
						}
						currLength += countsJOffset
						jOffset += countsJOffset
						j += countsJOffset
						if !(currLength <= lwMaxLength && j < pixCount && argb[jOffset] == argb[j]) { //nolint:staticcheck // as libwebp
							break
						}
					}
					// Only an uncapped walk is exact.
					if currLength <= lwMaxLength {
						matchEnd[k] = i + currLength
					}
				}
				if bestLength < currLength {
					bestOffset = wo
					if currLength >= lwMaxLength {
						bestLength = lwMaxLength
						break
					}
					bestLength = currLength
				}
			}
		}

		if bestLength <= lwMinLength {
			ol[i] = 0
			bestOffsetPrev = 0
			bestLengthPrev = 0
		} else {
			ol[i] = uint32(bestOffset<<lwMaxLengthBits) | uint32(bestLength)
			bestOffsetPrev = bestOffset
			bestLengthPrev = bestLength
		}
	}
	ol[0] = 0
	backwardReferencesLz77(xsize, ysize, argb, box, refs)
}

// BackwardReferences2DLocality, backward_references_enc.c:708; in place.
func backwardReferences2DLocality(xsize int, refs lwRefs) {
	for k := range refs {
		if refs[k].mode == lwCopy {
			dist := int(refs[k].argbOrDist)
			refs[k].argbOrDist = uint32(distanceToPlaneCode(xsize, dist))
		}
	}
}

// getBackwardReferences leaves the best refs, as plane codes, in
// refs[0]; refs[1] is refs_tmp. Uses e.hcBox and e.estHisto.
// GetBackwardReferences, backward_references_enc.c:882, via
// VP8LGetBackwardReferences :1009 (low_effort 0, cache 0, do_no_cache 0).
func (e *LW) getBackwardReferences(w, h int, argb []uint32, quality, lz77Types int, hc *lwHashChain, refs []lwRefs) {
	lz77TypeBest := 0
	bitCostBest := math.MaxFloat64
	histo := &e.estHisto
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
		// Cache bits 0, no do_no_cache pass.
		histo.create(*tmp)
		bitCost := histo.estimateBits()
		if bitCost < bitCostBest {
			refs[1], refs[0] = refs[0], refs[1]
			bitCostBest = bitCost
			lz77TypeBest = lz77Type
		}
	}

	// TraceBackwards only for quality >= 25.
	if (lz77TypeBest == lz77Standard || lz77TypeBest == lz77Box) && quality >= 25 {
		chain := hc
		if lz77TypeBest != lz77Standard {
			chain = &e.hcBox
		}
		e.backwardReferencesTraceBackwards(w, h, argb, chain, refs[0], &refs[1])
		histo.create(refs[1])
		if histo.estimateBits() < bitCostBest {
			refs[1], refs[0] = refs[0], refs[1]
		}
	}
	backwardReferences2DLocality(w, refs[0])
}
