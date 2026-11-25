// Port of libwebp 1.6.0 enc/predictor_enc.c, exact=1, no near-lossless or cross color. PredictBatch, GetResidual and CopyImageWithPrediction are the 1.2.2 port. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

const (
	v16HistoSize     = 4 * 256         // HISTO_SIZE
	v16SpatialBias   = int64(15) << 23 // kSpatialPredictorBias
	v16MaxTransBits  = 9               // MAX_TRANSFORM_BITS, vp8li_enc.h:41
	v16MaxPredImage  = 1 << 14         // MAX_PREDICTOR_IMAGE_SIZE, vp8l_enc.c:39
	v16ModeHistoSize = lwNumPredModes * v16HistoSize
)

// PredictionCostBias, predictor_enc.c:46; expVal is in 1/100.
func v16PredictionCostBias(counts []uint32, weight0, expVal uint64) int64 {
	const significantSymbols = 256 >> 4
	const expDecayFactor uint64 = 6 // in 1/10
	bits := (weight0 * uint64(counts[0])) << v16LogBits
	expVal <<= v16LogBits
	for i := 1; i < significantSymbols; i++ {
		bits += uint64(v16DivRound(int64(expVal*uint64(counts[i]+counts[256-i])), 100))
		expVal = uint64(v16DivRound(int64(expDecayFactor*expVal), 10))
	}
	return -v16DivRound(int64(bits), 10)
}

// PredictionCostSpatialHistogram, predictor_enc.c:60.
func v16PredictionCostSpatialHistogram(acc, tile []uint32, mode, leftMode, aboveMode int) int64 {
	var retval int64
	for i := 0; i < 4; i++ {
		const kExpValue = 94
		retval += v16PredictionCostBias(tile[i*256:], 1, kExpValue)
		retval += int64(v16CombinedShannonEntropy(tile[i*256:], acc[i*256:]))
	}
	// Favor keeping the areas locally similar.
	if mode == leftMode {
		retval -= v16SpatialBias
	}
	if mode == aboveMode {
		retval -= v16SpatialBias
	}
	return retval
}

// UpdateHisto, predictor_enc.c:80.
func v16UpdateHisto(h []uint32, argb uint32) {
	h[argb>>24]++
	h[256+(argb>>16)&0xff]++
	h[2*256+(argb>>8)&0xff]++
	h[3*256+argb&0xff]++
}

// GetBestPredictorForTile, predictor_enc.c:334. allArgb holds the
// mode histograms of each level, allAcc one histogram per level.
func v16GetBestPredictorForTile(allArgb []uint32, sub, tileX, tileY, tilesPerRow int, allAcc []uint32, allModes [][]uint32, allPred []uint32) {
	acc := allAcc[sub*v16HistoSize : (sub+1)*v16HistoSize]
	modes := allModes[sub]
	pred := allPred[sub*lwNumPredModes:]
	leftMode, aboveMode := 0xff, 0xff
	if tileX > 0 {
		leftMode = int(modes[tileY*tilesPerRow+tileX-1]>>8) & 0xff
	}
	if tileY > 0 {
		aboveMode = int(modes[(tileY-1)*tilesPerRow+tileX]>>8) & 0xff
	}
	bestDiff := int64(v16Int64Max)
	bestMode := 0
	best := allArgb[:v16HistoSize]
	for mode := 0; mode < lwNumPredModes; mode++ {
		o := sub*v16ModeHistoSize + mode*v16HistoSize
		h := allArgb[o : o+v16HistoSize]
		curDiff := v16PredictionCostSpatialHistogram(acc, h, mode, leftMode, aboveMode)
		if curDiff < bestDiff {
			best = h
			bestDiff = curDiff
			bestMode = mode
		}
	}
	for i := range acc {
		acc[i] += best[i]
	}
	modes[tileY*tilesPerRow+tileX] = lwARGBBlack | uint32(bestMode)<<8
	pred[bestMode]++
}

// computeResidualsForTile: scratch holds two rows of w+1.
// ComputeResidualsForTile, predictor_enc.c:380.
func v16ComputeResidualsForTile(w, h, tileX, tileY, minBits, updateUpTo int, allArgb, scratch, argb []uint32) {
	startX := tileX << minBits
	startY := tileY << minBits
	tileSize := 1 << minBits
	maxY := min(tileSize, h-startY)
	maxX := min(tileSize, w-startX)
	haveLeft := 0
	if startX > 0 {
		haveLeft = 1
	}
	ctxX := startX - haveLeft
	// Rows are w+1 wide so top-right wraps to the next row.
	upper := scratch[:w+1]
	cur := scratch[w+1 : 2*(w+1)]
	var residuals [1 << v16MaxTransBits]uint32
	for mode := 0; mode < lwNumPredModes; mode++ {
		histo := allArgb[mode*v16HistoSize : (mode+1)*v16HistoSize]
		if startY > 0 {
			o := (startY-1)*w + ctxX
			copy(cur[ctxX:], argb[o:o+maxX+haveLeft+1])
		}
		for ry := 0; ry < maxY; ry++ {
			y := startY + ry
			upper, cur = cur, upper
			n := maxX + haveLeft
			if y+1 < h {
				n++
			}
			o := y*w + ctxX
			copy(cur[ctxX:], argb[o:o+n])
			getResidual(w, h, upper, cur, mode, startX, startX+maxX, y, residuals[:])
			for rx := 0; rx < maxX; rx++ {
				v16UpdateHisto(histo, residuals[rx])
			}
			for s := 1; s <= updateUpTo; s++ {
				o := s*v16ModeHistoSize + mode*v16HistoSize
				super := allArgb[o : o+v16HistoSize]
				for rx := 0; rx < maxX; rx++ {
					v16UpdateHisto(super, residuals[rx])
				}
			}
		}
	}
}

// v16OptimizeSampling returns the bits of the largest uniform squares
// of image and subsamples it in place to them.
// VP8LOptimizeSampling, predictor_enc.c:526.
func v16OptimizeSampling(image []uint32, fullW, fullH, bits, maxBits int) int {
	width := subSampleSize(fullW, bits)
	height := subSampleSize(fullH, bits)
	bestBits := bits
	// Rows first.
	for bestBits < maxBits {
		newSquare := 1 << (bestBits + 1 - bits)
		square := 1 << (bestBits - bits)
		isGood := true
		for y := 0; y+square < height; y += newSquare {
			a := image[y*width : (y+1)*width]
			b := image[(y+square)*width : (y+square+1)*width]
			for i := range a {
				if a[i] != b[i] {
					isGood = false
					break
				}
			}
			if !isGood {
				break
			}
		}
		if !isGood {
			break
		}
		bestBits++
	}
	if bestBits == bits {
		return bits
	}
	// Then columns.
	for bestBits > bits {
		isGood := true
		square := 1 << (bestBits - bits)
		for y := 0; isGood && y < height; y++ {
			for x := 0; isGood && x < width; x += square {
				for i := x + 1; i < min(x+square, width); i++ {
					if image[y*width+i] != image[y*width+x] {
						isGood = false
						break
					}
				}
			}
		}
		if isGood {
			break
		}
		bestBits--
	}
	if bestBits == bits {
		return bits
	}
	oldWidth := width
	square := 1 << (bestBits - bits)
	width = subSampleSize(fullW, bestBits)
	height = subSampleSize(fullH, bestBits)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			image[y*width+x] = image[square*(y*oldWidth+x)]
		}
	}
	return bestBits
}

// getBestPredictorsAndSubSampling walks tiles in Z order over the
// levels minBits..maxBits; allModes[i] is the mode image at minBits+i.
// It returns the chosen bits and mode image, resampled. Uses s.predRaw.
// GetBestPredictorsAndSubSampling, predictor_enc.c:615.
func (s *lw16State) getBestPredictorsAndSubSampling(w, h, minBits, maxBits int, scratch, argb []uint32, allModes [][]uint32) (bestBits int, bestMode []uint32) {
	tilesPerRow := uint32(subSampleSize(w, minBits))
	tilesPerCol := uint32(subSampleSize(h, minBits))
	maxSub := uint32(maxBits - minBits)
	numArgb := int(maxSub+1) * v16ModeHistoSize
	numAcc := int(maxSub+1) * v16HistoSize
	numPred := int(maxSub+1) * lwNumPredModes
	n := numArgb + numAcc + numPred
	if cap(s.predRaw) < n {
		s.predRaw = make([]uint32, n)
	}
	raw := s.predRaw[:n]
	clear(raw) // calloc
	allArgb := raw[:numArgb]
	allAcc := raw[numArgb : numArgb+numAcc]
	allPred := raw[numArgb+numAcc:]
	maxTileSize := uint32(1) << maxSub
	// Levels up to this one take residuals directly.
	updateUpTo := uint32(max(min(4, maxBits), minBits) - minBits)
	var localX, localY, maxTileX, maxTileY, tileX, tileY uint32

	for tileY < tilesPerCol {
		v16ComputeResidualsForTile(w, h, int(tileX), int(tileY), minBits, int(updateUpTo), allArgb, scratch, argb)

		// Update every completed super-tile.
		sub := uint32(0)
		for {
			superX := tileX >> sub
			superY := tileY >> sub
			superPerRow := subSampleSize(w, minBits+int(sub))
			v16GetBestPredictorForTile(allArgb, int(sub), int(superX), int(superY), superPerRow, allAcc, allModes, allPred)
			if sub == maxSub {
				break
			}
			sub++
			if sub > updateUpTo && sub <= maxSub {
				dst := allArgb[int(sub)*v16ModeHistoSize : int(sub+1)*v16ModeHistoSize]
				src := allArgb[int(sub-1)*v16ModeHistoSize : int(sub)*v16ModeHistoSize]
				for i := range dst {
					dst[i] += src[i]
				}
			}
			if !((tileX == tilesPerRow-1 || (localX+1)%(1<<sub) == 0) && //nolint:staticcheck // as libwebp
				(tileY == tilesPerCol-1 || (localY+1)%(1<<sub) == 0)) {
				sub-- // last finished level
				break
			}
		}
		clear(allArgb[:int(sub+1)*v16ModeHistoSize])

		if sub == maxSub {
			// Start a new max-tile.
			if tileX == tilesPerRow-1 {
				maxTileX = 0
				maxTileY++
			} else {
				maxTileX++
			}
			localX, localY = 0, 0
		} else {
			// Z traversal.
			cx := localX >> sub
			cy := localY >> sub
			if tileX == tilesPerRow-1 && cx%2 == 0 {
				cy++
			} else if cx%2 == 0 {
				cx++
			} else {
				cy++
				cx--
			}
			localX = cx << sub
			localY = cy << sub
		}
		tileX = maxTileX*maxTileSize + localX
		tileY = maxTileY*maxTileSize + localY
	}

	bestCost := int64(v16Int64Max)
	for sub := 0; sub <= int(maxSub); sub++ {
		acc := allAcc[sub*v16HistoSize:]
		cost := int64(v16ShannonEntropy(allPred[sub*lwNumPredModes : (sub+1)*lwNumPredModes]))
		for plane := 0; plane < 4; plane++ {
			cost += int64(v16ShannonEntropy(acc[plane*256 : (plane+1)*256]))
		}
		if cost < bestCost {
			bestCost = cost
			bestBits = minBits + sub
			bestMode = allModes[sub]
		}
	}
	bestBits = v16OptimizeSampling(bestMode, w, h, bestBits, v16MaxTransBits)
	return bestBits, bestMode
}

// residualImage writes modes to image and residuals over argb, and
// returns the predictor bits. Uses s.modesRaw.
// VP8LResidualImage, predictor_enc.c:776, low_effort=0.
func (s *lw16State) residualImage(w, h, minBits, maxBits int, argb, scratch, image []uint32) int {
	var num [v16MaxTransBits + 1]int
	sum := 0
	for bits := minBits; bits <= maxBits; bits++ {
		num[bits] = subSampleSize(w, bits) * subSampleSize(h, bits)
		sum += num[bits]
	}
	if cap(s.modesRaw) < sum {
		s.modesRaw = make([]uint32, sum)
	}
	raw := s.modesRaw[:sum]
	var modes [v16MaxTransBits + 1][]uint32
	off := 0
	for bits := minBits; bits <= maxBits; bits++ {
		modes[bits] = raw[off : off+num[bits]]
		off += num[bits]
	}
	bestBits, bestMode := s.getBestPredictorsAndSubSampling(w, h, minBits, maxBits, scratch, argb, modes[minBits:maxBits+1])
	n := subSampleSize(w, bestBits) * subSampleSize(h, bestBits)
	copy(image[:n], bestMode[:n])
	copyImageWithPrediction(w, h, bestBits, image, scratch, argb)
	return bestBits
}
