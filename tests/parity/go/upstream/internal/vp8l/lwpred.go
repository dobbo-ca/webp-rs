// Port of libwebp 1.2.2 enc/predictor_enc.c, exact=1, no near-lossless or cross color. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

const (
	maxDiffCost          float32 = 1e30 // MAX_DIFF_COST
	spatialPredictorBias float32 = 15   // kSpatialPredictorBias
	expDecayFactor       float64 = 0.6
	spatialExpValue      float64 = 0.94
	spatialCostScale     float64 = -0.1
)

// PredictionCostSpatial, predictor_enc.c:33.
func predictionCostSpatial(counts *[256]int32, weight0 int, expVal float64) float32 {
	const significantSymbols = 256 >> 4
	bits := float64(int32(weight0) * counts[0])
	for i := 1; i < significantSymbols; i++ {
		bits += float64(expVal * float64(counts[i]+counts[256-i]))
		expVal *= expDecayFactor
	}
	return float32(spatialCostScale * bits)
}

// predictBatch writes out[0:n] for cur[xStart:xStart+n]; cur and
// upper are whole rows. PredictBatch, predictor_enc.c:68.
func predictBatch(mode, xStart, y, n int, cur, upper, out []uint32) {
	if xStart == 0 {
		if y == 0 {
			predictorsSub(0, cur, nil, 0, 1, out) // ARGB_BLACK
		} else {
			predictorsSub(2, cur, upper, 0, 1, out) // top
		}
		xStart++
		out = out[1:]
		n--
	}
	if y == 0 {
		predictorsSub(1, cur, nil, xStart, n, out) // left
	} else {
		predictorsSub(mode, cur, upper, xStart, n, out)
	}
}

// GetResidual, predictor_enc.c:233, exact=1 branch only.
func getResidual(w, h int, upper, cur []uint32, mode, xStart, xEnd, y int, out []uint32) {
	predictBatch(mode, xStart, y, xEnd-xStart, cur, upper, out)
}

// getBestPredictorForTile needs green-only argb (v<<8).
// GetBestPredictorForTile, predictor_enc.c:298.
func getBestPredictorForTile(w, h, tileX, tileY, bits int, acc *[4][256]int32, argb, modes []uint32) int {
	startX := tileX << bits
	startY := tileY << bits
	tileSize := 1 << bits
	maxY := min(tileSize, h-startY)
	maxX := min(tileSize, w-startX)
	tilesPerRow := subSampleSize(w, bits)
	leftMode, aboveMode := 0xff, 0xff
	if tileX > 0 {
		leftMode = int(modes[tileY*tilesPerRow+tileX-1]>>8) & 0xff
	}
	if tileY > 0 {
		aboveMode = int(modes[(tileY-1)*tilesPerRow+tileX]>>8) & 0xff
	}
	var green [lwNumPredModes][256]int32
	var black, inner int32 // black: all modes; inner: mode 0
	for y := startY; y < startY+maxY; y++ {
		row := argb[y*w : y*w+w]
		for x := startX; x < startX+maxX; x++ {
			g := int(uint8(row[x] >> 8))
			if y == 0 || x == 0 {
				// PredictBatch: black, left or top.
				p := 0
				if x > 0 {
					p = int(uint8(row[x-1] >> 8))
				} else if y > 0 {
					p = int(uint8(argb[(y-1)*w] >> 8))
				} else {
					black++
				}
				for m := range green {
					green[m][uint8(g-p)]++
				}
				continue
			}
			// TR wraps at x=w-1.
			up := argb[(y-1)*w+x-1:]
			l := int(uint8(row[x-1] >> 8))
			tl, t, tr := int(uint8(up[0]>>8)), int(uint8(up[1]>>8)), int(uint8(up[2]>>8))
			sel := l // Select(T, L, TL)
			if absInt(l-tl)-absInt(t-tl) <= 0 {
				sel = t
			}
			ave := (l + t) >> 1
			p := [lwNumPredModes]int{
				0, l, t, tr, tl,
				(((l + tr) >> 1) + t) >> 1,
				(l + tl) >> 1,
				ave,
				(tl + t) >> 1,
				(t + tr) >> 1,
				(((l + tl) >> 1) + ((t + tr) >> 1)) >> 1,
				sel,
				min(max(l+t-tl, 0), 255),
				min(max(ave+(ave-tl)/2, 0), 255),
			}
			for m := range green {
				green[m][uint8(g-p[m])]++
			}
			inner++
		}
	}

	n := int32(maxX * maxY)
	var alpha, rb [256]int32
	rb[0] = n
	psRB := predictionCostSpatial(&rb, 1, spatialExpValue)
	ceR := combinedShannonEntropy(&rb, &acc[1])
	ceB := combinedShannonEntropy(&rb, &acc[3])
	nAlpha1 := [2]int32{black + inner, black} // mode 0, others
	var psA, ceA [2]float32
	for i, k := range nAlpha1 {
		alpha[0], alpha[1] = n-k, k
		psA[i] = predictionCostSpatial(&alpha, 1, spatialExpValue)
		ceA[i] = combinedShannonEntropy(&alpha, &acc[0])
	}
	bestDiff := maxDiffCost
	bestMode := 0
	for mode := 0; mode < lwNumPredModes; mode++ {
		a := min(mode, 1)
		// PredictionCostSpatialHistogram, predictor_enc.c:46.
		var retval float64
		retval += float64(psA[a])
		retval += float64(ceA[a])
		retval += float64(psRB)
		retval += float64(ceR)
		retval += float64(predictionCostSpatial(&green[mode], 1, spatialExpValue))
		retval += float64(combinedShannonEntropy(&green[mode], &acc[2]))
		retval += float64(psRB)
		retval += float64(ceB)
		curDiff := float32(retval)
		// Favor keeping the areas locally similar.
		if mode == leftMode {
			curDiff -= spatialPredictorBias
		}
		if mode == aboveMode {
			curDiff -= spatialPredictorBias
		}
		if curDiff < bestDiff {
			bestDiff = curDiff
			bestMode = mode
		}
	}

	k := nAlpha1[min(bestMode, 1)]
	acc[0][0] += n - k
	acc[0][1] += k
	acc[1][0] += n
	for j, v := range green[bestMode] {
		acc[2][j] += v
	}
	acc[3][0] += n
	return bestMode
}

// CopyImageWithPrediction, predictor_enc.c:413.
func copyImageWithPrediction(w, h, bits int, modes, scratch, argb []uint32) {
	tilesPerRow := subSampleSize(w, bits)
	upper := scratch[:w+1]
	cur := scratch[w+1 : 2*(w+1)]
	for y := 0; y < h; y++ {
		upper, cur = cur, upper
		n := w
		if y+1 < h {
			n++
		}
		copy(cur, argb[y*w:y*w+n])
		for x := 0; x < w; {
			mode := int(modes[(y>>bits)*tilesPerRow+(x>>bits)]>>8) & 0xff
			xEnd := x + 1<<bits
			if xEnd > w {
				xEnd = w
			}
			getResidual(w, h, upper, cur, mode, x, xEnd, y, argb[y*w+x:])
			x = xEnd
		}
	}
}

// residualImage writes modes to image and residuals over argb.
// VP8LResidualImage, predictor_enc.c:475.
func residualImage(w, h, bits int, argb, scratch, image []uint32) {
	tilesPerRow := subSampleSize(w, bits)
	tilesPerCol := subSampleSize(h, bits)
	var histo [4][256]int32
	for tileY := 0; tileY < tilesPerCol; tileY++ {
		for tileX := 0; tileX < tilesPerRow; tileX++ {
			pred := getBestPredictorForTile(w, h, tileX, tileY, bits, &histo, argb, image)
			image[tileY*tilesPerRow+tileX] = lwARGBBlack | uint32(pred)<<8
		}
	}
	copyImageWithPrediction(w, h, bits, image, scratch, argb)
}
