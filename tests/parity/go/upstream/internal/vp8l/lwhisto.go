// Port of libwebp 1.2.2 enc/histogram_enc.c. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "slices"

const (
	lwMaxCost        float64 = 1.e38 // MAX_COST
	lwNumPartitions          = 4     // NUM_PARTITIONS
	lwBinSize                = lwNumPartitions * lwNumPartitions * lwNumPartitions
	lwMaxHistoGreedy         = 100 // MAX_HISTO_GREEDY

	lwInvalidHistoSym uint16 = 0xffff // kInvalidHistogramSymbol

	lwSmallBias   float64 = 9.1  // InitialHuffmanCost kSmallBias
	lwBinIDEps    float64 = 1e-6 // GetBinIdForEntropy
	lwBinIDFactor float64 = float64(lwNumPartitions) - lwBinIDEps
)

// alloc sizes s to n cleared histograms, h[i] = &pool[i].
// VP8LAllocateHistogramSet, histogram_enc.c:146.
func (s *lwHistoSet) alloc(n int) {
	if cap(s.pool) < n {
		s.pool = make([]lwHisto, n)
	}
	s.pool = s.pool[:n]
	if cap(s.h) < n {
		s.h = make([]*lwHisto, n)
	}
	s.max = n
	s.clear()
}

// clear also undoes pointer swaps, as C resets the pointers.
// VP8LHistogramSetClear, histogram_enc.c:165.
func (s *lwHistoSet) clear() {
	s.h = s.h[:s.max]
	for i := range s.h {
		s.pool[i] = lwHisto{}
		s.h[i] = &s.pool[i]
	}
}

// HistogramSetRemoveHistogram, histogram_enc.c:184.
func (s *lwHistoSet) remove(i int, numUsed *int) {
	s.h[:s.max][i] = nil
	*numUsed--
	if i == len(s.h)-1 {
		for len(s.h) >= 1 && s.h[len(s.h)-1] == nil {
			s.h = s.h[:len(s.h)-1]
		}
	}
}

// HistogramClear, histogram_enc.c:34. HistogramCopy :50 is *dst = *src.
func (h *lwHisto) clear() {
	*h = lwHisto{}
}

// VP8LHistogramInit, histogram_enc.c:96.
func (h *lwHisto) init(arrays bool) {
	if arrays {
		h.clear()
		return
	}
	h.trivialSymbol = 0
	h.bitCost = 0
	h.literalCost = 0
	h.redCost = 0
	h.blueCost = 0
	h.isUsed = [5]bool{}
}

// VP8LHistogramStoreRefs, histogram_enc.c:77.
func (h *lwHisto) storeRefs(refs lwRefs) {
	for i := range refs {
		h.addSinglePixOrCopy(&refs[i], 0)
	}
}

// VP8LHistogramCreate, histogram_enc.c:86.
func (h *lwHisto) create(refs lwRefs) {
	h.clear()
	h.storeRefs(refs)
}

// addSinglePixOrCopy maps distances through distanceToPlaneCode when
// planeXsize > 0 (CostModelBuild), else counts them raw.
// VP8LHistogramAddSinglePixOrCopy, histogram_enc.c:199.
func (h *lwHisto) addSinglePixOrCopy(v *pixOrCopy, planeXsize int) {
	if v.mode == lwLiteral {
		p := v.argbOrDist
		h.alpha[p>>24]++
		h.red[(p>>16)&0xff]++
		h.literal[(p>>8)&0xff]++
		h.blue[p&0xff]++
		return
	}
	code, _, _ := prefixBits(int(v.len))
	h.literal[lwNumLiteral+code]++
	dist := int(v.argbOrDist)
	if planeXsize > 0 {
		dist = distanceToPlaneCode(planeXsize, dist)
	}
	code, _, _ = prefixBits(dist)
	h.distance[code]++
}

// BitsEntropyRefine, histogram_enc.c:231.
func bitsEntropyRefine(b *bitEntropy) float64 {
	var mix float64
	if b.nonzeros < 5 {
		if b.nonzeros <= 1 {
			return 0
		}
		if b.nonzeros == 2 {
			return float64(0.99*float64(b.sum)) + float64(0.01*b.entropy)
		}
		if b.nonzeros == 3 {
			mix = 0.95
		} else {
			mix = 0.7
		}
	} else {
		mix = 0.627
	}
	minLimit := float64(2*b.sum - b.maxVal) // uint32 wrap as in C
	minLimit = float64(mix*minLimit) + float64((1.0-mix)*b.entropy)
	if b.entropy < minLimit {
		return minLimit
	}
	return b.entropy
}

// VP8LBitsEntropy, histogram_enc.c:263.
func bitsEntropy(a []uint32) float64 {
	var b bitEntropy
	bitsEntropyUnrefined(a, &b)
	return bitsEntropyRefine(&b)
}

// InitialHuffmanCost, histogram_enc.c:270.
func initialHuffmanCost() float64 {
	return float64(lwCodeLenCodes*3) - lwSmallBias
}

// FinalHuffmanCost, histogram_enc.c:279.
func finalHuffmanCost(s *streaks) float64 {
	retval := initialHuffmanCost()
	retval += float64(float64(s.counts[0])*1.5625) + float64(0.234375*float64(s.streaks[0][1]))
	retval += float64(float64(s.counts[1])*2.578125) + float64(0.703125*float64(s.streaks[1][1]))
	retval += float64(1.796875 * float64(s.streaks[0][0]))
	retval += float64(3.28125 * float64(s.streaks[1][0]))
	return retval
}

// populationCost: trivialSym may be nil.
// PopulationCost, histogram_enc.c:299.
func populationCost(pop []uint32, trivialSym *uint32, isUsed *bool) float64 {
	var b bitEntropy
	var s streaks
	getEntropyUnrefined(pop, &b, &s)
	if trivialSym != nil {
		if b.nonzeros == 1 {
			*trivialSym = b.nonzeroCode
		} else {
			*trivialSym = lwNonTrivialSym
		}
	}
	*isUsed = s.streaks[1][0] != 0 || s.streaks[1][1] != 0
	return bitsEntropyRefine(&b) + finalHuffmanCost(&s)
}

// GetCombinedEntropy, histogram_enc.c:317; length is len(x).
func getCombinedEntropy(x, y []uint32, isXUsed, isYUsed, trivialAtEnd bool) float64 {
	var s streaks
	length := len(x)
	if trivialAtEnd {
		s.streaks[1][0] = 1
		s.counts[0] = 1
		s.streaks[0][1] = length - 1
		return finalHuffmanCost(&s)
	}
	var b bitEntropy
	if isXUsed {
		if isYUsed {
			getCombinedEntropyUnrefined(x, y, &b, &s)
		} else {
			getEntropyUnrefined(x, &b, &s)
		}
	} else {
		if isYUsed {
			getEntropyUnrefined(y, &b, &s)
		} else {
			s.counts[0] = 1
			if length > 3 {
				s.streaks[0][1] = length
			} else {
				s.streaks[0][0] = length
			}
			b.init()
		}
	}
	return bitsEntropyRefine(&b) + finalHuffmanCost(&s)
}

// VP8LHistogramEstimateBits, histogram_enc.c:359.
func (h *lwHisto) estimateBits() float64 {
	return populationCost(h.literal[:], nil, &h.isUsed[0]) +
		populationCost(h.red[:], nil, &h.isUsed[1]) +
		populationCost(h.blue[:], nil, &h.isUsed[2]) +
		populationCost(h.alpha[:], nil, &h.isUsed[3]) +
		populationCost(h.distance[:], nil, &h.isUsed[4]) +
		extraCost(h.literal[lwNumLiteral:]) +
		extraCost(h.distance[:])
}

// GetCombinedHistogramEntropy, histogram_enc.c:374. Keep the early exits.
func getCombinedHistogramEntropy(a, b *lwHisto, threshold float64, cost *float64) bool {
	trivialAtEnd := false
	*cost += getCombinedEntropy(a.literal[:], b.literal[:], a.isUsed[0], b.isUsed[0], false)
	*cost += extraCostCombined(a.literal[lwNumLiteral:], b.literal[lwNumLiteral:])
	if *cost > threshold {
		return false
	}

	if a.trivialSymbol != lwNonTrivialSym && a.trivialSymbol == b.trivialSymbol {
		// A, R and B are all 0 or 0xff.
		ca := (a.trivialSymbol >> 24) & 0xff
		cr := (a.trivialSymbol >> 16) & 0xff
		cb := a.trivialSymbol & 0xff
		if (ca == 0 || ca == 0xff) && (cr == 0 || cr == 0xff) && (cb == 0 || cb == 0xff) {
			trivialAtEnd = true
		}
	}

	*cost += getCombinedEntropy(a.red[:], b.red[:], a.isUsed[1], b.isUsed[1], trivialAtEnd)
	if *cost > threshold {
		return false
	}
	*cost += getCombinedEntropy(a.blue[:], b.blue[:], a.isUsed[2], b.isUsed[2], trivialAtEnd)
	if *cost > threshold {
		return false
	}
	*cost += getCombinedEntropy(a.alpha[:], b.alpha[:], a.isUsed[3], b.isUsed[3], trivialAtEnd)
	if *cost > threshold {
		return false
	}
	*cost += getCombinedEntropy(a.distance[:], b.distance[:], a.isUsed[4], b.isUsed[4], false)
	*cost += extraCostCombined(a.distance[:], b.distance[:])
	return !(*cost > threshold) //nolint:staticcheck // as libwebp
}

// HistogramAdd, histogram_enc.c:427.
func histogramAdd(a, b, out *lwHisto) {
	vp8lHistogramAdd(a, b, out)
	if a.trivialSymbol == b.trivialSymbol {
		out.trivialSymbol = a.trivialSymbol
	} else {
		out.trivialSymbol = lwNonTrivialSym
	}
}

// HistogramAddEval, histogram_enc.c:442.
func histogramAddEval(a, b, out *lwHisto, threshold float64) float64 {
	cost := 0.0
	sumCost := a.bitCost + b.bitCost
	threshold += sumCost
	if getCombinedHistogramEntropy(a, b, threshold, &cost) {
		histogramAdd(a, b, out)
		out.bitCost = cost
	}
	return cost - sumCost
}

// HistogramAddThresh, histogram_enc.c:462.
func histogramAddThresh(a, b *lwHisto, threshold float64) float64 {
	cost := -a.bitCost
	getCombinedHistogramEntropy(a, b, threshold, &cost)
	return cost
}

// DominantCostRangeInit, histogram_enc.c:487.
func (c *dominantCostRange) init() {
	c.literalMax = 0
	c.literalMin = lwMaxCost
	c.redMax = 0
	c.redMin = lwMaxCost
	c.blueMax = 0
	c.blueMin = lwMaxCost
}

// UpdateDominantCostRange, histogram_enc.c:496.
func (c *dominantCostRange) update(h *lwHisto) {
	if c.literalMax < h.literalCost {
		c.literalMax = h.literalCost
	}
	if c.literalMin > h.literalCost {
		c.literalMin = h.literalCost
	}
	if c.redMax < h.redCost {
		c.redMax = h.redCost
	}
	if c.redMin > h.redCost {
		c.redMin = h.redCost
	}
	if c.blueMax < h.blueCost {
		c.blueMax = h.blueCost
	}
	if c.blueMin > h.blueCost {
		c.blueMin = h.blueCost
	}
}

// UpdateHistogramCost, histogram_enc.c:506.
func updateHistogramCost(h *lwHisto) {
	var alphaSym, redSym, blueSym uint32
	alphaCost := populationCost(h.alpha[:], &alphaSym, &h.isUsed[3])
	distanceCost := populationCost(h.distance[:], nil, &h.isUsed[4]) + extraCost(h.distance[:])
	h.literalCost = populationCost(h.literal[:], nil, &h.isUsed[0]) + extraCost(h.literal[lwNumLiteral:])
	h.redCost = populationCost(h.red[:], &redSym, &h.isUsed[1])
	h.blueCost = populationCost(h.blue[:], &blueSym, &h.isUsed[2])
	h.bitCost = h.literalCost + h.redCost + h.blueCost + alphaCost + distanceCost
	if alphaSym|redSym|blueSym == lwNonTrivialSym {
		h.trivialSymbol = lwNonTrivialSym
	} else {
		h.trivialSymbol = alphaSym<<24 | redSym<<16 | blueSym
	}
}

// GetBinIdForEntropy, histogram_enc.c:532.
func getBinIdForEntropy(lo, hi, val float64) int {
	rng := hi - lo
	if rng > 0 {
		delta := val - lo
		return int(lwBinIDFactor * delta / rng)
	}
	return 0
}

// GetHistoBinIndex, histogram_enc.c:542, low_effort=0.
func getHistoBinIndex(h *lwHisto, c *dominantCostRange) int {
	binID := getBinIdForEntropy(c.literalMin, c.literalMax, h.literalCost)
	binID = binID*lwNumPartitions + getBinIdForEntropy(c.redMin, c.redMax, h.redCost)
	binID = binID*lwNumPartitions + getBinIdForEntropy(c.blueMin, c.blueMax, h.blueCost)
	return binID
}

// HistogramBuild, histogram_enc.c:558.
func histogramBuild(xsize, histoBits int, refs lwRefs, set *lwHistoSet) {
	// Caller's alloc has cleared set.
	x, y := 0, 0
	histoXsize := subSampleSize(xsize, histoBits)
	for i := range refs {
		v := &refs[i]
		ix := (y>>histoBits)*histoXsize + (x >> histoBits)
		set.h[ix].addSinglePixOrCopy(v, 0)
		x += int(v.len)
		for x >= xsize {
			x -= xsize
			y++
		}
	}
}

// HistogramCopyAndAnalyze, histogram_enc.c:582.
func histogramCopyAndAnalyze(orig, image *lwHistoSet, numUsed *int, symbols []uint16) {
	numUsedOrig := *numUsed
	origH := orig.h[:orig.max]
	imgH := image.h[:image.max]
	clusterID := 0
	for i := 0; i < orig.max; i++ {
		histo := origH[i]
		updateHistogramCost(histo)
		// Empty tiles take the previous id, see HistogramRemap.
		if !histo.isUsed[0] && !histo.isUsed[1] && !histo.isUsed[2] &&
			!histo.isUsed[3] && !histo.isUsed[4] {
			image.remove(i, numUsed)
			orig.remove(i, &numUsedOrig)
			symbols[i] = lwInvalidHistoSym
		} else {
			*imgH[i] = *histo
			symbols[i] = uint16(clusterID)
			clusterID++
		}
	}
}

// HistogramAnalyzeEntropyBin, histogram_enc.c:617, low_effort=0.
func histogramAnalyzeEntropyBin(set *lwHistoSet, binMap []uint16) {
	var c dominantCostRange
	c.init()
	for _, h := range set.h {
		if h == nil {
			continue
		}
		c.update(h)
	}
	for i, h := range set.h {
		if h == nil {
			continue
		}
		binMap[i] = uint16(getHistoBinIndex(h, &c))
	}
}

// HistogramCombineEntropyBin, histogram_enc.c:644, low_effort=0.
// curCombo may be swapped into set.h.
func histogramCombineEntropyBin(set *lwHistoSet, numUsed *int, clusters, clusterMappings []uint16, curCombo *lwHisto, binMap []uint16, numBins int, factor float64) {
	const maxCombineFailures = 32
	var binInfo [lwBinSize]struct {
		first    int16
		failures uint16
	}
	for idx := 0; idx < numBins; idx++ {
		binInfo[idx].first = -1
		binInfo[idx].failures = 0
	}
	for idx := 0; idx < *numUsed; idx++ {
		clusterMappings[idx] = uint16(idx)
	}
	hs := set.h[:set.max]
	for idx := 0; idx < len(set.h); idx++ {
		if hs[idx] == nil {
			continue
		}
		binID := int(binMap[idx])
		first := int(binInfo[binID].first)
		if first == -1 {
			binInfo[binID].first = int16(idx)
			continue
		}
		bitCost := hs[idx].bitCost
		bitCostThresh := float64(-bitCost * factor)
		currCostDiff := histogramAddEval(hs[first], hs[idx], curCombo, bitCostThresh)
		if currCostDiff < bitCostThresh {
			tryCombine := curCombo.trivialSymbol != lwNonTrivialSym ||
				(hs[idx].trivialSymbol == lwNonTrivialSym && hs[first].trivialSymbol == lwNonTrivialSym)
			if tryCombine || binInfo[binID].failures >= maxCombineFailures {
				curCombo, hs[first] = hs[first], curCombo
				set.remove(idx, numUsed)
				clusterMappings[clusters[idx]] = clusters[first]
			} else {
				binInfo[binID].failures++
			}
		}
	}
}

// MyRand, histogram_enc.c:721. uint64 product.
func myRand(seed *uint32) uint32 {
	*seed = uint32(uint64(*seed) * 48271 % 2147483647)
	return *seed
}

// HistoQueueInit, histogram_enc.c:744; reuses q's capacity.
func (q *lwHistoQueue) init(size int) {
	q.size = 0
	q.max = size
	if cap(q.q) < size {
		q.q = make([]lwHistoPair, size)
	}
	q.q = q.q[:size]
}

// HistoQueuePopPair, histogram_enc.c:763.
func (q *lwHistoQueue) popPair(p *lwHistoPair) {
	*p = q.q[q.size-1]
	q.size--
}

// HistoQueueUpdateHead, histogram_enc.c:773.
func (q *lwHistoQueue) updateHead(p *lwHistoPair) {
	if p.costDiff < q.q[0].costDiff {
		tmp := q.q[0]
		q.q[0] = *p
		*p = tmp
	}
}

// HistoQueueUpdatePair, histogram_enc.c:789.
func histoQueueUpdatePair(h1, h2 *lwHisto, threshold float64, p *lwHistoPair) {
	sumCost := h1.bitCost + h2.bitCost
	p.costCombo = 0
	getCombinedHistogramEntropy(h1, h2, sumCost+threshold, &p.costCombo)
	p.costDiff = p.costCombo - sumCost
}

// HistoQueuePush, histogram_enc.c:802.
func (q *lwHistoQueue) push(hs []*lwHisto, idx1, idx2 int, threshold float64) float64 {
	if q.size == q.max {
		return 0
	}
	if idx1 > idx2 {
		idx1, idx2 = idx2, idx1
	}
	pair := lwHistoPair{idx1: idx1, idx2: idx2}
	histoQueueUpdatePair(hs[idx1], hs[idx2], threshold, &pair)
	if pair.costDiff >= threshold {
		return 0
	}
	q.q[q.size] = pair
	q.size++
	q.updateHead(&q.q[q.size-1])
	return pair.costDiff
}

// histogramCombineGreedy uses e.queue.
// HistogramCombineGreedy, histogram_enc.c:837.
func (e *LW) histogramCombineGreedy(set *lwHistoSet, numUsed *int) {
	q := &e.queue
	size := len(set.h)
	hs := set.h[:set.max]
	q.init(size * size)
	for i := 0; i < size; i++ {
		if hs[i] == nil {
			continue
		}
		for j := i + 1; j < size; j++ {
			if hs[j] == nil {
				continue
			}
			q.push(hs, i, j, 0)
		}
	}

	for q.size > 0 {
		idx1 := q.q[0].idx1
		idx2 := q.q[0].idx2
		histogramAdd(hs[idx2], hs[idx1], hs[idx1])
		hs[idx1].bitCost = q.q[0].costCombo
		set.remove(idx2, numUsed)

		// Remove pairs intersecting the merged pair.
		for i := 0; i < q.size; {
			p := &q.q[i]
			if p.idx1 == idx1 || p.idx2 == idx1 || p.idx1 == idx2 || p.idx2 == idx2 {
				q.popPair(p)
			} else {
				q.updateHead(p)
				i++
			}
		}

		for i := 0; i < len(set.h); i++ {
			if i == idx1 || hs[i] == nil {
				continue
			}
			q.push(hs, idx1, i, 0)
		}
	}
}

// histogramCombineStochastic uses e.queue and e.mappings; PairComparison
// :904 is a binary search on mappings.
// HistogramCombineStochastic, histogram_enc.c:908.
func (e *LW) histogramCombineStochastic(set *lwHistoSet, numUsed *int, minClusterSize int) (doGreedy bool) {
	const kHistoQueueSize = 9
	seed := uint32(1)
	triesWithNoSuccess := 0
	outerIters := *numUsed
	numTriesNoSuccess := outerIters / 2
	hs := set.h[:set.max]
	q := &e.queue

	if *numUsed < minClusterSize {
		return true
	}

	if cap(e.mappings) < *numUsed {
		e.mappings = make([]int, *numUsed)
	}
	mappings := e.mappings[:*numUsed]
	q.init(kHistoQueueSize)
	j := 0
	for iter := 0; iter < len(set.h); iter++ {
		if hs[iter] == nil {
			continue
		}
		mappings[j] = iter
		j++
	}

	for iter := 0; ; iter++ {
		// C's for condition, ++tries only if the rest holds.
		if !(iter < outerIters && *numUsed >= minClusterSize) { //nolint:staticcheck // as libwebp
			break
		}
		triesWithNoSuccess++
		if !(triesWithNoSuccess < numTriesNoSuccess) { //nolint:staticcheck // as libwebp
			break
		}

		bestCost := 0.0
		if q.size != 0 {
			bestCost = q.q[0].costDiff
		}
		randRange := uint32((*numUsed - 1) * *numUsed)
		numTries := *numUsed / 2

		for j := 0; *numUsed >= 2 && j < numTries; j++ {
			tmp := myRand(&seed) % randRange
			idx1 := tmp / uint32(*numUsed-1)
			idx2 := tmp % uint32(*numUsed-1)
			if idx2 >= idx1 {
				idx2++
			}
			idx1 = uint32(mappings[idx1])
			idx2 = uint32(mappings[idx2])

			currCost := q.push(hs, int(idx1), int(idx2), bestCost)
			if currCost < 0 {
				bestCost = currCost
				if q.size == q.max {
					break
				}
			}
		}
		if q.size == 0 {
			continue
		}

		bestIdx1 := q.q[0].idx1
		bestIdx2 := q.q[0].idx2
		k, _ := slices.BinarySearch(mappings[:*numUsed], bestIdx2)
		copy(mappings[k:*numUsed-1], mappings[k+1:*numUsed])
		histogramAdd(hs[bestIdx2], hs[bestIdx1], hs[bestIdx1])
		hs[bestIdx1].bitCost = q.q[0].costCombo
		set.remove(bestIdx2, numUsed)

		for j := 0; j < q.size; {
			p := &q.q[j]
			isIdx1Best := p.idx1 == bestIdx1 || p.idx1 == bestIdx2
			isIdx2Best := p.idx2 == bestIdx1 || p.idx2 == bestIdx2
			doEval := false
			// The front pair may be a duplicate from a random pick.
			if isIdx1Best && isIdx2Best {
				q.popPair(p)
				continue
			}
			if isIdx1Best {
				p.idx1 = bestIdx1
				doEval = true
			} else if isIdx2Best {
				p.idx2 = bestIdx1
				doEval = true
			}
			if p.idx1 > p.idx2 {
				p.idx1, p.idx2 = p.idx2, p.idx1
			}
			if doEval {
				histoQueueUpdatePair(hs[p.idx1], hs[p.idx2], 0, p)
				if p.costDiff >= 0 {
					q.popPair(p)
					continue
				}
			}
			q.updateHead(p)
			j++
		}
		triesWithNoSuccess = 0
	}
	return *numUsed <= minClusterSize
}

// HistogramRemap, histogram_enc.c:1049.
func histogramRemap(in, out *lwHistoSet, symbols []uint16) {
	inH := in.h[:in.max]
	outH := out.h[:out.max]
	inSize := out.max
	outSize := len(out.h)
	if outSize > 1 {
		for i := 0; i < inSize; i++ {
			bestOut := 0
			bestBits := lwMaxCost
			if inH[i] == nil {
				// Previous id helps later LZ77.
				symbols[i] = symbols[i-1]
				continue
			}
			for k := 0; k < outSize; k++ {
				curBits := histogramAddThresh(outH[k], inH[i], bestBits)
				if k == 0 || curBits < bestBits {
					bestBits = curBits
					bestOut = k
				}
			}
			symbols[i] = uint16(bestOut)
		}
	} else {
		for i := 0; i < inSize; i++ {
			symbols[i] = 0
		}
	}

	// Recompute each out from in and symbols.
	out.clear()
	out.h = out.h[:outSize]
	for i := 0; i < inSize; i++ {
		if inH[i] == nil {
			continue
		}
		idx := symbols[i]
		histogramAdd(inH[i], out.h[idx], out.h[idx])
	}
}

// GetCombineCostFactor, histogram_enc.c:1096.
func getCombineCostFactor(histoSize, quality int) float64 {
	f := 0.16
	if quality < 90 {
		if histoSize > 256 {
			f /= 2
		}
		if histoSize > 512 {
			f /= 2
		}
		if histoSize > 1024 {
			f /= 2
		}
		if quality <= 50 {
			f /= 2
		}
	}
	return f
}

// The C assert-only loop at :1151 is dropped.
// OptimizeHistogramSymbols, histogram_enc.c:1110.
func optimizeHistogramSymbols(set *lwHistoSet, clusterMappings []uint16, numClusters int, tmp, symbols []uint16) {
	doContinue := true
	// Assign the lowest cluster to each pixel.
	for doContinue {
		doContinue = false
		for i := 0; i < numClusters; i++ {
			k := clusterMappings[i]
			for k != clusterMappings[k] {
				clusterMappings[k] = clusterMappings[clusterMappings[k]]
				k = clusterMappings[k]
			}
			if k != clusterMappings[i] {
				doContinue = true
				clusterMappings[i] = k
			}
		}
	}
	clusterMax := uint16(0)
	clear(tmp[:set.max])
	for i := 0; i < set.max; i++ {
		if symbols[i] == lwInvalidHistoSym {
			continue
		}
		cluster := clusterMappings[symbols[i]]
		if cluster > 0 && tmp[cluster] == 0 {
			clusterMax++
			tmp[cluster] = clusterMax
		}
		symbols[i] = tmp[cluster]
	}
}

// RemoveEmptyHistograms, histogram_enc.c:1161.
func removeEmptyHistograms(set *lwHistoSet) {
	size := 0
	for i := 0; i < len(set.h); i++ {
		if set.h[i] == nil {
			continue
		}
		set.h[size] = set.h[i]
		size++
	}
	set.h = set.h[:size]
}

// getHistoImageSymbols uses e.histoOrig and e.mapTmp.
// VP8LGetHistoImageSymbols, histogram_enc.c:1171, low_effort=0, cache 0.
func (e *LW) getHistoImageSymbols(xsize, ysize int, refs lwRefs, quality, histoBits int, image *lwHistoSet, tmp *lwHisto, symbols []uint16) {
	histoXsize, histoYsize := 1, 1
	if histoBits != 0 {
		histoXsize = subSampleSize(xsize, histoBits)
		histoYsize = subSampleSize(ysize, histoBits)
	}
	rawSize := histoXsize * histoYsize
	e.histoOrig.alloc(rawSize)
	const entropyCombineNumBins = lwBinSize
	if cap(e.mapTmp) < 2*rawSize {
		e.mapTmp = make([]uint16, 2*rawSize)
	}
	mapTmp := e.mapTmp[:2*rawSize]
	clusterMappings := mapTmp[rawSize:]
	numUsed := rawSize

	histogramBuild(xsize, histoBits, refs, &e.histoOrig)
	histogramCopyAndAnalyze(&e.histoOrig, image, &numUsed, symbols)

	entropyCombine := numUsed > entropyCombineNumBins*2 && quality < 100
	if entropyCombine {
		binMap := mapTmp
		factor := getCombineCostFactor(rawSize, quality)
		numClusters := numUsed
		histogramAnalyzeEntropyBin(image, binMap)
		histogramCombineEntropyBin(image, &numUsed, symbols, clusterMappings, tmp, binMap, entropyCombineNumBins, factor)
		optimizeHistogramSymbols(image, clusterMappings, numClusters, mapTmp, symbols)
	}

	// float32 cubic ramp, as C's quality / 100.f.
	x := float32(quality) / float32(100)
	thresholdSize := int(float32(1) + float32(float32(x*x*x)*float32(lwMaxHistoGreedy-1)))
	if e.histogramCombineStochastic(image, &numUsed, thresholdSize) {
		removeEmptyHistograms(image)
		e.histogramCombineGreedy(image, &numUsed)
	}

	removeEmptyHistograms(image)
	histogramRemap(&e.histoOrig, image, symbols)
}
