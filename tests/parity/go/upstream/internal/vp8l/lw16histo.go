// Port of libwebp 1.6.0 enc/histogram_enc.c. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "math"

// HistogramIndex, histogram_enc.c:39.
const (
	v16Literal = iota
	v16Red
	v16Blue
	v16Alpha
	v16Distance
)

// lw16CacheIdx is kCacheIdx, backward_references_enc.h.
const lw16CacheIdx = 1

// v16NumCodes is VP8LHistogramNumCodes, histogram_enc.h.
func v16NumCodes(cacheBits int) int {
	if cacheBits > 0 {
		return lwNumCodes + 1<<cacheBits
	}
	return lwNumCodes
}

// lw16Histo is VP8LHistogram, histogram_enc.h:29. literal has
// v16NumCodes(paletteCodeBits) entries.
type lw16Histo struct {
	literal         []uint32
	red             [lwNumLiteral]uint32
	blue            [lwNumLiteral]uint32
	alpha           [lwNumLiteral]uint32
	distance        [lwNumDistance]uint32
	paletteCodeBits int
	trivialSymbol   [5]uint16
	bitCost         uint64
	costs           [5]uint64
	isUsed          [5]bool
	binID           uint16
}

// lw16HistoSet is VP8LHistogramSet: h[:size] in use, len(h) is max.
type lw16HistoSet struct {
	h    []*lw16Histo
	size int
	pool []lw16Histo
	lits []uint32
}

// setBits sizes literal for cacheBits; cap must allow it.
func (h *lw16Histo) setBits(cacheBits int) {
	h.paletteCodeBits = cacheBits
	h.literal = h.literal[:v16NumCodes(cacheBits)]
}

// alloc gives h room for cacheBits. VP8LAllocateHistogram, histogram_enc.c:122.
func (h *lw16Histo) alloc(cacheBits int) {
	if n := v16NumCodes(cacheBits); cap(h.literal) < n {
		h.literal = make([]uint32, n)
	}
	h.init(cacheBits, false)
}

// HistogramStatsClear, histogram_enc.c:55.
func (h *lw16Histo) statsClear() {
	for i := range h.trivialSymbol {
		h.trivialSymbol[i] = v16NonTrivialSym
		h.isUsed[i] = true
	}
	h.bitCost = 0
	h.costs = [5]uint64{}
}

// HistogramClear, histogram_enc.c:66.
func (h *lw16Histo) clear() {
	clear(h.literal)
	h.red = [lwNumLiteral]uint32{}
	h.blue = [lwNumLiteral]uint32{}
	h.alpha = [lwNumLiteral]uint32{}
	h.distance = [lwNumDistance]uint32{}
	h.binID = 0
	h.statsClear()
}

// zero is the memset of VP8LHistogramSetClear, histogram_enc.c:176.
func (h *lw16Histo) zero() {
	lit, bits := h.literal, h.paletteCodeBits
	clear(lit)
	*h = lw16Histo{literal: lit, paletteCodeBits: bits}
}

// HistogramCopy, histogram_enc.c:83.
func (h *lw16Histo) copyFrom(src *lw16Histo) {
	lit := h.literal
	*h = *src
	h.literal = lit
	copy(lit, src.literal)
}

// VP8LHistogramInit, histogram_enc.c:112.
func (h *lw16Histo) init(cacheBits int, arrays bool) {
	h.setBits(cacheBits)
	if arrays {
		h.clear()
	} else {
		h.statsClear()
	}
}

// VP8LHistogramCreate, histogram_enc.c:101.
func (h *lw16Histo) create(refs lwRefs, cacheBits int) {
	if cacheBits >= 0 {
		h.setBits(cacheBits)
	}
	h.clear()
	h.storeRefs(refs, 0)
}

// storeRefs maps distances to plane codes when planeXsize > 0.
// VP8LHistogramStoreRefs, histogram_enc.c:231.
func (h *lw16Histo) storeRefs(refs lwRefs, planeXsize int) {
	for i := range refs {
		h.addSinglePixOrCopy(&refs[i], planeXsize)
	}
}

// HistogramAddSinglePixOrCopy, histogram_enc.c:203.
func (h *lw16Histo) addSinglePixOrCopy(v *pixOrCopy, planeXsize int) {
	switch v.mode {
	case lwLiteral:
		p := v.argbOrDist
		h.alpha[p>>24]++
		h.red[(p>>16)&0xff]++
		h.literal[(p>>8)&0xff]++
		h.blue[p&0xff]++
	case lw16CacheIdx:
		h.literal[lwNumCodes+int(v.argbOrDist)]++
	default:
		code, _, _ := prefixBits(int(v.len))
		h.literal[lwNumLiteral+code]++
		dist := int(v.argbOrDist)
		if planeXsize > 0 {
			dist = distanceToPlaneCode(planeXsize, dist)
		}
		code, _, _ = prefixBits(dist)
		h.distance[code]++
	}
}

// pop is GetPopulationInfo, histogram_enc.c:337.
func (h *lw16Histo) pop(i int) []uint32 {
	switch i {
	case v16Literal:
		return h.literal
	case v16Red:
		return h.red[:]
	case v16Blue:
		return h.blue[:]
	case v16Alpha:
		return h.alpha[:]
	}
	return h.distance[:]
}

// alloc sizes s to n histograms for cacheBits, stats cleared.
// VP8LAllocateHistogramSet, histogram_enc.c:157.
func (s *lw16HistoSet) alloc(n, cacheBits int) {
	nc := v16NumCodes(cacheBits)
	if cap(s.pool) < n {
		s.pool = make([]lw16Histo, n)
	}
	s.pool = s.pool[:n]
	if cap(s.lits) < n*nc {
		s.lits = make([]uint32, n*nc)
	}
	if cap(s.h) < n {
		s.h = make([]*lw16Histo, n)
	}
	s.h = s.h[:n]
	s.size = n
	for i := range s.pool {
		p := &s.pool[i]
		p.literal = s.lits[i*nc : (i+1)*nc : (i+1)*nc]
		p.init(cacheBits, false)
		s.h[i] = p
	}
}

// setClear zeroes every histogram and resets the pointers.
// VP8LHistogramSetClear, histogram_enc.c:176.
func (s *lw16HistoSet) setClear() {
	for i := range s.pool {
		s.pool[i].zero()
		s.h[i] = &s.pool[i]
	}
	s.size = len(s.h)
}

// HistogramSetRemoveHistogram, histogram_enc.c:195.
func (s *lw16HistoSet) remove(i int) {
	s.h[i] = s.h[s.size-1]
	s.size--
}

// BitsEntropyRefine, histogram_enc.c:246.
func v16BitsEntropyRefine(b *v16BitEntropy) uint64 {
	var mix uint64
	if b.nonzeros < 5 {
		if b.nonzeros <= 1 {
			return 0
		}
		if b.nonzeros == 2 {
			return uint64(v16DivRound(int64(99*(uint64(b.sum)<<v16LogBits)+b.entropy), 100))
		}
		if b.nonzeros == 3 {
			mix = 950
		} else {
			mix = 700
		}
	} else {
		mix = 627
	}
	minLimit := uint64(2*b.sum-b.maxVal) << v16LogBits // uint32 wrap as in C
	minLimit = uint64(v16DivRound(int64(mix*minLimit+(1000-mix)*b.entropy), 1000))
	if b.entropy < minLimit {
		return minLimit
	}
	return b.entropy
}

// VP8LBitsEntropy, histogram_enc.c:282.
func v16BitsEntropy(a []uint32) uint64 {
	var b v16BitEntropy
	v16BitsEntropyUnrefined(a, &b)
	return v16BitsEntropyRefine(&b)
}

// InitialHuffmanCost, histogram_enc.c:289.
func v16InitialHuffmanCost() uint64 {
	return uint64(lwCodeLenCodes*3)<<v16LogBits - uint64(v16DivRound(91<<v16LogBits, 10))
}

// FinalHuffmanCost, histogram_enc.c:299; uint32 extra as in C.
func v16FinalHuffmanCost(s *streaks) uint64 {
	retval := v16InitialHuffmanCost()
	extra := uint32(s.counts[0]*1600 + 240*s.streaks[0][1])
	extra += uint32(s.counts[1]*2640 + 720*s.streaks[1][1])
	extra += uint32(1840 * s.streaks[0][0])
	extra += uint32(3360 * s.streaks[1][0])
	return retval + uint64(extra)<<(v16LogBits-10)
}

// v16PopulationCost: trivialSym and isUsed may be nil.
// PopulationCost, histogram_enc.c:319.
func v16PopulationCost(pop []uint32, trivialSym *uint16, isUsed *bool) uint64 {
	var b v16BitEntropy
	var s streaks
	v16EntropyUnrefined(pop, &b, &s)
	if trivialSym != nil {
		if b.nonzeros == 1 {
			*trivialSym = uint16(b.nonzeroCode)
		} else {
			*trivialSym = v16NonTrivialSym
		}
	}
	if isUsed != nil {
		*isUsed = s.streaks[1][0] != 0 || s.streaks[1][1] != 0
	}
	return v16BitsEntropyRefine(&b) + v16FinalHuffmanCost(&s)
}

// GetCombinedEntropy, histogram_enc.c:369.
func v16GetCombinedEntropy(h1, h2 *lw16Histo, i int) uint64 {
	used1, used2 := h1.isUsed[i], h2.isUsed[i]
	trivial := h1.trivialSymbol[i] != v16NonTrivialSym && h1.trivialSymbol[i] == h2.trivialSymbol[i]
	if trivial || !used1 || !used2 {
		if used1 {
			return h1.costs[i]
		}
		return h2.costs[i]
	}
	var b v16BitEntropy
	var s streaks
	v16CombinedEntropyUnrefined(h1.pop(i), h2.pop(i), &b, &s)
	return v16BitsEntropyRefine(&b) + v16FinalHuffmanCost(&s)
}

// VP8LHistogramEstimateBits, histogram_enc.c:395.
func (h *lw16Histo) estimateBits() uint64 {
	var cost uint64
	for i := 0; i < 5; i++ {
		cost += v16PopulationCost(h.pop(i), nil, nil)
	}
	extra := v16ExtraCost(h.literal[lwNumLiteral:lwNumCodes]) + v16ExtraCost(h.distance[:])
	return cost + uint64(extra)<<v16LogBits
}

// SaturateAdd, histogram_enc.c:416.
func v16SaturateAdd(a uint64, b *int64) {
	if *b < 0 || int64(a) <= v16Int64Max-*b {
		*b += int64(a)
	} else {
		*b = v16Int64Max
	}
}

// GetCombinedHistogramEntropy, histogram_enc.c:426.
func v16GetCombinedHistogramEntropy(a, b *lw16Histo, thresholdIn int64, cost *uint64, costs *[5]uint64) bool {
	threshold := uint64(thresholdIn)
	if thresholdIn <= 0 {
		return false
	}
	*cost = 0
	for i := 0; i < 5; i++ {
		costs[i] = v16GetCombinedEntropy(a, b, i)
		*cost += costs[i]
		if *cost >= threshold {
			return false
		}
	}
	return true
}

// HistogramAdd, histogram_enc.c:446; h2 may equal out.
func v16HistogramAdd(h1, h2, out *lw16Histo) {
	for i := 0; i < 5; i++ {
		p1, p2, po := h1.pop(i), h2.pop(i), out.pop(i)
		if h2 == out {
			if h1.isUsed[i] {
				if out.isUsed[i] {
					for k := range po {
						po[k] += p1[k]
					}
				} else {
					copy(po, p1)
				}
			}
		} else {
			if h1.isUsed[i] {
				if h2.isUsed[i] {
					for k := range po {
						po[k] = p1[k] + p2[k]
					}
				} else {
					copy(po, p1)
				}
			} else if h2.isUsed[i] {
				copy(po, p2)
			} else {
				clear(po)
			}
		}
	}
	for i := 0; i < 5; i++ {
		if h1.trivialSymbol[i] == h2.trivialSymbol[i] {
			out.trivialSymbol[i] = h1.trivialSymbol[i]
		} else {
			out.trivialSymbol[i] = v16NonTrivialSym
		}
		out.isUsed[i] = h1.isUsed[i] || h2.isUsed[i]
	}
}

// UpdateHistogramCost, histogram_enc.c:491.
func (h *lw16Histo) updateCost(bitCost uint64, costs *[5]uint64) {
	h.bitCost = bitCost
	h.costs = *costs
}

// HistogramAddEval, histogram_enc.c:508.
func v16HistogramAddEval(a, b, out *lw16Histo, threshold int64) bool {
	sumCost := a.bitCost + b.bitCost
	var bitCost uint64
	var costs [5]uint64
	v16SaturateAdd(sumCost, &threshold)
	if !v16GetCombinedHistogramEntropy(a, b, threshold, &bitCost, &costs) {
		return false
	}
	v16HistogramAdd(a, b, out)
	out.updateCost(bitCost, &costs)
	return true
}

// HistogramAddThresh, histogram_enc.c:529.
func v16HistogramAddThresh(a, b *lw16Histo, threshold int64, costOut *int64) bool {
	var cost uint64
	var costs [5]uint64
	v16SaturateAdd(a.bitCost, &threshold)
	if !v16GetCombinedHistogramEntropy(a, b, threshold, &cost, &costs) {
		return false
	}
	*costOut = int64(cost) - int64(a.bitCost)
	return true
}

// v16CostRange is DominantCostRange, histogram_enc.c:548.
type v16CostRange struct {
	literalMax, literalMin uint64
	redMax, redMin         uint64
	blueMax, blueMin       uint64
}

// DominantCostRangeInit, histogram_enc.c:557.
func (c *v16CostRange) init() {
	*c = v16CostRange{literalMin: math.MaxUint64, redMin: math.MaxUint64, blueMin: math.MaxUint64}
}

// UpdateDominantCostRange, histogram_enc.c:566.
func (c *v16CostRange) update(h *lw16Histo) {
	c.literalMax = max(c.literalMax, h.costs[v16Literal])
	c.literalMin = min(c.literalMin, h.costs[v16Literal])
	c.redMax = max(c.redMax, h.costs[v16Red])
	c.redMin = min(c.redMin, h.costs[v16Red])
	c.blueMax = max(c.blueMax, h.costs[v16Blue])
	c.blueMin = min(c.blueMin, h.costs[v16Blue])
}

// ComputeHistogramCost, histogram_enc.c:576.
func (h *lw16Histo) computeCost() {
	for i := 0; i < 5; i++ {
		h.costs[i] = v16PopulationCost(h.pop(i), &h.trivialSymbol[i], &h.isUsed[i])
	}
	h.bitCost = h.costs[v16Literal] + h.costs[v16Red] + h.costs[v16Blue] +
		h.costs[v16Alpha] + h.costs[v16Distance]
}

// GetBinIdForEntropy, histogram_enc.c:591; double as in C.
func v16GetBinIdForEntropy(lo, hi, val uint64) int {
	rng := hi - lo
	if rng > 0 {
		delta := val - lo
		return int(float64(lwBinIDFactor*float64(delta)) / float64(rng))
	}
	return 0
}

// GetHistoBinIndex, histogram_enc.c:601, low_effort=0.
func v16GetHistoBinIndex(h *lw16Histo, c *v16CostRange) int {
	binID := v16GetBinIdForEntropy(c.literalMin, c.literalMax, h.costs[v16Literal])
	binID = binID*lwNumPartitions + v16GetBinIdForEntropy(c.redMin, c.redMax, h.costs[v16Red])
	binID = binID*lwNumPartitions + v16GetBinIdForEntropy(c.blueMin, c.blueMax, h.costs[v16Blue])
	return binID
}

// HistogramBuild, histogram_enc.c:617.
func v16HistogramBuild(xsize, histoBits int, refs lwRefs, set *lw16HistoSet) {
	x, y := 0, 0
	histoXsize := subSampleSize(xsize, histoBits)
	set.setClear()
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

// HistogramCopyAndAnalyze, histogram_enc.c:640. Empty orig entries
// become nil.
func v16HistogramCopyAndAnalyze(orig, image *lw16HistoSet) {
	image.size = 0
	for i := range orig.h {
		histo := orig.h[i]
		histo.computeCost()
		if !histo.isUsed[0] && !histo.isUsed[1] && !histo.isUsed[2] &&
			!histo.isUsed[3] && !histo.isUsed[4] {
			orig.h[i] = nil
		} else {
			image.h[image.size].copyFrom(histo)
			image.size++
		}
	}
}

// HistogramAnalyzeEntropyBin, histogram_enc.c:669, low_effort=0.
func v16HistogramAnalyzeEntropyBin(set *lw16HistoSet) {
	var c v16CostRange
	c.init()
	for _, h := range set.h[:set.size] {
		c.update(h)
	}
	for _, h := range set.h[:set.size] {
		h.binID = uint16(v16GetHistoBinIndex(h, &c))
	}
}

// HistogramCombineEntropyBin, histogram_enc.c:693, low_effort=0.
// curCombo may be swapped into set.h.
func v16HistogramCombineEntropyBin(set *lw16HistoSet, curCombo *lw16Histo, numBins int, factor int32) {
	const maxCombineFailures = 32
	const nt = v16NonTrivialSym
	var binInfo [lwBinSize]struct {
		first    int16
		failures uint16
	}
	for idx := 0; idx < numBins; idx++ {
		binInfo[idx].first = -1
		binInfo[idx].failures = 0
	}
	hs := set.h
	for idx := 0; idx < set.size; {
		binID := int(hs[idx].binID)
		first := int(binInfo[binID].first)
		if first == -1 {
			binInfo[binID].first = int16(idx)
			idx++
			continue
		}
		bitCostThresh := -v16DivRound(int64(hs[idx].bitCost)*int64(factor), 100)
		if !v16HistogramAddEval(hs[first], hs[idx], curCombo, bitCostThresh) {
			idx++
			continue
		}
		c := &curCombo.trivialSymbol
		tryCombine := c[v16Red] != nt && c[v16Blue] != nt && c[v16Alpha] != nt
		if !tryCombine {
			a, b := &hs[idx].trivialSymbol, &hs[first].trivialSymbol
			tryCombine = (a[v16Red] == nt || a[v16Blue] == nt || a[v16Alpha] == nt) &&
				(b[v16Red] == nt || b[v16Blue] == nt || b[v16Alpha] == nt)
		}
		if tryCombine || binInfo[binID].failures >= maxCombineFailures {
			curCombo, hs[first] = hs[first], curCombo
			set.remove(idx)
		} else {
			binInfo[binID].failures++
			idx++
		}
	}
}

// lw16Pair is HistogramPair, histogram_enc.c:781.
type lw16Pair struct {
	idx1, idx2 int
	costDiff   int64
	costCombo  uint64
	costs      [5]uint64
}

// lw16Queue is HistoQueue, histogram_enc.c:789.
type lw16Queue struct {
	q    []lw16Pair
	size int
	max  int
}

// HistoQueueInit, histogram_enc.c:795; reuses q's capacity.
func (q *lw16Queue) init(n int) {
	q.size = 0
	q.max = n
	if cap(q.q) < n+1 {
		q.q = make([]lw16Pair, n+1)
	}
	q.q = q.q[:n+1]
}

// HistoQueuePopPair, histogram_enc.c:814.
func (q *lw16Queue) popPair(p *lw16Pair) {
	*p = q.q[q.size-1]
	q.size--
}

// HistoQueueUpdateHead, histogram_enc.c:824.
func (q *lw16Queue) updateHead(p *lw16Pair) {
	if p.costDiff < q.q[0].costDiff {
		tmp := q.q[0]
		q.q[0] = *p
		*p = tmp
	}
}

// HistoQueueFixPair, histogram_enc.c:839.
func v16FixPair(badID, goodID int, p *lw16Pair) {
	if p.idx1 == badID {
		p.idx1 = goodID
	}
	if p.idx2 == badID {
		p.idx2 = goodID
	}
	if p.idx1 > p.idx2 {
		p.idx1, p.idx2 = p.idx2, p.idx1
	}
}

// HistoQueueUpdatePair, histogram_enc.c:854.
func v16UpdatePair(h1, h2 *lw16Histo, threshold int64, p *lw16Pair) bool {
	sumCost := int64(h1.bitCost + h2.bitCost)
	v16SaturateAdd(uint64(sumCost), &threshold)
	if !v16GetCombinedHistogramEntropy(h1, h2, threshold, &p.costCombo, &p.costs) {
		return false
	}
	p.costDiff = int64(p.costCombo) - sumCost
	return true
}

// HistoQueuePush, histogram_enc.c:871.
func (q *lw16Queue) push(hs []*lw16Histo, idx1, idx2 int, threshold int64) int64 {
	if q.size == q.max {
		return 0
	}
	if idx1 > idx2 {
		idx1, idx2 = idx2, idx1
	}
	pair := lw16Pair{idx1: idx1, idx2: idx2}
	if !v16UpdatePair(hs[idx1], hs[idx2], threshold, &pair) {
		return 0
	}
	q.q[q.size] = pair
	q.size++
	q.updateHead(&q.q[q.size-1])
	return pair.costDiff
}

// HistogramCombineGreedy, histogram_enc.c:904.
func (s *lw16State) histogramCombineGreedy(set *lw16HistoSet) {
	q := &s.queue
	size := set.size
	hs := set.h
	q.init(size * size)
	for i := 0; i < size; i++ {
		for j := i + 1; j < size; j++ {
			q.push(hs, i, j, 0)
		}
	}
	for q.size > 0 {
		idx1 := q.q[0].idx1
		idx2 := q.q[0].idx2
		v16HistogramAdd(hs[idx2], hs[idx1], hs[idx1])
		hs[idx1].updateCost(q.q[0].costCombo, &q.q[0].costs)
		set.remove(idx2)

		// Remove pairs intersecting the merged pair.
		for i := 0; i < q.size; {
			p := &q.q[i]
			if p.idx1 == idx1 || p.idx2 == idx1 || p.idx1 == idx2 || p.idx2 == idx2 {
				q.popPair(p)
			} else {
				v16FixPair(set.size, idx2, p)
				q.updateHead(p)
				i++
			}
		}

		for i := 0; i < set.size; i++ {
			if i == idx1 {
				continue
			}
			q.push(hs, idx1, i, 0)
		}
	}
}

// HistogramCombineStochastic, histogram_enc.c:970.
func (s *lw16State) histogramCombineStochastic(set *lw16HistoSet, minClusterSize int) (doGreedy bool) {
	const kHistoQueueSize = 9
	seed := uint32(1)
	triesWithNoSuccess := 0
	outerIters := set.size
	numTriesNoSuccess := outerIters / 2
	hs := set.h
	q := &s.queue

	if set.size < minClusterSize {
		return true
	}
	q.init(kHistoQueueSize)

	for iter := 0; ; iter++ {
		// C's for condition, ++tries only if the rest holds.
		if !(iter < outerIters && set.size >= minClusterSize) { //nolint:staticcheck // as libwebp
			break
		}
		triesWithNoSuccess++
		if !(triesWithNoSuccess < numTriesNoSuccess) { //nolint:staticcheck // as libwebp
			break
		}

		bestCost := int64(0)
		if q.size != 0 {
			bestCost = q.q[0].costDiff
		}
		randRange := uint32((set.size - 1) * set.size)
		numTries := set.size / 2

		for j := 0; set.size >= 2 && j < numTries; j++ {
			tmp := myRand(&seed) % randRange
			idx1 := tmp / uint32(set.size-1)
			idx2 := tmp % uint32(set.size-1)
			if idx2 >= idx1 {
				idx2++
			}
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
		v16HistogramAdd(hs[bestIdx2], hs[bestIdx1], hs[bestIdx1])
		hs[bestIdx1].updateCost(q.q[0].costCombo, &q.q[0].costs)
		set.remove(bestIdx2)

		for j := 0; j < q.size; {
			p := &q.q[j]
			isIdx1Best := p.idx1 == bestIdx1 || p.idx1 == bestIdx2
			isIdx2Best := p.idx2 == bestIdx1 || p.idx2 == bestIdx2
			// The front pair may be a duplicate from a random pick.
			if isIdx1Best && isIdx2Best {
				q.popPair(p)
				continue
			}
			if isIdx1Best || isIdx2Best {
				v16FixPair(bestIdx2, bestIdx1, p)
				if !v16UpdatePair(hs[p.idx1], hs[p.idx2], 0, p) {
					q.popPair(p)
					continue
				}
			}
			v16FixPair(set.size, bestIdx2, p)
			q.updateHead(p)
			j++
		}
		triesWithNoSuccess = 0
	}
	return set.size <= minClusterSize
}

// HistogramRemap, histogram_enc.c:1078.
func v16HistogramRemap(in, out *lw16HistoSet, symbols []uint32) {
	inH := in.h
	inSize := len(out.h)
	outSize := out.size
	if outSize > 1 {
		for i := 0; i < inSize; i++ {
			if inH[i] == nil {
				// Previous id helps later LZ77.
				symbols[i] = symbols[i-1]
				continue
			}
			bestOut := 0
			bestBits := int64(v16Int64Max)
			for k := 0; k < outSize; k++ {
				var curBits int64
				if v16HistogramAddThresh(out.h[k], inH[i], bestBits, &curBits) {
					bestBits = curBits
					bestOut = k
				}
			}
			symbols[i] = uint32(bestOut)
		}
	} else {
		clear(symbols[:inSize])
	}

	// Recompute each out from in and symbols.
	out.setClear()
	out.size = outSize
	for i := 0; i < inSize; i++ {
		if inH[i] == nil {
			continue
		}
		idx := symbols[i]
		v16HistogramAdd(inH[i], out.h[idx], out.h[idx])
	}
}

// GetCombineCostFactor, histogram_enc.c:1125.
func v16GetCombineCostFactor(histoSize, quality int) int32 {
	f := int32(16)
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

// getHistoImageSymbols uses s.orig as orig_histo.
// VP8LGetHistoImageSymbols, histogram_enc.c:1136, low_effort=0.
func (s *lw16State) getHistoImageSymbols(xsize, ysize int, refs lwRefs, quality, histoBits, cacheBits int, image *lw16HistoSet, tmp *lw16Histo, symbols []uint32) {
	rawSize := subSampleSize(xsize, histoBits) * subSampleSize(ysize, histoBits)
	orig := &s.orig
	orig.alloc(rawSize, cacheBits)

	v16HistogramBuild(xsize, histoBits, refs, orig)
	v16HistogramCopyAndAnalyze(orig, image)
	if image.size > lwBinSize*2 && quality < 100 {
		factor := v16GetCombineCostFactor(rawSize, quality)
		v16HistogramAnalyzeEntropyBin(image)
		v16HistogramCombineEntropyBin(image, tmp, lwBinSize, factor)
	}

	// Cubic ramp between 1 and lwMaxHistoGreedy.
	thresholdSize := int(1 + v16DivRound(int64(quality*quality*quality*(lwMaxHistoGreedy-1)), 100*100*100))
	if s.histogramCombineStochastic(image, thresholdSize) {
		s.histogramCombineGreedy(image)
	}
	v16HistogramRemap(orig, image, symbols)
}
