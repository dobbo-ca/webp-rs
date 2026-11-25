// Port of libwebp 1.6.0 enc/backward_references_cost_enc.c, integer costs and color cache. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "slices"

// lw16CostModel is CostModel, backward_references_cost_enc.c:35.
type lw16CostModel struct {
	alpha    [256]uint32
	red      [256]uint32
	blue     [256]uint32
	distance [lwNumDistance]uint32
	literal  []uint32 // v16NumCodes(cache bits)
}

// lw16CostInterval is CostInterval, backward_references_cost_enc.c:161.
type lw16CostInterval struct {
	cost       int64
	start, end int
	index      int
	prev, next *lw16CostInterval
}

// lw16CostCacheInterval is CostCacheInterval; end is exclusive.
type lw16CostCacheInterval struct {
	cost       int64
	start, end int
}

// lw16CostManager is CostManager. ivs is the interval store; the live
// list never exceeds lwCostIntervalMax.
type lw16CostManager struct {
	head           *lw16CostInterval
	count          int
	cacheIntervals []lw16CostCacheInterval
	costCache      [lwMaxLength]int64 // GetLengthCost(k)
	costs          []int64
	distArray      []uint16
	ivs            [lwCostIntervalMax]lw16CostInterval
	free           *lw16CostInterval
}

// v16ColorCache is VP8LColorCache, utils/color_cache_utils.h.
type v16ColorCache struct {
	colors []uint32
	shift  int
}

// VP8LColorCacheInit, color_cache_utils.c:25; colors start at 0.
func (c *v16ColorCache) init(hashBits int) {
	n := 1 << hashBits
	if cap(c.colors) < n {
		c.colors = make([]uint32, n)
	}
	c.colors = c.colors[:n]
	clear(c.colors)
	c.shift = 32 - hashBits
}

// VP8LHashPix, color_cache_utils.h:38.
func v16HashPix(argb uint32, shift int) int {
	return int((argb * 0x1e35a7bd) >> shift)
}

// VP8LColorCacheContains, color_cache_utils.h:66.
func (c *v16ColorCache) contains(argb uint32) int {
	k := v16HashPix(argb, c.shift)
	if c.colors[k] == argb {
		return k
	}
	return -1
}

// VP8LColorCacheInsert, color_cache_utils.h:54.
func (c *v16ColorCache) insert(argb uint32) {
	c.colors[v16HashPix(argb, c.shift)] = argb
}

// ConvertPopulationCountTableToBitEstimates, backward_references_cost_enc.c:43.
func v16ConvertPopulation(counts, out []uint32) {
	var sum uint32
	nonzeros := 0
	for _, c := range counts {
		sum += c
		if c > 0 {
			nonzeros++
		}
	}
	if nonzeros <= 1 {
		clear(out[:len(counts)])
		return
	}
	logsum := v16FastLog2(sum)
	for i, c := range counts {
		out[i] = logsum - v16FastLog2(c)
	}
}

// costModelBuild uses s.costHisto with plane-code distances.
// CostModelBuild, backward_references_cost_enc.c:64.
func (s *lw16State) costModelBuild(m *lw16CostModel, xsize, cacheBits int, refs lwRefs) {
	h := &s.costHisto
	h.alloc(cacheBits)
	h.init(cacheBits, true)
	h.storeRefs(refs, xsize)
	nc := v16NumCodes(cacheBits)
	if cap(m.literal) < nc {
		m.literal = make([]uint32, nc)
	}
	m.literal = m.literal[:nc]
	v16ConvertPopulation(h.literal, m.literal)
	v16ConvertPopulation(h.red[:], m.red[:])
	v16ConvertPopulation(h.blue[:], m.blue[:])
	v16ConvertPopulation(h.alpha[:], m.alpha[:])
	v16ConvertPopulation(h.distance[:], m.distance[:])
}

// GetLiteralCost, backward_references_cost_enc.c:93.
func (m *lw16CostModel) literalCost(v uint32) int64 {
	return int64(m.alpha[v>>24]) + int64(m.red[(v>>16)&0xff]) +
		int64(m.literal[(v>>8)&0xff]) + int64(m.blue[v&0xff])
}

// GetCacheCost, backward_references_cost_enc.c:99.
func (m *lw16CostModel) cacheCost(idx int) int64 {
	return int64(m.literal[lwNumCodes+idx])
}

// GetLengthCost, backward_references_cost_enc.c:105.
func (m *lw16CostModel) lengthCost(n uint32) int64 {
	code, extra := prefixCodeBits(int(n))
	return int64(m.literal[lwNumLiteral+code]) + int64(extra)<<v16LogBits
}

// GetDistanceCost, backward_references_cost_enc.c:113.
func (m *lw16CostModel) distanceCost(d uint32) int64 {
	code, extra := prefixCodeBits(int(d))
	return int64(m.distance[code]) + int64(extra)<<v16LogBits
}

// v16AddSingleLiteralWithCostModel: cc is nil without a color cache.
// AddSingleLiteralWithCostModel, backward_references_cost_enc.c:121.
func v16AddSingleLiteralWithCostModel(argb []uint32, cc *v16ColorCache, m *lw16CostModel, idx int, prevCost int64, cost []int64, distArray []uint16) {
	costVal := prevCost
	color := argb[idx]
	ix := -1
	if cc != nil {
		ix = cc.contains(color)
	}
	if ix >= 0 {
		costVal += v16DivRound(m.cacheCost(ix)*68, 100)
	} else {
		if cc != nil {
			cc.insert(color)
		}
		costVal += v16DivRound(m.literalCost(color)*82, 100)
	}
	if cost[idx] > costVal {
		cost[idx] = costVal
		distArray[idx] = 1
	}
}

// CostIntervalAddToFreeList, backward_references_cost_enc.c:201.
func (cm *lw16CostManager) addToFreeList(iv *lw16CostInterval) {
	iv.next = cm.free
	cm.free = iv
}

// newInterval pops the free list; ivs never runs out (count <= 500).
func (cm *lw16CostManager) newInterval() *lw16CostInterval {
	iv := cm.free
	cm.free = iv.next
	return iv
}

// CostManagerInit, backward_references_cost_enc.c:249.
func (cm *lw16CostManager) init(distArray []uint16, pixCount int, m *lw16CostModel) {
	cacheSize := min(pixCount, lwMaxLength)
	cm.head = nil
	cm.count = 0
	cm.distArray = distArray
	cm.free = nil
	for i := range cm.ivs {
		cm.addToFreeList(&cm.ivs[i])
	}

	for i := 0; i < cacheSize; i++ {
		cm.costCache[i] = m.lengthCost(uint32(i))
	}

	// Equal neighbours share one interval.
	// One interval per length prefix code at most.
	cm.cacheIntervals = append(slices.Grow(cm.cacheIntervals[:0], lwNumLength), lw16CostCacheInterval{cost: cm.costCache[0], start: 0, end: 1})
	cur := &cm.cacheIntervals[0]
	for i := 1; i < cacheSize; i++ {
		costVal := cm.costCache[i]
		if costVal != cur.cost {
			cm.cacheIntervals = append(cm.cacheIntervals, lw16CostCacheInterval{cost: costVal, start: i})
			cur = &cm.cacheIntervals[len(cm.cacheIntervals)-1]
		}
		cur.end = i + 1
	}

	if cap(cm.costs) < pixCount {
		cm.costs = make([]int64, pixCount)
	}
	cm.costs = cm.costs[:pixCount]
	for i := range cm.costs {
		cm.costs[i] = v16Int64Max
	}
}

// UpdateCost, backward_references_cost_enc.c:327.
func (cm *lw16CostManager) updateCost(i, position int, cost int64) {
	k := i - position
	if cm.costs[i] > cost {
		cm.costs[i] = cost
		cm.distArray[i] = uint16(k + 1)
	}
}

// UpdateCostPerInterval, backward_references_cost_enc.c:340.
func (cm *lw16CostManager) updateCostPerInterval(start, end, position int, cost int64) {
	for i := start; i < end; i++ {
		cm.updateCost(i, position, cost)
	}
}

// ConnectIntervals, backward_references_cost_enc.c:348.
func (cm *lw16CostManager) connectIntervals(prev, next *lw16CostInterval) {
	if prev != nil {
		prev.next = next
	} else {
		cm.head = next
	}
	if next != nil {
		next.prev = prev
	}
}

// PopInterval, backward_references_cost_enc.c:361.
func (cm *lw16CostManager) popInterval(iv *lw16CostInterval) {
	if iv == nil {
		return
	}
	cm.connectIntervals(iv.prev, iv.next)
	cm.addToFreeList(iv)
	cm.count--
}

// UpdateCostAtIndex, backward_references_cost_enc.c:380.
func (cm *lw16CostManager) updateCostAtIndex(i int, clean bool) {
	cur := cm.head
	for cur != nil && cur.start <= i {
		next := cur.next
		if cur.end <= i {
			if clean {
				cm.popInterval(cur)
			}
		} else {
			cm.updateCost(i, cur.index, cur.cost)
		}
		cur = next
	}
}

// PositionOrphanInterval, backward_references_cost_enc.c:401.
func (cm *lw16CostManager) positionOrphanInterval(cur, prev *lw16CostInterval) {
	if prev == nil {
		prev = cm.head
	}
	for prev != nil && cur.start < prev.start {
		prev = prev.prev
	}
	for prev != nil && prev.next != nil && prev.next.start < cur.start {
		prev = prev.next
	}
	if prev != nil {
		cm.connectIntervals(cur, prev.next)
	} else {
		cm.connectIntervals(cur, cm.head)
	}
	cm.connectIntervals(prev, cur)
}

// InsertInterval, backward_references_cost_enc.c:425.
func (cm *lw16CostManager) insertInterval(in *lw16CostInterval, cost int64, position, start, end int) {
	if start >= end {
		return
	}
	if cm.count >= lwCostIntervalMax {
		cm.updateCostPerInterval(start, end, position, cost)
		return
	}
	iv := cm.newInterval()
	iv.cost = cost
	iv.index = position
	iv.start = start
	iv.end = end
	cm.positionOrphanInterval(iv, in)
	cm.count++
}

// PushInterval, backward_references_cost_enc.c:465.
func (cm *lw16CostManager) pushInterval(distCost int64, position, length int) {
	iv := cm.head
	var ivNext *lw16CostInterval
	ci := cm.cacheIntervals

	if length < lwSkipDistance {
		for j := position; j < position+length; j++ {
			k := j - position
			costTmp := distCost + cm.costCache[k]
			if cm.costs[j] > costTmp {
				cm.costs[j] = costTmp
				cm.distArray[j] = uint16(k + 1)
			}
		}
		return
	}

	for i := 0; i < len(ci) && ci[i].start < length; i++ {
		start := position + ci[i].start
		end := position + ci[i].end
		if ci[i].end > length {
			end = position + length
		}
		cost := distCost + ci[i].cost

		for ; iv != nil && iv.start < end; iv = ivNext {
			ivNext = iv.next

			if start >= iv.end {
				continue
			}

			if cost >= iv.cost {
				startNew := iv.end
				cm.insertInterval(iv, cost, position, start, iv.start)
				start = startNew
				if start >= end {
					break
				}
				continue
			}

			if start <= iv.start {
				if iv.end <= end {
					cm.popInterval(iv)
				} else {
					iv.start = end
					break
				}
			} else {
				if end < iv.end {
					endOriginal := iv.end
					iv.end = start
					cm.insertInterval(iv, iv.cost, iv.index, end, endOriginal)
					iv = iv.next
					break
				} else {
					iv.end = start
				}
			}
		}
		cm.insertInterval(iv, cost, position, start, end)
	}
}

// distanceOnly fills distArray using s.costModel and s.costMgr.
// BackwardReferencesHashChainDistanceOnly, backward_references_cost_enc.c:569.
func (s *lw16State) distanceOnly(xsize, ysize int, argb []uint32, cacheBits int, hc *lwHashChain, refs lwRefs, distArray []uint16) {
	pixCount := xsize * ysize
	m := &s.costModel
	cm := &s.costMgr
	var cc *v16ColorCache
	if cacheBits > 0 {
		cc = &s.cc
		cc.init(cacheBits)
	}
	offsetPrev, lenPrev := -1, -1
	offsetCost := int64(-1)
	firstOffsetIsConstant := -1
	reach := 0

	s.costModelBuild(m, xsize, cacheBits, refs)
	cm.init(distArray, pixCount, m)

	distArray[0] = 0
	v16AddSingleLiteralWithCostModel(argb, cc, m, 0, 0, cm.costs, distArray)

	for i := 1; i < pixCount; i++ {
		prevCost := cm.costs[i-1]
		offset, length := hc.findOffset(i), hc.findLength(i)

		v16AddSingleLiteralWithCostModel(argb, cc, m, i, prevCost, cm.costs, distArray)

		if length >= 2 {
			if offset != offsetPrev {
				code := distanceToPlaneCode(xsize, offset)
				offsetCost = m.distanceCost(uint32(code))
				firstOffsetIsConstant = 1
				cm.pushInterval(prevCost+offsetCost, i, length)
			} else {
				if firstOffsetIsConstant != 0 {
					reach = i - 1 + lenPrev - 1
					firstOffsetIsConstant = 0
				}

				if i+length-1 > reach {
					var offsetJ int
					lenJ := 0
					j := i
					for ; j <= reach; j++ {
						offsetJ, lenJ = hc.findOffset(j+1), hc.findLength(j+1)
						if offsetJ != offset {
							offsetJ, lenJ = hc.findOffset(j), hc.findLength(j) //nolint:staticcheck // as libwebp
							break
						}
					}
					cm.updateCostAtIndex(j-1, false)
					cm.updateCostAtIndex(j, false)

					cm.pushInterval(cm.costs[j-1]+offsetCost, j, lenJ)
					reach = j + lenJ - 1
				}
			}
		}

		cm.updateCostAtIndex(i, true)
		offsetPrev = offset
		lenPrev = length
	}
}

// followChosenPath rewrites refs from path.
// BackwardReferencesHashChainFollowChosenPath, backward_references_cost_enc.c:711.
func (s *lw16State) followChosenPath(argb []uint32, cacheBits int, path []uint16, hc *lwHashChain, refs *lwRefs) {
	var cc *v16ColorCache
	if cacheBits > 0 {
		cc = &s.cc
		cc.init(cacheBits)
	}
	i := 0
	*refs = slices.Grow((*refs)[:0], len(path))
	for _, l := range path {
		n := int(l)
		if n != 1 {
			offset := hc.findOffset(i)
			refs.add(pixOrCopy{mode: lwCopy, len: uint16(n), argbOrDist: uint32(offset)})
			if cc != nil {
				for k := 0; k < n; k++ {
					cc.insert(argb[i+k])
				}
			}
			i += n
			continue
		}
		idx := -1
		if cc != nil {
			idx = cc.contains(argb[i])
		}
		if idx >= 0 {
			refs.add(pixOrCopy{mode: lw16CacheIdx, len: 1, argbOrDist: uint32(idx)})
		} else {
			if cc != nil {
				cc.insert(argb[i])
			}
			refs.add(pixOrCopy{mode: lwLiteral, len: 1, argbOrDist: argb[i]})
		}
		i++
	}
}

// traceBackwards uses e.distArray; TraceBackwards :695 is shared.
// VP8LBackwardReferencesTraceBackwards, backward_references_cost_enc.c:767.
func (e *LW) v16TraceBackwards(xsize, ysize int, argb []uint32, cacheBits int, hc *lwHashChain, src lwRefs, dst *lwRefs) {
	n := xsize * ysize
	if cap(e.distArray) < n {
		e.distArray = make([]uint16, n)
	}
	e.distArray = e.distArray[:n]
	e.v16.distanceOnly(xsize, ysize, argb, cacheBits, hc, src, e.distArray)
	path := traceBackwards(e.distArray)
	e.v16.followChosenPath(argb, cacheBits, path, hc, dst)
}
