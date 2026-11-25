// Port of libwebp 1.2.2 enc/backward_references_cost_enc.c, without the color cache. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "slices"

const (
	lwCostMul1     float64 = 0.82 // mul1, AddSingleLiteralWithCostModel
	lwCostInit     float32 = 1e38 // 1e38f, CostManagerInit
	lwSkipDistance         = 10   // kSkipDistance, PushInterval
)

// ConvertPopulationCountTableToBitEstimates, backward_references_cost_enc.c:40.
func convertPopulationCountTableToBitEstimates(counts []uint32, out []float64) {
	var sum uint32
	nonzeros := 0
	for i := range counts {
		sum += counts[i]
		if counts[i] > 0 {
			nonzeros++
		}
	}
	if nonzeros <= 1 {
		clear(out[:len(counts)])
	} else {
		logsum := float64(fastLog2(sum))
		for i := range counts {
			out[i] = logsum - float64(fastLog2(counts[i]))
		}
	}
}

// costModelBuild uses e.costHisto with plane-code distances.
// CostModelBuild, backward_references_cost_enc.c:61.
func (e *LW) costModelBuild(m *lwCostModel, xsize int, refs lwRefs) {
	h := &e.costHisto
	h.init(true)
	for k := range refs {
		h.addSinglePixOrCopy(&refs[k], xsize)
	}
	convertPopulationCountTableToBitEstimates(h.literal[:], m.literal[:])
	convertPopulationCountTableToBitEstimates(h.red[:], m.red[:])
	convertPopulationCountTableToBitEstimates(h.blue[:], m.blue[:])
	convertPopulationCountTableToBitEstimates(h.alpha[:], m.alpha[:])
	convertPopulationCountTableToBitEstimates(h.distance[:], m.distance[:])
}

// GetLiteralCost, backward_references_cost_enc.c:95.
func (m *lwCostModel) literalCost(v uint32) float64 {
	return m.alpha[v>>24] +
		m.red[(v>>16)&0xff] +
		m.literal[(v>>8)&0xff] +
		m.blue[v&0xff]
}

// prefixCodeBits is VP8LPrefixEncodeBits; kPrefixEncodeCode[0] is {0, 0}.
func prefixCodeBits(v int) (code, extra int) {
	if v == 0 {
		return 0, 0
	}
	code, extra, _ = prefixBits(v)
	return code, extra
}

// GetLengthCost, backward_references_cost_enc.c:107.
func (m *lwCostModel) lengthCost(n uint32) float64 {
	code, extra := prefixCodeBits(int(n))
	return m.literal[lwNumLiteral+code] + float64(extra)
}

// GetDistanceCost, backward_references_cost_enc.c:114.
func (m *lwCostModel) distanceCost(d uint32) float64 {
	code, extra := prefixCodeBits(int(d))
	return m.distance[code] + float64(extra)
}

// AddSingleLiteralWithCostModel, backward_references_cost_enc.c:121, no cache.
func addSingleLiteralWithCostModel(argb []uint32, m *lwCostModel, idx int, prevCost float32, cost []float32, distArray []uint16) {
	costVal := float64(prevCost)
	costVal += float64(m.literalCost(argb[idx]) * lwCostMul1)
	if float64(cost[idx]) > costVal {
		cost[idx] = float32(costVal)
		distArray[idx] = 1
	}
}

// CostIntervalAddToFreeList, backward_references_cost_enc.c:202.
func (cm *lwCostManager) addToFreeList(iv *lwCostInterval) {
	iv.next = cm.free
	cm.free = iv
}

// newInterval pops the free list; ivs never runs out (count <= 500). InsertInterval, backward_references_cost_enc.c:431-444.
func (cm *lwCostManager) newInterval() *lwCostInterval {
	iv := cm.free
	cm.free = iv.next
	return iv
}

// CostManagerInit, backward_references_cost_enc.c:250 (and InitFreeList :214).
func (cm *lwCostManager) init(distArray []uint16, pixCount int, m *lwCostModel) {
	cacheSize := pixCount
	if pixCount > lwMaxLength {
		cacheSize = lwMaxLength
	}
	cm.head = nil
	cm.count = 0
	cm.distArray = distArray
	cm.free = nil
	for i := range cm.ivs {
		cm.addToFreeList(&cm.ivs[i])
	}

	cm.costCache[0] = m.lengthCost(0)
	for i := 1; i < cacheSize; i++ {
		cm.costCache[i] = m.lengthCost(uint32(i))
	}

	// Equal neighbours share one interval.
	// One interval per length prefix code at most.
	cm.cacheIntervals = append(slices.Grow(cm.cacheIntervals[:0], lwNumLength), lwCostCacheInterval{cost: cm.costCache[0], start: 0, end: 1})
	cur := &cm.cacheIntervals[0]
	for i := 1; i < cacheSize; i++ {
		costVal := cm.costCache[i]
		if costVal != cur.cost {
			cm.cacheIntervals = append(cm.cacheIntervals, lwCostCacheInterval{cost: costVal, start: i})
			cur = &cm.cacheIntervals[len(cm.cacheIntervals)-1]
		}
		cur.end = i + 1
	}

	if cap(cm.costs) < pixCount {
		cm.costs = make([]float32, pixCount)
	}
	cm.costs = cm.costs[:pixCount]
	for i := range cm.costs {
		cm.costs[i] = lwCostInit
	}
}

// UpdateCost, backward_references_cost_enc.c:321.
func (cm *lwCostManager) updateCost(i, position int, cost float32) {
	k := i - position
	if cm.costs[i] > cost {
		cm.costs[i] = cost
		cm.distArray[i] = uint16(k + 1)
	}
}

// UpdateCostPerInterval, backward_references_cost_enc.c:334.
func (cm *lwCostManager) updateCostPerInterval(start, end, position int, cost float32) {
	for i := start; i < end; i++ {
		cm.updateCost(i, position, cost)
	}
}

// ConnectIntervals, backward_references_cost_enc.c:342.
func (cm *lwCostManager) connectIntervals(prev, next *lwCostInterval) {
	if prev != nil {
		prev.next = next
	} else {
		cm.head = next
	}
	if next != nil {
		next.prev = prev
	}
}

// PopInterval, backward_references_cost_enc.c:355.
func (cm *lwCostManager) popInterval(iv *lwCostInterval) {
	if iv == nil {
		return
	}
	cm.connectIntervals(iv.prev, iv.next)
	cm.addToFreeList(iv)
	cm.count--
}

// UpdateCostAtIndex, backward_references_cost_enc.c:374.
func (cm *lwCostManager) updateCostAtIndex(i int, clean bool) {
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

// PositionOrphanInterval, backward_references_cost_enc.c:395.
func (cm *lwCostManager) positionOrphanInterval(cur, prev *lwCostInterval) {
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

// InsertInterval, backward_references_cost_enc.c:419.
func (cm *lwCostManager) insertInterval(in *lwCostInterval, cost float32, position, start, end int) {
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

// PushInterval, backward_references_cost_enc.c:459.
func (cm *lwCostManager) pushInterval(distCost float64, position, length int) {
	iv := cm.head
	var ivNext *lwCostInterval
	ci := cm.cacheIntervals

	if length < lwSkipDistance {
		for j := position; j < position+length; j++ {
			k := j - position
			costTmp := float32(distCost + cm.costCache[k])
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
		cost := float32(distCost + ci[i].cost)

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

// distanceOnly fills distArray using e.costModel and e.costMgr.
// BackwardReferencesHashChainDistanceOnly, backward_references_cost_enc.c:563.
func (e *LW) distanceOnly(xsize, ysize int, argb []uint32, hc *lwHashChain, refs lwRefs, distArray []uint16) {
	pixCount := xsize * ysize
	m := &e.costModel
	cm := &e.costMgr
	offsetPrev, lenPrev := -1, -1
	offsetCost := float64(-1)
	firstOffsetIsConstant := -1
	reach := 0

	e.costModelBuild(m, xsize, refs)
	cm.init(distArray, pixCount, m)

	distArray[0] = 0
	addSingleLiteralWithCostModel(argb, m, 0, 0, cm.costs, distArray)

	for i := 1; i < pixCount; i++ {
		prevCost := cm.costs[i-1]
		offset, length := hc.findOffset(i), hc.findLength(i)

		addSingleLiteralWithCostModel(argb, m, i, prevCost, cm.costs, distArray)

		if length >= 2 {
			if offset != offsetPrev {
				code := distanceToPlaneCode(xsize, offset)
				offsetCost = m.distanceCost(uint32(code))
				firstOffsetIsConstant = 1
				cm.pushInterval(float64(prevCost)+offsetCost, i, length)
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

					cm.pushInterval(float64(cm.costs[j-1])+offsetCost, j, lenJ)
					reach = j + lenJ - 1
				}
			}
		}

		cm.updateCostAtIndex(i, true)
		offsetPrev = offset
		lenPrev = length
	}
}

// traceBackwards packs the path at the end of distArray and returns it.
// TraceBackwards, backward_references_cost_enc.c:689.
func traceBackwards(distArray []uint16) []uint16 {
	p := len(distArray)
	for cur := len(distArray) - 1; cur >= 0; {
		k := int(distArray[cur])
		p--
		distArray[p] = uint16(k)
		cur -= k
	}
	return distArray[p:]
}

// followChosenPath rewrites refs from path.
// BackwardReferencesHashChainFollowChosenPath, backward_references_cost_enc.c:705.
func followChosenPath(argb []uint32, path []uint16, hc *lwHashChain, refs *lwRefs) {
	i := 0
	*refs = slices.Grow((*refs)[:0], len(path))
	for _, l := range path {
		n := int(l)
		if n != 1 {
			offset := hc.findOffset(i)
			refs.add(pixOrCopy{mode: lwCopy, len: uint16(n), argbOrDist: uint32(offset)})
			i += n
		} else {
			refs.add(pixOrCopy{mode: lwLiteral, len: 1, argbOrDist: argb[i]})
			i++
		}
	}
}

// backwardReferencesTraceBackwards uses e.distArray.
// VP8LBackwardReferencesTraceBackwards, backward_references_cost_enc.c:761.
func (e *LW) backwardReferencesTraceBackwards(xsize, ysize int, argb []uint32, hc *lwHashChain, src lwRefs, dst *lwRefs) {
	n := xsize * ysize
	if cap(e.distArray) < n {
		e.distArray = make([]uint16, n)
	}
	e.distArray = e.distArray[:n]
	e.distanceOnly(xsize, ysize, argb, hc, src, e.distArray)
	path := traceBackwards(e.distArray)
	followChosenPath(argb, path, hc, dst)
}
