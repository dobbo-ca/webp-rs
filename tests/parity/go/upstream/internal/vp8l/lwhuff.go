// Port of libwebp 1.2.2 utils/huffman_encode_utils.c. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import "slices"

// MAX_ALLOWED_CODE_LENGTH, format_constants.h.
const maxAllowedCodeLength = 15

// ValuesShouldBeCollapsedToStrideAverage, huffman_encode_utils.c:25.
func valuesShouldBeCollapsedToStrideAverage(a, b int) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 4
}

// OptimizeHuffmanForRle, huffman_encode_utils.c:31; length is len(counts).
func optimizeHuffmanForRle(goodForRle []uint8, counts []uint32) {
	length := len(counts)
	for ; length >= 0; length-- {
		if length == 0 {
			return // all zeros
		}
		if counts[length-1] != 0 {
			break
		}
	}
	// Mark runs that already code well with RLE.
	{
		symbol := counts[0]
		stride := 0
		for i := 0; i < length+1; i++ {
			if i == length || counts[i] != symbol {
				if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
					for k := 0; k < stride; k++ {
						goodForRle[i-k-1] = 1
					}
				}
				stride = 1
				if i != length {
					symbol = counts[i]
				}
			} else {
				stride++
			}
		}
	}
	// Replace counts that lead to more RLE codes.
	{
		var stride uint32
		limit := counts[0]
		var sum uint32
		for i := 0; i < length+1; i++ {
			if i == length || goodForRle[i] != 0 ||
				(i != 0 && goodForRle[i-1] != 0) ||
				!valuesShouldBeCollapsedToStrideAverage(int(int32(counts[i])), int(int32(limit))) {
				if stride >= 4 || (stride >= 3 && sum == 0) {
					count := (sum + stride/2) / stride
					if count < 1 {
						count = 1
					}
					if sum == 0 {
						count = 0
					}
					for k := uint32(0); k < stride; k++ {
						counts[i-int(k)-1] = count
					}
				}
				stride = 0
				sum = 0
				if i < length-3 {
					limit = (counts[i] + counts[i+1] + counts[i+2] + counts[i+3] + 2) / 4
				} else if i < length {
					limit = counts[i]
				} else {
					limit = 0
				}
			}
			stride++
			if i != length {
				sum += counts[i]
				if stride >= 4 {
					limit = (sum + stride/2) / stride
				}
			}
		}
	}
}

// compareHuffmanTrees is a slices.SortFunc cmp: count desc, value asc.
// CompareHuffmanTrees, huffman_encode_utils.c:122.
func compareHuffmanTrees(a, b *huffTree) int {
	if a.totalCount > b.totalCount {
		return -1
	} else if a.totalCount < b.totalCount {
		return 1
	}
	if a.value < b.value {
		return -1
	}
	return 1
}

// SetBitDepths, huffman_encode_utils.c:135.
func setBitDepths(t *huffTree, pool []huffTree, depths []uint8, level int) {
	if t.left >= 0 {
		setBitDepths(&pool[t.left], pool, depths, level+1)
		setBitDepths(&pool[t.right], pool, depths, level+1)
	} else {
		depths[t.value] = uint8(level)
	}
}

// generateOptimalTree: tree has room for 3*len(histogram) nodes.
// GenerateOptimalTree, huffman_encode_utils.c:165.
func generateOptimalTree(histogram []uint32, tree []huffTree, limit int, depths []uint8) {
	sizeOrig := 0
	for _, v := range histogram {
		if v != 0 {
			sizeOrig++
		}
	}
	if sizeOrig == 0 {
		return
	}
	pool := tree[sizeOrig:]
	for countMin := uint32(1); ; countMin *= 2 {
		size := sizeOrig
		idx := 0
		for j, v := range histogram {
			if v != 0 {
				count := v
				if v < countMin {
					count = countMin
				}
				tree[idx] = huffTree{totalCount: count, value: j, left: -1, right: -1}
				idx++
			}
		}
		// Values are distinct, so the order is total and qsort-equivalent.
		slices.SortFunc(tree[:size], func(a, b huffTree) int { return compareHuffmanTrees(&a, &b) })

		if size > 1 {
			poolSize := 0
			for size > 1 {
				pool[poolSize] = tree[size-1]
				poolSize++
				pool[poolSize] = tree[size-2]
				poolSize++
				count := pool[poolSize-1].totalCount + pool[poolSize-2].totalCount
				size -= 2
				k := 0
				for ; k < size; k++ {
					if tree[k].totalCount <= count {
						break
					}
				}
				copy(tree[k+1:size+1], tree[k:size])
				tree[k] = huffTree{totalCount: count, value: -1, left: poolSize - 1, right: poolSize - 2}
				size++
			}
			setBitDepths(&tree[0], pool, depths, 0)
		} else if size == 1 {
			depths[tree[0].value] = 1
		}

		maxDepth := int(depths[0])
		for j := 1; j < len(histogram); j++ {
			if maxDepth < int(depths[j]) {
				maxDepth = int(depths[j])
			}
		}
		if maxDepth <= limit {
			break
		}
	}
}

// codeRepeatedValues appends to tokens and returns it.
// CodeRepeatedValues, huffman_encode_utils.c:261.
func codeRepeatedValues(tokens []huffTreeToken, reps, value, prev int) []huffTreeToken {
	if value != prev {
		tokens = append(tokens, huffTreeToken{code: uint8(value)})
		reps--
	}
	for reps >= 1 {
		if reps < 3 {
			for i := 0; i < reps; i++ {
				tokens = append(tokens, huffTreeToken{code: uint8(value)})
			}
			break
		} else if reps < 7 {
			tokens = append(tokens, huffTreeToken{code: 16, extraBits: uint8(reps - 3)})
			break
		} else {
			tokens = append(tokens, huffTreeToken{code: 16, extraBits: 3})
			reps -= 6
		}
	}
	return tokens
}

// CodeRepeatedZeros, huffman_encode_utils.c:295.
func codeRepeatedZeros(tokens []huffTreeToken, reps int) []huffTreeToken {
	for reps >= 1 {
		if reps < 3 {
			for i := 0; i < reps; i++ {
				tokens = append(tokens, huffTreeToken{code: 0})
			}
			break
		} else if reps < 11 {
			tokens = append(tokens, huffTreeToken{code: 17, extraBits: uint8(reps - 3)})
			break
		} else if reps < 139 {
			tokens = append(tokens, huffTreeToken{code: 18, extraBits: uint8(reps - 11)})
			break
		} else {
			tokens = append(tokens, huffTreeToken{code: 18, extraBits: 0x7f})
			reps -= 138
		}
	}
	return tokens
}

// createCompressedHuffmanTree appends tokens to tokens[:0].
// VP8LCreateCompressedHuffmanTree, huffman_encode_utils.c:326.
func createCompressedHuffmanTree(c *huffTreeCode, tokens []huffTreeToken) []huffTreeToken {
	tokens = tokens[:0]
	n := c.numSymbols
	prev := 8 // initial RLE value
	i := 0
	for i < n {
		value := int(c.codeLengths[i])
		k := i + 1
		for k < n && int(c.codeLengths[k]) == value {
			k++
		}
		runs := k - i
		if value == 0 {
			tokens = codeRepeatedZeros(tokens, runs)
		} else {
			tokens = codeRepeatedValues(tokens, runs, value, prev)
			prev = value
		}
		i += runs
	}
	return tokens
}

// kReversedBits, huffman_encode_utils.c:356.
var reversedBits = [16]uint8{
	0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
	0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
}

// ReverseBits, huffman_encode_utils.c:361.
func reverseBits(n int, bits uint32) uint32 {
	var r uint32
	i := 0
	for i < n {
		i += 4
		r |= uint32(reversedBits[bits&0xf]) << (maxAllowedCodeLength + 1 - i)
		bits >>= 4
	}
	r >>= maxAllowedCodeLength + 1 - n
	return r
}

// ConvertBitDepthsToSymbols, huffman_encode_utils.c:374.
func convertBitDepthsToSymbols(c *huffTreeCode) {
	var nextCode [maxAllowedCodeLength + 1]uint32
	var depthCount [maxAllowedCodeLength + 1]int
	n := c.numSymbols
	for i := 0; i < n; i++ {
		depthCount[c.codeLengths[i]]++
	}
	depthCount[0] = 0 // unused symbols
	var code uint32
	for i := 1; i <= maxAllowedCodeLength; i++ {
		code = (code + uint32(depthCount[i-1])) << 1
		nextCode[i] = code
	}
	for i := 0; i < n; i++ {
		l := c.codeLengths[i]
		c.codes[i] = uint16(reverseBits(int(l), nextCode[l]))
		nextCode[l]++
	}
}

// createHuffmanTree mutates histogram (RLE optimisation) and fills c.
// bufRle and tree are scratch of len(histogram) and 3*len(histogram).
// c.codeLengths must be zeroed, as C's calloc'd buffer is.
// VP8LCreateHuffmanTree, huffman_encode_utils.c:406.
func createHuffmanTree(histogram []uint32, limit int, bufRle []uint8, tree []huffTree, c *huffTreeCode) {
	n := c.numSymbols
	clear(bufRle[:n])
	optimizeHuffmanForRle(bufRle, histogram[:n])
	generateOptimalTree(histogram[:n], tree, limit, c.codeLengths)
	convertBitDepthsToSymbols(c)
}
