// Package vp8l writes VP8L image streams for WebP ALPH chunks.
//
// Only what alpha needs: no transforms, no color cache, one prefix-code
// group, and LZ77 copies from the left or the row above. Output is
// lossless, so it decodes to the same alpha as libwebp's, though the
// bytes differ. The lw*.go files port libwebp's ALPH encoder instead
// (see LICENSE.libwebp).
package vp8l

import (
	"math/bits"
	"slices"
)

const (
	numLen    = 24 // length prefix codes
	numGreen  = 256 + numLen
	numDist   = 40
	maxLen    = 4096
	minCopy   = 3
	distAbove = 1 // plane code (0, 1)
	distLeft  = 2 // plane code (1, 0)
)

// kCodeLengthCodeOrder, spec section 3.7.2.1.2.
var clOrder = [19]uint8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// Encoder keeps scratch between calls so Alpha allocates only dst.
type Encoder struct {
	bw     bitWriter
	green  code
	dist   code
	cl     code
	clSyms []uint16 // code-length symbol, extra bits in the high byte
}

// Alpha appends the image stream of a (w x h, stride w) to dst.
func (e *Encoder) Alpha(dst []byte, a []uint8, w, h int) []byte {
	var gh [numGreen]uint32
	var dh [numDist]uint32
	walk(a, w, h, func(lit bool, v, n, d int) {
		if lit {
			gh[v]++
			return
		}
		gh[256+prefix(n)]++
		dh[prefix(d)]++
	})
	e.green.build(gh[:], 15)
	e.dist.build(dh[:], 15)

	e.bw = bitWriter{buf: dst}
	b := &e.bw
	b.put(0, 1) // no transform
	b.put(0, 1) // no color cache
	b.put(0, 1) // no meta prefix codes
	e.writeCode(&e.green)
	for range 3 { // red, blue, alpha: always 0
		b.put(1, 1) // simple code
		b.put(0, 1) // one symbol
		b.put(0, 1) // 1-bit symbol
		b.put(0, 1) // symbol 0
	}
	e.writeCode(&e.dist)
	walk(a, w, h, func(lit bool, v, n, d int) {
		if lit {
			e.green.put(b, v)
			return
		}
		p, xb, xv := prefixBits(n)
		e.green.put(b, 256+p)
		b.put(uint64(xv), xb)
		p, xb, xv = prefixBits(d)
		e.dist.put(b, p)
		b.put(uint64(xv), xb)
	})
	return b.flush()
}

// walk calls f with literals and greedy left or above copies.
func walk(a []uint8, w, h int, f func(lit bool, v, n, d int)) {
	n := w * h
	for i := 0; i < n; {
		best, dist := 0, 0
		if i >= 1 {
			k := run(a, i, 1, n)
			best, dist = k, distLeft
		}
		if i >= w {
			if k := run(a, i, w, n); k > best {
				best, dist = k, distAbove
			}
		}
		if best >= minCopy {
			f(false, 0, best, dist)
			i += best
			continue
		}
		f(true, int(a[i]), 0, 0)
		i++
	}
}

// run is how many pixels from i repeat those d back, up to maxLen.
func run(a []uint8, i, d, n int) int {
	k := 0
	for k < maxLen && i+k < n && a[i+k] == a[i+k-d] {
		k++
	}
	return k
}

// prefix is the prefix code of a length or distance value v >= 1.
func prefix(v int) int {
	p, _, _ := prefixBits(v)
	return p
}

// prefixBits is VP8LPrefixEncode: code, extra bit count and extra value.
func prefixBits(v int) (code, n, extra int) {
	v--
	if v < 4 {
		return v, 0, 0
	}
	hb := bits.Len(uint(v)) - 1
	second := (v >> (hb - 1)) & 1
	n = hb - 1
	return 2*hb + second, n, v & (1<<n - 1)
}

// code is a canonical prefix code; bits are stored reversed.
type code struct {
	len  [numGreen]uint8
	bits [numGreen]uint16
	n    int
	used int // symbols with a nonzero length
}

// build makes lengths up to limit from counts, then the codes.
func (c *code) build(counts []uint32, limit int) {
	c.n = len(counts)
	clear(c.len[:])
	huffLens(counts, c.len[:c.n], limit)
	c.used = 0
	for _, l := range c.len[:c.n] {
		if l != 0 {
			c.used++
		}
	}
	var next [16]uint16
	var cnt [16]uint16
	for _, l := range c.len[:c.n] {
		cnt[l]++
	}
	cnt[0] = 0
	v := uint16(0)
	for l := 1; l < 16; l++ {
		v = (v + cnt[l-1]) << 1
		next[l] = v
	}
	for s, l := range c.len[:c.n] {
		if l == 0 {
			continue
		}
		v := next[l]
		next[l]++
		r := uint16(0)
		for range l {
			r = r<<1 | v&1
			v >>= 1
		}
		c.bits[s] = r
	}
}

// put writes symbol s; a one-symbol code takes no bits.
func (c *code) put(b *bitWriter, s int) {
	if c.used > 1 {
		b.put(uint64(c.bits[s]), int(c.len[s]))
	}
}

// writeCode writes c, as a simple code when it has at most one symbol,
// else as a normal code with run-length coded lengths.
func (e *Encoder) writeCode(c *code) {
	b := &e.bw
	if c.used <= 1 {
		s := max(0, slices.IndexFunc(c.len[:c.n], func(l uint8) bool { return l != 0 }))
		b.put(1, 1) // simple code
		b.put(0, 1) // one symbol
		if s < 2 {
			b.put(0, 1)
			b.put(uint64(s), 1)
		} else {
			b.put(1, 1)
			b.put(uint64(s), 8)
		}
		return
	}
	syms := e.clSyms[:0]
	prev := uint8(8)
	lens := c.len[:c.n]
	for i := 0; i < len(lens); {
		v := lens[i]
		r := 1
		for i+r < len(lens) && lens[i+r] == v {
			r++
		}
		i += r
		if v == 0 {
			for r > 0 {
				switch {
				case r >= 11:
					k := min(r, 138)
					syms = append(syms, 18|uint16(k-11)<<8)
					r -= k
				case r >= 3:
					k := min(r, 10)
					syms = append(syms, 17|uint16(k-3)<<8)
					r -= k
				default:
					syms = append(syms, 0)
					r--
				}
			}
			continue
		}
		if v != prev {
			syms = append(syms, uint16(v))
			prev = v
			r--
		}
		for r >= 3 {
			k := min(r, 6)
			syms = append(syms, 16|uint16(k-3)<<8)
			r -= k
		}
		for ; r > 0; r-- {
			syms = append(syms, uint16(v))
		}
	}
	e.clSyms = syms

	var ch [19]uint32
	for _, s := range syms {
		ch[s&0xff]++
	}
	e.cl.build(ch[:], 7)
	k := 4
	for i, s := range clOrder {
		if e.cl.len[s] != 0 {
			k = max(k, i+1)
		}
	}
	b.put(0, 1) // normal code
	b.put(uint64(k-4), 4)
	for _, s := range clOrder[:k] {
		b.put(uint64(e.cl.len[s]), 3)
	}
	b.put(0, 1) // max_symbol is the alphabet size
	for _, s := range syms {
		e.cl.put(b, int(s&0xff))
		switch s & 0xff {
		case 16:
			b.put(uint64(s>>8), 2)
		case 17:
			b.put(uint64(s>>8), 3)
		case 18:
			b.put(uint64(s>>8), 7)
		}
	}
}

// huffLens sets Huffman code lengths of at most limit bits into lens.
// Zero counts get length 0; a lone symbol gets 1, which decoders read
// as a zero-bit code. ponytail: halves counts until the tree fits,
// package-merge if the extra bits ever matter.
func huffLens(counts []uint32, lens []uint8, limit int) {
	type node struct {
		w           uint32
		left, right int16 // children, -1 for a leaf
		sym         int16
	}
	var buf [2 * numGreen]node
	var c [numGreen]uint32
	copy(c[:], counts)
	for {
		nodes := buf[:0]
		for s, w := range c[:len(counts)] {
			if w > 0 {
				nodes = append(nodes, node{w, -1, -1, int16(s)})
			}
		}
		switch len(nodes) {
		case 0:
			return
		case 1:
			lens[nodes[0].sym] = 1
			return
		}
		slices.SortFunc(nodes, func(a, b node) int { return int(a.w) - int(b.w) })
		// Two-queue build: leaves sorted, internal nodes made in order.
		leaves := len(nodes)
		li, qi := 0, leaves
		pick := func() int {
			if li < leaves && (qi >= len(nodes) || nodes[li].w <= nodes[qi].w) {
				li++
				return li - 1
			}
			qi++
			return qi - 1
		}
		for len(nodes) < 2*leaves-1 {
			x, y := pick(), pick()
			nodes = append(nodes, node{nodes[x].w + nodes[y].w, int16(x), int16(y), -1})
		}
		var depth [2 * numGreen]uint8
		deep := 0
		for i := len(nodes) - 1; i >= 0; i-- {
			if nodes[i].left >= 0 {
				depth[nodes[i].left] = depth[i] + 1
				depth[nodes[i].right] = depth[i] + 1
			} else {
				deep = max(deep, int(depth[i]))
			}
		}
		if deep <= limit {
			for i := range leaves {
				lens[nodes[i].sym] = depth[i]
			}
			return
		}
		for s := range c[:len(counts)] {
			if c[s] > 0 {
				c[s] = max(1, c[s]>>1)
			}
		}
	}
}

// bitWriter packs bits LSB first, as VP8L reads them.
type bitWriter struct {
	buf  []byte
	acc  uint64
	nacc int
}

func (b *bitWriter) put(v uint64, n int) {
	b.acc |= v << b.nacc
	b.nacc += n
	for b.nacc >= 8 {
		b.buf = append(b.buf, byte(b.acc))
		b.acc >>= 8
		b.nacc -= 8
	}
}

func (b *bitWriter) flush() []byte {
	if b.nacc > 0 {
		b.buf = append(b.buf, byte(b.acc))
	}
	b.acc, b.nacc = 0, 0
	return b.buf
}
