// Package vp8ldec decodes VP8L lossless image streams (RFC 9649), bit
// exact with libwebp. The transforms and prefix-code reading follow the
// spec and libwebp's src/dec/vp8l_dec.c and src/dsp/lossless.c (BSD-3-Clause;
// see NOTICE).
package vp8ldec

import (
	"context"
	"encoding/binary"
	"errors"
)

var (
	errSig       = errors.New("webp: invalid VP8L header")
	errTruncated = errors.New("webp: truncated VP8L stream")
	errBitstream = errors.New("webp: invalid VP8L bitstream")
)

// MaxSide is the most pixels a side may have.
const MaxSide = 16383

// Header reads the image size and the alpha-is-used flag.
func Header(data []byte) (w, h int, alpha bool, err error) {
	if len(data) < 5 || data[0] != 0x2f {
		return 0, 0, false, errSig
	}
	v := binary.LittleEndian.Uint32(data[1:5])
	if v>>29 != 0 {
		return 0, 0, false, errSig // version
	}
	return int(v&0x3fff) + 1, int(v>>14&0x3fff) + 1, v>>28&1 != 0, nil
}

// Decode decodes data into dst, 4*w*h bytes of R, G, B, A (not
// premultiplied), w and h from Header. grow is asked, before each
// allocation, for its bytes; its error ends the decode. ctx is checked
// once a row.
func Decode(ctx context.Context, data, dst []byte, grow func(int64) error) error {
	w, h, _, err := Header(data)
	if err != nil {
		return err
	}
	d := &decoder{br: bitReader{b: data[5:]}, ctx: ctx, grow: grow}
	if err := d.decode(dst[:4*w*h], w, h); err != nil {
		return err
	}
	if d.br.over {
		return errTruncated
	}
	for i := 0; i < len(dst); i += 4 {
		dst[i], dst[i+2] = dst[i+2], dst[i] // BGRA to RGBA
	}
	return nil
}

// DecodeAlpha decodes the headerless VP8L stream of an ALPH chunk, w x h
// pixels, into alpha (w*h bytes, the green channel of each pixel). It holds
// the last ringPx pixels the stream can still copy from, not the image, and
// undoes the transforms a row at a time as the rows complete.
func DecodeAlpha(ctx context.Context, data []byte, w, h int, alpha []byte, grow func(int64) error) error {
	d := &decoder{br: bitReader{b: data}, ctx: ctx, grow: grow}
	ts, xs, err := d.readTransforms(w, h)
	if err != nil {
		return err
	}
	ring, err := d.alloc(4 * min(xs*h, ringPx))
	if err != nil {
		return err
	}
	rows, err := d.alloc(8 * w) // the row being converted and the last one the predictor made
	if err != nil {
		return err
	}
	cur, prev := rows[:4*w], rows[4*w:]
	d.sink = func(y int) error {
		for x := 0; x < xs; x++ {
			put(cur, x, get(ring, slot(len(ring)/4, y*xs+x)))
		}
		d.inverseRow(cur, prev, ts, y, w)
		for x := 0; x < w; x++ {
			alpha[y*w+x] = cur[4*x+1] // green, in BGRA order
		}
		return nil
	}
	if err := d.decodeImage(ring, xs, h, true); err != nil {
		return err
	}
	if d.br.over {
		return errTruncated
	}
	return nil
}

// ringPx is the farthest a copy reaches back (distance code 40 with all its
// extra bits is 1<<20 pixels) plus the pixels a run can write before its
// rows are converted (a run of 4096 and a row of 16383).
const ringPx = 1<<20 + 1<<15

type decoder struct {
	br   bitReader
	ctx  context.Context
	grow func(int64) error
	sink func(y int) error // DecodeAlpha's: row y of the main image is complete
}

// slot is where pixel pos lives in an array of n pixels used as a ring.
func slot(n, pos int) int {
	if pos >= n {
		return pos % n
	}
	return pos
}

func get(a []byte, i int) uint32    { return binary.LittleEndian.Uint32(a[4*i:]) }
func put(a []byte, i int, v uint32) { binary.LittleEndian.PutUint32(a[4*i:], v) }
func ceil(n int, bits uint) int     { return (n + 1<<bits - 1) >> bits }
func (d *decoder) row() error       { return d.ctx.Err() }
func (d *decoder) alloc(n int) ([]byte, error) {
	if err := d.grow(int64(n)); err != nil {
		return nil, err
	}
	return make([]byte, n), nil
}

type transform struct {
	kind int
	bits uint
	xs   int // image width this transform sees
	data []byte
}

const (
	tPredictor = iota
	tCross
	tSubGreen
	tIndex
)

func (d *decoder) decode(a []byte, w, h int) error {
	ts, xs, err := d.readTransforms(w, h)
	if err != nil {
		return err
	}
	if err := d.decodeImage(a[:4*xs*h], xs, h, true); err != nil {
		return err
	}
	for i := len(ts) - 1; i >= 0; i-- {
		if err := d.inverse(a, &ts[i], h, w); err != nil {
			return err
		}
	}
	return nil
}

// readTransforms reads the transforms and their images, and returns them
// with the width the main image has once they are applied.
func (d *decoder) readTransforms(w, h int) ([]transform, int, error) {
	var ts []transform
	var seen [4]bool
	xs := w
	for d.br.bit() {
		t := transform{kind: int(d.br.bits(2)), xs: xs}
		if seen[t.kind] {
			return nil, 0, errBitstream
		}
		seen[t.kind] = true
		switch t.kind {
		case tPredictor, tCross:
			t.bits = uint(d.br.bits(3)) + 2
			tw, th := ceil(xs, t.bits), ceil(h, t.bits)
			var err error
			if t.data, err = d.alloc(4 * tw * th); err != nil {
				return nil, 0, err
			}
			if err := d.decodeImage(t.data, tw, th, false); err != nil {
				return nil, 0, err
			}
		case tIndex:
			n := int(d.br.bits(8)) + 1
			switch {
			case n <= 2:
				t.bits = 3
			case n <= 4:
				t.bits = 2
			case n <= 16:
				t.bits = 1
			}
			// The table pads to every index its packed pixels can hold.
			var err error
			if t.data, err = d.alloc(4 << (8 >> t.bits)); err != nil {
				return nil, 0, err
			}
			if err := d.decodeImage(t.data[:4*n], n, 1, false); err != nil {
				return nil, 0, err
			}
			for i := 4; i < 4*n; i++ {
				t.data[i] += t.data[i-4]
			}
			xs = ceil(xs, t.bits)
		}
		if d.br.over {
			return nil, 0, errTruncated
		}
		ts = append(ts, t)
	}
	return ts, xs, nil
}

var kCLOrder = [19]uint8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// readCode reads one prefix code over an alphabet of len(lens) symbols.
func (d *decoder) readCode(c *code, lens []uint8, keep bool) error {
	br := &d.br
	clear(lens)
	if br.bit() { // simple
		n := br.bits(1) + 1
		first := uint(1)
		if br.bit() {
			first = 8
		}
		if s := int(br.bits(first)); s < len(lens) {
			lens[s] = 1
		}
		if n == 2 {
			if s := int(br.bits(8)); s < len(lens) {
				lens[s] = 1
			}
		}
	} else {
		var cl [19]uint8
		for i, n := 0, int(br.bits(4))+4; i < n; i++ {
			cl[kCLOrder[i]] = uint8(br.bits(3))
		}
		var clc code
		if err := clc.build(cl[:], func(int64) error { return nil }, true); err != nil {
			return err
		}
		maxSym := len(lens)
		if br.bit() {
			nb := 2 + 2*uint(br.bits(3))
			maxSym = 2 + int(br.bits(nb))
			if maxSym > len(lens) {
				return errBitstream
			}
		}
		prev := uint8(8)
		for s := 0; s < len(lens) && maxSym > 0; maxSym-- {
			if br.over {
				return errTruncated
			}
			v := clc.read(br)
			if v < 16 {
				lens[s] = uint8(v)
				s++
				if v != 0 {
					prev = uint8(v)
				}
				continue
			}
			var rep int
			val := uint8(0)
			switch v {
			case 16:
				rep, val = 3+int(br.bits(2)), prev
			case 17:
				rep = 3 + int(br.bits(3))
			default:
				rep = 11 + int(br.bits(7))
			}
			if s+rep > len(lens) {
				return errBitstream
			}
			for ; rep > 0; rep-- {
				lens[s] = val
				s++
			}
		}
	}
	if br.over {
		return errTruncated
	}
	return c.build(lens, d.grow, keep)
}

// groupAlpha is the alphabet size of each of a group's five codes, in
// stream order: green with lengths and cache, red, blue, alpha, distance.
func groupAlpha(cacheBits uint) [5]int {
	g := 256 + 24
	if cacheBits > 0 {
		g += 1 << cacheBits
	}
	return [5]int{g, 256, 256, 256, 40}
}

// decodeImage reads one entropy-coded image of xs x ys pixels into a.
// Only the main image has a meta prefix image.
func (d *decoder) decodeImage(a []byte, xs, ys int, main bool) error {
	br := &d.br
	var cacheBits uint
	if br.bit() {
		if cacheBits = uint(br.bits(4)); cacheBits < 1 || cacheBits > 11 {
			return errBitstream
		}
	}
	var meta []byte
	var mw int
	var pbits uint
	var slotOf []int32 // group number to slot in groups, -1 for a group no pixel uses
	nGroups, nSlots := 1, 1
	if main && br.bit() {
		pbits = uint(br.bits(3)) + 2
		mw = ceil(xs, pbits)
		mh := ceil(ys, pbits)
		var err error
		if meta, err = d.alloc(4 * mw * mh); err != nil {
			return err
		}
		if err := d.decodeImage(meta, mw, mh, false); err != nil {
			return err
		}
		for i := 0; i < mw*mh; i++ {
			nGroups = max(nGroups, int(get(meta, i)>>8&0xffff)+1)
		}
		if err := d.grow(4 * int64(nGroups)); err != nil {
			return err
		}
		slotOf = make([]int32, nGroups)
		for i := range slotOf {
			slotOf[i] = -1
		}
		nSlots = 0
		for i := 0; i < mw*mh; i++ {
			g := get(meta, i) >> 8 & 0xffff
			if slotOf[g] < 0 {
				slotOf[g] = int32(nSlots)
				nSlots++
			}
			put(meta, i, uint32(slotOf[g]))
		}
	}
	if err := d.grow(5 * codeBytes * int64(nSlots)); err != nil {
		return err
	}
	groups := make([]code, 5*nSlots)
	alpha := groupAlpha(cacheBits)
	lens := make([]uint8, alpha[0])
	var skip code
	for g := 0; g < nGroups; g++ {
		if err := d.ctx.Err(); err != nil {
			return err
		}
		slot := 0
		if slotOf != nil {
			slot = int(slotOf[g])
		}
		for k := 0; k < 5; k++ {
			c := &skip
			if slot >= 0 {
				c = &groups[5*slot+k]
			}
			if err := d.readCode(c, lens[:alpha[k]], slot >= 0); err != nil {
				return err
			}
		}
	}
	var cache []uint32
	if cacheBits > 0 {
		if err := d.grow(4 << cacheBits); err != nil {
			return err
		}
		cache = make([]uint32, 1<<cacheBits)
	}
	cshift := 32 - cacheBits

	total := xs * ys
	mask := 1<<pbits - 1
	x, y, curY := 0, 0, 0
	ringN := len(a) / 4 // pixels a holds: all of them but for DecodeAlpha's ring
	var g []code
	setGroup := func() {
		if meta != nil {
			slot := int(get(meta, (y>>pbits)*mw+x>>pbits))
			g = groups[5*slot : 5*slot+5]
		} else {
			g = groups
		}
	}
	setGroup()
	for pos := 0; pos < total; {
		if br.over {
			return errTruncated
		}
		for ; curY < y; curY++ {
			if err := d.ctx.Err(); err != nil {
				return err
			}
			if main && d.sink != nil {
				if err := d.sink(curY); err != nil {
					return err
				}
			}
		}
		s := g[0].read(br)
		switch {
		case s < 256:
			r := uint32(g[1].read(br))
			b := uint32(g[2].read(br))
			al := uint32(g[3].read(br))
			px := al<<24 | r<<16 | uint32(s)<<8 | b
			put(a, slot(ringN, pos), px)
			if cache != nil {
				cache[(px*0x1e35a7bd)>>cshift] = px
			}
			pos++
			if x++; x == xs {
				x, y = 0, y+1
			}
			if x&mask == 0 && meta != nil && pos < total {
				setGroup()
			}
		case s < 256+24:
			n := prefixValue(br, s-256)
			dist := planeDist(prefixValue(br, g[4].read(br)), xs)
			if pos < dist || total-pos < n {
				return errBitstream
			}
			for i := 0; i < n; i++ {
				px := get(a, slot(ringN, pos-dist))
				put(a, slot(ringN, pos), px)
				if cache != nil {
					cache[(px*0x1e35a7bd)>>cshift] = px
				}
				pos++
			}
			x += n
			for x >= xs {
				x -= xs
				y++
			}
			if pos < total && meta != nil {
				setGroup()
			}
		case s < 256+24+len(cache):
			put(a, slot(ringN, pos), cache[s-256-24])
			pos++
			if x++; x == xs {
				x, y = 0, y+1
			}
			if x&mask == 0 && meta != nil && pos < total {
				setGroup()
			}
		default:
			return errBitstream
		}
	}
	for ; main && d.sink != nil && curY < ys; curY++ {
		if err := d.sink(curY); err != nil {
			return err
		}
	}
	return nil
}

// prefixValue is a length or distance from its prefix symbol, plus its
// extra bits.
func prefixValue(br *bitReader, sym int) int {
	if sym < 4 {
		return sym + 1
	}
	nb := uint(sym-2) >> 1
	off := (2 + sym&1) << nb
	return off + int(br.bits(nb)) + 1
}

// planeDist is the pixel distance of distance code c in an image xs wide.
func planeDist(c, xs int) int {
	if c > 120 {
		return c - 120
	}
	e := distMap[c-1]
	return max(1, int(e[0])+int(e[1])*xs)
}

// distMap are the (x, y) offsets of distance codes 1 to 120, spec 5.2.2.
var distMap = [120][2]int8{
	{0, 1}, {1, 0}, {1, 1}, {-1, 1}, {0, 2}, {2, 0}, {1, 2}, {-1, 2},
	{2, 1}, {-2, 1}, {2, 2}, {-2, 2}, {0, 3}, {3, 0}, {1, 3}, {-1, 3},
	{3, 1}, {-3, 1}, {2, 3}, {-2, 3}, {3, 2}, {-3, 2}, {0, 4}, {4, 0},
	{1, 4}, {-1, 4}, {4, 1}, {-4, 1}, {3, 3}, {-3, 3}, {2, 4}, {-2, 4},
	{4, 2}, {-4, 2}, {0, 5}, {3, 4}, {-3, 4}, {4, 3}, {-4, 3}, {5, 0},
	{1, 5}, {-1, 5}, {5, 1}, {-5, 1}, {2, 5}, {-2, 5}, {5, 2}, {-5, 2},
	{4, 4}, {-4, 4}, {3, 5}, {-3, 5}, {5, 3}, {-5, 3}, {0, 6}, {6, 0},
	{1, 6}, {-1, 6}, {6, 1}, {-6, 1}, {2, 6}, {-2, 6}, {6, 2}, {-6, 2},
	{4, 5}, {-4, 5}, {5, 4}, {-5, 4}, {3, 6}, {-3, 6}, {6, 3}, {-6, 3},
	{0, 7}, {7, 0}, {1, 7}, {-1, 7}, {5, 5}, {-5, 5}, {7, 1}, {-7, 1},
	{4, 6}, {-4, 6}, {6, 4}, {-6, 4}, {2, 7}, {-2, 7}, {7, 2}, {-7, 2},
	{3, 7}, {-3, 7}, {7, 3}, {-7, 3}, {5, 6}, {-5, 6}, {6, 5}, {-6, 5},
	{8, 0}, {4, 7}, {-4, 7}, {7, 4}, {-7, 4}, {8, 1}, {8, 2}, {6, 6},
	{-6, 6}, {8, 3}, {5, 7}, {-5, 7}, {7, 5}, {-7, 5}, {8, 4}, {6, 7},
	{-6, 7}, {7, 6}, {-7, 6}, {8, 5}, {7, 7}, {-7, 7}, {8, 6}, {8, 7},
}
