package vp8dec

import (
	"encoding/binary"
	"math/bits"
)

// boolDec is the boolean entropy decoder of RFC 6386 7, in libwebp's
// form (src/utils/bit_reader_utils.c, BSD-3-Clause): rng is the range
// minus one and eof is set when a bit is needed past the end.
type boolDec struct {
	b    []byte
	pos  int
	val  uint64
	rng  uint32
	bits int
	eof  bool
}

func (d *boolDec) init(b []byte) {
	*d = boolDec{b: b, rng: 254, bits: -8}
	d.load()
}

func (d *boolDec) load() {
	switch {
	case d.pos+8 <= len(d.b):
		d.val = binary.BigEndian.Uint64(d.b[d.pos:])>>8 | d.val<<56
		d.pos += 7
		d.bits += 56
	case d.pos < len(d.b):
		d.val = uint64(d.b[d.pos]) | d.val<<8
		d.pos++
		d.bits += 8
	case !d.eof:
		d.val <<= 8
		d.bits += 8
		d.eof = true
	default:
		d.bits = 0 // avoid undefined shifts on garbage input
	}
}

// bit decodes one bit with probability prob/256 of being 0.
func (d *boolDec) bit(prob uint8) int {
	rng := d.rng
	if d.bits < 0 {
		d.load()
	}
	pos := uint(d.bits)
	split := rng * uint32(prob) >> 8
	var b int
	if uint32(d.val>>pos) > split {
		rng -= split
		d.val -= uint64(split+1) << pos
		b = 1
	} else {
		rng = split + 1
	}
	shift := 7 ^ (bits.Len32(rng) - 1)
	d.bits -= shift
	d.rng = rng<<uint(shift) - 1
	return b
}

func (d *boolDec) flag() bool { return d.bit(0x80) != 0 }

// value reads n bits, most significant first.
func (d *boolDec) value(n int) int {
	v := 0
	for ; n > 0; n-- {
		v = v<<1 | d.bit(0x80)
	}
	return v
}

// signed reads n bits and a sign.
func (d *boolDec) signed(n int) int {
	v := d.value(n)
	if d.flag() {
		return -v
	}
	return v
}
