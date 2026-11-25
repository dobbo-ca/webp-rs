package vp8enc

import "math/bits"

// Ported from libwebp src/utils/bit_writer_utils.c, BSD-3-Clause.

// VP8 boolean entropy encoder.
type boolEncoder struct {
	buf    []byte
	rng    int32 // range-1, 127..254 between calls
	value  int64
	run    int // pending 0xff bytes
	nbBits int32
}

func (e *boolEncoder) reset() {
	e.buf = e.buf[:0]
	e.rng = 254
	e.value = 0
	e.run = 0
	e.nbBits = -8
}

func (e *boolEncoder) putBit(bit bool, prob uint8) bool {
	split := (e.rng * int32(prob)) >> 8
	var m int32
	if bit {
		m = -1
	}
	// Masks avoid mispredicts on random bits.
	e.value += int64((split + 1) & m)
	r := split + (e.rng-2*split-1)&m + 1
	// sh is 0 when r >= 128.
	sh := int32(bits.LeadingZeros32(uint32(r))) - 24
	e.rng = r<<sh - 1
	e.value <<= sh
	e.nbBits += sh
	if e.nbBits > 0 {
		e.flush()
	}
	return bit
}

func (e *boolEncoder) putLiteral(v uint32, n int) {
	for i := n - 1; i >= 0; i-- {
		e.putBit(v>>uint(i)&1 == 1, 128)
	}
}

func (e *boolEncoder) flush() {
	s := 8 + e.nbBits
	b := e.value >> uint(s)
	e.value -= b << uint(s)
	e.nbBits -= 8
	if b&0xff == 0xff {
		e.run++
		return
	}
	fill := byte(0xff)
	if b&0x100 != 0 {
		fill = 0
		if len(e.buf) > 0 {
			e.buf[len(e.buf)-1]++
		}
	}
	for ; e.run > 0; e.run-- {
		e.buf = append(e.buf, fill)
	}
	e.buf = append(e.buf, byte(b))
}

func (e *boolEncoder) finish() []byte {
	e.putLiteral(0, int(9-e.nbBits))
	e.nbBits = 0
	e.flush()
	return e.buf
}
