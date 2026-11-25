package vp8ldec

import "encoding/binary"

// bitReader reads the LSB-first bit stream. Reading past the end yields
// zero bits and sets over.
type bitReader struct {
	b    []byte
	pos  int
	val  uint64
	n    uint
	over bool
}

func (r *bitReader) refill() {
	if r.pos+8 <= len(r.b) {
		r.val |= binary.LittleEndian.Uint64(r.b[r.pos:]) << r.n
		k := (63 - r.n) >> 3
		r.pos += int(k)
		r.n += k << 3
		return
	}
	for r.n <= 56 && r.pos < len(r.b) {
		r.val |= uint64(r.b[r.pos]) << r.n
		r.pos++
		r.n += 8
	}
}

// skip drops k bits.
func (r *bitReader) skip(k uint) {
	if k > r.n {
		r.over, r.val, r.n = true, 0, 0
		return
	}
	r.val >>= k
	r.n -= k
}

func (r *bitReader) bits(k uint) uint32 {
	if k == 0 {
		return 0
	}
	if r.n < k {
		r.refill()
	}
	v := uint32(r.val & (1<<k - 1))
	r.skip(k)
	return v
}

func (r *bitReader) bit() bool { return r.bits(1) != 0 }
