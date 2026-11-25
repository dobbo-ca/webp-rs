package vp8ldec

import "errors"

var errCode = errors.New("webp: invalid prefix code")

// codeBytes bounds the memory of a code besides its symbols (a test checks it).
const codeBytes = 576

// code is a canonical prefix code, packed MSB first like deflate's.
type code struct {
	single bool
	sym    uint16
	root   [256]uint16 // sym<<4 | len for codes of at most 8 bits, else 0
	count  [16]uint16
	syms   []uint16 // by length, then value
}

// build makes the code of lens. One used symbol codes in zero bits; any
// other code must be complete; with keep false it is only checked. grow is asked for the symbol table's bytes
// before it is allocated.
func (c *code) build(lens []uint8, grow func(int64) error, keep bool) error {
	*c = code{}
	for _, l := range lens {
		c.count[l]++
	}
	n := len(lens) - int(c.count[0])
	c.count[0] = 0
	if n == 0 {
		return errCode
	}
	if n == 1 {
		for s, l := range lens {
			if l != 0 {
				c.single, c.sym = true, uint16(s)
			}
		}
		return nil
	}
	left := 1
	for l := 1; l < 16; l++ {
		left = left<<1 - int(c.count[l])
		if left < 0 {
			return errCode
		}
	}
	if left != 0 {
		return errCode
	}
	if !keep {
		return nil // only checked
	}
	if err := grow(2 * int64(n)); err != nil {
		return err
	}
	c.syms = make([]uint16, n)
	var off [16]uint16
	for l := 1; l < 15; l++ {
		off[l+1] = off[l] + c.count[l]
	}
	var next [16]uint32 // next code of each length
	for l, cd := 1, uint32(0); l < 16; l++ {
		cd = (cd + uint32(c.count[l-1])) << 1
		next[l] = cd
	}
	for s, l := range lens {
		if l == 0 {
			continue
		}
		c.syms[off[l]] = uint16(s)
		off[l]++
		cd := next[l]
		next[l]++
		if l <= 8 {
			r := reverse(cd, uint(l))
			for j := r; j < 256; j += 1 << l {
				c.root[j] = uint16(s)<<4 | uint16(l)
			}
		}
	}
	return nil
}

func reverse(v uint32, n uint) uint32 {
	var r uint32
	for ; n > 0; n-- {
		r = r<<1 | v&1
		v >>= 1
	}
	return r
}

// read decodes one symbol; a stream that ends sets br.over.
func (c *code) read(r *bitReader) int {
	if c.single {
		return int(c.sym)
	}
	if r.n < 15 {
		r.refill()
	}
	v := uint32(r.val)
	if e := c.root[v&255]; e != 0 {
		r.skip(uint(e & 15))
		return int(e >> 4)
	}
	cd, first, index := 0, 0, 0
	for l := 1; l < 16; l++ {
		cd |= int(v>>(l-1)) & 1
		cnt := int(c.count[l])
		if cd-cnt < first {
			r.skip(uint(l))
			return int(c.syms[index+cd-first])
		}
		index += cnt
		first = (first + cnt) << 1
		cd <<= 1
	}
	r.over = true // not reached: the code is complete
	return 0
}
