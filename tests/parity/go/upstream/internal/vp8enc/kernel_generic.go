//go:build !arm64 || purego

package vp8enc

// collectMBGo is collect over one MB's 16 luma blocks.
func (e *Encoder) collectMBGo(d *hist4, src []uint8, stride int) {
	for n := 0; n < 16; n++ {
		y, x := 1+4*(n/4), 8+4*(n%4)
		e.collect(d, src[(y-1)*stride+x-8:], stride, y, x)
	}
}

func (s *costTabs) trellis(m *quantMatrix, in, out *[16]int16, ctx0, typ, lambda int) bool {
	return s.trellisGo(m, in, out, ctx0, typ, lambda)
}

// quantY16 is i16Quant's untrellised AC loop: lane 0 of each block
// moves to dc and the coefficient SSE is returned.
func (e *Encoder) quantY16(dq *[16][16]int16, dc *[16]int16) int {
	d := 0
	for n := range dq {
		b := &dq[n]
		dc[n] = b[0]
		b[0] = 0
		_, k := e.q.y1.quantSSE(b, &e.lv[n])
		d += k
	}
	return d
}

func iwhtY(c *[16]int16) [16]int16 { return iwht(c) }

// quantI16AC is quantI16's AC levels: lv[:16] and dq with lane 0 = dc.
// Lane 0 must be 0 on input; i16DC zeroes it outside NEON.
func (q *quant) quantI16AC(dq *[16][16]int16, lv *[25][16]int16, dc *[16]int16) int {
	nz := q.y1.quantizeN(dq[:], lv[:16])
	for n := range dq {
		dq[n][0] = dc[n]
	}
	return nz
}

// quantizeNDC is quantizeN over in with lane 0 of block i replaced by
// dc[i] first.
func (m *quantMatrix) quantizeNDC(in, out [][16]int16, dc *[8]int16) int {
	for n := range in {
		in[n][0] = dc[n]
	}
	return m.quantizeN(in, out)
}
