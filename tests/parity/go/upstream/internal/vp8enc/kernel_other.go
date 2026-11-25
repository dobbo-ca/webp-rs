//go:build !(amd64 || arm64) || purego

package vp8enc

// residualsY is residual over the MB's 16 luma blocks.
func (e *Encoder) residualsY(dq *[16][16]int16, src []uint8, ys int) {
	for n := range dq {
		y, x := 1+4*(n/4), 8+4*(n%4)
		e.residual(&dq[n], src[(y-1)*ys+x-8:], ys, y, x)
	}
}

// residualsUV is residual over the MB's 8 chroma blocks.
func (e *Encoder) residualsUV(dq *[8][16]int16, u, v []uint8, cs int) {
	for c := range dq {
		src, bx := u, 8
		if c >= 4 {
			src, bx = v, 24
		}
		oy, ox := 4*((c&3)>>1), 4*(c&1)
		e.residual(&dq[c], src[oy*cs+ox:], cs, 18+oy, bx+ox)
	}
}

// quantizeN is quantize over in; bit n of nz is set when block n has a
// nonzero level.
func (m *quantMatrix) quantizeN(in, out [][16]int16) (nz int) {
	for n := range in {
		if m.quantize(&in[n], &out[n]) >= 0 {
			nz |= 1 << n
		}
	}
	return nz
}

// idctAddsY is idctAdd over the MB's 16 luma blocks.
func (z *ybrBuf) idctAddsY(dq *[16][16]int16) {
	for n := range dq {
		z.idctAdd(&dq[n], 1+4*(n/4), 8+4*(n%4))
	}
}

// idctAddsUV is idctAdd over the MB's 8 chroma blocks.
func (z *ybrBuf) idctAddsUV(dq *[8][16]int16) {
	for c := range dq {
		oy, ox := 4*((c&3)>>1), 4*(c&1)
		z.idctAdd(&dq[c], 18+oy, 8+16*(c>>2)+ox)
	}
}

func (s *lwState) costLuma16(lv *[25][16]int16, l, t nzCtx) int { return s.costLuma16Go(lv, l, t) }

func (s *lwState) costUV(lv *[25][16]int16, l, t nzCtx) int { return s.costUVGo(lv, l, t) }

func costLuma16Nat(lv *[25][16]int16, l, t nzCtx) int { return costLuma(true, lv, &l, &t) }

func costUVNat(lv *[25][16]int16, l, t nzCtx) int { return costChroma(lv, &l, &t) }

func cost4(ctx, last int, lv *[16]int16) int {
	c, _ := costLast(3, ctx, 0, last, lv)
	return c
}

// costGridOK is false so pickI4 inlines resCostLast.
func costGridOK() bool { return false }

func (s *lwState) resCost4(ctx, last int, lv *[16]int16) int {
	c, _ := s.resCostLast(3, ctx, 0, last, lv)
	return c
}

// chromaRow leaves the chroma loop to Go.
func chromaRow(cb, cr, p0, p1 []uint8) (n int, and uint8) { return 0, 255 }
