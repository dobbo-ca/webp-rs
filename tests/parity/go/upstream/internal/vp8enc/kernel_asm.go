//go:build (arm64 || amd64) && !purego

package vp8enc

// quantAsm is quantSSEGo in NEON or SSE2.
//
//go:noescape
func quantAsm(m *quantMatrix, in, out *[16]int16) (last, d int)

func (m *quantMatrix) quantize(in, out *[16]int16) int {
	last, _ := quantAsm(m, in, out)
	return last
}

func (m *quantMatrix) quantSSE(in, out *[16]int16) (last, d int) {
	return quantAsm(m, in, out)
}

// i16DCAsm is the purego i16DC in NEON or SSE2; NEON leaves dq lane 0
// unzeroed, quantI16AC ignores it.
//
//go:noescape
func i16DCAsm(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) (nz bool)

func i16DC(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) bool {
	return i16DCAsm(m, dq, lv, dc)
}

// sse4x10Asm is sse4x10Go in NEON or SSE2.
//
//go:noescape
func sse4x10Asm(d *[10]int32, pr *[10][4]uint32, src *uint8, stride int)

func sse4x10(d *[10]int32, pr *[10][4]uint32, src []uint8, stride int) {
	_ = src[3*stride+3]
	sse4x10Asm(d, pr, &src[0], stride)
}

// sseAsm is predSSEGo in NEON or SSE2, w is 16 or 8.
//
//go:noescape
func sseAsm(src *uint8, stride int, pred *uint8, w, h int) int

func (z *ybrBuf) predSSE(src []uint8, stride, y, x, w, h int) int {
	_ = src[(h-1)*stride+w-1]
	_ = z[y+h-1][x+w-1]
	return sseAsm(&src[0], stride, &z[y][x], w, h)
}

// edgeSSEAsm is edgeSSEGo in NEON (UDOT with useDot) or SSE2, n is
// 16 or 8, edges as dcEdges.
//
//go:noescape
func edgeSSEAsm(d *[4]int, src *uint8, stride int, p *uint8, n, edges int)

func (z *ybrBuf) edgeSSE(d *[4]int, src []uint8, stride, n, y, x, mbx, mby int) {
	_ = src[(n-1)*stride+n-1]
	_ = z[y-1][x-1]
	_ = z[y+n-1][x+n-1]
	edgeSSEAsm(d, &src[0], stride, &z[y][x], n, dcEdges(mbx, mby))
}

// dcEdges is dcAvg's edges: bit 0 left, bit 1 top.
func dcEdges(mbx, mby int) int {
	return int(b2u(mbx != 0)) | int(b2u(mby != 0))<<1
}

// predTMAsm is predTMGo in NEON or SSE2, n is 16 or 8.
//
//go:noescape
func predTMAsm(dst *uint8, n int)

func (z *ybrBuf) predTM(n, y, x int) {
	_ = z[y-1][x-1]
	_ = z[y+n-1][x+n-1]
	predTMAsm(&z[y][x], n)
}

// predDCAsm and predHEAsm are predMBGo's DC and HE cases in
// NEON or SSE2, n is 16 or 8, edges as dcEdges.
//
//go:noescape
func predDCAsm(dst *uint8, n, edges int)

//go:noescape
func predHEAsm(dst *uint8, n int)

func (z *ybrBuf) predMB(n, y, x int, mode uint8, mbx, mby int) {
	switch mode {
	case predDC:
		_ = z[y-1][x-1]
		_ = z[y+n-1][x+n-1]
		predDCAsm(&z[y][x], n, dcEdges(mbx, mby))
	case predHE:
		_ = z[y][x-1]
		_ = z[y+n-1][x+n-1]
		predHEAsm(&z[y][x], n)
	default:
		z.predMBGo(n, y, x, mode, mbx, mby)
	}
}

// pred4AllAsm is pred4AllGo in NEON or SSE4.1, p = &z[y][x].
//
//go:noescape
func pred4AllAsm(pr *[10][4]uint32, p *uint8)

// ttMBAsm is ttMBGo in NEON or AVX2.
//
//go:noescape
func ttMBAsm(src *uint8, stride int, st *[16]int32)

// lumaAsm is lumaRowGo in NEON or SSE2, n%16 == 0.
//
//go:noescape
func lumaAsm(dst, src *uint8, n int)

func lumaRow(dst, src []uint8) {
	n := len(dst) &^ 15
	if n > 0 {
		_ = src[4*n-1]
		lumaAsm(&dst[0], &src[0], n)
	}
	lumaRowGo(dst[n:], src[4*n:])
}

// idctAddAsm is idctAddGo in NEON or SSE2, row stride 32.
//
//go:noescape
func idctAddAsm(c *[16]int16, dst *uint8)

func (z *ybrBuf) idctAdd(c *[16]int16, y, x int) {
	if *c == [16]int16{} {
		return
	}
	_ = z[y+3][x+3]
	idctAddAsm(c, &z[y][x])
}

// residualAsm is residualGo in NEON or SSE2, pred row stride 32.
//
//go:noescape
func residualAsm(pred *uint8, src *uint8, stride int, out *[16]int16)

// residual fills blk with the fdct of src minus the prediction at (y, x).
func (e *Encoder) residual(blk *[16]int16, src []uint8, stride, y, x int) {
	_ = src[3*stride+3]
	_ = e.z[y+3][x+3]
	residualAsm(&e.z[y][x], &src[0], stride, blk)
}

// levelCostsAsm is levelCostsGo in scalar asm.
//
//go:noescape
func levelCostsAsm(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int)

func levelCosts(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int) {
	return levelCostsAsm(t, lv, first, last, ctx)
}

// residualQuantAsm is residualQuantGo in NEON or SSE2.
//
//go:noescape
func residualQuantAsm(m *quantMatrix, pr *[4]uint32, src *uint8, stride int, in, out *[16]int16) (last, d int)

func (e *Encoder) residualQuant(blk, lv *[16]int16, pr *[4]uint32, src []uint8, stride, y, x int) (last, d int) {
	_ = src[3*stride+3]
	return residualQuantAsm(&e.q.y1, pr, &src[0], stride, blk, lv)
}

// recMetricsAsm is recMetricsGo in NEON or SSE2.
//
//go:noescape
func recMetricsAsm(c *[16]int16, dst *uint8, src *uint8, stride int) (sse, tt int)

// recMetrics adds the inverse DCT of c at (y, x) and returns the SSE
// of src against the result and its TTransform.
func (z *ybrBuf) recMetrics(c *[16]int16, y, x int, src []uint8, ss int) (sse, tt int) {
	_ = src[3*ss+3]
	return recMetricsAsm(c, &z[y][x], &src[0], ss)
}

// i4TrialsAsm is i4Trials for all ten modes in NEON or SSE2.
//
//go:noescape
func i4TrialsAsm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, dq *[16]int16)

// binStatsAsm is binStatsGo in AVX2 or NEON.
//
//go:noescape
func binStatsAsm(b *uint8, n int) (last, c0, c1, c2, c3 int)

// fastAlpha is a var so tests can force the scatter.
var fastAlpha = true

// binsAlpha is hist4.alpha of the bins b. When bins 0-3 hold a count at
// least the size of everything above bin 3, that count is the largest.
func binsAlpha(b []uint8) int {
	if fastAlpha {
		last, c0, c1, c2, c3 := binStatsAsm(&b[0], len(b))
		if m := max(c0, c1, c2, c3); m >= len(b)-(c0+c1+c2+c3) {
			return 510 * last / m // m >= len(b)/5, so above 1
		}
	}
	var d hist4
	for i := 0; i < len(b); i += 4 {
		d[0][b[i]&31]++
		d[1][b[i+1]&31]++
		d[2][b[i+2]&31]++
		d[3][b[i+3]&31]++
	}
	return d.alpha()
}
