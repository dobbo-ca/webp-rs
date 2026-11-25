//go:build !(arm64 || amd64) || purego

package vp8enc

func (m *quantMatrix) quantize(in, out *[16]int16) int { return m.quantizeGo(in, out) }

func (m *quantMatrix) quantSSE(in, out *[16]int16) (last, d int) {
	return m.quantSSEGo(in, out)
}

// i16DC is lwI16's DC path.
func i16DC(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) bool {
	for n := range dq {
		dc[n] = dq[n][0]
		dq[n][0] = 0
	}
	fwhtPlain(dc)
	nz := m.quantize(dc, lv) >= 0
	*dc = iwht(dc)
	return nz
}

func sse4x10(d *[10]int32, pr *[10][4]uint32, src []uint8, stride int) {
	sse4x10Go(d, pr, src, stride)
}

func (z *ybrBuf) predSSE(src []uint8, stride, y, x, w, h int) int {
	return z.predSSEGo(src, stride, y, x, w, h)
}

func (z *ybrBuf) edgeSSE(d *[4]int, src []uint8, stride, n, y, x, mbx, mby int) {
	z.edgeSSEGo(d, src, stride, n, y, x, mbx, mby)
}

func (z *ybrBuf) predTM(n, y, x int) { z.predTMGo(n, y, x) }

func (z *ybrBuf) predMB(n, y, x int, mode uint8, mbx, mby int) {
	z.predMBGo(n, y, x, mode, mbx, mby)
}

func (z *ybrBuf) pred4All(pr *[10][4]uint32, y, x int) { z.pred4AllGo(pr, y, x) }

func ttMB(src []uint8, ss int, st *[16]int32) { ttMBGo(src, ss, st) }

func lumaRow(dst, src []uint8) { lumaRowGo(dst, src) }

// recMetrics16 is recMetrics over the MB's 16 luma blocks, summing SSE
// and |tt - st[n]| >> 5.
func (z *ybrBuf) recMetrics16(c *[16][16]int16, src []uint8, ss int, st *[16]int32) (sse, sd int) {
	return z.recMetrics16Go(c, src, ss, st)
}

const resCostAsmOK = false

func (s *lwState) resCostL(typ, ctx, first, last int, lv *[16]int16) (int, uint8) {
	panic("no resCost asm")
}

// resCost is GetResidualCost_C: the cost of lv[first:] at ctx, and
// whether any level is nonzero.
func (s *lwState) resCost(typ, ctx, first int, lv *[16]int16) (int, uint8) {
	return s.resCostLast(typ, ctx, first, lastNZ(lv, first), lv)
}

func (z *ybrBuf) idctAdd(c *[16]int16, y, x int) { z.idctAddGo(c, y, x) }

func levelCosts(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int) {
	return levelCostsGo(t, lv, first, last, ctx)
}

func (e *Encoder) residual(blk *[16]int16, src []uint8, stride, y, x int) {
	e.residualGo(blk, src, stride, y, x)
}

func (e *Encoder) residualQuant(blk, lv *[16]int16, pr *[4]uint32, src []uint8, stride, y, x int) (last, d int) {
	e.z.put4(pr, y, x)
	e.residual(blk, src, stride, y, x)
	return e.q.y1.quantSSE(blk, lv)
}

// recMetrics adds the inverse DCT of c at (y, x) and returns the SSE
// of src against the result and its TTransform.
func (z *ybrBuf) recMetrics(c *[16]int16, y, x int, src []uint8, ss int) (sse, tt int) {
	return z.recMetricsGo(c, y, x, src, ss)
}

// i4Trials runs every i4 mode at (y, x): levels into lv, the
// reconstruction into rec, and last, SSE, TTransform and the nonzero AC
// count into res.
func (e *Encoder) i4Trials(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x int) {
	e.i4TrialsGo(pr, src, ss, lv, rec, res, y, x)
}

const useTrellisAsm = false

func trellisDPAsm(a *trellisArgs) int { panic("no trellis asm") }

// residualQuant3 leaves encodeMB's per-trial loop unchanged.
func (e *Encoder) residualQuant3(blk, lv *[3][16]int16, res *[3][2]int, pr *[10][4]uint32, try uint16, src []uint8, stride int) bool {
	return false
}

// alphaMB is the analysis alpha of the MB's 16 luma blocks.
func (e *Encoder) alphaMB(src []uint8, stride int) int {
	var d hist4
	e.collectMBGo(&d, src, stride)
	return d.alpha()
}

// alphaUV leaves the Go loop inline in analyze.
func (e *Encoder) alphaUV(u, v []uint8, stride int) (int, bool) { return 0, false }
