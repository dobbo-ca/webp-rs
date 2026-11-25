//go:build !purego

package vp8enc

import "math/bits"

// residualNAsm is residualAsm over a rows x cols block grid.
//
//go:noescape
func residualNAsm(pred, src *uint8, stride int, out *[16]int16, rows, cols int)

// residualsY is residual over the MB's 16 luma blocks.
func (e *Encoder) residualsY(dq *[16][16]int16, src []uint8, ys int) {
	_ = src[15*ys+15]
	if useAVX2 {
		residualsYAsm(&e.z[1][8], &src[0], ys, dq)
		return
	}
	residualNAsm(&e.z[1][8], &src[0], ys, &dq[0], 4, 4)
}

// residualsYAsm and residualsUVAsm are residualsY and residualsUV in
// AVX2; gen/batch.py generates them.
//
//go:noescape
func residualsYAsm(pred, src *uint8, stride int, out *[16][16]int16)

//go:noescape
func residualsUVAsm(pred, u, v *uint8, stride int, out *[8][16]int16)

// residualsUV is residual over the MB's 8 chroma blocks.
func (e *Encoder) residualsUV(dq *[8][16]int16, u, v []uint8, cs int) {
	_ = u[7*cs+7]
	_ = v[7*cs+7]
	if useAVX2 {
		residualsUVAsm(&e.z[18][8], &u[0], &v[0], cs, dq)
		return
	}
	residualNAsm(&e.z[18][8], &u[0], cs, &dq[0], 2, 2)
	residualNAsm(&e.z[18][24], &v[0], cs, &dq[4], 2, 2)
}

// quantNAsm is quantAsm over n blocks, nonzero blocks as bits.
//
//go:noescape
func quantNAsm(m *quantMatrix, in, out *[16]int16, n int) (nz int)

// quantizeN is quantize over in; bit n of nz is set when block n has a
// nonzero level.
func (m *quantMatrix) quantizeN(in, out [][16]int16) (nz int) {
	_ = out[len(in)-1]
	if useAVX2 && len(in)%2 == 0 {
		return quantNAVX2Asm(m, &in[0], &out[0], len(in))
	}
	return quantNAsm(m, &in[0], &out[0], len(in))
}

// quantNAVX2Asm is quantNAsm in AVX2 for even n, two blocks per pass;
// gen/batch.py generates it.
//
//go:noescape
func quantNAVX2Asm(m *quantMatrix, in, out *[16]int16, n int) (nz int)

// idctAddNAsm is idctAddAsm over a rows x cols block grid.
//
//go:noescape
func idctAddNAsm(c *[16]int16, dst *uint8, rows, cols int)

// idctAddsUVAsm is idctAddsUV in AVX2, a plane per pass. It returns
// false, leaving dst unchanged, unless every level is in [-2048, 2047].
//
//go:noescape
func idctAddsUVAsm(dq *[8][16]int16, dst *uint8) (ok bool)

// idctAddsY is idctAdd over the MB's 16 luma blocks.
func (z *ybrBuf) idctAddsY(dq *[16][16]int16) { idctAddNAsm(&dq[0], &z[1][8], 4, 4) }

// idctAddsUV is idctAdd over the MB's 8 chroma blocks.
func (z *ybrBuf) idctAddsUV(dq *[8][16]int16) {
	if useAVX2 && idctAddsUVAsm(dq, &z[18][8]) {
		return
	}
	idctAddNAsm(&dq[0], &z[18][8], 2, 2)
	idctAddNAsm(&dq[4], &z[18][24], 2, 2)
}

// collectAsm writes bin() of the 16 fdct blocks of src minus pred.
//
//go:noescape
func collectAsm(bins *[256]uint8, src *uint8, stride int, pred *uint8)

// collectUVAsm is collectAsm over the 8 chroma blocks.
//
//go:noescape
func collectUVAsm(bins *[128]uint8, u, v *uint8, stride int, pred *uint8)

// alphaMB is the analysis alpha of the MB's 16 luma blocks.
func (e *Encoder) alphaMB(src []uint8, stride int) int {
	if !useAVX2 {
		var d hist4
		e.collectMBGo(&d, src, stride)
		return d.alpha()
	}
	_ = src[15*stride+15]
	var b [256]uint8
	collectAsm(&b, &src[0], stride, &e.z[1][8])
	return binsAlpha(b[:])
}

// alphaUV is alphaMB over the 8 chroma blocks; false without AVX2.
func (e *Encoder) alphaUV(u, v []uint8, stride int) (int, bool) {
	if !useAVX2 {
		return 0, false
	}
	_ = u[7*stride+7]
	_ = v[7*stride+7]
	var b [128]uint8
	collectUVAsm(&b, &u[0], &v[0], stride, &e.z[18][8])
	return binsAlpha(b[:]), true
}

// recMetrics16Asm is recMetrics16Go in AVX2; gen/recm.py generates it.
//
//go:noescape
func recMetrics16Asm(c *[16][16]int16, dst *uint8, src *uint8, stride int, st *[16]int32) (sse, sd int)

// recMetrics16 is recMetrics over the MB's 16 luma blocks, summing SSE
// and |tt - st[n]| >> 5.
func (z *ybrBuf) recMetrics16(c *[16][16]int16, src []uint8, ss int, st *[16]int32) (sse, sd int) {
	if !useAVX2 {
		return z.recMetrics16Go(c, src, ss, st)
	}
	_ = src[15*ss+15]
	return recMetrics16Asm(c, &z[1][8], &src[0], ss, st)
}

// resCostAsm is resCost in SSE2 and scalar asm, returning last.
//
//go:noescape
func resCostAsm(s *costTabs, typ, ctx, first int, lv *[16]int16) (c, last int)

func (s *lwState) resCost(typ, ctx, first int, lv *[16]int16) (int, uint8) {
	c, last := resCostAsm(&s.costTabs, typ, ctx, first, lv)
	return c, b2u(last >= 0)
}

// resCostL is resCostLast's asm path; it finds last again.
func (s *lwState) resCostL(typ, ctx, first, last int, lv *[16]int16) (int, uint8) {
	c, n := resCostAsm(&s.costTabs, typ, ctx, first, lv)
	return c, b2u(n >= 0)
}

// resCostLast finds last again; SSE2 does that cheaply.
const resCostAsmOK = true

// costGridAsm is resCostAsm's cost summed over a rows x cols grid of
// blocks from lv, in AVX2. Block (y, x) has ctx bit y of lbits plus
// bit x of tbits, both then set to its nz, as costLuma16Go does.
//
//go:noescape
func costGridAsm(s *costTabs, typ, first int, lv *[16]int16, rows, cols, lbits, tbits int) (c int)

// costLuma16 is costLuma16Go with the AC grid in one call.
func (s *lwState) costLuma16(lv *[25][16]int16, l, t nzCtx) int {
	if !useAVX2 {
		return s.costLuma16Go(lv, l, t)
	}
	r, _ := resCostAsm(&s.costTabs, 1, int(l[8]+t[8]), 0, &lv[24])
	lb := int(l[0]) | int(l[1])<<1 | int(l[2])<<2 | int(l[3])<<3
	tb := int(t[0]) | int(t[1])<<1 | int(t[2])<<2 | int(t[3])<<3
	return r + costGridAsm(&s.costTabs, 0, 1, &lv[0], 4, 4, lb, tb)
}

// costLuma16Nat is costLuma on defTabs, the AC grid in one call.
func costLuma16Nat(lv *[25][16]int16, l, t nzCtx) int {
	if !useAVX2 {
		return costLuma(true, lv, &l, &t)
	}
	r, _ := resCostAsm(&defTabs, 1, int(l[8]+t[8]), 0, &lv[24])
	lb := int(l[0]) | int(l[1])<<1 | int(l[2])<<2 | int(l[3])<<3
	tb := int(t[0]) | int(t[1])<<1 | int(t[2])<<2 | int(t[3])<<3
	return r + costGridAsm(&defTabs, 0, 1, &lv[0], 4, 4, lb, tb)
}

// cost4Asm is resCostLast(3, ctx, 0, last, lv) in AVX2, last -1 or
// lastNZ(lv, 0).
//
//go:noescape
func cost4Asm(s *costTabs, ctx, last int, lv *[16]int16) (c int)

// cost4 is costLast(3, ctx, 0, last, lv).
func cost4(ctx, last int, lv *[16]int16) int {
	if useAVX2 {
		return cost4Asm(&defTabs, ctx, last, lv)
	}
	c, _ := costLast(3, ctx, 0, last, lv)
	return c
}

// costUVNat is costChroma on defTabs, a grid call per plane.
func costUVNat(lv *[25][16]int16, l, t nzCtx) int {
	if !useAVX2 {
		return costChroma(lv, &l, &t)
	}
	u := costGridAsm(&defTabs, 2, 0, &lv[16], 2, 2, int(l[4])|int(l[5])<<1, int(t[4])|int(t[5])<<1)
	return u + costGridAsm(&defTabs, 2, 0, &lv[20], 2, 2, int(l[6])|int(l[7])<<1, int(t[6])|int(t[7])<<1)
}

// costGridOK selects resCost4 for i4 trial costs.
func costGridOK() bool { return useAVX2 }

// resCost4 is resCostLast for an i4 trial, AVX2 only.
func (s *lwState) resCost4(ctx, last int, lv *[16]int16) int {
	return cost4Asm(&s.costTabs, ctx, last, lv)
}

// costUV is costUVGo with each plane in one call.
func (s *lwState) costUV(lv *[25][16]int16, l, t nzCtx) int {
	if !useAVX2 {
		return s.costUVGo(lv, l, t)
	}
	u := costGridAsm(&s.costTabs, 2, 0, &lv[16], 2, 2, int(l[4])|int(l[5])<<1, int(t[4])|int(t[5])<<1)
	return u + costGridAsm(&s.costTabs, 2, 0, &lv[20], 2, 2, int(l[6])|int(l[7])<<1, int(t[6])|int(t[7])<<1)
}

// i4Trials runs every i4 mode at (y, x): levels into lv, the
// reconstruction into rec, and last, SSE, TTransform and the nonzero AC
// count into res.
func (e *Encoder) i4Trials(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x int) {
	_ = src[3*ss+3]
	if useAVX2 {
		i4TrialsAVX2Asm(&e.q.y1, pr, &src[0], ss, lv, rec, res)
		return
	}
	if useSSE41 {
		i4Trials41Asm(&e.q.y1, pr, &src[0], ss, lv, rec, res)
		return
	}
	var dq [16]int16
	i4TrialsAsm(&e.q.y1, pr, &src[0], ss, lv, rec, res, &dq)
}

// i4Trials41Asm is i4TrialsAsm in SSSE3/SSE4.1, two modes per pass;
// gen/i4sse.py generates it.
//
//go:noescape
func i4Trials41Asm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32)

// i4TrialsAVX2Asm is i4Trials41Asm with each YMM lane on its own
// modes; gen/i4avx2.py generates it.
//
//go:noescape
func i4TrialsAVX2Asm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32)

// useTrellisAsm is a var so tests can run the Go loop.
var useTrellisAsm = true

// trellisDPAsm is lwState.trellis's DP loop in scalar asm. It fills
// a.nodes and returns the best end node as n<<1|d, or -1.
//
//go:noescape
func trellisDPAsm(a *trellisArgs) int

// pred4All is pred4AllGo, in SSE4.1 when useSSE41.
func (z *ybrBuf) pred4All(pr *[10][4]uint32, y, x int) {
	if !useSSE41 {
		z.pred4AllGo(pr, y, x)
		return
	}
	_ = z[y-1][x+7]
	_ = z[y+3][x-1]
	pred4AllAsm(pr, &z[y][x])
}

// ttMB is ttMBGo, in AVX2 when useAVX2.
func ttMB(src []uint8, ss int, st *[16]int32) {
	if !useAVX2 {
		ttMBGo(src, ss, st)
		return
	}
	_ = src[15*ss+15]
	ttMBAsm(&src[0], ss, st)
}

func cpuid1() (ecx uint32)

// useSSE41 selects the SSSE3/SSE4.1/POPCNT kernels; tests clear it.
var useSSE41 = cpuid1()&(1<<9|1<<19|1<<23) == 1<<9|1<<19|1<<23

func cpuid7() (ebx uint32)
func xgetbv0() (eax uint32)

// useAVX2 selects the AVX2 kernels: AVX2 (CPUID.7 EBX bit 5), and the
// OS saves YMM state (OSXSAVE, then XCR0 bits 1 and 2); tests clear it.
var useAVX2 = useSSE41 && cpuid1()&(1<<27|1<<28) == 1<<27|1<<28 &&
	xgetbv0()&6 == 6 && cpuid7()&(1<<5) != 0

// chromaAVX2Asm is rgbaToYCbCr's opaque chroma over n blocks, n%8 == 0,
// returning the AND of their alpha bytes.
//
//go:noescape
func chromaAVX2Asm(cb, cr, p0, p1 *uint8, n int) (and uint8)

// chromaRow runs chromaAVX2Asm over the row pair's whole 8-block groups,
// returning how many blocks it did and their alpha AND.
func chromaRow(cb, cr, p0, p1 []uint8) (n int, and uint8) {
	n = len(p0) / 8 &^ 7
	if !useAVX2 || n == 0 {
		return 0, 255
	}
	_ = cb[n-1]
	_ = cr[n-1]
	_ = p1[8*n-1]
	return n, chromaAVX2Asm(&cb[0], &cr[0], &p0[0], &p1[0], n)
}

// residualQuant3Asm is residualQuant3 in AVX2.
//
//go:noescape
func residualQuant3Asm(m *quantMatrix, pr *[10][4]uint32, modes int, src *uint8, stride int, blk, lv *[3][16]int16, res *[3][2]int)

// residualQuant3 batches try's 3 i4 trials.
func (e *Encoder) residualQuant3(blk, lv *[3][16]int16, res *[3][2]int, pr *[10][4]uint32, try uint16, src []uint8, stride int) bool {
	if !useAVX2 {
		return false
	}
	_ = src[3*stride+3]
	m0 := bits.TrailingZeros16(try)
	try &= try - 1
	m1 := bits.TrailingZeros16(try)
	try &= try - 1
	m2 := bits.TrailingZeros16(try)
	residualQuant3Asm(&e.q.y1, pr, m0|m1<<4|m2<<8, &src[0], stride, blk, lv, res)
	return true
}
