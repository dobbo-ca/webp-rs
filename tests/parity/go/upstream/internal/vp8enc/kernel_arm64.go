//go:build !purego

package vp8enc

import (
	"encoding/binary"
	"math/bits"
	"os"
	"sync"
)

// collectAsm writes bin() of the 16 fdct blocks of src minus pred.
//
//go:noescape
func collectAsm(bins *[256]uint8, src *uint8, stride int, pred *uint8)

// alphaMB is the analysis alpha of the MB's 16 luma blocks.
func (e *Encoder) alphaMB(src []uint8, stride int) int {
	_ = src[15*stride+15]
	var b [256]uint8
	collectAsm(&b, &src[0], stride, &e.z[1][8])
	return binsAlpha(b[:])
}

// alphaUV is alphaMB over the 8 chroma blocks.
func (e *Encoder) alphaUV(u, v []uint8, stride int) (int, bool) {
	_ = u[7*stride+7]
	_ = v[7*stride+7]
	var b [128]uint8
	collectUVAsm(&b, &u[0], &v[0], stride, &e.z[18][8])
	return binsAlpha(b[:]), true
}

// collectUVAsm is collectAsm over the 8 chroma blocks.
//
//go:noescape
func collectUVAsm(bins *[128]uint8, u, v *uint8, stride int, pred *uint8)

// residualQuant3Asm is residualQuantAsm for the 3 modes packed in modes.
//
//go:noescape
func residualQuant3Asm(m *quantMatrix, pr *[10][4]uint32, modes int, src *uint8, stride int, blk, lv *[3][16]int16, res *[3][2]int)

// residualQuant3 runs the 3 i4 trials in try, lowest mode first.
// try must have exactly 3 bits: the asm does not mask mode 2.
func (e *Encoder) residualQuant3(blk, lv *[3][16]int16, res *[3][2]int, pr *[10][4]uint32, try uint16, src []uint8, stride int) bool {
	_ = src[3*stride+3]
	m0 := bits.TrailingZeros16(try)
	try &= try - 1
	m1 := bits.TrailingZeros16(try)
	try &= try - 1
	m2 := bits.TrailingZeros16(try)
	residualQuant3Asm(&e.q.y1, pr, m0|m1<<4|m2<<8, &src[0], stride, blk, lv, res)
	return true
}

// recMetrics16Asm is recMetrics16Go in NEON for the first n blocks; it
// stops at a row with a coefficient above 2047 in magnitude.
//
//go:noescape
func recMetrics16Asm(c *[16][16]int16, dst *uint8, src *uint8, stride int, st *[16]int32) (sse, sd, n int)

// recMetrics16 is recMetrics over the MB's 16 luma blocks, summing SSE
// and |tt - st[n]| >> 5.
func (z *ybrBuf) recMetrics16(c *[16][16]int16, src []uint8, ss int, st *[16]int32) (sse, sd int) {
	_ = src[15*ss+15]
	sse, sd, n := recMetrics16Asm(c, &z[1][8], &src[0], ss, st)
	for ; n < 16; n++ {
		y, x := 1+4*(n/4), 8+4*(n%4)
		k, t := z.recMetrics(&c[n], y, x, src[(y-1)*ss+x-8:], ss)
		sse += k
		sd += abs(t-int(st[n])) >> 5
	}
	return sse, sd
}

func (z *ybrBuf) pred4All(pr *[10][4]uint32, y, x int) {
	_ = z[y-1][x+7]
	_ = z[y+3][x-1]
	pred4AllAsm(pr, &z[y][x])
}

func ttMB(src []uint8, ss int, st *[16]int32) {
	_ = src[15*ss+15]
	ttMBAsm(&src[0], ss, st)
}

// useDot picks i4TrialsDotAsm and edgeSSEAsm's dot loop; it needs
// UDOT (HWCAP_ASIMDDP).
var useDot = hwcap()&(1<<20) != 0

// hwcap reads AT_HWCAP from the aux vector.
func hwcap() uint64 {
	b, _ := os.ReadFile("/proc/self/auxv")
	for ; len(b) >= 16; b = b[16:] {
		if binary.LittleEndian.Uint64(b) == 16 {
			return binary.LittleEndian.Uint64(b[8:])
		}
	}
	return 0
}

// i4TrialsDotAsm is i4TrialsAsm with dot products and two modes in
// flight.
//
//go:noescape
func i4TrialsDotAsm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32)

// i4Trials runs every i4 mode at (y, x): levels into lv, the
// reconstruction into rec, and last, SSE, TTransform and the nonzero AC
// count into res.
func (e *Encoder) i4Trials(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x int) {
	_ = src[3*ss+3]
	if useDot {
		i4TrialsDotAsm(&e.q.y1, pr, &src[0], ss, lv, rec, res)
		return
	}
	var dq [16]int16
	i4TrialsAsm(&e.q.y1, pr, &src[0], ss, lv, rec, res, &dq)
}

// trellisAsm is trellisGo in scalar asm.
//
//go:noescape
func trellisAsm(s *costTabs, m *quantMatrix, in, out *[16]int16, ctx0, typ, lambda int) bool

func (s *costTabs) trellis(m *quantMatrix, in, out *[16]int16, ctx0, typ, lambda int) bool {
	return trellisAsm(s, m, in, out, ctx0, typ, lambda)
}

// resCostAsm is resCost in NEON and scalar asm, returning last.
//
//go:noescape
func resCostAsm(s *costTabs, typ, ctx, first int, lv *[16]int16) (c, last int)

func (s *lwState) resCost(typ, ctx, first int, lv *[16]int16) (int, uint8) {
	c, last := resCostAsm(&s.costTabs, typ, ctx, first, lv)
	return c, b2u(last >= 0)
}

// resCostLastAsm is resCostAsm with last, -1 or at least first,
// passed in.
//
//go:noescape
func resCostLastAsm(s *costTabs, typ, ctx, first, last int, lv *[16]int16) (c int)

// resCostL is resCostLast's asm path.
func (s *lwState) resCostL(typ, ctx, first, last int, lv *[16]int16) (int, uint8) {
	return resCostLastAsm(&s.costTabs, typ, ctx, first, last, lv), b2u(last >= 0)
}

// resCostLast skips finding last again.
const resCostAsmOK = true

const useTrellisAsm = false

func trellisDPAsm(a *trellisArgs) int { panic("no trellis asm") }

// residualsYAsm and residualsUVAsm are residualsY and residualsUV in
// NEON, four blocks per pass.
//
//go:noescape
func residualsYAsm(pred, src *uint8, stride int, out *[16][16]int16)

//go:noescape
func residualsUVAsm(pred, u, v *uint8, stride int, out *[8][16]int16)

// residualsY is residual over the MB's 16 luma blocks.
func (e *Encoder) residualsY(dq *[16][16]int16, src []uint8, ys int) {
	_ = src[15*ys+15]
	residualsYAsm(&e.z[1][8], &src[0], ys, dq)
}

// residualsUV is residual over the MB's 8 chroma blocks.
func (e *Encoder) residualsUV(dq *[8][16]int16, u, v []uint8, cs int) {
	_ = u[7*cs+7]
	_ = v[7*cs+7]
	residualsUVAsm(&e.z[18][8], &u[0], &v[0], cs, dq)
}

// quantNAsm is quantAsm over n blocks, nonzero blocks as bits.
//
//go:noescape
func quantNAsm(m *quantMatrix, in, out *[16]int16, n int) (nz int)

// quantizeN is quantize over in; bit n of nz is set when block n has a
// nonzero level.
func (m *quantMatrix) quantizeN(in, out [][16]int16) (nz int) {
	_ = out[len(in)-1]
	return quantNAsm(m, &in[0], &out[0], len(in))
}

// quantNDCAsm is quantNAsm with lane 0 of block i read from dc[i].
//
//go:noescape
func quantNDCAsm(m *quantMatrix, in, out *[16]int16, n int, dc *[8]int16) (nz int)

// quantizeNDC is quantizeN over in with lane 0 of block i replaced by
// dc[i] first; at most 8 blocks.
func (m *quantMatrix) quantizeNDC(in, out [][16]int16, dc *[8]int16) (nz int) {
	_ = out[len(in)-1]
	_ = dc[len(in)-1]
	return quantNDCAsm(m, &in[0], &out[0], len(in), dc)
}

// quantI16ACAsm is quantNAsm over dq's 16 blocks with lane 0 zeroed
// on input and set to dc[n] in the dequantized block.
//
//go:noescape
func quantI16ACAsm(m *quantMatrix, dq, lv *[16][16]int16, dc *[16]int16) (nz int)

// quantI16AC is quantI16's AC levels: lv[:16] and dq with lane 0 = dc.
func (q *quant) quantI16AC(dq *[16][16]int16, lv *[25][16]int16, dc *[16]int16) int {
	return quantI16ACAsm(&q.y1, dq, (*[16][16]int16)(lv[:16]), dc)
}

// quantY16Asm is quantAsm over dq's 16 blocks with lane 0 moved to dc
// and zeroed; d sums the per-block SSE.
//
//go:noescape
func quantY16Asm(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) (d int)

// quantY16 is i16Quant's untrellised AC loop: lane 0 of each block
// moves to dc and the coefficient SSE is returned.
func (e *Encoder) quantY16(dq *[16][16]int16, dc *[16]int16) int {
	return quantY16Asm(&e.q.y1, dq, &e.lv[0], dc)
}

// iwhtAsm is iwht in NEON.
//
//go:noescape
func iwhtAsm(c, o *[16]int16)

func iwhtY(c *[16]int16) (o [16]int16) {
	iwhtAsm(c, &o)
	return o
}

// idctAddNAsm is idctAddAsm over a rows x cols block grid.
//
//go:noescape
func idctAddNAsm(c *[16]int16, dst *uint8, rows, cols int)

// idctAddsY is idctAdd over the MB's 16 luma blocks.
func (z *ybrBuf) idctAddsY(dq *[16][16]int16) { idctAddNAsm(&dq[0], &z[1][8], 4, 4) }

// idctAddsUV is idctAdd over the MB's 8 chroma blocks.
func (z *ybrBuf) idctAddsUV(dq *[8][16]int16) {
	idctAddNAsm(&dq[0], &z[18][8], 2, 2)
	idctAddNAsm(&dq[4], &z[18][24], 2, 2)
}

// costGridAsm is resCostAsm's cost summed over a rows x cols grid of
// blocks from lv. Block (y, x) has ctx bit y of lbits plus bit x of
// tbits, both then set to its nz, as costLuma16Go does. rows*cols <= 16.
//
//go:noescape
func costGridAsm(s *costTabs, typ, first int, lv *[16]int16, rows, cols, lbits, tbits int) (c int)

// costLuma16 is costLuma16Go with the AC grid in one call.
func (s *lwState) costLuma16(lv *[25][16]int16, l, t nzCtx) int {
	r, _ := resCostAsm(&s.costTabs, 1, int(l[8]+t[8]), 0, &lv[24])
	lb := int(l[0]) | int(l[1])<<1 | int(l[2])<<2 | int(l[3])<<3
	tb := int(t[0]) | int(t[1])<<1 | int(t[2])<<2 | int(t[3])<<3
	return r + costGridAsm(&s.costTabs, 0, 1, &lv[0], 4, 4, lb, tb)
}

// costLuma16Nat is costLuma on defTabs, the AC grid in one call.
func costLuma16Nat(lv *[25][16]int16, l, t nzCtx) int {
	r, _ := resCostAsm(&defTabs, 1, int(l[8]+t[8]), 0, &lv[24])
	lb := int(l[0]) | int(l[1])<<1 | int(l[2])<<2 | int(l[3])<<3
	tb := int(t[0]) | int(t[1])<<1 | int(t[2])<<2 | int(t[3])<<3
	return r + costGridAsm(&defTabs, 0, 1, &lv[0], 4, 4, lb, tb)
}

// cost4 is costLast(3, ctx, 0, last, lv) in NEON.
func cost4(ctx, last int, lv *[16]int16) int {
	if last >= 0 {
		return resCostLastAsm(&defTabs, 3, ctx, 0, last, lv)
	}
	// costLast's empty-block cost, inline: most i4 blocks are empty.
	return bitCost[0][defaultProbs[3*264+ctx*11]]
}

// costUV is costUVGo with each plane in one call.
func (s *lwState) costUV(lv *[25][16]int16, l, t nzCtx) int {
	u := costGridAsm(&s.costTabs, 2, 0, &lv[16], 2, 2, int(l[4])|int(l[5])<<1, int(t[4])|int(t[5])<<1)
	return u + costGridAsm(&s.costTabs, 2, 0, &lv[20], 2, 2, int(l[6])|int(l[7])<<1, int(t[6])|int(t[7])<<1)
}

// costUVNat is costChroma on defTabs, a grid call per plane.
func costUVNat(lv *[25][16]int16, l, t nzCtx) int {
	u := costGridAsm(&defTabs, 2, 0, &lv[16], 2, 2, int(l[4])|int(l[5])<<1, int(t[4])|int(t[5])<<1)
	return u + costGridAsm(&defTabs, 2, 0, &lv[20], 2, 2, int(l[6])|int(l[7])<<1, int(t[6])|int(t[7])<<1)
}

// costGridOK is false so pickI4 inlines resCostLast; a wrapper
// there made c7g lw1 0.66% slower.
func costGridOK() bool { return false }

func (s *lwState) resCost4(ctx, last int, lv *[16]int16) int {
	c, _ := s.resCostLast(3, ctx, 0, last, lv)
	return c
}

// chromaTabs are chromaAsm's tables and constants in load order.
//
// h[v>>4] + v (byte wrap), >>4, is gammaToLinear[v]>>8. g(255) = 4095,
// so the high byte is <= 15 and H = 15 never steps. A run of 16 inputs
// crosses at most one multiple of 256: g spans [0,425], [447,759],
// [778,1059], [1076,1338], [1355,1604] for runs 0 to 4, later spans are
// narrower (concave), under 256. With the step at offset s, h[q] =
// 16H+16-s-16q gives 16H+16-s+r in [16H, 16H+30], no overflow past 255.
type chromaTabs struct {
	h    [16]uint8    // high byte offsets
	g    [4][64]uint8 // gammaToLinear low bytes by quarter
	l, d [32]uint8    // linearToGamma[p], linearToGamma[p+1]-[p]
	k    [5][8]int16  // cb R, G, B; cr G, B (cr R is cb B)
	bias [4]int32
}

// chromaTab is built on first use: gammaToLinear fills in a later init.
var chromaTab = sync.OnceValue(func() *chromaTabs {
	t := new(chromaTabs)
	for v, g := range &gammaToLinear {
		t.g[v/64][v%64] = uint8(g)
	}
	hi := func(v int) int { return int(gammaToLinear[v] >> 8) }
	for q := range t.h {
		s := 16
		for r := 1; r < 16; r++ {
			if hi(16*q+r) != hi(16*q) {
				s = r
				break
			}
		}
		t.h[q] = uint8(16*hi(16*q) + 16 - s - 16*q)
	}
	for p := range t.l {
		t.l[p] = uint8(linearToGamma[p])
		t.d[p] = uint8(linearToGamma[p+1] - linearToGamma[p])
	}
	for i, c := range [5]int16{-9719, -19081, 28800, -24116, -4684} {
		for j := range t.k[i] {
			t.k[i][j] = c
		}
	}
	for j := range t.bias {
		t.bias[j] = 1<<17 + 128<<18
	}
	return t
})

// chromaAsm is rgbaToYCbCr's opaque chroma over n blocks, n%8 == 0,
// returning the AND of their alpha bytes.
//
//go:noescape
func chromaAsm(t *chromaTabs, cb, cr, p0, p1 *uint8, n int) (and uint8)

// chromaRow runs chromaAsm over the row pair's whole 8-block groups,
// returning how many blocks it did and their alpha AND.
func chromaRow(cb, cr, p0, p1 []uint8) (n int, and uint8) {
	n = len(p0) / 8 &^ 7
	if n == 0 {
		return 0, 255
	}
	_ = cb[n-1]
	_ = cr[n-1]
	_ = p1[8*n-1]
	return n, chromaAsm(chromaTab(), &cb[0], &cr[0], &p0[0], &p1[0], n)
}
