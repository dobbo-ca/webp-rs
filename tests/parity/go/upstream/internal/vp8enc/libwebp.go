package vp8enc

// MatchLibwebp mode: libwebp 1.6.0's RD_OPT_BASIC path (method 3 and 4)
// ported as-is, so output bytes equal cwebp's. Sources are libwebp
// quant_enc.c (VP8Decimate), frame_enc.c (VP8EncTokenLoop), cost_enc.c
// and filter_enc.c, BSD-3-Clause.

import (
	"math/bits"
	"slices"
)

// lwState is libwebp's running state for one encode.
type lwState struct {
	costTabs              // proba->coeffs and its costs, with VP8LevelFixedCosts
	stats    [1057]uint32 // proba->stats: total<<16 | ones; 1056 takes sign bits
	cnt, max int          // mid-pass refresh countdown
	maxEdge  [4]int
	diffuse  bool
	topDerr  [][2][2]int8 // per MB column: [ch][top0, top1]
	leftDerr [2][2]int8
	tmp      lwScratch // encodeMBLW scratch
}

// statSlot maps a token's prob index, with fixedTok at bit 11, to its
// stats slot: prob 10 counts in 9, fixed-prob bits in the unused 1056.
var statSlot = func() (t [4096]uint16) {
	for i := range t {
		t[i] = uint16(i)
		if i%11 == 10 {
			t[i]--
		}
		if i >= 1056 {
			t[i] = 1056
		}
	}
	return t
}()

// kWeightY, quant_enc.c:517.
var weightY = [16]int{38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2}

// VP8FixedCostsUV, cost_enc.c:103.
var uvCostLW = [4]int{302, 984, 439, 642}

const flatPenalty = 140 // FLATNESS_PENALTY

// lwFixed is VP8LevelFixedCosts, which has 640 at 9 and 10.
func lwFixed(v int) int {
	if v == 9 || v == 10 {
		return 640
	}
	return fixedCost[v]
}

func (s *lwState) reset(mbw, mbh int, quality float32) {
	s.probs = defaultProbs
	s.buildCosts()
	s.stats = [1057]uint32{}
	s.max = max(mbw*mbh>>3, 96) // MIN_COUNT
	s.cnt = s.max
	s.maxEdge = [4]int{}
	s.diffuse = quality <= 98 // ERROR_DIFFUSION_QUALITY
	if cap(s.topDerr) < mbw {
		s.topDerr = make([][2][2]int8, mbw)
	}
	s.topDerr = s.topDerr[:mbw]
	clear(s.topDerr)
}

// levelToks[v] are level v's prob-coded tokens at prob offset 0.
var levelToks [68][]uint16

// classToks are the distinct levelToks; classOf maps v to one.
var (
	classToks [][]uint16
	classOf   [68]uint8
)

func init() {
	for v := 1; v < 68; v++ {
		c, _ := recordLevel(nil, v, 0)
		for _, tok := range c {
			if tok&fixedTok == 0 {
				levelToks[v] = append(levelToks[v], tok)
			}
		}
		k := slices.IndexFunc(classToks, func(t []uint16) bool { return slices.Equal(t, levelToks[v]) })
		if k < 0 {
			k = len(classToks)
			classToks = append(classToks, levelToks[v])
		}
		classOf[v] = uint8(k)
	}
}

// buildCosts is VP8CalculateLevelCosts over s.probs.
func (s *lwState) buildCosts() {
	var sum [68]int
	for typ := range s.cost {
		for n := range s.cost[typ] {
			m := 0 // first position in n's band
			for bands[m] != bands[n] {
				m++
			}
			for ctx := 0; ctx < 3; ctx++ {
				t := &s.cost[typ][n][ctx]
				if m < n {
					copy(t[:68], s.cost[typ][m][ctx][:68])
				} else {
					p := typ*264 + (int(bands[n])*3+ctx)*11
					c0 := 0
					if ctx > 0 {
						c0 = bitCost[1][s.probs[p]]
					}
					t[0] = uint16(bitCost[0][s.probs[p+1]] + c0)
					for c, toks := range classToks {
						k := bitCost[1][s.probs[p+1]] + c0
						for _, tok := range toks {
							k += bitCost[tok>>15][s.probs[p+int(tok&0x7ff)]]
						}
						sum[c] = k
					}
					for v := 1; v < 68; v++ {
						t[v] = uint16(sum[classOf[v]] + lwFixed(v))
					}
				}
				if n < 15 {
					s.eob[typ][n][ctx] = uint16(bitCost[0][s.probs[typ*264+(int(bands[n+1])*3+ctx)*11]])
				}
			}
		}
	}
}

// finalize is FinalizeTokenProbas; it reports a prob off its default.
func (s *lwState) finalize() bool {
	changed := false
	for i := range s.probs {
		nb, total := int(s.stats[i]&0xffff), int(s.stats[i]>>16)
		np := 255
		if nb != 0 {
			np = 255 - nb*255/total
		}
		bc := func(q int) int { return nb*bitCost[1][q] + (total-nb)*bitCost[0][q] }
		old := bc(int(defaultProbs[i])) + bitCost[0][updateProbs[i]]
		nw := bc(np) + bitCost[1][updateProbs[i]] + 8*256
		s.probs[i] = defaultProbs[i]
		if old > nw {
			s.probs[i] = uint8(np)
			changed = changed || uint8(np) != defaultProbs[i]
		}
	}
	return changed
}

// refresh runs before each MB, frame_enc.c:832-837.
func (s *lwState) refresh() {
	if s.cnt--; s.cnt < 0 {
		if s.finalize() {
			s.buildCosts()
		}
		s.cnt = s.max
	}
}

// record is VP8RecordStats over one MB's tokens. libwebp counts the
// cat5/cat6 bit (prob 10) in slot 9, token_enc.c:172,178.
func (s *lwState) record(t *tokenBuf, a, b tokPos) {
	for pg := a.pg; pg <= b.pg; pg++ {
		p := t.page(int(pg))
		if pg == b.pg {
			p = p[:b.off]
		}
		if pg == a.pg {
			p = p[a.off:]
		}
		for _, v := range p {
			i := statSlot[(v&0x7ff|v>>3&0x800)&0xfff]
			x := s.stats[i]
			if x >= 0xfffe0000 {
				x = (x + 1) >> 1 & 0x7fff7fff
			}
			s.stats[i] = x + 0x10000 + uint32(v>>15)
		}
	}
}

// resCostLast is resCost with the last nonzero index known.
func (s *lwState) resCostLast(typ, ctx, first, last int, lv *[16]int16) (int, uint8) {
	if resCostAsmOK {
		return s.resCostL(typ, ctx, first, last, lv)
	}
	base := typ * 264
	p0 := s.probs[base+(int(bands[first])*3+ctx)*11]
	if last < 0 {
		return bitCost[0][p0], 0
	}
	c := 0
	if ctx == 0 {
		c = bitCost[1][p0]
	}
	k, v := levelCosts(&s.cost[typ&3], lv, first, last, ctx)
	c += k
	if last < 15 {
		c += bitCost[0][s.probs[base+(int(bands[last+1])*3+v)*11]]
	}
	return c, 1
}

// levelCostsGo is the cost of lv[first:last+1] from ctx on the rows t,
// and the context after it. first <= last.
func levelCostsGo(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int) {
	v := 0
	for n := first; n <= last; n++ {
		v = int(lv[n&15])
		m := v >> 63
		v = (v ^ m) - m
		if v > 67 { // rare
			c += lwFixed(v&2047) - fixedCost[67]
		}
		c += int(t[n&15][ctx&3][min(uint(v), 67)&127])
		k := v - 2
		ctx = 2 + k&(k>>63)
	}
	return c, ctx
}

// costLuma16Go is VP8GetCostLuma16.
func (s *lwState) costLuma16Go(lv *[25][16]int16, l, t nzCtx) int {
	r, nz := s.resCost(1, int(l[8]+t[8]), 0, &lv[24])
	_ = nz
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			c, nz := s.resCost(0, int(l[y]+t[x]), 1, &lv[4*y+x])
			l[y], t[x] = nz, nz
			r += c
		}
	}
	return r
}

// costUVGo is VP8GetCostUV.
func (s *lwState) costUVGo(lv *[25][16]int16, l, t nzCtx) int {
	r := 0
	for c := 0; c < 4; c += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				k, nz := s.resCost(2, int(l[4+c+y]+t[4+c+x]), 0, &lv[16+2*c+2*y+x])
				l[4+c+y], t[4+c+x] = nz, nz
				r += k
			}
		}
	}
	return r
}

// isFlat is IsFlat_C over n blocks of levels. Blocks whose nz
// bit is clear hold no levels, so they are skipped.
func isFlat(lv [][16]int16, nz, thresh int) bool {
	k := 0
	for i := range lv {
		if nz>>i&1 == 0 {
			continue
		}
		for _, v := range lv[i][1:] {
			k += int(uint16(v)|-uint16(v)) >> 15
		}
	}
	return k <= thresh
}

// tt is TTransform of the 4x4 block at p.
func tt(p []uint8, stride int) int {
	var t [16]int
	for i := 0; i < 4; i++ {
		r := p[i*stride : i*stride+4]
		a0, a1 := int(r[0])+int(r[2]), int(r[1])+int(r[3])
		a2, a3 := int(r[1])-int(r[3]), int(r[0])-int(r[2])
		t[4*i], t[4*i+1], t[4*i+2], t[4*i+3] = a0+a1, a3+a2, a3-a2, a0-a1
	}
	sum := 0
	for i := 0; i < 4; i++ {
		a0, a1 := t[i]+t[8+i], t[4+i]+t[12+i]
		a2, a3 := t[4+i]-t[12+i], t[i]-t[8+i]
		sum += weightY[i]*abs(a0+a1) + weightY[4+i]*abs(a3+a2) +
			weightY[8+i]*abs(a3-a2) + weightY[12+i]*abs(a0-a1)
	}
	return sum
}

// ttMBGo is tt over the MB's 16 luma blocks of src.
func ttMBGo(src []uint8, ss int, st *[16]int32) {
	for n := range st {
		y, x := 4*(n/4), 4*(n%4)
		st[n] = int32(tt(src[y*ss+x:], ss))
	}
}

// tt is TTransform of the 4x4 block of z at (y, x).
func (z *ybrBuf) tt(y, x int) int {
	var t [16]int
	for i := 0; i < 4; i++ {
		r := z[y+i][x : x+4 : x+4]
		a0, a1 := int(r[0])+int(r[2]), int(r[1])+int(r[3])
		a2, a3 := int(r[1])-int(r[3]), int(r[0])-int(r[2])
		t[4*i], t[4*i+1], t[4*i+2], t[4*i+3] = a0+a1, a3+a2, a3-a2, a0-a1
	}
	sum := 0
	for i := 0; i < 4; i++ {
		a0, a1 := t[i]+t[8+i], t[4+i]+t[12+i]
		a2, a3 := t[4+i]-t[12+i], t[i]-t[8+i]
		sum += weightY[i]*abs(a0+a1) + weightY[4+i]*abs(a3+a2) +
			weightY[8+i]*abs(a3-a2) + weightY[12+i]*abs(a0-a1)
	}
	return sum
}

func mult8b(a, b int) int { return (a*b + 128) >> 8 }

// rdScore is SetRDScore.
func rdScore(lambda, r, h, d, sd int) int { return (r+h)*lambda + 256*(d+sd) }

// quantSingle is QuantizeSingle: quantizes *v and returns err >> 1.
func quantSingle(v *int16, m *quantMatrix) int {
	V := int(*v)
	neg := V < 0
	if neg {
		V = -V
	}
	qV := int((uint32(V)*m.iq[0]+m.bias[0])>>17) * int(m.q[0])
	err := V - qV
	if neg {
		*v, err = int16(-qV), -err
	} else {
		*v = int16(qV)
	}
	return err >> 1
}

// correctDC is CorrectDCValues on the 8 chroma blocks; the corrected
// DCs go to dc, not dq, so no narrow store meets quantizeNDC's load.
func (s *lwState) correctDC(m *quantMatrix, dq *[8][16]int16, dc *[8]int16, mbx int, derr *[2][3]int8) {
	for ch := 0; ch < 2; ch++ {
		top, left := &s.topDerr[mbx][ch], &s.leftDerr[ch]
		c, d := dq[4*ch:4*ch+4], dc[4*ch:4*ch+4]
		d[0] = c[0][0] + int16((7*int(top[0])+8*int(left[0]))>>3)
		e0 := quantSingle(&d[0], m)
		d[1] = c[1][0] + int16((7*int(top[1])+8*e0)>>3)
		e1 := quantSingle(&d[1], m)
		d[2] = c[2][0] + int16((7*e0+8*int(left[1]))>>3)
		e2 := quantSingle(&d[2], m)
		d[3] = c[3][0] + int16((7*e1+8*e2)>>3)
		e3 := quantSingle(&d[3], m)
		derr[ch] = [3]int8{int8(e1), int8(e2), int8(e3)}
	}
}

// storeDerr is StoreDiffusionErrors.
func (s *lwState) storeDerr(derr [2][3]int8, mbx int) {
	for ch := 0; ch < 2; ch++ {
		l := &s.leftDerr[ch]
		t := &s.topDerr[mbx][ch]
		l[0] = derr[ch][0]
		l[1] = int8(3 * int(derr[ch][2]) >> 2)
		t[0] = derr[ch][1]
		t[1] = int8(int(derr[ch][2]) - int(l[1]))
	}
}

// fwhtPlain is FTransformWHT_C: fwht without the x/image rounding.
func fwhtPlain(dc *[16]int16) {
	var t [16]int32
	for i := 0; i < 4; i++ {
		p0, p1, p2, p3 := int32(dc[4*i]), int32(dc[4*i+1]), int32(dc[4*i+2]), int32(dc[4*i+3])
		a0, a1, a2, a3 := p0+p2, p1+p3, p1-p3, p0-p2
		t[4*i], t[4*i+1], t[4*i+2], t[4*i+3] = a0+a1, a3+a2, a3-a2, a0-a1
	}
	for i := 0; i < 4; i++ {
		a0, a1 := t[i]+t[8+i], t[4+i]+t[12+i]
		a2, a3 := t[4+i]-t[12+i], t[i]-t[8+i]
		dc[i] = int16((a0 + a1) >> 1)
		dc[4+i] = int16((a3 + a2) >> 1)
		dc[8+i] = int16((a3 - a2) >> 1)
		dc[12+i] = int16((a0 - a1) >> 1)
	}
}

// lwI16 is one i16 trial up to the dequantized coefficients in dq;
// the caller adds them to the prediction. It returns nz as libwebp packs
// it (bit 24 for the DC block).
func (e *Encoder) lwI16(dq *[16][16]int16, lv *[25][16]int16) int {
	e.residualsY(dq, e.lwSrc, e.img.YStride)
	return e.q.quantI16(dq, lv)
}

// quantI16 is lwI16 after the residuals.
func (q *quant) quantI16(dq *[16][16]int16, lv *[25][16]int16) int {
	var dc [16]int16
	y2 := i16DC(&q.y2, dq, &lv[24], &dc)
	nz := q.quantI16AC(dq, lv, &dc)
	if y2 {
		nz |= 1 << 24
	}
	return nz
}

// lwUV is one chroma trial: ReconstructUV into z. derr is by pointer to
// skip a result copy.
func (e *Encoder) lwUV(dq *[8][16]int16, lv *[25][16]int16, srcU, srcV []uint8, cs, mbx int, derr *[2][3]int8) (nz int) {
	e.residualsUV(dq, srcU, srcV, cs)
	if e.lw.diffuse {
		var dc [8]int16
		e.lw.correctDC(&e.q.uv, dq, &dc, mbx, derr)
		nz = e.q.uv.quantizeNDC(dq[:], lv[16:24], &dc)
	} else {
		nz = e.q.uv.quantizeN(dq[:], lv[16:24])
	}
	e.z.idctAddsUV(dq)
	return nz
}

// lwScratch is encodeMBLW's scratch. Every element read is written first,
// so it is never cleared.
type lwScratch struct {
	dq16 [16][16]int16
	blv  [25][16]int16
	brec [16][16]uint8
	dqUV [8][16]int16
	lvUV [8][16]int16
	buv  [8][2][8]uint8
	pr   [10][4]uint32
	lv   [10][16]int16
	rec  [10][4]uint32
	res  [10][4]int32
	p16  [4][16][16]uint8 // i16 predictions, for lwDedup
	pUV  [4][8][16]uint8
}

// lwDedup reuses the trial of an earlier mode with the same
// prediction; a var so tests can compare it against full trials.
var lwDedup = true

// wideInt gates the rate skip: a 32-bit int wraps rdScore at low
// quality, so the score is not monotone in the rate there.
const wideInt = bits.UintSize == 64

// lwTrial is one i16 or chroma trial's result. r and flat are filled
// on first use.
type lwTrial struct {
	nz, d, sd, r int
	derr         [2][3]int8
	rOK, flat    bool
	flatOK       bool
}

// i16Trial runs the i16 trial on the prediction in z.
func (e *Encoder) i16Trial(t *lwTrial, dq16 *[16][16]int16, left *nzCtx, mbx int, srcY []uint8, ys int, st *[16]int32, tlambda int) {
	if e.method >= 6 {
		t.nz = e.lwI16T(dq16, &e.lv, *left, e.top[mbx])
	} else {
		t.nz = e.lwI16(dq16, &e.lv)
	}
	d, sd := e.z.recMetrics16(dq16, srcY, ys, st)
	t.d, t.sd = d, mult8b(tlambda, sd)
}

// uvTrial runs the chroma trial on the prediction in z.
func (e *Encoder) uvTrial(t *lwTrial, dq *[8][16]int16, srcU, srcV []uint8, cs, mbx int) {
	t.nz = e.lwUV(dq, &e.lv, srcU, srcV, cs, mbx, &t.derr)
	t.d = e.z.predSSE(srcU, cs, 18, 8, 8, 8) + e.z.predSSE(srcV, cs, 18, 24, 8, 8)
}

// predY is the 16x16 luma prediction in z.
func (z *ybrBuf) predY() (p [16][16]uint8) {
	for j := range p {
		p[j] = [16]uint8(z[1+j][8:24])
	}
	return p
}

// predUV is the 8x8 U and V predictions in z, one row of each per row.
func (z *ybrBuf) predUV() (p [8][16]uint8) {
	for j := range p {
		copy(p[j][:8], z[18+j][8:16])
		copy(p[j][8:], z[18+j][24:32])
	}
	return p
}

// pr4Same reports all ten i4 predictions are equal.
func pr4Same(pr *[10][4]uint32) bool {
	for m := 1; m < len(pr); m++ {
		if pr[m] != pr[0] {
			return false
		}
	}
	return true
}

// i4Trial1 is i4Trials (i4TrialsTrellis at method 6) when every
// prediction is equal: mode 0's trial, which stands for every mode.
// Only a 32-bit int copies it to the rest, for pickI4's full loop.
func (e *Encoder) i4Trial1(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x, ctx int) {
	if e.method >= 6 {
		e.i4TrialTrellis(0, pr, src, ss, lv, rec, res, y, x, ctx)
	} else {
		e.i4TrialGo(0, pr, src, ss, lv, rec, res, y, x)
	}
	if !wideInt {
		for m := 1; m < len(pr); m++ {
			lv[m], rec[m], res[m] = lv[0], rec[0], res[0]
		}
	}
}

// isFlatSource16 is IsFlatSource16.
func isFlatSource16(src []uint8, ys int) bool { return flat(src, 0, ys, 16, 16, 1) }

// encodeMBLW is VP8Decimate at RD_OPT_BASIC, then RecordTokens.
func (e *Encoder) encodeMBLW(mbx, mby int, left *nzCtx) {
	z := &e.z
	s := e.lw
	z.prepare(&e.img, mbx, mby, e.mbw)
	ys := e.img.YStride
	srcY := e.img.Y[mby*16*ys+mbx*16:]
	e.lwSrc = srcY
	mb := &e.mbs[mby*e.mbw+mbx]
	e.q, e.lam = e.segQ[mb.seg], e.segLam[mb.seg]
	lam := &e.lam
	qi4 := qa(&e.q.y1)
	tlambda := max(50*qi4>>5, 1)
	if e.method < 4 {
		tlambda = 1
	}

	// TTransform of the source blocks, shared by all luma trials.
	var st [16]int32
	ttMB(srcY, ys, &st)

	// PickBestIntra16. A trial is a function of its prediction, so a
	// mode predicting an earlier mode's bytes reuses that trial (rep).
	// A trial reruns only where its levels or reconstruction are gone.
	tmp := &s.tmp
	dq16, blv, brec := &tmp.dq16, &tmp.blv, &tmp.brec
	best16, bs, bnz := uint8(0), 0, 0
	bD, bSD, bR, bH := 0, 0, 0, 0
	flat16 := isFlatSource16(srcY, ys)
	var c16 [4]lwTrial
	bRep, lvRep := -1, -1 // reps held by blv/brec and by e.lv
	for m := uint8(0); m < 4; m++ {
		z.predMB(16, 1, 8, m, mbx, mby)
		rep, zRec := int(m), -1 // zRec: rep whose reconstruction z holds
		if lwDedup {
			tmp.p16[m] = z.predY()
			for j := range m {
				if tmp.p16[j] == tmp.p16[m] {
					rep = int(j)
					break
				}
			}
		}
		t := &c16[rep]
		if rep == int(m) {
			e.i16Trial(t, dq16, left, mbx, srcY, ys, &st, tlambda)
			lvRep, zRec = rep, rep
		}
		nz, d, sd := t.nz, t.d, t.sd
		h := i16Cost[m]
		if flat16 {
			if flat16 = nz&0xffff == 0; flat16 {
				d, sd = 2*d, 2*sd
			}
		}
		// Rate is never negative.
		if lwDedup && wideInt && m > 0 && rdScore(lam.i16, 0, h, d, sd) >= bs {
			continue
		}
		if !t.rOK {
			if lvRep != rep { // z still holds the prediction
				e.i16Trial(t, dq16, left, mbx, srcY, ys, &st, tlambda)
				lvRep, zRec = rep, rep
			}
			t.r, t.rOK = s.costLuma16(&e.lv, *left, e.top[mbx]), true
		}
		if sc := rdScore(lam.i16, t.r, h, d, sd); m == 0 || sc < bs {
			if bRep != rep {
				if zRec != rep {
					e.i16Trial(t, dq16, left, mbx, srcY, ys, &st, tlambda)
					lvRep = rep
				}
				for j := range brec {
					brec[j] = [16]uint8(z[1+j][8:24])
				}
				copy(blv[:16], e.lv[:16])
				blv[24] = e.lv[24]
				bRep = rep
			}
			best16, bs, bnz = m, sc, nz
			bD, bSD, bR, bH = d, sd, t.r, h
		}
	}
	s16 := rdScore(lam.mode, bR, bH, bD, bSD)
	if bnz&0x100ffff == 0x1000000 && bD > 20*int(e.q.y1.q[0]) {
		dc := &blv[24]
		s.maxEdge[mb.seg] = max(s.maxEdge[mb.seg], abs(int(dc[1])), abs(int(dc[2])), abs(int(dc[4])))
	}

	// PickBestIntra4.
	i16 := false
	var modes [16]uint8
	pr, lv, rec, res := &tmp.pr, &tmp.lv, &tmp.rec, &tmp.res
	{
		sc := rdScore(lam.mode, 0, 211, 0, 0)
		hdr := 0
		l, t := *left, e.top[mbx]
		for n := 0; n < 16; n++ {
			y, x := 1+4*(n/4), 8+4*(n%4)
			src := srcY[(y-1)*ys+x-8:]
			above, lft := e.lwModes(&modes, mbx, mby, n)
			ctx := int(l[n/4] + t[n%4])
			z.pred4All(pr, y, x)
			same := lwDedup && pr4Same(pr)
			if same {
				e.i4Trial1(pr, src, ys, lv, rec, res, y, x, ctx)
			} else if e.method >= 6 {
				e.i4TrialsTrellis(pr, src, ys, lv, rec, res, y, x, ctx)
			} else {
				e.i4Trials(pr, src, ys, lv, rec, res, y, x)
			}
			best, b4D, b4SD, b4R, b4H := s.pickI4(res, lv, &i4Cost[above][lft], st[n], ctx, lam.i4, tlambda, same)
			sc += rdScore(lam.mode, b4R, b4H, b4D, b4SD)
			if sc >= s16 {
				i16 = true
				break
			}
			if hdr += b4H; hdr > 65536 { // max_i4_header_bits at method 4
				i16 = true
				break
			}
			k := best
			if same { // i4Trial1 filled mode 0 only
				k = 0
			}
			z.put4(&rec[k], y, x)
			e.lv[n] = lv[k]
			modes[n] = uint8(best)
			nz := uint8(0)
			if res[k][0] >= 0 {
				nz = 1
			}
			l[n/4], t[n%4] = nz, nz
		}
	}
	if i16 {
		for j := range brec {
			*(*[16]uint8)(z[1+j][8:24]) = brec[j]
		}
		copy(e.lv[:16], blv[:16])
		e.lv[24] = blv[24]
		for n := range modes {
			modes[n] = best16
		}
	}

	// PickBestUV.
	cs := e.img.CStride
	srcU := e.img.Cb[mby*8*cs+mbx*8:]
	srcV := e.img.Cr[mby*8*cs+mbx*8:]
	dqUV, lvUV, buv := &tmp.dqUV, &tmp.lvUV, &tmp.buv
	var bderr [2][3]int8
	bestUV, bsUV := uint8(0), 0
	// As PickBestIntra16: equal predictions share a trial.
	var cUV [4]lwTrial
	bRepUV, lvRepUV, zRecUV := -1, -1, -1
	for m := uint8(0); m < 4; m++ {
		z.predMB(8, 18, 8, m, mbx, mby)
		z.predMB(8, 18, 24, m, mbx, mby)
		rep := int(m)
		zRecUV = -1
		if lwDedup {
			tmp.pUV[m] = z.predUV()
			for j := range m {
				if tmp.pUV[j] == tmp.pUV[m] {
					rep = int(j)
					break
				}
			}
		}
		t := &cUV[rep]
		if rep == int(m) {
			e.uvTrial(t, dqUV, srcU, srcV, cs, mbx)
			lvRepUV, zRecUV = rep, rep
		}
		if lwDedup && wideInt && m > 0 && rdScore(lam.uv, 0, uvCostLW[m], t.d, 0) >= bsUV {
			continue
		}
		if !t.rOK || m > 0 && !t.flatOK {
			if lvRepUV != rep { // z still holds the prediction
				e.uvTrial(t, dqUV, srcU, srcV, cs, mbx)
				lvRepUV, zRecUV = rep, rep
			}
			if !t.rOK {
				t.r, t.rOK = s.costUV(&e.lv, *left, e.top[mbx]), true
			}
			if m > 0 && !t.flatOK {
				t.flat, t.flatOK = isFlat(e.lv[16:24], t.nz, 2), true
			}
		}
		r := t.r
		if m > 0 && t.flat {
			r += flatPenalty * 8
		}
		if sc := rdScore(lam.uv, r, uvCostLW[m], t.d, 0); m == 0 || sc < bsUV {
			if bRepUV != rep {
				if zRecUV != rep {
					e.uvTrial(t, dqUV, srcU, srcV, cs, mbx)
					lvRepUV, zRecUV = rep, rep
				}
				for j := range buv {
					buv[j][0] = [8]uint8(z[18+j][8:16])
					buv[j][1] = [8]uint8(z[18+j][24:32])
				}
				copy(lvUV[:], e.lv[16:24])
				bRepUV = rep
			}
			bestUV, bsUV = m, sc
			bderr = t.derr
		}
	}
	if zRecUV != bRepUV { // z holds bestUV's reconstruction otherwise
		for j := range buv {
			*(*[8]uint8)(z[18+j][8:16]) = buv[j][0]
			*(*[8]uint8)(z[18+j][24:32]) = buv[j][1]
		}
	}
	copy(e.lv[16:24], lvUV[:])
	if s.diffuse {
		s.storeDerr(bderr, mbx)
	}
	if e.method == 5 {
		e.simpleQuantize(i16, &modes, bestUV, mbx, mby, left)
	}

	// RecordTokens: every MB, no skip flag.
	a := e.tok.pos()
	recordLuma(&e.tok, i16, &e.lv, left, &e.top[mbx])
	recordChroma(&e.tok, &e.lv, left, &e.top[mbx])
	s.record(&e.tok, a, e.tok.pos())

	for j := 0; j < 16; j++ {
		copy(srcY[j*ys:j*ys+16], z[1+j][8:24])
	}
	for j := 0; j < 8; j++ {
		copy(srcU[j*cs:j*cs+8], z[18+j][8:16])
		copy(srcV[j*cs:j*cs+8], z[18+j][24:32])
	}
	*mb = mbInfo{i16: i16, modes: modes, uv: bestUV, seg: mb.seg}
}

// lwModes is ctxModes with this MB's i4 modes still in progress.
func (e *Encoder) lwModes(modes *[16]uint8, mbx, mby, n int) (above, left uint8) {
	above, left = predDC, predDC
	if n >= 4 {
		above = modes[n-4]
	} else if mby > 0 {
		above = e.mbs[(mby-1)*e.mbw+mbx].modes[12+n]
	}
	if n&3 != 0 {
		left = modes[n-1]
	} else if mbx > 0 {
		left = e.mbs[mby*e.mbw+mbx-1].modes[n+3]
	}
	return above, left
}

// pickI4 is PickBestIntra4's mode choice. same: every mode has mode
// 0's trial, read from index 0 only.
func (s *lwState) pickI4(res *[10][4]int32, lv *[10][16]int16, hc *[10]int, st int32, ctx, lambda, tlambda int, same bool) (best, d, sd, r, h int) {
	if same && wideInt {
		return s.pickI4Same(res, lv, hc, st, ctx, tlambda)
	}
	bs, bc, sc := 0, 0, -1
	for m := range res {
		sd := mult8b(tlambda, abs(int(res[m][2]-st))>>5)
		r := 0
		if m > 0 && res[m][3] <= 3 {
			r = flatPenalty
		}
		k := rdScore(lambda, r, hc[m], int(res[m][1]), sd)
		if m > 0 && k >= bs {
			continue
		}
		var c int
		if sc >= 0 {
			c = sc
		} else if res[m][0] < 0 { // no levels: resCostLast's EOB-only cost
			c = bitCost[0][s.probs[3*264+(int(bands[0])*3+ctx)*11]]
		} else if costGridOK() {
			c = s.resCost4(ctx, int(res[m][0]), &lv[m])
		} else {
			c, _ = s.resCostLast(3, ctx, 0, int(res[m][0]), &lv[m])
		}
		if same {
			sc = c
		}
		if k += c * lambda; m == 0 || k < bs {
			best, bs, bc = m, k, c
		}
	}
	d, h = int(res[best][1]), hc[best]
	sd = mult8b(tlambda, abs(int(res[best][2]-st))>>5)
	if r = bc; best > 0 && res[best][3] <= 3 {
		r += flatPenalty
	}
	return best, d, sd, r, h
}

// pickI4Same is pickI4 when every mode has mode 0's trial. The scores
// then differ only in header cost plus flatness penalty, times a
// lambda of at least 1, so the pick is their first argmin, costed once.
func (s *lwState) pickI4Same(res *[10][4]int32, lv *[10][16]int16, hc *[10]int, st int32, ctx, tlambda int) (best, d, sd, r, h int) {
	pen := 0
	if res[0][3] <= 3 {
		pen = flatPenalty
	}
	bh := hc[0]
	for m := 1; m < len(hc); m++ {
		if k := hc[m] + pen; k < bh {
			best, bh = m, k
		}
	}
	switch {
	case res[0][0] < 0: // no levels: resCostLast's EOB-only cost
		r = bitCost[0][s.probs[3*264+(int(bands[0])*3+ctx)*11]]
	case costGridOK():
		r = s.resCost4(ctx, int(res[0][0]), &lv[0])
	default:
		r, _ = s.resCostLast(3, ctx, 0, int(res[0][0]), &lv[0])
	}
	if best > 0 {
		r += pen
	}
	d, h = int(res[0][1]), hc[best]
	sd = mult8b(tlambda, abs(int(res[0][2]-st))>>5)
	return best, d, sd, r, h
}

// adjustFilter is VP8AdjustFilterStrength with sharpness 0.
func (e *Encoder) adjustFilter(s *segs) {
	for i := range s.level {
		if lv := min(e.lw.maxEdge[i]*int(e.segQ[i].y2.q[1])>>3, 63); lv > s.level[i] {
			s.level[i] = lv
		}
	}
}

// sse4 is SSE4x4 of src against z at (y, x).
func (z *ybrBuf) sse4(src []uint8, ss, y, x int) int {
	d := 0
	for j := 0; j < 4; j++ {
		s, r := src[j*ss:j*ss+4:j*ss+4], z[y+j][x:x+4:x+4]
		a, b := int(s[0])-int(r[0]), int(s[1])-int(r[1])
		c, e := int(s[2])-int(r[2]), int(s[3])-int(r[3])
		d += a*a + b*b + c*c + e*e
	}
	return d
}

// emit1 codes all tokens into one partition.
func (t *tokenBuf) emit1(a *boolEncoder, p *[1056]uint8) {
	for i := 0; i <= t.n; i++ {
		emitPage(a, t.page(i), p)
	}
}

func (z *ybrBuf) recMetricsGo(c *[16]int16, y, x int, src []uint8, ss int) (sse, tt int) {
	z.idctAddGo(c, y, x)
	return z.sse4(src, ss, y, x), z.tt(y, x)
}

func (e *Encoder) i4TrialsGo(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x int) {
	for m := range pr {
		e.i4TrialGo(m, pr, src, ss, lv, rec, res, y, x)
	}
}

// i4TrialGo is i4TrialsGo for mode m.
func (e *Encoder) i4TrialGo(m int, pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x int) {
	z := &e.z
	var blk [16]int16
	last, _ := e.residualQuant(&blk, &lv[m], &pr[m], src, ss, y, x)
	z.put4(&pr[m], y, x)
	d, t := z.recMetrics(&blk, y, x, src, ss)
	nz := int32(0)
	for _, v := range lv[m][1:] {
		if v != 0 {
			nz++
		}
	}
	for j := range rec[m] {
		r := z[y+j][x : x+4 : x+4]
		rec[m][j] = uint32(r[0]) | uint32(r[1])<<8 | uint32(r[2])<<16 | uint32(r[3])<<24
	}
	res[m] = [4]int32{int32(last), int32(d), int32(t), nz}
}

func (z *ybrBuf) recMetrics16Go(c *[16][16]int16, src []uint8, ss int, st *[16]int32) (sse, sd int) {
	for n := range c {
		y, x := 1+4*(n/4), 8+4*(n%4)
		k, t := z.recMetrics(&c[n], y, x, src[(y-1)*ss+x-8:], ss)
		sse += k
		sd += abs(t-int(st[n])) >> 5
	}
	return sse, sd
}

// trellisGo is TrellisQuantizeBlock (quant_enc.c:597-768) on s's
// probs and level costs.
func (s *costTabs) trellisGo(m *quantMatrix, in, out *[16]int16, ctx0, typ, lambda int) bool {
	first, pb := 0, typ*264
	if typ == 0 {
		first = 1
	}
	lam := int64(lambda)
	p0 := s.probs[pb+(int(bands[first])*3+ctx0)*11]
	thresh := int(m.q[1] * m.q[1] / 4)
	last := first - 1
	for n := 15; n >= first; n-- {
		if v := int(in[zigzag[n]]); v*v > thresh {
			last = n
			break
		}
	}
	if last < 15 {
		last++
	}
	tc := &s.cost[typ&3]
	// Previous position's node scores and cost rows.
	var ps0, ps1 int64
	if ctx0 == 0 {
		ps0 = int64(bitCost[1][p0]) * lam
		ps1 = ps0
	}
	pc0 := &tc[first&15][ctx0&3]
	pc1 := pc0
	best := int64(bitCost[0][p0]) * lam
	pathN, pathD := -1, 0
	var nodes [16][2]lwNode
	if useTrellisAsm {
		a := trellisArgs{in: in, m: m, tc: tc, eob: &s.eob[typ&3], nodes: &nodes, pc0: pc0,
			lam: lam, ps0: ps0, ps1: ps1, best: best, first: first, last: last}
		p := trellisDPAsm(&a)
		pathN, pathD = p>>1, p&1
	} else {
		for n := first; n <= last; n++ {
			j := zigzag[n&15] & 15
			k := min(j, 1)
			neg := in[j] >> 15
			c0 := uint32((in[j]^neg)-neg) + m.sharpen[j]
			l0 := min(int(c0*m.iq[k]>>17), 2047)
			tl := min(int((c0*m.iq[k]+0x80<<9)>>17), 2047)
			row := &tc[min(n+1, 15)]
			if tl == 0 { // only a level 0 node, which cannot end the block
				cs0, bp := ps0+int64(pc0[0])*lam, int8(0)
				if s1 := ps1 + int64(pc1[0])*lam; s1 < cs0 {
					cs0, bp = s1, 1
				}
				nodes[n&15][0] = lwNode{prev: bp}
				ps0, ps1, pc0, pc1 = cs0, maxCost, &row[0], &row[0]
				continue
			}
			nc0, nc1 := &row[min(l0, 2)], &row[min(l0+1, 2)]
			q, w, cc := int64(m.q[k]), 256*trellisWeight[j], int64(c0)*int64(c0)
			nd := &nodes[n&15]
			eob := &s.eob[typ&3][n&15]

			// Node 0 always lives: tl >= l0.
			e := int64(c0) - int64(l0)*q
			cs0, bp := ps0+lwCost(pc0, l0)*lam, int8(0)
			if s1 := ps1 + lwCost(pc1, l0)*lam; s1 < cs0 {
				cs0, bp = s1, 1
			}
			cs0 += w * (e*e - cc)
			nd[0] = lwNode{prev: bp, level: (int16(l0) ^ neg) - neg}
			if l0 != 0 && cs0 < best {
				if t := cs0 + int64(eob[min(l0, 2)])*lam; t < best {
					best, pathN, pathD = t, n, 0
				}
			}

			cs1 := int64(maxCost)
			if l1 := l0 + 1; l1 <= tl {
				e := int64(c0) - int64(l1)*q
				cs1, bp = ps0+lwCost(pc0, l1)*lam, 0
				if s1 := ps1 + lwCost(pc1, l1)*lam; s1 < cs1 {
					cs1, bp = s1, 1
				}
				cs1 += w * (e*e - cc)
				nd[1] = lwNode{prev: bp, level: (int16(l1) ^ neg) - neg}
				if cs1 < best {
					if t := cs1 + int64(eob[min(l1, 2)])*lam; t < best {
						best, pathN, pathD = t, n, 1
					}
				}
			}
			ps0, ps1, pc0, pc1 = cs0, cs1, nc0, nc1
		}
	}
	dc, lv0 := in[0], out[0]
	*in, *out = [16]int16{}, [16]int16{}
	if first == 1 {
		in[0], out[0] = dc, lv0
	}
	if pathN < 0 {
		return false
	}
	d := pathD
	for n := pathN; n >= first; n-- {
		nd := &nodes[n&15][d&1]
		j := zigzag[n&15] & 15
		out[n&15] = nd.level
		in[j] = nd.level * int16(m.q[min(j, 1)])
		d = int(nd.prev)
	}
	return true // the end node's level is nonzero
}

// trellisArgs is trellisDPAsm's input.
type trellisArgs struct {
	in                  *[16]int16
	m                   *quantMatrix
	tc                  *[16][4][128]uint16
	eob                 *[16][4]uint16
	nodes               *[16][2]lwNode
	pc0                 *[128]uint16
	lam, ps0, ps1, best int64
	first, last         int
}

// lwNode is a trellis node with a signed level.
type lwNode struct {
	prev  int8
	level int16
}

// lwCost is VP8LevelCost on a fixed-cost-folded row.
func lwCost(t *[128]uint16, level int) int64 {
	c := int64(t[min(level, 67)&127])
	if level > 67 {
		c += int64(lwFixed(level&2047) - fixedCost[67])
	}
	return c
}

// lwI16T is lwI16 with trellis on the AC blocks, contexts from l and t
// as ReconstructIntra16 updates them.
func (e *Encoder) lwI16T(dq *[16][16]int16, lv *[25][16]int16, l, t nzCtx) int {
	srcY, ys := e.lwSrc, e.img.YStride
	var dc [16]int16
	nz := 0
	for n := range dq {
		y, x := 1+4*(n/4), 8+4*(n%4)
		e.residual(&dq[n], srcY[(y-1)*ys+x-8:], ys, y, x)
		dc[n] = dq[n][0]
	}
	fwhtPlain(&dc)
	if e.q.y2.quantize(&dc, &lv[24]) >= 0 {
		nz |= 1 << 24
	}
	for n := range dq {
		y, x := n/4, n%4
		k := e.lw.trellis(&e.q.y1, &dq[n], &lv[n], int(l[y]+t[x]), 0, e.lam.ti16)
		l[y], t[x] = b2u(k), b2u(k)
		lv[n][0] = 0
		if k {
			nz |= 1 << n
		}
	}
	q := iwht(&dc)
	for n := range dq {
		dq[n][0] = q[n]
	}
	return nz
}

// i4TrialsTrellis is i4Trials with TrellisQuantizeBlock at context ctx.
func (e *Encoder) i4TrialsTrellis(pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x, ctx int) {
	for m := range pr {
		e.i4TrialTrellis(m, pr, src, ss, lv, rec, res, y, x, ctx)
	}
}

// i4TrialTrellis is i4TrialsTrellis for mode m.
func (e *Encoder) i4TrialTrellis(m int, pr *[10][4]uint32, src []uint8, ss int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, y, x, ctx int) {
	z := &e.z
	var blk [16]int16
	z.put4(&pr[m], y, x)
	e.residual(&blk, src, ss, y, x)
	e.lw.trellis(&e.q.y1, &blk, &lv[m], ctx, 3, e.lam.ti4)
	d, t := z.recMetrics(&blk, y, x, src, ss)
	nz := int32(0)
	for _, v := range lv[m][1:] {
		if v != 0 {
			nz++
		}
	}
	for j := range rec[m] {
		r := z[y+j][x : x+4 : x+4]
		rec[m][j] = uint32(r[0]) | uint32(r[1])<<8 | uint32(r[2])<<16 | uint32(r[3])<<24
	}
	res[m] = [4]int32{int32(lastNZ(&lv[m], 0)), int32(d), int32(t), nz}
}

// simpleQuantize is SimpleQuantize for Method 5: the chosen modes again
// with trellis. As in libwebp the i4 contexts stay at the MB's (the
// iterator never updates them there) and UV reads the diffusion errors
// PickBestUV already stored.
func (e *Encoder) simpleQuantize(i16 bool, modes *[16]uint8, uv uint8, mbx, mby int, left *nzCtx) {
	z := &e.z
	srcY, ys := e.lwSrc, e.img.YStride
	if i16 {
		var dq [16][16]int16
		z.predMB(16, 1, 8, modes[0], mbx, mby)
		e.lwI16T(&dq, &e.lv, *left, e.top[mbx])
		for n := range dq {
			z.idctAdd(&dq[n], 1+4*(n/4), 8+4*(n%4))
		}
	} else {
		l, t := *left, e.top[mbx]
		for n := 0; n < 16; n++ {
			y, x := 1+4*(n/4), 8+4*(n%4)
			var pr [10][4]uint32
			z.pred4All(&pr, y, x)
			z.put4(&pr[modes[n]], y, x)
			var blk [16]int16
			e.residual(&blk, srcY[(y-1)*ys+x-8:], ys, y, x)
			e.lw.trellis(&e.q.y1, &blk, &e.lv[n], int(l[n/4]+t[n%4]), 3, e.lam.ti4)
			z.idctAdd(&blk, y, x)
		}
	}
	cs := e.img.CStride
	var dq [8][16]int16
	z.predMB(8, 18, 8, uv, mbx, mby)
	z.predMB(8, 18, 24, uv, mbx, mby)
	var derr [2][3]int8
	e.lwUV(&dq, &e.lv, e.img.Cb[mby*8*cs+mbx*8:], e.img.Cr[mby*8*cs+mbx*8:], cs, mbx, &derr)
}
