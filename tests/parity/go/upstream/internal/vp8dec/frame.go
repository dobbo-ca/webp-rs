package vp8dec

import "github.com/dobbo-ca/webp-go/internal/vp8enc"

// Prediction modes, libwebp's order. The 16x16 modes share the first four.
const (
	bDC = iota
	bTM
	bVE
	bHE
	bRD
	bVR
	bLD
	bVL
	bHD
	bHU
)

// kYModesIntra4 is libwebp's tree for sub-block modes (negated leaves).
var kYModesIntra4 = [18]int8{
	-bDC, 1,
	-bTM, 2,
	-bVE, 3,
	4, 6,
	-bHE, 5,
	-bRD, -bVR,
	-bLD, 7,
	-bVL, 8,
	-bHD, -bHU,
}

func (d *dec) row(my int) error {
	if err := d.ctx.Err(); err != nil {
		return err
	}
	d.intraL = [4]uint8{}
	d.nzL = [9]uint8{}
	for x := 0; x < d.mbw; x++ {
		d.parseIntraMode(x)
	}
	if d.br.eof {
		return errTruncated
	}
	tb := &d.parts[my&(d.nparts-1)]
	for x := 0; x < d.mbw; x++ {
		m := &d.mbs[x]
		skip := d.useSkip && m.skip
		if !skip {
			skip = d.parseResiduals(x, m, tb)
		} else {
			t := &d.nzT[x]
			for i := 0; i < 8; i++ {
				t[i], d.nzL[i] = 0, 0
			}
			if !m.i4 {
				t[8], d.nzL[8] = 0, 0
			}
			d.nonzero = [24]bool{}
		}
		fi := &d.finfos[x]
		if d.fType > 0 {
			*fi = d.fs[m.segment][b2i(m.i4)]
			fi.inner = fi.inner || !skip
		}
		if tb.eof {
			return errTruncated
		}
		d.recon(x, my, m)
	}
	if d.fType > 0 {
		d.filterRow(my)
	}
	return nil
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (d *dec) parseIntraMode(x int) {
	br := &d.br
	m := &d.mbs[x]
	*m = mb{}
	if d.seg.updateMap {
		if br.bit(d.seg.probs[0]) == 0 {
			m.segment = uint8(br.bit(d.seg.probs[1]))
		} else {
			m.segment = uint8(br.bit(d.seg.probs[2])) + 2
		}
	}
	if d.useSkip {
		m.skip = br.bit(d.skipP) != 0
	}
	top := d.intraT[4*x : 4*x+4]
	left := &d.intraL
	m.i4 = br.bit(145) == 0
	if !m.i4 {
		var y uint8
		if br.bit(156) != 0 {
			y = bHE
			if br.bit(128) != 0 {
				y = bTM
			}
		} else {
			y = bDC
			if br.bit(163) != 0 {
				y = bVE
			}
		}
		for i := range m.modes {
			m.modes[i] = y
		}
		for i := 0; i < 4; i++ {
			top[i], left[i] = y, y
		}
	} else {
		for y := 0; y < 4; y++ {
			ym := left[y]
			for x := 0; x < 4; x++ {
				p := &vp8enc.BModeProbs[top[x]][ym]
				i := int(kYModesIntra4[br.bit(p[0])])
				for i > 0 {
					i = int(kYModesIntra4[2*i+br.bit(p[i])])
				}
				ym = uint8(-i)
				top[x] = ym
				m.modes[4*y+x] = ym
			}
			left[y] = ym
		}
	}
	switch {
	case br.bit(142) == 0:
		m.uvmode = bDC
	case br.bit(114) == 0:
		m.uvmode = bVE
	case br.bit(183) != 0:
		m.uvmode = bTM
	default:
		m.uvmode = bHE
	}
}

// parseResiduals reads one macroblock's coefficients into d.coeffs and
// reports whether it has none (libwebp's "skip").
func (d *dec) parseResiduals(x int, m *mb, br *boolDec) bool {
	d.coeffs = [384]int16{}
	q := &d.q[m.segment]
	top, left := &d.nzT[x], &d.nzL
	any := false
	first, t := 0, 3
	if !m.i4 {
		var dc [16]int16
		ctx := int(top[8] + left[8])
		nz := d.getCoeffs(br, 1, ctx, &q.y2, 0, dc[:])
		top[8], left[8] = b2u(nz > 0), b2u(nz > 0)
		if nz > 1 {
			wht(&dc, &d.coeffs)
		} else {
			dc0 := int16((int32(dc[0]) + 3) >> 3)
			for i := 0; i < 256; i += 16 {
				d.coeffs[i] = dc0
			}
		}
		first, t = 1, 0
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			n := 4*y + x
			ctx := int(left[y] + top[x])
			nz := d.getCoeffs(br, t, ctx, &q.y1, first, d.coeffs[16*n:16*n+16])
			l := b2u(nz > first)
			left[y], top[x] = l, l
			d.nonzero[n] = nz > 1 || d.coeffs[16*n] != 0
			any = any || d.nonzero[n]
		}
	}
	for ch := 0; ch < 2; ch++ { // U then V
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				n := 16 + 4*ch + 2*y + x
				ctx := int(left[4+2*ch+y] + top[4+2*ch+x])
				nz := d.getCoeffs(br, 2, ctx, &q.uv, 0, d.coeffs[16*n:16*n+16])
				l := b2u(nz > 0)
				left[4+2*ch+y], top[4+2*ch+x] = l, l
				d.nonzero[n] = nz > 1 || d.coeffs[16*n] != 0
				any = any || d.nonzero[n]
			}
		}
	}
	return !any
}

func b2u(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}

// getCoeffs reads one block's tokens from position n and returns the
// position after the last nonzero coefficient (n if there is none).
func (d *dec) getCoeffs(br *boolDec, t, ctx int, dq *[2]int32, n int, out []int16) int {
	pr := &d.probs[t]
	p := &pr[vp8enc.Bands[n]][ctx]
	for ; n < 16; n++ {
		if br.bit(p[0]) == 0 {
			return n
		}
		for br.bit(p[1]) == 0 {
			n++
			p = &pr[vp8enc.Bands[n]][0]
			if n == 16 {
				return 16
			}
		}
		pn := &pr[vp8enc.Bands[n+1]]
		var v int32
		if br.bit(p[2]) == 0 {
			v = 1
			p = &pn[1]
		} else {
			v = largeValue(br, p)
			p = &pn[2]
		}
		if br.bit(0x80) != 0 {
			v = -v
		}
		out[vp8enc.Zigzag[n]] = int16(v * dq[b2i(n > 0)])
	}
	return 16
}

func largeValue(br *boolDec, p *[11]uint8) int32 {
	if br.bit(p[3]) == 0 {
		if br.bit(p[4]) == 0 {
			return 2
		}
		return 3 + int32(br.bit(p[5]))
	}
	if br.bit(p[6]) == 0 {
		if br.bit(p[7]) == 0 {
			return 5 + int32(br.bit(159))
		}
		v := 7 + 2*int32(br.bit(165))
		return v + int32(br.bit(145))
	}
	bit1 := br.bit(p[8])
	bit0 := br.bit(p[9+bit1])
	cat := 2*bit1 + bit0
	var v int32
	for _, pr := range vp8enc.CatProbs[cat] {
		v += v + int32(br.bit(pr))
	}
	return v + 3 + 8<<uint(cat)
}
