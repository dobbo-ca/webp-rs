package vp8enc

import "github.com/dobbo-ca/webp-go/internal/fmath"

// RFC 14.1 dc/ac_qlookup.
var dcTable = [128]uint16{
	4, 5, 6, 7, 8, 9, 10, 10,
	11, 12, 13, 14, 15, 16, 17, 17,
	18, 19, 20, 20, 21, 21, 22, 22,
	23, 23, 24, 25, 25, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36,
	37, 37, 38, 39, 40, 41, 42, 43,
	44, 45, 46, 46, 47, 48, 49, 50,
	51, 52, 53, 54, 55, 56, 57, 58,
	59, 60, 61, 62, 63, 64, 65, 66,
	67, 68, 69, 70, 71, 72, 73, 74,
	75, 76, 76, 77, 78, 79, 80, 81,
	82, 83, 84, 85, 86, 87, 88, 89,
	91, 93, 95, 96, 98, 100, 101, 102,
	104, 106, 108, 110, 112, 114, 116, 118,
	122, 124, 126, 128, 130, 132, 134, 136,
	138, 140, 143, 145, 148, 151, 154, 157,
}

var acTable = [128]uint16{
	4, 5, 6, 7, 8, 9, 10, 11,
	12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 24, 25, 26, 27,
	28, 29, 30, 31, 32, 33, 34, 35,
	36, 37, 38, 39, 40, 41, 42, 43,
	44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 60,
	62, 64, 66, 68, 70, 72, 74, 76,
	78, 80, 82, 84, 86, 88, 90, 92,
	94, 96, 98, 100, 102, 104, 106, 108,
	110, 112, 114, 116, 119, 122, 125, 128,
	131, 134, 137, 140, 143, 146, 149, 152,
	155, 158, 161, 164, 167, 170, 173, 177,
	181, 185, 189, 193, 197, 201, 205, 209,
	213, 217, 221, 225, 229, 234, 239, 245,
	249, 254, 259, 264, 269, 274, 279, 284,
}

// RFC 13.3 zigzag scan order.
var zigzag = [16]uint8{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}

// freqSharpen is libwebp kFreqSharpening, quant_enc.c:178.
var freqSharpen = [16]uint32{0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90}

// libwebp VP8Matrix, indexed [dc, ac].
type quantMatrix struct {
	q, iq, bias [2]uint32
	sharpen     [16]uint32
}

// quant mirrors x/image vp8 quant.go:10-14.
type quant struct{ y1, y2, uv quantMatrix }

// libwebp quant_enc.c ExpandMatrix.
func (m *quantMatrix) expand(biasDC, biasAC byte, sharpenAC bool) {
	b := [2]uint32{uint32(biasDC), uint32(biasAC)}
	for k := 0; k < 2; k++ {
		m.iq[k] = (1 << 17) / m.q[k]
		m.bias[k] = b[k] << 9
	}
	if sharpenAC {
		for j := range m.sharpen {
			k := j
			if k > 1 {
				k = 1
			}
			m.sharpen[j] = freqSharpen[j] * m.q[k] >> 11
		}
	}
}

// Fixed by libwebp, quant_enc.c:408-418.
const dqUVDC = -2

func clip(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Y1 and Y2 deltas are 0.
func newQuant(qi, dqUVAC int) quant {
	var q quant
	q.y1.q = [2]uint32{uint32(dcTable[qi]), uint32(acTable[qi])}
	q.y1.expand(96, 110, true)

	y2AC := uint32(acTable[qi]) * 155 / 100
	if y2AC < 8 {
		y2AC = 8
	}
	q.y2.q = [2]uint32{uint32(dcTable[qi]) * 2, y2AC}
	q.y2.expand(96, 108, false)

	q.uv.q = [2]uint32{uint32(dcTable[clip(qi+dqUVDC, 0, 117)]), uint32(acTable[clip(qi+dqUVAC, 0, 127)])}
	q.uv.expand(110, 115, false)

	return q
}

// QuantizeBlock_C minus zthresh and MAX_LEVEL.
// Branchless: hottest function at method 4.
// Returns the last nonzero zigzag index, or -1.
func (m *quantMatrix) quantizeGo(in, out *[16]int16) int {
	v := int32(in[0])
	s := v >> 31
	a := uint32(v^s-s) + m.sharpen[0]
	nz := int32((a*m.iq[0]+m.bias[0])>>17) ^ s - s
	out[0] = int16(nz)
	in[0] = int16(nz * int32(m.q[0]))
	last := -1
	if nz != 0 {
		last = 0
	}
	// DC peeled so the AC loop carries no reloads.
	iq, bias, q := m.iq[1], m.bias[1], int32(m.q[1])
	for n := 1; n < 16; n++ {
		j := zigzag[n] & 15
		v := int32(in[j])
		s := v >> 31
		a := uint32(v^s-s) + m.sharpen[j]
		level := int32((a*iq+bias)>>17) ^ s - s
		out[n] = int16(level)
		in[j] = int16(level * q)
		if level != 0 {
			last = n
		}
	}
	return last
}

// quantSSEGo is quantizeGo that also returns the coefficient SSE,
// so callers need no copy of the input.
func (m *quantMatrix) quantSSEGo(in, out *[16]int16) (last, d int) {
	v := int32(in[0])
	s := v >> 31
	a := uint32(v^s-s) + m.sharpen[0]
	nz := int32((a*m.iq[0]+m.bias[0])>>17) ^ s - s
	out[0] = int16(nz)
	dq := nz * int32(m.q[0])
	in[0] = int16(dq)
	d = int((v - dq) * (v - dq))
	last = -1
	if nz != 0 {
		last = 0
	}
	iq, bias, q := m.iq[1], m.bias[1], int32(m.q[1])
	// Unrolled so every index is a constant.
	ac := func(n, j int) {
		v := int32(in[j])
		s := v >> 31
		a := uint32(v^s-s) + m.sharpen[j]
		level := int32((a*iq+bias)>>17) ^ s - s
		out[n] = int16(level)
		dq := level * q
		in[j] = int16(dq)
		d += int((v - dq) * (v - dq))
		if level != 0 {
			last = n
		}
	}
	ac(1, 1)
	ac(2, 4)
	ac(3, 8)
	ac(4, 5)
	ac(5, 2)
	ac(6, 3)
	ac(7, 6)
	ac(8, 9)
	ac(9, 12)
	ac(10, 13)
	ac(11, 10)
	ac(12, 7)
	ac(13, 11)
	ac(14, 14)
	ac(15, 15)
	return last, d
}

// QualityToCompression, quant_enc.c:302-313. Pow mirrors C.
func qualityCompression(q float32) float64 {
	c := float64(q) / 100
	lin := 2*c - 1
	if c < 0.75 {
		lin = c * (2. / 3.)
	}
	return fmath.Pow(lin, 1./3)
}
