// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fmath is the math.Pow webp-go uses, written so that every
// architecture computes what amd64 with FMA (c7i) computes, bit for bit.
//
// On amd64 math.Pow calls archExp (exp_amd64.s, its VFMADD path when the
// CPU has FMA) and archLog (log_amd64.s); arm64 runs other code and Go
// may fuse x*y+z there. These are math's pow.go, exp_amd64.s and
// log_amd64.s (Go 1.27) in Go: math.FMA where the asm fuses, and
// float64(x*y) everywhere else, which the Go spec never fuses.
package fmath

import "math"

// Pow is math.Pow on amd64 with FMA.
func Pow(x, y float64) float64 {
	switch {
	case y == 0 || x == 1:
		return 1
	case y == 1:
		return x
	case math.IsNaN(x) || math.IsNaN(y):
		return math.NaN()
	case x == 0:
		switch {
		case y < 0:
			if math.Signbit(x) && isOddInt(y) {
				return math.Inf(-1)
			}
			return math.Inf(1)
		case y > 0:
			if math.Signbit(x) && isOddInt(y) {
				return x
			}
			return 0
		}
	case math.IsInf(y, 0):
		switch {
		case x == -1:
			return 1
		case (math.Abs(x) < 1) == math.IsInf(y, 1):
			return 0
		default:
			return math.Inf(1)
		}
	case math.IsInf(x, 0):
		if math.IsInf(x, -1) {
			return Pow(1/x, -y) // Pow(-0, -y)
		}
		switch {
		case y < 0:
			return 0
		case y > 0:
			return math.Inf(1)
		}
	case y == 0.5:
		return math.Sqrt(x)
	case y == -0.5:
		return 1 / math.Sqrt(x)
	}

	yi, yf := math.Modf(math.Abs(y))
	if yf != 0 && x < 0 {
		return math.NaN()
	}
	if yi >= 1<<63 {
		switch {
		case x == -1:
			return 1
		case (math.Abs(x) < 1) == (y > 0):
			return 0
		default:
			return math.Inf(1)
		}
	}

	a1 := 1.0
	ae := 0
	if yf != 0 {
		if yf > 0.5 {
			yf--
			yi++
		}
		a1 = exp(float64(yf * log(x)))
	}
	x1, xe := math.Frexp(x)
	for i := int64(yi); i != 0; i >>= 1 {
		if xe < -1<<12 || 1<<12 < xe {
			ae += xe
			break
		}
		if i&1 == 1 {
			a1 = float64(a1 * x1)
			ae += xe
		}
		x1 = float64(x1 * x1)
		xe <<= 1
		if x1 < .5 {
			x1 += x1
			xe--
		}
	}
	if y < 0 {
		a1 = 1 / a1
		ae = -ae
	}
	return math.Ldexp(a1, ae)
}

func isOddInt(x float64) bool {
	if math.Abs(x) >= 1<<53 {
		return false
	}
	xi, xf := math.Modf(x)
	return xf == 0 && int64(xi)&1 == 1
}

// exp is exp_amd64.s's avxfma path.
func exp(x float64) float64 {
	const (
		log2e    = 1.4426950408889634073599246810018920
		ln2U     = 0.69314718055966295651160180568695068359375
		ln2L     = 0.28235290563031577122588448175013436025525412068e-12
		overflow = 7.09782712893384e+02
	)
	b := math.Float64bits(x)
	if b&^(1<<63) >= 0x7FF0000000000000 {
		if b == 0xFFF0000000000000 {
			return 0
		}
		return x // NaN or +Inf
	}
	if x > overflow {
		return math.Inf(1)
	}
	k := cvtsd2sl(float64(log2e * x))
	kf := float64(k)
	x = math.FMA(-kf, ln2U, x)
	x = math.FMA(-kf, ln2L, x)
	x = float64(x * 0.0625)
	p := 2.4801587301587301587e-5
	for _, c := range [...]float64{1.9841269841269841270e-4, 1.3888888888888888889e-3, 8.3333333333333333333e-3,
		4.1666666666666666667e-2, 1.6666666666666666667e-1, 0.5, 1.0} {
		p = math.FMA(p, x, c)
	}
	x = float64(x * p)
	for range 3 {
		p = x + 2
		x = float64(x * p)
	}
	p = x + 2
	x = math.FMA(x, p, 1)
	// fr * 2**k, as the asm's ldexp
	bx := int64(k) + 0x3FF
	if bx <= 0 {
		if bx < -52 {
			return 0
		}
		x = float64(x * math.Float64frombits(uint64(bx+0x3FE)<<52))
		return float64(x * math.Float64frombits(1<<52))
	}
	if bx >= 0x7FF {
		return math.Inf(1)
	}
	return float64(x * math.Float64frombits(uint64(bx)<<52))
}

// cvtsd2sl is CVTSD2SL: round to nearest even, 0x80000000 if out of range.
func cvtsd2sl(t float64) int32 {
	r := math.RoundToEven(t)
	if !(r >= math.MinInt32 && r <= math.MaxInt32) {
		return math.MinInt32
	}
	return int32(r)
}

// log is log_amd64.s, its Frexp from the exponent bits included.
func log(x float64) float64 {
	const (
		hSqrt2 = 7.07106781186547524401e-01
		ln2Hi  = 6.93147180369123816490e-01
		ln2Lo  = 1.90821492927058770002e-10
		l1     = 6.666666666666735130e-01
		l2     = 3.999999999940941908e-01
		l3     = 2.857142874366239149e-01
		l4     = 2.222219843214978396e-01
		l5     = 1.818357216161805012e-01
		l6     = 1.531383769920937332e-01
		l7     = 1.479819860511658591e-01
	)
	b := math.Float64bits(x)
	switch {
	case b&^(1<<63) == 0:
		return math.Inf(-1)
	case int64(b) < 0:
		return math.Float64frombits(0x7FF8000000000001)
	case b >= 0x7FF0000000000000:
		return x // +Inf or NaN
	}
	f1 := math.Float64frombits(b&0x000FFFFFFFFFFFFF | 0x3FE0000000000000)
	k := float64(int32(b>>52&0x7FF) - 0x3FE)
	if hSqrt2 >= f1 { // CMPSD $5: not less than
		k--
		f1 = float64(f1 * 2)
	}
	f := f1 - 1
	s := f / (2 + f)
	s2 := float64(s * s)
	s4 := float64(s2 * s2)
	t1 := float64(float64(l7*s4) + l5)
	t1 = float64(float64(t1*s4) + l3)
	t1 = float64(float64(t1*s4) + l1)
	t1 = float64(s2 * t1)
	t2 := float64(float64(l6*s4) + l4)
	t2 = float64(float64(t2*s4) + l2)
	t2 = float64(s4 * t2)
	r := t1 + t2
	hfsq := float64(float64(0.5*f) * f)
	u := float64(s*(hfsq+r)) + float64(ln2Lo*k)
	return float64(k*ln2Hi) - ((hfsq - u) - f)
}
