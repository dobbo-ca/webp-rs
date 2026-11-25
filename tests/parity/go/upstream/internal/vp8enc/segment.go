package vp8enc

import "github.com/dobbo-ca/webp-go/internal/fmath"

// Per frame segment parameters, libwebp VP8EncSegmentHeader plus dqm.
type segs struct {
	n            int
	quant, level [4]int
	probs        [3]uint8
	dqUVAC       int
	updateMap    bool
}

// GetProba, frame_enc.c:189-193.
func segProb(a, b int) uint8 {
	if a+b == 0 {
		return 255
	}
	return uint8((255*a + (a+b)/2) / (a + b))
}

// SetSegmentProbas, frame_enc.c:202-230.
func (s *segs) setProbs(mbs []mbInfo) {
	var c [4]int
	for i := range mbs {
		c[mbs[i].seg]++
	}
	s.probs = [3]uint8{segProb(c[0]+c[1], c[2]+c[3]), segProb(c[0], c[1]), segProb(c[2], c[3])}
	s.updateMap = s.n > 1 && s.probs != [3]uint8{255, 255, 255}
	if !s.updateMap {
		for i := range mbs {
			mbs[i].seg = 0
		}
	}
}

// GetAlpha, analysis_enc.c:116-125, over a CollectHistogram bin count.
func alphaOf(d *[32]int) int {
	maxV, last := 0, 1
	for k, v := range d {
		if v > 0 {
			maxV = max(maxV, v)
			last = k
		}
	}
	if maxV <= 1 {
		return 0
	}
	return 510 * last / maxV
}

// hist4 splits counts to break store chains.
type hist4 [4][32]int32

// alpha is alphaOf of the four rows' sum.
func (h *hist4) alpha() int {
	maxV, last := 0, 1
	for k := range 32 {
		if v := int(h[0][k] + h[1][k] + h[2][k] + h[3][k]); v > 0 {
			maxV = max(maxV, v)
			last = k
		}
	}
	if maxV <= 1 {
		return 0
	}
	return 510 * last / maxV
}

// bin maps a coefficient to its histogram bucket.
func bin(c int16) int {
	m := c >> 15 // branchless abs, signs are unpredictable
	return min(int((c^m)-m)>>3, 31)
}

// CollectHistogram_C for one block, dsp/enc.c:64-84. Fdct fused, split
// across hist4's rows so the d[bin]++ chain isn't a single memory RMW.
func (e *Encoder) collect(d *hist4, src []uint8, stride, y, x int) {
	z := &e.z
	t00, t01, t02, t03 := fdctRow(diff4(src, z[y][x:]))
	t10, t11, t12, t13 := fdctRow(diff4(src[stride:], z[y+1][x:]))
	t20, t21, t22, t23 := fdctRow(diff4(src[2*stride:], z[y+2][x:]))
	t30, t31, t32, t33 := fdctRow(diff4(src[3*stride:], z[y+3][x:]))
	c0, c1, c2, c3 := fdctCol(t00, t10, t20, t30)
	d[0][bin(c0)]++
	d[1][bin(c1)]++
	d[2][bin(c2)]++
	d[3][bin(c3)]++
	c0, c1, c2, c3 = fdctCol(t01, t11, t21, t31)
	d[1][bin(c0)]++
	d[2][bin(c1)]++
	d[3][bin(c2)]++
	d[0][bin(c3)]++
	c0, c1, c2, c3 = fdctCol(t02, t12, t22, t32)
	d[2][bin(c0)]++
	d[3][bin(c1)]++
	d[0][bin(c2)]++
	d[1][bin(c3)]++
	c0, c1, c2, c3 = fdctCol(t03, t13, t23, t33)
	d[3][bin(c0)]++
	d[0][bin(c1)]++
	d[1][bin(c2)]++
	d[2][bin(c3)]++
}

// AssignSegments, analysis_enc.c:135-222. mb.seg holds alphas on entry.
func kMeans(hist *[256]int, mbs []mbInfo) (centers [4]int, avg int) {
	minA, maxA := 0, 255
	for minA < 255 && hist[minA] == 0 {
		minA++
	}
	for maxA > minA && hist[maxA] == 0 {
		maxA--
	}
	for k := range centers {
		centers[k] = minA + (2*k+1)*(maxA-minA)/8
	}
	var seg [256]uint8
	for iter := 0; iter < 6; iter++ {
		var acc, dist [4]int
		n := 0
		for a := minA; a <= maxA; a++ {
			if hist[a] == 0 {
				continue
			}
			for n+1 < 4 && abs(a-centers[n+1]) < abs(a-centers[n]) {
				n++
			}
			seg[a] = uint8(n)
			dist[n] += a * hist[a]
			acc[n] += hist[a]
		}
		displaced, sum, tw := 0, 0, 0
		for n := range centers {
			if acc[n] == 0 {
				continue
			}
			c := (dist[n] + acc[n]/2) / acc[n]
			displaced += abs(centers[n] - c)
			centers[n] = c
			sum += c * acc[n]
			tw += acc[n]
		}
		avg = (sum + tw/2) / tw
		if displaced < 5 {
			break
		}
	}
	for i := range mbs {
		mbs[i].seg = seg[mbs[i].seg]
	}
	return centers, avg
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// anaDedup skips the TM histogram when TM predicts DC's bytes; a var
// so tests can compare it against both histograms.
var anaDedup = true

// VP8EncAnalyze over the padded source, analysis_enc.c:317-342, 431-485.
func (e *Encoder) analyze(q float32, method int) segs {
	z := &e.z
	ys, cs := e.img.YStride, e.img.CStride
	var hist [256]int
	sumUV := 0
	for mby := 0; mby < e.mbh; mby++ {
		if e.stopped() {
			return segs{} // Encode reports ErrCanceled
		}
		for mbx := 0; mbx < e.mbw; mbx++ {
			z.prepare(&e.img, mbx, mby, e.mbw)
			srcY := e.img.Y[mby*16*ys+mbx*16:]
			srcU := e.img.Cb[mby*8*cs+mbx*8:]
			srcV := e.img.Cr[mby*8*cs+mbx*8:]
			// prepare copied the left column from a prediction.
			if mbx > 0 {
				oy, oc := mby*16*ys+mbx*16-1, mby*8*cs+mbx*8-1
				for j := 0; j < 16; j++ {
					z[1+j][7] = e.img.Y[oy+j*ys]
				}
				for j := 0; j < 8; j++ {
					z[18+j][7] = e.img.Cb[oc+j*cs]
					z[18+j][23] = e.img.Cr[oc+j*cs]
				}
			}
			// Every mode predicts a flat MB exactly: alpha 0.
			if flat(srcY, 0, ys, 16, 16, 1) && flat(srcU, 0, cs, 8, 8, 1) && flat(srcV, 0, cs, 8, 8, 1) &&
				z.edgesAre(1, 8, 16, srcY[0]) && z.edgesAre(18, 8, 8, srcU[0]) && z.edgesAre(18, 24, 8, srcV[0]) {
				hist[255]++
				e.mbs[mby*e.mbw+mbx].seg = 255
				continue
			}
			// FastMBAnalyze returns 0, analysis_enc.c:260-286.
			a := 0
			if method > 1 {
				var p0 [16][16]uint8
				for m := uint8(predDC); m <= predTM; m++ {
					z.predMB(16, 1, 8, m, mbx, mby)
					// TM predicting DC's bytes has DC's alpha.
					if anaDedup && m == predDC {
						p0 = z.predY()
					} else if anaDedup && z.predY() == p0 {
						continue
					}
					a = max(a, e.alphaMB(srcY, ys))
				}
			}
			uva := 0
			// Gray has no chroma; skip UV mode search except at borders.
			if !e.gray || mbx == 0 || mby == 0 {
				var p0 [8][16]uint8
				for m := uint8(predDC); m <= predTM; m++ {
					z.predMB(8, 18, 8, m, mbx, mby)
					z.predMB(8, 18, 24, m, mbx, mby)
					if anaDedup && m == predDC {
						p0 = z.predUV()
					} else if anaDedup && z.predUV() == p0 {
						continue
					}
					ua, ok := e.alphaUV(srcU, srcV, cs)
					if !ok {
						var d hist4
						for n := 0; n < 8; n++ {
							src, bx := srcU, 8
							if n >= 4 {
								src, bx = srcV, 24
							}
							oy, ox := 4*((n&3)>>1), 4*(n&1)
							e.collect(&d, src[oy*cs+ox:], cs, 18+oy, bx+ox)
						}
						ua = d.alpha()
					}
					uva = max(uva, ua)
				}
			}
			mix := clip(255-((3*a+uva+2)>>2), 0, 255)
			hist[mix]++
			e.mbs[mby*e.mbw+mbx].seg = uint8(mix)
			sumUV += uva
		}
	}
	uvAlpha := sumUV / len(e.mbs)
	centers, avg := kMeans(&hist, e.mbs)

	// SetSegmentAlphas, analysis_enc.c:80-100.
	lo, hi := centers[0], centers[0]
	for _, c := range centers {
		lo, hi = min(lo, c), max(hi, c)
	}
	if hi == lo {
		hi = lo + 1
	}
	// VP8SetSegmentParams, quant_enc.c:378-430, sns 50.
	var s segs
	const amp = 0.9 * 50 / 100. / 128
	c0 := qualityCompression(q)
	for i, c := range centers {
		alpha := clip(255*(c-avg)/(hi-lo), -127, 127)
		beta := clip(255*(c-lo)/(hi-lo), 0, 255)
		// float64() stops arm64 fusing the product.
		s.quant[i] = clip(int(127*(1-fmath.Pow(c0, 1-float64(amp*float64(alpha))))), 0, 127)
		s.level[i] = filterLevel(s.quant[i], beta)
	}
	s.dqUVAC = clip((uvAlpha-64)*10/70*50/100, -4, 6)

	// SimplifySegments, quant_enc.c:338-376.
	var remap [4]uint8
	n := 1
	for s1 := 1; s1 < 4; s1++ {
		s2 := 0
		for s2 < n && (s.quant[s2] != s.quant[s1] || s.level[s2] != s.level[s1]) {
			s2++
		}
		remap[s1] = uint8(s2)
		if s2 == n {
			s.quant[n], s.level[n] = s.quant[s1], s.level[s1]
			n++
		}
	}
	if n < 4 {
		for i := range e.mbs {
			e.mbs[i].seg = remap[e.mbs[i].seg]
		}
		for i := n; i < 4; i++ {
			s.quant[i], s.level[i] = s.quant[n-1], s.level[n-1]
		}
	}
	s.n = n
	s.setProbs(e.mbs)
	return s
}

// oneSegment is VP8EncAnalyze with segments 1: no analysis, every
// alpha 0 (ResetAllMBInfo, analysis_enc.c:353-365, 470-472), then
// VP8SetSegmentParams as in analyze.
func (e *Encoder) oneSegment(q float32) segs {
	for i := range e.mbs {
		e.mbs[i].seg = 0
	}
	var s segs
	qi := clip(int(127*(1-qualityCompression(q))), 0, 127)
	for i := range s.quant {
		s.quant[i], s.level[i] = qi, filterLevel(qi, 0)
	}
	s.dqUVAC = clip((0-64)*10/70*50/100, -4, 6)
	s.n = 1
	s.setProbs(e.mbs)
	return s
}

// edgesAre reports whether the block's top and left edges equal v.
func (z *ybrBuf) edgesAre(y, x, n int, v uint8) bool {
	for i := x - 1; i < x+n; i++ {
		if z[y-1][i] != v {
			return false
		}
	}
	for j := y; j < y+n; j++ {
		if z[j][x-1] != v {
			return false
		}
	}
	return true
}

// binStatsGo is the largest of b and the counts of 0, 1, 2 and 3.
func binStatsGo(b []uint8) (last, c0, c1, c2, c3 int) {
	var c [4]int
	for _, v := range b {
		last = max(last, int(v))
		if v < 4 {
			c[v]++
		}
	}
	return last, c[0], c[1], c[2], c[3]
}
