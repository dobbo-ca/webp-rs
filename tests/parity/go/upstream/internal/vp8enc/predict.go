// Ported from golang.org/x/image/vp8 predfunc.go and reconstruct.go, BSD-3-Clause.

package vp8enc

import (
	"encoding/binary"
	"image"
)

// ybrBuf is the x/image prediction workspace.
// Y at (1,8), U at (18,8), V at (18,24).
type ybrBuf [26][32]uint8

// Mode numbers double as implied 4x4 submodes.
const (
	predDC = iota
	predTM
	predVE
	predHE
	predRD
	predVR
	predLD
	predVL
	predHD
	predHU
)

// prepare fills the edges, x/image prepareYBR.
func (z *ybrBuf) prepare(f *image.YCbCr, mbx, mby, mbw int) {
	if mbx == 0 {
		for y := 0; y < 17; y++ {
			z[y][7] = 0x81
		}
		for y := 17; y < 26; y++ {
			z[y][7] = 0x81
			z[y][23] = 0x81
		}
	} else {
		for y := 0; y < 17; y++ {
			z[y][7] = z[y][7+16]
		}
		for y := 17; y < 26; y++ {
			z[y][7] = z[y][15]
			z[y][23] = z[y][31]
		}
	}
	if mby == 0 {
		for x := 7; x < 28; x++ {
			z[0][x] = 0x7f
		}
		for x := 7; x < 16; x++ {
			z[17][x] = 0x7f
		}
		for x := 23; x < 32; x++ {
			z[17][x] = 0x7f
		}
	} else {
		yoff := (16*mby-1)*f.YStride + 16*mbx
		ysrc := f.Y[yoff : yoff+16 : yoff+16]
		binary.LittleEndian.PutUint64(z[0][8:16], binary.LittleEndian.Uint64(ysrc[0:8]))
		binary.LittleEndian.PutUint64(z[0][16:24], binary.LittleEndian.Uint64(ysrc[8:16]))

		coff := (8*mby-1)*f.CStride + 8*mbx
		binary.LittleEndian.PutUint64(z[17][8:16], binary.LittleEndian.Uint64(f.Cb[coff:coff+8]))
		binary.LittleEndian.PutUint64(z[17][24:32], binary.LittleEndian.Uint64(f.Cr[coff:coff+8]))

		if mbx == mbw-1 {
			v := uint32(ysrc[15])
			binary.LittleEndian.PutUint32(z[0][24:28], v*0x01010101)
		} else {
			binary.LittleEndian.PutUint32(z[0][24:28], binary.LittleEndian.Uint32(f.Y[yoff+16:yoff+20]))
		}
	}
	rt := binary.LittleEndian.Uint32(z[0][24:28])
	binary.LittleEndian.PutUint32(z[4][24:28], rt)
	binary.LittleEndian.PutUint32(z[8][24:28], rt)
	binary.LittleEndian.PutUint32(z[12][24:28], rt)
}

// predMBGo predicts an n by n block, n 16 or 8.
func (z *ybrBuf) predMBGo(n, y, x int, mode uint8, mbx, mby int) {
	switch mode {
	case predDC:
		avg := z.dcAvg(n, y, x, mbx, mby)
		for j := 0; j < n; j++ {
			fill8(z[y+j][x:x+n], avg)
		}
	case predTM:
		z.predTM(n, y, x)
	case predVE:
		if n == 16 { // 16-byte stores match the 16-byte reloads
			top := [16]uint8(z[y-1][x : x+16])
			for j := 0; j < 16; j++ {
				*(*[16]uint8)(z[y+j][x:]) = top
			}
			break
		}
		w := binary.LittleEndian.Uint64(z[y-1][x:])
		for j := 0; j < n; j++ {
			binary.LittleEndian.PutUint64(z[y+j][x:], w)
		}
	case predHE:
		for j := 0; j < n; j++ {
			fill8(z[y+j][x:x+n], z[y+j][x-1])
		}
	}
}

// dcAvg is the DC prediction value of an n by n block.
func (z *ybrBuf) dcAvg(n, y, x, mbx, mby int) uint8 {
	log2n := 3
	if n == 16 {
		log2n = 4
	}
	sum, avg := 0, 128
	switch {
	case mbx == 0 && mby == 0:
	case mby == 0:
		for j := 0; j < n; j++ {
			sum += int(z[y+j][x-1])
		}
		avg = (sum + n/2) >> log2n
	case mbx == 0:
		for i := 0; i < n; i++ {
			sum += int(z[y-1][x+i])
		}
		avg = (sum + n/2) >> log2n
	default:
		for i := 0; i < n; i++ {
			sum += int(z[y-1][x+i]) + int(z[y+i][x-1])
		}
		avg = (sum + n) >> (log2n + 1)
	}
	return uint8(avg)
}

// edgeSSEGo is predSSEGo of the DC, TM, VE and HE
// predictions at (y, x), leaving z unchanged.
func (z *ybrBuf) edgeSSEGo(d *[4]int, src []uint8, stride, n, y, x, mbx, mby int) {
	dc := int32(z.dcAvg(n, y, x, mbx, mby))
	a := int32(z[y-1][x-1])
	top := z[y-1][x : x+n]
	*d = [4]int{}
	for j := 0; j < n; j++ {
		l := int32(z[y+j][x-1])
		for i, v := range src[j*stride : j*stride+n] {
			s, t := int32(v), int32(top[i])
			k0, k1 := s-dc, s-int32(clip8(l-a+t))
			k2, k3 := s-t, s-l
			d[0] += int(k0 * k0)
			d[1] += int(k1 * k1)
			d[2] += int(k2 * k2)
			d[3] += int(k3 * k3)
		}
	}
}

// predTMGo is TrueMotion prediction of an n by n block.
func (z *ybrBuf) predTMGo(n, y, x int) {
	delta0 := -int32(z[y-1][x-1])
	for j := 0; j < n; j++ {
		delta1 := delta0 + int32(z[y+j][x-1])
		for i := 0; i < n; i++ {
			z[y+j][x+i] = clip8(delta1 + int32(z[y-1][x+i]))
		}
	}
}

// fill8 sets b, a multiple of 8 long, to v a word at a time.
func fill8(b []uint8, v uint8) {
	w := uint64(v) * 0x0101010101010101
	for i := 0; i+8 <= len(b); i += 8 {
		binary.LittleEndian.PutUint64(b[i:], w)
	}
}

// pred4AllGo writes all ten 4x4 predictions at (y, x) to pr as
// little-endian rows, gathering the edges once.
func (z *ybrBuf) pred4AllGo(pr *[10][4]uint32, y, x int) {
	t := z[y-1][x-1 : x+8 : x+8]
	A := int32(t[0])
	T0, T1, T2, T3 := int32(t[1]), int32(t[2]), int32(t[3]), int32(t[4])
	T4, T5, T6, T7 := int32(t[5]), int32(t[6]), int32(t[7]), int32(t[8])
	L0, L1, L2, L3 := int32(z[y][x-1]), int32(z[y+1][x-1]), int32(z[y+2][x-1]), int32(z[y+3][x-1])

	dc := uint32(T0+T1+T2+T3+L0+L1+L2+L3+4) >> 3 * 0x01010101
	pr[predDC] = [4]uint32{dc, dc, dc, dc}
	for j, l := range [4]int32{L0, L1, L2, L3} {
		d := l - A
		pr[predTM][j] = word(uint32(clip8(d+T0)), uint32(clip8(d+T1)), uint32(clip8(d+T2)), uint32(clip8(d+T3)))
	}
	ve := word(avg3(A, T0, T1), avg3(T0, T1, T2), avg3(T1, T2, T3), avg3(T2, T3, T4))
	pr[predVE] = [4]uint32{ve, ve, ve, ve}
	pr[predHE] = [4]uint32{avg3(A, L0, L1) * 0x01010101, avg3(L0, L1, L2) * 0x01010101,
		avg3(L1, L2, L3) * 0x01010101, avg3(L2, L3, L3) * 0x01010101}

	// Left and top-left three-tap filters.
	srq, rqp, qpa, pab := avg3(L3, L2, L1), avg3(L2, L1, L0), avg3(L1, L0, A), avg3(L0, A, T0)
	// Top three-tap filters, abc centred on T0.
	abc, bcd, cde := avg3(A, T0, T1), avg3(T0, T1, T2), avg3(T1, T2, T3)
	pr[predRD] = [4]uint32{word(pab, abc, bcd, cde), word(qpa, pab, abc, bcd),
		word(rqp, qpa, pab, abc), word(srq, rqp, qpa, pab)}

	ab, bc, cd, de := avg2(A, T0), avg2(T0, T1), avg2(T1, T2), avg2(T2, T3)
	pr[predVR] = [4]uint32{word(ab, bc, cd, de), word(pab, abc, bcd, cde),
		word(qpa, ab, bc, cd), word(rqp, pab, abc, bcd)}

	// LD and VL filter only the top row, t0 centred on T1.
	t0, t1, t2, t3 := avg3(T0, T1, T2), avg3(T1, T2, T3), avg3(T2, T3, T4), avg3(T3, T4, T5)
	t4, t5, t6 := avg3(T4, T5, T6), avg3(T5, T6, T7), avg3(T6, T7, T7)
	pr[predLD] = [4]uint32{word(t0, t1, t2, t3), word(t1, t2, t3, t4),
		word(t2, t3, t4, t5), word(t3, t4, t5, t6)}
	v0, v1, v2, v3 := avg2(T0, T1), avg2(T1, T2), avg2(T2, T3), avg2(T3, T4)
	pr[predVL] = [4]uint32{word(v0, v1, v2, v3), word(t0, t1, t2, t3),
		word(v1, v2, v3, t4), word(t1, t2, t3, t5)}

	sr, rq, qp, pa := avg2(L3, L2), avg2(L2, L1), avg2(L1, L0), avg2(L0, A)
	pr[predHD] = [4]uint32{word(pa, pab, abc, bcd), word(qp, qpa, pa, pab),
		word(rq, rqp, qp, qpa), word(sr, srq, rq, rqp)}

	pq, qr, rs := avg2(L0, L1), avg2(L1, L2), avg2(L2, L3)
	pqr, qrs, rss, s := avg3(L0, L1, L2), avg3(L1, L2, L3), avg3(L2, L3, L3), uint32(L3)
	pr[predHU] = [4]uint32{word(pq, pqr, qr, qrs), word(qr, qrs, rs, rss),
		word(rs, rss, s, s), s * 0x01010101}
}

func avg2(a, b int32) uint32    { return uint32(a+b+1) >> 1 }
func avg3(a, b, c int32) uint32 { return uint32(a+2*b+c+2) >> 2 }

func word(a, b, c, d uint32) uint32 { return a | b<<8 | c<<16 | d<<24 }
