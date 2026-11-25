package vp8dec

// The work buffer holds one macroblock with its neighbours, as libwebp's
// yuv_b: 32 bytes a row, the row above and the column to the left at -1.
const (
	bps  = 32
	yOff = bps + 8
	uOff = yOff + bps*16 + bps
	vOff = uOff + 16
)

var kScan [16]int

func init() {
	for n := range kScan {
		kScan[n] = n&3*4 + n>>2*4*bps
	}
}

func clip8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func avg2(a, b uint8) uint8    { return uint8((int(a) + int(b) + 1) >> 1) }
func avg3(a, b, c uint8) uint8 { return uint8((int(a) + 2*int(b) + int(c) + 2) >> 2) }

func trueMotion(b []uint8, o, size int) {
	tl := int(b[o-bps-1])
	for y := 0; y < size; y++ {
		l := int(b[o+y*bps-1]) - tl
		for x := 0; x < size; x++ {
			b[o+y*bps+x] = clip8(int(b[o-bps+x]) + l)
		}
	}
}

func fill(b []uint8, o, size int, v uint8) {
	for y := 0; y < size; y++ {
		row := b[o+y*bps : o+y*bps+size]
		for x := range row {
			row[x] = v
		}
	}
}

// pred predicts a size x size block (16 luma, 8 chroma) at o; noTop and
// noLeft say that edge is the frame's, which only DC cares about.
func pred(b []uint8, o, size int, mode uint8, noTop, noLeft bool) {
	switch mode {
	case bDC:
		var sum, n int
		if !noTop {
			for i := 0; i < size; i++ {
				sum += int(b[o-bps+i])
			}
			n += size
		}
		if !noLeft {
			for i := 0; i < size; i++ {
				sum += int(b[o+i*bps-1])
			}
			n += size
		}
		dc := uint8(0x80)
		if n > 0 {
			dc = uint8((sum + n/2) / n)
		}
		fill(b, o, size, dc)
	case bTM:
		trueMotion(b, o, size)
	case bVE:
		for y := 0; y < size; y++ {
			copy(b[o+y*bps:o+y*bps+size], b[o-bps:o-bps+size])
		}
	case bHE:
		for y := 0; y < size; y++ {
			v := b[o+y*bps-1]
			row := b[o+y*bps : o+y*bps+size]
			for x := range row {
				row[x] = v
			}
		}
	}
}

// pred4 predicts the 4x4 block at o from the 13 pixels around it.
func pred4(b []uint8, o int, mode uint8) {
	dst := func(x, y int, v uint8) { b[o+x+y*bps] = v }
	top := func(i int) uint8 { return b[o-bps+i] }
	left := func(i int) uint8 { return b[o-1+i*bps] }
	X := b[o-bps-1]
	I, J, K, L := left(0), left(1), left(2), left(3)
	A, B, C, D := top(0), top(1), top(2), top(3)
	switch mode {
	case bDC:
		dc := 4
		for i := 0; i < 4; i++ {
			dc += int(top(i)) + int(left(i))
		}
		fill(b, o, 4, uint8(dc>>3))
	case bTM:
		trueMotion(b, o, 4)
	case bVE:
		v := [4]uint8{avg3(X, A, B), avg3(A, B, C), avg3(B, C, D), avg3(C, D, top(4))}
		for y := 0; y < 4; y++ {
			copy(b[o+y*bps:o+y*bps+4], v[:])
		}
	case bHE:
		r := [4]uint8{avg3(X, I, J), avg3(I, J, K), avg3(J, K, L), avg3(K, L, L)}
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				b[o+y*bps+x] = r[y]
			}
		}
	case bRD:
		dst(0, 3, avg3(J, K, L))
		v := avg3(I, J, K)
		dst(1, 3, v)
		dst(0, 2, v)
		v = avg3(X, I, J)
		dst(2, 3, v)
		dst(1, 2, v)
		dst(0, 1, v)
		v = avg3(A, X, I)
		dst(3, 3, v)
		dst(2, 2, v)
		dst(1, 1, v)
		dst(0, 0, v)
		v = avg3(B, A, X)
		dst(3, 2, v)
		dst(2, 1, v)
		dst(1, 0, v)
		v = avg3(C, B, A)
		dst(3, 1, v)
		dst(2, 0, v)
		dst(3, 0, avg3(D, C, B))
	case bVR:
		v := avg2(X, A)
		dst(0, 0, v)
		dst(1, 2, v)
		v = avg2(A, B)
		dst(1, 0, v)
		dst(2, 2, v)
		v = avg2(B, C)
		dst(2, 0, v)
		dst(3, 2, v)
		dst(3, 0, avg2(C, D))
		dst(0, 3, avg3(K, J, I))
		dst(0, 2, avg3(J, I, X))
		v = avg3(I, X, A)
		dst(0, 1, v)
		dst(1, 3, v)
		v = avg3(X, A, B)
		dst(1, 1, v)
		dst(2, 3, v)
		v = avg3(A, B, C)
		dst(2, 1, v)
		dst(3, 3, v)
		dst(3, 1, avg3(B, C, D))
	case bLD:
		E, F, G, H := top(4), top(5), top(6), top(7)
		dst(0, 0, avg3(A, B, C))
		v := avg3(B, C, D)
		dst(1, 0, v)
		dst(0, 1, v)
		v = avg3(C, D, E)
		dst(2, 0, v)
		dst(1, 1, v)
		dst(0, 2, v)
		v = avg3(D, E, F)
		dst(3, 0, v)
		dst(2, 1, v)
		dst(1, 2, v)
		dst(0, 3, v)
		v = avg3(E, F, G)
		dst(3, 1, v)
		dst(2, 2, v)
		dst(1, 3, v)
		v = avg3(F, G, H)
		dst(3, 2, v)
		dst(2, 3, v)
		dst(3, 3, avg3(G, H, H))
	case bVL:
		E, F, G, H := top(4), top(5), top(6), top(7)
		dst(0, 0, avg2(A, B))
		v := avg2(B, C)
		dst(1, 0, v)
		dst(0, 2, v)
		v = avg2(C, D)
		dst(2, 0, v)
		dst(1, 2, v)
		v = avg2(D, E)
		dst(3, 0, v)
		dst(2, 2, v)
		dst(0, 1, avg3(A, B, C))
		v = avg3(B, C, D)
		dst(1, 1, v)
		dst(0, 3, v)
		v = avg3(C, D, E)
		dst(2, 1, v)
		dst(1, 3, v)
		v = avg3(D, E, F)
		dst(3, 1, v)
		dst(2, 3, v)
		dst(3, 2, avg3(E, F, G))
		dst(3, 3, avg3(F, G, H))
	case bHD:
		v := avg2(I, X)
		dst(0, 0, v)
		dst(2, 1, v)
		v = avg2(J, I)
		dst(0, 1, v)
		dst(2, 2, v)
		v = avg2(K, J)
		dst(0, 2, v)
		dst(2, 3, v)
		dst(0, 3, avg2(L, K))
		dst(3, 0, avg3(A, B, C))
		dst(2, 0, avg3(X, A, B))
		v = avg3(I, X, A)
		dst(1, 0, v)
		dst(3, 1, v)
		v = avg3(J, I, X)
		dst(1, 1, v)
		dst(3, 2, v)
		v = avg3(K, J, I)
		dst(1, 2, v)
		dst(3, 3, v)
		dst(1, 3, avg3(L, K, J))
	case bHU:
		dst(0, 0, avg2(I, J))
		v := avg2(J, K)
		dst(2, 0, v)
		dst(0, 1, v)
		v = avg2(K, L)
		dst(2, 1, v)
		dst(0, 2, v)
		dst(1, 0, avg3(I, J, K))
		v = avg3(J, K, L)
		dst(3, 0, v)
		dst(1, 1, v)
		v = avg3(K, L, L)
		dst(3, 1, v)
		dst(1, 2, v)
		for _, p := range [][2]int{{3, 2}, {2, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 3}} {
			dst(p[0], p[1], L)
		}
	}
}

// wht inverts the luma DC transform into the DC of each of 16 blocks.
func wht(in *[16]int16, out *[384]int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[i]) + int32(in[12+i])
		a1 := int32(in[4+i]) + int32(in[8+i])
		a2 := int32(in[4+i]) - int32(in[8+i])
		a3 := int32(in[i]) - int32(in[12+i])
		tmp[i] = a0 + a1
		tmp[8+i] = a0 - a1
		tmp[4+i] = a3 + a2
		tmp[12+i] = a3 - a2
	}
	for i := 0; i < 4; i++ {
		dc := tmp[i*4] + 3
		a0 := dc + tmp[3+i*4]
		a1 := tmp[1+i*4] + tmp[2+i*4]
		a2 := tmp[1+i*4] - tmp[2+i*4]
		a3 := dc - tmp[3+i*4]
		out[64*i] = int16((a0 + a1) >> 3)
		out[64*i+16] = int16((a3 + a2) >> 3)
		out[64*i+32] = int16((a0 - a1) >> 3)
		out[64*i+48] = int16((a3 - a2) >> 3)
	}
}

func mul1(a int32) int32 { return (a*20091)>>16 + a }
func mul2(a int32) int32 { return (a * 35468) >> 16 }

// idctAdd adds the inverse transform of in to the 4x4 block at o.
func idctAdd(in []int16, b []uint8, o int) {
	var c [16]int32
	for i := 0; i < 4; i++ {
		a := int32(in[i]) + int32(in[8+i])
		bb := int32(in[i]) - int32(in[8+i])
		cc := mul2(int32(in[4+i])) - mul1(int32(in[12+i]))
		dd := mul1(int32(in[4+i])) + mul2(int32(in[12+i]))
		c[4*i], c[4*i+1], c[4*i+2], c[4*i+3] = a+dd, bb+cc, bb-cc, a-dd
	}
	for i := 0; i < 4; i++ {
		dc := c[i] + 4
		a := dc + c[8+i]
		bb := dc - c[8+i]
		cc := mul2(c[4+i]) - mul1(c[12+i])
		dd := mul1(c[4+i]) + mul2(c[12+i])
		r := o + i*bps
		b[r] = clip8(int(b[r]) + int((a+dd)>>3))
		b[r+1] = clip8(int(b[r+1]) + int((bb+cc)>>3))
		b[r+2] = clip8(int(b[r+2]) + int((bb-cc)>>3))
		b[r+3] = clip8(int(b[r+3]) + int((a-dd)>>3))
	}
}

// recon predicts macroblock (mx, my) in the work buffer, adds its residuals
// and stores it in the planes.
func (d *dec) recon(mx, my int, m *mb) {
	f, y := d.f, &d.yuv
	if mx == 0 {
		for j := 0; j < 16; j++ {
			y[yOff+j*bps-1] = 129
		}
		for j := 0; j < 8; j++ {
			y[uOff+j*bps-1], y[vOff+j*bps-1] = 129, 129
		}
		if my > 0 {
			y[yOff-bps-1], y[uOff-bps-1], y[vOff-bps-1] = 129, 129, 129
		} else {
			for i := yOff - bps - 1; i < yOff-bps+20; i++ {
				y[i] = 127
			}
			for i := uOff - bps - 1; i < uOff-bps+8; i++ {
				y[i] = 127
			}
			for i := vOff - bps - 1; i < vOff-bps+8; i++ {
				y[i] = 127
			}
		}
	} else {
		for j := -1; j < 16; j++ {
			copy(y[yOff+j*bps-4:yOff+j*bps], y[yOff+j*bps+12:yOff+j*bps+16])
		}
		for j := -1; j < 8; j++ {
			copy(y[uOff+j*bps-4:uOff+j*bps], y[uOff+j*bps+4:uOff+j*bps+8])
			copy(y[vOff+j*bps-4:vOff+j*bps], y[vOff+j*bps+4:vOff+j*bps+8])
		}
	}
	if my > 0 {
		copy(y[yOff-bps:yOff-bps+16], d.topY[16*mx:])
		copy(y[uOff-bps:uOff-bps+8], d.topU[8*mx:])
		copy(y[vOff-bps:vOff-bps+8], d.topV[8*mx:])
	}
	if m.i4 {
		tr := yOff - bps + 16
		if my > 0 {
			if mx >= d.mbw-1 {
				v := d.topY[16*mx+15]
				y[tr], y[tr+1], y[tr+2], y[tr+3] = v, v, v, v
			} else {
				copy(y[tr:tr+4], d.topY[16*(mx+1):])
			}
		}
		for r := 1; r < 4; r++ { // replicate the top-right pixels below
			copy(y[tr+4*r*bps:tr+4*r*bps+4], y[tr:tr+4])
		}
		for n := 0; n < 16; n++ {
			o := yOff + kScan[n]
			pred4(y[:], o, m.modes[n])
			if d.nonzero[n] {
				idctAdd(d.coeffs[16*n:], y[:], o)
			}
		}
	} else {
		pred(y[:], yOff, 16, m.modes[0], my == 0, mx == 0)
		for n := 0; n < 16; n++ {
			if d.nonzero[n] {
				idctAdd(d.coeffs[16*n:], y[:], yOff+kScan[n])
			}
		}
	}
	pred(y[:], uOff, 8, m.uvmode, my == 0, mx == 0)
	pred(y[:], vOff, 8, m.uvmode, my == 0, mx == 0)
	for n := 0; n < 4; n++ {
		o := bps*(n>>1)*4 + n&1*4
		if d.nonzero[16+n] {
			idctAdd(d.coeffs[16*(16+n):], y[:], uOff+o)
		}
		if d.nonzero[20+n] {
			idctAdd(d.coeffs[16*(20+n):], y[:], vOff+o)
		}
	}
	if my < d.mbh-1 {
		copy(d.topY[16*mx:16*mx+16], y[yOff+15*bps:])
		copy(d.topU[8*mx:8*mx+8], y[uOff+7*bps:])
		copy(d.topV[8*mx:8*mx+8], y[vOff+7*bps:])
	}
	for j := 0; j < 16; j++ {
		copy(f.Y[(16*my+j)*f.YStride+16*mx:], y[yOff+j*bps:yOff+j*bps+16])
	}
	for j := 0; j < 8; j++ {
		copy(f.U[(8*my+j)*f.CStrd+8*mx:], y[uOff+j*bps:uOff+j*bps+8])
		copy(f.V[(8*my+j)*f.CStrd+8*mx:], y[vOff+j*bps:vOff+j*bps+8])
	}
}
