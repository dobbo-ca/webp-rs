package vp8dec

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func sclip1(v int) int { return max(-128, min(v, 127)) }
func sclip2(v int) int { return max(-16, min(v, 15)) }

// px is the byte at p + k*step.
func filter2(b []uint8, p, step int) {
	p1, p0, q0, q1 := int(b[p-2*step]), int(b[p-step]), int(b[p]), int(b[p+step])
	a := 3*(q0-p0) + sclip1(p1-q1)
	a1 := sclip2((a + 4) >> 3)
	a2 := sclip2((a + 3) >> 3)
	b[p-step] = clip8(p0 + a2)
	b[p] = clip8(q0 - a1)
}

func filter4(b []uint8, p, step int) {
	p1, p0, q0, q1 := int(b[p-2*step]), int(b[p-step]), int(b[p]), int(b[p+step])
	a := 3 * (q0 - p0)
	a1 := sclip2((a + 4) >> 3)
	a2 := sclip2((a + 3) >> 3)
	a3 := (a1 + 1) >> 1
	b[p-2*step] = clip8(p1 + a3)
	b[p-step] = clip8(p0 + a2)
	b[p] = clip8(q0 - a1)
	b[p+step] = clip8(q1 - a3)
}

func filter6(b []uint8, p, step int) {
	p2, p1, p0 := int(b[p-3*step]), int(b[p-2*step]), int(b[p-step])
	q0, q1, q2 := int(b[p]), int(b[p+step]), int(b[p+2*step])
	a := sclip1(3*(q0-p0) + sclip1(p1-q1))
	a1 := (27*a + 63) >> 7
	a2 := (18*a + 63) >> 7
	a3 := (9*a + 63) >> 7
	b[p-3*step] = clip8(p2 + a3)
	b[p-2*step] = clip8(p1 + a2)
	b[p-step] = clip8(p0 + a1)
	b[p] = clip8(q0 - a1)
	b[p+step] = clip8(q1 - a2)
	b[p+2*step] = clip8(q2 - a3)
}

func hev(b []uint8, p, step, thresh int) bool {
	p1, p0, q0, q1 := int(b[p-2*step]), int(b[p-step]), int(b[p]), int(b[p+step])
	return abs(p1-p0) > thresh || abs(q1-q0) > thresh
}

func needs(b []uint8, p, step, t int) bool {
	p1, p0, q0, q1 := int(b[p-2*step]), int(b[p-step]), int(b[p]), int(b[p+step])
	return 4*abs(p0-q0)+abs(p1-q1) <= t
}

func needs2(b []uint8, p, step, t, it int) bool {
	p3, p2, p1, p0 := int(b[p-4*step]), int(b[p-3*step]), int(b[p-2*step]), int(b[p-step])
	q0, q1, q2, q3 := int(b[p]), int(b[p+step]), int(b[p+2*step]), int(b[p+3*step])
	if 4*abs(p0-q0)+abs(p1-q1) > t {
		return false
	}
	return abs(p3-p2) <= it && abs(p2-p1) <= it && abs(p1-p0) <= it &&
		abs(q3-q2) <= it && abs(q2-q1) <= it && abs(q1-q0) <= it
}

// simple filters size pixels along an edge: hstride crosses the edge,
// vstride walks along it.
func simple(b []uint8, p, hstride, vstride, thresh int) {
	t := 2*thresh + 1
	for i := 0; i < 16; i++ {
		if needs(b, p+i*vstride, hstride, t) {
			filter2(b, p+i*vstride, hstride)
		}
	}
}

func loop(b []uint8, p, hstride, vstride, size, thresh, ithresh, hevT int, edge bool) {
	t := 2*thresh + 1
	for ; size > 0; size-- {
		if needs2(b, p, hstride, t, ithresh) {
			switch {
			case hev(b, p, hstride, hevT):
				filter2(b, p, hstride)
			case edge:
				filter6(b, p, hstride)
			default:
				filter4(b, p, hstride)
			}
		}
		p += vstride
	}
}

// filterRow runs the loop filter over macroblock row my, in place.
func (d *dec) filterRow(my int) {
	f := d.f
	for mx := 0; mx < d.mbw; mx++ {
		fi := &d.finfos[mx]
		limit := int(fi.limit)
		if limit == 0 {
			continue
		}
		y := 16*my*f.YStride + 16*mx
		u := 8*my*f.CStrd + 8*mx
		ys, cs := f.YStride, f.CStrd
		if d.fType == 1 {
			if mx > 0 {
				simple(f.Y, y, 1, ys, limit+4)
			}
			if fi.inner {
				for k := 4; k < 16; k += 4 {
					simple(f.Y, y+k, 1, ys, limit)
				}
			}
			if my > 0 {
				simple(f.Y, y, ys, 1, limit+4)
			}
			if fi.inner {
				for k := 4; k < 16; k += 4 {
					simple(f.Y, y+k*ys, ys, 1, limit)
				}
			}
			continue
		}
		il, ht := int(fi.ilevel), int(fi.hev)
		if mx > 0 {
			loop(f.Y, y, 1, ys, 16, limit+4, il, ht, true)
			loop(f.U, u, 1, cs, 8, limit+4, il, ht, true)
			loop(f.V, u, 1, cs, 8, limit+4, il, ht, true)
		}
		if fi.inner {
			for k := 4; k < 16; k += 4 {
				loop(f.Y, y+k, 1, ys, 16, limit, il, ht, false)
			}
			loop(f.U, u+4, 1, cs, 8, limit, il, ht, false)
			loop(f.V, u+4, 1, cs, 8, limit, il, ht, false)
		}
		if my > 0 {
			loop(f.Y, y, ys, 1, 16, limit+4, il, ht, true)
			loop(f.U, u, cs, 1, 8, limit+4, il, ht, true)
			loop(f.V, u, cs, 1, 8, limit+4, il, ht, true)
		}
		if fi.inner {
			for k := 4; k < 16; k += 4 {
				loop(f.Y, y+k*ys, ys, 1, 16, limit, il, ht, false)
			}
			loop(f.U, u+4*cs, cs, 1, 8, limit, il, ht, false)
			loop(f.V, u+4*cs, cs, 1, 8, limit, il, ht, false)
		}
	}
}
