package vp8ldec

// inverse undoes transform t on a, whose rows are t.xs pixels wide until
// a color-indexing transform widens them to w.
func (d *decoder) inverse(a []byte, t *transform, h, w int) error {
	switch t.kind {
	case tPredictor:
		for y := 0; y < h; y++ {
			if err := d.row(); err != nil {
				return err
			}
			var prev []byte
			if y > 0 {
				prev = a[4*(y-1)*t.xs:]
			}
			predictorRow(a[4*y*t.xs:4*(y+1)*t.xs], prev, y, t)
		}
	case tCross:
		for y := 0; y < h; y++ {
			if err := d.row(); err != nil {
				return err
			}
			crossRow(a[4*y*t.xs:], y, t)
		}
	case tSubGreen:
		subGreen(a[:4*t.xs*h])
	case tIndex:
		pw := ceil(w, t.bits)
		for y := h - 1; y >= 0; y-- {
			if err := d.row(); err != nil {
				return err
			}
			indexRow(a[4*y*w:], a[4*y*pw:], w, t)
		}
	}
	return nil
}

// inverseRow undoes the transforms ts, last first, on row y, which is in
// cur and w pixels wide when done. prev holds the row the predictor made
// last, and cur needs room for w pixels.
func (d *decoder) inverseRow(cur, prev []byte, ts []transform, y, w int) {
	for i := len(ts) - 1; i >= 0; i-- {
		t := &ts[i]
		switch t.kind {
		case tPredictor:
			var p []byte
			if y > 0 {
				p = prev
			}
			predictorRow(cur[:4*t.xs], p, y, t)
			copy(prev, cur[:4*t.xs])
		case tCross:
			crossRow(cur, y, t)
		case tSubGreen:
			subGreen(cur[:4*t.xs])
		case tIndex:
			indexRow(cur, cur, w, t)
		}
	}
}

func crossRow(row []byte, y int, t *transform) {
	tw := ceil(t.xs, t.bits)
	for x := 0; x < t.xs; x++ {
		m := get(t.data, (y>>t.bits)*tw+x>>t.bits)
		gr, gb, rb := int8(m), int8(m>>8), int8(m>>16)
		px := get(row, x)
		g := int8(px >> 8)
		r := uint8(px>>16) + uint8(int32(gr)*int32(g)>>5)
		b := uint8(px) + uint8(int32(gb)*int32(g)>>5) + uint8(int32(rb)*int32(int8(r))>>5)
		put(row, x, px&0xff00ff00|uint32(r)<<16|uint32(b))
	}
}

func subGreen(a []byte) {
	for i := 0; i < len(a)/4; i++ {
		px := get(a, i)
		g := px >> 8 & 0xff
		put(a, i, px&0xff00ff00|(px&0xff0000+g<<16)&0xff0000|(px&0xff+g)&0xff)
	}
}

// indexRow writes the w pixels of a row from the packed indexes in src,
// last first, so src may be dst.
func indexRow(dst, src []byte, w int, t *transform) {
	per := uint(1) << t.bits
	bpp := 8 / per
	for x := w - 1; x >= 0; x-- {
		g := get(src, x>>t.bits) >> 8
		i := g >> (bpp * (uint(x) & (per - 1))) & (1<<bpp - 1)
		put(dst, x, get(t.data, int(i)))
	}
}

// predictorRow undoes the predictor transform on row y of t.xs pixels. prev
// is the row above, nil for the first row. The rightmost pixel's top right
// is the first pixel of its own row.
func predictorRow(cur, prev []byte, y int, t *transform) {
	xs := t.xs
	if prev == nil {
		put(cur, 0, add(get(cur, 0), 0xff000000))
		for x := 1; x < xs; x++ {
			put(cur, x, add(get(cur, x), get(cur, x-1)))
		}
		return
	}
	tw := ceil(xs, t.bits)
	put(cur, 0, add(get(cur, 0), get(prev, 0)))
	mode := 0
	for x := 1; x < xs; x++ {
		if x&(1<<t.bits-1) == 0 || x == 1 {
			mode = int(get(t.data, (y>>t.bits)*tw+x>>t.bits) >> 8 & 15)
		}
		l, tp, tl := get(cur, x-1), get(prev, x), get(prev, x-1)
		tr := get(cur, 0)
		if x+1 < xs {
			tr = get(prev, x+1)
		}
		var p uint32
		switch mode {
		case 0, 14, 15:
			p = 0xff000000
		case 1:
			p = l
		case 2:
			p = tp
		case 3:
			p = tr
		case 4:
			p = tl
		case 5:
			p = avg2(avg2(l, tr), tp)
		case 6:
			p = avg2(l, tl)
		case 7:
			p = avg2(l, tp)
		case 8:
			p = avg2(tl, tp)
		case 9:
			p = avg2(tp, tr)
		case 10:
			p = avg2(avg2(l, tl), avg2(tp, tr))
		case 11:
			p = selectPx(l, tp, tl)
		case 12:
			p = clampAddSub(l, tp, tl)
		case 13:
			p = clampAddSubHalf(avg2(l, tp), tl)
		}
		put(cur, x, add(get(cur, x), p))
	}
}

// add adds per channel, modulo 256.
func add(a, b uint32) uint32 {
	return (a&0xff00ff00+b&0xff00ff00)&0xff00ff00 | (a&0x00ff00ff+b&0x00ff00ff)&0x00ff00ff
}

func avg2(a, b uint32) uint32 { return (a^b)&0xfefefefe>>1 + a&b }

func ch(v uint32, s uint) int { return int(v >> s & 0xff) }

// selectPx is the spec's Select(L, T, TL).
func selectPx(l, t, tl uint32) uint32 {
	var pl, pt int
	for s := uint(0); s < 32; s += 8 {
		pl += abs(ch(t, s) - ch(tl, s))
		pt += abs(ch(l, s) - ch(tl, s))
	}
	if pl < pt {
		return l
	}
	return t
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func clamp(v int) uint32 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint32(v)
}

func clampAddSub(a, b, c uint32) uint32 {
	var r uint32
	for s := uint(0); s < 32; s += 8 {
		r |= clamp(ch(a, s)+ch(b, s)-ch(c, s)) << s
	}
	return r
}

func clampAddSubHalf(a, b uint32) uint32 {
	var r uint32
	for s := uint(0); s < 32; s += 8 {
		r |= clamp(ch(a, s)+(ch(a, s)-ch(b, s))/2) << s
	}
	return r
}
