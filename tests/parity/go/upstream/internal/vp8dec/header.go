// Package vp8dec decodes VP8 key frames (RFC 6386) the way libwebp does, so
// the pixels are bit exact with dwebp. It follows libwebp's src/dec
// (vp8_dec.c, tree_dec.c, quant_dec.c, frame_dec.c, io_dec.c) and src/dsp
// (dec.c, upsampling.c, yuv.h), BSD-3-Clause; see NOTICE.
package vp8dec

import (
	"context"
	"encoding/binary"
	"errors"

	"github.com/dobbo-ca/webp-go/internal/vp8enc"
)

var (
	errHeader    = errors.New("webp: invalid VP8 frame header")
	errTruncated = errors.New("webp: truncated VP8 data")
)

// Frame is a decoded key frame: 4:2:0 planes of the macroblock-padded size.
type Frame struct {
	W, H           int
	Y, U, V        []byte
	YStride, CStrd int
}

// Header checks the key frame tag and returns the frame size.
func Header(data []byte) (w, h int, err error) {
	if len(data) < 10 || data[0]&1 != 0 || data[0]>>1&7 > 3 || data[0]>>4&1 == 0 ||
		data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a {
		return 0, 0, errHeader
	}
	if int(binary.LittleEndian.Uint32(data[:4])&0xffffff>>5) > len(data)-10 {
		return 0, 0, errHeader
	}
	w = int(binary.LittleEndian.Uint16(data[6:]) & 0x3fff)
	h = int(binary.LittleEndian.Uint16(data[8:]) & 0x3fff)
	if w == 0 || h == 0 {
		return 0, 0, errHeader
	}
	return w, h, nil
}

// PlaneBytes is the size of the padded planes of a w x h frame.
func PlaneBytes(w, h int) int64 {
	return int64((w+15)&^15) * int64((h+15)&^15) * 3 / 2
}

// ScratchBytes bounds what Decode allocates besides the planes, for a w x h frame.
func ScratchBytes(w int) int64 { return int64((w+15)>>4)*160 + 4096 }

type segHeader struct {
	use, updateMap, absDelta bool
	quant, strength          [4]int
	probs                    [3]uint8
}

type filterHeader struct {
	simple, useDelta bool
	level, sharpness int
	ref, mode        [4]int
}

type quant struct {
	y1, y2, uv [2]int32 // dc, ac
}

type finfo struct {
	limit, ilevel, hev uint8
	inner              bool
}

type mb struct {
	segment uint8
	skip    bool
	i4      bool
	uvmode  uint8
	modes   [16]uint8
}

type dec struct {
	ctx      context.Context
	br       boolDec
	parts    [8]boolDec
	nparts   int
	mbw, mbh int
	seg      segHeader
	flt      filterHeader
	fType    int // 0 none, 1 simple, 2 complex
	q        [4]quant
	probs    [4][8][3][11]uint8
	useSkip  bool
	skipP    uint8
	fs       [4][2]finfo

	intraT           []uint8
	intraL           [4]uint8
	nzT              [][9]uint8
	nzL              [9]uint8
	topY, topU, topV []uint8
	mbs              []mb
	finfos           []finfo

	f       *Frame
	yuv     [bps*17 + bps*9]uint8
	coeffs  [384]int16
	nonzero [24]bool // per 4x4 block: needs the inverse transform
}

// Decode decodes the key frame in data into planes y, u and v, which hold
// PlaneBytes(w, h) in all (y first, then u and v each a quarter of the
// padded area), with w and h from Header. grow is asked for the scratch
// before it is allocated. ctx is checked once a macroblock row.
func Decode(ctx context.Context, data []byte, y, u, v []byte, grow func(int64) error) (*Frame, error) {
	w, h, err := Header(data)
	if err != nil {
		return nil, err
	}
	if err := grow(ScratchBytes(w)); err != nil {
		return nil, err
	}
	d := &dec{ctx: ctx}
	d.mbw, d.mbh = (w+15)>>4, (h+15)>>4
	d.f = &Frame{W: w, H: h, Y: y, U: u, V: v, YStride: d.mbw * 16, CStrd: d.mbw * 8}
	partLen := int(binary.LittleEndian.Uint32(data[:4]) & 0xffffff >> 5)
	d.br.init(data[10 : 10+partLen])
	if err := d.parseHeaders(data[10+partLen:]); err != nil {
		return nil, err
	}
	d.intraT = make([]uint8, 4*d.mbw)
	d.nzT = make([][9]uint8, d.mbw)
	d.topY, d.topU, d.topV = make([]uint8, 16*d.mbw), make([]uint8, 8*d.mbw), make([]uint8, 8*d.mbw)
	d.mbs = make([]mb, d.mbw)
	d.finfos = make([]finfo, d.mbw)
	for my := 0; my < d.mbh; my++ {
		if err := d.row(my); err != nil {
			return nil, err
		}
	}
	return d.f, nil
}

func (d *dec) parseHeaders(rest []byte) error {
	br := &d.br
	br.value(2) // color space and clamping type
	s := &d.seg
	s.probs = [3]uint8{255, 255, 255}
	s.absDelta = true
	if s.use = br.flag(); s.use {
		s.updateMap = br.flag()
		if br.flag() { // update data
			s.absDelta = br.flag()
			for i := range s.quant {
				s.quant[i] = 0
				if br.flag() {
					s.quant[i] = br.signed(7)
				}
			}
			for i := range s.strength {
				s.strength[i] = 0
				if br.flag() {
					s.strength[i] = br.signed(6)
				}
			}
		}
		if s.updateMap {
			for i := range s.probs {
				if br.flag() {
					s.probs[i] = uint8(br.value(8))
				}
			}
		}
	}
	if br.eof {
		return errTruncated
	}
	f := &d.flt
	f.simple = br.flag()
	f.level = br.value(6)
	f.sharpness = br.value(3)
	if f.useDelta = br.flag(); f.useDelta && br.flag() {
		for i := range f.ref {
			if br.flag() {
				f.ref[i] = br.signed(6)
			}
		}
		for i := range f.mode {
			if br.flag() {
				f.mode[i] = br.signed(6)
			}
		}
	}
	switch {
	case f.level == 0:
		d.fType = 0
	case f.simple:
		d.fType = 1
	default:
		d.fType = 2
	}
	if br.eof {
		return errTruncated
	}
	if err := d.parsePartitions(rest); err != nil {
		return err
	}
	d.parseQuant()
	br.flag() // update_proba: ignored on key frames
	for t := range d.probs {
		for b := range d.probs[t] {
			for c := range d.probs[t][b] {
				for p := range d.probs[t][b][c] {
					if br.bit(vp8enc.CoeffUpdateProbs[t][b][c][p]) != 0 {
						d.probs[t][b][c][p] = uint8(br.value(8))
					} else {
						d.probs[t][b][c][p] = vp8enc.CoeffProbs[t][b][c][p]
					}
				}
			}
		}
	}
	if d.useSkip = br.flag(); d.useSkip {
		d.skipP = uint8(br.value(8))
	}
	if br.eof {
		return errTruncated
	}
	d.filterStrengths()
	return nil
}

func (d *dec) parsePartitions(buf []byte) error {
	d.nparts = 1 << d.br.value(2)
	last := d.nparts - 1
	if len(buf) < 3*last {
		return errTruncated
	}
	sz, start := buf, buf[3*last:]
	for p := 0; p < last; p++ {
		n := min(int(sz[0])|int(sz[1])<<8|int(sz[2])<<16, len(start))
		d.parts[p].init(start[:n])
		start = start[n:]
		sz = sz[3:]
	}
	d.parts[last].init(start)
	if len(start) == 0 {
		return errTruncated // libwebp: the last partition must not be empty
	}
	return nil
}

func clip(v, m int) int { return max(0, min(v, m)) }

func (d *dec) parseQuant() {
	br := &d.br
	base := br.value(7)
	var dq [5]int
	for i := range dq {
		if br.flag() {
			dq[i] = br.signed(4)
		}
	}
	dy1dc, dy2dc, dy2ac, duvdc, duvac := dq[0], dq[1], dq[2], dq[3], dq[4]
	for i := 0; i < 4; i++ {
		var q int
		if d.seg.use {
			q = d.seg.quant[i]
			if !d.seg.absDelta {
				q += base
			}
		} else if i > 0 {
			d.q[i] = d.q[0]
			continue
		} else {
			q = base
		}
		m := &d.q[i]
		m.y1[0] = int32(vp8enc.DCQuant[clip(q+dy1dc, 127)])
		m.y1[1] = int32(vp8enc.ACQuant[clip(q, 127)])
		m.y2[0] = int32(vp8enc.DCQuant[clip(q+dy2dc, 127)]) * 2
		m.y2[1] = max(int32(vp8enc.ACQuant[clip(q+dy2ac, 127)])*101581>>16, 8)
		m.uv[0] = int32(vp8enc.DCQuant[clip(q+duvdc, 117)])
		m.uv[1] = int32(vp8enc.ACQuant[clip(q+duvac, 127)])
	}
}

func (d *dec) filterStrengths() {
	if d.fType == 0 {
		return
	}
	h := &d.flt
	for s := 0; s < 4; s++ {
		base := h.level
		if d.seg.use {
			base = d.seg.strength[s]
			if !d.seg.absDelta {
				base += h.level
			}
		}
		for i4 := 0; i4 < 2; i4++ {
			level := base
			if h.useDelta {
				level += h.ref[0]
				if i4 == 1 {
					level += h.mode[0]
				}
			}
			level = clip(level, 63)
			fi := &d.fs[s][i4]
			*fi = finfo{inner: i4 == 1}
			if level == 0 {
				continue
			}
			il := level
			if h.sharpness > 0 {
				if h.sharpness > 4 {
					il >>= 2
				} else {
					il >>= 1
				}
				il = min(il, 9-h.sharpness)
			}
			il = max(il, 1)
			fi.ilevel = uint8(il)
			fi.limit = uint8(2*level + il)
			switch {
			case level >= 40:
				fi.hev = 2
			case level >= 15:
				fi.hev = 1
			}
		}
	}
}
