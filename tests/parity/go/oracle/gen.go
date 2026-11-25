package main

// The inputs: the fixtures and their mutations, junk files, and the images of webp-go's encoder
// tests (the generators of internal/webp/encode_test.go, alpha_test.go, libwebp_test.go, written
// with the same PCG seeds) under many options, plus random images under random options. Go
// supplies the expected bytes; the Rust side is only given the inputs.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"

	webp "github.com/dobbo-ca/webp-go"
)

func put(dir, name string, b []byte) { must(os.WriteFile(filepath.Join(dir, name), b, 0o644)) }

func gen(fix, out string, benchOnly bool) {
	must(os.MkdirAll(out, 0o755))
	if benchOnly {
		genBench(fix, out)
		return
	}
	genDecode(fix, out)
	genEncode(fix, out)
}

// xs is a xorshift64 generator seeded from a name, so the inputs do not depend on the Go version.
type xs struct{ s uint64 }

func newXS(name string) *xs {
	h := uint64(0xcbf29ce484222325)
	for _, c := range []byte(name) {
		h ^= uint64(c)
		h *= 0x100000001b3
	}
	return &xs{h | 1}
}

func (x *xs) next() uint64 {
	x.s ^= x.s << 13
	x.s ^= x.s >> 7
	x.s ^= x.s << 17
	return x.s
}

func (x *xs) n(k int) int { return int(x.next() % uint64(k)) }

type mut struct {
	tag string
	b   []byte
}

// mutate returns the mutations of a file: cuts (all of them for a file under 1200 bytes), cuts with
// the RIFF size fixed, bit flips, byte sets and, when heavy, RIFF and chunk size edits, a sweep of
// the header bytes and of the first 20 bytes of every chunk, every value of the first ALPH byte, and
// appended bytes.
func mutate(d []byte, name string, heavy bool) []mut {
	x := newXS(name)
	n := len(d)
	var ms []mut
	add := func(tag string, b []byte) { ms = append(ms, mut{tag, b}) }
	with := func(f func(b []byte)) []byte {
		b := append([]byte(nil), d...)
		f(b)
		return b
	}
	lim, steps, nf, ns := 64, 48, 64, 32
	if !heavy {
		lim, steps, nf, ns = 40, 16, 24, 12
	}
	cuts := map[int]bool{}
	if heavy && n <= 1200 {
		for i := 0; i < n; i++ {
			cuts[i] = true
		}
	} else {
		for i := 0; i < min(n, lim); i++ {
			cuts[i] = true
		}
		for i := 1; i <= steps; i++ {
			cuts[n*i/(steps+1)] = true
		}
		for _, k := range []int{1, 2, 3, 9} {
			if n-k >= 0 {
				cuts[n-k] = true
			}
		}
	}
	var cs []int
	for k := range cuts {
		cs = append(cs, k)
	}
	sort.Ints(cs)
	for _, k := range cs {
		add(fmt.Sprintf("t%d", k), d[:k])
	}
	for i := 1; i <= steps; i++ {
		if k := n * i / (steps + 1); k >= 8 {
			b := append([]byte(nil), d[:k]...)
			binary.LittleEndian.PutUint32(b[4:], uint32(k-8))
			add(fmt.Sprintf("u%d", k), b)
		}
	}
	for i := 0; i < nf && n > 0; i++ {
		bit := x.n(8 * n)
		add(fmt.Sprintf("f%d", bit), with(func(b []byte) { b[bit/8] ^= 1 << (bit % 8) }))
	}
	for i := 0; i < ns && n > 0; i++ {
		pos := x.n(n)
		v := []byte{0, 0xff, byte(x.next())}[i%3]
		add(fmt.Sprintf("s%d_%d", pos, v), with(func(b []byte) { b[pos] = v }))
	}
	if !heavy || n < 12 || string(d[:4]) != "RIFF" {
		return ms
	}
	for k, v := range []uint32{0, 4, 11, uint32(n - 9), uint32(n - 7), uint32(n), 0xffffffff, 0x7fffffff} {
		add(fmt.Sprintf("r%d", k), with(func(b []byte) { binary.LittleEndian.PutUint32(b[4:], v) }))
	}
	type chunk struct {
		tag        string
		hdr, data  int
		size, next int
	}
	var chunks []chunk
	for pos := 12; pos+8 <= n; {
		sz := int(binary.LittleEndian.Uint32(d[pos+4:]))
		c := chunk{string(d[pos : pos+4]), pos, pos + 8, sz, pos + 8 + sz + sz&1}
		chunks = append(chunks, c)
		if sz > n || c.next > n {
			break
		}
		pos = c.next
	}
	sweep := map[int]bool{}
	for i := 8; i < min(n, 48); i++ {
		sweep[i] = true
	}
	for _, c := range chunks {
		for i := c.data; i < min(n, c.data+20); i++ {
			sweep[i] = true
		}
	}
	var ps []int
	for p := range sweep {
		ps = append(ps, p)
	}
	sort.Ints(ps)
	for _, p := range ps {
		for k, v := range []byte{0, 0xff, d[p] ^ 8} {
			add(fmt.Sprintf("h%d_%d", p, k), with(func(b []byte) { b[p] = v }))
		}
	}
	for ci, c := range chunks {
		for k, v := range []uint32{0, uint32(c.size - 1), uint32(c.size + 1), uint32(c.size + 2), 0xffffffff} {
			add(fmt.Sprintf("z%d_%d", ci, k), with(func(b []byte) { binary.LittleEndian.PutUint32(b[c.hdr+4:], v) }))
		}
		if c.tag == "ALPH" && c.data < n {
			for v := 0; v < 256; v++ {
				add(fmt.Sprintf("p%d", v), with(func(b []byte) { b[c.data] = byte(v) }))
			}
		}
	}
	for _, k := range []int{1, 7, 100} {
		add(fmt.Sprintf("a%d", k), append(append([]byte(nil), d...), make([]byte, k)...))
		b := append(append([]byte(nil), d...), make([]byte, k+k&1)...)
		binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
		add(fmt.Sprintf("A%d", k), b)
	}
	return ms
}

func riffOne(tag string, body []byte) []byte {
	b := append([]byte("RIFF\x00\x00\x00\x00WEBP"), tag...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(body)))
	b = append(b, body...)
	if len(body)&1 == 1 {
		b = append(b, 0)
	}
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	return b
}

func genDecode(fix, out string) {
	for _, f := range sortedNames(fix, ".webp") {
		d, err := os.ReadFile(filepath.Join(fix, f))
		must(err)
		base := "fx-" + strings.TrimSuffix(f, ".webp")
		put(out, base+".webp", d)
		for _, m := range mutate(d, base, true) {
			put(out, base+"~"+m.tag+".webp", m.b)
		}
	}
	// Junk: random chunk soups, and VP8 and VP8L chunks with valid-looking headers and random
	// bodies, which the decoders parse deep into before they fail.
	r := newXS("junk")
	put(out, "junk-empty.webp", nil)
	put(out, "junk-riff.webp", []byte("RIFF"))
	tags := []string{"VP8 ", "VP8L", "VP8X", "ALPH", "EXIF", "XMP ", "ANIM", "ANMF", "JUNK"}
	for i := 0; i < 200; i++ {
		var body []byte
		for c := 1 + r.n(3); c > 0; c-- {
			n := r.n(40)
			sz := n
			if r.n(8) == 0 {
				sz = r.n(100)
			}
			body = append(body, tags[r.n(len(tags))]...)
			body = binary.LittleEndian.AppendUint32(body, uint32(sz))
			for j := 0; j < n; j++ {
				body = append(body, byte(r.next()))
			}
			if n&1 == 1 {
				body = append(body, 0)
			}
		}
		b := append([]byte("RIFF\x00\x00\x00\x00WEBP"), body...)
		binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
		put(out, fmt.Sprintf("junk-%03d.webp", i), b)
	}
	for i := 0; i < 150; i++ {
		w, h := 1+r.n(80), 1+r.n(80)
		n := 20 + r.n(400)
		tail := make([]byte, n)
		for j := range tail {
			tail[j] = byte(r.next())
		}
		ft := uint32(r.n(n))<<5 | 1<<4 | uint32(r.n(4))<<1
		body := []byte{byte(ft), byte(ft >> 8), byte(ft >> 16), 0x9d, 0x01, 0x2a}
		body = binary.LittleEndian.AppendUint16(body, uint16(w|r.n(4)<<14))
		body = binary.LittleEndian.AppendUint16(body, uint16(h|r.n(4)<<14))
		put(out, fmt.Sprintf("junk-vp8-%03d.webp", i), riffOne("VP8 ", append(body, tail...)))
		hdr := uint32(w-1) | uint32(h-1)<<14 | uint32(r.n(2))<<28 | uint32(r.n(8)/7)<<29
		body = binary.LittleEndian.AppendUint32([]byte{0x2f}, hdr)
		put(out, fmt.Sprintf("junk-vp8l-%03d.webp", i), riffOne("VP8L", append(body, tail...)))
	}
}

// Encode inputs.

const (
	fMatch  = 1
	fA16    = 2
	fLegacy = 4
	fNil    = 8
)

// opt is the options of one encode.
type opt struct {
	q     float32
	m     int
	flags byte
	seg   int
}

func (o opt) sig() string { return fmt.Sprintf("q%g_m%d_f%d_s%d", o.q, o.m, o.flags, o.seg) }

func plain(q float32, m int) opt           { return opt{q: q, m: m} }
func lw(q float32, m, seg int) opt         { return opt{q: q, m: m, flags: fMatch, seg: seg} }
func nilOpts() opt                         { return opt{flags: fNil} }
func lwFlags(q float32, m int, f byte) opt { return opt{q: q, m: m, flags: fMatch | f} }

type encGen struct{ out string }

func (g *encGen) add(name string, kind byte, w, h int, pix []byte, o opt) {
	c := encIn{kind: kind, method: o.m, flags: o.flags, seg: o.seg, quality: o.q, w: w, h: h, pix: pix}
	put(g.out, fmt.Sprintf("enc-%s-%s.enc", name, o.sig()), c.bytes())
}

// addEncoded writes the file Go encodes from the case as a decode input, with light mutations.
func (g *encGen) addEncoded(name string, kind byte, w, h int, pix []byte, o opt) {
	c := encIn{kind: kind, method: o.m, flags: o.flags, seg: o.seg, quality: o.q, w: w, h: h, pix: pix}
	m, op, err := parseEnc(c.bytes())
	must(err)
	var buf bytes.Buffer
	must(encode(m, op, nil, &buf))
	base := "ge-" + name + "-" + o.sig()
	put(g.out, base+".webp", buf.Bytes())
	for _, mu := range mutate(buf.Bytes(), base, false) {
		put(g.out, base+"~"+mu.tag+".webp", mu.b)
	}
}

// tightPix returns the image as tight NRGBA, or tight Gray pixels for a Gray image.
func tightPix(m image.Image) (kind byte, w, h int, pix []byte) {
	b := m.Bounds()
	w, h = b.Dx(), b.Dy()
	if g, ok := m.(*image.Gray); ok {
		for y := 0; y < h; y++ {
			pix = append(pix, g.Pix[y*g.Stride:y*g.Stride+w]...)
		}
		return 2, w, h, pix
	}
	o := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(o, o.Rect, m, b.Min, draw.Src)
	return 0, w, h, o.Pix
}

func (g *encGen) addImage(name string, m image.Image, opts []opt) {
	kind, w, h, pix := tightPix(m)
	for _, o := range opts {
		g.add(name, kind, w, h, pix, o)
	}
}

// halve is one 2x2 box downsample of tight NRGBA pixels, as a DZI pyramid level.
func halve(w, h int, p []byte) (int, int, []byte) {
	ow, oh := (w+1)/2, (h+1)/2
	o := make([]byte, 4*ow*oh)
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			var s [4]int
			n := 0
			for j := 2 * y; j < min(2*y+2, h); j++ {
				for i := 2 * x; i < min(2*x+2, w); i++ {
					for c := range s {
						s[c] += int(p[4*(j*w+i)+c])
					}
					n++
				}
			}
			for c := range s {
				o[4*(y*ow+x)+c] = uint8((s[c] + n/2) / n)
			}
		}
	}
	return ow, oh, o
}

// holes is an NRGBA tile with transparent holes and soft alpha edges (TestEncodeAlpha).
func holes(w, h int, seed uint64) []byte {
	r := rand.New(rand.NewPCG(seed, 1))
	p := make([]byte, 4*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8(255)
			switch (x/20 + y/13) % 5 {
			case 0:
				a = 0
			case 1:
				a = uint8(x * 7)
			}
			o := 4 * (y*w + x)
			p[o], p[o+1], p[o+2], p[o+3] = uint8(r.Uint32()), uint8(x), uint8(y), a
		}
	}
	return p
}

// premul is the premultiplied form of NRGBA pixels.
func premul(p []byte) []byte {
	o := append([]byte(nil), p...)
	for i := 0; i < len(o); i += 4 {
		for c := 0; c < 3; c++ {
			o[i+c] = byte((int(p[i+c])*int(p[i+3]) + 127) / 255)
		}
	}
	return o
}

func opaque(p []byte) []byte {
	o := append([]byte(nil), p...)
	for i := 3; i < len(o); i += 4 {
		o[i] = 255
	}
	return o
}

func solid(w, h int, a byte) []byte {
	p := make([]byte, 4*w*h)
	for i := 0; i < len(p); i += 4 {
		p[i], p[i+1], p[i+2], p[i+3] = 10, 200, 90, a
	}
	return p
}

func loadImage(fix, name string) image.Image {
	d, err := os.ReadFile(filepath.Join(fix, name))
	must(err)
	if strings.HasSuffix(name, ".webp") {
		m, err := webp.Decode(d, nil)
		must(err)
		return m
	}
	m, err := png.Decode(bytes.NewReader(d))
	must(err)
	return m
}

func genEncode(fix, out string) {
	g := &encGen{out: out}
	lwGolden := []opt{lw(90, 4, 0), lw(50, 6, 0), lw(90, 4, 1)}
	big := append([]opt{plain(75, 4), plain(90, 4), plain(60, 0), plain(60, 2)}, lwGolden...)
	mid := append(append([]opt{}, big...), plain(60, 1), plain(60, 3), plain(60, 5), lw(40, 3, 0), lw(95, 5, 0))
	var grid []opt
	for m := 0; m <= 6; m++ {
		for _, f := range []byte{0, fMatch} {
			for _, q := range []float32{0, 30, 75, 100} {
				for _, seg := range []int{0, 1} {
					grid = append(grid, opt{q: q, m: m, flags: f, seg: seg})
				}
			}
		}
	}
	grid = append(grid, lwGolden...)

	// The fixtures of libwebp_test.go's lwCases and TestGrayMatchesRGBA.
	fl := loadImage(fix, "fl_768.png")
	g.addImage("fl768", fl, big)
	if gr, ok := fl.(*image.Gray); ok {
		// The same picture as NRGBA, to see that Gray equals RGBA.
		_, w, h, _ := tightPix(fl)
		n := make([]byte, 4*w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := gr.Pix[y*gr.Stride+x]
				copy(n[4*(y*w+x):], []byte{v, v, v, 255})
			}
		}
		g.add("fl768n", 0, w, h, n, plain(75, 4))
	}
	g.addImage("he768", loadImage(fix, "he_768.webp"), big)
	k05 := loadImage(fix, "kodim05_192.png")
	g.addImage("k05", k05, grid)
	g.addImage("k05", k05, []opt{nilOpts()})
	g.addImage("k23", loadImage(fix, "kodim23_192.png"), grid)

	// The noise pyramid of lwCases (PCG 21, 22), as NRGBA with alpha 255.
	r := rand.New(rand.NewPCG(21, 22))
	nw, nh := 1600, 1200
	np := make([]byte, 4*nw*nh)
	for i := range np {
		np[i] = uint8(r.Uint32())
		if i%4 == 3 {
			np[i] = 255
		}
	}
	for l := 1; nw > 10; l++ {
		nw, nh, np = halve(nw, nh, np)
		if nw <= 200 {
			opts := mid
			if l == 3 {
				opts = grid
			}
			for _, o := range opts {
				g.add(fmt.Sprintf("noise%dx%d", nw, nh), 0, nw, nh, np, o)
			}
		}
	}

	// Alpha (alpha_test.go): TestEncodeAlpha sizes with seed = width, TestMatchLibwebpAlphaALPH
	// (every method with every ALPH writer), TestSegmentsAlphaAndErrors, TestEncodeSolidAlpha.
	for _, sz := range [][2]int{{1, 1}, {7, 5}, {33, 17}, {101, 250}, {770, 770}} {
		g.add(fmt.Sprintf("holes%dx%d", sz[0], sz[1]), 0, sz[0], sz[1], holes(sz[0], sz[1], uint64(sz[0])), plain(90, 4))
	}
	h101 := holes(101, 250, 5)
	for m := 0; m <= 6; m++ {
		for _, match := range []bool{false, true} {
			for _, a16 := range []bool{false, true} {
				for _, legacy := range []bool{false, true} {
					o := plain(90, m)
					if match {
						o.flags |= fMatch
					}
					if a16 {
						o.flags |= fA16
					}
					if legacy {
						o.flags |= fLegacy
					}
					g.add("holesA101x250", 0, 101, 250, h101, o)
				}
			}
		}
	}
	for _, sz := range [][2]int{{7, 5}, {33, 17}} {
		p := holes(sz[0], sz[1], 5)
		for m := 3; m <= 5; m++ {
			for f := byte(0); f < 8; f++ {
				g.add(fmt.Sprintf("holesA%dx%d", sz[0], sz[1]), 0, sz[0], sz[1], p, opt{q: 90, m: m, flags: f})
			}
		}
	}
	for _, f := range []byte{0, fMatch} {
		for _, seg := range []int{0, 1, 4, -1, 2, 3, 5} {
			g.add("holesA101x250", 0, 101, 250, h101, opt{q: 90, m: 4, flags: f, seg: seg})
		}
	}
	for _, a := range []byte{0, 1, 128, 254} {
		s := 96
		if a == 0 || a == 128 {
			s = 770
		}
		for _, o := range []opt{plain(90, 4), lw(90, 4, 0)} {
			g.add(fmt.Sprintf("solid%d_%d", a, s), 0, s, s, solid(s, s, a), o)
		}
	}
	// TestEncodeAlphaKinds: premultiplied RGBA with nil Options, and opaque NRGBA.
	h64 := holes(64, 48, 3)
	g.add("premul64x48", 1, 64, 48, premul(h64), nilOpts())
	g.add("premul64x48", 1, 64, 48, premul(h64), lw(90, 4, 0))
	g.add("opaque64x48", 0, 64, 48, opaque(h64), nilOpts())

	// A NaN quality: Go clamps it to NaN, so a solid image (which never reads the quality) encodes
	// as ever and any other takes quality index 0 with no error diffusion, on every architecture.
	nan := float32(math.NaN())
	for _, o := range []opt{plain(nan, 4), lw(nan, 4, 0), lw(nan, 4, 1)} {
		g.add("nanq-solid", 0, 32, 32, solid(32, 32, 255), o)
		g.add("nanq-textured", 0, 64, 48, opaque(h64), o)
		g.add("nanq-alpha", 0, 64, 48, h64, o)
	}

	// Flat grey (flatSizes), and the sizes TestEncodeBadSize refuses.
	for _, s := range [][2]int{{1, 1}, {16, 16}, {17, 33}, {768, 768}, {1000, 1}} {
		flat := bytes.Repeat([]byte{130}, s[0]*s[1])
		g.add(fmt.Sprintf("flat%dx%d", s[0], s[1]), 2, s[0], s[1], flat, nilOpts())
		g.add(fmt.Sprintf("flat%dx%d", s[0], s[1]), 2, s[0], s[1], flat, lw(75, 4, 0))
	}
	g.add("bad0x0", 2, 0, 0, nil, nilOpts())
	g.add("bad0x5", 0, 0, 5, nil, nilOpts())
	g.add("bad16384x1", 2, 16384, 1, make([]byte, 16384), nilOpts())
	g.add("bad1x16384", 2, 1, 16384, make([]byte, 16384), nilOpts())

	// TestEncodeYCbCr (37x21, Y = i*7) and the in-gamut planes of TestEncodeYCbCrOddOrigin.
	y := make([]byte, 37*21)
	for i := range y {
		y[i] = uint8(i * 7)
	}
	g.add("ycc37x21", 3, 37, 21, append(append(y, make([]byte, 19*11)...), make([]byte, 19*11)...), nilOpts())
	for _, s := range [][2]int{{1, 1}, {2, 2}, {16, 16}, {17, 17}, {31, 19}, {40, 30}} {
		w, h := s[0], s[1]
		cw, ch := (w+1)/2, (h+1)/2
		p := make([]byte, w*h+2*cw*ch)
		for i := 0; i < w*h; i++ {
			p[i] = uint8(60 + i*7%140)
		}
		for i := 0; i < cw*ch; i++ {
			p[w*h+i] = uint8(118 + i*13%20)
			p[w*h+cw*ch+i] = uint8(118 + i*29%20)
		}
		for _, o := range []opt{nilOpts(), plain(90, 4), lw(75, 4, 0), plain(30, 0)} {
			g.add(fmt.Sprintf("ycc%dx%d", w, h), 3, w, h, p, o)
		}
	}

	// Random images under random options.
	rr := rand.New(rand.NewPCG(1234, 5678))
	dims := []int{1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49}
	for i := 0; i < 300; i++ {
		w, h := 1+rr.IntN(96), 1+rr.IntN(96)
		if rr.IntN(3) == 0 {
			w, h = dims[rr.IntN(len(dims))], dims[rr.IntN(len(dims))]
		}
		kind := []byte{0, 0, 0, 1, 2, 3}[rr.IntN(6)]
		g.add(fmt.Sprintf("r%03d_%dx%d_k%d", i, w, h, kind), kind, w, h, randPix(rr, kind, w, h), randOpt(rr))
	}

	// Files the encoder writes, as decode inputs: VP8X with every ALPH writer, plain VP8, Gray.
	g.addEncoded("holes64x48", 0, 64, 48, h64, plain(90, 4))
	g.addEncoded("premul64x48", 1, 64, 48, premul(h64), nilOpts())
	for m := 3; m <= 5; m++ {
		for _, f := range []byte{0, fA16, fLegacy} {
			g.addEncoded("holesA101x250", 0, 101, 250, h101, lwFlags(90, m, f))
		}
	}
	g.addEncoded("holesA101x250", 0, 101, 250, h101, plain(90, 4))
	g.addEncoded("holesA101x250", 0, 101, 250, h101, plain(90, 0))
	kind, w, h, p := tightPix(k05)
	g.addEncoded("k05", kind, w, h, p, lw(90, 4, 0))
	g.addEncoded("k05", kind, w, h, p, plain(75, 4))
	g.addEncoded("flat17x33", 2, 17, 33, bytes.Repeat([]byte{130}, 17*33), nilOpts())
}

// randPix is a random w by h image of the kind: noise under a mask, a gradient, a checkerboard, a
// bump or a flat colour with a few noisy pixels, with alpha holes for a third of them.
func randPix(r *rand.Rand, kind byte, w, h int) []byte {
	p := make([]byte, 4*w*h)
	style, period := r.IntN(5), 1+r.IntN(20)
	mask := []byte{0xff, 0xf0, 0xc0, 0x80, 0}[r.IntN(5)]
	withHoles := r.IntN(3) == 0
	flat := [3]byte{byte(r.Uint32()), byte(r.Uint32()), byte(r.Uint32())}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := 4 * (y*w + x)
			var v [3]byte
			switch style {
			case 0:
				v = [3]byte{byte(r.Uint32()) & mask, byte(r.Uint32()) & mask, byte(r.Uint32()) & mask}
			case 1:
				v = [3]byte{byte(x * 255 / max(w-1, 1)), byte(y * 255 / max(h-1, 1)), byte((x + y) * 255 / max(w+h-2, 1))}
				v[0] ^= byte(r.Uint32()) & 3
			case 2:
				if (x/period+y/period)%2 == 0 {
					v = [3]byte{230, 40, 40}
				} else {
					v = [3]byte{20, 20, 200}
				}
			case 3:
				b := byte((x*x + y*y) >> 3)
				v = [3]byte{b, b ^ 0x55, byte(x + y)}
			default:
				v = flat
				if r.IntN(16) == 0 {
					v[r.IntN(3)] = byte(r.Uint32())
				}
			}
			a := byte(255)
			if withHoles {
				switch (x/period + y/period) % 4 {
				case 0:
					a = 0
				case 1:
					a = byte(r.Uint32())
				}
			}
			p[o], p[o+1], p[o+2], p[o+3] = v[0], v[1], v[2], a
		}
	}
	switch kind {
	case 0:
		return p
	case 1:
		return premul(p)
	}
	cw, ch := (w+1)/2, (h+1)/2
	pl := make([]byte, w*h, w*h+2*cw*ch)
	for i := range pl {
		pl[i] = byte((int(p[4*i]) + 2*int(p[4*i+1]) + int(p[4*i+2])) / 4)
	}
	if kind == 2 {
		return pl
	}
	for c := 0; c < 2; c++ {
		for i := 0; i < cw*ch; i++ {
			pl = append(pl, byte(128+(i*(13+16*c))%60-30)^byte(r.Uint32())&3)
		}
	}
	return pl
}

func randOpt(r *rand.Rand) opt {
	o := opt{q: float32(r.IntN(101)), m: r.IntN(7)}
	if r.IntN(4) == 0 {
		o.q += 0.5
	}
	switch r.IntN(24) {
	case 0:
		o.m = -1
	case 1:
		o.m = 9
	case 2:
		o.q = -10
	case 3:
		o.q = 150
	}
	if r.IntN(2) == 0 {
		o.flags |= fMatch
	}
	if r.IntN(3) == 0 {
		o.flags |= fA16
	}
	if r.IntN(4) == 0 {
		o.flags |= fLegacy
	}
	o.seg = []int{0, 0, 1, 4}[r.IntN(4)]
	if r.IntN(40) == 0 {
		o.seg = []int{-1, 2, 3, 5}[r.IntN(4)]
	}
	if r.IntN(15) == 0 {
		o.flags = fNil
	}
	return o
}

// genBench writes the benchmark inputs: every fixture as a decode input named by its kind, and
// four images (and a 512 by 512 alpha tile) under the options whose speed matters.
func genBench(fix, out string) {
	for _, f := range sortedNames(fix, ".webp") {
		d, err := os.ReadFile(filepath.Join(fix, f))
		must(err)
		c, err := webp.DecodeConfig(d)
		if err != nil {
			continue
		}
		cl := "lossy"
		switch {
		case strings.HasPrefix(f, "anim_"):
			cl = "anim"
		case c.Lossless:
			cl = "ll"
		case c.HasAlpha:
			cl = "lossya"
		}
		put(out, fmt.Sprintf("dec-%s-%s", cl, f), d)
	}
	g := &encGen{out: out}
	for _, o := range []opt{plain(75, 0), plain(75, 4), plain(75, 6), lw(75, 4, 0), lw(75, 6, 0)} {
		for _, f := range []string{"kodim05_192.png", "kodim23_192.png", "he_768.webp", "fl_768.png"} {
			g.addImage(strings.TrimSuffix(strings.TrimSuffix(f, ".png"), ".webp"), loadImage(fix, f), []opt{o})
		}
	}
	a := holes(512, 512, 7)
	for _, o := range []opt{plain(75, 4), lw(75, 4, 0), lwFlags(75, 4, fA16)} {
		g.add("alpha512", 0, 512, 512, a, o)
	}
}
