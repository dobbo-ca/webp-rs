// Command oracle is the Go side of the parity check with webp-rs (see docs/parity.md).
//
//	oracle <inputs> <outputs>        run every input and write <name>.<op>.out or <name>.<op>.err
//	oracle -gen <testdata> <inputs>  write the inputs: fixtures, mutations, generated images
//	oracle -genbench <testdata> <d>  write the benchmark inputs
//	oracle -qenc <images> <webp> <dec> encode the PPM/PGM images under the bench settings, then decode them
//	oracle -bench N <dir>            print name, pixels and ns per operation for each input
//
// An input is a .webp file (decoded: ops cfg, dec, planar, meta, ...) or a .enc file (encoded: op
// enc; the format is in encFormat below).
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"time"

	webp "github.com/dobbo-ca/webp-go"
)

// maxPixels caps every decode and every encode of this program, so a hostile header cannot make
// the oracle allocate more than a few tens of MB (it runs in a memory-limited CI container).
const maxPixels = 1 << 22

// maxFile is the largest input read.
const maxFile = 16 << 20

func main() {
	debug.SetMemoryLimit(1 << 30)
	webp.SetMemoryLimit(0)
	a := os.Args[1:]
	switch {
	case len(a) == 3 && a[0] == "-gen":
		gen(a[1], a[2], false)
	case len(a) == 3 && a[0] == "-genbench":
		gen(a[1], a[2], true)
	case len(a) == 3 && a[0] == "-bench":
		var n int
		if _, err := fmt.Sscan(a[1], &n); err != nil || n < 1 {
			fatal(fmt.Errorf("bad N %q", a[1]))
		}
		bench(n, a[2])
	case len(a) == 4 && a[0] == "-qenc":
		qenc(a[1], a[2], a[3])
	case len(a) == 2 && !strings.HasPrefix(a[0], "-"):
		run(a[0], a[1])
	default:
		fmt.Fprintln(os.Stderr, "usage: oracle <inputs> <outputs> | -gen <testdata> <inputs> | -genbench <testdata> <dir> | -bench N <dir> | -qenc <images> <webp-out> <dec-out>")
		os.Exit(2)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "oracle:", err)
	os.Exit(1)
}

func must(err error) {
	if err != nil {
		fatal(err)
	}
}

var errBadInput = errors.New("bad-input")

// class is the short stable token of an error; the table is in docs/parity.md.
func class(err error) string {
	switch {
	case errors.Is(err, errBadInput):
		return "bad-input"
	case errors.Is(err, webp.ErrLimit), errors.Is(err, webp.ErrMemoryLimit):
		return "limit"
	case errors.Is(err, webp.ErrFormat):
		return "format"
	case errors.Is(err, webp.ErrTruncated):
		return "truncated"
	case errors.Is(err, webp.ErrAnimated):
		return "animated"
	case errors.Is(err, webp.ErrNotLossy):
		return "not-lossy"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	}
	msg := err.Error()
	switch msg {
	case "webp: invalid VP8 frame header":
		return "vp8-header"
	case "webp: truncated VP8 data":
		return "vp8-truncated"
	case "webp: invalid VP8L header":
		return "vp8l-header"
	case "webp: truncated VP8L stream":
		return "vp8l-truncated"
	case "webp: invalid VP8L bitstream":
		return "vp8l-bitstream"
	case "webp: invalid prefix code":
		return "prefix-code"
	}
	for _, s := range []string{"dimensions out of range", "Segments must be", "YCbCr planes shorter", "only 4:2:0"} {
		if strings.Contains(msg, s) {
			return "argument"
		}
	}
	if strings.Contains(msg, "partition too large") {
		return "unsupported"
	}
	return "other:" + msg
}

// digest is the FNV-1a 64 and the length of b, as a line.
func digest(b []byte) []byte {
	h := uint64(0xcbf29ce484222325)
	for _, c := range b {
		h ^= uint64(c)
		h *= 0x100000001b3
	}
	return []byte(fmt.Sprintf("%016x %d\n", h, len(b)))
}

// emitter writes the results of one input. A mutation (`~` in the name) writes the digest of each
// output, not the bytes, to keep the trees small.
type emitter struct {
	dir, name string
	mut       bool
}

func (e emitter) put(op string, b []byte, err error) {
	if e.mut && err == nil {
		b = digest(b)
	}
	e.write(op, b, err)
}

// putd is put with the digest always.
func (e emitter) putd(op string, b []byte, err error) {
	if err == nil {
		b = digest(b)
	}
	e.write(op, b, err)
}

func (e emitter) write(op string, b []byte, err error) {
	base := filepath.Join(e.dir, e.name+"."+op)
	if err != nil {
		must(os.WriteFile(base+".err", []byte(class(err)+"\n"), 0o644))
		return
	}
	must(os.WriteFile(base+".out", b, 0o644))
}

func run(in, out string) {
	must(os.MkdirAll(out, 0o755))
	ents, err := os.ReadDir(in)
	must(err)
	for _, ent := range ents {
		name := ent.Name()
		ext := filepath.Ext(name)
		if ext != ".webp" && ext != ".enc" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(in, name))
		must(err)
		if len(data) > maxFile {
			continue
		}
		e := emitter{dir: out, name: name, mut: strings.Contains(name, "~")}
		if ext == ".webp" {
			webpOps(e, data)
		} else {
			encOps(e, data)
		}
	}
}

func limits() *webp.Limits { return &webp.Limits{MaxPixels: maxPixels} }

// webpOps decodes one file every way the API has: cfg (DecodeConfig), dec (Decode: width and height
// as u32 LE, then the NRGBA pixels), planar (DecodePlanar: width, height, the cropped Y, Cb, Cr
// planes, a flag and the alpha plane, then RGB of all rows), meta (Metadata). Go returns no
// output with an error in any of them, so there is no partial output to compare.
func webpOps(e emitter, data []byte) {
	ctx := context.Background()
	cfg, cerr := webp.DecodeConfig(data)
	e.put("cfg", []byte(fmt.Sprintf("%d %d %t %t %d\n", cfg.Width, cfg.Height, cfg.HasAlpha, cfg.Lossless, cfg.FramesDropped)), cerr)

	m, err := webp.Decode(data, limits())
	var out []byte
	if err == nil {
		out = pixOut(m)
	}
	e.put("dec", out, err)

	p, err := webp.DecodePlanar(ctx, data, limits())
	out = nil
	if err == nil {
		out, err = planarOut(ctx, p)
	}
	e.put("planar", out, err)

	meta, err := webp.Metadata(data, nil)
	e.put("meta", metaOut(meta), err)

	if e.mut {
		return
	}
	// The limits: one pixel under the image fails, exactly the image passes.
	if cerr == nil {
		n := int64(cfg.Width) * int64(cfg.Height)
		for _, d := range []int64{-1, 0} {
			if n+d < 1 {
				continue
			}
			m, err := webp.Decode(data, &webp.Limits{MaxPixels: n + d})
			var out []byte
			if err == nil {
				out = pixOut(m)
			}
			e.putd(fmt.Sprintf("decl%+d", d), out, err)
		}
	}
	for _, n := range []int64{33, 100} {
		meta, err := webp.Metadata(data, &webp.Limits{MaxMetadataBytes: n})
		e.put(fmt.Sprintf("metal%d", n), metaOut(meta), err)
	}
}

func pixOut(m *image.NRGBA) []byte {
	w, h := m.Rect.Dx(), m.Rect.Dy()
	b := binary.LittleEndian.AppendUint32(nil, uint32(w))
	b = binary.LittleEndian.AppendUint32(b, uint32(h))
	for y := 0; y < h; y++ {
		b = append(b, m.Pix[y*m.Stride:y*m.Stride+4*w]...)
	}
	return b
}

func planarOut(ctx context.Context, p *webp.Planar) ([]byte, error) {
	w, h := p.Width, p.Height
	cw, ch := (w+1)/2, (h+1)/2
	b := binary.LittleEndian.AppendUint32(nil, uint32(w))
	b = binary.LittleEndian.AppendUint32(b, uint32(h))
	for y := 0; y < h; y++ {
		b = append(b, p.Y[y*p.YStride:y*p.YStride+w]...)
	}
	for _, pl := range [][]byte{p.Cb, p.Cr} {
		for y := 0; y < ch; y++ {
			b = append(b, pl[y*p.CStride:y*p.CStride+cw]...)
		}
	}
	if p.Alpha != nil {
		b = append(b, 1)
		b = append(b, p.Alpha...)
	} else {
		b = append(b, 0)
	}
	rgb := make([]byte, 3*w*h)
	if err := p.RGB(ctx, rgb, 0, h); err != nil {
		return nil, err
	}
	return append(b, rgb...), nil
}

// metaOut is one line per chunk: kind, index, truncated, and the digest of its bytes.
func metaOut(m webp.Meta) []byte {
	var b bytes.Buffer
	for i, c := range m.EXIF {
		fmt.Fprintf(&b, "exif %d %t %s", i, c.Truncated, digest(c.Data))
	}
	for i, c := range m.XMP {
		fmt.Fprintf(&b, "xmp %d %t %s", i, c.Truncated, digest(c.Data))
	}
	return b.Bytes()
}

// encFormat is the input of the encode op:
//
//	[kind u8: 0 NRGBA, 1 RGBA (premultiplied), 2 Gray, 3 YCbCr 4:2:0]
//	[method i8] [flags u8: 1 match_libwebp, 2 alpha_libwebp16, 4 legacy_alpha, 8 nil Options]
//	[segments i8] [quality f32 LE] [width u16 LE] [height u16 LE]
//	pixels: 4 bytes a pixel for kinds 0 and 1, 1 for kind 2, for kind 3 the Y plane then the Cb and
//	Cr planes of (w+1)/2 by (h+1)/2; all tight (stride is the row length).
const encHeader = 12

type encIn struct {
	kind, flags byte
	method, seg int
	quality     float32
	w, h        int
	pix         []byte
}

func (c *encIn) bytes() []byte {
	b := []byte{c.kind, byte(int8(c.method)), c.flags, byte(int8(c.seg))}
	b = binary.LittleEndian.AppendUint32(b, math.Float32bits(c.quality))
	b = binary.LittleEndian.AppendUint16(b, uint16(c.w))
	b = binary.LittleEndian.AppendUint16(b, uint16(c.h))
	return append(b, c.pix...)
}

func parseEnc(d []byte) (image.Image, *webp.Options, error) {
	if len(d) < encHeader {
		return nil, nil, errBadInput
	}
	c := encIn{kind: d[0], method: int(int8(d[1])), flags: d[2], seg: int(int8(d[3])),
		quality: math.Float32frombits(binary.LittleEndian.Uint32(d[4:8])),
		w:       int(binary.LittleEndian.Uint16(d[8:10])), h: int(binary.LittleEndian.Uint16(d[10:12])), pix: d[encHeader:]}
	if c.w*c.h > maxPixels {
		return nil, nil, errBadInput
	}
	cw, ch := (c.w+1)/2, (c.h+1)/2
	var need int
	switch c.kind {
	case 0, 1:
		need = 4 * c.w * c.h
	case 2:
		need = c.w * c.h
	case 3:
		need = c.w*c.h + 2*cw*ch
	default:
		return nil, nil, errBadInput
	}
	if len(c.pix) != need {
		return nil, nil, errBadInput
	}
	r := image.Rect(0, 0, c.w, c.h)
	var m image.Image
	switch c.kind {
	case 0:
		m = &image.NRGBA{Pix: c.pix, Stride: 4 * c.w, Rect: r}
	case 1:
		m = &image.RGBA{Pix: c.pix, Stride: 4 * c.w, Rect: r}
	case 2:
		m = &image.Gray{Pix: c.pix, Stride: c.w, Rect: r}
	case 3:
		n := c.w * c.h
		m = &image.YCbCr{Y: c.pix[:n], Cb: c.pix[n : n+cw*ch], Cr: c.pix[n+cw*ch:], YStride: c.w, CStride: cw,
			SubsampleRatio: image.YCbCrSubsampleRatio420, Rect: r}
	}
	var o *webp.Options
	if c.flags&8 == 0 {
		o = &webp.Options{Quality: c.quality, Method: c.method, MatchLibwebp: c.flags&1 != 0,
			AlphaLibwebp16: c.flags&2 != 0, LegacyAlpha: c.flags&4 != 0, Segments: c.seg}
	}
	return m, o, nil
}

func encode(m image.Image, o *webp.Options, enc *webp.Encoder, buf *bytes.Buffer) error {
	if enc != nil {
		buf.Reset()
		enc.Reset(buf)
		return enc.Encode(m, o)
	}
	switch m := m.(type) {
	case *image.Gray:
		return webp.EncodeGray(buf, m, o)
	case *image.YCbCr:
		return webp.EncodeYCbCr(buf, m, o)
	}
	return webp.Encode(buf, m, o)
}

func encOps(e emitter, data []byte) {
	m, o, err := parseEnc(data)
	var buf bytes.Buffer
	if err == nil {
		err = encode(m, o, nil, &buf)
	}
	e.put("enc", buf.Bytes(), err)
}

// bench prints, for each input of dir, the name, the pixels and the ns per operation: a decode
// for a .webp that decodes, an encode for a .enc that encodes. Each runs once to time it, then
// as many times as fit in about 200 ms, between 1 and n. Go reuses its Encoder and the output
// buffer.
func bench(n int, dir string) {
	ents, err := os.ReadDir(dir)
	must(err)
	var enc webp.Encoder
	var buf bytes.Buffer
	for _, ent := range ents {
		name := ent.Name()
		data, err := os.ReadFile(filepath.Join(dir, name))
		must(err)
		var op func() error
		var px int
		switch filepath.Ext(name) {
		case ".webp":
			c, err := webp.DecodeConfig(data)
			if err != nil {
				continue
			}
			px = c.Width * c.Height
			op = func() error { _, err := webp.Decode(data, limits()); return err }
		case ".enc":
			m, o, err := parseEnc(data)
			if err != nil {
				continue
			}
			px = m.Bounds().Dx() * m.Bounds().Dy()
			op = func() error { return encode(m, o, &enc, &buf) }
		default:
			continue
		}
		t := time.Now()
		if op() != nil {
			continue
		}
		iters := n
		if one := time.Since(t); one > 0 {
			iters = max(1, min(n, int(200*time.Millisecond/one)))
		}
		t = time.Now()
		for i := 0; i < iters; i++ {
			must(op())
		}
		fmt.Printf("%s\t%d\t%d\n", name, px, time.Since(t).Nanoseconds()/int64(iters))
	}
}

// sortedNames lists the files of dir with the extension ext, sorted.
func sortedNames(dir, ext string) []string {
	ents, err := os.ReadDir(dir)
	must(err)
	var s []string
	for _, e := range ents {
		if filepath.Ext(e.Name()) == ext {
			s = append(s, e.Name())
		}
	}
	sort.Strings(s)
	return s
}
