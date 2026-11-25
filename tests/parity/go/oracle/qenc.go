package main

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	webp "github.com/dobbo-ca/webp-go"
)

// qencSettings are the five encode settings of -genbench, all at quality 75 and segments 0: plain
// methods 0, 4 and 6, and match_libwebp methods 4 and 6. The tag is part of the output names.
var qencSettings = []struct {
	tag    string
	method int
	match  bool
}{{"m0", 0, false}, {"m4", 4, false}, {"m6", 6, false}, {"lw4", 4, true}, {"lw6", 6, true}}

// pnmImage reads a binary PGM (P5) or PPM (P6) with maxval 255 as image.Gray or image.NRGBA
// (alpha 255). The header is at most 64 bytes of tokens and the pixel count is capped by maxPixels.
// Tokens are split by space, tab, CR and LF only, there are no comments, and a number is plain
// decimal (strconv.Atoi), as in examples/quality.rs, so the two accept the same headers.
func pnmImage(d []byte) (image.Image, error) {
	pos := 0
	tok := func() string {
		for pos < len(d) && (d[pos] == ' ' || d[pos] == '\n' || d[pos] == '\t' || d[pos] == '\r') {
			pos++
		}
		s := pos
		for pos < len(d) && pos-s < 16 && d[pos] != ' ' && d[pos] != '\n' && d[pos] != '\t' && d[pos] != '\r' {
			pos++
		}
		return string(d[s:pos])
	}
	magic := tok()
	w, errW := strconv.Atoi(tok())
	h, errH := strconv.Atoi(tok())
	mx, errMx := strconv.Atoi(tok())
	if errW != nil || errH != nil || errMx != nil {
		return nil, errBadInput
	}
	pos++ // the single whitespace byte after maxval
	if (magic != "P5" && magic != "P6") || mx != 255 || w < 1 || h < 1 || w > 1<<16 || h > 1<<16 || w*h > maxPixels || pos > len(d) {
		return nil, errBadInput
	}
	ch := 1
	if magic == "P6" {
		ch = 3
	}
	pix := d[pos:]
	if len(pix) != ch*w*h {
		return nil, errBadInput
	}
	r := image.Rect(0, 0, w, h)
	if ch == 1 {
		return &image.Gray{Pix: pix, Stride: w, Rect: r}, nil
	}
	m := &image.NRGBA{Pix: make([]byte, 4*w*h), Stride: 4 * w, Rect: r}
	for i := 0; i < w*h; i++ {
		copy(m.Pix[4*i:], pix[3*i:3*i+3])
		m.Pix[4*i+3] = 255
	}
	return m, nil
}

// ppmOut is the P6 file of an image, alpha dropped.
func ppmOut(m *image.NRGBA) []byte {
	w, h := m.Rect.Dx(), m.Rect.Dy()
	b := []byte(fmt.Sprintf("P6\n%d %d\n255\n", w, h))
	for y := 0; y < h; y++ {
		row := m.Pix[y*m.Stride:]
		for x := 0; x < w; x++ {
			b = append(b, row[4*x:4*x+3]...)
		}
	}
	return b
}

// qenc is `oracle -qenc <images> <webp-out> <dec-out>`: every .ppm and .pgm of the images directory
// (sorted) is encoded under each of qencSettings with one reused Encoder, to <webp-out>/<stem>_<tag>.webp,
// then decoded back to <dec-out>/<stem>_<tag>.ppm. An error writes <webp-out>/<stem>_<tag>.err with
// its class token.
func qenc(cells, webpOut, decOut string) {
	must(os.MkdirAll(webpOut, 0o755))
	must(os.MkdirAll(decOut, 0o755))
	ents, err := os.ReadDir(cells)
	must(err)
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		if ext := filepath.Ext(e.Name()); ext == ".ppm" || ext == ".pgm" {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	var enc webp.Encoder
	var buf bytes.Buffer
	for _, name := range names {
		stem := strings.TrimSuffix(name, filepath.Ext(name))
		d, err := os.ReadFile(filepath.Join(cells, name))
		must(err)
		m, perr := pnmImage(d)
		for _, s := range qencSettings {
			base := stem + "_" + s.tag
			err := perr
			if err == nil {
				err = encode(m, &webp.Options{Quality: 75, Method: s.method, MatchLibwebp: s.match}, &enc, &buf)
			}
			if err != nil {
				must(os.WriteFile(filepath.Join(webpOut, base+".err"), []byte(class(err)+"\n"), 0o644))
				continue
			}
			must(os.WriteFile(filepath.Join(webpOut, base+".webp"), buf.Bytes(), 0o644))
			img, err := webp.Decode(buf.Bytes(), limits())
			if err != nil {
				must(os.WriteFile(filepath.Join(decOut, base+".err"), []byte(class(err)+"\n"), 0o644))
				continue
			}
			must(os.WriteFile(filepath.Join(decOut, base+".ppm"), ppmOut(img), 0o644))
		}
	}
}
