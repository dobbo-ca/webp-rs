package webp

import (
	"context"
	"errors"
	"fmt"
	"image"

	"github.com/dobbo-ca/webp-go/internal/riff"
	"github.com/dobbo-ca/webp-go/internal/vp8dec"
	"github.com/dobbo-ca/webp-go/internal/vp8ldec"
)

var (
	// ErrAnimated is returned by DecodePlanar for an animated WebP, whose
	// canvas it cannot hold; Decode places the first frame on it.
	ErrAnimated = riff.ErrAnimated
	// ErrFormat is returned for data that is not a valid WebP file.
	ErrFormat = riff.ErrFormat
	// ErrTruncated is returned for a file that ends before its sizes say.
	ErrTruncated = riff.ErrTruncated
	// ErrLimit is wrapped by every *LimitError. It is permanent: the same
	// input and Limits fail again.
	ErrLimit = errors.New("webp: limit exceeded")
	// ErrMemoryLimit is wrapped when the memory budget is held by other
	// work and the decode gave up waiting (its ctx ended) or could not grow.
	// The same call can succeed later.
	ErrMemoryLimit = errors.New("webp: memory budget exhausted")
)

// LimitError reports a cap that the input is over.
type LimitError struct {
	Name     string
	Got, Max int64
}

func (e *LimitError) Error() string {
	return fmt.Sprintf("webp: %s %d over limit %d", e.Name, e.Got, e.Max)
}

func (e *LimitError) Unwrap() error { return ErrLimit }

// Limits caps one decode. The zero value is the safe default.
type Limits struct {
	// MaxPixels is the most pixels (width x height) a decode may produce;
	// 0 means 1<<26. Sides are always capped at 16383.
	MaxPixels int64

	// MaxMetadataBytes is the most EXIF and XMP bytes Metadata may return
	// together, each chunk also costing the 32 bytes of its slice entry;
	// 0 means 1<<24.
	MaxMetadataBytes int64
}

const (
	maxSide          = 16383
	defaultMaxPixels = 1 << 26
)

// Config is what a header says.
type Config struct {
	Width, Height int
	// HasAlpha is the VP8X alpha flag or, for a bare VP8L image, its
	// alpha-is-used bit. Decode returns the alpha the stream codes.
	HasAlpha bool
	Lossless bool
	// FramesDropped is how many frames of an animated file Decode leaves
	// out: all but the first. 0 for a still image.
	FramesDropped int
}

func parse(data []byte) (riff.File, Config, error) {
	f, err := riff.Parse(data)
	if err != nil {
		return f, Config{}, err
	}
	c := Config{Lossless: f.Lossless, HasAlpha: f.Alpha}
	w, h := 0, 0
	if f.Lossless {
		var a bool
		if w, h, a, err = vp8ldec.Header(f.Image); err != nil {
			return f, c, fmt.Errorf("%w: %w", ErrFormat, err)
		}
		c.HasAlpha = c.HasAlpha || a
	} else {
		if w, h, err = vp8dec.Header(f.Image); err != nil {
			return f, c, fmt.Errorf("%w: %w", ErrFormat, err)
		}
	}
	if w < 1 || h < 1 {
		return f, c, ErrFormat
	}
	if f.Frames == 0 {
		if f.Width != 0 && (f.Width != w || f.Height != h) {
			return f, c, ErrFormat // VP8X canvas and image disagree
		}
		f.FW, f.FH = w, h
	} else if f.FW != w || f.FH != h {
		return f, c, ErrFormat // ANMF and image disagree
	}
	c.Width, c.Height = w, h
	if f.Frames > 0 {
		c.Width, c.Height = f.Width, f.Height // the canvas
		c.FramesDropped = f.Frames - 1
	}
	return f, c, nil
}

func (l *Limits) check(c Config) error {
	max := int64(defaultMaxPixels)
	if l != nil && l.MaxPixels > 0 {
		max = l.MaxPixels
	}
	for _, s := range []struct {
		n string
		v int
	}{{"Width", c.Width}, {"Height", c.Height}} {
		if s.v > maxSide {
			return &LimitError{s.n, int64(s.v), maxSide}
		}
	}
	if p := int64(c.Width) * int64(c.Height); p > max {
		return &LimitError{"MaxPixels", p, max}
	}
	return nil
}

// DecodeConfig reads the image size from the headers, walking the chunks
// like Decode does. It allocates no pixels and takes no budget.
func DecodeConfig(data []byte) (Config, error) {
	_, c, err := parse(data)
	return c, err
}

// Decode is DecodeContext with a background context.
func Decode(data []byte, l *Limits) (*image.NRGBA, error) {
	return DecodeContext(context.Background(), data, l)
}

// DecodeContext decodes a still WebP, or the first frame of an animated
// one, placed at its offset on a transparent canvas of the file's size (as
// libwebp's WebPAnimDecoder does); Config.FramesDropped counts the rest.
// Lossless images decode bit exact
// with libwebp. The pixels are reserved from the SetMemoryLimit budget
// before anything is allocated: 4 bytes per pixel plus the VP8L transform
// images, prefix codes and colour cache, each reserved before it is
// allocated. If the budget is held by other work it waits in arrival order
// until ctx is done, then returns an error wrapping ErrMemoryLimit and
// ctx.Err(). ctx is checked once a pixel row and a transform row.
func DecodeContext(ctx context.Context, data []byte, l *Limits) (*image.NRGBA, error) {
	f, c, err := parse(data)
	if err != nil {
		return nil, err
	}
	if err := l.check(c); err != nil {
		return nil, err
	}
	if !f.Lossless && f.Alph != nil && badAlph(f.Alph) {
		return nil, fmt.Errorf("%w: ALPH header", ErrFormat)
	}
	fc := c // the frame; the canvas, if bigger, is c
	fc.Width, fc.Height = f.FW, f.FH
	w, h := int64(fc.Width), int64(fc.Height)
	n := 4 * w * h
	if !f.Lossless {
		n += vp8dec.PlaneBytes(fc.Width, fc.Height)
		if f.Alph != nil {
			n += w * h
		}
	}
	bigger := fc.Width != c.Width || fc.Height != c.Height
	if bigger {
		n += 4 * int64(c.Width) * int64(c.Height)
	}
	r, err := reserve(ctx, n)
	if err != nil {
		return nil, err
	}
	defer r.release()
	m := image.NewNRGBA(image.Rect(0, 0, fc.Width, fc.Height))
	if f.Lossless {
		err = vp8ldec.Decode(ctx, f.Image, m.Pix, r.grow)
	} else {
		err = decodeLossy(ctx, f.Image, f.Alph, fc, m.Pix, r)
	}
	if err != nil {
		return nil, err
	}
	if bigger {
		canvas := image.NewNRGBA(image.Rect(0, 0, c.Width, c.Height))
		for y := 0; y < fc.Height; y++ {
			copy(canvas.Pix[4*((f.Y+y)*c.Width+f.X):], m.Pix[4*y*fc.Width:4*(y+1)*fc.Width])
		}
		m = canvas
	}
	return m, nil
}

func badAlph(a []byte) bool {
	return len(a) < 2 || a[0]>>6 != 0 || a[0]>>4&3 > 1 || a[0]&3 > 1
}

// decodeLossy decodes a VP8 frame to YUV planes, then to RGBA in pix, with
// the alpha of an ALPH chunk if there is one.
func decodeLossy(ctx context.Context, data, alph []byte, c Config, pix []byte, r *decodeRes) error {
	n := c.Width * c.Height
	var alpha []byte
	if alph != nil {
		alpha = make([]byte, n)
		if err := readAlph(ctx, alph, c, alpha, r); err != nil {
			return err
		}
	}
	fr, err := decodePlanes(ctx, data, c, r)
	if err != nil {
		return err
	}
	if err := fr.RGBA(ctx, pix); err != nil {
		return err
	}
	for i, a := range alpha {
		pix[4*i+3] = a
	}
	return nil
}

// readAlph reads an ALPH chunk (WebP container spec, "Alpha") into the
// plane, one byte per pixel.
func readAlph(ctx context.Context, alph []byte, c Config, plane []byte, r *decodeRes) error {
	w, h := c.Width, c.Height
	filter := alph[0] >> 2 & 3
	if alph[0]&3 == 0 {
		if len(alph)-1 < w*h {
			return fmt.Errorf("%w: ALPH data", ErrTruncated)
		}
		copy(plane, alph[1:])
	} else if err := vp8ldec.DecodeAlpha(ctx, alph[1:], w, h, plane, r.grow); err != nil {
		return fmt.Errorf("%w: ALPH: %w", ErrFormat, err)
	}
	unfilter(plane, w, h, filter)
	return nil
}

// unfilter undoes the alpha prediction filters (0 none, 1 horizontal,
// 2 vertical, 3 gradient) in place. The first row, and the first pixel of
// every row but for the vertical and gradient filters, predict from the
// left or are raw, as libwebp's HorizontalUnfilter, VerticalUnfilter and
// GradientUnfilter do.
func unfilter(a []byte, w, h int, filter byte) {
	if filter == 0 {
		return
	}
	for y := 0; y < h; y++ {
		row := a[y*w : (y+1)*w]
		var prev []byte
		if y > 0 {
			prev = a[(y-1)*w : y*w]
		}
		if prev == nil || filter == 1 {
			pred := byte(0)
			if prev != nil {
				pred = prev[0]
			}
			for i := range row {
				row[i] += pred
				pred = row[i]
			}
			continue
		}
		if filter == 2 {
			for i := range row {
				row[i] += prev[i]
			}
			continue
		}
		left, tl := prev[0], prev[0]
		for i := range row {
			top := prev[i]
			g := int(left) + int(top) - int(tl)
			row[i] += byte(max(0, min(g, 255)))
			left, tl = row[i], top
		}
	}
}

func decodePlanes(ctx context.Context, data []byte, c Config, r *decodeRes) (*vp8dec.Frame, error) {
	pw, ph := (c.Width+15)&^15, (c.Height+15)&^15
	planes := make([]byte, pw*ph*3/2)
	return vp8dec.Decode(ctx, data, planes[:pw*ph], planes[pw*ph:pw*ph*5/4], planes[pw*ph*5/4:], r.grow)
}

// ErrNotLossy is returned by DecodePlanar for a lossless image, which has
// no YCbCr planes.
var ErrNotLossy = errors.New("webp: image is lossless")

// Planar is a lossy image as the 4:2:0 planes the VP8 frame decodes to.
type Planar struct {
	Width, Height    int
	Y, Cb, Cr        []byte
	YStride, CStride int
	// Alpha is one byte a pixel, Width per row; nil if there is no ALPH chunk.
	Alpha []byte
	f     *vp8dec.Frame
}

// DecodePlanar decodes a lossy image to its planes only, reserving
// vp8dec.PlaneBytes (1.5 bytes a pixel, the sides rounded up to 16) from the
// budget, not the 4 bytes a pixel of RGBA that DecodeContext adds. Convert
// with RGB, a band of rows at a time, into memory the caller holds.
func DecodePlanar(ctx context.Context, data []byte, l *Limits) (*Planar, error) {
	f, c, err := parse(data)
	if err != nil {
		return nil, err
	}
	if err := l.check(c); err != nil {
		return nil, err
	}
	if f.Frames > 0 {
		return nil, ErrAnimated
	}
	if f.Lossless {
		return nil, ErrNotLossy
	}
	if f.Alph != nil && badAlph(f.Alph) {
		return nil, fmt.Errorf("%w: ALPH header", ErrFormat)
	}
	n := vp8dec.PlaneBytes(c.Width, c.Height)
	if f.Alph != nil {
		n += int64(c.Width) * int64(c.Height)
	}
	r, err := reserve(ctx, n)
	if err != nil {
		return nil, err
	}
	defer r.release()
	var alpha []byte
	if f.Alph != nil {
		alpha = make([]byte, c.Width*c.Height)
		if err := readAlph(ctx, f.Alph, c, alpha, r); err != nil {
			return nil, err
		}
	}
	fr, err := decodePlanes(ctx, f.Image, c, r)
	if err != nil {
		return nil, err
	}
	return &Planar{c.Width, c.Height, fr.Y, fr.U, fr.V, fr.YStride, fr.CStrd, alpha, fr}, nil
}

// RGB writes rows y0 to y1 (exclusive) as 3 bytes a pixel into dst, which
// must hold exactly those rows. The conversion is DecodeContext's: libwebp's
// fancy upsampling, so a whole image in bands equals the one-shot RGBA.
func (p *Planar) RGB(ctx context.Context, dst []byte, y0, y1 int) error {
	if y0 < 0 || y1 > p.Height || y0 > y1 || len(dst) != 3*p.Width*(y1-y0) {
		return fmt.Errorf("webp: RGB rows %d-%d of %d with %d bytes", y0, y1, p.Height, len(dst))
	}
	return p.f.Rows(ctx, dst, y0, y1, 3)
}

// decodeRes is one decode's share of the budget.
type decodeRes struct{ held int64 }

type decodeWaiter struct {
	need  int64
	ready chan struct{}
}

// reserve takes need bytes of the budget, waiting its turn.
func reserve(ctx context.Context, need int64) (*decodeRes, error) {
	lim.mu.Lock()
	if b := budget(); need > b {
		lim.mu.Unlock()
		return nil, &LimitError{"MemoryBudget", need, b}
	}
	if len(lim.dq) == 0 && lim.reserved+need+garbageNow() <= budget() {
		lim.reserved += need
		lim.mu.Unlock()
		return &decodeRes{need}, nil
	}
	w := &decodeWaiter{need, make(chan struct{})}
	lim.dq = append(lim.dq, w)
	collect()
	lim.mu.Unlock()
	select {
	case <-w.ready:
		return &decodeRes{need}, nil
	case <-ctx.Done():
	}
	lim.mu.Lock()
	defer lim.mu.Unlock()
	for i, q := range lim.dq {
		if q == w {
			lim.dq = append(lim.dq[:i], lim.dq[i+1:]...)
			wake() // w may have blocked the head
			return nil, fmt.Errorf("%w: %w", ErrMemoryLimit, ctx.Err())
		}
	}
	// Admitted as ctx ended: give the bytes straight back.
	lim.reserved -= need
	wake()
	return nil, fmt.Errorf("%w: %w", ErrMemoryLimit, ctx.Err())
}

// grow reserves n more bytes at once, without waiting: a decode that
// waited while holding some could deadlock with another doing the same.
func (r *decodeRes) grow(n int64) error {
	lim.mu.Lock()
	defer lim.mu.Unlock()
	if b := budget(); r.held+n > b {
		return &LimitError{"MemoryBudget", r.held + n, b}
	}
	if lim.reserved+n+garbageNow() > budget() {
		return fmt.Errorf("%w: need %d more bytes", ErrMemoryLimit, n)
	}
	lim.reserved += n
	r.held += n
	return nil
}

func (r *decodeRes) release() {
	lim.mu.Lock()
	lim.reserved -= r.held
	wake()
	lim.mu.Unlock()
}
