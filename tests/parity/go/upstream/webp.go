// Package webp encodes lossy WebP and decodes still WebP (see Decode).
//
// At most GOMAXPROCS encodes run at once, served in arrival order. With
// GOMEMLIMIT set, encodes in flight also hold at most half of it: each is
// admitted on a per-pixel estimate learned from finished encodes and
// charges its token pages as they grow, each with the most partition and
// output bytes its tokens can code to, and its ALPH chunk. One that would
// go past the budget stops and reruns from scratch, unless it is the
// oldest in flight, which waits instead; an image too big for the budget
// runs alone. An Encoder that waits, or finishes while others wait, parks
// its buffers for the next encode. Outside the budget an Encoder whose
// last encode was opaque holds its planes, small slices and at most 4 MiB
// of idle partition, alpha and pooled output buffers; after an encode
// with alpha it also keeps its alpha plane, ALPH and alpha coder state.
// While the Writer runs, the trims are done and the output buffer, up to
// 5/4 of the output, stands in for the pooled one. Set GOMEMLIMIT 10 to 15% below the container memory
// limit and to at least twice the largest single encode's peak, see
// docs/memory.md. MaxEncodeCharge gives one encode's most.
// SetMemoryLimit overrides the budget this package takes from
// GOMEMLIMIT, for a process that shares it with other codecs.
//
// An encode codes into its Encoder's own buffer and gives its slot and
// budget back before it writes, so a slow Writer holds neither and a
// Writer may call Encode. EncodeContext stops waiting or encoding when
// its context is done.
package webp

import (
	"context"
	"image"
	"io"

	iwebp "github.com/dobbo-ca/webp-go/internal/webp"
)

type (
	// Options controls the encoder. Method is 0..6 like cwebp -m; nil
	// means Quality 75 Method 4, a zero Options is Method 0.
	//
	// MatchLibwebp makes Methods 3 to 6 decide and write exactly as
	// libwebp 1.6.0 (and 1.2.2) does with default settings, so the output
	// bytes equal cwebp -q Quality -m Method. It costs speed: every intra
	// mode gets a full rate-distortion trial. Methods 0 to 2 ignore it.
	// ALPH matches libwebp 1.2.2, or 1.6.0 with AlphaLibwebp16.
	Options = iwebp.Options

	// Encoder keeps its buffers between calls.
	Encoder = iwebp.Encoder

	// Limits caps one decode; the zero value is the safe default.
	Limits = iwebp.Limits

	// LimitError reports a cap the input is over; it wraps ErrLimit.
	LimitError = iwebp.LimitError

	// Config is what a WebP header says.
	Config = iwebp.Config

	// Meta is the EXIF and XMP chunks of a file; see Metadata.
	Meta = iwebp.Meta

	// MetaChunk is one metadata chunk body and whether it was cut short.
	MetaChunk = iwebp.MetaChunk

	// Planar is a lossy image as 4:2:0 planes; see DecodePlanar.
	Planar = iwebp.Planar
)

var (
	ErrAnimated    = iwebp.ErrAnimated
	ErrFormat      = iwebp.ErrFormat
	ErrTruncated   = iwebp.ErrTruncated
	ErrLimit       = iwebp.ErrLimit
	ErrMemoryLimit = iwebp.ErrMemoryLimit
	ErrNotLossy    = iwebp.ErrNotLossy
)

// DecodeConfig reads the size and kind of a WebP from its headers
// without allocating pixels. An animated file reports its canvas size and,
// in FramesDropped, the frames Decode leaves out.
func DecodeConfig(data []byte) (Config, error) {
	return iwebp.DecodeConfig(data)
}

// Decode decodes a WebP to non-premultiplied RGBA, bit exact with
// dwebp (libwebp's default fancy chroma upsampling, no dithering; alpha
// from an ALPH chunk, filters and compression included). An animated
// file decodes to its first frame only, placed at its ANMF offset on a
// transparent canvas of the file's size (as libwebp's WebPAnimDecoder
// does), a frame outside the canvas is ErrFormat, and Limits apply to
// the canvas. DecodeConfig counts the dropped frames. l may be nil. The pixels (4 bytes each, and the decoder's working
// memory) are taken from the SetMemoryLimit budget before they are
// allocated.
func Decode(data []byte, l *Limits) (*image.NRGBA, error) {
	return iwebp.Decode(data, l)
}

// DecodeContext is Decode that waits for budget, and checks ctx once a
// pixel row, until ctx is done.
func DecodeContext(ctx context.Context, data []byte, l *Limits) (*image.NRGBA, error) {
	return iwebp.DecodeContext(ctx, data, l)
}

// Metadata returns every EXIF and XMP chunk of a WebP file in file order;
// see internal/riff for the walk. Bytes alias data, a chunk cut short by
// the file or RIFF size has Truncated set, and more than
// Limits.MaxMetadataBytes of them returns ErrLimit. l may be nil.
func Metadata(data []byte, l *Limits) (Meta, error) {
	return iwebp.Metadata(data, l)
}

// DecodePlanar decodes a lossy still WebP to its Y, Cb and Cr planes, which
// take 1.5 bytes a pixel of the SetMemoryLimit budget (the sides rounded up
// to 16) where Decode takes 5.5. Planar.RGB then converts a band of rows at
// a time, bit exact with Decode, into memory the caller holds. Alpha from an
// ALPH chunk is a 1 byte plane (2.5 bytes a pixel in all); a compressed
// one also holds 4 bytes a pixel while it decodes. A lossless
// image returns ErrNotLossy; an animated file returns ErrAnimated. Check
// DecodeConfig first.
func DecodePlanar(ctx context.Context, data []byte, l *Limits) (*Planar, error) {
	return iwebp.DecodePlanar(ctx, data, l)
}

func Encode(w io.Writer, m image.Image, o *Options) error {
	return iwebp.Encode(w, m, o)
}

// EncodeContext is Encode that gives up, returning ctx.Err(), when ctx is
// done while it waits for a slot or budget or while it encodes. Nothing
// is written then.
func EncodeContext(ctx context.Context, w io.Writer, m image.Image, o *Options) error {
	return iwebp.EncodeContext(ctx, w, m, o)
}

// EncodeGray matches Encode on the same image.
func EncodeGray(w io.Writer, m *image.Gray, o *Options) error {
	return iwebp.EncodeGray(w, m, o)
}

func EncodeYCbCr(w io.Writer, m *image.YCbCr, o *Options) error {
	return iwebp.EncodeYCbCr(w, m, o)
}

// SetMemoryLimit sets how many bytes of heap the encodes in flight may
// hold together, and returns the previous limit. n <= 0 means no limit.
// Until it is called, the limit is half of GOMEMLIMIT (see
// runtime/debug.SetMemoryLimit) when one is set, else, on Linux, half of
// the cgroup memory limit (v2 memory.max for this process's cgroup, else
// the root's, else v1 memory.limit_in_bytes) read once, else none. Lowering the limit can
// make an encode already running restart, as in docs/memory.md; a waiting
// encode is woken to recheck it either way.
func SetMemoryLimit(n int64) int64 {
	return iwebp.SetMemoryLimit(n)
}

// MaxEncodeCharge is the most budget one opaque w x h encode can charge,
// on a fresh Encoder or one whose last encode was opaque, last used on
// images no bigger. Under a SetMemoryLimit budget below it, such an
// encode can end up running alone. Encodes with alpha also charge their
// ALPH chunk and its scratch, and an Encoder holding alpha state from an
// earlier encode charges that state on every encode.
func MaxEncodeCharge(w, h int) int64 {
	return iwebp.MaxEncodeCharge(w, h)
}

// MaxEncodedSize is the most bytes one opaque w x h encode writes. The
// Encoder's output buffer, up to 5/4 of that, sits outside the budget
// while the Writer runs.
func MaxEncodedSize(w, h int) int64 {
	return iwebp.MaxEncodedSize(w, h)
}
