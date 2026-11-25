package riff

import (
	"encoding/binary"
	"errors"
)

var (
	// ErrFormat is returned for data that is not a WebP file.
	ErrFormat = errors.New("webp: invalid format")
	// ErrTruncated is returned when a size field reaches past the file.
	ErrTruncated = errors.New("webp: truncated file")
	// ErrAnimated is returned where an animated WebP cannot be handled.
	ErrAnimated = errors.New("webp: animated images are not supported")
)

// File is the still image, or the first frame of an animation, found in a
// WebP file. Slices alias the input.
type File struct {
	Lossless      bool // Image is a VP8L chunk, else VP8
	Width, Height int  // canvas from VP8X, else the image's own
	Alpha         bool // the VP8X alpha flag; VP8L's own flag is in its header
	Image         []byte
	Alph          []byte // last ALPH chunk before a VP8 chunk, as libwebp reads it, else nil

	// Frames counts the ANMF chunks of an animation, 0 for a still. Then
	// X, Y, FW and FH are the first frame's ANMF offset and size on the canvas.
	Frames, X, Y, FW, FH int
}

// Parse walks the RIFF and VP8X chunks, checking every size against the
// data. It reads no image header: Width and Height are the VP8X canvas,
// or 0 for a file without one. An animation yields its first frame; the
// others are only counted.
func Parse(data []byte) (File, error) {
	var f File
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return f, ErrFormat
	}
	if end := uint64(binary.LittleEndian.Uint32(data[4:8])) + 8; end > uint64(len(data)) {
		return f, ErrTruncated
	} else if end < 12 {
		return f, ErrFormat
	} else {
		data = data[:end]
	}
	pos, vp8x, anim := 12, false, false
	for pos+8 <= len(data) {
		tag, n := string(data[pos:pos+4]), uint64(binary.LittleEndian.Uint32(data[pos+4:pos+8]))
		if n > uint64(len(data)-pos-8) {
			return f, ErrTruncated
		}
		body := data[pos+8 : pos+8+int(n)]
		pos += 8 + int(n) + int(n&1)
		switch tag {
		case "VP8X":
			if vp8x || n != 10 {
				return f, ErrFormat
			}
			vp8x = true
			anim = body[0]&0x02 != 0
			f.Alpha = body[0]&0x10 != 0
			f.Width = 1 + (int(body[4]) | int(body[5])<<8 | int(body[6])<<16)
			f.Height = 1 + (int(body[7]) | int(body[8])<<8 | int(body[9])<<16)
		case "ALPH":
			if anim {
				return f, ErrFormat // an animation's ALPH lives inside its ANMF
			}
			if vp8x {
				f.Alph = body
			}
		case "ANMF":
			if !anim {
				return f, ErrFormat
			}
			// Every ANMF must hold an image and lie in the canvas (stricter than
			// libwebp, which skips an image-less one); only the first is kept.
			g := &f
			if f.Frames++; f.Frames > 1 {
				g = &File{Width: f.Width, Height: f.Height}
			}
			if err := firstFrame(body, g); err != nil {
				return f, err
			}
		case "VP8 ", "VP8L":
			if anim {
				return f, ErrFormat
			}
			f.Lossless = tag == "VP8L"
			f.Image = body
			if f.Lossless {
				f.Alph = nil // ALPH belongs to VP8 only
			}
			return f, nil
		}
		if !vp8x && tag != "VP8X" {
			return f, ErrFormat // the first chunk is VP8, VP8L or VP8X
		}
	}
	if f.Frames > 0 {
		return f, nil
	}
	return f, ErrFormat
}

// firstFrame reads an ANMF body: the frame's offset and size (checked to lie
// in the canvas), then its ALPH and VP8 or VP8L chunks.
func firstFrame(b []byte, f *File) error {
	if len(b) < 16 {
		return ErrFormat
	}
	u24 := func(i int) int { return int(b[i]) | int(b[i+1])<<8 | int(b[i+2])<<16 }
	f.X, f.Y, f.FW, f.FH = 2*u24(0), 2*u24(3), 1+u24(6), 1+u24(9)
	if f.X+f.FW > f.Width || f.Y+f.FH > f.Height {
		return ErrFormat
	}
	for pos := 16; pos+8 <= len(b); {
		tag, n := string(b[pos:pos+4]), uint64(binary.LittleEndian.Uint32(b[pos+4:pos+8]))
		if n > uint64(len(b)-pos-8) {
			return ErrTruncated
		}
		body := b[pos+8 : pos+8+int(n)]
		pos += 8 + int(n) + int(n&1)
		switch tag {
		case "ALPH":
			f.Alph = body
		case "VP8 ", "VP8L":
			f.Lossless = tag == "VP8L"
			f.Image = body
			if f.Lossless {
				f.Alph = nil
			}
			return nil
		}
	}
	return ErrFormat
}
