package riff

import (
	"encoding/binary"
	"errors"
	"strings"
	"unsafe"
)

// ErrMetaLimit is returned by Metadata when the chunks cost more than max.
var ErrMetaLimit = errors.New("webp: metadata over limit")

// ChunkCost is what each collected chunk is charged on top of its bytes: the
// Chunk it takes in a slice, so empty chunks cannot be collected for free.
const ChunkCost = int64(unsafe.Sizeof(Chunk{}))

// Chunk is one metadata chunk body. Data aliases the input and holds only
// the bytes inside the file and the RIFF size; Truncated is set when the
// chunk's declared size is more than that.
type Chunk struct {
	Data      []byte
	Truncated bool
}

// Meta is every EXIF and XMP chunk, each in file order.
type Meta struct{ EXIF, XMP []Chunk }

// Metadata walks all top-level chunks, past the image, and collects the
// EXIF and XMP ones. A header cut short by the end of the file or RIFF size
// after the start of an EXIF or XMP tag is a Truncated chunk with no data. A RIFF size past the file is clamped to it. Each
// collected chunk costs ChunkCost plus its bytes; as soon as the total is
// over max the walk stops with ErrMetaLimit and the total so far.
func Metadata(data []byte, max int64) (m Meta, got int64, err error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return m, 0, ErrFormat
	}
	end := uint64(binary.LittleEndian.Uint32(data[4:8])) + 8
	if end < 12 {
		return m, 0, ErrFormat
	}
	if end > uint64(len(data)) {
		end = uint64(len(data))
	}
	pos := uint64(12)
	for pos+8 <= end {
		tag, n := string(data[pos:pos+4]), uint64(binary.LittleEndian.Uint32(data[pos+4:pos+8]))
		pos += 8
		c := Chunk{Truncated: n > end-pos}
		if c.Truncated {
			n = end - pos
		}
		c.Data = data[pos : pos+n]
		if tag == "EXIF" || tag == "XMP " {
			if got += ChunkCost + int64(n); got > max {
				return Meta{}, got, ErrMetaLimit
			}
		}
		switch tag {
		case "EXIF":
			m.EXIF = append(m.EXIF, c)
		case "XMP ":
			m.XMP = append(m.XMP, c)
		}
		if c.Truncated {
			return m, got, nil
		}
		pos += n + n&1
	}
	// Under 8 bytes left: a header cut off after the start of an EXIF or XMP
	// tag is that chunk, cut, with no data.
	if pos < end {
		t := string(data[pos:min(pos+4, end)])
		for _, tag := range []string{"EXIF", "XMP "} {
			if strings.HasPrefix(tag, t) {
				if got += ChunkCost; got > max {
					return Meta{}, got, ErrMetaLimit
				}
				c := Chunk{Truncated: true}
				if tag == "EXIF" {
					m.EXIF = append(m.EXIF, c)
				} else {
					m.XMP = append(m.XMP, c)
				}
			}
		}
	}
	return m, got, nil
}
