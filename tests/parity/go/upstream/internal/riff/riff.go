package riff

import (
	"encoding/binary"
	"io"
)

var padByte = []byte{0}

// Write emits a simple-format VP8 WebP file.
// Chunk size excludes pad, per spec.
// padSize counts the pad in the chunk size, as libwebp does.
func Write(w io.Writer, hdr *[20]byte, padSize bool, chunk ...[]byte) error {
	n := 0
	for _, c := range chunk {
		n += len(c)
	}
	pad := n & 1

	copy(hdr[0:4], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(4+8+n+pad))
	copy(hdr[8:12], "WEBP")
	copy(hdr[12:16], "VP8 ")
	sz := n
	if padSize {
		sz += pad
	}
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(sz))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	for _, c := range chunk {
		if _, err := w.Write(c); err != nil {
			return err
		}
	}
	if pad != 0 {
		if _, err := w.Write(padByte); err != nil {
			return err
		}
	}
	return nil
}

// WriteAlpha emits a VP8X file: VP8X with the alpha flag, ALPH, then VP8.
func WriteAlpha(w io.Writer, hdr *[46]byte, alph []byte, width, height int, padSize bool, chunk ...[]byte) error {
	n := 0
	for _, c := range chunk {
		n += len(c)
	}
	pad, apad := n&1, len(alph)&1

	copy(hdr[0:4], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(4+18+8+len(alph)+apad+8+n+pad))
	copy(hdr[8:12], "WEBP")
	copy(hdr[12:16], "VP8X")
	binary.LittleEndian.PutUint32(hdr[16:20], 10)
	hdr[20] = 0x10 // alpha
	hdr[21], hdr[22], hdr[23] = 0, 0, 0
	put24(hdr[24:27], width-1)
	put24(hdr[27:30], height-1)
	copy(hdr[30:34], "ALPH")
	binary.LittleEndian.PutUint32(hdr[34:38], uint32(len(alph)))
	copy(hdr[38:42], "VP8 ")
	sz := n
	if padSize {
		sz += pad
	}
	binary.LittleEndian.PutUint32(hdr[42:46], uint32(sz))
	if _, err := w.Write(hdr[:38]); err != nil {
		return err
	}
	if _, err := w.Write(alph); err != nil {
		return err
	}
	if apad != 0 {
		if _, err := w.Write(padByte); err != nil {
			return err
		}
	}
	if _, err := w.Write(hdr[38:]); err != nil {
		return err
	}
	for _, c := range chunk {
		if _, err := w.Write(c); err != nil {
			return err
		}
	}
	if pad != 0 {
		if _, err := w.Write(padByte); err != nil {
			return err
		}
	}
	return nil
}

func put24(b []byte, v int) { b[0], b[1], b[2] = byte(v), byte(v>>8), byte(v>>16) }
