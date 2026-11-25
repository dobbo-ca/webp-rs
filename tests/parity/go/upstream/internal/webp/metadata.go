package webp

import (
	"errors"

	"github.com/dobbo-ca/webp-go/internal/riff"
)

type (
	// MetaChunk is one EXIF or XMP chunk body; see Metadata.
	MetaChunk = riff.Chunk
	// Meta holds every EXIF and XMP chunk, each in file order.
	Meta = riff.Meta
)

const defaultMaxMetadata = 1 << 24

// Metadata returns every EXIF and XMP chunk of a WebP file, still or
// animated, in file order. The bytes alias data. A chunk whose declared
// size runs past the end of the file or the RIFF size is returned cut
// there with Truncated set. If the bytes of all the chunks together, plus
// riff.ChunkCost for each chunk, are over Limits.MaxMetadataBytes it stops
// walking and returns a *LimitError and no chunks. l may be nil.
func Metadata(data []byte, l *Limits) (Meta, error) {
	max := int64(defaultMaxMetadata)
	if l != nil && l.MaxMetadataBytes > 0 {
		max = l.MaxMetadataBytes
	}
	m, got, err := riff.Metadata(data, max)
	if errors.Is(err, riff.ErrMetaLimit) {
		return Meta{}, &LimitError{"MetadataBytes", got, max}
	}
	return m, err
}
