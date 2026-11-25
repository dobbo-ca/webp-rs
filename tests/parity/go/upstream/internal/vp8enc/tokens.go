package vp8enc

import (
	"encoding/binary"
	"errors"
	"math/bits"
)

// RFC 6386 13.4, BSD-3 spec code.
var coeffUpdateProbs = [4][8][3][11]uint8{
	{
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255},
			{250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255},
			{234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255},
		},
		{
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255},
			{234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255},
			{251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255},
		},
		{
			{255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
	{
		{
			{248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255},
			{248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
	},
}

// From x/image/vp8 defaultTokenProb (BSD-3), equals libwebp VP8CoeffsProba0.
var coeffProbs = [4][8][3][11]uint8{
	{
		{
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128},
			{189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128},
			{106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128},
		},
		{
			{1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128},
			{181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128},
			{78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128},
		},
		{
			{1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128},
			{184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128},
			{77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128},
		},
		{
			{1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128},
			{170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128},
			{37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128},
		},
		{
			{1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128},
			{207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128},
			{102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128},
		},
		{
			{1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128},
			{177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128},
			{80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128},
		},
		{
			{1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62},
			{131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1},
			{68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128},
		},
		{
			{1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128},
			{184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128},
			{81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128},
		},
		{
			{1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128},
			{99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128},
			{23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128},
		},
		{
			{1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128},
			{109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128},
			{44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128},
		},
		{
			{1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128},
			{94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128},
			{22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128},
		},
		{
			{1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128},
			{124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128},
			{35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128},
		},
		{
			{1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128},
			{121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128},
			{45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128},
		},
		{
			{1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128},
			{203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128},
			{175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128},
			{73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128},
		},
		{
			{1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128},
			{239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128},
			{155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128},
		},
		{
			{1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128},
			{201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128},
			{69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128},
		},
		{
			{1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128},
			{223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128},
			{141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128},
			{149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128},
			{213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128},
			{55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
		{
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
	{
		{
			{202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255},
			{126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128},
			{61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128},
		},
		{
			{1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128},
			{166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128},
			{39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128},
		},
		{
			{1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128},
			{124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128},
			{24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128},
		},
		{
			{1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128},
			{149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128},
			{28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128},
		},
		{
			{1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128},
			{123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128},
			{20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128},
		},
		{
			{1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128},
			{168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128},
			{47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128},
		},
		{
			{1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128},
			{141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128},
			{42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128},
		},
		{
			{1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
		},
	},
}

var bands = [17]uint8{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0}

// RFC 6386 13.2 extra-bit probabilities for cat3..cat6.
var catProbs = [4][]uint8{
	{173, 148, 140},
	{176, 155, 140, 135},
	{180, 157, 141, 134, 130},
	{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129},
}

// Nonzero flags: Y 0..3, U 4..5, V 6..7, Y2 8.
type nzCtx [9]uint8

// Ported from libwebp PutCoeffs (frame_enc.c), BSD-3-Clause.
// tokenBuf holds (bit, prob) pairs until probs are final.
// Bit 15 is the value, bit 14 marks a fixed prob in bits 0-7,
// otherwise bits 0-10 index a flat prob table.
// Fixed pages, as libwebp token_enc.c: no append growth garbage.
// maxBlockTok bounds tokens one block can record: per coeff a zero
// flag, then recordLevel's worst case (v>1, v>4, v>10, k>=2, k&1, 11
// cat6 bits = 16), then a sign and EOB (19), plus a leading EOB.
const maxBlockTok = 16*19 + 1

type tokenBuf struct {
	pages [][]uint16 // kept across encodes; pages[:n] are filled
	n     int
	cur   []uint16 // pages[n], being filled
	base  int64    // bytes held besides pages
	parts int64    // partition 0 bound, charged with the pages
	lim   Limiter
}

// Limiter hands out pages; ok false abandons the encode.
type Limiter interface {
	Page(held int64) (p []uint16, ok bool)
}

// ErrRetry means the limiter abandoned the encode.
var ErrRetry = errors.New("vp8enc: retry")

const pageBytes = tokPage * 2

// A token is one putBit, which shifts out at most 7 bits (r >= 1 in
// putBit, the &7 in emitPage and emitPair), so a page codes to at most
// partPage bytes. Pages charge it twice: partition and output copy.
const partPage = tokPage * 7 / 8

// parts0 bounds partition 0 and its output copy: at most 119 decisions
// per MB at 7 bits; 1<<16 covers the frame header, chunk headers and
// size classes.
func parts0(mbs int) int64 { return 2*105*int64(mbs) + 1<<16 }

// MaxCharge is the most an opaque w x h encode passes to Page: planes,
// a generic source's alpha and both strips, parts0, and every token
// page plus the spare, each holding at least tokPage-maxBlockTok+1 tokens.
func MaxCharge(w, h int) int64 {
	mbs := int64((w+15)/16) * int64((h+15)/16)
	base := 384*mbs + int64(w)*int64(h) + 128*int64(w)
	pages := 25*maxBlockTok*mbs/(tokPage-maxBlockTok+1) + 2
	return base + parts0(int(mbs)) + pages*(pageBytes+2*partPage)
}

// MaxOutput bounds an opaque w x h file: every token at 7 bits, plus
// partition 0 and headers, half of parts0.
func MaxOutput(w, h int) int64 {
	mbs := int64((w+15)/16) * int64((h+15)/16)
	return 7*25*maxBlockTok*mbs/8 + parts0(int(mbs))/2
}

const (
	fixedTok = 1 << 14
	tokPage  = 32768
)

func (t *tokenBuf) reset() {
	t.n, t.cur = 0, nil
	if len(t.pages) > 0 {
		t.cur = t.pages[0][:0]
	}
}

func (t *tokenBuf) add(bit bool, v int) bool {
	if bit {
		v |= 1 << 15
	}
	if len(t.cur) == cap(t.cur) {
		t.next()
	}
	t.cur = append(t.cur, uint16(v))
	return bit
}

// put appends one (bit, prob) token to a local slice.
func put(c []uint16, bit bool, v int) []uint16 {
	if bit {
		v |= 1 << 15
	}
	return append(c, uint16(v))
}

func (t *tokenBuf) next() {
	if cap(t.cur) > 0 {
		t.pages[t.n] = t.cur
		t.n++
	}
	if t.n == len(t.pages) {
		t.pages = append(t.pages, make([]uint16, 0, tokPage))
	}
	t.cur = t.pages[t.n][:0]
}

// refill adds the next page before a macroblock, which fits in one.
func (t *tokenBuf) refill() bool {
	p, ok := t.lim.Page(t.charged(0))
	if !ok {
		return false
	}
	if p == nil {
		p = make([]uint16, 0, tokPage)
	}
	t.pages = append(t.pages, p)
	return true
}

// charged is what holding one more page and extra bytes charges.
func (t *tokenBuf) charged(extra int64) int64 {
	return t.base + t.parts + extra + int64(len(t.pages)+1)*(pageBytes+2*partPage)
}

// more charges extra bytes without needing a page; a page Page hands
// out anyway goes back with the rest.
func (t *tokenBuf) more(extra int64) bool {
	if t.lim == nil {
		return true
	}
	p, ok := t.lim.Page(t.charged(extra))
	if p != nil {
		t.pages = append(t.pages, p)
	}
	return ok
}

// page returns page i of n+1, the last partial.
func (t *tokenBuf) page(i int) []uint16 {
	if i == t.n {
		return t.cur
	}
	return t.pages[i]
}

func tokProb(v uint16, p *[1056]uint8) uint8 {
	if v&fixedTok != 0 {
		return uint8(v)
	}
	return p[v&0x7ff]
}

func (t *tokenBuf) cost(p *[1056]uint8) int {
	c := 0
	for i := 0; i <= t.n; i++ {
		for _, v := range t.page(i) {
			c += bitCost[v>>15][tokProb(v, p)]
		}
	}
	return c
}

// tokPos is a page and offset in a tokenBuf.
type tokPos struct{ pg, off int32 }

func (t *tokenBuf) pos() tokPos { return tokPos{int32(t.n), int32(len(t.cur))} }

// cursor walks MB rows r, r+2, ... of one partition.
type cursor struct {
	t       *tokenBuf
	rows    []tokPos
	r       int
	pg, off int
}

// next returns the next non-empty run, or nil.
func (c *cursor) next() []uint16 {
	for c.r+1 < len(c.rows) {
		end := c.rows[c.r+1]
		p := c.t.page(c.pg)
		last := c.pg == int(end.pg)
		if last {
			p = p[:end.off]
		}
		s := p[c.off:]
		if !last {
			c.pg, c.off = c.pg+1, 0
		} else if c.r += 2; c.r < len(c.rows) {
			c.pg, c.off = int(c.rows[c.r].pg), int(c.rows[c.r].off)
		}
		if len(s) > 0 {
			return s
		}
	}
	return nil
}

// emit2 codes even MB rows into a, odd into b.
func (t *tokenBuf) emit2(a, b *boolEncoder, rows []tokPos, p *[1056]uint8) {
	ca := cursor{t, rows, 0, int(rows[0].pg), int(rows[0].off)}
	cb := cursor{t, rows, 1, int(rows[1].pg), int(rows[1].off)}
	sa, sb := ca.next(), cb.next()
	for len(sa) > 0 && len(sb) > 0 {
		k := min(len(sa), len(sb))
		emitPair(a, b, sa[:k], sb[:k], p)
		sa, sb = sa[k:], sb[k:]
		if len(sa) == 0 {
			sa = ca.next()
		}
		if len(sb) == 0 {
			sb = cb.next()
		}
	}
	for ; len(sa) > 0; sa = ca.next() {
		emitPage(a, sa, p)
	}
	for ; len(sb) > 0; sb = cb.next() {
		emitPage(b, sb, p)
	}
}

// emitPair is emitPage on two streams; the chains overlap.
func emitPair(a, b *boolEncoder, ta, tb []uint16, p *[1056]uint8) {
	na, va, nba := uint64(a.rng+1)<<24, a.value, a.nbBits
	nb_, vb, nbb := uint64(b.rng+1)<<24, b.value, b.nbBits
	tb = tb[:len(ta)]
	for i, v := range ta {
		pr := uint64(p[v&0x7ff])
		if v&fixedTok != 0 {
			pr = uint64(v & 0xff)
		}
		q := 256 - pr
		m, d := pr, q<<24
		if v&0x8000 != 0 {
			m, d = q, (pr-1)<<24
		}
		va += int64((na*pr+q<<24)>>32) & -int64(v>>15)
		x := na*m + d
		lz := bits.LeadingZeros64(x)
		na = x >> 32 << (lz & 63)
		sh := int32(lz-24) & 7
		va <<= sh
		nba += sh

		w := tb[i]
		pw := uint64(p[w&0x7ff])
		if w&fixedTok != 0 {
			pw = uint64(w & 0xff)
		}
		qw := 256 - pw
		mw, dw := pw, qw<<24
		if w&0x8000 != 0 {
			mw, dw = qw, (pw-1)<<24
		}
		vb += int64((nb_*pw+qw<<24)>>32) & -int64(w>>15)
		y := nb_*mw + dw
		lzw := bits.LeadingZeros64(y)
		nb_ = y >> 32 << (lzw & 63)
		shw := int32(lzw-24) & 7
		vb <<= shw
		nbb += shw

		if nba > 24 {
			a.value, a.nbBits = va, nba
			for a.nbBits > 0 {
				a.flush()
			}
			va, nba = a.value, a.nbBits
		}
		if nbb > 24 {
			b.value, b.nbBits = vb, nbb
			for b.nbBits > 0 {
				b.flush()
			}
			vb, nbb = b.value, b.nbBits
		}
	}
	a.rng, a.value, a.nbBits = int32(na>>24)-1, va, nba
	b.rng, b.value, b.nbBits = int32(nb_>>24)-1, vb, nbb
	for a.nbBits > 0 {
		a.flush()
	}
	for b.nbBits > 0 {
		b.flush()
	}
}

// putBit inlined; coder state stays in registers.
// n is range<<clz(range), shortening the dependency chain.
func emitPage(e *boolEncoder, t []uint16, p *[1056]uint8) {
	n, value, nb := uint64(e.rng+1)<<24, e.value, e.nbBits
	for _, v := range t {
		pr := uint64(p[v&0x7ff])
		if v&fixedTok != 0 {
			pr = uint64(v & 0xff)
		}
		q := 256 - pr
		a, b := pr, q<<24 // split+1 in the top half
		if v&0x8000 != 0 {
			a, b = q, (pr-1)<<24 // range-split in the top half
		}
		value += int64((n*pr+q<<24)>>32) & -int64(v>>15)
		x := n*a + b
		lz := bits.LeadingZeros64(x)
		n = x >> 32 << (lz & 63) // mask drops Go's shift check
		sh := int32(lz-24) & 7
		value <<= sh
		nb += sh
		if nb > 24 { // 4 bytes per flush
			// No carry, no 0xff, no pending run: flush appends them as is.
			s := uint(nb - 16)
			w := uint64(value) >> s
			if x := ^uint32(w); w>>32 == 0 && e.run == 0 && (x-0x01010101)&^x&0x80808080 == 0 {
				e.buf = binary.BigEndian.AppendUint32(e.buf, uint32(w))
				value &= 1<<s - 1
				nb -= 32
				continue
			}
			e.value, e.nbBits = value, nb
			for e.nbBits > 0 {
				e.flush()
			}
			value, nb = e.value, e.nbBits
		}
	}
	e.rng, e.value, e.nbBits = int32(n>>24)-1, value, nb
	for e.nbBits > 0 {
		e.flush()
	}
}

func flatten(p *[4][8][3][11]uint8) (f [1056]uint8) {
	n := 0
	for i := range p {
		for j := range p[i] {
			for k := range p[i][j] {
				n += copy(f[n:], p[i][j][k][:])
			}
		}
	}
	return f
}

var defaultProbs, updateProbs = flatten(&coeffProbs), flatten(&coeffUpdateProbs)

// libwebp FinalizeTokenProbas (frame_enc.c:136-181), BSD-3-Clause.
// Key frames reset to the defaults, so old is always the default.
func finalizeProbs(t *tokenBuf, p *[1056]uint8) {
	var cnt [2048]uint64 // nb<<32 | total
	for i := 0; i <= t.n; i++ {
		for _, v := range t.page(i) {
			j := (v | -(v >> 14 & 1)) & 0x7ff // fixed tokens to 2047
			cnt[j] += 1 | uint64(v>>15)<<32
		}
	}
	*p = defaultProbs
	for i := range p {
		total, nb := int(uint32(cnt[i])), int(cnt[i]>>32)
		if nb == 0 {
			continue
		}
		np := 255 - nb*255/total
		bc := func(q uint8) int { return nb*bitCost[1][q] + (total-nb)*bitCost[0][q] }
		old := bc(defaultProbs[i]) + bitCost[0][updateProbs[i]]
		new := bc(uint8(np)) + bitCost[1][updateProbs[i]] + 8*256
		if old > new {
			p[i] = uint8(np)
		}
	}
}

// Ported from libwebp PutCoeffs (frame_enc.c), BSD-3-Clause.
func recordCoeffs(t *tokenBuf, typ, ctx, first int, lv *[16]int16) uint8 {
	if cap(t.cur)-len(t.cur) < maxBlockTok {
		t.next()
	}
	var nz uint8
	t.cur, nz = putCoeffs(t.cur, typ, ctx, first, lv)
	return nz
}

// putCoeffs writes zero and one levels by index without branches:
// nz and the gt1 slot are set for every n, the gt1 slot speculative
// for zeros, and the cursor only advances past what's real. Only
// |v|>=2 (recordLevelTail) still branches.
func putCoeffs(c []uint16, typ, ctx, first int, lv *[16]int16) ([]uint16, uint8) {
	last := lastNZ(lv, first)
	base := typ * 264
	p := base + (int(bands[first])*3+ctx)*11
	c = put(c, last >= 0, p)
	if last < 0 {
		return c, 0
	}
	i := len(c)
	c = c[:cap(c)]
	for n := first; n <= last; n++ {
		v := int(lv[n])
		s := v >> 63
		a := (v ^ s) - s
		nz := min(a, 1)
		g := uint16(min(a, 2) >> 1)
		c[i] = uint16(nz)<<15 | uint16(p+1)
		c[i+1] = g<<15 | uint16(p+2)
		i += 1 + nz
		if a > 1 {
			i = recordLevelTail(c, i, a, p)
		}
		pn := base + int(bands[n+1])*33 + 11*min(a, 2)
		c[i] = uint16(s&1)<<15 | fixedTok | 128
		c[i+1] = uint16(b2i(n < last))<<15 | uint16(pn)
		i += nz * (1 + b2i(n < 15))
		p = pn
	}
	return c[:i], 1
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// recordLevel codes |v| >= 1 from the v>1 node; used only by init
// cost tables in rdo.go. The putCoeffs hot path uses recordLevelTail.
func recordLevel(c []uint16, v, p int) ([]uint16, bool) {
	gt1 := v > 1
	c = put(c, gt1, p+2)
	if !gt1 {
		return c, false
	}
	i := len(c)
	c = append(c, make([]uint16, 16)...)
	i = recordLevelTail(c, i, v, p)
	return c[:i], true
}

// recordLevelTail codes |v| > 1 from the gt4 node, writing at c[i:].
func recordLevelTail(c []uint16, i, v, p int) int {
	gt4 := v > 4
	c[i] = uint16(b2i(gt4))<<15 | uint16(p+3)
	i++
	if !gt4 {
		ne2 := v != 2
		c[i] = uint16(b2i(ne2))<<15 | uint16(p+4)
		i++
		if ne2 {
			c[i] = uint16(b2i(v == 4))<<15 | uint16(p+5)
			i++
		}
		return i
	}
	gt10 := v > 10
	c[i] = uint16(b2i(gt10))<<15 | uint16(p+6)
	i++
	if !gt10 {
		gt6 := v > 6
		c[i] = uint16(b2i(gt6))<<15 | uint16(p+7)
		i++
		if !gt6 {
			c[i] = uint16(b2i(v == 6))<<15 | fixedTok | 159
			i++
		} else {
			c[i] = uint16(b2i(v >= 9))<<15 | fixedTok | 165
			i++
			c[i] = uint16(b2i(v&1 == 0))<<15 | fixedTok | 145
			i++
		}
		return i
	}
	k := 3
	switch {
	case v < 19:
		k = 0
	case v < 35:
		k = 1
	case v < 67:
		k = 2
	}
	c[i] = uint16(b2i(k >= 2))<<15 | uint16(p+8)
	i++
	c[i] = uint16(b2i(k&1 == 1))<<15 | uint16(p+9+k>>1)
	i++
	v -= 3 + 8<<k
	nb := len(catProbs[k])
	for j, pb := range catProbs[k] {
		c[i] = uint16(b2i(v>>(nb-1-j)&1 == 1))<<15 | fixedTok | uint16(pb)
		i++
	}
	return i
}

// Ported from libwebp CodeResiduals (frame_enc.c), BSD-3-Clause.
func recordLuma(t *tokenBuf, i16 bool, lv *[25][16]int16, l, tp *nzCtx) {
	typ, first := 3, 0
	if i16 {
		nz := recordCoeffs(t, 1, int(l[8]+tp[8]), 0, &lv[24])
		l[8], tp[8] = nz, nz
		typ, first = 0, 1
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			nz := recordCoeffs(t, typ, int(l[y]+tp[x]), first, &lv[4*y+x])
			l[y], tp[x] = nz, nz
		}
	}
}

func recordChroma(t *tokenBuf, lv *[25][16]int16, l, tp *nzCtx) {
	for c := 0; c < 4; c += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				nz := recordCoeffs(t, 2, int(l[4+c+y]+tp[4+c+x]), 0, &lv[16+2*c+2*y+x])
				l[4+c+y], tp[4+c+x] = nz, nz
			}
		}
	}
}

// costCoeffs is libwebp GetResidualCost_C (dsp/cost.c) with default
// probs; it equals recordCoeffs followed by cost(&defaultProbs).
func costCoeffs(typ, ctx, first int, lv *[16]int16) (int, uint8) {
	return costLast(typ, ctx, first, lastNZ(lv, first), lv)
}

// lastNZ is a branch-free SWAR scan.
func lastNZ(lv *[16]int16, first int) int {
	m := nz4(lv, 0) | nz4(lv, 4)<<4 | nz4(lv, 8)<<8 | nz4(lv, 12)<<12
	return bits.Len16(m>>first<<first) - 1
}

// nz4 maps lv[i:i+4] nonzero to bits 0-3.
func nz4(lv *[16]int16, i int) uint16 {
	w := uint64(uint16(lv[i])) | uint64(uint16(lv[i+1]))<<16 | uint64(uint16(lv[i+2]))<<32 | uint64(uint16(lv[i+3]))<<48
	// Lane bit 15 set iff nonzero.
	h := ((w & 0x7fff7fff7fff7fff) + 0x7fff7fff7fff7fff | w) & 0x8000800080008000
	// Lanes land on bits 60-63.
	return uint16(h * 0x0000200040008001 >> 60)
}

// costLast is costCoeffs with last already known.
func costLast(typ, ctx, first, last int, lv *[16]int16) (int, uint8) {
	base := typ * 264
	p0 := defaultProbs[base+(int(bands[first])*3+ctx)*11]
	if last < 0 {
		return bitCost[0][p0], 0
	}
	c := 0
	if ctx == 0 { // tables carry the EOB bit only for ctx > 0
		c = bitCost[1][p0]
	}
	pc := &defTabs.cost[typ&3]
	v := 0
	for n := first; n <= last; n++ {
		// Branch-free: signs and sizes are random.
		v = int(lv[n&15])
		s := v >> 63
		v = (v ^ s) - s
		u := uint(v)
		if u > 67 { // rare
			c += fixedCost[u&2047] - fixedCost[67]
			u = 67
		}
		c += int(pc[n&15][ctx][u])
		k := v - 2
		ctx = 2 + k&(k>>63)
	}
	if last < 15 {
		c += int(defTabs.eob[typ&3][last&15][min(v, 2)&3])
	}
	return c, 1
}

// costLuma mirrors recordLuma.
func costLuma(i16 bool, lv *[25][16]int16, l, tp *nzCtx) int {
	typ, first, r := 3, 0, 0
	if i16 {
		c, nz := costCoeffs(1, int(l[8]+tp[8]), 0, &lv[24])
		l[8], tp[8] = nz, nz
		typ, first, r = 0, 1, c
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			c, nz := costCoeffs(typ, int(l[y]+tp[x]), first, &lv[4*y+x])
			l[y], tp[x] = nz, nz
			r += c
		}
	}
	return r
}

// costChroma mirrors recordChroma.
func costChroma(lv *[25][16]int16, l, tp *nzCtx) int {
	r := 0
	for c := 0; c < 4; c += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				k, nz := costCoeffs(2, int(l[4+c+y]+tp[4+c+x]), 0, &lv[16+2*c+2*y+x])
				l[4+c+y], tp[4+c+x] = nz, nz
				r += k
			}
		}
	}
	return r
}
