package vp8enc

import "encoding/binary"

// Key frame tag, RFC 6386 9.1.
func frameTag(width, height, size0 int) [10]byte {
	v := uint32(size0)<<5 | 1<<4
	var t [10]byte
	t[0] = byte(v)
	t[1] = byte(v >> 8)
	t[2] = byte(v >> 16)
	t[3] = 0x9d
	t[4] = 0x01
	t[5] = 0x2a
	binary.LittleEndian.PutUint16(t[6:8], uint16(width))
	binary.LittleEndian.PutUint16(t[8:10], uint16(height))
	return t
}

// Compressed frame header, RFC 6386 19.2.
// lw writes one DCT partition and no skip flags, as libwebp's token loop.
func putFrameHeader(e *boolEncoder, s *segs, skipProb uint8, probs *[1056]uint8, lw bool) {
	e.putLiteral(0, 1) // color_space
	e.putLiteral(0, 1) // clamping_type

	// PutSegmentHeader, syntax_enc.c:185-212.
	if e.putBit(s.n > 1, 128) {
		e.putBit(s.updateMap, 128)
		e.putLiteral(1, 1) // update_segment_feature_data
		e.putLiteral(1, 1) // absolute values, x/image decode.go:220
		for i := range s.quant {
			putSigned(e, s.quant[i], 7)
		}
		for i := range s.level {
			putSigned(e, s.level[i], 6)
		}
		if s.updateMap {
			for _, p := range s.probs {
				if e.putBit(p != 255, 128) {
					e.putLiteral(uint32(p), 8)
				}
			}
		}
	}

	// Decoders skip filtering when the frame level is 0.
	level := max(s.level[0], s.level[1], s.level[2], s.level[3])
	e.putLiteral(0, 1)             // filter_type
	e.putLiteral(uint32(level), 6) // loop_filter_level
	e.putLiteral(0, 3)             // sharpness_level

	e.putLiteral(0, 1) // loop_filter_adj_enable

	if lw {
		e.putLiteral(0, 2) // log2_nbr_of_dct_partitions
	} else {
		e.putLiteral(1, 2)
	}

	e.putLiteral(uint32(s.quant[0]), 7) // y_ac_qi
	putSigned(e, 0, 4)                  // y_dc
	putSigned(e, 0, 4)                  // y2_dc
	putSigned(e, 0, 4)                  // y2_ac
	putSigned(e, dqUVDC, 4)
	putSigned(e, s.dqUVAC, 4)

	e.putLiteral(0, 1) // refresh_entropy_probs

	// VP8WriteProbas (tree_enc.c:417-432): update iff changed.
	for i := range probs {
		if e.putBit(probs[i] != defaultProbs[i], updateProbs[i]) {
			e.putLiteral(uint32(probs[i]), 8)
		}
	}

	if lw {
		e.putLiteral(0, 1) // mb_no_skip_coeff
		return
	}
	e.putLiteral(1, 1)                // mb_no_skip_coeff
	e.putLiteral(uint32(skipProb), 8) // prob_skip_false
}

// VP8PutSignedBits, bit_writer_utils.c:151-158.
func putSigned(e *boolEncoder, v, n int) {
	if !e.putBit(v != 0, 128) {
		return
	}
	neg := v < 0
	if neg {
		v = -v
	}
	e.putLiteral(uint32(v), n)
	e.putBit(neg, 128)
}

// From x/image/vp8 predProb (BSD-3), equals libwebp kBModesProba.
var predProb = [10][10][9]uint8{
	{
		{231, 120, 48, 89, 115, 113, 120, 152, 112},
		{152, 179, 64, 126, 170, 118, 46, 70, 95},
		{175, 69, 143, 80, 85, 82, 72, 155, 103},
		{56, 58, 10, 171, 218, 189, 17, 13, 152},
		{114, 26, 17, 163, 44, 195, 21, 10, 173},
		{121, 24, 80, 195, 26, 62, 44, 64, 85},
		{144, 71, 10, 38, 171, 213, 144, 34, 26},
		{170, 46, 55, 19, 136, 160, 33, 206, 71},
		{63, 20, 8, 114, 114, 208, 12, 9, 226},
		{81, 40, 11, 96, 182, 84, 29, 16, 36},
	},
	{
		{134, 183, 89, 137, 98, 101, 106, 165, 148},
		{72, 187, 100, 130, 157, 111, 32, 75, 80},
		{66, 102, 167, 99, 74, 62, 40, 234, 128},
		{41, 53, 9, 178, 241, 141, 26, 8, 107},
		{74, 43, 26, 146, 73, 166, 49, 23, 157},
		{65, 38, 105, 160, 51, 52, 31, 115, 128},
		{104, 79, 12, 27, 217, 255, 87, 17, 7},
		{87, 68, 71, 44, 114, 51, 15, 186, 23},
		{47, 41, 14, 110, 182, 183, 21, 17, 194},
		{66, 45, 25, 102, 197, 189, 23, 18, 22},
	},
	{
		{88, 88, 147, 150, 42, 46, 45, 196, 205},
		{43, 97, 183, 117, 85, 38, 35, 179, 61},
		{39, 53, 200, 87, 26, 21, 43, 232, 171},
		{56, 34, 51, 104, 114, 102, 29, 93, 77},
		{39, 28, 85, 171, 58, 165, 90, 98, 64},
		{34, 22, 116, 206, 23, 34, 43, 166, 73},
		{107, 54, 32, 26, 51, 1, 81, 43, 31},
		{68, 25, 106, 22, 64, 171, 36, 225, 114},
		{34, 19, 21, 102, 132, 188, 16, 76, 124},
		{62, 18, 78, 95, 85, 57, 50, 48, 51},
	},
	{
		{193, 101, 35, 159, 215, 111, 89, 46, 111},
		{60, 148, 31, 172, 219, 228, 21, 18, 111},
		{112, 113, 77, 85, 179, 255, 38, 120, 114},
		{40, 42, 1, 196, 245, 209, 10, 25, 109},
		{88, 43, 29, 140, 166, 213, 37, 43, 154},
		{61, 63, 30, 155, 67, 45, 68, 1, 209},
		{100, 80, 8, 43, 154, 1, 51, 26, 71},
		{142, 78, 78, 16, 255, 128, 34, 197, 171},
		{41, 40, 5, 102, 211, 183, 4, 1, 221},
		{51, 50, 17, 168, 209, 192, 23, 25, 82},
	},
	{
		{138, 31, 36, 171, 27, 166, 38, 44, 229},
		{67, 87, 58, 169, 82, 115, 26, 59, 179},
		{63, 59, 90, 180, 59, 166, 93, 73, 154},
		{40, 40, 21, 116, 143, 209, 34, 39, 175},
		{47, 15, 16, 183, 34, 223, 49, 45, 183},
		{46, 17, 33, 183, 6, 98, 15, 32, 183},
		{57, 46, 22, 24, 128, 1, 54, 17, 37},
		{65, 32, 73, 115, 28, 128, 23, 128, 205},
		{40, 3, 9, 115, 51, 192, 18, 6, 223},
		{87, 37, 9, 115, 59, 77, 64, 21, 47},
	},
	{
		{104, 55, 44, 218, 9, 54, 53, 130, 226},
		{64, 90, 70, 205, 40, 41, 23, 26, 57},
		{54, 57, 112, 184, 5, 41, 38, 166, 213},
		{30, 34, 26, 133, 152, 116, 10, 32, 134},
		{39, 19, 53, 221, 26, 114, 32, 73, 255},
		{31, 9, 65, 234, 2, 15, 1, 118, 73},
		{75, 32, 12, 51, 192, 255, 160, 43, 51},
		{88, 31, 35, 67, 102, 85, 55, 186, 85},
		{56, 21, 23, 111, 59, 205, 45, 37, 192},
		{55, 38, 70, 124, 73, 102, 1, 34, 98},
	},
	{
		{125, 98, 42, 88, 104, 85, 117, 175, 82},
		{95, 84, 53, 89, 128, 100, 113, 101, 45},
		{75, 79, 123, 47, 51, 128, 81, 171, 1},
		{57, 17, 5, 71, 102, 57, 53, 41, 49},
		{38, 33, 13, 121, 57, 73, 26, 1, 85},
		{41, 10, 67, 138, 77, 110, 90, 47, 114},
		{115, 21, 2, 10, 102, 255, 166, 23, 6},
		{101, 29, 16, 10, 85, 128, 101, 196, 26},
		{57, 18, 10, 102, 102, 213, 34, 20, 43},
		{117, 20, 15, 36, 163, 128, 68, 1, 26},
	},
	{
		{102, 61, 71, 37, 34, 53, 31, 243, 192},
		{69, 60, 71, 38, 73, 119, 28, 222, 37},
		{68, 45, 128, 34, 1, 47, 11, 245, 171},
		{62, 17, 19, 70, 146, 85, 55, 62, 70},
		{37, 43, 37, 154, 100, 163, 85, 160, 1},
		{63, 9, 92, 136, 28, 64, 32, 201, 85},
		{75, 15, 9, 9, 64, 255, 184, 119, 16},
		{86, 6, 28, 5, 64, 255, 25, 248, 1},
		{56, 8, 17, 132, 137, 255, 55, 116, 128},
		{58, 15, 20, 82, 135, 57, 26, 121, 40},
	},
	{
		{164, 50, 31, 137, 154, 133, 25, 35, 218},
		{51, 103, 44, 131, 131, 123, 31, 6, 158},
		{86, 40, 64, 135, 148, 224, 45, 183, 128},
		{22, 26, 17, 131, 240, 154, 14, 1, 209},
		{45, 16, 21, 91, 64, 222, 7, 1, 197},
		{56, 21, 39, 155, 60, 138, 23, 102, 213},
		{83, 12, 13, 54, 192, 255, 68, 47, 28},
		{85, 26, 85, 85, 128, 128, 32, 146, 171},
		{18, 11, 7, 63, 144, 171, 4, 4, 246},
		{35, 27, 10, 146, 174, 171, 12, 26, 128},
	},
	{
		{190, 80, 35, 99, 180, 80, 126, 54, 45},
		{85, 126, 47, 87, 176, 51, 41, 20, 32},
		{101, 75, 128, 139, 118, 146, 116, 128, 85},
		{56, 41, 15, 176, 236, 85, 37, 9, 62},
		{71, 30, 17, 119, 118, 255, 17, 18, 138},
		{101, 38, 60, 138, 55, 70, 43, 26, 142},
		{146, 36, 19, 30, 171, 255, 97, 27, 20},
		{138, 45, 61, 62, 219, 1, 81, 188, 64},
		{32, 41, 20, 117, 151, 142, 20, 21, 163},
		{112, 19, 12, 61, 195, 128, 48, 4, 24},
	},
}

// Mode writers mirror libwebp tree_enc.c Put*Mode, BSD-3-Clause.
func putI16Mode(put func(bool, uint8) bool, m uint8) {
	if put(m == predTM || m == predHE, 156) {
		put(m == predTM, 128)
	} else {
		put(m == predVE, 163)
	}
}

func putUVMode(put func(bool, uint8) bool, m uint8) {
	if put(m != predDC, 142) && put(m != predVE, 114) {
		put(m != predHE, 183)
	}
}

func putI4Mode(put func(bool, uint8) bool, m uint8, p *[9]uint8) {
	if !put(m != predDC, p[0]) || !put(m != predTM, p[1]) || !put(m != predVE, p[2]) {
		return
	}
	if !put(m >= predLD, p[3]) {
		if put(m != predHE, p[4]) {
			put(m != predRD, p[5])
		}
	} else if put(m != predLD, p[6]) && put(m != predVL, p[7]) {
		put(m != predHD, p[8])
	}
}
