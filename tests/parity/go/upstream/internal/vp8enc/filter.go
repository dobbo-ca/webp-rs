package vp8enc

// libwebp SetupFilterStrength, strength 60, sharpness 0 (quant_enc.c:272-290).
func filterLevel(qi, beta int) int {
	s := min(int(acTable[qi])>>2, 63)
	f := s * 300 / (256 + beta)
	if f < 2 {
		return 0
	}
	return min(f, 63)
}
