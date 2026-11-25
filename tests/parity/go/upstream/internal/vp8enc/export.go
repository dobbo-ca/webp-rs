package vp8enc

// Tables the decoder (internal/vp8dec) shares with the encoder.
var (
	DCQuant          = dcTable          // RFC 6386 14.1 dc_qlookup
	ACQuant          = acTable          // ac_qlookup
	CoeffProbs       = coeffProbs       // default token probabilities, 13.5
	CoeffUpdateProbs = coeffUpdateProbs // 13.4
	BModeProbs       = predProb         // key frame sub-block modes by [above][left], 11.5
	Bands            = bands            // 13.3
	CatProbs         = catProbs         // extra bits of tokens cat3 to cat6, 13.2
	Zigzag           = zigzag           // 13.3
)
