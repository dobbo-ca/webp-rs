// Shared types for the libwebp 1.2.2 ALPH port (enc/vp8li_enc.h, enc/backward_references_enc.h, enc/histogram_enc.h, utils/huffman_encode_utils.h, dsp/lossless.h). Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

// Constants shared across lw*.go. File-local ones live with their users.
const (
	lwMaxLength     = 4095 // MAX_LENGTH, backward_references_enc.h
	lwMaxLengthBits = 12   // MAX_LENGTH_BITS
	lwHashBits      = 18   // HASH_BITS
	lwHashSize      = 1 << lwHashBits
	lwWindowSize    = 1<<20 - 120 // WINDOW_SIZE, backward_references_enc.c:30

	lwNumLiteral   = 256                        // NUM_LITERAL_CODES
	lwNumLength    = 24                         // NUM_LENGTH_CODES
	lwNumDistance  = 40                         // NUM_DISTANCE_CODES
	lwNumCodes     = lwNumLiteral + lwNumLength // VP8LHistogramNumCodes(0)
	lwCodeLenCodes = 19                         // CODE_LENGTH_CODES
	lwMaxPalette   = 256                        // MAX_PALETTE_SIZE
	lwMinHuffBits  = 2                          // MIN_HUFFMAN_BITS
	lwMaxHuffBits  = 9                          // MAX_HUFFMAN_BITS
	lwMaxHuffImage = 2600                       // MAX_HUFF_IMAGE_SIZE
	lwMaxTransBits = 6                          // MAX_TRANSFORM_BITS
	lwNumPredModes = 14                         // kNumPredModes

	lwARGBBlack     = 0xff000000 // ARGB_BLACK
	lwNonTrivialSym = 0xffffffff // VP8L_NON_TRIVIAL_SYM

	lwCostIntervalMax = 500 // COST_CACHE_INTERVAL_SIZE_MAX
	lwHistoTotal      = 13  // kHistoTotal, vp8l_enc.c:357
)

// EntropyIx, vp8l_enc.c:326.
const (
	kDirect = iota
	kSpatial
	kSubGreen
	kSpatialSubGreen
	kPalette
)

// VP8LLZ77Type, backward_references_enc.h:214.
const (
	lz77Standard = 1
	lz77RLE      = 2
	lz77Box      = 4
)

// WEBP_FILTER_TYPE, filters_utils.h.
const (
	filterNone = iota
	filterHorizontal
	filterVertical
	filterGradient
)

// pixOrCopy modes; kCacheIdx is unreachable with use_cache=0.
const (
	lwLiteral = 0
	lwCopy    = 2
)

// lwConfig is the sub-encode WebPConfig, alpha_enc.c:65-89.
type lwConfig struct {
	method  int // config->method
	quality int // (int)config->quality, 8*method
	effort  int // alpha effort_level, alpha_enc.c:369; equals method
}

// pixOrCopy is PixOrCopy: a literal argb or a copy (len, distance).
type pixOrCopy struct {
	mode       uint8
	len        uint16
	argbOrDist uint32 // argb, raw distance, or plane code after 2D locality
}

// lwRefs is VP8LBackwardRefs as one flat slice, same iteration order.
type lwRefs []pixOrCopy

// lwHashChain is VP8LHashChain: offset<<12 | length per pixel.
// hashChainFill uses it as an int32 chain first, cast through uint32.
type lwHashChain struct {
	offLen []uint32
}

// lwHisto is VP8LHistogram with cache_bits 0 (palette_code_bits_ 0).
type lwHisto struct {
	literal       [lwNumCodes]uint32 // green, then length prefixes
	red           [lwNumLiteral]uint32
	blue          [lwNumLiteral]uint32
	alpha         [lwNumLiteral]uint32
	distance      [lwNumDistance]uint32
	trivialSymbol uint32
	bitCost       float64
	literalCost   float64
	redCost       float64
	blueCost      float64
	isUsed        [5]bool // literal, red, blue, alpha, distance
}

// lwHistoSet is VP8LHistogramSet. h[i] may be nil once removed;
// len(h) is size. Pointers may be swapped with a scratch lwHisto.
type lwHistoSet struct {
	h    []*lwHisto
	max  int       // max_size
	pool []lwHisto // backing for h, reused across calls
}

// bitEntropy is VP8LBitEntropy, dsp/lossless.h:200.
type bitEntropy struct {
	entropy     float64
	sum         uint32
	nonzeros    int
	maxVal      uint32
	nonzeroCode uint32
}

// streaks is VP8LStreaks, dsp/lossless.h:195.
type streaks struct {
	counts  [2]int    // zero streak, non-zero streak
	streaks [2][2]int // [zero/non-zero][streak<=3 / streak>3]
}

// dominantCostRange is DominantCostRange, histogram_enc.c:478.
type dominantCostRange struct {
	literalMax, literalMin float64
	redMax, redMin         float64
	blueMax, blueMin       float64
}

// lwHistoPair is HistogramPair; idx1 < 0 means out of date.
type lwHistoPair struct {
	idx1, idx2 int
	costDiff   float64
	costCombo  float64
}

// lwHistoQueue is HistoQueue; q has len max.
type lwHistoQueue struct {
	q    []lwHistoPair
	size int
	max  int
}

// lwCostModel is CostModel, backward_references_cost_enc.c:32.
type lwCostModel struct {
	alpha    [256]float64
	red      [256]float64
	blue     [256]float64
	distance [lwNumDistance]float64
	literal  [lwNumCodes]float64
}

// lwCostInterval is CostInterval: cost for [start, end) from index.
type lwCostInterval struct {
	cost       float32
	start, end int
	index      int
	prev, next *lwCostInterval
}

// lwCostCacheInterval is CostCacheInterval; end is exclusive.
type lwCostCacheInterval struct {
	cost       float64
	start, end int
}

// lwCostManager is CostManager. ivs is the interval store; the live
// list never exceeds lwCostIntervalMax.
type lwCostManager struct {
	head           *lwCostInterval
	count          int
	cacheIntervals []lwCostCacheInterval
	costCache      [lwMaxLength]float64 // GetLengthCost(k)
	costs          []float32
	distArray      []uint16
	ivs            [lwCostIntervalMax]lwCostInterval
	free           *lwCostInterval
}

// huffTreeToken is HuffmanTreeToken.
type huffTreeToken struct {
	code      uint8 // 0..15 or escape 16, 17, 18
	extraBits uint8
}

// huffTreeCode is HuffmanTreeCode; slices point into lwHuffCodes backing.
type huffTreeCode struct {
	numSymbols  int
	codeLengths []uint8
	codes       []uint16
}

// huffTree is HuffmanTree, a node of the tree-building pool.
type huffTree struct {
	totalCount uint32
	value      int
	left       int // pool_index_left_
	right      int // pool_index_right_
}

// lwHuffCodes holds 5 codes per histogram and their backing.
type lwHuffCodes struct {
	codes []huffTreeCode
	lens  []uint8
	bits  []uint16
}

// crunchSubConfig is CrunchSubConfig; do_no_cache_ is always 0 here.
type crunchSubConfig struct {
	lz77 int
}

// crunchConfig is CrunchConfig; sorting is kMinimizeDelta when palette.
type crunchConfig struct {
	entropyIdx int
	sub        [2]crunchSubConfig
	subSize    int
}

// LW is the libwebp 1.2.2 or 1.6.0 ALPH encoder. It keeps scratch between calls.
type LW struct {
	// Libwebp16 writes ALPH as libwebp 1.6.0 does (lw16*.go).
	Libwebp16 bool

	cfg lwConfig
	v16 *lw16State // made on first 1.6.0 encode

	// alpha_enc.c
	filt  []uint8 // tmp_alpha, filtered plane
	trial []byte  // second filter trial, header + payload

	// VP8LEncoder, vp8li_enc.h
	pic           []uint32 // picture argb, alpha<<8
	argb          []uint32 // argb_, transformed image
	argbScratch   []uint32 // argb_scratch_, predictor rows
	transformData []uint32 // transform_data_, predictor modes
	currentWidth  int
	histoBits     int
	transformBits int
	usePredict    bool
	usePalette    bool
	paletteSize   int
	palette       [lwMaxPalette]uint32
	paletteSorted [lwMaxPalette]uint32
	refs          [4]lwRefs
	hc            lwHashChain // hash_chain_

	bw     bitWriter // stream writer
	bwBest bitWriter // EncodeImageInternal best sub-config

	// backward_references_enc.c
	hcBox     lwHashChain
	hcHisto   lwHashChain // hash_chain_histogram
	hashFirst []int32     // hash_to_first_index, lwHashSize
	boxCounts []uint16    // counts_ini
	estHisto  lwHisto     // GetBackwardReferences histo

	// backward_references_cost_enc.c
	costModel lwCostModel
	costHisto lwHisto
	costMgr   lwCostManager
	distArray []uint16

	// histogram_enc.c
	histoImage   lwHistoSet
	histoOrig    lwHistoSet
	histoNH      lwHistoSet // EncodeImageNoHuffman, size 1
	tmpHisto     lwHisto
	histoSymbols []uint16
	histoArgb    []uint32
	mapTmp       []uint16
	mappings     []int
	queue        lwHistoQueue
	aeHisto      [lwHistoTotal * 256]uint32 // AnalyzeEntropy

	// huffman_encode_utils.c and code storage
	huff     lwHuffCodes // main image codes
	huffNH   lwHuffCodes // EncodeImageNoHuffman codes
	bufRle   []uint8
	huffTree []huffTree
	tokens   []huffTreeToken

	palRow []uint8 // ApplyPalette tmp_row
}
