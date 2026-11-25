package vp8enc

// libwebp VP8EntropyCost (dsp/cost.c:25-47), BSD-3-Clause.
// Cost of a 0 bit at prob p, in 1/256 bit.
var entropyCost = [256]int{
	1792, 1792, 1792, 1536, 1536, 1408, 1366, 1280, 1280, 1216, 1178, 1152,
	1110, 1076, 1061, 1024, 1024, 992, 968, 951, 939, 911, 896, 878,
	871, 854, 838, 820, 811, 794, 786, 768, 768, 752, 740, 732,
	720, 709, 704, 690, 683, 672, 666, 655, 647, 640, 631, 622,
	615, 607, 598, 592, 586, 576, 572, 564, 559, 555, 547, 541,
	534, 528, 522, 512, 512, 504, 500, 494, 488, 483, 477, 473,
	467, 461, 458, 452, 448, 443, 438, 434, 427, 424, 419, 415,
	410, 406, 403, 399, 394, 390, 384, 384, 377, 374, 370, 366,
	362, 359, 355, 351, 347, 342, 342, 336, 333, 330, 326, 323,
	320, 316, 312, 308, 305, 302, 299, 296, 293, 288, 287, 283,
	280, 277, 274, 272, 268, 266, 262, 256, 256, 256, 251, 248,
	245, 242, 240, 237, 234, 232, 228, 226, 223, 221, 218, 216,
	214, 211, 208, 205, 203, 201, 198, 196, 192, 191, 188, 187,
	183, 181, 179, 176, 175, 171, 171, 168, 165, 163, 160, 159,
	156, 154, 152, 150, 148, 146, 144, 142, 139, 138, 135, 133,
	131, 128, 128, 125, 123, 121, 119, 117, 115, 113, 111, 110,
	107, 105, 103, 102, 100, 98, 96, 94, 92, 91, 89, 86,
	86, 83, 82, 80, 77, 76, 74, 73, 71, 69, 67, 66,
	64, 63, 61, 59, 57, 55, 54, 52, 51, 49, 47, 46,
	44, 43, 41, 40, 38, 36, 35, 33, 32, 30, 29, 27,
	25, 24, 22, 21, 19, 18, 16, 15, 13, 12, 10, 9,
	7, 6, 4, 3}

// bitCost[bit][p] is VP8BitCost(bit, p).
var bitCost [2][256]int

// Mode costs in 1/256 bit, i16Cost includes the is_i16 bit.
var i16Cost, uvCost [4]int
var i4Cost [10][10][10]int

func init() {
	for p := range entropyCost {
		bitCost[0][p] = entropyCost[p]
		bitCost[1][p] = entropyCost[255-p]
	}
	for m := uint8(0); m < 4; m++ {
		i16Cost[m] = treeCost(func(put func(bool, uint8) bool) {
			put(true, 145)
			putI16Mode(put, m)
		})
		uvCost[m] = treeCost(func(put func(bool, uint8) bool) { putUVMode(put, m) })
	}
	for a := range i4Cost {
		for l := range i4Cost[a] {
			for m := uint8(0); m < 10; m++ {
				i4Cost[a][l][m] = treeCost(func(put func(bool, uint8) bool) { putI4Mode(put, m, &predProb[a][l]) })
			}
		}
	}
}

// treeCost sums bitCost over a mode writer's bits.
func treeCost(f func(put func(bool, uint8) bool)) int {
	c := 0
	f(func(b bool, p uint8) bool {
		if b {
			c += bitCost[1][p]
		} else {
			c += bitCost[0][p]
		}
		return b
	})
	return c
}

// libwebp quant_enc.c:184-212, 239-247.
type lambdas struct{ i4, i16, uv, mode, ti4, ti16 int }

func qa(m *quantMatrix) int { return int(m.q[0]+15*m.q[1]+8) >> 4 }

func newLambdas(q *quant) lambdas {
	y1, y2, uv := qa(&q.y1), qa(&q.y2), qa(&q.uv)
	return lambdas{
		i4:   max(3*y1*y1>>7, 1),
		i16:  max(3*y2*y2, 1),
		uv:   max(3*uv*uv>>6, 1),
		mode: max(y1*y1>>7, 1),
		ti4:  max(7*y1*y1>>3, 1),
		ti16: max(y2*y2>>2, 1),
	}
}

// libwebp VP8LevelFixedCosts and level_cost, cost_enc.c:66-100.
var fixedCost [2048]int
var levelCost [4][8][3][68]int

// costTabs are the trellis cost tables, laid out as the asm expects.
type costTabs struct {
	probs [1056]uint8
	cost  [4][16][4][128]uint16 // level_cost plus fixed costs; ctx 3 and v > 67 unused
	eob   [4][16][4]uint16      // cost of ending after position n at ctx
}

// defTabs are costTabs at defaultProbs with fixedCost.
var defTabs costTabs

func init() {
	var b [16]uint16
	for v := 1; v < 2048; v++ {
		c, _ := recordLevel(b[:0], v, 0)
		fixedCost[v] = 256
		for _, tok := range c {
			if tok&fixedTok != 0 {
				fixedCost[v] += bitCost[tok>>15][uint8(tok)]
			}
		}
	}
	for typ := 0; typ < 4; typ++ {
		for band := 0; band < 8; band++ {
			for ctx := 0; ctx < 3; ctx++ {
				p := typ*264 + (band*3+ctx)*11
				c0 := 0
				if ctx > 0 {
					c0 = bitCost[1][defaultProbs[p]]
				}
				lc := &levelCost[typ][band][ctx]
				lc[0] = bitCost[0][defaultProbs[p+1]] + c0
				for v := 1; v < 68; v++ {
					c, _ := recordLevel(b[:0], v, p)
					lc[v] = bitCost[1][defaultProbs[p+1]] + c0
					for _, tok := range c {
						if tok&fixedTok == 0 {
							lc[v] += bitCost[tok>>15][defaultProbs[tok&0x7ff]]
						}
					}
				}
			}
		}
	}
	defTabs.probs = defaultProbs
	for typ := range defTabs.cost {
		for n := range defTabs.cost[typ] {
			for ctx := 0; ctx < 3; ctx++ {
				for v := 0; v < 68; v++ {
					defTabs.cost[typ][n][ctx][v] = uint16(levelCost[typ][bands[n]][ctx][v] + fixedCost[v])
				}
				if n < 15 {
					defTabs.eob[typ][n][ctx] = uint16(bitCost[0][defaultProbs[typ*264+(int(bands[n+1])*3+ctx)*11]])
				}
			}
		}
	}
}

// kWeightTrellis, quant_enc.c:492-499.
var trellisWeight = [16]int64{30, 27, 19, 11, 27, 24, 17, 10, 19, 17, 12, 8, 11, 10, 8, 6}

const maxCost = 1<<55 - 1

// Port of libwebp TrellisQuantizeBlock (quant_enc.c:569-751), BSD-3-Clause.
// Same conventions as quantize: raster in dequantized in place, zigzag out.
func (m *quantMatrix) trellis(in, out *[16]int16, ctx0, typ, lambda int) bool {
	return defTabs.trellis(m, in, out, ctx0, typ, lambda)
}
