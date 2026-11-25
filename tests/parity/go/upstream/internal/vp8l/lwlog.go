// Port of libwebp 1.2.2 dsp/lossless_enc.c, dsp/lossless.c and dsp/lossless_common.h, with clog from Go's math/log.go. Copyright 2012 Google Inc. BSD-3, see LICENSE.libwebp.

package vp8l

import (
	"math"
	"math/bits"
)

// lossless_common.h:63-66.
const (
	approxLogWithCorrectionMax         = 65536
	approxLogMax                       = 4096
	log2Recip                  float64 = 1.44269504088896338700465094007086
	logLookupIdxMax                    = 256
)

// kLog2Table, lossless_enc.c:28. Go float32 constants round like C's f literals.
var log2Table = [256]float32{
	0.0000000000000000, 0.0000000000000000,
	1.0000000000000000, 1.5849625007211560,
	2.0000000000000000, 2.3219280948873621,
	2.5849625007211560, 2.8073549220576041,
	3.0000000000000000, 3.1699250014423121,
	3.3219280948873621, 3.4594316186372973,
	3.5849625007211560, 3.7004397181410921,
	3.8073549220576041, 3.9068905956085187,
	4.0000000000000000, 4.0874628412503390,
	4.1699250014423121, 4.2479275134435852,
	4.3219280948873626, 4.3923174227787606,
	4.4594316186372973, 4.5235619560570130,
	4.5849625007211560, 4.6438561897747243,
	4.7004397181410917, 4.7548875021634682,
	4.8073549220576037, 4.8579809951275718,
	4.9068905956085187, 4.9541963103868749,
	5.0000000000000000, 5.0443941193584533,
	5.0874628412503390, 5.1292830169449663,
	5.1699250014423121, 5.2094533656289501,
	5.2479275134435852, 5.2854022188622487,
	5.3219280948873626, 5.3575520046180837,
	5.3923174227787606, 5.4262647547020979,
	5.4594316186372973, 5.4918530963296747,
	5.5235619560570130, 5.5545888516776376,
	5.5849625007211560, 5.6147098441152083,
	5.6438561897747243, 5.6724253419714951,
	5.7004397181410917, 5.7279204545631987,
	5.7548875021634682, 5.7813597135246599,
	5.8073549220576037, 5.8328900141647412,
	5.8579809951275718, 5.8826430493618415,
	5.9068905956085187, 5.9307373375628866,
	5.9541963103868749, 5.9772799234999167,
	6.0000000000000000, 6.0223678130284543,
	6.0443941193584533, 6.0660891904577720,
	6.0874628412503390, 6.1085244567781691,
	6.1292830169449663, 6.1497471195046822,
	6.1699250014423121, 6.1898245588800175,
	6.2094533656289501, 6.2288186904958804,
	6.2479275134435852, 6.2667865406949010,
	6.2854022188622487, 6.3037807481771030,
	6.3219280948873626, 6.3398500028846243,
	6.3575520046180837, 6.3750394313469245,
	6.3923174227787606, 6.4093909361377017,
	6.4262647547020979, 6.4429434958487279,
	6.4594316186372973, 6.4757334309663976,
	6.4918530963296747, 6.5077946401986963,
	6.5235619560570130, 6.5391588111080309,
	6.5545888516776376, 6.5698556083309478,
	6.5849625007211560, 6.5999128421871278,
	6.6147098441152083, 6.6293566200796094,
	6.6438561897747243, 6.6582114827517946,
	6.6724253419714951, 6.6865005271832185,
	6.7004397181410917, 6.7142455176661224,
	6.7279204545631987, 6.7414669864011464,
	6.7548875021634682, 6.7681843247769259,
	6.7813597135246599, 6.7944158663501061,
	6.8073549220576037, 6.8201789624151878,
	6.8328900141647412, 6.8454900509443747,
	6.8579809951275718, 6.8703647195834047,
	6.8826430493618415, 6.8948177633079437,
	6.9068905956085187, 6.9188632372745946,
	6.9307373375628866, 6.9425145053392398,
	6.9541963103868749, 6.9657842846620869,
	6.9772799234999167, 6.9886846867721654,
	7.0000000000000000, 7.0112272554232539,
	7.0223678130284543, 7.0334230015374501,
	7.0443941193584533, 7.0552824355011898,
	7.0660891904577720, 7.0768155970508308,
	7.0874628412503390, 7.0980320829605263,
	7.1085244567781691, 7.1189410727235076,
	7.1292830169449663, 7.1395513523987936,
	7.1497471195046822, 7.1598713367783890,
	7.1699250014423121, 7.1799090900149344,
	7.1898245588800175, 7.1996723448363644,
	7.2094533656289501, 7.2191685204621611,
	7.2288186904958804, 7.2384047393250785,
	7.2479275134435852, 7.2573878426926521,
	7.2667865406949010, 7.2761244052742375,
	7.2854022188622487, 7.2946207488916270,
	7.3037807481771030, 7.3128829552843557,
	7.3219280948873626, 7.3309168781146167,
	7.3398500028846243, 7.3487281542310771,
	7.3575520046180837, 7.3663222142458160,
	7.3750394313469245, 7.3837042924740519,
	7.3923174227787606, 7.4008794362821843,
	7.4093909361377017, 7.4178525148858982,
	7.4262647547020979, 7.4346282276367245,
	7.4429434958487279, 7.4512111118323289,
	7.4594316186372973, 7.4676055500829976,
	7.4757334309663976, 7.4838157772642563,
	7.4918530963296747, 7.4998458870832056,
	7.5077946401986963, 7.5156998382840427,
	7.5235619560570130, 7.5313814605163118,
	7.5391588111080309, 7.5468944598876364,
	7.5545888516776376, 7.5622424242210728,
	7.5698556083309478, 7.5774288280357486,
	7.5849625007211560, 7.5924570372680806,
	7.5999128421871278, 7.6073303137496104,
	7.6147098441152083, 7.6220518194563764,
	7.6293566200796094, 7.6366246205436487,
	7.6438561897747243, 7.6510516911789281,
	7.6582114827517946, 7.6653359171851764,
	7.6724253419714951, 7.6794800995054464,
	7.6865005271832185, 7.6934869574993252,
	7.7004397181410917, 7.7073591320808825,
	7.7142455176661224, 7.7210991887071855,
	7.7279204545631987, 7.7347096202258383,
	7.7414669864011464, 7.7481928495894605,
	7.7548875021634682, 7.7615512324444795,
	7.7681843247769259, 7.7747870596011736,
	7.7813597135246599, 7.7879025593914317,
	7.7944158663501061, 7.8008998999203047,
	7.8073549220576037, 7.8137811912170374,
	7.8201789624151878, 7.8265484872909150,
	7.8328900141647412, 7.8392037880969436,
	7.8454900509443747, 7.8517490414160571,
	7.8579809951275718, 7.8641861446542797,
	7.8703647195834047, 7.8765169465649993,
	7.8826430493618415, 7.8887432488982591,
	7.8948177633079437, 7.9008668079807486,
	7.9068905956085187, 7.9128893362299619,
	7.9188632372745946, 7.9248125036057812,
	7.9307373375628866, 7.9366379390025709,
	7.9425145053392398, 7.9483672315846778,
	7.9541963103868749, 7.9600019320680805,
	7.9657842846620869, 7.9715435539507719,
	7.9772799234999167, 7.9829935746943103,
	7.9886846867721654, 7.9943534368588577,
}

// kSLog2Table, lossless_enc.c:159.
var slog2Table = [256]float32{
	0.00000000, 0.00000000, 2.00000000, 4.75488750,
	8.00000000, 11.60964047, 15.50977500, 19.65148445,
	24.00000000, 28.52932501, 33.21928095, 38.05374781,
	43.01955001, 48.10571634, 53.30296891, 58.60335893,
	64.00000000, 69.48686830, 75.05865003, 80.71062276,
	86.43856190, 92.23866588, 98.10749561, 104.04192499,
	110.03910002, 116.09640474, 122.21143267, 128.38196256,
	134.60593782, 140.88144886, 147.20671787, 153.58008562,
	160.00000000, 166.46500594, 172.97373660, 179.52490559,
	186.11730005, 192.74977453, 199.42124551, 206.13068654,
	212.87712380, 219.65963219, 226.47733176, 233.32938445,
	240.21499122, 247.13338933, 254.08384998, 261.06567603,
	268.07820003, 275.12078236, 282.19280949, 289.29369244,
	296.42286534, 303.57978409, 310.76392512, 317.97478424,
	325.21187564, 332.47473081, 339.76289772, 347.07593991,
	354.41343574, 361.77497759, 369.16017124, 376.56863518,
	384.00000000, 391.45390785, 398.93001188, 406.42797576,
	413.94747321, 421.48818752, 429.04981119, 436.63204548,
	444.23460010, 451.85719280, 459.49954906, 467.16140179,
	474.84249102, 482.54256363, 490.26137307, 497.99867911,
	505.75424759, 513.52785023, 521.31926438, 529.12827280,
	536.95466351, 544.79822957, 552.65876890, 560.53608414,
	568.42998244, 576.34027536, 584.26677867, 592.20931226,
	600.16769996, 608.14176943, 616.13135206, 624.13628279,
	632.15640007, 640.19154569, 648.24156472, 656.30630539,
	664.38561898, 672.47935976, 680.58738488, 688.70955430,
	696.84573069, 704.99577935, 713.15956818, 721.33696754,
	729.52785023, 737.73209140, 745.94956849, 754.18016116,
	762.42375127, 770.68022275, 778.94946161, 787.23135586,
	795.52579543, 803.83267219, 812.15187982, 820.48331383,
	828.82687147, 837.18245171, 845.54995518, 853.92928416,
	862.32034249, 870.72303558, 879.13727036, 887.56295522,
	896.00000000, 904.44831595, 912.90781569, 921.37841320,
	929.86002376, 938.35256392, 946.85595152, 955.37010560,
	963.89494641, 972.43039537, 980.97637504, 989.53280911,
	998.09962237, 1006.67674069, 1015.26409097, 1023.86160116,
	1032.46920021, 1041.08681805, 1049.71438560, 1058.35183469,
	1066.99909811, 1075.65610955, 1084.32280357, 1092.99911564,
	1101.68498204, 1110.38033993, 1119.08512727, 1127.79928282,
	1136.52274614, 1145.25545758, 1153.99735821, 1162.74838989,
	1171.50849518, 1180.27761738, 1189.05570047, 1197.84268914,
	1206.63852876, 1215.44316535, 1224.25654560, 1233.07861684,
	1241.90932703, 1250.74862473, 1259.59645914, 1268.45278005,
	1277.31753781, 1286.19068338, 1295.07216828, 1303.96194457,
	1312.85996488, 1321.76618236, 1330.68055071, 1339.60302413,
	1348.53355734, 1357.47210556, 1366.41862452, 1375.37307041,
	1384.33539991, 1393.30557020, 1402.28353887, 1411.26926400,
	1420.26270412, 1429.26381818, 1438.27256558, 1447.28890615,
	1456.31280014, 1465.34420819, 1474.38309138, 1483.42941118,
	1492.48312945, 1501.54420843, 1510.61261078, 1519.68829949,
	1528.77123795, 1537.86138993, 1546.95871952, 1556.06319119,
	1565.17476976, 1574.29342040, 1583.41910860, 1592.55180020,
	1601.69146137, 1610.83805860, 1619.99155871, 1629.15192882,
	1638.31913637, 1647.49314911, 1656.67393509, 1665.86146266,
	1675.05570047, 1684.25661744, 1693.46418280, 1702.67836605,
	1711.89913698, 1721.12646563, 1730.36032233, 1739.60067768,
	1748.84750254, 1758.10076802, 1767.36044551, 1776.62650662,
	1785.89892323, 1795.17766747, 1804.46271172, 1813.75402857,
	1823.05159087, 1832.35537170, 1841.66534438, 1850.98148244,
	1860.30375965, 1869.63214999, 1878.96662767, 1888.30716711,
	1897.65374295, 1907.00633003, 1916.36490342, 1925.72943838,
	1935.09991037, 1944.47629506, 1953.85856831, 1963.24670620,
	1972.64068498, 1982.04048108, 1991.44607117, 2000.85743204,
	2010.27454072, 2019.69737440, 2029.12591044, 2038.56012640,
}

// clog is a pure-Go copy of Go's math.Log, so amd64 and arm64 agree.
// Products are wrapped in float64() to block FMA fusion.
func clog(x float64) float64 {
	const (
		Ln2Hi = 6.93147180369123816490e-01 /* 3fe62e42 fee00000 */
		Ln2Lo = 1.90821492927058770002e-10 /* 3dea39ef 35793c76 */
		L1    = 6.666666666666735130e-01   /* 3FE55555 55555593 */
		L2    = 3.999999999940941908e-01   /* 3FD99999 9997FA04 */
		L3    = 2.857142874366239149e-01   /* 3FD24924 94229359 */
		L4    = 2.222219843214978396e-01   /* 3FCC71C5 1D8E78AF */
		L5    = 1.818357216161805012e-01   /* 3FC74664 96CB03DE */
		L6    = 1.531383769920937332e-01   /* 3FC39A09 D078C69F */
		L7    = 1.479819860511658591e-01   /* 3FC2F112 DF3E5244 */
	)

	// special cases
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	}

	// reduce
	f1, ki := math.Frexp(x)
	if f1 < math.Sqrt2/2 {
		f1 *= 2
		ki--
	}
	f := f1 - 1
	k := float64(ki)

	// compute
	s := f / (2 + f)
	s2 := float64(s * s)
	s4 := float64(s2 * s2)
	t1 := float64(s2 * (L1 + float64(s4*(L3+float64(s4*(L5+float64(s4*L7)))))))
	t2 := float64(s4 * (L2 + float64(s4*(L4+float64(s4*L6)))))
	R := t1 + t2
	hfsq := float64(float64(0.5*f) * f)
	return float64(k*Ln2Hi) - ((hfsq - (float64(s*(hfsq+R)) + float64(k*Ln2Lo))) - f)
}

// FastSLog2Slow_C, lossless_enc.c:329.
func fastSLog2Slow(v uint32) float32 {
	if v < approxLogWithCorrectionMax {
		logCnt := bits.Len32(v) - 8 // BitsLog2Floor(v) - 7
		y := uint32(1) << logCnt
		vf := float32(v)
		origV := v
		v >>= logCnt
		correction := int((23 * (origV & (y - 1))) >> 4)
		return float32(vf*(log2Table[v]+float32(logCnt))) + float32(correction)
	}
	return float32(float64(log2Recip*float64(v)) * clog(float64(v)))
}

// FastLog2Slow_C, lossless_enc.c:365.
func fastLog2Slow(v uint32) float32 {
	if v < approxLogWithCorrectionMax {
		logCnt := bits.Len32(v) - 8 // BitsLog2Floor(v) - 7
		y := uint32(1) << logCnt
		origV := v
		v >>= logCnt
		log2 := float64(log2Table[v] + float32(logCnt))
		if origV >= approxLogMax {
			correction := int((23 * (origV & (y - 1))) >> 4)
			log2 += float64(correction) / float64(origV)
		}
		return float32(log2)
	}
	return float32(log2Recip * clog(float64(v)))
}

// VP8LFastSLog2, lossless_common.h:89.
func fastSLog2(v uint32) float32 {
	if v < logLookupIdxMax {
		return slog2Table[v]
	}
	return fastSLog2Slow(v)
}

// VP8LFastLog2, lossless_common.h:85.
func fastLog2(v uint32) float32 {
	if v < logLookupIdxMax {
		return log2Table[v]
	}
	return fastLog2Slow(v)
}

// subSampleSize is VP8LSubSampleSize, lossless_common.h:51.
func subSampleSize(size, bits int) int {
	return (size + 1<<bits - 1) >> bits
}

// CombinedShannonEntropy_C, lossless_enc.c:403. SSE2 sums the same way.
func combinedShannonEntropy(x, y *[256]int32) float32 {
	retval := 0.0
	var sumX, sumXY int32
	for i := 0; i < 256; i++ {
		xi := x[i]
		if xi != 0 {
			xy := xi + y[i]
			sumX += xi
			retval -= float64(fastSLog2(uint32(xi)))
			sumXY += xy
			retval -= float64(fastSLog2(uint32(xy)))
		} else if y[i] != 0 {
			sumXY += y[i]
			retval -= float64(fastSLog2(uint32(y[i])))
		}
	}
	retval += float64(fastSLog2(uint32(sumX)) + fastSLog2(uint32(sumXY)))
	return float32(retval)
}

// VP8LBitEntropyInit, lossless_enc.c:424.
func (b *bitEntropy) init() {
	b.entropy = 0
	b.sum = 0
	b.nonzeros = 0
	b.maxVal = 0
	b.nonzeroCode = lwNonTrivialSym
}

// streaks zeroing is a plain streaks{} literal (memset in C).

// VP8LBitsEntropyUnrefined, lossless_enc.c:432; n is len(a).
func bitsEntropyUnrefined(a []uint32, b *bitEntropy) {
	b.init()
	for i := range a {
		if a[i] != 0 {
			b.sum += a[i]
			b.nonzeroCode = uint32(i)
			b.nonzeros++
			b.entropy -= float64(fastSLog2(a[i]))
			if b.maxVal < a[i] {
				b.maxVal = a[i]
			}
		}
	}
	b.entropy += float64(fastSLog2(b.sum))
}

// GetEntropyUnrefinedHelper, lossless_enc.c:452.
func getEntropyUnrefinedHelper(val uint32, i int, valPrev *uint32, iPrev *int, b *bitEntropy, s *streaks) {
	streak := i - *iPrev

	// Gather info for the bit entropy.
	if *valPrev != 0 {
		b.sum += *valPrev * uint32(streak)
		b.nonzeros += streak
		b.nonzeroCode = uint32(*iPrev)
		b.entropy -= float64(float32(fastSLog2(*valPrev) * float32(streak)))
		if b.maxVal < *valPrev {
			b.maxVal = *valPrev
		}
	}

	// Gather info for the Huffman cost.
	nz := 0
	if *valPrev != 0 {
		nz = 1
	}
	long := 0
	if streak > 3 {
		long = 1
	}
	s.counts[nz] += long
	s.streaks[nz][long] += streak

	*valPrev = val
	*iPrev = i
}

// GetEntropyUnrefined_C, lossless_enc.c:476; length is len(x).
func getEntropyUnrefined(x []uint32, b *bitEntropy, s *streaks) {
	iPrev := 0
	xPrev := x[0]

	*s = streaks{}
	b.init()

	i := 1
	for ; i < len(x); i++ {
		if x[i] != xPrev {
			getEntropyUnrefinedHelper(x[i], i, &xPrev, &iPrev, b, s)
		}
	}
	getEntropyUnrefinedHelper(0, i, &xPrev, &iPrev, b, s)

	b.entropy += float64(fastSLog2(b.sum))
}

// GetCombinedEntropyUnrefined_C, lossless_enc.c:497; length is len(x).
func getCombinedEntropyUnrefined(x, y []uint32, b *bitEntropy, s *streaks) {
	y = y[:len(x)]
	iPrev := 0
	xyPrev := x[0] + y[0]

	*s = streaks{}
	b.init()

	i := 1
	for ; i < len(x); i++ {
		xy := x[i] + y[i]
		if xy != xyPrev {
			getEntropyUnrefinedHelper(xy, i, &xyPrev, &iPrev, b, s)
		}
	}
	getEntropyUnrefinedHelper(0, i, &xyPrev, &iPrev, b, s)

	b.entropy += float64(fastSLog2(b.sum))
}

// VectorMismatch_C, lossless_enc.c:606.
// Compares two pixels per 64-bit load.
func vectorMismatch(a, b []uint32, length int) int {
	a = a[:length]
	b = b[:len(a)]
	i := 0
	for ; i < len(a)-1; i += 2 {
		p, q := (*[2]uint32)(a[i:]), (*[2]uint32)(b[i:])
		x := uint64(p[0]) | uint64(p[1])<<32
		y := uint64(q[0]) | uint64(q[1])<<32
		if x != y {
			return i + bits.TrailingZeros64(x^y)>>5
		}
	}
	if i < len(a) && a[i] == b[i] {
		i++
	}
	return i
}

// VP8LBundleColorMap_C, lossless_enc.c:617.
func bundleColorMap(row []uint8, width, xbits int, dst []uint32) {
	if xbits > 0 {
		bitDepth := 1 << (3 - xbits)
		mask := 1<<xbits - 1
		code := uint32(0xff000000)
		for x := 0; x < width; x++ {
			xsub := x & mask
			if xsub == 0 {
				code = 0xff000000
			}
			code |= uint32(row[x]) << (8 + bitDepth*xsub)
			dst[x>>xbits] = code
		}
	} else {
		for x := 0; x < width; x++ {
			dst[x] = 0xff000000 | uint32(row[x])<<8
		}
	}
}

// ExtraCost_C, lossless_enc.c:639; length is len(pop). uint32 product.
func extraCost(pop []uint32) float64 {
	cost := 0.0
	for i := 2; i < len(pop)-2; i++ {
		cost += float64(uint32(i>>1) * pop[i+2])
	}
	return cost
}

// ExtraCostCombined_C, lossless_enc.c:646; length is len(x). int32 product.
func extraCostCombined(x, y []uint32) float64 {
	cost := 0.0
	for i := 2; i < len(x)-2; i++ {
		xy := int32(x[i+2] + y[i+2])
		cost += float64(int32(i>>1) * xy)
	}
	return cost
}

// parts lists the five histogram arrays in is_used order.
func (h *lwHisto) parts() [5][]uint32 {
	return [5][]uint32{h.literal[:], h.red[:], h.blue[:], h.alpha[:], h.distance[:]}
}

// VP8LHistogramAdd, lossless_enc.c:694. b may equal out.
func vp8lHistogramAdd(a, b, out *lwHisto) {
	ap, bp, op := a.parts(), b.parts(), out.parts()
	if b != out {
		for k := 0; k < 5; k++ { // ADD
			if a.isUsed[k] {
				if b.isUsed[k] {
					for i := range op[k] {
						op[k][i] = ap[k][i] + bp[k][i]
					}
				} else {
					copy(op[k], ap[k])
				}
			} else if b.isUsed[k] {
				copy(op[k], bp[k])
			} else {
				clear(op[k])
			}
		}
		for k := 0; k < 5; k++ {
			out.isUsed[k] = a.isUsed[k] || b.isUsed[k]
		}
	} else {
		for k := 0; k < 5; k++ { // ADD_EQ
			if a.isUsed[k] {
				if out.isUsed[k] {
					for i := range op[k] {
						op[k][i] += ap[k][i]
					}
				} else {
					copy(op[k], ap[k])
				}
			}
		}
		for k := 0; k < 5; k++ {
			out.isUsed[k] = out.isUsed[k] || a.isUsed[k]
		}
	}
}

// VP8LSubPixels, lossless_common.h:160.
func subPixels(a, b uint32) uint32 {
	alphaAndGreen := 0x00ff00ff + (a & 0xff00ff00) - (b & 0xff00ff00)
	redAndBlue := 0xff00ff00 + (a & 0x00ff00ff) - (b & 0x00ff00ff)
	return alphaAndGreen&0xff00ff00 | redAndBlue&0x00ff00ff
}

// Average2, lossless.c:29.
func average2(a, b uint32) uint32 {
	return ((a^b)&0xfefefefe)>>1 + (a & b)
}

// Average3, lossless.c:33.
func average3(a, b, c uint32) uint32 {
	return average2(average2(a, c), b)
}

// Average4, lossless.c:37.
func average4(a, b, c, d uint32) uint32 {
	return average2(average2(a, b), average2(c, d))
}

// Clip255, lossless.c:42.
func clip255(a uint32) uint32 {
	if a < 256 {
		return a
	}
	// 0 when a is negative, 255 when positive.
	return ^a >> 24
}

// AddSubtractComponentFull, lossless.c:51.
func addSubtractComponentFull(a, b, c int) int {
	return int(clip255(uint32(a + b - c)))
}

// ClampedAddSubtractFull, lossless.c:55.
func clampedAddSubtractFull(c0, c1, c2 uint32) uint32 {
	a := addSubtractComponentFull(int(c0>>24), int(c1>>24), int(c2>>24))
	r := addSubtractComponentFull(int((c0>>16)&0xff), int((c1>>16)&0xff), int((c2>>16)&0xff))
	g := addSubtractComponentFull(int((c0>>8)&0xff), int((c1>>8)&0xff), int((c2>>8)&0xff))
	b := addSubtractComponentFull(int(c0&0xff), int(c1&0xff), int(c2&0xff))
	return uint32(a)<<24 | uint32(r)<<16 | uint32(g)<<8 | uint32(b)
}

// AddSubtractComponentHalf, lossless.c:68.
func addSubtractComponentHalf(a, b int) int {
	return int(clip255(uint32(a + (a-b)/2)))
}

// ClampedAddSubtractHalf, lossless.c:72.
func clampedAddSubtractHalf(c0, c1, c2 uint32) uint32 {
	ave := average2(c0, c1)
	a := addSubtractComponentHalf(int(ave>>24), int(c2>>24))
	r := addSubtractComponentHalf(int((ave>>16)&0xff), int((c2>>16)&0xff))
	g := addSubtractComponentHalf(int((ave>>8)&0xff), int((c2>>8)&0xff))
	b := addSubtractComponentHalf(int(ave&0xff), int(c2&0xff))
	return uint32(a)<<24 | uint32(r)<<16 | uint32(g)<<8 | uint32(b)
}

// Sub3, lossless.c:90.
func sub3(a, b, c int) int {
	pb := b - c
	pa := a - c
	return absInt(pb) - absInt(pa)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// Select, lossless.c:98.
func selectPred(a, b, c uint32) uint32 {
	paMinusPb := sub3(int(a>>24), int(b>>24), int(c>>24)) +
		sub3(int((a>>16)&0xff), int((b>>16)&0xff), int((c>>16)&0xff)) +
		sub3(int((a>>8)&0xff), int((b>>8)&0xff), int((c>>8)&0xff)) +
		sub3(int(a&0xff), int(b&0xff), int(c&0xff))
	if paMinusPb <= 0 {
		return a
	}
	return b
}

// predictor is VP8LPredictor2_C..13_C, lossless.c:121-182, for mode
// 2..13. top is upper[x-1:], so top[0]=TL, top[1]=T, top[2]=TR.
func predictor(mode int, left uint32, top []uint32) uint32 {
	switch mode {
	case 2:
		return top[1]
	case 3:
		return top[2]
	case 4:
		return top[0]
	case 5:
		return average3(left, top[1], top[2])
	case 6:
		return average2(left, top[0])
	case 7:
		return average2(left, top[1])
	case 8:
		return average2(top[0], top[1])
	case 9:
		return average2(top[1], top[2])
	case 10:
		return average4(left, top[0], top[1], top[2])
	case 11:
		return selectPred(top[1], left, top[0])
	case 12:
		return clampedAddSubtractFull(left, top[1], top[0])
	case 13:
		return clampedAddSubtractHalf(left, top[1], top[0])
	}
	panic("vp8l: bad predictor mode")
}

// predictorsSub is VP8LPredictorsSub[mode], lossless_enc.c:724-764.
// It writes out[0:n] for in[x:x+n], reading in[x-1] and upper[x-1:x+n+1].
// upper may be nil for modes 0 and 1.
func predictorsSub(mode int, in, upper []uint32, x, n int, out []uint32) {
	switch mode {
	case 0:
		for i := 0; i < n; i++ {
			out[i] = subPixels(in[x+i], lwARGBBlack)
		}
	case 1:
		for i := 0; i < n; i++ {
			out[i] = subPixels(in[x+i], in[x+i-1])
		}
	case 2: // PredictBatch calls it at x 0
		for i := 0; i < n; i++ {
			out[i] = subPixels(in[x+i], upper[x+i])
		}
	default:
		for i := 0; i < n; i++ {
			out[i] = subPixels(in[x+i], predictor(mode, in[x+i-1], upper[x+i-1:]))
		}
	}
}
