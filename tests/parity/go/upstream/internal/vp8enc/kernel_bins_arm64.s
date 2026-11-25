//go:build !purego

#include "textflag.h"

// func binStatsAsm(b *uint8, n int) (last, c0, c1, c2, c3 int)
// NEON; n is a nonzero multiple of 16 and at most 4080.
TEXT ·binStatsAsm(SB), NOSPLIT, $0-56
	MOVD  b+0(FP), R0
	MOVD  n+8(FP), R1
	VEOR  V0.B16, V0.B16, V0.B16 // max
	VEOR  V1.B16, V1.B16, V1.B16 // counts of 0, 1, 2, 3
	VEOR  V2.B16, V2.B16, V2.B16
	VEOR  V3.B16, V3.B16, V3.B16
	VEOR  V4.B16, V4.B16, V4.B16
	VEOR  V8.B16, V8.B16, V8.B16
	VMOVI $1, V5.B16
	VMOVI $2, V6.B16
	VMOVI $3, V7.B16

loop:
	VLD1.P 16(R0), [V9.B16]
	VUMAX  V9.B16, V0.B16, V0.B16
	VCMEQ  V8.B16, V9.B16, V10.B16
	VSUB   V10.B16, V1.B16, V1.B16
	VCMEQ  V5.B16, V9.B16, V10.B16
	VSUB   V10.B16, V2.B16, V2.B16
	VCMEQ  V6.B16, V9.B16, V10.B16
	VSUB   V10.B16, V3.B16, V3.B16
	VCMEQ  V7.B16, V9.B16, V10.B16
	VSUB   V10.B16, V4.B16, V4.B16
	SUBS   $16, R1
	BNE    loop

	VUMAXV  V0.B16, V0
	VMOV    V0.B[0], R2
	MOVD    R2, last+16(FP)
	VUADDLV V1.B16, V1
	VMOV    V1.H[0], R2
	MOVD    R2, c0+24(FP)
	VUADDLV V2.B16, V2
	VMOV    V2.H[0], R2
	MOVD    R2, c1+32(FP)
	VUADDLV V3.B16, V3
	VMOV    V3.H[0], R2
	MOVD    R2, c2+40(FP)
	VUADDLV V4.B16, V4
	VMOV    V4.H[0], R2
	MOVD    R2, c3+48(FP)
	RET
