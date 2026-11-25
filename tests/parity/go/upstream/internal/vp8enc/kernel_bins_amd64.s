//go:build !purego

#include "textflag.h"

// func binStatsAsm(b *uint8, n int) (last, c0, c1, c2, c3 int)
// AVX2; n is a nonzero multiple of 32 and at most 8160.
TEXT ·binStatsAsm(SB), NOSPLIT, $0-56
	MOVQ     b+0(FP), SI
	MOVQ     n+8(FP), CX
	VPXOR    Y0, Y0, Y0 // max
	VPXOR    Y1, Y1, Y1 // counts of 0, 1, 2, 3
	VPXOR    Y2, Y2, Y2
	VPXOR    Y3, Y3, Y3
	VPXOR    Y4, Y4, Y4
	VPXOR    Y5, Y5, Y5
	VPCMPEQB Y6, Y6, Y6
	VPSUBB   Y6, Y5, Y7 // 1
	VPADDB   Y7, Y7, Y8 // 2
	VPADDB   Y7, Y8, Y9 // 3

loop:
	VMOVDQU  (SI), Y10
	VPMAXUB  Y10, Y0, Y0
	VPCMPEQB Y5, Y10, Y11
	VPSUBB   Y11, Y1, Y1
	VPCMPEQB Y7, Y10, Y11
	VPSUBB   Y11, Y2, Y2
	VPCMPEQB Y8, Y10, Y11
	VPSUBB   Y11, Y3, Y3
	VPCMPEQB Y9, Y10, Y11
	VPSUBB   Y11, Y4, Y4
	ADDQ     $32, SI
	SUBQ     $32, CX
	JNZ      loop

	VEXTRACTI128 $1, Y0, X10
	VPMAXUB      X10, X0, X0
	VPSRLDQ      $8, X0, X10
	VPMAXUB      X10, X0, X0
	VPSRLDQ      $4, X0, X10
	VPMAXUB      X10, X0, X0
	VPSRLDQ      $2, X0, X10
	VPMAXUB      X10, X0, X0
	VPSRLDQ      $1, X0, X10
	VPMAXUB      X10, X0, X0
	VMOVQ        X0, AX
	MOVBQZX      AL, AX
	MOVQ         AX, last+16(FP)

#define BSUM(y, x, off) \
	VPSADBW      Y5, y, y \
	VEXTRACTI128 $1, y, X10 \
	VPADDQ       X10, x, x \
	VPSRLDQ      $8, x, X10 \
	VPADDQ       X10, x, x \
	VMOVQ        x, AX \
	MOVQ         AX, off(FP)

	BSUM(Y1, X1, c0+24)
	BSUM(Y2, X2, c1+32)
	BSUM(Y3, X3, c2+40)
	BSUM(Y4, X4, c3+48)
	VZEROUPPER
	RET
