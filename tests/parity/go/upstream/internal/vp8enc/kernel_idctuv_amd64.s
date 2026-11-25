//go:build !purego

#include "textflag.h"

// Words: k1 = 20091, k2 = -30068, 4, 2048.
DATA iuvk<>+0(SB)/8, $0x4e7b4e7b4e7b4e7b
DATA iuvk<>+8(SB)/8, $0x4e7b4e7b4e7b4e7b
DATA iuvk<>+16(SB)/8, $0x4e7b4e7b4e7b4e7b
DATA iuvk<>+24(SB)/8, $0x4e7b4e7b4e7b4e7b
DATA iuvk<>+32(SB)/8, $0x8a8c8a8c8a8c8a8c
DATA iuvk<>+40(SB)/8, $0x8a8c8a8c8a8c8a8c
DATA iuvk<>+48(SB)/8, $0x8a8c8a8c8a8c8a8c
DATA iuvk<>+56(SB)/8, $0x8a8c8a8c8a8c8a8c
DATA iuvk<>+64(SB)/8, $0x0004000400040004
DATA iuvk<>+72(SB)/8, $0x0004000400040004
DATA iuvk<>+80(SB)/8, $0x0004000400040004
DATA iuvk<>+88(SB)/8, $0x0004000400040004
DATA iuvk<>+96(SB)/8, $0x0800080008000800
DATA iuvk<>+104(SB)/8, $0x0800080008000800
DATA iuvk<>+112(SB)/8, $0x0800080008000800
DATA iuvk<>+120(SB)/8, $0x0800080008000800
GLOBL iuvk<>(SB), RODATA, $128

// IPASSY is IPASS16 on YMM rows r0-r3 in place, k1 in Y14 and k2 in
// Y15; t0-t3 are scratch.
#define IPASSY(r0, r1, r2, r3, t0, t1, t2, t3) \
	VPSUBW  r2, r0, t0 \
	VPADDW  r2, r0, r0 \
	VPMULHW Y15, r1, t1 \
	VPMULHW Y14, r3, t2 \
	VPSUBW  t2, t1, t1 \
	VPSUBW  r3, r1, t2 \
	VPADDW  t2, t1, t1 \
	VPMULHW Y14, r1, t2 \
	VPMULHW Y15, r3, t3 \
	VPADDW  t3, t2, t2 \
	VPADDW  r3, r1, r1 \
	VPADDW  t2, r1, r1 \
	VPSUBW  r1, r0, r3 \
	VPADDW  r1, r0, r0 \
	VPSUBW  t1, t0, r2 \
	VPADDW  t1, t0, r1

// TRANSPOSEY transposes the 4x4 word block in each qword of a-d into
// w-z, clobbering a-d.
#define TRANSPOSEY(a, b, c, d, w, x, y, z) \
	VPUNPCKLWD  b, a, w \
	VPUNPCKHWD  b, a, x \
	VPUNPCKLWD  d, c, y \
	VPUNPCKHWD  d, c, z \
	VPUNPCKLDQ  y, w, a \
	VPUNPCKHDQ  y, w, b \
	VPUNPCKLDQ  z, x, c \
	VPUNPCKHDQ  z, x, d \
	VPUNPCKLQDQ c, a, w \
	VPUNPCKHQDQ c, a, x \
	VPUNPCKLQDQ d, b, y \
	VPUNPCKHQDQ d, b, z

// ADDROW adds r to dst rows j (lane 0) and j+4 (lane 1).
#define ADDROW(off, r) \
	VMOVQ     off(DI), X12 \
	VPINSRQ   $1, (128+off)(DI), X12, X12 \
	VPMOVZXBW X12, Y12 \
	VPADDSW   Y12, r, r

// IDCTUV is the 16-bit inverse DCT of the 4 blocks at (SI) added to
// the 8x8 plane at (DI). Qword n of each row holds block n.
#define IDCTUV \
	VMOVDQU     (SI), X4 \
	VINSERTI128 $1, 64(SI), Y4, Y4 \
	VMOVDQU     32(SI), X5 \
	VINSERTI128 $1, 96(SI), Y5, Y5 \
	VMOVDQU     16(SI), X6 \
	VINSERTI128 $1, 80(SI), Y6, Y6 \
	VMOVDQU     48(SI), X7 \
	VINSERTI128 $1, 112(SI), Y7, Y7 \
	VPUNPCKLQDQ Y5, Y4, Y0 \
	VPUNPCKHQDQ Y5, Y4, Y1 \
	VPUNPCKLQDQ Y7, Y6, Y2 \
	VPUNPCKHQDQ Y7, Y6, Y3 \
	IPASSY(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7) \
	TRANSPOSEY(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7) \
	VPADDW      Y13, Y4, Y4 \
	IPASSY(Y4, Y5, Y6, Y7, Y0, Y1, Y2, Y3) \
	VPSRAW      $3, Y4, Y4 \
	VPSRAW      $3, Y5, Y5 \
	VPSRAW      $3, Y6, Y6 \
	VPSRAW      $3, Y7, Y7 \
	TRANSPOSEY(Y4, Y5, Y6, Y7, Y0, Y1, Y2, Y3) \
	ADDROW(0, Y0) \
	ADDROW(32, Y1) \
	ADDROW(64, Y2) \
	ADDROW(96, Y3) \
	VPACKUSWB    Y1, Y0, Y0 \
	VPACKUSWB    Y3, Y2, Y2 \
	VMOVQ        X0, (DI) \
	VPEXTRQ      $1, X0, 32(DI) \
	VEXTRACTI128 $1, Y0, X0 \
	VMOVQ        X0, 128(DI) \
	VPEXTRQ      $1, X0, 160(DI) \
	VMOVQ        X2, 64(DI) \
	VPEXTRQ      $1, X2, 96(DI) \
	VEXTRACTI128 $1, Y2, X2 \
	VMOVQ        X2, 192(DI) \
	VPEXTRQ      $1, X2, 224(DI)

// func idctAddsUVAsm(dq *[8][16]int16, dst *uint8) (ok bool)
// Returns false, dst untouched, unless every level is in [-2048, 2047].
TEXT ·idctAddsUVAsm(SB), NOSPLIT, $0-17
	MOVQ    dq+0(FP), SI
	MOVQ    dst+8(FP), DI
	VMOVDQU iuvk<>+96(SB), Y13
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 64(SI), Y2
	VMOVDQU 96(SI), Y3
	VMOVDQU 128(SI), Y4
	VMOVDQU 160(SI), Y5
	VMOVDQU 192(SI), Y6
	VMOVDQU 224(SI), Y7

	// Y8 and Y10 are nonzero for a nonzero U and V plane.
	VPOR Y1, Y0, Y8
	VPOR Y3, Y2, Y9
	VPOR Y9, Y8, Y8
	VPOR Y5, Y4, Y10
	VPOR Y7, Y6, Y11
	VPOR Y11, Y10, Y10

	// x+2048 below 4096 for every level.
	VPADDW Y13, Y0, Y0
	VPADDW Y13, Y1, Y1
	VPADDW Y13, Y2, Y2
	VPADDW Y13, Y3, Y3
	VPADDW Y13, Y4, Y4
	VPADDW Y13, Y5, Y5
	VPADDW Y13, Y6, Y6
	VPADDW Y13, Y7, Y7
	VPOR   Y1, Y0, Y0
	VPOR   Y3, Y2, Y2
	VPOR   Y5, Y4, Y4
	VPOR   Y7, Y6, Y6
	VPOR   Y2, Y0, Y0
	VPOR   Y6, Y4, Y4
	VPOR   Y4, Y0, Y0
	VPSRLW $12, Y0, Y0
	VPTEST Y0, Y0
	JNZ    iuvslow

	VMOVDQU iuvk<>+0(SB), Y14
	VMOVDQU iuvk<>+32(SB), Y15
	VMOVDQU iuvk<>+64(SB), Y13
	VPTEST  Y8, Y8
	JZ      iuvv
	IDCTUV

iuvv:
	VPTEST Y10, Y10
	JZ     iuvdone
	ADDQ   $128, SI
	ADDQ   $16, DI
	IDCTUV

iuvdone:
	MOVB $1, ok+16(FP)
	VZEROUPPER
	RET

iuvslow:
	MOVB $0, ok+16(FP)
	VZEROUPPER
	RET
