//go:build !purego

#include "go_asm.h"
#include "textflag.h"

// X9/X11/X13 carry DC in lane 0.
// Sharpen fits int16, so signed packing is exact.
// a = |v| + sharpen, as uint16.
// level = a*iq+bias >> 17, 32-bit product from mullo/mulhi.
// Restore sign: X5, X2 hold levels 0-7, 8-15.
// in = level*q, truncated like int16(dq).
// 16-bit v-dq is exact: |v-dq| < 2^15.
// libwebp zigzag shuffles, then swap 3 and 12.
// last = highest nonzero lane, or -1.
#define QUANTK \
	MOVL    quantMatrix_q+4(AX), BX \
	MOVQ    BX, X8 \
	PSHUFLW $0, X8, X8 \
	PSHUFL  $0, X8, X8 \
	MOVO    X8, X9 \
	MOVL    quantMatrix_q(AX), BX \
	PINSRW  $0, BX, X9 \
	MOVL    quantMatrix_iq+4(AX), BX \
	MOVQ    BX, X10 \
	PSHUFLW $0, X10, X10 \
	PSHUFL  $0, X10, X10 \
	MOVO    X10, X11 \
	MOVL    quantMatrix_iq(AX), BX \
	PINSRW  $0, BX, X11 \
	MOVL    quantMatrix_bias+4(AX), BX \
	MOVQ    BX, X12 \
	PSHUFL  $0, X12, X12 \
	MOVO    X12, X13 \
	MOVL    quantMatrix_bias(AX), BX \
	MOVQ    BX, X4 \
	MOVSS   X4, X13 \
	MOVOU    quantMatrix_sharpen(AX), X14 \
	MOVOU    quantMatrix_sharpen+16(AX), X4 \
	PACKSSLW X4, X14 \
	MOVOU    quantMatrix_sharpen+32(AX), X15 \
	MOVOU    quantMatrix_sharpen+48(AX), X4 \
	PACKSSLW X4, X15

// QUANTB is the quantizer with QUANTK's constants in X8-X15.
#define QUANTB QUANTBX(QABS2, QZZ2)

// QUANTB41 is QUANTB with SSSE3 abs and zigzag.
#define QUANTB41 QUANTBX(QABS41, QZZ41)

#define QABS2 \
	MOVO  X0, X6 \
	PSRAW $15, X6 \
	MOVO  X1, X7 \
	PSRAW $15, X7 \
	MOVO  X0, X2 \
	PXOR  X6, X2 \
	PSUBW X6, X2 \
	PADDW X14, X2 \
	MOVO  X1, X3 \
	PXOR  X7, X3 \
	PSUBW X7, X3 \
	PADDW X15, X3

#define QABS41 \
	MOVO  X0, X6 \
	PSRAW $15, X6 \
	MOVO  X1, X7 \
	PSRAW $15, X7 \
	PABSW X0, X2 \
	PADDW X14, X2 \
	PABSW X1, X3 \
	PADDW X15, X3

#define QZZ2 \
	PSHUFHW $0x9c, X5, X5 \
	PSHUFL  $0xd8, X5, X5 \
	PSHUFHW $0xd2, X5, X5 \
	PSHUFLW $0xc9, X2, X2 \
	PSHUFL  $0xd8, X2, X2 \
	PSHUFLW $0x78, X2, X2 \
	PEXTRW  $3, X5, BX \
	PEXTRW  $4, X2, DX \
	PINSRW  $3, DX, X5 \
	PINSRW  $4, BX, X2

// Each output half ORs two PSHUFBs.
#define QZZ41 \
	MOVO   X5, X3 \
	MOVO   X2, X4 \
	PSHUFB zz41<>+0(SB), X5 \
	PSHUFB zz41<>+16(SB), X4 \
	POR    X4, X5 \
	PSHUFB zz41<>+32(SB), X2 \
	PSHUFB zz41<>+48(SB), X3 \
	POR    X3, X2

// Words 0,1,4,-,5,2,3,6 of lo; word 0 of hi into
// lane 3; words 1,4,5,2,-,3,6,7 of hi; word 7 of lo into lane 4.
DATA zz41<>+0(SB)/8, $0x8080090803020100
DATA zz41<>+8(SB)/8, $0x0d0c070605040b0a
DATA zz41<>+16(SB)/8, $0x0100808080808080
DATA zz41<>+24(SB)/8, $0x8080808080808080
DATA zz41<>+32(SB)/8, $0x05040b0a09080302
DATA zz41<>+40(SB)/8, $0x0f0e0d0c07068080
DATA zz41<>+48(SB)/8, $0x8080808080808080
DATA zz41<>+56(SB)/8, $0x8080808080800f0e
GLOBL zz41<>(SB), RODATA, $64

#define QUANTBX(qabs, qzz) \
	qabs \
	MOVO      X2, X4 \
	PMULLW    X11, X4 \
	PMULHUW   X11, X2 \
	MOVO      X4, X5 \
	PUNPCKLWL X2, X5 \
	PUNPCKHWL X2, X4 \
	PADDL     X13, X5 \
	PADDL     X12, X4 \
	PSRLL     $17, X5 \
	PSRLL     $17, X4 \
	PACKSSLW  X4, X5 \
	MOVO      X3, X4 \
	PMULLW    X10, X4 \
	PMULHUW   X10, X3 \
	MOVO      X4, X2 \
	PUNPCKLWL X3, X2 \
	PUNPCKHWL X3, X4 \
	PADDL     X12, X2 \
	PADDL     X12, X4 \
	PSRLL     $17, X2 \
	PSRLL     $17, X4 \
	PACKSSLW  X4, X2 \
	PXOR  X6, X5 \
	PSUBW X6, X5 \
	PXOR  X7, X2 \
	PSUBW X7, X2 \
	MOVO   X5, X3 \
	PMULLW X9, X3 \
	MOVO   X2, X4 \
	PMULLW X8, X4 \
	MOVOU  X3, (SI) \
	MOVOU  X4, 16(SI) \
	PSUBW   X3, X0 \
	PSUBW   X4, X1 \
	PMADDWL X0, X0 \
	PMADDWL X1, X1 \
	PADDL   X1, X0 \
	PSHUFL  $0x4e, X0, X1 \
	PADDL   X1, X0 \
	PSHUFL  $0xb1, X0, X1 \
	PADDL   X1, X0 \
	MOVL    X0, CX \
	qzz \
	MOVOU   X5, (DI) \
	MOVOU   X2, 16(DI) \
	PXOR     X3, X3 \
	PCMPEQW  X3, X5 \
	PCMPEQW  X3, X2 \
	PACKSSWB X2, X5 \
	PMOVMSKB X5, AX \
	XORL     $0xffff, AX \
	MOVQ     $-1, DX \
	BSRQ     AX, BX \
	CMOVQEQ  DX, BX

// QUANT is QUANTB with m's constants from AX.
#define QUANT \
	QUANTK \
	QUANTB

#define QUANT41 \
	QUANTK \
	QUANTB41

// NZ2 counts nonzero AC levels: PSADBW over a 0/1 byte per lane.
#define NZ2 \
	MOVOU    (DI), X0 \
	MOVOU    16(DI), X1 \
	PXOR     X3, X3 \
	PCMPEQW  X3, X0 \
	PCMPEQW  X3, X1 \
	PACKSSWB X1, X0 \
	MOVOU    nzAC<>(SB), X1 \
	PANDN    X1, X0 \
	PSADBW   X3, X0 \
	PSHUFL   $0x4e, X0, X1 \
	PADDL    X1, X0 \
	MOVL     X0, AX \
	MOVL     AX, 12(R13)

// I4TRIAL is the ten-trial loop of i4TrialsAsm with the quantizer qb,
// nonzero AC count nz, abs ab and horizontal pass hp. Per trial: FDCTB,
// quantizer, idct plus prediction to rec, then SSE and TTransform.
#define I4TRIAL(qb, nz, ab, hp, trial, slow, add) \
trial: \
	MOVOU 0(SP), X8 \
	MOVOU 16(SP), X9 \
	MOVOU 32(SP), X10 \
	MOVOU 48(SP), X11 \
	MOVOU 64(SP), X12 \
	MOVOU 80(SP), X13 \
	MOVOU 96(SP), X14 \
	MOVOU 112(SP), X15 \
	MOVOU (R8), X6 \
	FDCTB \
	MOVOU 128(SP), X8 \
	MOVOU 144(SP), X9 \
	MOVOU 160(SP), X10 \
	MOVOU 176(SP), X11 \
	MOVOU 192(SP), X12 \
	MOVOU 208(SP), X13 \
	MOVOU 224(SP), X14 \
	MOVOU 240(SP), X15 \
	qb \
	MOVL BX, (R13) \
	nz \
	IDCT16(slow, add) \
slow: \
	MOVL   $35468, AX \
	MOVQ   AX, X14 \
	PSHUFL $0, X14, X14 \
	MOVL   $85627, AX \
	MOVQ   AX, X15 \
	PSHUFL $0, X15, X15 \
	MOVOU     (SI), X4 \
	MOVOU     16(SI), X6 \
	MOVO      X4, X0 \
	PUNPCKLWL X4, X0 \
	PSRAL     $16, X0 \
	MOVO      X4, X1 \
	PUNPCKHWL X4, X1 \
	PSRAL     $16, X1 \
	MOVO      X6, X2 \
	PUNPCKLWL X6, X2 \
	PSRAL     $16, X2 \
	MOVO      X6, X3 \
	PUNPCKHWL X6, X3 \
	PSRAL     $16, X3 \
	PASS(X0, X1, X2, X3) \
	TRANSPOSE(X0, X1, X2, X3, X4) \
	MOVL   $4, AX \
	MOVQ   AX, X5 \
	PSHUFL $0, X5, X5 \
	PADDL  X5, X0 \
	PASS(X0, X1, X2, X3) \
	PSRAL  $3, X0 \
	PSRAL  $3, X1 \
	PSRAL  $3, X2 \
	PSRAL  $3, X3 \
	TRANSPOSE(X0, X1, X2, X3, X4) \
	PACKSSLW  X1, X0 \
	PACKSSLW  X3, X2 \
add: \
	MOVOU     (R8), X4 \
	PSHUFL    $0x4e, X4, X6 \
	PXOR      X7, X7 \
	PUNPCKLBW X7, X4 \
	PUNPCKLBW X7, X6 \
	PADDSW    X4, X0 \
	PADDSW    X6, X2 \
	PACKUSWB  X2, X0 \
	MOVO      X0, X13 \
	MOVOU     X13, (R12) \
	MOVL       (R9), X4 \
	MOVL       (R9)(R10*1), X5 \
	PUNPCKLLQ  X5, X4 \
	LEAQ       (R9)(R10*2), R11 \
	MOVL       (R11), X6 \
	MOVL       (R11)(R10*1), X7 \
	PUNPCKLLQ  X7, X6 \
	PUNPCKLQDQ X6, X4 \
	MOVO       X13, X5 \
	PSUBUSB    X4, X5 \
	MOVO       X4, X6 \
	PSUBUSB    X13, X6 \
	POR        X6, X5 \
	PXOR       X7, X7 \
	MOVO       X5, X6 \
	PUNPCKLBW  X7, X5 \
	PUNPCKHBW  X7, X6 \
	PMADDWL    X5, X5 \
	PMADDWL    X6, X6 \
	PADDL      X6, X5 \
	PSHUFL     $0x4e, X5, X6 \
	PADDL      X6, X5 \
	PSHUFL     $0xb1, X5, X6 \
	PADDL      X6, X5 \
	MOVQ       X5, AX \
	MOVL       AX, AX \
	MOVL       AX, 4(R13) \
	MOVO       X13, X8 \
	PUNPCKLBW  X7, X8 \
	MOVO       X13, X9 \
	PUNPCKHBW  X7, X9 \
	MOVO       X8, X0 \
	PADDW      X9, X0 \
	MOVO       X8, X1 \
	PSUBW      X9, X1 \
	PSHUFL     $0x4e, X0, X2 \
	PSHUFL     $0x4e, X1, X3 \
	MOVO       X0, X4 \
	PADDW      X2, X4 \
	PSUBW      X2, X0 \
	MOVO       X1, X5 \
	PADDW      X3, X5 \
	PSUBW      X3, X1 \
	MOVO       X4, X8 \
	PUNPCKLQDQ X5, X8 \
	MOVO       X1, X9 \
	PUNPCKLQDQ X0, X9 \
	hp(X8, X0, X1, X2, X3, X10) \
	hp(X9, X4, X5, X6, X7, X10) \
	ab(X0, X10) \
	ab(X1, X10) \
	ab(X2, X10) \
	ab(X3, X10) \
	ab(X4, X10) \
	ab(X5, X10) \
	ab(X6, X10) \
	ab(X7, X10) \
	MOVOU      ttWeights<>+0(SB), X11 \
	PMADDWL    X11, X0 \
	MOVOU      ttWeights<>+16(SB), X11 \
	PMADDWL    X11, X2 \
	MOVOU      ttWeights<>+32(SB), X11 \
	PMADDWL    X11, X3 \
	MOVOU      ttWeights<>+48(SB), X11 \
	PMADDWL    X11, X1 \
	MOVOU      ttWeights<>+64(SB), X11 \
	PMADDWL    X11, X4 \
	MOVOU      ttWeights<>+80(SB), X11 \
	PMADDWL    X11, X6 \
	MOVOU      ttWeights<>+96(SB), X11 \
	PMADDWL    X11, X7 \
	MOVOU      ttWeights<>+112(SB), X11 \
	PMADDWL    X11, X5 \
	PADDL      X1, X0 \
	PADDL      X2, X0 \
	PADDL      X3, X0 \
	PADDL      X4, X0 \
	PADDL      X5, X0 \
	PADDL      X6, X0 \
	PADDL      X7, X0 \
	PSHUFL     $0x4e, X0, X1 \
	PADDL      X1, X0 \
	PSHUFL     $0xb1, X0, X1 \
	PADDL      X1, X0 \
	MOVQ       X0, AX \
	MOVL       AX, AX \
	MOVL       AX, 8(R13) \
	ADDQ $16, R8 \
	ADDQ $32, DI \
	ADDQ $16, R12 \
	ADDQ $16, R13 \
	DECQ R14 \
	JNZ  trial

// func quantAsm(m *quantMatrix, in, out *[16]int16) (last, d int)
// RQ3K copies X to both lanes.
#define RQ3K(X, off) \
	MOVOU X, off(SP) \
	MOVOU X, off+16(SP)

// RQ3P is residualQuantAsm per YMM lane.
// In: Y6 pred, Y13 zero, Y14-Y15 source.
// Out: Y8-Y11 dq, levels; Y0 d; AX mask.
#define RQ3P \
	VPSHUFLW    $0xd8, Y6, Y6 \
	VPSHUFHW    $0xd8, Y6, Y6 \
	VPUNPCKHBW  Y13, Y6, Y5 \
	VPUNPCKLBW  Y13, Y6, Y6 \
	VPSUBW      Y6, Y14, Y2 \
	VPSUBW      Y5, Y15, Y4 \
	VPSHUFHW    $0xb1, Y2, Y2 \
	VPSHUFHW    $0xb1, Y4, Y4 \
	VPUNPCKHQDQ Y4, Y2, Y3 \
	VPUNPCKLQDQ Y4, Y2, Y2 \
	VPSUBW      Y3, Y2, Y5 \
	VPADDW      Y3, Y2, Y2 \
	VPMADDWD    288(SP), Y2, Y6 \
	VPMADDWD    256(SP), Y2, Y2 \
	VPMADDWD    352(SP), Y5, Y3 \
	VPMADDWD    320(SP), Y5, Y5 \
	VPADDD      384(SP), Y5, Y5 \
	VPADDD      416(SP), Y3, Y3 \
	VPSRAD      $9, Y5, Y5 \
	VPSRAD      $9, Y3, Y3 \
	VPACKSSDW   Y6, Y2, Y2 \
	VPACKSSDW   Y3, Y5, Y5 \
	VPUNPCKHWD  Y5, Y2, Y6 \
	VPUNPCKLWD  Y5, Y2, Y2 \
	VPUNPCKHDQ  Y6, Y2, Y3 \
	VPUNPCKLDQ  Y6, Y2, Y2 \
	VPSHUFD     $0x4e, Y3, Y3 \
	VPSUBW      Y3, Y2, Y4 \
	VPADDW      Y3, Y2, Y2 \
	VPUNPCKHQDQ Y4, Y4, Y5 \
	VPUNPCKLWD  Y4, Y5, Y5 \
	VPMADDWD    480(SP), Y5, Y6 \
	VPMADDWD    448(SP), Y5, Y5 \
	VPADDD      512(SP), Y5, Y5 \
	VPADDD      544(SP), Y6, Y6 \
	VPSRAD      $16, Y5, Y5 \
	VPSRAD      $16, Y6, Y6 \
	VPACKSSDW   Y5, Y5, Y5 \
	VPACKSSDW   Y6, Y6, Y6 \
	VPCMPEQW    Y13, Y4, Y7 \
	VPADDW      Y7, Y5, Y5 \
	VPUNPCKHQDQ Y2, Y2, Y3 \
	VPADDW      576(SP), Y2, Y2 \
	VPADDW      Y3, Y2, Y0 \
	VPSUBW      Y3, Y2, Y1 \
	VPSRAW      $4, Y0, Y0 \
	VPSRAW      $4, Y1, Y1 \
	VPUNPCKLQDQ Y5, Y0, Y0 \
	VPUNPCKLQDQ Y6, Y1, Y1 \
	VPSRAW      $15, Y0, Y6 \
	VPSRAW      $15, Y1, Y7 \
	VPABSW      Y0, Y2 \
	VPADDW      192(SP), Y2, Y2 \
	VPABSW      Y1, Y3 \
	VPADDW      224(SP), Y3, Y3 \
	VPMULLW     96(SP), Y2, Y4 \
	VPMULHUW    96(SP), Y2, Y2 \
	VPUNPCKLWD  Y2, Y4, Y5 \
	VPUNPCKHWD  Y2, Y4, Y4 \
	VPADDD      160(SP), Y5, Y5 \
	VPADDD      128(SP), Y4, Y4 \
	VPSRLD      $17, Y5, Y5 \
	VPSRLD      $17, Y4, Y4 \
	VPACKSSDW   Y4, Y5, Y5 \
	VPMULLW     64(SP), Y3, Y4 \
	VPMULHUW    64(SP), Y3, Y3 \
	VPUNPCKLWD  Y3, Y4, Y2 \
	VPUNPCKHWD  Y3, Y4, Y4 \
	VPADDD      128(SP), Y2, Y2 \
	VPADDD      128(SP), Y4, Y4 \
	VPSRLD      $17, Y2, Y2 \
	VPSRLD      $17, Y4, Y4 \
	VPACKSSDW   Y4, Y2, Y2 \
	VPXOR       Y6, Y5, Y5 \
	VPSUBW      Y6, Y5, Y5 \
	VPXOR       Y7, Y2, Y2 \
	VPSUBW      Y7, Y2, Y2 \
	VPMULLW     32(SP), Y5, Y8 \
	VPMULLW     0(SP), Y2, Y9 \
	VPSUBW      Y8, Y0, Y0 \
	VPSUBW      Y9, Y1, Y1 \
	VPMADDWD    Y0, Y0, Y0 \
	VPMADDWD    Y1, Y1, Y1 \
	VPADDD      Y1, Y0, Y0 \
	VPSHUFD     $0x4e, Y0, Y1 \
	VPADDD      Y1, Y0, Y0 \
	VPSHUFD     $0xb1, Y0, Y1 \
	VPADDD      Y1, Y0, Y0 \
	VPSHUFB     608(SP), Y5, Y10 \
	VPSHUFB     640(SP), Y2, Y4 \
	VPOR        Y4, Y10, Y10 \
	VPSHUFB     672(SP), Y2, Y11 \
	VPSHUFB     704(SP), Y5, Y4 \
	VPOR        Y4, Y11, Y11 \
	VPCMPEQW    Y13, Y10, Y4 \
	VPCMPEQW    Y13, Y11, Y5 \
	VPACKSSWB   Y5, Y4, Y4 \
	VPMOVMSKB   Y4, AX \
	NOTL        AX

// func residualQuant3Asm(m *quantMatrix, pr *[10][4]uint32, modes int, src *uint8, stride int, blk, lv *[3][16]int16, res *[3][2]int)
// Frame: QUANTK, FDCTK, FDCTB, zz41 constants.
TEXT ·quantAsm(SB), NOSPLIT, $0-40
	MOVQ m+0(FP), AX
	MOVQ in+8(FP), SI
	MOVQ out+16(FP), DI

	MOVOU (SI), X0
	MOVOU 16(SI), X1
	CMPB  ·useSSE41(SB), $0
	JNE   q41
	QUANT

	MOVQ BX, last+24(FP)
	MOVQ CX, d+32(FP)
	RET

q41:
	QUANT41
	MOVQ BX, last+24(FP)
	MOVQ CX, d+32(FP)
	RET

#define KD(v, X) \
	MOVL $(v), BX \
	MOVQ BX, X \
	PSHUFL $0, X, X

// FDCTK loads FDCTB's constants into X8-X15.
#define FDCTK \
	KD(8|8<<16, X8) \
	KD(8|-8<<16, X9) \
	KD(5352|2217<<16, X10) \
	KD(2217|-5352<<16, X11) \
	KD(1812, X12) \
	KD(937, X13) \
	KD(2217|5352<<16, X14) \
	KD(-5352&0xffff|2217<<16, X15)

// FDCTB is the fdct of src at R9 (stride R10) minus the prediction in
// X6, raster order in X0, X1, with FDCTK's constants.
#define FDCTB \
	MOVL      (R9), X2 \
	MOVL      (R9)(R10*1), X3 \
	LEAQ      (R9)(R10*2), R11 \
	MOVL      (R11), X4 \
	MOVL      (R11)(R10*1), X5 \
	PUNPCKLWL X3, X2 \
	PUNPCKLWL X5, X4 \
	PSHUFLW   $0xd8, X6, X6 \
	PSHUFHW   $0xd8, X6, X6 \
	PXOR      X7, X7 \
	PUNPCKLBW X7, X2 \
	PUNPCKLBW X7, X4 \
	MOVO      X6, X5 \
	PUNPCKLBW X7, X6 \
	PUNPCKHBW X7, X5 \
	PSUBW     X6, X2 \
	PSUBW     X5, X4 \
	PSHUFHW    $0xb1, X2, X2 \
	PSHUFHW    $0xb1, X4, X4 \
	MOVO       X2, X3 \
	PUNPCKLQDQ X4, X2 \
	PUNPCKHQDQ X4, X3 \
	MOVO       X2, X5 \
	PADDW      X3, X2 \
	PSUBW      X3, X5 \
	MOVO       X2, X6 \
	PMADDWL    X8, X2 \
	PMADDWL    X9, X6 \
	MOVO       X5, X3 \
	PMADDWL    X10, X5 \
	PMADDWL    X11, X3 \
	PADDL      X12, X5 \
	PADDL      X13, X3 \
	PSRAL      $9, X5 \
	PSRAL      $9, X3 \
	PACKSSLW   X6, X2 \
	PACKSSLW   X3, X5 \
	MOVO       X2, X6 \
	PUNPCKLWL  X5, X2 \
	PUNPCKHWL  X5, X6 \
	MOVO       X2, X3 \
	PUNPCKLLQ  X6, X2 \
	PUNPCKHLQ  X6, X3 \
	PSHUFL     $0x4e, X3, X3 \
	KD(12000+65536, X8) \
	KD(51000, X9) \
	KD(7|7<<16, X10) \
	MOVO       X2, X4 \
	PSUBW      X3, X4 \
	PADDW      X3, X2 \
	MOVO       X4, X5 \
	PUNPCKHQDQ X5, X5 \
	PUNPCKLWL  X4, X5 \
	MOVO       X5, X6 \
	PMADDWL    X14, X5 \
	PMADDWL    X15, X6 \
	PADDL      X8, X5 \
	PADDL      X9, X6 \
	PSRAL      $16, X5 \
	PSRAL      $16, X6 \
	PACKSSLW   X5, X5 \
	PACKSSLW   X6, X6 \
	PXOR       X7, X7 \
	PCMPEQW    X4, X7 \
	PADDW      X7, X5 \
	MOVO       X2, X3 \
	PUNPCKHQDQ X3, X3 \
	PADDW      X10, X2 \
	MOVO       X2, X1 \
	MOVO       X2, X0 \
	PADDW      X3, X0 \
	PSUBW      X3, X1 \
	PSRAW      $4, X0 \
	PSRAW      $4, X1 \
	PUNPCKLQDQ X5, X0 \
	PUNPCKLQDQ X6, X1

// FDCT6X is FDCTB with its constants.
#define FDCT6X \
	FDCTK \
	FDCTB

// func residualQuantAsm(m *quantMatrix, pr *[4]uint32, src *uint8, stride int, in, out *[16]int16) (last, d int)
TEXT ·residualQuantAsm(SB), NOSPLIT, $0-64
	MOVQ m+0(FP), AX
	MOVQ pr+8(FP), R8
	MOVQ src+16(FP), R9
	MOVQ stride+24(FP), R10
	MOVQ in+32(FP), SI
	MOVQ out+40(FP), DI

	KD(8|8<<16, X8)
	KD(8|-8<<16, X9)
	KD(5352|2217<<16, X10)
	KD(2217|-5352<<16, X11)
	KD(1812, X12)
	KD(937, X13)
	KD(2217|5352<<16, X14)
	KD(-5352&0xffff|2217<<16, X15)

	MOVL      (R9), X2
	MOVL      (R9)(R10*1), X3
	LEAQ      (R9)(R10*2), R11
	MOVL      (R11), X4
	MOVL      (R11)(R10*1), X5
	PUNPCKLWL X3, X2
	PUNPCKLWL X5, X4
	MOVOU     (R8), X6
	PSHUFLW   $0xd8, X6, X6
	PSHUFHW   $0xd8, X6, X6
	PXOR      X7, X7
	PUNPCKLBW X7, X2
	PUNPCKLBW X7, X4
	MOVO      X6, X5
	PUNPCKLBW X7, X6
	PUNPCKHBW X7, X5
	PSUBW     X6, X2
	PSUBW     X5, X4

	PSHUFHW    $0xb1, X2, X2
	PSHUFHW    $0xb1, X4, X4
	MOVO       X2, X3
	PUNPCKLQDQ X4, X2
	PUNPCKHQDQ X4, X3
	MOVO       X2, X5
	PADDW      X3, X2
	PSUBW      X3, X5
	MOVO       X2, X6
	PMADDWL    X8, X2
	PMADDWL    X9, X6
	MOVO       X5, X3
	PMADDWL    X10, X5
	PMADDWL    X11, X3
	PADDL      X12, X5
	PADDL      X13, X3
	PSRAL      $9, X5
	PSRAL      $9, X3
	PACKSSLW   X6, X2
	PACKSSLW   X3, X5
	MOVO       X2, X6
	PUNPCKLWL  X5, X2
	PUNPCKHWL  X5, X6
	MOVO       X2, X3
	PUNPCKLLQ  X6, X2
	PUNPCKHLQ  X6, X3
	PSHUFL     $0x4e, X3, X3

	KD(12000+65536, X8)
	KD(51000, X9)
	KD(7|7<<16, X10)
	MOVO       X2, X4
	PSUBW      X3, X4
	PADDW      X3, X2
	MOVO       X4, X5
	PUNPCKHQDQ X5, X5
	PUNPCKLWL  X4, X5
	MOVO       X5, X6
	PMADDWL    X14, X5
	PMADDWL    X15, X6
	PADDL      X8, X5
	PADDL      X9, X6
	PSRAL      $16, X5
	PSRAL      $16, X6
	PACKSSLW   X5, X5
	PACKSSLW   X6, X6
	PXOR       X7, X7
	PCMPEQW    X4, X7
	PADDW      X7, X5
	MOVO       X2, X3
	PUNPCKHQDQ X3, X3
	PADDW      X10, X2
	MOVO       X2, X1
	MOVO       X2, X0
	PADDW      X3, X0
	PSUBW      X3, X1
	PSRAW      $4, X0
	PSRAW      $4, X1
	PUNPCKLQDQ X5, X0
	PUNPCKLQDQ X6, X1

	CMPB ·useSSE41(SB), $0
	JNE  rq41
	QUANT

	MOVQ BX, last+48(FP)
	MOVQ CX, d+56(FP)
	RET

rq41:
	QUANT41
	MOVQ BX, last+48(FP)
	MOVQ CX, d+56(FP)
	RET

TEXT ·residualQuant3Asm(SB), 0, $736-64
	MOVQ m+0(FP), AX
	QUANTK
	RQ3K(X8, 0)
	RQ3K(X9, 32)
	RQ3K(X10, 64)
	RQ3K(X11, 96)
	RQ3K(X12, 128)
	RQ3K(X13, 160)
	RQ3K(X14, 192)
	RQ3K(X15, 224)
	FDCTK
	RQ3K(X8, 256)
	RQ3K(X9, 288)
	RQ3K(X10, 320)
	RQ3K(X11, 352)
	RQ3K(X12, 384)
	RQ3K(X13, 416)
	RQ3K(X14, 448)
	RQ3K(X15, 480)
	KD(12000+65536, X8)
	KD(51000, X9)
	KD(7|7<<16, X10)
	RQ3K(X8, 512)
	RQ3K(X9, 544)
	RQ3K(X10, 576)
	MOVOU zz41<>+0(SB), X0
	RQ3K(X0, 608)
	MOVOU zz41<>+16(SB), X0
	RQ3K(X0, 640)
	MOVOU zz41<>+32(SB), X0
	RQ3K(X0, 672)
	MOVOU zz41<>+48(SB), X0
	RQ3K(X0, 704)

	MOVQ      src+24(FP), R9
	MOVQ      stride+32(FP), R10
	MOVL      (R9), X2
	MOVL      (R9)(R10*1), X3
	LEAQ      (R9)(R10*2), R11
	MOVL      (R11), X4
	MOVL      (R11)(R10*1), X5
	PUNPCKLWL X3, X2
	PUNPCKLWL X5, X4
	PXOR      X7, X7
	PUNPCKLBW X7, X2
	PUNPCKLBW X7, X4
	VINSERTI128 $1, X2, Y2, Y14
	VINSERTI128 $1, X4, Y4, Y15
	VPXOR       Y13, Y13, Y13

	MOVQ pr+8(FP), R8
	MOVQ modes+16(FP), CX
	MOVQ blk+40(FP), SI
	MOVQ lv+48(FP), DI
	MOVQ res+56(FP), R13
	MOVQ $-1, DX

	// First two modes in lanes 0, 1.
	MOVQ        CX, BX
	ANDQ        $15, BX
	SHLQ        $4, BX
	VMOVDQU     (R8)(BX*1), X6
	MOVQ        CX, BX
	SHRQ        $4, BX
	ANDQ        $15, BX
	SHLQ        $4, BX
	VINSERTI128 $1, (R8)(BX*1), Y6, Y6
	RQ3P
	VMOVDQU      X8, (SI)
	VMOVDQU      X9, 16(SI)
	VEXTRACTI128 $1, Y8, 32(SI)
	VEXTRACTI128 $1, Y9, 48(SI)
	VMOVDQU      X10, (DI)
	VMOVDQU      X11, 16(DI)
	VEXTRACTI128 $1, Y10, 32(DI)
	VEXTRACTI128 $1, Y11, 48(DI)
	VMOVD        X0, BX
	MOVQ         BX, 8(R13)
	VEXTRACTI128 $1, Y0, X1
	VMOVD        X1, BX
	MOVQ         BX, 24(R13)
	MOVL         AX, BX
	ANDL         $0xffff, BX
	BSRQ         BX, BX
	CMOVQEQ      DX, BX
	MOVQ         BX, (R13)
	SHRL         $16, AX
	BSRQ         AX, BX
	CMOVQEQ      DX, BX
	MOVQ         BX, 16(R13)

	// Third mode in lane 0.
	SHRQ    $8, CX
	SHLQ    $4, CX
	VMOVDQU (R8)(CX*1), X6
	RQ3P
	VMOVDQU X8, 64(SI)
	VMOVDQU X9, 80(SI)
	VMOVDQU X10, 64(DI)
	VMOVDQU X11, 80(DI)
	VMOVD   X0, BX
	MOVQ    BX, 40(R13)
	ANDL    $0xffff, AX
	BSRQ    AX, BX
	CMOVQEQ DX, BX
	MOVQ    BX, 32(R13)
	VZEROUPPER
	RET

// func residualAsm(pred *uint8, src *uint8, stride int, out *[16]int16)
TEXT ·residualAsm(SB), NOSPLIT, $0-32
	MOVQ      pred+0(FP), R8
	MOVQ      src+8(FP), R9
	MOVQ      stride+16(FP), R10
	MOVQ      out+24(FP), DI
	MOVL      (R8), X6
	MOVL      32(R8), X0
	MOVL      64(R8), X1
	MOVL      96(R8), X2
	PUNPCKLLQ X0, X6
	PUNPCKLLQ X2, X1
	PUNPCKLQDQ X1, X6
	FDCT6X
	MOVOU X0, (DI)
	MOVOU X1, 16(DI)
	RET

// func levelCostsAsm(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int)
// R14 walks the 1024-byte position rows, R9 is the row plus ctx*256.
TEXT ·levelCostsAsm(SB), NOSPLIT, $0-56
	MOVQ    t+0(FP), R14
	MOVQ    lv+8(FP), R8
	MOVQ    first+16(FP), DI
	MOVQ    last+24(FP), SI
	MOVQ    ctx+32(FP), R10
	MOVQ    DI, AX
	SHLQ    $10, AX
	ADDQ    AX, R14
	SHLQ    $8, R10
	LEAQ    (R14)(R10*1), R9
	XORQ    R12, R12
	MOVQ    $512, R13
	LEAQ    ·fixedCost(SB), DX

lcloop:
	MOVWQSX (R8)(DI*2), AX
	MOVQ    AX, BX
	SARQ    $63, BX
	XORQ    BX, AX
	SUBQ    BX, AX
	CMPQ    AX, $67
	JHI     lcbig

lccost:
	MOVWLZX (R9)(AX*2), R11
	ADDQ    R11, R12
	MOVQ    AX, R10
	SHLQ    $8, R10
	CMPQ    R10, R13
	CMOVQHI R13, R10
	ADDQ    $1024, R14
	LEAQ    (R14)(R10*1), R9
	INCQ    DI
	CMPQ    DI, SI
	JLE     lcloop
	SHRQ    $8, R10
	MOVQ    R12, c+40(FP)
	MOVQ    R10, next+48(FP)
	RET

	// Level above 67: its fixed cost beyond fixedCost[67], then row 67.
lcbig:
	ANDQ    $2047, AX
	ADDQ    (DX)(AX*8), R12
	SUBQ    536(DX), R12
	MOVQ    $67, AX
	JMP     lccost

// func sse4x10Asm(d *[10]int32, pr *[10][4]uint32, src *uint8, stride int)
TEXT ·sse4x10Asm(SB), NOSPLIT, $0-32
	MOVQ d+0(FP), DI
	MOVQ pr+8(FP), SI
	MOVQ src+16(FP), AX
	MOVQ stride+24(FP), BX
	PXOR X7, X7
	MOVL       (AX), X1
	MOVL       (AX)(BX*1), X2
	PUNPCKLLQ  X2, X1
	LEAQ       (AX)(BX*2), AX
	MOVL       (AX), X3
	MOVL       (AX)(BX*1), X4
	PUNPCKLLQ  X4, X3
	PUNPCKLQDQ X3, X1
	MOVO  X1, X2
	PUNPCKLBW X7, X1
	PUNPCKHBW X7, X2
	MOVQ  $10, CX

loop:
	MOVOU     (SI), X3
	MOVO      X3, X4
	PUNPCKLBW X7, X3
	PUNPCKHBW X7, X4
	PSUBW     X1, X3
	PSUBW     X2, X4
	PMADDWL   X3, X3
	PMADDWL   X4, X4
	PADDL     X4, X3
	PSHUFL    $0x4e, X3, X4
	PADDL     X4, X3
	PSHUFL    $0xb1, X3, X4
	PADDL     X4, X3
	MOVL      X3, (DI)
	ADDQ      $16, SI
	ADDQ      $4, DI
	DECQ      CX
	JNZ       loop
	RET

// func sseAsm(src *uint8, stride int, pred *uint8, w, h int) int
TEXT ·sseAsm(SB), NOSPLIT, $0-48
	MOVQ src+0(FP), SI
	MOVQ stride+8(FP), BX
	MOVQ pred+16(FP), DI
	MOVQ w+24(FP), AX
	MOVQ h+32(FP), CX
	PXOR X7, X7
	PXOR X6, X6
	CMPQ AX, $16
	JNE  row8

row16:
	MOVOU     (SI), X0
	MOVOU     (DI), X1
	// |a-b| from two saturating subtracts.
	MOVO      X0, X2
	PSUBUSB   X1, X2
	PSUBUSB   X0, X1
	POR       X1, X2
	MOVO      X2, X3
	PUNPCKLBW X7, X2
	PUNPCKHBW X7, X3
	PMADDWL   X2, X2
	PMADDWL   X3, X3
	PADDL     X2, X6
	PADDL     X3, X6
	ADDQ      BX, SI
	ADDQ      $32, DI
	DECQ      CX
	JNZ       row16
	JMP       done

row8:
	MOVQ      (SI), X0
	MOVQ      (DI), X1
	PUNPCKLBW X7, X0
	PUNPCKLBW X7, X1
	PSUBW     X1, X0
	PMADDWL   X0, X0
	PADDL     X0, X6
	ADDQ      BX, SI
	ADDQ      $32, DI
	DECQ      CX
	JNZ       row8

done:
	PSHUFL $0x4e, X6, X1
	PADDL  X1, X6
	PSHUFL $0xb1, X6, X1
	PADDL  X1, X6
	MOVL   X6, AX
	MOVQ   AX, ret+40(FP)
	RET

// Adds (X3-X4)^2 pairs into acc.
#define ESSE(acc) \
	PSUBW   X4, X3 \
	PMADDWL X3, X3 \
	PADDL   X3, acc

// Sums acc into the int at off(R8).
#define EHSUM(acc, off) \
	PSHUFL $0x4e, acc, X3 \
	PADDL  X3, acc \
	PSHUFL $0xb1, acc, X3 \
	PADDL  X3, acc \
	MOVL   acc, AX \
	MOVQ   AX, off(R8)

// Adds the left byte at off(DI) to R10.
#define LADD(off) \
	MOVBLZX off(DI), R11 \
	ADDQ    R11, R10

// Sets AX to dcAvg of the n x n block at DI, n in CX and
// edges in DX, bit 0 left and bit 1 top; uses X0, X1, R9-R11.
#define DCAVG \
	MOVBLZX -1(DI), R10 \
	LADD(31) \
	LADD(63) \
	LADD(95) \
	LADD(127) \
	LADD(159) \
	LADD(191) \
	LADD(223) \
	PXOR    X1, X1 \
	CMPQ    CX, $16 \
	JNE     dctop8 \
	LADD(255) \
	LADD(287) \
	LADD(319) \
	LADD(351) \
	LADD(383) \
	LADD(415) \
	LADD(447) \
	LADD(479) \
	MOVOU   -32(DI), X0 \
	PSADBW  X1, X0 \
	PSHUFL  $0x4e, X0, X1 \
	PADDQ   X1, X0 \
	JMP     dctop \
dctop8: \
	MOVQ    -32(DI), X0 \
	PSADBW  X1, X0 \
dctop: \
	MOVQ    X0, R9 \
	MOVQ    DX, R11 \
	ANDQ    $1, R11 \
	NEGQ    R11 \
	ANDQ    R11, R10 \
	MOVQ    DX, R11 \
	SHRQ    $1, R11 \
	NEGQ    R11 \
	ANDQ    R11, R9 \
	LEAQ    (R9)(R10*1), AX \
	MOVQ    CX, R11 \
	SHRQ    $4, CX \
	ADDQ    $2, CX \
	MOVQ    DX, R9 \
	ANDQ    $1, R9 \
	ADDQ    R9, CX \
	MOVQ    DX, R9 \
	SHRQ    $1, R9 \
	ADDQ    R9, CX \
	MOVL    $1, R9 \
	SHLQ    CX, R9 \
	SHRQ    $1, R9 \
	ADDQ    R9, AX \
	SHRQ    CX, AX \
	MOVQ    R11, CX \
	MOVL    $128, R9 \
	TESTQ   DX, DX \
	CMOVQEQ R9, AX

// func edgeSSEAsm(d *[4]int, src *uint8, stride int, p *uint8, n, edges int)
TEXT ·edgeSSEAsm(SB), NOSPLIT, $0-48
	MOVQ      d+0(FP), R8
	MOVQ      src+8(FP), SI
	MOVQ      stride+16(FP), BX
	MOVQ      p+24(FP), DI
	MOVQ      n+32(FP), CX
	MOVQ      edges+40(FP), DX
	DCAVG
	PXOR      X5, X5
	MOVQ      AX, X10
	PSHUFLW   $0, X10, X10
	PSHUFL    $0, X10, X10
	MOVL      $255, AX
	MOVQ      AX, X11
	PSHUFLW   $0, X11, X11
	PSHUFL    $0, X11, X11
	MOVBLZX   -33(DI), AX
	MOVQ      AX, X3
	PSHUFLW   $0, X3, X3
	PSHUFL    $0, X3, X3
	MOVOU     -32(DI), X6
	MOVO      X6, X7
	PUNPCKLBW X5, X6
	PUNPCKHBW X5, X7
	// top - corner fits int16.
	MOVO      X6, X8
	MOVO      X7, X9
	PSUBW     X3, X8
	PSUBW     X3, X9
	PXOR      X12, X12
	PXOR      X13, X13
	PXOR      X14, X14
	PXOR      X15, X15
	CMPQ      CX, $16
	JNE       erow8

erow16:
	MOVOU     (SI), X0
	MOVO      X0, X1
	PUNPCKLBW X5, X0
	PUNPCKHBW X5, X1
	MOVBLZX   -1(DI), AX
	MOVQ      AX, X2
	PSHUFLW   $0, X2, X2
	PSHUFL    $0, X2, X2
	MOVO      X10, X3
	MOVO      X0, X4
	ESSE(X12)
	MOVO      X10, X3
	MOVO      X1, X4
	ESSE(X12)
	// clip8 of L - A + T in [-255, 510].
	MOVO      X2, X3
	PADDW     X8, X3
	PMAXSW    X5, X3
	PMINSW    X11, X3
	MOVO      X0, X4
	ESSE(X13)
	MOVO      X2, X3
	PADDW     X9, X3
	PMAXSW    X5, X3
	PMINSW    X11, X3
	MOVO      X1, X4
	ESSE(X13)
	MOVO      X6, X3
	MOVO      X0, X4
	ESSE(X14)
	MOVO      X7, X3
	MOVO      X1, X4
	ESSE(X14)
	MOVO      X2, X3
	MOVO      X0, X4
	ESSE(X15)
	MOVO      X2, X3
	MOVO      X1, X4
	ESSE(X15)
	ADDQ      BX, SI
	ADDQ      $32, DI
	DECQ      CX
	JNZ       erow16
	JMP       edone

erow8:
	MOVQ      (SI), X0
	PUNPCKLBW X5, X0
	MOVBLZX   -1(DI), AX
	MOVQ      AX, X2
	PSHUFLW   $0, X2, X2
	PSHUFL    $0, X2, X2
	MOVO      X10, X3
	MOVO      X0, X4
	ESSE(X12)
	MOVO      X2, X3
	PADDW     X8, X3
	PMAXSW    X5, X3
	PMINSW    X11, X3
	MOVO      X0, X4
	ESSE(X13)
	MOVO      X6, X3
	MOVO      X0, X4
	ESSE(X14)
	MOVO      X2, X3
	MOVO      X0, X4
	ESSE(X15)
	ADDQ      BX, SI
	ADDQ      $32, DI
	DECQ      CX
	JNZ       erow8

edone:
	EHSUM(X12, 0)
	EHSUM(X13, 8)
	EHSUM(X14, 16)
	EHSUM(X15, 24)
	RET

// func predTMAsm(dst *uint8, n int)
TEXT ·predTMAsm(SB), NOSPLIT, $0-16
	MOVQ      dst+0(FP), DI
	MOVQ      n+8(FP), CX
	PXOR      X7, X7
	MOVBLZX   -33(DI), AX
	MOVQ      AX, X3
	PSHUFLW   $0, X3, X3
	PSHUFL    $0, X3, X3

	// top - corner fits int16.
	MOVOU     -32(DI), X1
	MOVO      X1, X2
	PUNPCKLBW X7, X1
	PUNPCKHBW X7, X2
	PSUBW     X3, X1
	PSUBW     X3, X2
	CMPQ      CX, $16
	JNE       row8

row16:
	MOVBLZX  -1(DI), AX
	MOVQ     AX, X4
	PSHUFLW  $0, X4, X4
	PSHUFL   $0, X4, X4
	MOVO     X1, X5
	MOVO     X2, X6
	PADDW    X4, X5
	PADDW    X4, X6
	// Saturating pack is clip8.
	PACKUSWB X6, X5
	MOVOU    X5, (DI)
	ADDQ     $32, DI
	DECQ     CX
	JNZ      row16
	RET

row8:
	MOVBLZX  -1(DI), AX
	MOVQ     AX, X4
	PSHUFLW  $0, X4, X4
	PSHUFL   $0, X4, X4
	MOVO     X1, X5
	PADDW    X4, X5
	PACKUSWB X5, X5
	MOVQ     X5, (DI)
	ADDQ     $32, DI
	DECQ     CX
	JNZ      row8
	RET

// func predDCAsm(dst *uint8, n, edges int)
TEXT ·predDCAsm(SB), NOSPLIT, $0-24
	MOVQ       dst+0(FP), DI
	MOVQ       n+8(FP), CX
	MOVQ       edges+16(FP), DX
	DCAVG
	MOVQ       $0x0101010101010101, R9
	IMULQ      R9, AX
	CMPQ       CX, $16
	JNE        dc8
	MOVQ       AX, X0
	PUNPCKLQDQ X0, X0
	MOVOU      X0, (DI)
	MOVOU      X0, 32(DI)
	MOVOU      X0, 64(DI)
	MOVOU      X0, 96(DI)
	MOVOU      X0, 128(DI)
	MOVOU      X0, 160(DI)
	MOVOU      X0, 192(DI)
	MOVOU      X0, 224(DI)
	MOVOU      X0, 256(DI)
	MOVOU      X0, 288(DI)
	MOVOU      X0, 320(DI)
	MOVOU      X0, 352(DI)
	MOVOU      X0, 384(DI)
	MOVOU      X0, 416(DI)
	MOVOU      X0, 448(DI)
	MOVOU      X0, 480(DI)
	RET

dc8:
	MOVQ AX, (DI)
	MOVQ AX, 32(DI)
	MOVQ AX, 64(DI)
	MOVQ AX, 96(DI)
	MOVQ AX, 128(DI)
	MOVQ AX, 160(DI)
	MOVQ AX, 192(DI)
	MOVQ AX, 224(DI)
	RET

// func predHEAsm(dst *uint8, n int)
TEXT ·predHEAsm(SB), NOSPLIT, $0-16
	MOVQ $0x0101010101010101, R9
	MOVQ dst+0(FP), DI
	MOVQ n+8(FP), CX
	CMPQ CX, $16
	JNE  he8

he16:
	MOVBLZX -1(DI), AX
	IMULQ   R9, AX
	MOVQ    AX, (DI)
	MOVQ    AX, 8(DI)
	ADDQ    $32, DI
	DECQ    CX
	JNZ     he16
	RET

he8:
	MOVBLZX -1(DI), AX
	IMULQ   R9, AX
	MOVQ    AX, (DI)
	ADDQ    $32, DI
	DECQ    CX
	JNZ     he8
	RET


// pred4All masks: V, ones, DC, 4, TM, modes.
DATA p4k<>+0(SB)/8, $0x020100090a0b0c0c
DATA p4k<>+8(SB)/8, $0x8008080706050403
DATA p4k<>+16(SB)/8, $0x0101010101010101
DATA p4k<>+24(SB)/8, $0x0101010101010101
DATA p4k<>+32(SB)/8, $0x0908070604030201
DATA p4k<>+40(SB)/8, $0x8080808080808080
DATA p4k<>+48(SB)/8, $0x0000000000000004
DATA p4k<>+56(SB)/8, $0x0000000000000000
DATA p4k<>+64(SB)/8, $0x8009800880078006
DATA p4k<>+72(SB)/8, $0x8009800880078006
DATA p4k<>+80(SB)/8, $0x8005800580058005
DATA p4k<>+88(SB)/8, $0x8005800580058005
DATA p4k<>+96(SB)/8, $0x8004800480048004
DATA p4k<>+104(SB)/8, $0x8003800380038003
DATA p4k<>+112(SB)/8, $0x8002800280028002
DATA p4k<>+120(SB)/8, $0x8001800180018001
DATA p4k<>+128(SB)/8, $0x0807060508070605
DATA p4k<>+136(SB)/8, $0x0807060508070605
DATA p4k<>+144(SB)/8, $0x0202020203030303
DATA p4k<>+152(SB)/8, $0x0000000001010101
DATA p4k<>+160(SB)/8, $0x0605040307060504
DATA p4k<>+168(SB)/8, $0x0403020105040302
DATA p4k<>+176(SB)/8, $0x0a09080709080706
DATA p4k<>+184(SB)/8, $0x0c0b0a090b0a0908
DATA p4k<>+192(SB)/8, $0x0706050480808080
DATA p4k<>+200(SB)/8, $0x0605040280808003
DATA p4k<>+208(SB)/8, $0x8080808008070605
DATA p4k<>+216(SB)/8, $0x8080808007060580
DATA p4k<>+224(SB)/8, $0x0908070680808080
DATA p4k<>+232(SB)/8, $0x0b0908070a808080
DATA p4k<>+240(SB)/8, $0x8080808009080706
DATA p4k<>+248(SB)/8, $0x8080808080090807
DATA p4k<>+256(SB)/8, $0x0480038006050480
DATA p4k<>+264(SB)/8, $0x0280018003800280
DATA p4k<>+272(SB)/8, $0x8004800380808004
DATA p4k<>+280(SB)/8, $0x8002800180038002
DATA p4k<>+288(SB)/8, $0x0080018001800280
DATA p4k<>+296(SB)/8, $0x8080808080800080
DATA p4k<>+304(SB)/8, $0x8001800280028003
DATA p4k<>+312(SB)/8, $0x0000000000008001
GLOBL p4k<>(SB), RODATA, $320

// Two-source mode: A3 and A2 picks ORed.
#define P4PAIR(off, dst) \
	MOVO   X4, X5 \
	PSHUFB p4k<>+off(SB), X5 \
	MOVO   X3, X6 \
	PSHUFB p4k<>+16+off(SB), X6 \
	POR    X6, X5 \
	MOVOU  X5, dst(DI)

// func pred4AllAsm(pr *[10][4]uint32, p *uint8)
TEXT ·pred4AllAsm(SB), NOSPLIT, $0-16
	MOVQ   pr+0(FP), DI
	MOVQ   p+8(FP), SI
	// A, T0..T7, then L0..L3 in bytes 9..12.
	// Bytes 0..8 only: skips the fresh put4 of block n-1.
	MOVQ   -33(SI), X0
	PINSRB $8, -25(SI), X0
	PINSRB $9, -1(SI), X0
	PINSRB $10, 31(SI), X0
	PINSRB $11, 63(SI), X0
	PINSRB $12, 95(SI), X0
	// V = L3 L3 L2 L1 L0 A T0..T7 T7.
	PSHUFB p4k<>+0(SB), X0
	MOVO   X0, X1
	PSRLDQ $1, X1
	MOVO   X0, X2
	PSRLDQ $2, X2
	// A2[i] = avg2(V[i], V[i+1]).
	MOVO   X0, X3
	PAVGB  X1, X3
	// A3[i] = avg3(V[i], V[i+1], V[i+2]).
	MOVO   X0, X4
	PAVGB  X2, X4
	MOVO   X0, X5
	PXOR   X2, X5
	PAND   p4k<>+16(SB), X5
	PSUBB  X5, X4
	PAVGB  X1, X4

	// DC
	MOVO   X0, X5
	PSHUFB p4k<>+32(SB), X5
	PXOR   X6, X6
	PSADBW X6, X5
	PADDW  p4k<>+48(SB), X5
	PSRLW  $3, X5
	PSHUFB X6, X5
	MOVOU  X5, 0(DI)

	// TM: L - A + T fits int16.
	MOVO     X0, X5
	PSHUFB   p4k<>+64(SB), X5
	MOVO     X0, X6
	PSHUFB   p4k<>+80(SB), X6
	PSUBW    X6, X5
	MOVO     X0, X6
	PSHUFB   p4k<>+96(SB), X6
	PADDW    X5, X6
	MOVO     X0, X7
	PSHUFB   p4k<>+112(SB), X7
	PADDW    X5, X7
	// Saturating pack is clip8.
	PACKUSWB X7, X6
	MOVOU    X6, 16(DI)

	// VE, HE, RD, LD
	MOVO   X4, X5
	PSHUFB p4k<>+128(SB), X5
	MOVOU  X5, 32(DI)
	MOVO   X4, X5
	PSHUFB p4k<>+144(SB), X5
	MOVOU  X5, 48(DI)
	MOVO   X4, X5
	PSHUFB p4k<>+160(SB), X5
	MOVOU  X5, 64(DI)
	MOVO   X4, X5
	PSHUFB p4k<>+176(SB), X5
	MOVOU  X5, 96(DI)

	// VR, VL, HD, HU
	P4PAIR(192, 80)
	P4PAIR(224, 112)
	P4PAIR(256, 128)
	P4PAIR(288, 144)
	RET


// func lumaAsm(dst, src *uint8, n int)
TEXT ·lumaAsm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	TESTQ CX, CX
	JZ   done
	MOVL $0xff, AX
	MOVQ AX, X12
	PSHUFL $0, X12, X12
	MOVL $0xff0000, AX
	MOVQ AX, X13
	PSHUFL $0, X13, X13
	MOVL $(16675<<16 | 16839), AX
	MOVQ AX, X14
	PSHUFL $0, X14, X14
	MOVL $(6420<<16 | 16384), AX
	MOVQ AX, X15
	PSHUFL $0, X15, X15
	MOVL $(1<<15 + 16<<16), AX
	MOVQ AX, X11
	PSHUFL $0, X11, X11

#define Y4(off, R) \
	MOVOU off(SI), R \
	MOVO  R, X1 \
	PSLLL $8, X1 \
	PAND  X13, X1 \
	MOVO  R, X2 \
	PAND  X12, X2 \
	POR   X1, X2 \
	MOVO  R, X3 \
	PSRLL $8, X3 \
	PAND  X12, X3 \
	PAND  X13, R \
	POR   X3, R \
	PMADDWL X14, X2 \
	PMADDWL X15, R \
	PADDL X2, R \
	PADDL X11, R \
	PSRLL $16, R

loop:
	Y4(0, X0)
	Y4(16, X4)
	Y4(32, X5)
	Y4(48, X6)
	PACKSSLW X4, X0
	PACKSSLW X6, X5
	PACKUSWB X5, X0
	MOVOU X0, (DI)
	ADDQ $64, SI
	ADDQ $16, DI
	SUBQ $16, CX
	JNZ  loop
done:
	RET

// Frees the register name for later code.
#undef Y4

// Go-wrapped x*k>>16 per int32 lane.
#define MULHI(x, k, t1, t2) \
	MOVO      x, t1        \
	PMULULQ   k, t1        \
	MOVO      x, t2        \
	PSRLQ     $32, t2      \
	PMULULQ   k, t2        \
	PSHUFL    $0x08, t1, t1 \
	PSHUFL    $0x08, t2, t2 \
	PUNPCKLLQ t2, t1       \
	PSRAL     $16, t1

// 4x4 int32 transpose, clobbers t.
#define TRANSPOSE(a, b, c, d, t) \
	MOVO       a, t   \
	PUNPCKLLQ  b, a   \
	PUNPCKHLQ  b, t   \
	MOVO       c, b   \
	PUNPCKLLQ  d, c   \
	PUNPCKHLQ  d, b   \
	MOVO       a, d   \
	PUNPCKLQDQ c, a   \
	PUNPCKHQDQ c, d   \
	MOVO       t, c   \
	PUNPCKLQDQ b, t   \
	PUNPCKHQDQ b, c   \
	MOVO       d, b   \
	MOVO       c, d   \
	MOVO       t, c

// 1-D pass in place, uses X8-X12.
#define PASS(a, b, c, d) \
	MOVO  a, X8           \
	PADDL c, a            \
	PSUBL c, X8           \
	MULHI(b, X14, X9, X12) \
	MULHI(d, X15, X10, X12) \
	PSUBL X10, X9         \
	MULHI(b, X15, X10, X12) \
	MULHI(d, X14, X11, X12) \
	PADDL X11, X10        \
	MOVO  a, d            \
	PADDL X10, a          \
	PSUBL X10, d          \
	MOVO  X8, b           \
	PADDL X9, b           \
	MOVO  X8, c           \
	PSUBL X9, c

// IPASS16 is one inverse DCT pass on int16 rows r0-r3 in place, as
// libwebp's Transform_SSE2: (x*K)>>16 is mulhi(x, K-65536)+x, with k1
// in X14 and k2 in X15. Uses X4-X7.
#define IPASS16(r0, r1, r2, r3) \
	MOVO   r0, X4 \
	PADDW  r2, r0 \
	PSUBW  r2, X4 \
	MOVO   r1, X5 \
	PMULHW X15, X5 \
	MOVO   r3, X6 \
	PMULHW X14, X6 \
	PSUBW  X6, X5 \
	MOVO   r1, X6 \
	PSUBW  r3, X6 \
	PADDW  X6, X5 \
	MOVO   r1, X6 \
	PMULHW X14, X6 \
	MOVO   r3, X7 \
	PMULHW X15, X7 \
	PADDW  X7, X6 \
	PADDW  r3, r1 \
	PADDW  X6, r1 \
	MOVO   r0, r3 \
	PSUBW  r1, r3 \
	PADDW  r1, r0 \
	MOVO   X4, r2 \
	PSUBW  X5, r2 \
	MOVO   X4, r1 \
	PADDW  X5, r1

// TRANSPOSE16 transposes the int16 4x4 in the low halves of r0-r3,
// leaving columns 0|1 in r0 and 2|3 in r2.
#define TRANSPOSE16(r0, r1, r2, r3) \
	PUNPCKLWL r1, r0 \
	PUNPCKLWL r3, r2 \
	MOVO      r0, r1 \
	PUNPCKLLQ r2, r0 \
	PUNPCKHLQ r2, r1 \
	MOVO      r1, r2 \
	PSHUFL    $0x4e, r0, r1 \
	PSHUFL    $0x4e, r2, r3

// IDCT16 is the inverse DCT of (SI) into int16 rows 0|1 in X0 and 2|3
// in X2, then a jump to done. It jumps to slow unless every level is in
// [-2048, 2047], which keeps each intermediate exact in int16.
#define IDCT16(slow, done) \
	MOVOU    (SI), X0 \
	MOVOU    16(SI), X2 \
	KD(2048|2048<<16, X5) \
	MOVO     X0, X4 \
	PADDW    X5, X4 \
	MOVO     X2, X6 \
	PADDW    X5, X6 \
	POR      X6, X4 \
	PSRLW    $12, X4 \
	PXOR     X7, X7 \
	PCMPEQW  X7, X4 \
	PMOVMSKB X4, AX \
	CMPL     AX, $0xffff \
	JNE      slow \
	KD(20091|20091<<16, X14) \
	KD(-30068&0xffff|-30068<<16, X15) \
	PSHUFL   $0x4e, X0, X1 \
	PSHUFL   $0x4e, X2, X3 \
	IPASS16(X0, X1, X2, X3) \
	TRANSPOSE16(X0, X1, X2, X3) \
	KD(4|4<<16, X7) \
	PADDW    X7, X0 \
	IPASS16(X0, X1, X2, X3) \
	PSRAW    $3, X0 \
	PSRAW    $3, X1 \
	PSRAW    $3, X2 \
	PSRAW    $3, X3 \
	TRANSPOSE16(X0, X1, X2, X3) \
	JMP      done

// func idctAddAsm(c *[16]int16, dst *uint8)
TEXT ·idctAddAsm(SB), NOSPLIT, $0-16
	MOVQ   c+0(FP), SI
	MOVQ   dst+8(FP), DI
	IDCT16(ia32, iaadd)

ia32:
	MOVL   $35468, AX
	MOVQ   AX, X14
	PSHUFL $0, X14, X14
	MOVL   $85627, AX
	MOVQ   AX, X15
	PSHUFL $0, X15, X15

	// Sign-extend rows to int32.
	MOVOU     (SI), X4
	MOVOU     16(SI), X6
	MOVO      X4, X0
	PUNPCKLWL X4, X0
	PSRAL     $16, X0
	MOVO      X4, X1
	PUNPCKHWL X4, X1
	PSRAL     $16, X1
	MOVO      X6, X2
	PUNPCKLWL X6, X2
	PSRAL     $16, X2
	MOVO      X6, X3
	PUNPCKHWL X6, X3
	PSRAL     $16, X3

	PASS(X0, X1, X2, X3)
	TRANSPOSE(X0, X1, X2, X3, X4)
	MOVL   $4, AX
	MOVQ   AX, X5
	PSHUFL $0, X5, X5
	PADDL  X5, X0
	PASS(X0, X1, X2, X3)
	PSRAL  $3, X0
	PSRAL  $3, X1
	PSRAL  $3, X2
	PSRAL  $3, X3
	TRANSPOSE(X0, X1, X2, X3, X4)

	// Saturation equals clip8 of int32 sum.
	PACKSSLW  X1, X0
	PACKSSLW  X3, X2

iaadd:
	MOVL      (DI), X4
	MOVL      32(DI), X5
	PUNPCKLLQ X5, X4
	MOVL      64(DI), X6
	MOVL      96(DI), X7
	PUNPCKLLQ X7, X6
	PXOR      X7, X7
	PUNPCKLBW X7, X4
	PUNPCKLBW X7, X6
	PADDSW    X4, X0
	PADDSW    X6, X2
	PACKUSWB  X2, X0
	MOVQ      X0, AX
	MOVL      AX, (DI)
	SHRQ      $32, AX
	MOVL      AX, 32(DI)
	PSRLO     $8, X0
	MOVQ      X0, AX
	MOVL      AX, 64(DI)
	SHRQ      $32, AX
	MOVL      AX, 96(DI)
	RET

// Horizontal TTransform pass on 2 rows of 4; each output sits in word
// lanes 0 and 4. T is scratch.
#define HPASSX(X, S2, S3, D2, D3, T) \
	PSHUFL  $0xb1, X, T \
	MOVO    X, S2 \
	PADDW   T, S2 \
	MOVO    X, D2 \
	PSUBW   T, D2 \
	PSHUFLW $0xb1, S2, T \
	PSHUFHW $0xb1, T, T \
	MOVO    S2, S3 \
	PADDW   T, S2 \
	PSUBW   T, S3 \
	PSHUFLW $0xb1, D2, T \
	PSHUFHW $0xb1, T, T \
	MOVO    D2, D3 \
	PADDW   T, D2 \
	PSUBW   T, D3

// ABSW is |v| on int16 lanes without SSSE3.
#define ABSW(v, t) \
	MOVO  v, t \
	PSRAW $15, t \
	PXOR  t, v \
	PSUBW t, v

// kWeightY pairs for the horizontal-pass outputs, words 0 and 4.
DATA ttWeights<>+0(SB)/8, $0x26
DATA ttWeights<>+8(SB)/8, $0x20
DATA ttWeights<>+16(SB)/8, $0x20
DATA ttWeights<>+24(SB)/8, $0x1c
DATA ttWeights<>+32(SB)/8, $0x14
DATA ttWeights<>+40(SB)/8, $0x11
DATA ttWeights<>+48(SB)/8, $0x09
DATA ttWeights<>+56(SB)/8, $0x07
DATA ttWeights<>+64(SB)/8, $0x14
DATA ttWeights<>+72(SB)/8, $0x09
DATA ttWeights<>+80(SB)/8, $0x11
DATA ttWeights<>+88(SB)/8, $0x07
DATA ttWeights<>+96(SB)/8, $0x0a
DATA ttWeights<>+104(SB)/8, $0x04
DATA ttWeights<>+112(SB)/8, $0x04
DATA ttWeights<>+120(SB)/8, $0x02
GLOBL ttWeights<>(SB), RODATA, $128

// nzAC is 1 in byte lanes 1-15.
DATA nzAC<>+0(SB)/8, $0x0101010101010100
DATA nzAC<>+8(SB)/8, $0x0101010101010101
GLOBL nzAC<>(SB), RODATA, $16

// func recMetricsAsm(c *[16]int16, dst *uint8, src *uint8, stride int) (sse, tt int)
// idctAddAsm, then SSE of src against the result and its TTransform.
TEXT ·recMetricsAsm(SB), NOSPLIT, $0-48
	MOVQ c+0(FP), SI
	MOVQ dst+8(FP), DI
	MOVQ src+16(FP), R8
	MOVQ stride+24(FP), R9
	IDCT16(rm32, rmadd)

rm32:
	MOVL   $35468, AX
	MOVQ   AX, X14
	PSHUFL $0, X14, X14
	MOVL   $85627, AX
	MOVQ   AX, X15
	PSHUFL $0, X15, X15

	// Sign-extend rows to int32.
	MOVOU     (SI), X4
	MOVOU     16(SI), X6
	MOVO      X4, X0
	PUNPCKLWL X4, X0
	PSRAL     $16, X0
	MOVO      X4, X1
	PUNPCKHWL X4, X1
	PSRAL     $16, X1
	MOVO      X6, X2
	PUNPCKLWL X6, X2
	PSRAL     $16, X2
	MOVO      X6, X3
	PUNPCKHWL X6, X3
	PSRAL     $16, X3

	PASS(X0, X1, X2, X3)
	TRANSPOSE(X0, X1, X2, X3, X4)
	MOVL   $4, AX
	MOVQ   AX, X5
	PSHUFL $0, X5, X5
	PADDL  X5, X0
	PASS(X0, X1, X2, X3)
	PSRAL  $3, X0
	PSRAL  $3, X1
	PSRAL  $3, X2
	PSRAL  $3, X3
	TRANSPOSE(X0, X1, X2, X3, X4)

	// Saturation equals clip8 of int32 sum.
	PACKSSLW  X1, X0
	PACKSSLW  X3, X2

rmadd:
	MOVL      (DI), X4
	MOVL      32(DI), X5
	PUNPCKLLQ X5, X4
	MOVL      64(DI), X6
	MOVL      96(DI), X7
	PUNPCKLLQ X7, X6
	PXOR      X7, X7
	PUNPCKLBW X7, X4
	PUNPCKLBW X7, X6
	PADDSW    X4, X0
	PADDSW    X6, X2
	PACKUSWB  X2, X0
	MOVO      X0, X13
	MOVQ      X0, AX
	MOVL      AX, (DI)
	SHRQ      $32, AX
	MOVL      AX, 32(DI)
	PSRLO     $8, X0
	MOVQ      X0, AX
	MOVL      AX, 64(DI)
	SHRQ      $32, AX
	MOVL      AX, 96(DI)

	// SSE against the four source rows.
	MOVL       (R8), X4
	MOVL       (R8)(R9*1), X5
	PUNPCKLLQ  X5, X4
	LEAQ       (R8)(R9*2), R10
	MOVL       (R10), X6
	MOVL       (R10)(R9*1), X7
	PUNPCKLLQ  X7, X6
	PUNPCKLQDQ X6, X4
	MOVO       X13, X5
	PSUBUSB    X4, X5
	MOVO       X4, X6
	PSUBUSB    X13, X6
	POR        X6, X5
	PXOR       X7, X7
	MOVO       X5, X6
	PUNPCKLBW  X7, X5
	PUNPCKHBW  X7, X6
	PMADDWL    X5, X5
	PMADDWL    X6, X6
	PADDL      X6, X5
	PSHUFL     $0x4e, X5, X6
	PADDL      X6, X5
	PSHUFL     $0xb1, X5, X6
	PADDL      X6, X5
	MOVQ       X5, AX
	MOVL       AX, AX
	MOVQ       AX, sse+32(FP)

	// TTransform: rows 0|1 in X8, rows 2|3 in X9, vertical pass first.
	MOVO       X13, X8
	PUNPCKLBW  X7, X8
	MOVO       X13, X9
	PUNPCKHBW  X7, X9
	MOVO       X8, X0
	PADDW      X9, X0
	MOVO       X8, X1
	PSUBW      X9, X1
	PSHUFL     $0x4e, X0, X2
	PSHUFL     $0x4e, X1, X3
	MOVO       X0, X4
	PADDW      X2, X4
	PSUBW      X2, X0
	MOVO       X1, X5
	PADDW      X3, X5
	PSUBW      X3, X1
	MOVO       X4, X8
	PUNPCKLQDQ X5, X8
	MOVO       X1, X9
	PUNPCKLQDQ X0, X9
	HPASSX(X8, X0, X1, X2, X3, X10)
	HPASSX(X9, X4, X5, X6, X7, X10)
	ABSW(X0, X10)
	ABSW(X1, X10)
	ABSW(X2, X10)
	ABSW(X3, X10)
	ABSW(X4, X10)
	ABSW(X5, X10)
	ABSW(X6, X10)
	ABSW(X7, X10)
	MOVOU      ttWeights<>+0(SB), X11
	PMADDWL    X11, X0
	MOVOU      ttWeights<>+16(SB), X11
	PMADDWL    X11, X2
	MOVOU      ttWeights<>+32(SB), X11
	PMADDWL    X11, X3
	MOVOU      ttWeights<>+48(SB), X11
	PMADDWL    X11, X1
	MOVOU      ttWeights<>+64(SB), X11
	PMADDWL    X11, X4
	MOVOU      ttWeights<>+80(SB), X11
	PMADDWL    X11, X6
	MOVOU      ttWeights<>+96(SB), X11
	PMADDWL    X11, X7
	MOVOU      ttWeights<>+112(SB), X11
	PMADDWL    X11, X5
	PADDL      X1, X0
	PADDL      X2, X0
	PADDL      X3, X0
	PADDL      X4, X0
	PADDL      X5, X0
	PADDL      X6, X0
	PADDL      X7, X0
	PSHUFL     $0x4e, X0, X1
	PADDL      X1, X0
	PSHUFL     $0xb1, X0, X1
	PADDL      X1, X0
	MOVQ       X0, AX
	MOVL       AX, AX
	MOVQ       AX, tt+40(FP)
	RET

// func i4TrialsAsm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, dq *[16]int16)
// All ten i4 modes: FDCTB and QUANTB, then the recMetricsAsm body with the
// prediction from pr and the result to rec. res holds last, SSE, TT and
// the count of nonzero AC levels per mode.
TEXT ·i4TrialsAsm(SB), NOSPLIT, $256-64
	MOVQ pr+8(FP), R8
	MOVQ src+16(FP), R9
	MOVQ stride+24(FP), R10
	MOVQ lv+32(FP), DI
	MOVQ rec+40(FP), R12
	MOVQ res+48(FP), R13
	MOVQ dq+56(FP), SI
	MOVQ $10, R14

	// Constants for all ten trials: fdct at 0(SP), quantizer at 128(SP).
	FDCTK
	MOVOU X8, 0(SP)
	MOVOU X9, 16(SP)
	MOVOU X10, 32(SP)
	MOVOU X11, 48(SP)
	MOVOU X12, 64(SP)
	MOVOU X13, 80(SP)
	MOVOU X14, 96(SP)
	MOVOU X15, 112(SP)
	MOVQ  m+0(FP), AX
	QUANTK
	MOVOU X8, 128(SP)
	MOVOU X9, 144(SP)
	MOVOU X10, 160(SP)
	MOVOU X11, 176(SP)
	MOVOU X12, 192(SP)
	MOVOU X13, 208(SP)
	MOVOU X14, 224(SP)
	MOVOU X15, 240(SP)

	I4TRIAL(QUANTB, NZ2, ABSW, HPASSX, trial10, idct32, addpred)
	RET

// func cpuid1() (ecx uint32)
TEXT ·cpuid1(SB), NOSPLIT, $0-4
	MOVL $1, AX
	XORL CX, CX
	CPUID
	MOVL CX, ecx+0(FP)
	RET

// func cpuid7() (ebx uint32)
// Zero when the max basic leaf is below 7.
TEXT ·cpuid7(SB), NOSPLIT, $0-4
	XORL AX, AX
	CPUID
	CMPL AX, $7
	JB none
	MOVL $7, AX
	XORL CX, CX
	CPUID
	MOVL BX, ebx+0(FP)
	RET
none:
	MOVL $0, ebx+0(FP)
	RET

// func xgetbv0() (eax uint32)
TEXT ·xgetbv0(SB), NOSPLIT, $0-4
	XORL CX, CX
	XGETBV
	MOVL AX, eax+0(FP)
	RET

// func residualNAsm(pred *uint8, src *uint8, stride int, out *[16]int16, rows, cols int)
// residualAsm over a rows x cols grid of blocks, raster order.
TEXT ·residualNAsm(SB), NOSPLIT, $0-48
	MOVQ pred+0(FP), R12
	MOVQ src+8(FP), R13
	MOVQ stride+16(FP), R10
	MOVQ out+24(FP), DI
	MOVQ rows+32(FP), DX
	FDCTK
rnrow:
	MOVQ cols+40(FP), CX
	MOVQ R12, R8
	MOVQ R13, R9
rnblk:
	MOVL      (R8), X6
	MOVL      32(R8), X0
	MOVL      64(R8), X1
	MOVL      96(R8), X2
	PUNPCKLLQ X0, X6
	PUNPCKLLQ X2, X1
	PUNPCKLQDQ X1, X6
	FDCTB
	MOVOU X0, (DI)
	MOVOU X1, 16(DI)
	// FDCTB overwrites these three.
	KD(8|8<<16, X8)
	KD(8|-8<<16, X9)
	KD(5352|2217<<16, X10)
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  rnblk
	ADDQ $128, R12
	LEAQ (R13)(R10*4), R13
	DECQ DX
	JNZ  rnrow
	RET

// func quantNAsm(m *quantMatrix, in, out *[16]int16, n int) (nz int)
// quantAsm over n blocks; bit i of nz is set when block i has a level.
TEXT ·quantNAsm(SB), NOSPLIT, $0-40
	MOVQ m+0(FP), AX
	MOVQ in+8(FP), SI
	MOVQ out+16(FP), DI
	MOVQ n+24(FP), R8
	QUANTK
	XORQ R9, R9
	XORQ R10, R10
	CMPB ·useSSE41(SB), $0
	JNE  qn41
qnblk:
	MOVOU (SI), X0
	MOVOU 16(SI), X1
	QUANTB
	CMPQ  BX, $0
	JLT   qnzero
	BTSQ  R10, R9
qnzero:
	ADDQ $32, SI
	ADDQ $32, DI
	INCQ R10
	CMPQ R10, R8
	JNE  qnblk
	MOVQ R9, nz+32(FP)
	RET

qn41:
	MOVOU (SI), X0
	MOVOU 16(SI), X1
	QUANTB41
	CMPQ  BX, $0
	JLT   qnzero41
	BTSQ  R10, R9
qnzero41:
	ADDQ $32, SI
	ADDQ $32, DI
	INCQ R10
	CMPQ R10, R8
	JNE  qn41
	MOVQ R9, nz+32(FP)
	RET

// func idctAddNAsm(c *[16]int16, dst *uint8, rows, cols int)
// idctAddAsm over a rows x cols grid of blocks, skipping zero blocks.
// Locals: 16 c, 24 dst, 32 row dst, 40 cols left, 48 rows left.
TEXT ·idctAddNAsm(SB), NOSPLIT, $56-32
	MOVQ c+0(FP), SI
	MOVQ SI, 16(SP)
	MOVQ dst+8(FP), DI
	MOVQ rows+16(FP), DX
	MOVQ DX, 48(SP)
inrow:
	MOVQ DI, 24(SP)
	MOVQ DI, 32(SP)
	MOVQ cols+24(FP), CX
	MOVQ CX, 40(SP)
inblk:
	MOVQ     16(SP), SI
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	POR      X1, X0
	PXOR     X1, X1
	PCMPEQB  X1, X0
	PMOVMSKB X0, AX
	CMPL     AX, $0xffff
	JEQ      inskip
	MOVQ     SI, 0(SP)
	MOVQ     24(SP), DI
	MOVQ     DI, 8(SP)
	CALL     ·idctAddAsm(SB)
inskip:
	ADDQ $32, 16(SP)
	ADDQ $4, 24(SP)
	DECQ 40(SP)
	JNZ  inblk
	MOVQ 32(SP), DI
	ADDQ $128, DI
	DECQ 48(SP)
	JNZ  inrow
	RET

// func trellisDPAsm(a *trellisArgs) int
// DI = n, R8/R9 = ps0/ps1, R10/R11 = pc0/pc1, R13 = best, R14 = a.
TEXT ·trellisDPAsm(SB), NOSPLIT, $40-16
	MOVQ    a+0(FP), R14
	MOVQ    $-1, path-8(SP)
	MOVQ    trellisArgs_ps0(R14), R8
	MOVQ    trellisArgs_ps1(R14), R9
	MOVQ    trellisArgs_pc0(R14), R10
	MOVQ    R10, R11
	MOVQ    trellisArgs_best(R14), R13
	MOVQ    trellisArgs_first(R14), DI
	CMPQ    DI, trellisArgs_last(R14)
	JGT     tdone

tloop:
	// c0 = uint32 of the int16 |in[j]|, plus sharpen[j].
	LEAQ    ·zigzag(SB), AX
	MOVBLZX (AX)(DI*1), SI
	MOVQ    trellisArgs_in(R14), AX
	MOVWLSX (AX)(SI*2), AX
	MOVL    AX, CX
	SARL    $31, CX
	XORL    CX, AX
	SUBL    CX, AX
	MOVWLSX AX, AX
	MOVQ    trellisArgs_m(R14), BX
	ADDL    quantMatrix_sharpen(BX)(SI*4), AX
	XORL    DX, DX
	TESTL   SI, SI
	SETNE   DL
	MOVL    quantMatrix_q(BX)(DX*4), R12
	MOVL    quantMatrix_iq(BX)(DX*4), BX
	IMULL   AX, BX
	MOVL    BX, DX
	SHRL    $17, DX
	ADDL    $0x10000, BX
	SHRL    $17, BX
	CMPL    DX, $2047
	JHI     tcapl0

tl0ok:
	CMPL    BX, $2047
	JHI     tcaptl

ttlok:
	TESTL   BX, BX
	JEQ     tzero
	MOVQ    CX, neg-16(SP)
	MOVQ    BX, tl-24(SP)

	// d0, d1 = w*(e*e - c0*c0) for levels l0 and l0+1.
	LEAQ    ·trellisWeight(SB), CX
	MOVQ    (CX)(SI*8), CX
	SHLQ    $8, CX
	MOVQ    AX, SI
	IMULQ   SI, SI
	MOVQ    DX, BX
	IMULQ   R12, BX
	SUBQ    BX, AX
	MOVQ    AX, BX
	SUBQ    R12, BX
	IMULQ   AX, AX
	SUBQ    SI, AX
	IMULQ   CX, AX
	MOVQ    AX, d0-32(SP)
	IMULQ   BX, BX
	SUBQ    SI, BX
	IMULQ   CX, BX
	MOVQ    BX, d1-40(SP)

	// Node 0, level DX = l0.
	CMPQ    DX, $67
	JHI     tbig0
	MOVWLZX (R10)(DX*2), AX
	MOVWLZX (R11)(DX*2), BX

tcost0:
	IMULQ   trellisArgs_lam(R14), AX
	ADDQ    R8, AX
	IMULQ   trellisArgs_lam(R14), BX
	ADDQ    R9, BX
	XORL    CX, CX
	CMPQ    BX, AX
	SETLT   CL
	CMOVQLT BX, AX
	ADDQ    d0-32(SP), AX
	MOVQ    trellisArgs_nodes(R14), SI
	LEAQ    (SI)(DI*8), SI
	MOVB    CL, (SI)
	MOVL    DX, BX
	XORL    neg-16(SP), BX
	SUBL    neg-16(SP), BX
	MOVW    BX, 2(SI)
	TESTQ   DX, DX
	JEQ     tnode1
	CMPQ    AX, R13
	JGE     tnode1
	MOVQ    $2, BX
	CMPQ    DX, BX
	CMOVQLT DX, BX
	MOVQ    trellisArgs_eob(R14), CX
	LEAQ    (CX)(DI*8), CX
	MOVWLZX (CX)(BX*2), CX
	IMULQ   trellisArgs_lam(R14), CX
	ADDQ    AX, CX
	CMPQ    CX, R13
	JGE     tnode1
	MOVQ    CX, R13
	LEAQ    (DI)(DI*1), CX
	MOVQ    CX, path-8(SP)

tnode1:
	// Node 1, level DX = l0+1, if <= tl.
	MOVQ    AX, R12
	INCQ    DX
	MOVQ    $0x7fffffffffffff, AX
	CMPQ    DX, tl-24(SP)
	JGT     tnext
	CMPQ    DX, $67
	JHI     tbig1
	MOVWLZX (R10)(DX*2), AX
	MOVWLZX (R11)(DX*2), BX

tcost1:
	IMULQ   trellisArgs_lam(R14), AX
	ADDQ    R8, AX
	IMULQ   trellisArgs_lam(R14), BX
	ADDQ    R9, BX
	XORL    CX, CX
	CMPQ    BX, AX
	SETLT   CL
	CMOVQLT BX, AX
	ADDQ    d1-40(SP), AX
	MOVB    CL, 4(SI)
	MOVL    DX, BX
	XORL    neg-16(SP), BX
	SUBL    neg-16(SP), BX
	MOVW    BX, 6(SI)
	CMPQ    AX, R13
	JGE     tnext
	MOVQ    $2, BX
	CMPQ    DX, BX
	CMOVQLT DX, BX
	MOVQ    trellisArgs_eob(R14), CX
	LEAQ    (CX)(DI*8), CX
	MOVWLZX (CX)(BX*2), CX
	IMULQ   trellisArgs_lam(R14), CX
	ADDQ    AX, CX
	CMPQ    CX, R13
	JGE     tnext
	MOVQ    CX, R13
	LEAQ    1(DI)(DI*1), CX
	MOVQ    CX, path-8(SP)

tnext:
	// pc0, pc1 = rows min(l0, 2), min(l0+1, 2) of position min(n+1, 15).
	MOVQ    R12, R8
	MOVQ    AX, R9
	LEAQ    1(DI), CX
	MOVQ    CX, BX
	SHRQ    $4, BX
	SUBQ    BX, CX
	SHLQ    $10, CX
	ADDQ    trellisArgs_tc(R14), CX
	MOVQ    $2, BX
	CMPQ    DX, BX
	CMOVQLT DX, BX
	SHLQ    $8, BX
	LEAQ    (CX)(BX*1), R11
	DECQ    DX
	MOVQ    $2, BX
	CMPQ    DX, BX
	CMOVQLT DX, BX
	SHLQ    $8, BX
	LEAQ    (CX)(BX*1), R10
	INCQ    DI
	CMPQ    DI, trellisArgs_last(R14)
	JLE     tloop

tdone:
	MOVQ    path-8(SP), AX
	MOVQ    AX, ret+8(FP)
	RET

	// tl == 0: only a level 0 node, which cannot end the block.
tzero:
	MOVWLZX (R10), AX
	IMULQ   trellisArgs_lam(R14), AX
	ADDQ    R8, AX
	MOVWLZX (R11), BX
	IMULQ   trellisArgs_lam(R14), BX
	ADDQ    R9, BX
	XORL    CX, CX
	CMPQ    BX, AX
	SETLT   CL
	CMOVQLT BX, AX
	MOVQ    trellisArgs_nodes(R14), SI
	MOVL    CX, (SI)(DI*8)
	MOVQ    AX, R8
	MOVQ    $0x7fffffffffffff, R9
	LEAQ    1(DI), CX
	MOVQ    CX, BX
	SHRQ    $4, BX
	SUBQ    BX, CX
	SHLQ    $10, CX
	ADDQ    trellisArgs_tc(R14), CX
	MOVQ    CX, R10
	MOVQ    CX, R11
	INCQ    DI
	CMPQ    DI, trellisArgs_last(R14)
	JLE     tloop
	JMP     tdone

tcapl0:
	MOVL    $2047, DX
	JMP     tl0ok

tcaptl:
	MOVL    $2047, BX
	JMP     ttlok

	// Levels above 67: fixed cost beyond fixedCost[67], then row 67.
tbig0:
	LEAQ    ·fixedCost(SB), CX
	MOVQ    (CX)(DX*8), BX
	SUBQ    536(CX), BX
	MOVWLZX 134(R10), AX
	ADDQ    BX, AX
	MOVWLZX 134(R11), CX
	ADDQ    CX, BX
	JMP     tcost0

tbig1:
	LEAQ    ·fixedCost(SB), CX
	MOVQ    (CX)(DX*8), BX
	SUBQ    536(CX), BX
	MOVWLZX 134(R10), AX
	ADDQ    BX, AX
	MOVWLZX 134(R11), CX
	ADDQ    CX, BX
	JMP     tcost1

// resCostAsm constants: word lanes n, n*512, then 2, 67, 127.
DATA rck<>+0(SB)/8, $0x0003000200010000
DATA rck<>+8(SB)/8, $0x0007000600050004
DATA rck<>+16(SB)/8, $0x000b000a00090008
DATA rck<>+24(SB)/8, $0x000f000e000d000c
DATA rck<>+32(SB)/8, $0x0600040002000000
DATA rck<>+40(SB)/8, $0x0e000c000a000800
DATA rck<>+48(SB)/8, $0x1600140012001000
DATA rck<>+56(SB)/8, $0x1e001c001a001800
DATA rck<>+64(SB)/8, $0x0002000200020002
DATA rck<>+72(SB)/8, $0x0002000200020002
DATA rck<>+80(SB)/8, $0x0043004300430043
DATA rck<>+88(SB)/8, $0x0043004300430043
DATA rck<>+96(SB)/8, $0x007f007f007f007f
DATA rck<>+104(SB)/8, $0x007f007f007f007f
GLOBL rck<>(SB), RODATA|NOPTR, $112

// func resCostAsm(s *costTabs, typ, ctx, first int, lv *[16]int16) (c, last int)
// resCost in SSE2 and scalar asm, as the arm64 one: all 16 cost rows
// indexed at once, positions outside [first, last] read cost[0][0][127],
// never set. first is 0 or 1, so bands[first] == first.
TEXT ·resCostAsm(SB), NOSPLIT, $32-56
	MOVQ s+0(FP), R8
	MOVQ typ+8(FP), R9
	MOVQ ctx+16(FP), R10
	MOVQ first+24(FP), CX
	MOVQ lv+32(FP), SI

	// last = highest nonzero n >= first, else -1.
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	PXOR     X7, X7
	MOVO     X0, X2
	MOVO     X1, X3
	PCMPEQW  X7, X2
	PCMPEQW  X7, X3
	PACKSSWB X3, X2
	PMOVMSKB X2, AX
	XORL     $0xffff, AX
	SHRL     CX, AX
	SHLL     CX, AX
	MOVQ     $-1, DX
	BSRL     AX, BX
	CMOVLEQ  DX, BX
	MOVLQSX  BX, BX
	MOVQ     BX, last+48(FP)

	// p0 = probs[typ*264+(first*3+ctx)*11]
	LEAQ    costTabs_probs(R8), DI
	IMUL3Q  $264, R9, R12
	LEAQ    (CX)(CX*2), R13
	ADDQ    R10, R13
	IMUL3Q  $11, R13, R13
	ADDQ    R12, R13
	MOVBLZX (DI)(R13*1), R13
	LEAQ    ·bitCost(SB), R14
	TESTQ   BX, BX
	JS      rcnone
	XORL    AX, AX
	MOVQ    2048(R14)(R13*8), R11
	TESTQ   R10, R10
	CMOVQEQ R11, AX
	MOVQ    R12, R13

	// Row index n*512 + ctx*128 + min(|v|, 67), where ctx is
	// min(|v|, 2) of the previous position.
	PXOR    X2, X2
	PSUBW   X0, X2
	PMAXSW  X0, X2
	PXOR    X3, X3
	PSUBW   X1, X3
	PMAXSW  X1, X3
	MOVOU   rck<>+64(SB), X4
	MOVO    X4, X5
	PMINSW  X2, X4
	PMINSW  X3, X5
	MOVO    X4, X6
	PSRLDQ  $14, X6
	PSLLDQ  $2, X5
	POR     X6, X5
	PSLLDQ  $2, X4
	TESTQ   CX, CX
	JNE     rcfirst1
	PINSRW  $0, R10, X4
	JMP     rcidx

rcfirst1:
	PINSRW $1, R10, X4

rcidx:
	PSLLW   $7, X4
	PSLLW   $7, X5
	MOVOU   rck<>+80(SB), X6
	MOVO    X6, X7
	PMINSW  X2, X6
	PMINSW  X3, X7
	PADDW   X6, X4
	PADDW   X7, X5
	MOVOU   rck<>+32(SB), X6
	MOVOU   rck<>+48(SB), X7
	PADDW   X6, X4
	PADDW   X7, X5

	// out = n < first || n > last, as words.
	MOVQ      CX, X6
	PSHUFLW   $0, X6, X6
	PSHUFL    $0, X6, X6
	MOVQ      BX, X7
	PSHUFLW   $0, X7, X7
	PSHUFL    $0, X7, X7
	MOVOU     rck<>+0(SB), X8
	MOVOU     rck<>+16(SB), X9
	MOVO      X6, X10
	MOVO      X6, X11
	PCMPGTW   X8, X10
	PCMPGTW   X9, X11
	MOVO      X8, X12
	MOVO      X9, X13
	PCMPGTW   X7, X12
	PCMPGTW   X7, X13
	POR       X12, X10
	POR       X13, X11
	MOVOU     rck<>+96(SB), X12
	MOVO      X12, X13
	PAND      X10, X12
	PAND      X11, X13
	PANDN     X4, X10
	PANDN     X5, X11
	POR       X12, X10
	POR       X13, X11
	MOVOU     X10, 0(SP)
	MOVOU     X11, 16(SP)

	// Sum the 16 rows, four chains.
	LEAQ    costTabs_cost(R8), R14
	SHLQ    $14, R9
	ADDQ    R9, R14
	XORL    R10, R10
	XORL    R11, R11
	XORL    R12, R12
	MOVQ    $0, R8
rcsum:
	MOVWLZX 0(SP)(R8*2), DX
	MOVWLZX (R14)(DX*2), DX
	ADDQ    DX, AX
	MOVWLZX 2(SP)(R8*2), DX
	MOVWLZX (R14)(DX*2), DX
	ADDQ    DX, R10
	MOVWLZX 4(SP)(R8*2), DX
	MOVWLZX (R14)(DX*2), DX
	ADDQ    DX, R11
	MOVWLZX 6(SP)(R8*2), DX
	MOVWLZX (R14)(DX*2), DX
	ADDQ    DX, R12
	ADDQ    $4, R8
	CMPQ    R8, $16
	JNE     rcsum
	ADDQ    R10, AX
	ADDQ    R12, R11
	ADDQ    R11, AX

	// Levels above 67 add fixedCost[v] - fixedCost[67].
	MOVOU    rck<>+80(SB), X6
	PCMPGTW  X6, X2
	PCMPGTW  X6, X3
	POR      X3, X2
	PMOVMSKB X2, DX
	TESTL    DX, DX
	JNE      rcbig

rctail:
	CMPQ    BX, $15
	JEQ     rcdone
	MOVWQSX (SI)(BX*2), DX
	MOVQ    DX, R8
	SARQ    $63, R8
	XORQ    R8, DX
	SUBQ    R8, DX
	MOVQ    $2, R8
	CMPQ    DX, R8
	CMOVQHI R8, DX
	LEAQ    ·bands(SB), R8
	MOVBLZX 1(R8)(BX*1), R8
	LEAQ    (R8)(R8*2), R8
	ADDQ    DX, R8
	IMUL3Q  $11, R8, R8
	ADDQ    R13, R8
	MOVBLZX (DI)(R8*1), R8
	LEAQ    ·bitCost(SB), R9
	ADDQ    (R9)(R8*8), AX

rcdone:
	MOVQ AX, c+40(FP)
	RET

rcnone:
	MOVQ (R14)(R13*8), AX
	MOVQ AX, c+40(FP)
	RET

rcbig:
	LEAQ ·fixedCost(SB), R8
	MOVQ 536(R8), R9
rcbloop:
	MOVWQSX (SI)(CX*2), DX
	MOVQ    DX, R10
	SARQ    $63, R10
	XORQ    R10, DX
	SUBQ    R10, DX
	CMPQ    DX, $67
	JLS     rcbnext
	ANDQ    $2047, DX
	ADDQ    (R8)(DX*8), AX
	SUBQ    R9, AX

rcbnext:
	INCQ CX
	CMPQ CX, BX
	JLE  rcbloop
	JMP  rctail

// linearToGamma[0:32], then its steps; checked by TestChromaKernel.
// Then dword constants: shuffle offsets, 511, 64,
// Cb and Cr weights (R | G<<16, then B), rounding.
DATA chk<>+0(SB)/8, $0x261f19130d080300
DATA chk<>+8(SB)/8, $0x635b534b433c342d
DATA chk<>+16(SB)/8, $0xa9a0978e857c746b
DATA chk<>+24(SB)/8, $0xf5ebe2d8cec5bbb2
DATA chk<>+32(SB)/8, $0x0707060606050503
DATA chk<>+40(SB)/8, $0x0808080808070807
DATA chk<>+48(SB)/8, $0x0909090909090809
DATA chk<>+56(SB)/8, $0x0a0a090a0a090a09
DATA chk<>+64(SB)/8, $0x8080807080808070
DATA chk<>+72(SB)/8, $0x8080807080808070
DATA chk<>+80(SB)/8, $0x8080807080808070
DATA chk<>+88(SB)/8, $0x8080807080808070
DATA chk<>+96(SB)/8, $0x808080f0808080f0
DATA chk<>+104(SB)/8, $0x808080f0808080f0
DATA chk<>+112(SB)/8, $0x808080f0808080f0
DATA chk<>+120(SB)/8, $0x808080f0808080f0
DATA chk<>+128(SB)/8, $0x000001ff000001ff
DATA chk<>+136(SB)/8, $0x000001ff000001ff
DATA chk<>+144(SB)/8, $0x000001ff000001ff
DATA chk<>+152(SB)/8, $0x000001ff000001ff
DATA chk<>+160(SB)/8, $0x0000004000000040
DATA chk<>+168(SB)/8, $0x0000004000000040
DATA chk<>+176(SB)/8, $0x0000004000000040
DATA chk<>+184(SB)/8, $0x0000004000000040
DATA chk<>+192(SB)/8, $0xb577da09b577da09
DATA chk<>+200(SB)/8, $0xb577da09b577da09
DATA chk<>+208(SB)/8, $0xb577da09b577da09
DATA chk<>+216(SB)/8, $0xb577da09b577da09
DATA chk<>+224(SB)/8, $0xa1cc7080a1cc7080
DATA chk<>+232(SB)/8, $0xa1cc7080a1cc7080
DATA chk<>+240(SB)/8, $0xa1cc7080a1cc7080
DATA chk<>+248(SB)/8, $0xa1cc7080a1cc7080
DATA chk<>+256(SB)/8, $0x0000708000007080
DATA chk<>+264(SB)/8, $0x0000708000007080
DATA chk<>+272(SB)/8, $0x0000708000007080
DATA chk<>+280(SB)/8, $0x0000708000007080
DATA chk<>+288(SB)/8, $0x0000edb40000edb4
DATA chk<>+296(SB)/8, $0x0000edb40000edb4
DATA chk<>+304(SB)/8, $0x0000edb40000edb4
DATA chk<>+312(SB)/8, $0x0000edb40000edb4
DATA chk<>+320(SB)/8, $0x0202000002020000
DATA chk<>+328(SB)/8, $0x0202000002020000
DATA chk<>+336(SB)/8, $0x0202000002020000
DATA chk<>+344(SB)/8, $0x0202000002020000
GLOBL chk<>(SB), RODATA|NOPTR, $352

// func chromaAVX2Asm(cb, cr, p0, p1 *uint8, n int) (and uint8)
// n > 0 and n%8 == 0.
TEXT ·chromaAVX2Asm(SB), NOSPLIT, $0-41
	MOVQ cb+0(FP), DI
	MOVQ cr+8(FP), SI
	MOVQ p0+16(FP), R8
	MOVQ p1+24(FP), R9
	MOVQ n+32(FP), CX
	LEAQ ·gammaToLinear(SB), AX
	VBROADCASTI128 chk<>+0(SB), Y10
	VBROADCASTI128 chk<>+16(SB), Y11
	VBROADCASTI128 chk<>+32(SB), Y12
	VBROADCASTI128 chk<>+48(SB), Y13
	VPCMPEQD Y14, Y14, Y14
	VPSRLD   $24, Y14, Y15
// Per channel: gather gammaToLinear of each pixel,
// sum 2x2 (block order 0,1,4,5 | 2,3,6,7), then
// linToGamma as 4*L[p] + (D[p]*x+64)>>7.
chloop:
	VPAND (R8), Y14, Y14
	VPAND 32(R8), Y14, Y14
	VPAND (R9), Y14, Y14
	VPAND 32(R9), Y14, Y14
	VMOVDQU    0(R8), Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y2, Y2, Y2
	VPGATHERDD Y1, (AX)(Y0*4), Y2
	VMOVDQU    0(R9), Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VPADDD  Y3, Y2, Y2
	VMOVDQU    32(R8), Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VMOVDQU    32(R9), Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y4, Y4, Y4
	VPGATHERDD Y1, (AX)(Y0*4), Y4
	VPADDD  Y4, Y3, Y3
	VPHADDD Y3, Y2, Y2
	VPSRLD   $9, Y2, Y3
	VPADDB   chk<>+64(SB), Y3, Y4
	VPADDB   chk<>+96(SB), Y3, Y5
	VPSHUFB  Y4, Y10, Y3
	VPSHUFB  Y5, Y11, Y6
	VPOR     Y6, Y3, Y3
	VPSHUFB  Y4, Y12, Y4
	VPSHUFB  Y5, Y13, Y5
	VPOR     Y5, Y4, Y4
	VPAND    chk<>+128(SB), Y2, Y2
	VPMADDWD Y2, Y4, Y4
	VPADDD   chk<>+160(SB), Y4, Y4
	VPSRLD   $7, Y4, Y4
	VPSLLD   $2, Y3, Y3
	VPADDD   Y3, Y4, Y7
	VMOVDQU    0(R8), Y0
	VPSRLD     $8, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y2, Y2, Y2
	VPGATHERDD Y1, (AX)(Y0*4), Y2
	VMOVDQU    0(R9), Y0
	VPSRLD     $8, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VPADDD  Y3, Y2, Y2
	VMOVDQU    32(R8), Y0
	VPSRLD     $8, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VMOVDQU    32(R9), Y0
	VPSRLD     $8, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y4, Y4, Y4
	VPGATHERDD Y1, (AX)(Y0*4), Y4
	VPADDD  Y4, Y3, Y3
	VPHADDD Y3, Y2, Y2
	VPSRLD   $9, Y2, Y3
	VPADDB   chk<>+64(SB), Y3, Y4
	VPADDB   chk<>+96(SB), Y3, Y5
	VPSHUFB  Y4, Y10, Y3
	VPSHUFB  Y5, Y11, Y6
	VPOR     Y6, Y3, Y3
	VPSHUFB  Y4, Y12, Y4
	VPSHUFB  Y5, Y13, Y5
	VPOR     Y5, Y4, Y4
	VPAND    chk<>+128(SB), Y2, Y2
	VPMADDWD Y2, Y4, Y4
	VPADDD   chk<>+160(SB), Y4, Y4
	VPSRLD   $7, Y4, Y4
	VPSLLD   $2, Y3, Y3
	VPADDD   Y3, Y4, Y8
	VMOVDQU    0(R8), Y0
	VPSRLD     $16, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y2, Y2, Y2
	VPGATHERDD Y1, (AX)(Y0*4), Y2
	VMOVDQU    0(R9), Y0
	VPSRLD     $16, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VPADDD  Y3, Y2, Y2
	VMOVDQU    32(R8), Y0
	VPSRLD     $16, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y3, Y3, Y3
	VPGATHERDD Y1, (AX)(Y0*4), Y3
	VMOVDQU    32(R9), Y0
	VPSRLD     $16, Y0, Y0
	VPAND      Y15, Y0, Y0
	VPCMPEQD   Y1, Y1, Y1
	VPXOR      Y4, Y4, Y4
	VPGATHERDD Y1, (AX)(Y0*4), Y4
	VPADDD  Y4, Y3, Y3
	VPHADDD Y3, Y2, Y2
	VPSRLD   $9, Y2, Y3
	VPADDB   chk<>+64(SB), Y3, Y4
	VPADDB   chk<>+96(SB), Y3, Y5
	VPSHUFB  Y4, Y10, Y3
	VPSHUFB  Y5, Y11, Y6
	VPOR     Y6, Y3, Y3
	VPSHUFB  Y4, Y12, Y4
	VPSHUFB  Y5, Y13, Y5
	VPOR     Y5, Y4, Y4
	VPAND    chk<>+128(SB), Y2, Y2
	VPMADDWD Y2, Y4, Y4
	VPADDD   chk<>+160(SB), Y4, Y4
	VPSRLD   $7, Y4, Y4
	VPSLLD   $2, Y3, Y3
	VPADDD   Y3, Y4, Y9
	VPSLLD   $16, Y8, Y8
	VPOR     Y8, Y7, Y7
	VPMADDWD chk<>+192(SB), Y7, Y0
	VPMADDWD chk<>+224(SB), Y7, Y1
	VPMADDWD chk<>+256(SB), Y9, Y2
	VPMADDWD chk<>+288(SB), Y9, Y3
	VPADDD   Y2, Y0, Y0
	VPADDD   Y3, Y1, Y1
	VPADDD   chk<>+320(SB), Y0, Y0
	VPADDD   chk<>+320(SB), Y1, Y1
	VPSRAD   $18, Y0, Y0
	VPSRAD   $18, Y1, Y1
	// Cb and Cr lie in [16, 240], so packs are exact.
	VPERMQ       $0xd8, Y0, Y0
	VPERMQ       $0xd8, Y1, Y1
	VPACKSSDW    Y1, Y0, Y0
	VPERMQ       $0xd8, Y0, Y0
	VEXTRACTI128 $1, Y0, X1
	VPACKUSWB    X1, X0, X0
	VMOVQ        X0, (DI)
	VPEXTRQ      $1, X0, (SI)
	ADDQ $64, R8
	ADDQ $64, R9
	ADDQ $8, DI
	ADDQ $8, SI
	SUBQ $8, CX
	JNZ  chloop
	VEXTRACTI128 $1, Y14, X0
	VPAND        X0, X14, X0
	VPSHUFD      $0x4e, X0, X1
	VPAND        X1, X0, X0
	VPSHUFD      $0xb1, X0, X1
	VPAND        X1, X0, X0
	VMOVD        X0, AX
	SHRL         $24, AX
	MOVB         AX, and+40(FP)
	VZEROUPPER
	RET

// ttMB weights: [w0 w4 w8 w12 w1 w5 w9 w13], [w3 .. w15 w2 .. w14].
DATA ttw<>+0(SB)/8, $0x0009001400200026
DATA ttw<>+8(SB)/8, $0x00070011001c0020
DATA ttw<>+16(SB)/8, $0x0009001400200026
DATA ttw<>+24(SB)/8, $0x00070011001c0020
DATA ttw<>+32(SB)/8, $0x0002000400070009
DATA ttw<>+40(SB)/8, $0x0004000a00110014
DATA ttw<>+48(SB)/8, $0x0002000400070009
DATA ttw<>+56(SB)/8, $0x0004000a00110014
GLOBL ttw<>(SB), RODATA, $64

// U0 = cols 0|1, U1 = cols 2|3, k in words; R = weighted |T|.
#define TTH(U0, U1, R) \
	VPADDW      U1, U0, Y4 \
	VPSUBW      U1, U0, Y5 \
	VPUNPCKLQDQ Y5, Y4, Y6 \
	VPUNPCKHQDQ Y5, Y4, Y7 \
	VPADDW      Y7, Y6, Y4 \
	VPSUBW      Y7, Y6, Y5 \
	VPABSW      Y4, Y4 \
	VPABSW      Y5, Y5 \
	VPMADDWD    Y8, Y4, Y4 \
	VPMADDWD    Y9, Y5, Y5 \
	VPADDD      Y5, Y4, R

// One block row: A = blocks 0|2, B = blocks 1|3.
// Vertical pass first; sums stay within 4080.
#define TTROW(A, B) \
	VPMOVZXBW  (SI), Y0 \
	VPMOVZXBW  (SI)(BX*1), Y1 \
	VPMOVZXBW  (SI)(BX*2), Y2 \
	VPMOVZXBW  (SI)(CX*1), Y3 \
	LEAQ       (SI)(BX*4), SI \
	VPADDW     Y2, Y0, Y4 \
	VPSUBW     Y2, Y0, Y5 \
	VPADDW     Y3, Y1, Y6 \
	VPSUBW     Y3, Y1, Y7 \
	VPADDW     Y6, Y4, Y0 \
	VPSUBW     Y6, Y4, Y3 \
	VPADDW     Y7, Y5, Y1 \
	VPSUBW     Y7, Y5, Y2 \
	VPUNPCKLWD Y1, Y0, Y4 \
	VPUNPCKHWD Y1, Y0, Y5 \
	VPUNPCKLWD Y3, Y2, Y6 \
	VPUNPCKHWD Y3, Y2, Y7 \
	VPUNPCKLDQ Y6, Y4, Y0 \
	VPUNPCKHDQ Y6, Y4, Y1 \
	VPUNPCKLDQ Y7, Y5, Y2 \
	VPUNPCKHDQ Y7, Y5, Y3 \
	TTH(Y0, Y1, A) \
	TTH(Y2, Y3, B)

// Two block rows' sums to (DI) in raster order.
#define TTPAIR \
	TTROW(Y10, Y11) \
	TTROW(Y12, Y13) \
	VPHADDD Y11, Y10, Y10 \
	VPHADDD Y13, Y12, Y12 \
	VPHADDD Y12, Y10, Y10 \
	VPERMQ  $0xd8, Y10, Y10 \
	VMOVDQU Y10, (DI)

// func ttMBAsm(src *uint8, stride int, st *[16]int32)
TEXT ·ttMBAsm(SB), NOSPLIT, $0-24
	MOVQ    src+0(FP), SI
	MOVQ    stride+8(FP), BX
	MOVQ    st+16(FP), DI
	LEAQ    (BX)(BX*2), CX
	VMOVDQU ttw<>+0(SB), Y8
	VMOVDQU ttw<>+32(SB), Y9
	TTPAIR
	ADDQ    $32, DI
	TTPAIR
	VZEROUPPER
	RET

// WHT4: p0+p2, p1+p3 butterfly into p0, p1, t0, p2.
#define WHT4(p0, p1, p2, p3, t0, t1) \
	MOVO  p0, t0 \
	PADDL p2, p0 \
	PSUBL p2, t0 \
	MOVO  p1, t1 \
	PADDL p3, p1 \
	PSUBL p3, t1 \
	MOVO  p0, p2 \
	PADDL p1, p0 \
	PSUBL p1, p2 \
	MOVO  t0, p1 \
	PADDL t1, p1 \
	PSUBL t1, t0

// SEXT4: x's words as int32.
#define SEXT4(x, lo, hi) \
	MOVO      x, lo \
	PUNPCKLWL x, lo \
	PSRAL     $16, lo \
	MOVO      x, hi \
	PUNPCKHWL x, hi \
	PSRAL     $16, hi

// Y2Q is QUANTBX without SSE or last:
// dequantized in X0, X1, ZF if all zero.
#define Y2Q(qabs, qzz) \
	qabs \
	MOVO      X2, X4 \
	PMULLW    X11, X4 \
	PMULHUW   X11, X2 \
	MOVO      X4, X5 \
	PUNPCKLWL X2, X5 \
	PUNPCKHWL X2, X4 \
	PADDL     X13, X5 \
	PADDL     X12, X4 \
	PSRLL     $17, X5 \
	PSRLL     $17, X4 \
	PACKSSLW  X4, X5 \
	MOVO      X3, X4 \
	PMULLW    X10, X4 \
	PMULHUW   X10, X3 \
	MOVO      X4, X2 \
	PUNPCKLWL X3, X2 \
	PUNPCKHWL X3, X4 \
	PADDL     X12, X2 \
	PADDL     X12, X4 \
	PSRLL     $17, X2 \
	PSRLL     $17, X4 \
	PACKSSLW  X4, X2 \
	PXOR      X6, X5 \
	PSUBW     X6, X5 \
	PXOR      X7, X2 \
	PSUBW     X7, X2 \
	MOVO      X5, X0 \
	PMULLW    X9, X0 \
	MOVO      X2, X1 \
	PMULLW    X8, X1 \
	qzz \
	MOVOU     X5, (DI) \
	MOVOU     X2, 16(DI) \
	POR       X2, X5 \
	PXOR      X3, X3 \
	PCMPEQW   X3, X5 \
	PMOVMSKB  X5, AX \
	XORL      $0xffff, AX

// func i16DCAsm(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) (nz bool)
TEXT ·i16DCAsm(SB), NOSPLIT, $0-33
	MOVQ m+0(FP), AX
	MOVQ dq+8(FP), R8
	MOVQ lv+16(FP), DI
	MOVQ dc+24(FP), SI

	// X0 = DC columns 0|1, X1 = 2|3.
	PINSRW $0, 0(R8), X0
	PINSRW $1, 128(R8), X0
	PINSRW $2, 256(R8), X0
	PINSRW $3, 384(R8), X0
	PINSRW $4, 32(R8), X0
	PINSRW $5, 160(R8), X0
	PINSRW $6, 288(R8), X0
	PINSRW $7, 416(R8), X0
	PINSRW $0, 64(R8), X1
	PINSRW $1, 192(R8), X1
	PINSRW $2, 320(R8), X1
	PINSRW $3, 448(R8), X1
	PINSRW $4, 96(R8), X1
	PINSRW $5, 224(R8), X1
	PINSRW $6, 352(R8), X1
	PINSRW $7, 480(R8), X1
	XORL   CX, CX
	MOVW   CX, 0(R8)
	MOVW   CX, 32(R8)
	MOVW   CX, 64(R8)
	MOVW   CX, 96(R8)
	MOVW   CX, 128(R8)
	MOVW   CX, 160(R8)
	MOVW   CX, 192(R8)
	MOVW   CX, 224(R8)
	MOVW   CX, 256(R8)
	MOVW   CX, 288(R8)
	MOVW   CX, 320(R8)
	MOVW   CX, 352(R8)
	MOVW   CX, 384(R8)
	MOVW   CX, 416(R8)
	MOVW   CX, 448(R8)
	MOVW   CX, 480(R8)

	// fwhtPlain, int32 like the Go.
	SEXT4(X0, X2, X3)
	SEXT4(X1, X4, X5)
	WHT4(X2, X3, X4, X5, X6, X7)
	TRANSPOSE(X2, X3, X6, X4, X5)
	WHT4(X2, X3, X6, X4, X5, X7)

	// Shifts give int16(x >> 1).
	PSLLL    $15, X2
	PSLLL    $15, X3
	PSLLL    $15, X5
	PSLLL    $15, X6
	PSRAL    $16, X2
	PSRAL    $16, X3
	PSRAL    $16, X5
	PSRAL    $16, X6
	PACKSSLW X3, X2
	PACKSSLW X6, X5
	MOVO     X2, X0
	MOVO     X5, X1
	QUANTK
	CMPB     ·useSSE41(SB), $0
	JNE      dc41
	Y2Q(QABS2, QZZ2)
	JMP      dciwht

dc41:
	Y2Q(QABS41, QZZ41)

dciwht:
	SETNE AX
	MOVB  AX, nz+32(FP)

	// iwht of the dequantized DCs.
	SEXT4(X0, X2, X3)
	SEXT4(X1, X4, X5)
	WHT4(X2, X3, X5, X4, X6, X7)
	TRANSPOSE(X2, X3, X5, X6, X4)
	MOVL   $3, AX
	MOVQ   AX, X7
	PSHUFL $0, X7, X7
	PADDL  X7, X2
	WHT4(X2, X3, X6, X5, X4, X7)

	// Shifts give int16(x >> 3).
	PSLLL    $13, X2
	PSLLL    $13, X3
	PSLLL    $13, X4
	PSLLL    $13, X6
	PSRAL    $16, X2
	PSRAL    $16, X3
	PSRAL    $16, X4
	PSRAL    $16, X6
	TRANSPOSE(X2, X3, X6, X4, X5)
	PACKSSLW X3, X2
	PACKSSLW X4, X6
	MOVOU    X2, (SI)
	MOVOU    X6, 16(SI)
	RET
