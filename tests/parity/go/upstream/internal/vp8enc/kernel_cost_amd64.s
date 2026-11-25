//go:build !purego

#include "go_asm.h"
#include "textflag.h"

// costGridAsm constant: word lanes n.
DATA cgn<>+0(SB)/8, $0x0003000200010000
DATA cgn<>+8(SB)/8, $0x0007000600050004
DATA cgn<>+16(SB)/8, $0x000b000a00090008
DATA cgn<>+24(SB)/8, $0x000f000e000d000c
GLOBL cgn<>(SB), RODATA|NOPTR, $32

// One cost row of the gather, into acc.
#define CGROW(off, acc) MOVWLZX off(SP), DX; MOVWLZX (R14)(DX*2), DX; ADDQ DX, acc

// func costGridAsm(s *costTabs, typ, first int, lv *[16]int16, rows, cols, lbits, tbits int) (c int)
// resCostAsm over a rows x cols block grid in AVX2, one block per
// YMM, with the same arithmetic. Block (y, x) has ctx lbits bit y plus
// tbits bit x before the grid; then both take that block's nz.
TEXT ·costGridAsm(SB), NOSPLIT, $40-72
	MOVQ s+0(FP), R8
	MOVQ typ+8(FP), AX
	MOVQ first+16(FP), CX
	MOVQ lv+24(FP), SI

	// Lane constants: 2, 67, 127, n, n == first, n < first, n*512.
	MOVL         $2, DX
	VMOVD        DX, X8
	VPBROADCASTW X8, Y8
	MOVL         $67, DX
	VMOVD        DX, X9
	VPBROADCASTW X9, Y9
	MOVL         $127, DX
	VMOVD        DX, X10
	VPBROADCASTW X10, Y10
	VMOVDQU      cgn<>(SB), Y11
	VMOVD        CX, X12
	VPBROADCASTW X12, Y12
	VPCMPGTW     Y11, Y12, Y15
	VPCMPEQW     Y11, Y12, Y12
	VPSLLW       $9, Y11, Y13
	VPXOR        Y14, Y14, Y14

	// DI = probs[typ*264:], 32(SP) = probs[typ*264+first*33:],
	// R14 = cost[typ], CX = 2*first.
	LEAQ   costTabs_probs(R8), DI
	IMUL3Q $264, AX, DX
	ADDQ   DX, DI
	IMUL3Q $33, CX, DX
	ADDQ   DI, DX
	MOVQ   DX, 32(SP)
	LEAQ   costTabs_cost(R8), R14
	SHLQ   $14, AX
	ADDQ   AX, R14
	ADDQ   CX, CX

	// R8 sum, R9 x, R10 y, R11 top bits, R12 left bit.
	XORL R8, R8
	XORL R9, R9
	XORL R10, R10
	MOVQ tbits+56(FP), R11
	MOVQ lbits+48(FP), R12
	ANDQ $1, R12

cgblk:
	// DX = ctx, AX = p0 = probs[typ*264+(first*3+ctx)*11].
	MOVQ    R12, DX
	BTQ     R9, R11
	ADCQ    $0, DX
	LEAQ    (DX)(DX*4), AX
	LEAQ    (DX)(AX*2), AX
	MOVQ    32(SP), BX
	MOVBLZX (BX)(AX*1), AX
	LEAQ    ·bitCost(SB), BX

	// R13 = last = highest nonzero n >= first.
	VMOVDQU   (SI), Y0
	VPCMPEQW  Y0, Y14, Y1
	VPMOVMSKB Y1, R13
	NOTL      R13
	SHRL      CX, R13
	SHLL      CX, R13
	BSRL      R13, R13
	JEQ       cgnone
	SHRL      $1, R13

	// bitCost[1][p0] when ctx == 0.
	MOVQ    2048(BX)(AX*8), AX
	XORL    BX, BX
	TESTQ   DX, DX
	CMOVQNE BX, AX
	ADDQ    AX, R8

	// Row index n*512 + ctx*128 + min(|v|, 67), where ctx is
	// min(|v|, 2) of the previous position, or DX at first.
	VPABSW       Y0, Y1
	VPMINSW      Y8, Y1, Y2
	VPMINSW      Y9, Y1, Y3
	VPERM2I128   $0x08, Y2, Y2, Y4
	VPALIGNR     $14, Y4, Y2, Y2
	VPSLLW       $7, Y2, Y2
	VPANDN       Y2, Y12, Y2
	VPADDW       Y3, Y2, Y2
	VPADDW       Y13, Y2, Y2
	SHLQ         $7, DX
	VMOVD        DX, X4
	VPBROADCASTW X4, Y4
	VPAND        Y12, Y4, Y4
	VPADDW       Y4, Y2, Y2

	// n < first || n > last read 127.
	VMOVD        R13, X5
	VPBROADCASTW X5, Y5
	VPCMPGTW     Y5, Y11, Y5
	VPOR         Y15, Y5, Y5
	VPBLENDVB    Y5, Y10, Y2, Y2
	VMOVDQU      Y2, 0(SP)

	// Sum the 16 rows, two chains.
	XORL  AX, AX
	XORL  BX, BX
	CGROW(0, AX)
	CGROW(2, BX)
	CGROW(4, AX)
	CGROW(6, BX)
	CGROW(8, AX)
	CGROW(10, BX)
	CGROW(12, AX)
	CGROW(14, BX)
	CGROW(16, AX)
	CGROW(18, BX)
	CGROW(20, AX)
	CGROW(22, BX)
	CGROW(24, AX)
	CGROW(26, BX)
	CGROW(28, AX)
	CGROW(30, BX)
	ADDQ  AX, R8
	ADDQ  BX, R8

	// Levels above 67 add fixedCost[v] - fixedCost[67].
	VPCMPGTW  Y9, Y1, Y6
	VPMOVMSKB Y6, DX
	TESTL     DX, DX
	JNE       cgbig

cgtail:
	CMPQ    R13, $15
	JEQ     cgset
	MOVWQSX (SI)(R13*2), DX
	MOVQ    DX, AX
	SARQ    $63, AX
	XORQ    AX, DX
	SUBQ    AX, DX
	MOVQ    $2, AX
	CMPQ    DX, AX
	CMOVQHI AX, DX
	LEAQ    ·bands(SB), AX
	MOVBLZX 1(AX)(R13*1), AX
	LEAQ    (AX)(AX*2), AX
	ADDQ    DX, AX
	IMUL3Q  $11, AX, AX
	MOVBLZX (DI)(AX*1), AX
	LEAQ    ·bitCost(SB), DX
	ADDQ    (DX)(AX*8), R8

cgset:
	BTSQ R9, R11
	MOVL $1, R12

cgnext:
	ADDQ $32, SI
	INCQ R9
	CMPQ R9, cols+40(FP)
	JLT  cgblk
	XORL R9, R9
	INCQ R10
	CMPQ R10, rows+32(FP)
	JGE  cgdone
	MOVQ lbits+48(FP), DX
	XORL R12, R12
	BTQ  R10, DX
	ADCL $0, R12
	JMP  cgblk

cgdone:
	MOVQ R8, c+64(FP)
	VZEROUPPER
	RET

cgnone:
	ADDQ (BX)(AX*8), R8
	BTRQ R9, R11
	XORL R12, R12
	JMP  cgnext

cgbig:
	LEAQ  ·fixedCost(SB), AX
	MOVQ  CX, BX
	SHRQ  $1, BX
cgbloop:
	MOVWQSX (SI)(BX*2), DX
	TESTQ   DX, DX
	JNS     cgbpos
	NEGQ    DX
cgbpos:
	CMPQ DX, $67
	JLS  cgbnext
	ANDQ $2047, DX
	ADDQ (AX)(DX*8), R8
	SUBQ 536(AX), R8

cgbnext:
	INCQ BX
	CMPQ BX, R13
	JLE  cgbloop
	JMP  cgtail
