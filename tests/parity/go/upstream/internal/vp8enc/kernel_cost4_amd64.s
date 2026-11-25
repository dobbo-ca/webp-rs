//go:build !purego

#include "go_asm.h"
#include "textflag.h"

// cost4Asm constants: n*512 per word lane n.
DATA c4n<>+0(SB)/8, $0x0600040002000000
DATA c4n<>+8(SB)/8, $0x0e000c000a000800
DATA c4n<>+16(SB)/8, $0x1600140012001000
DATA c4n<>+24(SB)/8, $0x1e001c001a001800
GLOBL c4n<>(SB), RODATA|NOPTR, $32

DATA c4two<>+0(SB)/8, $0x0002000200020002
DATA c4two<>+8(SB)/8, $0x0002000200020002
DATA c4two<>+16(SB)/8, $0x0002000200020002
DATA c4two<>+24(SB)/8, $0x0002000200020002
GLOBL c4two<>(SB), RODATA|NOPTR, $32

DATA c4lim<>+0(SB)/8, $0x0043004300430043
DATA c4lim<>+8(SB)/8, $0x0043004300430043
DATA c4lim<>+16(SB)/8, $0x0043004300430043
DATA c4lim<>+24(SB)/8, $0x0043004300430043
GLOBL c4lim<>(SB), RODATA|NOPTR, $32

// 16 words 0 then 16 words 127; read at word 15-last, lanes n > last
// get 127, a zero cost entry.
DATA c4m<>+0(SB)/8, $0
DATA c4m<>+8(SB)/8, $0
DATA c4m<>+16(SB)/8, $0
DATA c4m<>+24(SB)/8, $0
DATA c4m<>+32(SB)/8, $0x007f007f007f007f
DATA c4m<>+40(SB)/8, $0x007f007f007f007f
DATA c4m<>+48(SB)/8, $0x007f007f007f007f
DATA c4m<>+56(SB)/8, $0x007f007f007f007f
GLOBL c4m<>(SB), RODATA|NOPTR, $64

// Byte n*4+c is (bands[n]*3+c)*11 for n 1..15, c 0..2.
DATA c4e<>+0(SB)/8, $0x00372c2100000000
DATA c4e<>+8(SB)/8, $0x00796e6300584d42
DATA c4e<>+16(SB)/8, $0x009a8f8400dcd1c6
DATA c4e<>+24(SB)/8, $0x00dcd1c600bbb0a5
DATA c4e<>+32(SB)/8, $0x00dcd1c600dcd1c6
DATA c4e<>+40(SB)/8, $0x00dcd1c600dcd1c6
DATA c4e<>+48(SB)/8, $0x00dcd1c600dcd1c6
DATA c4e<>+56(SB)/8, $0x00fdf2e700dcd1c6
GLOBL c4e<>(SB), RODATA|NOPTR, $64

// One cost row of the gather, into acc.
#define C4ROW(off, acc) MOVWLZX off(SP), R11; MOVWLZX (R14)(R11*2), R11; ADDQ R11, acc

// func cost4Asm(s *costTabs, ctx, last int, lv *[16]int16) (c int)
// costGridAsm's 1x1 block at typ 3, first 0, ctx 0..2, with last
// passed in: -1 or lastNZ(lv, 0). The EOB term reads live probs.
TEXT ·cost4Asm(SB), NOSPLIT, $32-40
	MOVQ s+0(FP), R8
	MOVQ ctx+8(FP), CX
	MOVQ last+16(FP), DX
	MOVQ lv+24(FP), SI

	// DI = probs[3*264:], AX = p0 = probs[3*264+ctx*11].
	LEAQ    costTabs_probs+792(R8), DI
	LEAQ    (CX)(CX*4), AX
	LEAQ    (CX)(AX*2), AX
	MOVBLZX (DI)(AX*1), AX
	LEAQ    ·bitCost(SB), BX
	TESTQ   DX, DX
	JS      c4none

	// R9 = bitCost[1][p0] when ctx == 0.
	MOVQ    2048(BX)(AX*8), R9
	XORL    AX, AX
	TESTQ   CX, CX
	CMOVQNE AX, R9

	// Row index n*512 + ctx*128 + min(|v|, 67), where ctx is
	// min(|v|, 2) of the previous position, or ctx at 0.
	VMOVDQU    (SI), Y0
	VPABSW     Y0, Y1
	VPMINUW    c4two<>(SB), Y1, Y2
	VPMINUW    c4lim<>(SB), Y1, Y3
	VPERM2I128 $0x08, Y2, Y2, Y4
	VPALIGNR   $14, Y4, Y2, Y2
	VMOVD      CX, X4
	VPOR       Y4, Y2, Y2
	VPSLLW     $7, Y2, Y2
	VPADDW     Y3, Y2, Y2
	VPADDW     c4n<>(SB), Y2, Y2
	MOVQ       DX, AX
	NEGQ       AX
	LEAQ       c4m<>+30(SB), R10
	VPOR       (R10)(AX*2), Y2, Y2
	VMOVDQU    Y2, 0(SP)

	// Sum the 16 rows of cost[3], two chains.
	LEAQ  costTabs_cost+49152(R8), R14
	XORL  AX, AX
	XORL  R12, R12
	C4ROW(0, AX)
	C4ROW(2, R12)
	C4ROW(4, AX)
	C4ROW(6, R12)
	C4ROW(8, AX)
	C4ROW(10, R12)
	C4ROW(12, AX)
	C4ROW(14, R12)
	C4ROW(16, AX)
	C4ROW(18, R12)
	C4ROW(20, AX)
	C4ROW(22, R12)
	C4ROW(24, AX)
	C4ROW(26, R12)
	C4ROW(28, AX)
	C4ROW(30, R12)
	ADDQ  AX, R9
	ADDQ  R12, R9

	// Levels above 67 add fixedCost[v] - fixedCost[67].
	VPCMPEQW  Y3, Y1, Y5
	VPMOVMSKB Y5, R11
	CMPL      R11, $-1
	JNE       c4big

c4tail:
	// Row last+1 >> 7 is (last+1)*4 + min(|lv[last]|, 2).
	CMPQ    DX, $15
	JEQ     c4done
	MOVWLZX 2(SP)(DX*2), AX
	SHRL    $7, AX
	LEAQ    c4e<>(SB), R11
	MOVBLZX (R11)(AX*1), AX
	MOVBLZX (DI)(AX*1), AX
	ADDQ    (BX)(AX*8), R9

c4done:
	MOVQ R9, c+32(FP)
	VZEROUPPER
	RET

c4none:
	MOVQ (BX)(AX*8), AX
	MOVQ AX, c+32(FP)
	RET

c4big:
	LEAQ ·fixedCost(SB), R10
	XORL CX, CX

c4bloop:
	MOVWQSX (SI)(CX*2), R11
	TESTQ   R11, R11
	JNS     c4bpos
	NEGQ    R11

c4bpos:
	CMPQ R11, $67
	JLS  c4bnext
	ANDQ $2047, R11
	ADDQ (R10)(R11*8), R9
	SUBQ 536(R10), R9

c4bnext:
	INCQ CX
	CMPQ CX, DX
	JLE  c4bloop
	JMP  c4tail
