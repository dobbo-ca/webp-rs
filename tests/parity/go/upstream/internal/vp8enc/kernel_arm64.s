//go:build !purego

#include "go_asm.h"
#include "textflag.h"

// V21/V23/V25 carry DC in lane 0.
// level = (|v|+sharpen)*iq+bias >> 17
// Restore sign.
// in = level*q, truncated like int16(dq).
// 16-bit v-dq is exact: |v-dq| < 2^15.
// Zigzag byte indices into [V8, V9].
// last = max(n+1 where nonzero) - 1
#define QUANTK \
	MOVWU quantMatrix_q+4(R0), R3 \
	MOVWU quantMatrix_q(R0), R4 \
	VDUP  R3, V20.H8 \
	VMOV  V20.B16, V21.B16 \
	VMOV  R4, V21.H[0] \
	MOVWU quantMatrix_iq+4(R0), R3 \
	MOVWU quantMatrix_iq(R0), R4 \
	VDUP  R3, V22.H8 \
	VMOV  V22.B16, V23.B16 \
	VMOV  R4, V23.H[0] \
	MOVWU quantMatrix_bias+4(R0), R3 \
	MOVWU quantMatrix_bias(R0), R4 \
	VDUP  R3, V24.S4 \
	VMOV  V24.B16, V25.B16 \
	VMOV  R4, V25.S[0] \
	ADD   $quantMatrix_sharpen, R0, R5 \
	VLD1  (R5), [V16.S4, V17.S4, V18.S4, V19.S4] \
	VUZP1 V17.H8, V16.H8, V16.H8 \
	VUZP1 V19.H8, V18.H8, V17.H8

// QUANTB quantizes V0, V1 with the QUANTK constants: dequantized
// coefficients to (R1), levels to (R2), last in R7, SSE in R6.
#define QUANTB \
	VABS    V0.H8, V2.H8 \
	VABS    V1.H8, V3.H8 \
	VADD    V16.H8, V2.H8, V2.H8 \
	VADD    V17.H8, V3.H8, V3.H8 \
	VUMULL  V23.H4, V2.H4, V4.S4 \
	VUMULL2 V23.H8, V2.H8, V5.S4 \
	VUMULL  V22.H4, V3.H4, V6.S4 \
	VUMULL2 V22.H8, V3.H8, V7.S4 \
	VADD    V25.S4, V4.S4, V4.S4 \
	VADD    V24.S4, V5.S4, V5.S4 \
	VADD    V24.S4, V6.S4, V6.S4 \
	VADD    V24.S4, V7.S4, V7.S4 \
	VSHRN   $16, V4.S4, V8.H4 \
	VSHRN2  $16, V5.S4, V8.H8 \
	VSHRN   $16, V6.S4, V9.H4 \
	VSHRN2  $16, V7.S4, V9.H8 \
	VUSHR   $1, V8.H8, V8.H8 \
	VUSHR   $1, V9.H8, V9.H8 \
	VSSHR $15, V0.H8, V10.H8 \
	VSSHR $15, V1.H8, V11.H8 \
	VEOR  V10.B16, V8.B16, V8.B16 \
	VEOR  V11.B16, V9.B16, V9.B16 \
	VSUB  V10.H8, V8.H8, V8.H8 \
	VSUB  V11.H8, V9.H8, V9.H8 \
	VMUL V21.H8, V8.H8, V12.H8 \
	VMUL V20.H8, V9.H8, V13.H8 \
	VST1 [V12.H8, V13.H8], (R1) \
	VSUB    V12.H8, V0.H8, V14.H8 \
	VSUB    V13.H8, V1.H8, V15.H8 \
	VSMULL  V14.H4, V14.H4, V26.S4 \
	VSMULL2 V14.H8, V14.H8, V27.S4 \
	VSMLAL  V15.H4, V15.H4, V26.S4 \
	VSMLAL2 V15.H8, V15.H8, V27.S4 \
	VADD    V27.S4, V26.S4, V26.S4 \
	VADDV   V26.S4, V26 \
	VMOV    V26.S[0], R6 \
	VMOVQ $0x1110090803020100, $0x0d0c070605040b0a, V28 \
	VMOVQ $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29 \
	VTBL  V28.B16, [V8.B16, V9.B16], V10.B16 \
	VTBL  V29.B16, [V8.B16, V9.B16], V11.B16 \
	VST1  [V10.H8, V11.H8], (R2) \
	VMOVQ  $0x0004000300020001, $0x0008000700060005, V30 \
	VMOVQ  $0x000c000b000a0009, $0x0010000f000e000d, V31 \
	VCMTST V10.H8, V10.H8, V10.H8 \
	VCMTST V11.H8, V11.H8, V11.H8 \
	VAND   V30.B16, V10.B16, V10.B16 \
	VAND   V31.B16, V11.B16, V11.B16 \
	VUMAX  V11.H8, V10.H8, V10.H8 \
	VUMAXV V10.H8, V10 \
	VMOV   V10.H[0], R7 \
	SUB    $1, R7

#define QUANT \
	QUANTK \
	QUANTB

// func quantAsm(m *quantMatrix, in, out *[16]int16) (last, d int)
TEXT ·quantAsm(SB), NOSPLIT, $0-40
	MOVD m+0(FP), R0
	MOVD in+8(FP), R1
	MOVD out+16(FP), R2

	VLD1 (R1), [V0.H8, V1.H8]
	QUANT

	MOVD R7, last+24(FP)
	MOVD R6, d+32(FP)
	RET

// FDCT34 is the fdct of src at R12 (advanced) minus the prediction
// rows 0, 2 in V3 and 1, 3 in V4, raster order in V0, V1.
#define FDCT34K \
	VMOVQ $0x08a908a908a908a9, $0x08a908a908a908a9, V28 \
	VMOVQ $0x14e814e814e814e8, $0x14e814e814e814e8, V29 \
	VMOVQ $0x0000071400000714, $0x0000071400000714, V30 \
	VMOVQ $0x000003a9000003a9, $0x000003a9000003a9, V31 \
	VMOVQ $0x0007000700070007, $0x0007000700070007, V24 \
	VMOVQ $0x00002ee000002ee0, $0x00002ee000002ee0, V25 \
	VMOVQ $0x0000c7380000c738, $0x0000c7380000c738, V26

#define FDCT34B \
	MOVWU (R12), R6 \
	ADD   R13, R12, R12 \
	MOVWU (R12), R7 \
	ADD   R13, R12, R12 \
	MOVWU (R12), R8 \
	ADD   R13, R12, R12 \
	MOVWU (R12), R9 \
	ORR   R8<<32, R6, R6 \
	ORR   R9<<32, R7, R7 \
	VMOV  R6, V2.D[0] \
	VMOV  R7, V2.D[1] \
	VUXTL  V2.B8, V5.H8 \
	VUXTL2 V2.B16, V6.H8 \
	VUXTL  V3.B8, V3.H8 \
	VUXTL  V4.B8, V4.H8 \
	VSUB   V3.H8, V5.H8, V5.H8 \
	VSUB   V4.H8, V6.H8, V6.H8 \
	VTRN1 V6.H8, V5.H8, V7.H8 \
	VTRN2 V6.H8, V5.H8, V8.H8 \
	VUZP1 V8.S4, V7.S4, V9.S4 \
	VUZP2 V7.S4, V8.S4, V10.S4 \
	VADD    V10.H8, V9.H8, V11.H8 \
	VSUB    V10.H8, V9.H8, V12.H8 \
	VEXT    $8, V11.B16, V11.B16, V13.B16 \
	VADD    V13.H8, V11.H8, V14.H8 \
	VSUB    V13.H8, V11.H8, V15.H8 \
	VZIP1   V15.D2, V14.D2, V14.D2 \
	VSHL    $3, V14.H8, V14.H8 \
	VSMULL2 V28.H8, V12.H8, V16.S4 \
	VSMULL  V28.H4, V12.H4, V17.S4 \
	VSMLAL  V29.H4, V12.H4, V16.S4 \
	VSMLSL2 V29.H8, V12.H8, V17.S4 \
	VADD    V30.S4, V16.S4, V16.S4 \
	VADD    V31.S4, V17.S4, V17.S4 \
	VSHRN   $9, V16.S4, V18.H4 \
	VSHRN2  $9, V17.S4, V18.H8 \
	VTRN1 V18.H8, V14.H8, V7.H8 \
	VTRN2 V18.H8, V14.H8, V8.H8 \
	VUZP1 V8.S4, V7.S4, V9.S4 \
	VUZP2 V7.S4, V8.S4, V10.S4 \
	VADD    V10.H8, V9.H8, V11.H8 \
	VSUB    V10.H8, V9.H8, V12.H8 \
	VEXT    $8, V11.B16, V11.B16, V13.B16 \
	VADD    V24.H8, V11.H8, V11.H8 \
	VADD    V13.H8, V11.H8, V14.H8 \
	VSUB    V13.H8, V11.H8, V15.H8 \
	VZIP1   V15.D2, V14.D2, V14.D2 \
	VSSHR   $4, V14.H8, V14.H8 \
	VSMULL2 V28.H8, V12.H8, V16.S4 \
	VSMULL  V28.H4, V12.H4, V17.S4 \
	VSMLAL  V29.H4, V12.H4, V16.S4 \
	VSMLSL2 V29.H8, V12.H8, V17.S4 \
	VADD    V25.S4, V16.S4, V16.S4 \
	VADD    V26.S4, V17.S4, V17.S4 \
	VSHRN   $16, V16.S4, V18.H4 \
	VSHRN2  $16, V17.S4, V18.H8 \
	VCMTST  V12.H4, V12.H4, V19.H4 \
	VSUB    V19.H8, V18.H8, V18.H8 \
	VZIP1   V18.D2, V14.D2, V0.D2 \
	VZIP2   V18.D2, V14.D2, V1.D2

#define FDCT34 \
	FDCT34K \
	FDCT34B

// FDCT is FDCT34 with the prediction pr at R11.
#define FDCT \
	VLD2 (R11), [V3.S2, V4.S2] \
	FDCT34

// One fused trial: src at R12 (advanced), pr at R11.
#define RESQ \
	FDCT \
	QUANT

// func residualQuantAsm(m *quantMatrix, pr *[4]uint32, src *uint8, stride int, in, out *[16]int16) (last, d int)
TEXT ·residualQuantAsm(SB), NOSPLIT, $0-64
	MOVD m+0(FP), R0
	MOVD pr+8(FP), R11
	MOVD src+16(FP), R12
	MOVD stride+24(FP), R13
	MOVD in+32(FP), R1
	MOVD out+40(FP), R2

	RESQ

	MOVD R7, last+48(FP)
	MOVD R6, d+56(FP)
	RET

// func residualAsm(pred *uint8, src *uint8, stride int, out *[16]int16)
TEXT ·residualAsm(SB), NOSPLIT, $0-32
	MOVD  pred+0(FP), R11
	MOVD  src+8(FP), R12
	MOVD  stride+16(FP), R13
	MOVD  out+24(FP), R2
	MOVWU (R11), R14
	MOVWU 64(R11), R15
	ORR   R15<<32, R14, R14
	VMOV  R14, V3.D[0]
	MOVWU 32(R11), R14
	MOVWU 96(R11), R15
	ORR   R15<<32, R14, R14
	VMOV  R14, V4.D[0]
	FDCT34
	VST1 [V0.H8, V1.H8], (R2)
	RET

// func levelCostsAsm(t *[16][4][128]uint16, lv *[16]int16, first, last, ctx int) (c, next int)
// R0 walks the 1024-byte position rows, R5 is the row plus ctx*256.
TEXT ·levelCostsAsm(SB), NOSPLIT, $0-56
	MOVD t+0(FP), R0
	MOVD lv+8(FP), R1
	MOVD first+16(FP), R2
	MOVD last+24(FP), R3
	MOVD ctx+32(FP), R4
	ADD  R2<<10, R0, R0
	ADD  R4<<8, R0, R5
	MOVD $0, R6
	MOVD $·fixedCost(SB), R7
	MOVD $512, R8

lcloop:
	MOVH  (R1)(R2<<1), R9
	CMP   $0, R9
	CNEG  LT, R9, R9
	CMP   $67, R9
	BHI   lcbig

lccost:
	MOVHU (R5)(R9<<1), R10
	ADD   R10, R6, R6
	LSL   $8, R9, R11
	CMP   R8, R11
	CSEL  HI, R8, R11, R11
	ADD   $1024, R0, R0
	ADD   R11, R0, R5
	ADD   $1, R2, R2
	CMP   R3, R2
	BLE   lcloop
	LSR   $8, R11, R11
	MOVD  R6, c+40(FP)
	MOVD  R11, next+48(FP)
	RET

	// Level above 67: its fixed cost beyond fixedCost[67], then row 67.
lcbig:
	AND  $2047, R9, R9
	MOVD (R7)(R9<<3), R10
	ADD  R10, R6, R6
	MOVD 536(R7), R10
	SUB  R10, R6, R6
	MOVD $67, R9
	B    lccost

// func sse4x10Asm(d *[10]int32, pr *[10][4]uint32, src *uint8, stride int)
TEXT ·sse4x10Asm(SB), NOSPLIT, $0-32
	MOVD  d+0(FP), R0
	MOVD  pr+8(FP), R1
	MOVD  src+16(FP), R2
	MOVD  stride+24(FP), R4
	// Rows read here: Go stores stall VLD1.
	ADD   R4<<1, R2, R5
	MOVWU (R2), R6
	MOVWU (R2)(R4), R7
	MOVWU (R5), R8
	MOVWU (R5)(R4), R9
	ORR   R7<<32, R6, R6
	ORR   R9<<32, R8, R8
	VMOV  R6, V0.D[0]
	VMOV  R8, V0.D[1]
	MOVD  $10, R3

loop:
	// No VUABD in Go asm: max minus min.
	VLD1.P  16(R1), [V1.B16]
	VUMAX   V1.B16, V0.B16, V2.B16
	VUMIN   V1.B16, V0.B16, V3.B16
	VSUB    V3.B16, V2.B16, V2.B16
	VUMULL  V2.B8, V2.B8, V4.H8
	VUMULL2 V2.B16, V2.B16, V5.H8
	VUADDLV V4.H8, V6
	VUADDLV V5.H8, V7
	VADD    V7.S4, V6.S4, V6.S4
	FMOVS.P F6, 4(R0)
	SUBS    $1, R3
	BNE     loop
	RET

// func sseAsm(src *uint8, stride int, pred *uint8, w, h int) int
TEXT ·sseAsm(SB), NOSPLIT, $0-48
	MOVD  src+0(FP), R0
	MOVD  stride+8(FP), R1
	MOVD  pred+16(FP), R2
	MOVD  w+24(FP), R3
	MOVD  h+32(FP), R4
	VEOR  V16.B16, V16.B16, V16.B16
	VEOR  V17.B16, V17.B16, V17.B16
	CMP   $16, R3
	BNE   row8

row16:
	VLD1    (R0), [V0.B16]
	VLD1    (R2), [V1.B16]
	ADD     R1, R0
	ADD     $32, R2
	VUMAX   V1.B16, V0.B16, V2.B16
	VUMIN   V1.B16, V0.B16, V3.B16
	VSUB    V3.B16, V2.B16, V2.B16
	VUMULL  V2.B8, V2.B8, V4.H8
	VUMULL2 V2.B16, V2.B16, V5.H8
	VUADDW  V4.H4, V16.S4, V16.S4
	VUADDW2 V4.H8, V17.S4, V17.S4
	VUADDW  V5.H4, V16.S4, V16.S4
	VUADDW2 V5.H8, V17.S4, V17.S4
	SUBS    $1, R4
	BNE     row16
	B       done

row8:
	VLD1    (R0), [V0.B8]
	// Two 4-byte loads: each matches one block store.
	FMOVS   (R2), F1
	MOVWU   4(R2), R5
	VMOV    R5, V1.S[1]
	ADD     R1, R0
	ADD     $32, R2
	VUMAX   V1.B8, V0.B8, V2.B8
	VUMIN   V1.B8, V0.B8, V3.B8
	VSUB    V3.B8, V2.B8, V2.B8
	VUMULL  V2.B8, V2.B8, V4.H8
	VUADDW  V4.H4, V16.S4, V16.S4
	VUADDW2 V4.H8, V17.S4, V17.S4
	SUBS    $1, R4
	BNE     row8

done:
	VADD  V17.S4, V16.S4, V16.S4
	VADDV V16.S4, V16
	VMOV  V16.S[0], R5
	MOVD  R5, ret+40(FP)
	RET

// Adds (V0-pred)^2 over 16 bytes into lo, hi.
#define ESSE16(pred, lo, hi) \
	VUMAX   pred.B16, V0.B16, V4.B16 \
	VUMIN   pred.B16, V0.B16, V5.B16 \
	VSUB    V5.B16, V4.B16, V4.B16 \
	VUMULL  V4.B8, V4.B8, V5.H8 \
	VUMULL2 V4.B16, V4.B16, V6.H8 \
	VUADDW  V5.H4, lo.S4, lo.S4 \
	VUADDW2 V5.H8, hi.S4, hi.S4 \
	VUADDW  V6.H4, lo.S4, lo.S4 \
	VUADDW2 V6.H8, hi.S4, hi.S4

// Adds (V0-pred)^2 over 8 bytes into lo, hi.
#define ESSE8(pred, lo, hi) \
	VUMAX   pred.B8, V0.B8, V4.B8 \
	VUMIN   pred.B8, V0.B8, V5.B8 \
	VSUB    V5.B8, V4.B8, V4.B8 \
	VUMULL  V4.B8, V4.B8, V5.H8 \
	VUADDW  V5.H4, lo.S4, lo.S4 \
	VUADDW2 V5.H8, hi.S4, hi.S4

// Sums lo, hi into the int at off(R0).
#define EHSUM(lo, hi, off) \
	VADD  hi.S4, lo.S4, lo.S4 \
	VADDV lo.S4, lo \
	VMOV  lo.S[0], R5 \
	MOVD  R5, off(R0)

// Adds the left byte at off(p) to R10.
#define LADD(p, off) \
	MOVBU off(p), R11 \
	ADD   R11, R10, R10

// Sets R9 to dcAvg of the n x n block at p, edges e bit 0 left
// and bit 1 top; uses V0, R10-R13.
#define DCAVG(p, n, e) \
	MOVBU   -1(p), R10 \
	LADD(p, 31) \
	LADD(p, 63) \
	LADD(p, 95) \
	LADD(p, 127) \
	LADD(p, 159) \
	LADD(p, 191) \
	LADD(p, 223) \
	SUB     $32, p, R13 \
	CMP     $16, n \
	BNE     dctop8 \
	LADD(p, 255) \
	LADD(p, 287) \
	LADD(p, 319) \
	LADD(p, 351) \
	LADD(p, 383) \
	LADD(p, 415) \
	LADD(p, 447) \
	LADD(p, 479) \
	VLD1    (R13), [V0.B16] \
	B       dctop \
dctop8: \
	VLD1    (R13), [V0.B8] \
dctop: \
	VUADDLV V0.B16, V0 \
	VMOV    V0.H[0], R12 \
	TST     $1, e \
	CSEL    EQ, ZR, R10, R10 \
	TST     $2, e \
	CSEL    EQ, ZR, R12, R12 \
	ADD     R12, R10, R9 \
	LSR     $4, n, R12 \
	ADD     $2, R12, R12 \
	AND     $1, e, R11 \
	ADD     R11, R12, R12 \
	ADD     e>>1, R12, R12 \
	MOVD    $1, R11 \
	LSL     R12, R11, R11 \
	ADD     R11>>1, R9, R9 \
	LSR     R12, R9, R9 \
	MOVD    $128, R11 \
	CMP     $0, e \
	CSEL    EQ, R11, R9, R9

// func edgeSSEAsm(d *[4]int, src *uint8, stride int, p *uint8, n, edges int)
TEXT ·edgeSSEAsm(SB), NOSPLIT, $0-48
	MOVD   d+0(FP), R0
	MOVD   src+8(FP), R1
	MOVD   stride+16(FP), R2
	MOVD   p+24(FP), R3
	MOVD   n+32(FP), R4
	MOVD   edges+40(FP), R5
	DCAVG(R3, R4, R5)
	VDUP   R9, V16.B16
	MOVBU  -33(R3), R5
	VDUP   R5, V2.H8
	SUB    $32, R3, R6
	VLD1   (R6), [V17.B16]
	// top - corner fits int16.
	VUXTL  V17.B8, V18.H8
	VUXTL2 V17.B16, V19.H8
	VSUB   V2.H8, V18.H8, V18.H8
	VSUB   V2.H8, V19.H8, V19.H8
	VEOR   V24.B16, V24.B16, V24.B16
	VEOR   V25.B16, V25.B16, V25.B16
	VEOR   V26.B16, V26.B16, V26.B16
	VEOR   V27.B16, V27.B16, V27.B16
	VEOR   V28.B16, V28.B16, V28.B16
	VEOR   V29.B16, V29.B16, V29.B16
	VEOR   V30.B16, V30.B16, V30.B16
	VEOR   V31.B16, V31.B16, V31.B16
	MOVBU  ·useDot(SB), R6
	CBNZ   R6, dot
	CMP    $16, R4
	BNE    erow8

erow16:
	VLD1     (R1), [V0.B16]
	ADD      R2, R1
	MOVBU    -1(R3), R5
	ADD      $32, R3
	VDUP     R5, V1.B16
	VDUP     R5, V3.H8
	// Saturating narrow is clip8.
	VADD     V3.H8, V18.H8, V7.H8
	VADD     V3.H8, V19.H8, V8.H8
	VSQXTUN  V7.H8, V9.B8
	VSQXTUN2 V8.H8, V9.B16
	ESSE16(V16, V24, V25)
	ESSE16(V9, V26, V27)
	ESSE16(V17, V28, V29)
	ESSE16(V1, V30, V31)
	SUBS     $1, R4
	BNE      erow16
	B        edone

erow8:
	VLD1    (R1), [V0.B8]
	ADD     R2, R1
	MOVBU   -1(R3), R5
	ADD     $32, R3
	VDUP    R5, V1.B16
	VDUP    R5, V3.H8
	VADD    V3.H8, V18.H8, V7.H8
	VSQXTUN V7.H8, V9.B8
	ESSE8(V16, V24, V25)
	ESSE8(V9, V26, V27)
	ESSE8(V17, V28, V29)
	ESSE8(V1, V30, V31)
	SUBS    $1, R4
	BNE     erow8

edone:
	EHSUM(V24, V25, 0)
	EHSUM(V26, V27, 8)
	EHSUM(V28, V29, 16)
	EHSUM(V30, V31, 24)
	RET

// UDOT of |s-p| with itself sums four squares per lane.
dot:
	SUB  $1, R3, R7
	MOVD $32, R8
	CMP  $16, R4
	BNE  drow8

drow16:
	VLD1.P   (R1)(R2), [V0.B16]
	VLD1R.P  (R7)(R8), [V1.B16]
	// T - A plus L, exact in int16.
	VUADDW   V1.B8, V18.H8, V7.H8
	VUADDW2  V1.B16, V19.H8, V8.H8
	VSQXTUN  V7.H8, V9.B8
	VSQXTUN2 V8.H8, V9.B16
	WORD     $0x6e307404 // uabd v4.16b, v0.16b, v16.16b
	WORD     $0x6e297405 // uabd v5.16b, v0.16b, v9.16b
	WORD     $0x6e317406 // uabd v6.16b, v0.16b, v17.16b
	WORD     $0x6e21740a // uabd v10.16b, v0.16b, v1.16b
	WORD     $0x6e849498 // udot v24.4s, v4.16b, v4.16b
	WORD     $0x6e8594ba // udot v26.4s, v5.16b, v5.16b
	WORD     $0x6e8694dc // udot v28.4s, v6.16b, v6.16b
	WORD     $0x6e8a955e // udot v30.4s, v10.16b, v10.16b
	SUBS     $1, R4
	BNE      drow16
	B        dsum

drow8:
	VLD1.P  (R1)(R2), [V0.B8]
	VLD1R.P (R7)(R8), [V1.B8]
	VUADDW  V1.B8, V18.H8, V7.H8
	VSQXTUN V7.H8, V9.B8
	WORD    $0x2e307404 // uabd v4.8b, v0.8b, v16.8b
	WORD    $0x2e297405 // uabd v5.8b, v0.8b, v9.8b
	WORD    $0x2e317406 // uabd v6.8b, v0.8b, v17.8b
	WORD    $0x2e21740a // uabd v10.8b, v0.8b, v1.8b
	WORD    $0x2e849498 // udot v24.2s, v4.8b, v4.8b
	WORD    $0x2e8594ba // udot v26.2s, v5.8b, v5.8b
	WORD    $0x2e8694dc // udot v28.2s, v6.8b, v6.8b
	WORD    $0x2e8a955e // udot v30.2s, v10.8b, v10.8b
	SUBS    $1, R4
	BNE     drow8

dsum:
	// Pairwise adds leave d[0..3] in lanes.
	VADDP  V26.S4, V24.S4, V24.S4
	VADDP  V30.S4, V28.S4, V28.S4
	VADDP  V28.S4, V24.S4, V24.S4
	VUXTL  V24.S2, V25.D2
	VUXTL2 V24.S4, V26.D2
	VST1   [V25.D2, V26.D2], (R0)
	RET

// func predTMAsm(dst *uint8, n int)
TEXT ·predTMAsm(SB), NOSPLIT, $0-16
	MOVD   dst+0(FP), R0
	MOVD   n+8(FP), R1
	MOVBU  -33(R0), R2
	VDUP   R2, V2.H8
	SUB    $32, R0, R3
	VLD1   (R3), [V0.B16]
	MOVD   R1, R4

	// top - corner fits int16.
	VUXTL  V0.B8, V16.H8
	VUXTL2 V0.B16, V17.H8
	VSUB   V2.H8, V16.H8, V16.H8
	VSUB   V2.H8, V17.H8, V17.H8
	CMP    $16, R1
	BNE    row8

row16:
	MOVBU    -1(R0), R2
	VDUP     R2, V3.H8
	VADD     V3.H8, V16.H8, V4.H8
	VADD     V3.H8, V17.H8, V5.H8
	// Saturating narrow is clip8.
	VSQXTUN  V4.H8, V6.B8
	VSQXTUN2 V5.H8, V6.B16
	VST1     [V6.B16], (R0)
	ADD      $32, R0
	SUBS     $1, R4
	BNE      row16
	RET

row8:
	MOVBU   -1(R0), R2
	VDUP    R2, V3.H8
	VADD    V3.H8, V16.H8, V4.H8
	VSQXTUN V4.H8, V6.B8
	VST1    [V6.B8], (R0)
	ADD     $32, R0
	SUBS    $1, R4
	BNE     row8
	RET

// func predDCAsm(dst *uint8, n, edges int)
TEXT ·predDCAsm(SB), NOSPLIT, $0-24
	MOVD  dst+0(FP), R0
	MOVD  n+8(FP), R1
	MOVD  edges+16(FP), R2
	DCAVG(R0, R1, R2)
	VDUP  R9, V0.B16
	CMP   $16, R1
	BNE   dc8
	FMOVQ F0, (R0)
	FMOVQ F0, 32(R0)
	FMOVQ F0, 64(R0)
	FMOVQ F0, 96(R0)
	FMOVQ F0, 128(R0)
	FMOVQ F0, 160(R0)
	FMOVQ F0, 192(R0)
	FMOVQ F0, 224(R0)
	FMOVQ F0, 256(R0)
	FMOVQ F0, 288(R0)
	FMOVQ F0, 320(R0)
	FMOVQ F0, 352(R0)
	FMOVQ F0, 384(R0)
	FMOVQ F0, 416(R0)
	FMOVQ F0, 448(R0)
	FMOVQ F0, 480(R0)
	RET

dc8:
	FMOVD F0, (R0)
	FMOVD F0, 32(R0)
	FMOVD F0, 64(R0)
	FMOVD F0, 96(R0)
	FMOVD F0, 128(R0)
	FMOVD F0, 160(R0)
	FMOVD F0, 192(R0)
	FMOVD F0, 224(R0)
	RET

// func predHEAsm(dst *uint8, n int)
TEXT ·predHEAsm(SB), NOSPLIT, $0-16
	MOVD dst+0(FP), R0
	MOVD n+8(FP), R1
	CMP  $16, R1
	BNE  he8

he16:
	MOVBU -1(R0), R2
	VDUP  R2, V0.B16
	VST1  [V0.B16], (R0)
	ADD   $32, R0
	SUBS  $1, R1
	BNE   he16
	RET

he8:
	MOVBU -1(R0), R2
	VDUP  R2, V0.B8
	VST1  [V0.B8], (R0)
	ADD   $32, R0
	SUBS  $1, R1
	BNE   he8
	RET


// pred4All TBL indices: V, DC, TM, then modes.
// Indices 16 and up pick from A2.
DATA p4n<>+0(SB)/8, $0x020100090a0b0c0c
DATA p4n<>+8(SB)/8, $0x8008080706050403
DATA p4n<>+16(SB)/8, $0x0908070604030201
DATA p4n<>+24(SB)/8, $0x8080808080808080
DATA p4n<>+32(SB)/8, $0x8009800880078006
DATA p4n<>+40(SB)/8, $0x8009800880078006
DATA p4n<>+48(SB)/8, $0x8005800580058005
DATA p4n<>+56(SB)/8, $0x8005800580058005
DATA p4n<>+64(SB)/8, $0x8004800480048004
DATA p4n<>+72(SB)/8, $0x8003800380038003
DATA p4n<>+80(SB)/8, $0x8002800280028002
DATA p4n<>+88(SB)/8, $0x8001800180018001
DATA p4n<>+96(SB)/8, $0x0807060508070605
DATA p4n<>+104(SB)/8, $0x0807060508070605
DATA p4n<>+112(SB)/8, $0x0202020203030303
DATA p4n<>+120(SB)/8, $0x0000000001010101
DATA p4n<>+128(SB)/8, $0x0605040307060504
DATA p4n<>+136(SB)/8, $0x0403020105040302
DATA p4n<>+144(SB)/8, $0x0706050418171615
DATA p4n<>+152(SB)/8, $0x0605040217161503
DATA p4n<>+160(SB)/8, $0x0a09080709080706
DATA p4n<>+168(SB)/8, $0x0c0b0a090b0a0908
DATA p4n<>+176(SB)/8, $0x0908070619181716
DATA p4n<>+184(SB)/8, $0x0b0908070a191817
DATA p4n<>+192(SB)/8, $0x0414031306050414
DATA p4n<>+200(SB)/8, $0x0212011103130212
DATA p4n<>+208(SB)/8, $0x0011011201120213
DATA p4n<>+216(SB)/8, $0x1010101010100011
GLOBL p4n<>(SB), RODATA|NOPTR, $224

// func pred4AllAsm(pr *[10][4]uint32, p *uint8)
TEXT ·pred4AllAsm(SB), NOSPLIT, $0-16
	MOVD   pr+0(FP), R0
	MOVD   p+8(FP), R1
	MOVD   $p4n<>(SB), R2
	VLD1.P 64(R2), [V6.B16, V7.B16, V8.B16, V9.B16]
	VLD1.P 64(R2), [V10.B16, V11.B16, V12.B16, V13.B16]
	VLD1.P 64(R2), [V26.B16, V27.B16, V28.B16, V29.B16]
	VLD1   (R2), [V30.B16, V31.B16]
	// A, T0..T7, then L0..L3 in bytes 9..12.
	// Bytes 0..8 only: skips the fresh put4 of block n-1.
	SUB    $33, R1, R3
	FMOVD  (R3), F3
	MOVBU  8(R3), R4
	VMOV   R4, V3.B[8]
	MOVBU  -1(R1), R4
	VMOV   R4, V3.B[9]
	MOVBU  31(R1), R4
	VMOV   R4, V3.B[10]
	MOVBU  63(R1), R4
	VMOV   R4, V3.B[11]
	MOVBU  95(R1), R4
	VMOV   R4, V3.B[12]
	// V = L3 L3 L2 L1 L0 A T0..T7 T7.
	VTBL   V6.B16, [V3.B16], V0.B16
	VEXT   $1, V0.B16, V0.B16, V1.B16
	VEXT   $2, V0.B16, V0.B16, V2.B16
	// A3[i] = avg3(V[i], V[i+1], V[i+2]).
	VUHADD  V2.B16, V0.B16, V4.B16
	VURHADD V1.B16, V4.B16, V4.B16
	// A2[i] = avg2(V[i], V[i+1]).
	VURHADD V1.B16, V0.B16, V5.B16

	// DC
	VTBL    V7.B16, [V0.B16], V3.B16
	VUADDLV V3.B16, V3
	VMOV    V3.H[0], R4
	ADD     $4, R4
	LSR     $3, R4
	VDUP    R4, V16.B16

	// TM: L - A + T fits int16.
	VTBL     V8.B16, [V0.B16], V3.B16
	VTBL     V9.B16, [V0.B16], V14.B16
	VSUB     V14.H8, V3.H8, V3.H8
	VTBL     V10.B16, [V0.B16], V14.B16
	VTBL     V11.B16, [V0.B16], V15.B16
	VADD     V3.H8, V14.H8, V14.H8
	VADD     V3.H8, V15.H8, V15.H8
	// Saturating narrow is clip8.
	VSQXTUN  V14.H8, V17.B8
	VSQXTUN2 V15.H8, V17.B16

	VTBL   V12.B16, [V4.B16], V18.B16
	VTBL   V13.B16, [V4.B16], V19.B16
	VTBL   V26.B16, [V4.B16], V20.B16
	VTBL   V27.B16, [V4.B16, V5.B16], V21.B16
	VTBL   V28.B16, [V4.B16], V22.B16
	VTBL   V29.B16, [V4.B16, V5.B16], V23.B16
	VTBL   V30.B16, [V4.B16, V5.B16], V24.B16
	VTBL   V31.B16, [V4.B16, V5.B16], V25.B16
	VST1.P [V16.B16, V17.B16, V18.B16, V19.B16], 64(R0)
	VST1.P [V20.B16, V21.B16, V22.B16, V23.B16], 64(R0)
	VST1   [V24.B16, V25.B16], (R0)
	RET


// func lumaAsm(dst, src *uint8, n int)
TEXT ·lumaAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD n+16(FP), R2
	CBZ  R2, done
	MOVD $16839, R3
	VDUP R3, V16.H8
	MOVD $33059, R3
	VDUP R3, V17.H8
	MOVD $6420, R3
	VDUP R3, V18.H8
	MOVD $(1<<15 + 16<<16), R3
	VDUP R3, V19.S4
loop:
	VLD4.P  64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]
	VUXTL   V0.B8, V4.H8
	VUXTL2  V0.B16, V5.H8
	VUXTL   V1.B8, V6.H8
	VUXTL2  V1.B16, V7.H8
	VUXTL   V2.B8, V8.H8
	VUXTL2  V2.B16, V9.H8
	VMOV    V19.B16, V10.B16
	VMOV    V19.B16, V11.B16
	VMOV    V19.B16, V12.B16
	VMOV    V19.B16, V13.B16
	VUMLAL  V16.H4, V4.H4, V10.S4
	VUMLAL2 V16.H8, V4.H8, V11.S4
	VUMLAL  V16.H4, V5.H4, V12.S4
	VUMLAL2 V16.H8, V5.H8, V13.S4
	VUMLAL  V17.H4, V6.H4, V10.S4
	VUMLAL2 V17.H8, V6.H8, V11.S4
	VUMLAL  V17.H4, V7.H4, V12.S4
	VUMLAL2 V17.H8, V7.H8, V13.S4
	VUMLAL  V18.H4, V8.H4, V10.S4
	VUMLAL2 V18.H8, V8.H8, V11.S4
	VUMLAL  V18.H4, V9.H4, V12.S4
	VUMLAL2 V18.H8, V9.H8, V13.S4
	VSHRN   $16, V10.S4, V14.H4
	VSHRN2  $16, V11.S4, V14.H8
	VSHRN   $16, V12.S4, V15.H4
	VSHRN2  $16, V13.S4, V15.H8
	VXTN    V14.H8, V20.B8
	VXTN2   V15.H8, V20.B16
	VST1.P  [V20.B16], 16(R0)
	SUBS    $16, R2, R2
	BNE     loop
done:
	RET

// IDCTADD adds the inverse DCT of (R0) to the 4x4 block at R1.
// Columns, transpose, rows, transpose back.
// Saturation equals clip8 of the int32 sum.
#define IDCTADD \
	MOVW $85627, R2 \
	MOVW $35468, R3 \
	MOVW $4, R4 \
	VDUP R2, V30.S4 \
	VDUP R3, V31.S4 \
	VDUP R4, V29.S4 \
	VLD1   (R0), [V0.H8, V1.H8] \
	VSXTL  V0.H4, V2.S4 \
	VSXTL2 V0.H8, V3.S4 \
	VSXTL  V1.H4, V4.S4 \
	VSXTL2 V1.H8, V5.S4 \
	VADD  V4.S4, V2.S4, V6.S4 \
	VSUB  V4.S4, V2.S4, V7.S4 \
	VMUL  V31.S4, V3.S4, V8.S4 \
	VMUL  V30.S4, V5.S4, V9.S4 \
	VMUL  V30.S4, V3.S4, V10.S4 \
	VMUL  V31.S4, V5.S4, V11.S4 \
	VSSHR $16, V8.S4, V8.S4 \
	VSSHR $16, V9.S4, V9.S4 \
	VSSHR $16, V10.S4, V10.S4 \
	VSSHR $16, V11.S4, V11.S4 \
	VSUB  V9.S4, V8.S4, V8.S4 \
	VADD  V11.S4, V10.S4, V10.S4 \
	VADD  V10.S4, V6.S4, V12.S4 \
	VADD  V8.S4, V7.S4, V13.S4 \
	VSUB  V8.S4, V7.S4, V14.S4 \
	VSUB  V10.S4, V6.S4, V15.S4 \
	VTRN1 V13.S4, V12.S4, V16.S4 \
	VTRN2 V13.S4, V12.S4, V17.S4 \
	VTRN1 V15.S4, V14.S4, V18.S4 \
	VTRN2 V15.S4, V14.S4, V19.S4 \
	VTRN1 V18.D2, V16.D2, V2.D2 \
	VTRN1 V19.D2, V17.D2, V3.D2 \
	VTRN2 V18.D2, V16.D2, V4.D2 \
	VTRN2 V19.D2, V17.D2, V5.D2 \
	VADD  V29.S4, V2.S4, V2.S4 \
	VADD  V4.S4, V2.S4, V6.S4 \
	VSUB  V4.S4, V2.S4, V7.S4 \
	VMUL  V31.S4, V3.S4, V8.S4 \
	VMUL  V30.S4, V5.S4, V9.S4 \
	VMUL  V30.S4, V3.S4, V10.S4 \
	VMUL  V31.S4, V5.S4, V11.S4 \
	VSSHR $16, V8.S4, V8.S4 \
	VSSHR $16, V9.S4, V9.S4 \
	VSSHR $16, V10.S4, V10.S4 \
	VSSHR $16, V11.S4, V11.S4 \
	VSUB  V9.S4, V8.S4, V8.S4 \
	VADD  V11.S4, V10.S4, V10.S4 \
	VADD  V10.S4, V6.S4, V12.S4 \
	VADD  V8.S4, V7.S4, V13.S4 \
	VSUB  V8.S4, V7.S4, V14.S4 \
	VSUB  V10.S4, V6.S4, V15.S4 \
	VSSHR $3, V12.S4, V12.S4 \
	VSSHR $3, V13.S4, V13.S4 \
	VSSHR $3, V14.S4, V14.S4 \
	VSSHR $3, V15.S4, V15.S4 \
	VTRN1 V13.S4, V12.S4, V16.S4 \
	VTRN2 V13.S4, V12.S4, V17.S4 \
	VTRN1 V15.S4, V14.S4, V18.S4 \
	VTRN2 V15.S4, V14.S4, V19.S4 \
	VTRN1 V18.D2, V16.D2, V2.D2 \
	VTRN1 V19.D2, V17.D2, V3.D2 \
	VTRN2 V18.D2, V16.D2, V4.D2 \
	VTRN2 V19.D2, V17.D2, V5.D2 \
	VSQXTN  V2.S4, V24.H4 \
	VSQXTN2 V3.S4, V24.H8 \
	VSQXTN  V4.S4, V25.H4 \
	VSQXTN2 V5.S4, V25.H8 \
	MOVWU   (R1), R4 \
	MOVWU   32(R1), R5 \
	ORR     R5<<32, R4, R4 \
	VMOV    R4, V26.D[0] \
	MOVWU   64(R1), R4 \
	MOVWU   96(R1), R5 \
	ORR     R5<<32, R4, R4 \
	VMOV    R4, V26.D[1] \
	VUXTL   V26.B8, V27.H8 \
	VUXTL2  V26.B16, V28.H8 \
	VSQADD  V27.H8, V24.H8, V24.H8 \
	VSQADD  V28.H8, V25.H8, V25.H8 \
	VSQXTUN  V24.H8, V26.B8 \
	VSQXTUN2 V25.H8, V26.B16 \
	VMOV    V26.D[0], R4 \
	MOVW    R4, (R1) \
	LSR     $32, R4 \
	MOVW    R4, 32(R1) \
	VMOV    V26.D[1], R4 \
	MOVW    R4, 64(R1) \
	LSR     $32, R4 \
	MOVW    R4, 96(R1)

// func idctAddAsm(c *[16]int16, dst *uint8)
TEXT ·idctAddAsm(SB), NOSPLIT, $0-16
	MOVD c+0(FP), R0
	MOVD dst+8(FP), R1
	IDCTADD
	RET

// func collectAsm(bins *[256]uint8, src *uint8, stride int, pred *uint8)
// Lane j of a half is block (j/4, j%4); TBL gathers and zero-extends.
TEXT ·collectAsm(SB), NOSPLIT, $0-32
	MOVD bins+0(FP), R3
	MOVD src+8(FP), R0
	MOVD stride+16(FP), R1
	MOVD pred+24(FP), R2

	VMOVQ $0xFF0CFF08FF04FF00, $0xFF1CFF18FF14FF10, V24
	VMOVQ $0xFF0DFF09FF05FF01, $0xFF1DFF19FF15FF11, V25
	VMOVQ $0xFF0EFF0AFF06FF02, $0xFF1EFF1AFF16FF12, V26
	VMOVQ $0xFF0FFF0BFF07FF03, $0xFF1FFF1BFF17FF13, V27
	VMOVQ $0x08a908a908a908a9, $0x08a908a908a908a9, V28
	VMOVQ $0x14e814e814e814e8, $0x14e814e814e814e8, V29
	LSL   $2, R1, R9
	MOVD  $2, R4

half:
// V0,V1 hold src rows r and r+4, V2,V3 the pred rows.
#define LOADY \
	ADD   R9, R0, R6 \
	VLD1  (R0), [V0.B16] \
	VLD1  (R6), [V1.B16] \
	VLD1  (R2), [V2.B16] \
	ADD   $128, R2, R7 \
	VLD1  (R7), [V3.B16] \
	ADD   R1, R0 \
	ADD   $32, R2

#define ROW(T0, T1, T2, T3) \
	VTBL  V24.B16, [V0.B16, V1.B16], V4.B16 \
	VTBL  V24.B16, [V2.B16, V3.B16], T0.B16 \
	VSUB  T0.H8, V4.H8, V4.H8 \
	VTBL  V25.B16, [V0.B16, V1.B16], V5.B16 \
	VTBL  V25.B16, [V2.B16, V3.B16], T0.B16 \
	VSUB  T0.H8, V5.H8, V5.H8 \
	VTBL  V26.B16, [V0.B16, V1.B16], V6.B16 \
	VTBL  V26.B16, [V2.B16, V3.B16], T0.B16 \
	VSUB  T0.H8, V6.H8, V6.H8 \
	VTBL  V27.B16, [V0.B16, V1.B16], V7.B16 \
	VTBL  V27.B16, [V2.B16, V3.B16], T0.B16 \
	VSUB  T0.H8, V7.H8, V7.H8 \
	VADD  V7.H8, V4.H8, T0.H8 \
	VSUB  V7.H8, V4.H8, V7.H8 \
	VADD  V6.H8, V5.H8, T1.H8 \
	VSUB  V6.H8, V5.H8, V6.H8 \
	VADD  T1.H8, T0.H8, V4.H8 \
	VSUB  T1.H8, T0.H8, V5.H8 \
	VSHL  $3, V4.H8, T0.H8 \
	VSHL  $3, V5.H8, T2.H8 \
	VMOVQ   $0x0000071400000714, $0x0000071400000714, V2 \
	VSMULL  V28.H4, V6.H4, V0.S4 \
	VSMULL2 V28.H8, V6.H8, V1.S4 \
	VSMLAL  V29.H4, V7.H4, V0.S4 \
	VSMLAL2 V29.H8, V7.H8, V1.S4 \
	VADD    V2.S4, V0.S4, V0.S4 \
	VADD    V2.S4, V1.S4, V1.S4 \
	VSHRN   $9, V0.S4, T1.H4 \
	VSHRN2  $9, V1.S4, T1.H8 \
	VMOVQ   $0x000003a9000003a9, $0x000003a9000003a9, V2 \
	VSMULL  V28.H4, V7.H4, V0.S4 \
	VSMULL2 V28.H8, V7.H8, V1.S4 \
	VSMLSL  V29.H4, V6.H4, V0.S4 \
	VSMLSL2 V29.H8, V6.H8, V1.S4 \
	VADD    V2.S4, V0.S4, V0.S4 \
	VADD    V2.S4, V1.S4, V1.S4 \
	VSHRN   $9, V0.S4, T3.H4 \
	VSHRN2  $9, V1.S4, T3.H8

	LOADY
	ROW(V8, V9, V10, V11)
	LOADY
	ROW(V12, V13, V14, V15)
	LOADY
	ROW(V16, V17, V18, V19)
	LOADY
	ROW(V20, V21, V22, V23)

// t0..t3 are one column's row outputs; out1 in T1, out3 in T2.
#define COL(T0, T1, T2, T3) \
	VADD  T3.H8, T0.H8, V4.H8 \
	VSUB  T3.H8, T0.H8, V7.H8 \
	VADD  T2.H8, T1.H8, V5.H8 \
	VSUB  T2.H8, T1.H8, V6.H8 \
	VMOVQ $0x0007000700070007, $0x0007000700070007, V3 \
	VADD  V5.H8, V4.H8, V0.H8 \
	VSUB  V5.H8, V4.H8, V1.H8 \
	VADD  V3.H8, V0.H8, V0.H8 \
	VADD  V3.H8, V1.H8, V1.H8 \
	VSSHR $4, V0.H8, V0.H8 \
	VSSHR $4, V1.H8, V1.H8 \
	VMOVQ   $0x00002ee000002ee0, $0x00002ee000002ee0, T0 \
	VSMULL  V28.H4, V6.H4, V2.S4 \
	VSMULL2 V28.H8, V6.H8, V3.S4 \
	VSMLAL  V29.H4, V7.H4, V2.S4 \
	VSMLAL2 V29.H8, V7.H8, V3.S4 \
	VADD    T0.S4, V2.S4, V2.S4 \
	VADD    T0.S4, V3.S4, V3.S4 \
	VSHRN   $16, V2.S4, T1.H4 \
	VSHRN2  $16, V3.S4, T1.H8 \
	VCMTST  V7.H8, V7.H8, T3.H8 \
	VSUB    T3.H8, T1.H8, T1.H8 \
	VMOVQ   $0x0000c7380000c738, $0x0000c7380000c738, T0 \
	VSMULL  V28.H4, V7.H4, V2.S4 \
	VSMULL2 V28.H8, V7.H8, V3.S4 \
	VSMLSL  V29.H4, V6.H4, V2.S4 \
	VSMLSL2 V29.H8, V6.H8, V3.S4 \
	VADD    T0.S4, V2.S4, V2.S4 \
	VADD    T0.S4, V3.S4, V3.S4 \
	VSHRN   $16, V2.S4, T2.H4 \
	VSHRN2  $16, V3.S4, T2.H8 \
	VABS    V0.H8, V0.H8 \
	VABS    T1.H8, T1.H8 \
	VABS    V1.H8, V1.H8 \
	VABS    T2.H8, T2.H8 \
	VSHRN   $3, V0.H8, V4.B8 \
	VSHRN2  $3, T1.H8, V4.B16 \
	VSHRN   $3, V1.H8, V5.B8 \
	VSHRN2  $3, T2.H8, V5.B16 \
	VMOVQ   $0x1f1f1f1f1f1f1f1f, $0x1f1f1f1f1f1f1f1f, V3 \
	VUMIN   V3.B16, V4.B16, V4.B16 \
	VUMIN   V3.B16, V5.B16, V5.B16 \
	VST1.P  [V4.B16, V5.B16], 32(R3)

	COL(V8, V12, V16, V20)
	COL(V9, V13, V17, V21)
	COL(V10, V14, V18, V22)
	COL(V11, V15, V19, V23)

	ADD  R9, R0
	ADD  $128, R2
	SUBS $1, R4
	BNE  half
	RET

// U and V rows share a register, so 8 blocks fill one half.
#define LOADUV \
	ADD   R9, R0, R6 \
	ADD   R9, R5, R8 \
	MOVD  (R0), R10 \
	MOVD  (R5), R11 \
	MOVD  (R6), R12 \
	MOVD  (R8), R13 \
	MOVD  (R2), R14 \
	MOVD  16(R2), R15 \
	VMOV  R10, V0.D[0] \
	VMOV  R11, V0.D[1] \
	VMOV  R12, V1.D[0] \
	VMOV  R13, V1.D[1] \
	VMOV  R14, V2.D[0] \
	VMOV  R15, V2.D[1] \
	MOVD  128(R2), R14 \
	MOVD  144(R2), R15 \
	VMOV  R14, V3.D[0] \
	VMOV  R15, V3.D[1] \
	ADD   R1, R0 \
	ADD   R1, R5 \
	ADD   $32, R2

// func collectUVAsm(bins *[128]uint8, u, v *uint8, stride int, pred *uint8)
TEXT ·collectUVAsm(SB), NOSPLIT, $0-40
	MOVD bins+0(FP), R3
	MOVD u+8(FP), R0
	MOVD v+16(FP), R5
	MOVD stride+24(FP), R1
	MOVD pred+32(FP), R2

	VMOVQ $0xFF0CFF08FF04FF00, $0xFF1CFF18FF14FF10, V24
	VMOVQ $0xFF0DFF09FF05FF01, $0xFF1DFF19FF15FF11, V25
	VMOVQ $0xFF0EFF0AFF06FF02, $0xFF1EFF1AFF16FF12, V26
	VMOVQ $0xFF0FFF0BFF07FF03, $0xFF1FFF1BFF17FF13, V27
	VMOVQ $0x08a908a908a908a9, $0x08a908a908a908a9, V28
	VMOVQ $0x14e814e814e814e8, $0x14e814e814e814e8, V29
	LSL   $2, R1, R9

	LOADUV
	ROW(V8, V9, V10, V11)
	LOADUV
	ROW(V12, V13, V14, V15)
	LOADUV
	ROW(V16, V17, V18, V19)
	LOADUV
	ROW(V20, V21, V22, V23)

	COL(V8, V12, V16, V20)
	COL(V9, V13, V17, V21)
	COL(V10, V14, V18, V22)
	COL(V11, V15, V19, V23)
	RET

// func residualQuant3Asm(m *quantMatrix, pr *[10][4]uint32, modes int, src *uint8, stride int, blk, lv *[3][16]int16, res *[3][2]int)
// Straight-line: the three trials are woven round robin,
// so the three fdct and quantizer chains overlap. The fdct and
// quantizer are i4TrialsDotAsm's, constants from i4Tab. WORD lines
// hold the instruction in the comment.
// Generated and simulated by gen/rq3.py; edit that, not this.
TEXT ·residualQuant3Asm(SB), NOSPLIT, $0-64
	MOVD m+0(FP), R0
	MOVD pr+8(FP), R21
	MOVD modes+16(FP), R14
	MOVD src+24(FP), R19
	MOVD stride+32(FP), R13
	MOVD blk+40(FP), R1
	MOVD lv+48(FP), R2
	MOVD res+56(FP), R20
	MOVD $·i4Tab(SB), R17
	WORD $0x92400dc9 // and x9, x14, #15
	WORD $0xd3441dca // ubfx x10, x14, #4, #4
	WORD $0xd348fdcb // lsr x11, x14, #8
	WORD $0x0d408261 // ld1 {v1.s}[0], [x19]
	WORD $0x8b0d026f // add x15, x19, x13
	WORD $0x0d4091e1 // ld1 {v1.s}[1], [x15]
	WORD $0x8b0d01ef // add x15, x15, x13
	WORD $0x4d4081e1 // ld1 {v1.s}[2], [x15]
	WORD $0x8b0d01ef // add x15, x15, x13
	WORD $0x4d4091e1 // ld1 {v1.s}[3], [x15]
	MOVWU quantMatrix_q+4(R0), R3
	MOVWU quantMatrix_q(R0), R4
	MOVWU quantMatrix_iq+4(R0), R5
	MOVWU quantMatrix_iq(R0), R6
	MOVWU quantMatrix_bias+4(R0), R7
	MOVWU quantMatrix_bias(R0), R8
	ADD $quantMatrix_sharpen, R0, R12
	WORD $0x4c40298c // ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x12]
	WORD $0x4e4d1986 // uzp1 v6.8h, v12.8h, v13.8h
	WORD $0x4e4f19c7 // uzp1 v7.8h, v14.8h, v15.8h
	WORD $0x3dc00220 // ldr q0, [x17]
	WORD $0x4e0a1c60 // ins v0.h[2], w3
	WORD $0x4e0e1ca0 // ins v0.h[3], w5
	WORD $0x4e020c69 // dup v9.8h, w3
	WORD $0x4e021c89 // ins v9.h[0], w4
	WORD $0x4e020ca8 // dup v8.8h, w5
	WORD $0x4e021cc8 // ins v8.h[0], w6
	WORD $0x4e040ceb // dup v11.4s, w7
	WORD $0x4eab1d6a // mov v10.16b, v11.16b
	WORD $0x4e041d0a // ins v10.s[0], w8
	WORD $0xd28001f7 // mov x23, #15
	WORD $0x3ce97aa2 // ldr q2, [x21, x9, lsl #4]
	WORD $0x3ceb7aa3 // ldr q3, [x21, x11, lsl #4]
	WORD $0x3cea7aa4 // ldr q4, [x21, x10, lsl #4]
	WORD $0x2e222025 // usubl v5.8h, v1.8b, v2.8b
	WORD $0x2e23202c // usubl v12.8h, v1.8b, v3.8b
	WORD $0x2e24202d // usubl v13.8h, v1.8b, v4.8b
	WORD $0x6e22202e // usubl2 v14.8h, v1.16b, v2.16b
	WORD $0x6e232022 // usubl2 v2.8h, v1.16b, v3.16b
	WORD $0x6e242023 // usubl2 v3.8h, v1.16b, v4.16b
	WORD $0x4e4e18a4 // uzp1 v4.8h, v5.8h, v14.8h
	WORD $0x4e42198f // uzp1 v15.8h, v12.8h, v2.8h
	WORD $0x4e4319b0 // uzp1 v16.8h, v13.8h, v3.8h
	WORD $0x4e4e58b1 // uzp2 v17.8h, v5.8h, v14.8h
	WORD $0x4e425985 // uzp2 v5.8h, v12.8h, v2.8h
	WORD $0x4e4359ae // uzp2 v14.8h, v13.8h, v3.8h
	WORD $0x4e511892 // uzp1 v18.8h, v4.8h, v17.8h
	WORD $0x4e4519f3 // uzp1 v19.8h, v15.8h, v5.8h
	WORD $0x4e4e1a14 // uzp1 v20.8h, v16.8h, v14.8h
	WORD $0x4e445a35 // uzp2 v21.8h, v17.8h, v4.8h
	WORD $0x4e4f58b1 // uzp2 v17.8h, v5.8h, v15.8h
	WORD $0x4e5059cf // uzp2 v15.8h, v14.8h, v16.8h
	WORD $0x4e758650 // add v16.8h, v18.8h, v21.8h
	WORD $0x4e718676 // add v22.8h, v19.8h, v17.8h
	WORD $0x4e6f8697 // add v23.8h, v20.8h, v15.8h
	WORD $0x6e758658 // sub v24.8h, v18.8h, v21.8h
	WORD $0x6e718672 // sub v18.8h, v19.8h, v17.8h
	WORD $0x6e6f8695 // sub v21.8h, v20.8h, v15.8h
	WORD $0x6e104211 // ext v17.16b, v16.16b, v16.16b, #8
	WORD $0x6e1642d9 // ext v25.16b, v22.16b, v22.16b, #8
	WORD $0x6e1742fa // ext v26.16b, v23.16b, v23.16b, #8
	WORD $0x4e71861b // add v27.8h, v16.8h, v17.8h
	WORD $0x4e7986dc // add v28.8h, v22.8h, v25.8h
	WORD $0x4e7a86fd // add v29.8h, v23.8h, v26.8h
	WORD $0x6e71861e // sub v30.8h, v16.8h, v17.8h
	WORD $0x6e7986df // sub v31.8h, v22.8h, v25.8h
	WORD $0x6e7a86f6 // sub v22.8h, v23.8h, v26.8h
	WORD $0x3dc00a37 // ldr q23, [x17, #32]
	WORD $0x3dc00a2c // ldr q12, [x17, #32]
	WORD $0x3dc00a22 // ldr q2, [x17, #32]
	WORD $0x4f402317 // smlal2 v23.4s, v24.8h, v0.h[0]
	WORD $0x4f40224c // smlal2 v12.4s, v18.8h, v0.h[0]
	WORD $0x4f4022a2 // smlal2 v2.4s, v21.8h, v0.h[0]
	WORD $0x0f502317 // smlal v23.4s, v24.4h, v0.h[1]
	WORD $0x0f50224c // smlal v12.4s, v18.4h, v0.h[1]
	WORD $0x0f5022a2 // smlal v2.4s, v21.4h, v0.h[1]
	WORD $0x3dc00e2d // ldr q13, [x17, #48]
	WORD $0x3dc00e23 // ldr q3, [x17, #48]
	WORD $0x3dc00e24 // ldr q4, [x17, #48]
	WORD $0x0f40230d // smlal v13.4s, v24.4h, v0.h[0]
	WORD $0x0f402243 // smlal v3.4s, v18.4h, v0.h[0]
	WORD $0x0f4022a4 // smlal v4.4s, v21.4h, v0.h[0]
	WORD $0x4f50630d // smlsl2 v13.4s, v24.8h, v0.h[1]
	WORD $0x4f506243 // smlsl2 v3.4s, v18.8h, v0.h[1]
	WORD $0x4f5062a4 // smlsl2 v4.4s, v21.8h, v0.h[1]
	WORD $0x4f135765 // shl v5.8h, v27.8h, #3
	WORD $0x4f13578e // shl v14.8h, v28.8h, #3
	WORD $0x4f1357b3 // shl v19.8h, v29.8h, #3
	WORD $0x4f1786e5 // shrn2 v5.8h, v23.4s, #9
	WORD $0x4f17858e // shrn2 v14.8h, v12.4s, #9
	WORD $0x4f178453 // shrn2 v19.8h, v2.4s, #9
	WORD $0x4f1357cc // shl v12.8h, v30.8h, #3
	WORD $0x4f1357e2 // shl v2.8h, v31.8h, #3
	WORD $0x4f1356d4 // shl v20.8h, v22.8h, #3
	WORD $0x4f1785ac // shrn2 v12.8h, v13.4s, #9
	WORD $0x4f178462 // shrn2 v2.8h, v3.4s, #9
	WORD $0x4f178494 // shrn2 v20.8h, v4.4s, #9
	WORD $0x4e4c18ad // uzp1 v13.8h, v5.8h, v12.8h
	WORD $0x4e4219cf // uzp1 v15.8h, v14.8h, v2.8h
	WORD $0x4e541a70 // uzp1 v16.8h, v19.8h, v20.8h
	WORD $0x4e4c58b1 // uzp2 v17.8h, v5.8h, v12.8h
	WORD $0x4e4259d2 // uzp2 v18.8h, v14.8h, v2.8h
	WORD $0x4e545a6c // uzp2 v12.8h, v19.8h, v20.8h
	WORD $0x4e5119ae // uzp1 v14.8h, v13.8h, v17.8h
	WORD $0x4e5219f9 // uzp1 v25.8h, v15.8h, v18.8h
	WORD $0x4e4c1a1a // uzp1 v26.8h, v16.8h, v12.8h
	WORD $0x4e4d5a38 // uzp2 v24.8h, v17.8h, v13.8h
	WORD $0x4e4f5a51 // uzp2 v17.8h, v18.8h, v15.8h
	WORD $0x4e50598d // uzp2 v13.8h, v12.8h, v16.8h
	WORD $0x4e7885cc // add v12.8h, v14.8h, v24.8h
	WORD $0x4e718735 // add v21.8h, v25.8h, v17.8h
	WORD $0x4e6d875b // add v27.8h, v26.8h, v13.8h
	WORD $0x6e7885dc // sub v28.8h, v14.8h, v24.8h
	WORD $0x6e71873d // sub v29.8h, v25.8h, v17.8h
	WORD $0x6e6d8757 // sub v23.8h, v26.8h, v13.8h
	WORD $0x3dc00636 // ldr q22, [x17, #16]
	WORD $0x4e76859e // add v30.8h, v12.8h, v22.8h
	WORD $0x4e7686bf // add v31.8h, v21.8h, v22.8h
	WORD $0x4e768763 // add v3.8h, v27.8h, v22.8h
	WORD $0x6e1e43c2 // ext v2.16b, v30.16b, v30.16b, #8
	WORD $0x6e1f43f6 // ext v22.16b, v31.16b, v31.16b, #8
	WORD $0x6e034064 // ext v4.16b, v3.16b, v3.16b, #8
	WORD $0x4e6c8445 // add v5.8h, v2.8h, v12.8h
	WORD $0x4e7586c3 // add v3.8h, v22.8h, v21.8h
	WORD $0x4e7b8493 // add v19.8h, v4.8h, v27.8h
	WORD $0x6e6c8454 // sub v20.8h, v2.8h, v12.8h
	WORD $0x6e7586c2 // sub v2.8h, v22.8h, v21.8h
	WORD $0x6e7b8492 // sub v18.8h, v4.8h, v27.8h
	WORD $0x4ed478a4 // zip2 v4.2d, v5.2d, v20.2d
	WORD $0x4ec27865 // zip2 v5.2d, v3.2d, v2.2d
	WORD $0x4ed27a6f // zip2 v15.2d, v19.2d, v18.2d
	WORD $0x4f1c0484 // sshr v4.8h, v4.8h, #4
	WORD $0x4f1c04a5 // sshr v5.8h, v5.8h, #4
	WORD $0x4f1c05ef // sshr v15.8h, v15.8h, #4
	WORD $0x3dc01230 // ldr q16, [x17, #64]
	WORD $0x3dc0122e // ldr q14, [x17, #64]
	WORD $0x3dc01238 // ldr q24, [x17, #64]
	WORD $0x0f502390 // smlal v16.4s, v28.4h, v0.h[1]
	WORD $0x0f5023ae // smlal v14.4s, v29.4h, v0.h[1]
	WORD $0x0f5022f8 // smlal v24.4s, v23.4h, v0.h[1]
	WORD $0x4f402390 // smlal2 v16.4s, v28.8h, v0.h[0]
	WORD $0x4f4023ae // smlal2 v14.4s, v29.8h, v0.h[0]
	WORD $0x4f4022f8 // smlal2 v24.4s, v23.8h, v0.h[0]
	WORD $0x3dc01639 // ldr q25, [x17, #80]
	WORD $0x3dc01631 // ldr q17, [x17, #80]
	WORD $0x3dc0163a // ldr q26, [x17, #80]
	WORD $0x0f402399 // smlal v25.4s, v28.4h, v0.h[0]
	WORD $0x0f4023b1 // smlal v17.4s, v29.4h, v0.h[0]
	WORD $0x0f4022fa // smlal v26.4s, v23.4h, v0.h[0]
	WORD $0x4f506399 // smlsl2 v25.4s, v28.8h, v0.h[1]
	WORD $0x4f5063b1 // smlsl2 v17.4s, v29.8h, v0.h[1]
	WORD $0x4f5062fa // smlsl2 v26.4s, v23.8h, v0.h[1]
	WORD $0x0f10861b // shrn v27.4h, v16.4s, #16
	WORD $0x0f1085d0 // shrn v16.4h, v14.4s, #16
	WORD $0x0f10870d // shrn v13.4h, v24.4s, #16
	WORD $0x4f10873b // shrn2 v27.8h, v25.4s, #16
	WORD $0x4f108630 // shrn2 v16.8h, v17.4s, #16
	WORD $0x4f10874d // shrn2 v13.8h, v26.4s, #16
	WORD $0x0e7c8f8c // cmtst v12.4h, v28.4h, v28.4h
	WORD $0x0e7d8fae // cmtst v14.4h, v29.4h, v29.4h
	WORD $0x0e778efe // cmtst v30.4h, v23.4h, v23.4h
	WORD $0x6e6c877b // sub v27.8h, v27.8h, v12.8h
	WORD $0x6e6e8610 // sub v16.8h, v16.8h, v14.8h
	WORD $0x6e7e85ad // sub v13.8h, v13.8h, v30.8h
	WORD $0x4edb388c // zip1 v12.2d, v4.2d, v27.2d
	WORD $0x4ed038ae // zip1 v14.2d, v5.2d, v16.2d
	WORD $0x4ecd39ff // zip1 v31.2d, v15.2d, v13.2d
	WORD $0x4edb7896 // zip2 v22.2d, v4.2d, v27.2d
	WORD $0x4ed078b5 // zip2 v21.2d, v5.2d, v16.2d
	WORD $0x4ecd79f4 // zip2 v20.2d, v15.2d, v13.2d
	WORD $0x4e60b98d // abs v13.8h, v12.8h
	WORD $0x4e60b9c3 // abs v3.8h, v14.8h
	WORD $0x4e60bbe2 // abs v2.8h, v31.8h
	WORD $0x4e60bad3 // abs v19.8h, v22.8h
	WORD $0x4e60bab2 // abs v18.8h, v21.8h
	WORD $0x4e60ba98 // abs v24.8h, v20.8h
	WORD $0x4e6685ad // add v13.8h, v13.8h, v6.8h
	WORD $0x4e668463 // add v3.8h, v3.8h, v6.8h
	WORD $0x4e668442 // add v2.8h, v2.8h, v6.8h
	WORD $0x4e678673 // add v19.8h, v19.8h, v7.8h
	WORD $0x4e678652 // add v18.8h, v18.8h, v7.8h
	WORD $0x4e678718 // add v24.8h, v24.8h, v7.8h
	WORD $0x2e68c1b7 // umull v23.4s, v13.4h, v8.4h
	WORD $0x2e68c079 // umull v25.4s, v3.4h, v8.4h
	WORD $0x2e68c051 // umull v17.4s, v2.4h, v8.4h
	WORD $0x6f70a1ba // umull2 v26.4s, v13.8h, v0.h[3]
	WORD $0x6f70a06d // umull2 v13.4s, v3.8h, v0.h[3]
	WORD $0x6f70a05c // umull2 v28.4s, v2.8h, v0.h[3]
	WORD $0x2f70a27b // umull v27.4s, v19.4h, v0.h[3]
	WORD $0x2f70a25d // umull v29.4s, v18.4h, v0.h[3]
	WORD $0x2f70a31e // umull v30.4s, v24.4h, v0.h[3]
	WORD $0x6f70a264 // umull2 v4.4s, v19.8h, v0.h[3]
	WORD $0x6f70a245 // umull2 v5.4s, v18.8h, v0.h[3]
	WORD $0x6f70a310 // umull2 v16.4s, v24.8h, v0.h[3]
	WORD $0x0e6a42e2 // addhn v2.4h, v23.4s, v10.4s
	WORD $0x0e6a4332 // addhn v18.4h, v25.4s, v10.4s
	WORD $0x0e6a4237 // addhn v23.4h, v17.4s, v10.4s
	WORD $0x4e6b4342 // addhn2 v2.8h, v26.4s, v11.4s
	WORD $0x4e6b41b2 // addhn2 v18.8h, v13.4s, v11.4s
	WORD $0x4e6b4397 // addhn2 v23.8h, v28.4s, v11.4s
	WORD $0x0e6b4363 // addhn v3.4h, v27.4s, v11.4s
	WORD $0x0e6b43b3 // addhn v19.4h, v29.4s, v11.4s
	WORD $0x0e6b43d8 // addhn v24.4h, v30.4s, v11.4s
	WORD $0x4e6b4083 // addhn2 v3.8h, v4.4s, v11.4s
	WORD $0x4e6b40b3 // addhn2 v19.8h, v5.4s, v11.4s
	WORD $0x4e6b4218 // addhn2 v24.8h, v16.4s, v11.4s
	WORD $0x6f1f0442 // ushr v2.8h, v2.8h, #1
	WORD $0x6f1f0652 // ushr v18.8h, v18.8h, #1
	WORD $0x6f1f06f7 // ushr v23.8h, v23.8h, #1
	WORD $0x6f1f0463 // ushr v3.8h, v3.8h, #1
	WORD $0x6f1f0673 // ushr v19.8h, v19.8h, #1
	WORD $0x6f1f0718 // ushr v24.8h, v24.8h, #1
	WORD $0x4e60a98d // cmlt v13.8h, v12.8h, #0
	WORD $0x4e60a9cf // cmlt v15.8h, v14.8h, #0
	WORD $0x4e60abf9 // cmlt v25.8h, v31.8h, #0
	WORD $0x4e60aad1 // cmlt v17.8h, v22.8h, #0
	WORD $0x4e60aab0 // cmlt v16.8h, v21.8h, #0
	WORD $0x4e60aa9a // cmlt v26.8h, v20.8h, #0
	WORD $0x6e2d1c42 // eor v2.16b, v2.16b, v13.16b
	WORD $0x6e2f1e52 // eor v18.16b, v18.16b, v15.16b
	WORD $0x6e391ef7 // eor v23.16b, v23.16b, v25.16b
	WORD $0x6e311c63 // eor v3.16b, v3.16b, v17.16b
	WORD $0x6e301e73 // eor v19.16b, v19.16b, v16.16b
	WORD $0x6e3a1f18 // eor v24.16b, v24.16b, v26.16b
	WORD $0x6e6d8442 // sub v2.8h, v2.8h, v13.8h
	WORD $0x6e6f8652 // sub v18.8h, v18.8h, v15.8h
	WORD $0x6e7986f7 // sub v23.8h, v23.8h, v25.8h
	WORD $0x6e718463 // sub v3.8h, v3.8h, v17.8h
	WORD $0x6e708673 // sub v19.8h, v19.8h, v16.8h
	WORD $0x6e7a8718 // sub v24.8h, v24.8h, v26.8h
	WORD $0x4e699c4d // mul v13.8h, v2.8h, v9.8h
	WORD $0x4e699e5c // mul v28.8h, v18.8h, v9.8h
	WORD $0x4e699efb // mul v27.8h, v23.8h, v9.8h
	WORD $0x4f60807d // mul v29.8h, v3.8h, v0.h[2]
	WORD $0x4f60827e // mul v30.8h, v19.8h, v0.h[2]
	WORD $0x4f608304 // mul v4.8h, v24.8h, v0.h[2]
	WORD $0x6e6d8585 // sub v5.8h, v12.8h, v13.8h
	WORD $0x6e7c85cc // sub v12.8h, v14.8h, v28.8h
	WORD $0x6e7b87ef // sub v15.8h, v31.8h, v27.8h
	WORD $0x6e7d86ce // sub v14.8h, v22.8h, v29.8h
	WORD $0x6e7e86bf // sub v31.8h, v21.8h, v30.8h
	WORD $0x6e648699 // sub v25.8h, v20.8h, v4.8h
	WORD $0xad00742d // stp q13, q29, [x1, #0]
	WORD $0xad02783c // stp q28, q30, [x1, #64]
	WORD $0xad01103b // stp q27, q4, [x1, #32]
	WORD $0x3dc01e2d // ldr q13, [x17, #112]
	WORD $0x4e0d2044 // tbl v4.16b, {v2.16b, v3.16b}, v13.16b
	WORD $0x4e0d2251 // tbl v17.16b, {v18.16b, v19.16b}, v13.16b
	WORD $0x4e0d22f0 // tbl v16.16b, {v23.16b, v24.16b}, v13.16b
	WORD $0x3dc0222d // ldr q13, [x17, #128]
	WORD $0x4e0d205a // tbl v26.16b, {v2.16b, v3.16b}, v13.16b
	WORD $0x4e0d2256 // tbl v22.16b, {v18.16b, v19.16b}, v13.16b
	WORD $0x4e0d22f5 // tbl v21.16b, {v23.16b, v24.16b}, v13.16b
	WORD $0xad006844 // stp q4, q26, [x2, #0]
	WORD $0xad025851 // stp q17, q22, [x2, #64]
	WORD $0xad015450 // stp q16, q21, [x2, #32]
	WORD $0x4e648c8d // cmtst v13.8h, v4.8h, v4.8h
	WORD $0x4e718e34 // cmtst v20.8h, v17.8h, v17.8h
	WORD $0x4e708e1d // cmtst v29.8h, v16.8h, v16.8h
	WORD $0x4e7a8f5e // cmtst v30.8h, v26.8h, v26.8h
	WORD $0x4e768edc // cmtst v28.8h, v22.8h, v22.8h
	WORD $0x4e758ebb // cmtst v27.8h, v21.8h, v21.8h
	WORD $0x4e1e19ba // uzp1 v26.16b, v13.16b, v30.16b
	WORD $0x4e1c1a8d // uzp1 v13.16b, v20.16b, v28.16b
	WORD $0x4e1b1bbe // uzp1 v30.16b, v29.16b, v27.16b
	WORD $0x0f0c8742 // shrn v2.8b, v26.8h, #4
	WORD $0x0f0c85a3 // shrn v3.8b, v13.8h, #4
	WORD $0x0f0c87c4 // shrn v4.8b, v30.8h, #4
	WORD $0x9e660043 // fmov x3, d2
	WORD $0x9e660064 // fmov x4, d3
	WORD $0x9e660085 // fmov x5, d4
	WORD $0xdac01066 // clz x6, x3
	WORD $0xdac01087 // clz x7, x4
	WORD $0xdac010a8 // clz x8, x5
	WORD $0xcb460aec // sub x12, x23, x6, lsr #2
	WORD $0xcb470aef // sub x15, x23, x7, lsr #2
	WORD $0xcb480af6 // sub x22, x23, x8, lsr #2
	WORD $0x0e65c0ad // smull v13.4s, v5.4h, v5.4h
	WORD $0x0e6cc192 // smull v18.4s, v12.4h, v12.4h
	WORD $0x0e6fc1f3 // smull v19.4s, v15.4h, v15.4h
	WORD $0x4e65c0b7 // smull2 v23.4s, v5.8h, v5.8h
	WORD $0x4e6cc198 // smull2 v24.4s, v12.8h, v12.8h
	WORD $0x4e6fc1ec // smull2 v12.4s, v15.8h, v15.8h
	WORD $0x0e6e81cd // smlal v13.4s, v14.4h, v14.4h
	WORD $0x0e7f83f2 // smlal v18.4s, v31.4h, v31.4h
	WORD $0x0e798333 // smlal v19.4s, v25.4h, v25.4h
	WORD $0x4e6e81d7 // smlal2 v23.4s, v14.8h, v14.8h
	WORD $0x4e7f83f8 // smlal2 v24.4s, v31.8h, v31.8h
	WORD $0x4e79832c // smlal2 v12.4s, v25.8h, v25.8h
	WORD $0x4eb785ad // add v13.4s, v13.4s, v23.4s
	WORD $0x4eb88652 // add v18.4s, v18.4s, v24.4s
	WORD $0x4eac8673 // add v19.4s, v19.4s, v12.4s
	WORD $0x4eb1b9ad // addv s13, v13.4s
	WORD $0x4eb1ba52 // addv s18, v18.4s
	WORD $0x4eb1ba73 // addv s19, v19.4s
	WORD $0x1e2601b8 // fmov w24, s13
	WORD $0x1e260259 // fmov w25, s18
	WORD $0x1e260263 // fmov w3, s19
	WORD $0xa900628c // stp x12, x24, [x20, #0]
	WORD $0xa902668f // stp x15, x25, [x20, #32]
	WORD $0xa9010e96 // stp x22, x3, [x20, #16]
	RET

#define HPASS(X, S2, S3, D2, D3) \
	VREV64 X.S4, V10.S4 \
	VADD   V10.H8, X.H8, V11.H8 \
	VSUB   V10.H8, X.H8, V12.H8 \
	VREV32 V11.H8, V13.H8 \
	VREV32 V12.H8, V14.H8 \
	VADD   V13.H8, V11.H8, S2.H8 \
	VSUB   V13.H8, V11.H8, S3.H8 \
	VADD   V14.H8, V12.H8, D2.H8 \
	VSUB   V14.H8, V12.H8, D3.H8 \
	VABS   S2.H8, S2.H8 \
	VABS   S3.H8, S3.H8 \
	VABS   D2.H8, D2.H8 \
	VABS   D3.H8, D3.H8

// RECM is recMetricsAsm: c at R0, dst at R1, src at R2 with stride R3;
// sse in R5 and tt in R6. Clobbers R4 and V0-V31.
#define RECM \
	MOVW $85627, R4 \
	MOVW $35468, R5 \
	MOVW $4, R6 \
	VDUP R4, V30.S4 \
	VDUP R5, V31.S4 \
	VDUP R6, V29.S4 \
	VLD1   (R0), [V0.H8, V1.H8] \
	VSXTL  V0.H4, V2.S4 \
	VSXTL2 V0.H8, V3.S4 \
	VSXTL  V1.H4, V4.S4 \
	VSXTL2 V1.H8, V5.S4 \
	VADD  V4.S4, V2.S4, V6.S4 \
	VSUB  V4.S4, V2.S4, V7.S4 \
	VMUL  V31.S4, V3.S4, V8.S4 \
	VMUL  V30.S4, V5.S4, V9.S4 \
	VMUL  V30.S4, V3.S4, V10.S4 \
	VMUL  V31.S4, V5.S4, V11.S4 \
	VSSHR $16, V8.S4, V8.S4 \
	VSSHR $16, V9.S4, V9.S4 \
	VSSHR $16, V10.S4, V10.S4 \
	VSSHR $16, V11.S4, V11.S4 \
	VSUB  V9.S4, V8.S4, V8.S4 \
	VADD  V11.S4, V10.S4, V10.S4 \
	VADD  V10.S4, V6.S4, V12.S4 \
	VADD  V8.S4, V7.S4, V13.S4 \
	VSUB  V8.S4, V7.S4, V14.S4 \
	VSUB  V10.S4, V6.S4, V15.S4 \
	VTRN1 V13.S4, V12.S4, V16.S4 \
	VTRN2 V13.S4, V12.S4, V17.S4 \
	VTRN1 V15.S4, V14.S4, V18.S4 \
	VTRN2 V15.S4, V14.S4, V19.S4 \
	VTRN1 V18.D2, V16.D2, V2.D2 \
	VTRN1 V19.D2, V17.D2, V3.D2 \
	VTRN2 V18.D2, V16.D2, V4.D2 \
	VTRN2 V19.D2, V17.D2, V5.D2 \
	VADD  V29.S4, V2.S4, V2.S4 \
	VADD  V4.S4, V2.S4, V6.S4 \
	VSUB  V4.S4, V2.S4, V7.S4 \
	VMUL  V31.S4, V3.S4, V8.S4 \
	VMUL  V30.S4, V5.S4, V9.S4 \
	VMUL  V30.S4, V3.S4, V10.S4 \
	VMUL  V31.S4, V5.S4, V11.S4 \
	VSSHR $16, V8.S4, V8.S4 \
	VSSHR $16, V9.S4, V9.S4 \
	VSSHR $16, V10.S4, V10.S4 \
	VSSHR $16, V11.S4, V11.S4 \
	VSUB  V9.S4, V8.S4, V8.S4 \
	VADD  V11.S4, V10.S4, V10.S4 \
	VADD  V10.S4, V6.S4, V12.S4 \
	VADD  V8.S4, V7.S4, V13.S4 \
	VSUB  V8.S4, V7.S4, V14.S4 \
	VSUB  V10.S4, V6.S4, V15.S4 \
	VSSHR $3, V12.S4, V12.S4 \
	VSSHR $3, V13.S4, V13.S4 \
	VSSHR $3, V14.S4, V14.S4 \
	VSSHR $3, V15.S4, V15.S4 \
	VTRN1 V13.S4, V12.S4, V16.S4 \
	VTRN2 V13.S4, V12.S4, V17.S4 \
	VTRN1 V15.S4, V14.S4, V18.S4 \
	VTRN2 V15.S4, V14.S4, V19.S4 \
	VTRN1 V18.D2, V16.D2, V2.D2 \
	VTRN1 V19.D2, V17.D2, V3.D2 \
	VTRN2 V18.D2, V16.D2, V4.D2 \
	VTRN2 V19.D2, V17.D2, V5.D2 \
	VSQXTN  V2.S4, V24.H4 \
	VSQXTN2 V3.S4, V24.H8 \
	VSQXTN  V4.S4, V25.H4 \
	VSQXTN2 V5.S4, V25.H8 \
	MOVWU   (R1), R4 \
	MOVWU   32(R1), R5 \
	ORR     R5<<32, R4, R4 \
	VMOV    R4, V26.D[0] \
	MOVWU   64(R1), R4 \
	MOVWU   96(R1), R5 \
	ORR     R5<<32, R4, R4 \
	VMOV    R4, V26.D[1] \
	VUXTL   V26.B8, V27.H8 \
	VUXTL2  V26.B16, V28.H8 \
	VSQADD  V27.H8, V24.H8, V24.H8 \
	VSQADD  V28.H8, V25.H8, V25.H8 \
	VSQXTUN  V24.H8, V26.B8 \
	VSQXTUN2 V25.H8, V26.B16 \
	VMOV    V26.D[0], R4 \
	MOVW    R4, (R1) \
	LSR     $32, R4 \
	MOVW    R4, 32(R1) \
	VMOV    V26.D[1], R4 \
	MOVW    R4, 64(R1) \
	LSR     $32, R4 \
	MOVW    R4, 96(R1) \
	MOVWU (R2), R4 \
	ADD   R3, R2, R6 \
	MOVWU (R6), R5 \
	ORR   R5<<32, R4, R4 \
	VMOV  R4, V27.D[0] \
	ADD   R3, R6 \
	MOVWU (R6), R4 \
	ADD   R3, R6 \
	MOVWU (R6), R5 \
	ORR   R5<<32, R4, R4 \
	VMOV  R4, V27.D[1] \
	VUMAX   V27.B16, V26.B16, V28.B16 \
	VUMIN   V27.B16, V26.B16, V20.B16 \
	VSUB    V20.B16, V28.B16, V28.B16 \
	VUMULL  V28.B8, V28.B8, V20.H8 \
	VUMULL2 V28.B16, V28.B16, V21.H8 \
	VEOR    V16.B16, V16.B16, V16.B16 \
	VEOR    V17.B16, V17.B16, V17.B16 \
	VUADDW  V20.H4, V16.S4, V16.S4 \
	VUADDW2 V20.H8, V17.S4, V17.S4 \
	VUADDW  V21.H4, V16.S4, V16.S4 \
	VUADDW2 V21.H8, V17.S4, V17.S4 \
	VADD    V17.S4, V16.S4, V16.S4 \
	VADDV   V16.S4, V16 \
	VMOV    V16.S[0], R5 \
	VUXTL  V26.B8, V18.H8 \
	VUXTL2 V26.B16, V19.H8 \
	VADD   V19.H8, V18.H8, V0.H8 \
	VSUB   V19.H8, V18.H8, V1.H8 \
	VEXT   $8, V0.B16, V0.B16, V2.B16 \
	VEXT   $8, V1.B16, V1.B16, V3.B16 \
	VADD   V2.H8, V0.H8, V4.H8 \
	VSUB   V2.H8, V0.H8, V5.H8 \
	VADD   V3.H8, V1.H8, V6.H8 \
	VSUB   V3.H8, V1.H8, V7.H8 \
	VZIP1  V6.D2, V4.D2, V8.D2 \
	VZIP1  V5.D2, V7.D2, V9.D2 \
	HPASS(V8, V16, V17, V18, V19) \
	HPASS(V9, V20, V21, V22, V23) \
	VMOVQ   $0x0000000000000026, $0x0000000000000020, V0 \
	VMOVQ   $0x0000000000000020, $0x000000000000001c, V1 \
	VMOVQ   $0x0000000000000014, $0x0000000000000011, V2 \
	VMOVQ   $0x0000000000000009, $0x0000000000000007, V3 \
	VSMULL  V0.H4, V16.H4, V24.S4 \
	VSMULL2 V0.H8, V16.H8, V25.S4 \
	VSMLAL  V1.H4, V18.H4, V24.S4 \
	VSMLAL2 V1.H8, V18.H8, V25.S4 \
	VSMLAL  V2.H4, V19.H4, V24.S4 \
	VSMLAL2 V2.H8, V19.H8, V25.S4 \
	VSMLAL  V3.H4, V17.H4, V24.S4 \
	VSMLAL2 V3.H8, V17.H8, V25.S4 \
	VMOVQ   $0x0000000000000014, $0x0000000000000009, V0 \
	VMOVQ   $0x0000000000000011, $0x0000000000000007, V1 \
	VMOVQ   $0x000000000000000a, $0x0000000000000004, V2 \
	VMOVQ   $0x0000000000000004, $0x0000000000000002, V3 \
	VSMLAL  V0.H4, V20.H4, V24.S4 \
	VSMLAL2 V0.H8, V20.H8, V25.S4 \
	VSMLAL  V1.H4, V22.H4, V24.S4 \
	VSMLAL2 V1.H8, V22.H8, V25.S4 \
	VSMLAL  V2.H4, V23.H4, V24.S4 \
	VSMLAL2 V2.H8, V23.H8, V25.S4 \
	VSMLAL  V3.H4, V21.H4, V24.S4 \
	VSMLAL2 V3.H8, V21.H8, V25.S4 \
	VADD    V25.S4, V24.S4, V24.S4 \
	VADDV   V24.S4, V24 \
	VMOV    V24.S[0], R6

// func recMetricsAsm(c *[16]int16, dst *uint8, src *uint8, stride int) (sse, tt int)
// idctAddAsm, then SSE of src against the result and TTransform of it.
// TT runs the vertical pass first; the transform is exact, so the order
// does not matter. kWeightY sits in lanes 0 and 4.
TEXT ·recMetricsAsm(SB), NOSPLIT, $0-48
	MOVD c+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD src+16(FP), R2
	MOVD stride+24(FP), R3
	RECM
	MOVD R5, sse+32(FP)
	MOVD R6, tt+40(FP)
	RET

// func i4TrialsAsm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32, dq *[16]int16)
// All ten i4 modes: RESQ, then the recMetricsAsm body with the
// prediction taken from pr and the result written to rec. res holds
// last, SSE, TT and the count of nonzero AC levels per mode.
TEXT ·i4TrialsAsm(SB), NOSPLIT, $0-64
	MOVD m+0(FP), R0
	MOVD pr+8(FP), R21
	MOVD src+16(FP), R19
	MOVD stride+24(FP), R13
	MOVD lv+32(FP), R2
	MOVD rec+40(FP), R24
	MOVD res+48(FP), R20
	MOVD dq+56(FP), R1
	MOVD $10, R22

trial10:
	MOVD R21, R11
	MOVD R19, R12
	RESQ

	// Nonzero AC levels: all lanes minus the DC lane.
	VLD1    (R2), [V14.H8, V15.H8]
	VCMTST  V14.H8, V14.H8, V14.H8
	VCMTST  V15.H8, V15.H8, V15.H8
	VUSHR   $15, V14.H8, V14.H8
	VUSHR   $15, V15.H8, V15.H8
	VMOV    V14.H[0], R10
	VADD    V15.H8, V14.H8, V14.H8
	VADDV   V14.H8, V14
	VMOV    V14.H[0], R9
	SUB     R10, R9, R9
	MOVW    R7, (R20)
	MOVW    R9, 12(R20)

	MOVW $85627, R9
	MOVW $35468, R10
	MOVW $4, R23
	VDUP R9, V30.S4
	VDUP R10, V31.S4
	VDUP R23, V29.S4
	VSXTL  V12.H4, V2.S4
	VSXTL2 V12.H8, V3.S4
	VSXTL  V13.H4, V4.S4
	VSXTL2 V13.H8, V5.S4

	VADD  V4.S4, V2.S4, V6.S4
	VSUB  V4.S4, V2.S4, V7.S4
	VMUL  V31.S4, V3.S4, V8.S4
	VMUL  V30.S4, V5.S4, V9.S4
	VMUL  V30.S4, V3.S4, V10.S4
	VMUL  V31.S4, V5.S4, V11.S4
	VSSHR $16, V8.S4, V8.S4
	VSSHR $16, V9.S4, V9.S4
	VSSHR $16, V10.S4, V10.S4
	VSSHR $16, V11.S4, V11.S4
	VSUB  V9.S4, V8.S4, V8.S4
	VADD  V11.S4, V10.S4, V10.S4
	VADD  V10.S4, V6.S4, V12.S4
	VADD  V8.S4, V7.S4, V13.S4
	VSUB  V8.S4, V7.S4, V14.S4
	VSUB  V10.S4, V6.S4, V15.S4

	VTRN1 V13.S4, V12.S4, V16.S4
	VTRN2 V13.S4, V12.S4, V17.S4
	VTRN1 V15.S4, V14.S4, V18.S4
	VTRN2 V15.S4, V14.S4, V19.S4
	VTRN1 V18.D2, V16.D2, V2.D2
	VTRN1 V19.D2, V17.D2, V3.D2
	VTRN2 V18.D2, V16.D2, V4.D2
	VTRN2 V19.D2, V17.D2, V5.D2

	VADD  V29.S4, V2.S4, V2.S4
	VADD  V4.S4, V2.S4, V6.S4
	VSUB  V4.S4, V2.S4, V7.S4
	VMUL  V31.S4, V3.S4, V8.S4
	VMUL  V30.S4, V5.S4, V9.S4
	VMUL  V30.S4, V3.S4, V10.S4
	VMUL  V31.S4, V5.S4, V11.S4
	VSSHR $16, V8.S4, V8.S4
	VSSHR $16, V9.S4, V9.S4
	VSSHR $16, V10.S4, V10.S4
	VSSHR $16, V11.S4, V11.S4
	VSUB  V9.S4, V8.S4, V8.S4
	VADD  V11.S4, V10.S4, V10.S4
	VADD  V10.S4, V6.S4, V12.S4
	VADD  V8.S4, V7.S4, V13.S4
	VSUB  V8.S4, V7.S4, V14.S4
	VSUB  V10.S4, V6.S4, V15.S4
	VSSHR $3, V12.S4, V12.S4
	VSSHR $3, V13.S4, V13.S4
	VSSHR $3, V14.S4, V14.S4
	VSSHR $3, V15.S4, V15.S4

	VTRN1 V13.S4, V12.S4, V16.S4
	VTRN2 V13.S4, V12.S4, V17.S4
	VTRN1 V15.S4, V14.S4, V18.S4
	VTRN2 V15.S4, V14.S4, V19.S4
	VTRN1 V18.D2, V16.D2, V2.D2
	VTRN1 V19.D2, V17.D2, V3.D2
	VTRN2 V18.D2, V16.D2, V4.D2
	VTRN2 V19.D2, V17.D2, V5.D2

	VSQXTN  V2.S4, V24.H4
	VSQXTN2 V3.S4, V24.H8
	VSQXTN  V4.S4, V25.H4
	VSQXTN2 V5.S4, V25.H8
	VLD1    (R21), [V26.B16]
	VUXTL   V26.B8, V27.H8
	VUXTL2  V26.B16, V28.H8
	VSQADD  V27.H8, V24.H8, V24.H8
	VSQADD  V28.H8, V25.H8, V25.H8
	VSQXTUN  V24.H8, V26.B8
	VSQXTUN2 V25.H8, V26.B16
	VST1    [V26.B16], (R24)

	MOVWU (R19), R9
	ADD   R13, R19, R23
	MOVWU (R23), R10
	ORR   R10<<32, R9, R9
	VMOV  R9, V27.D[0]
	ADD   R13, R23
	MOVWU (R23), R9
	ADD   R13, R23
	MOVWU (R23), R10
	ORR   R10<<32, R9, R9
	VMOV  R9, V27.D[1]
	VUMAX   V27.B16, V26.B16, V28.B16
	VUMIN   V27.B16, V26.B16, V20.B16
	VSUB    V20.B16, V28.B16, V28.B16
	VUMULL  V28.B8, V28.B8, V20.H8
	VUMULL2 V28.B16, V28.B16, V21.H8
	VEOR    V16.B16, V16.B16, V16.B16
	VEOR    V17.B16, V17.B16, V17.B16
	VUADDW  V20.H4, V16.S4, V16.S4
	VUADDW2 V20.H8, V17.S4, V17.S4
	VUADDW  V21.H4, V16.S4, V16.S4
	VUADDW2 V21.H8, V17.S4, V17.S4
	VADD    V17.S4, V16.S4, V16.S4
	VADDV   V16.S4, V16
	VMOV    V16.S[0], R9
	MOVW    R9, 4(R20)

	VUXTL  V26.B8, V18.H8
	VUXTL2 V26.B16, V19.H8
	VADD   V19.H8, V18.H8, V0.H8
	VSUB   V19.H8, V18.H8, V1.H8
	VEXT   $8, V0.B16, V0.B16, V2.B16
	VEXT   $8, V1.B16, V1.B16, V3.B16
	VADD   V2.H8, V0.H8, V4.H8
	VSUB   V2.H8, V0.H8, V5.H8
	VADD   V3.H8, V1.H8, V6.H8
	VSUB   V3.H8, V1.H8, V7.H8
	VZIP1  V6.D2, V4.D2, V8.D2
	VZIP1  V5.D2, V7.D2, V9.D2
	HPASS(V8, V16, V17, V18, V19)
	HPASS(V9, V20, V21, V22, V23)
	VMOVQ   $0x0000000000000026, $0x0000000000000020, V0
	VMOVQ   $0x0000000000000020, $0x000000000000001c, V1
	VMOVQ   $0x0000000000000014, $0x0000000000000011, V2
	VMOVQ   $0x0000000000000009, $0x0000000000000007, V3
	VSMULL  V0.H4, V16.H4, V24.S4
	VSMULL2 V0.H8, V16.H8, V25.S4
	VSMLAL  V1.H4, V18.H4, V24.S4
	VSMLAL2 V1.H8, V18.H8, V25.S4
	VSMLAL  V2.H4, V19.H4, V24.S4
	VSMLAL2 V2.H8, V19.H8, V25.S4
	VSMLAL  V3.H4, V17.H4, V24.S4
	VSMLAL2 V3.H8, V17.H8, V25.S4
	VMOVQ   $0x0000000000000014, $0x0000000000000009, V0
	VMOVQ   $0x0000000000000011, $0x0000000000000007, V1
	VMOVQ   $0x000000000000000a, $0x0000000000000004, V2
	VMOVQ   $0x0000000000000004, $0x0000000000000002, V3
	VSMLAL  V0.H4, V20.H4, V24.S4
	VSMLAL2 V0.H8, V20.H8, V25.S4
	VSMLAL  V1.H4, V22.H4, V24.S4
	VSMLAL2 V1.H8, V22.H8, V25.S4
	VSMLAL  V2.H4, V23.H4, V24.S4
	VSMLAL2 V2.H8, V23.H8, V25.S4
	VSMLAL  V3.H4, V21.H4, V24.S4
	VSMLAL2 V3.H8, V21.H8, V25.S4
	VADD    V25.S4, V24.S4, V24.S4
	VADDV   V24.S4, V24
	VMOV    V24.S[0], R9
	MOVW    R9, 8(R20)

	ADD  $16, R21
	ADD  $32, R2
	ADD  $16, R24
	ADD  $16, R20
	SUBS $1, R22
	BNE  trial10
	RET

DATA ·i4Tab+0(SB)/8, $0x0000000014e808a9
DATA ·i4Tab+8(SB)/8, $0x000000004e7b4546
DATA ·i4Tab+16(SB)/8, $0x0007000700070007
DATA ·i4Tab+24(SB)/8, $0x0007000700070007
DATA ·i4Tab+32(SB)/8, $0x0000071400000714
DATA ·i4Tab+40(SB)/8, $0x0000071400000714
DATA ·i4Tab+48(SB)/8, $0x000003a9000003a9
DATA ·i4Tab+56(SB)/8, $0x000003a9000003a9
DATA ·i4Tab+64(SB)/8, $0x00002ee000002ee0
DATA ·i4Tab+72(SB)/8, $0x00002ee000002ee0
DATA ·i4Tab+80(SB)/8, $0x0000c7380000c738
DATA ·i4Tab+88(SB)/8, $0x0000c7380000c738
DATA ·i4Tab+96(SB)/8, $0x0000000000000000
DATA ·i4Tab+104(SB)/8, $0x0000000000000000
DATA ·i4Tab+112(SB)/8, $0x1110090803020100
DATA ·i4Tab+120(SB)/8, $0x0d0c070605040b0a
DATA ·i4Tab+128(SB)/8, $0x15141b1a19181312
DATA ·i4Tab+136(SB)/8, $0x1f1e1d1c17160f0e
DATA ·i4Tab+144(SB)/8, $0x0d0905010c080400
DATA ·i4Tab+152(SB)/8, $0x0f0b07030e0a0602
DATA ·i4Tab+160(SB)/8, $0x0014000900200026
DATA ·i4Tab+168(SB)/8, $0x0004000200070009
DATA ·i4Tab+176(SB)/8, $0x00110007001c0020
DATA ·i4Tab+184(SB)/8, $0x000a000400110014
GLOBL ·i4Tab(SB), RODATA|NOPTR, $192

// func i4TrialsDotAsm(m *quantMatrix, pr *[10][4]uint32, src *uint8, stride int, lv *[10][16]int16, rec *[10][4]uint32, res *[10][4]int32)
// i4TrialsAsm for wide cores: two modes per pass, src and the
// quantizer set up once, UDOT for the SSE, a 16-bit IDCT with one
// mode per register half, and a butterfly TTransform. Each pass
// weaves the next pair's FDCT and quantizer into this pair's IDCT
// and reconstruction so the two chains overlap. WORD lines
// hold the instruction in the comment; the Go assembler lacks
// most of them. |dq| <= 2168 for 8-bit src and pred (y1 from
// newQuant) bounds every sum under 2^15, so it matches the int32 Go.
// Generated and simulated by gen/i4dot.py; edit that, not this.
TEXT ·i4TrialsDotAsm(SB), NOSPLIT, $104-56
	MOVD m+0(FP), R0
	MOVD pr+8(FP), R21
	MOVD src+16(FP), R19
	MOVD stride+24(FP), R13
	MOVD lv+32(FP), R2
	MOVD rec+40(FP), R24
	MOVD res+48(FP), R20
	MOVD $·i4Tab(SB), R17
	MOVWU quantMatrix_q+4(R0), R3
	MOVWU quantMatrix_q(R0), R4
	MOVWU quantMatrix_iq+4(R0), R5
	MOVWU quantMatrix_iq(R0), R6
	MOVWU quantMatrix_bias+4(R0), R7
	MOVWU quantMatrix_bias(R0), R8
	ADD $quantMatrix_sharpen, R0, R9
	WORD $0x4c40292c // ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
	WORD $0x4e4d1986 // uzp1 v6.8h, v12.8h, v13.8h
	WORD $0x4e4f19c7 // uzp1 v7.8h, v14.8h, v15.8h
	WORD $0x3dc00220 // ldr q0, [x17]
	WORD $0x4e0a1c60 // ins v0.h[2], w3
	WORD $0x4e0e1ca0 // ins v0.h[3], w5
	WORD $0x4e020c69 // dup v9.8h, w3
	WORD $0x4e021c89 // ins v9.h[0], w4
	WORD $0x4e020ca8 // dup v8.8h, w5
	WORD $0x4e021cc8 // ins v8.h[0], w6
	WORD $0x4e040ceb // dup v11.4s, w7
	WORD $0x4eab1d6a // mov v10.16b, v11.16b
	WORD $0x4e041d0a // ins v10.s[0], w8
	WORD $0x0d408261 // ld1 {v1.s}[0], [x19]
	WORD $0x8b0d0269 // add x9, x19, x13
	WORD $0x0d409121 // ld1 {v1.s}[1], [x9]
	WORD $0x8b0d0129 // add x9, x9, x13
	WORD $0x4d408121 // ld1 {v1.s}[2], [x9]
	WORD $0x8b0d0129 // add x9, x9, x13
	WORD $0x4d409121 // ld1 {v1.s}[3], [x9]
	WORD $0x528001f7 // mov w23, #15
	WORD $0xd2800096 // mov x22, #4
	ADD $16, RSP, R25
	WORD $0xad001f26 // stp q6, q7, [x25]
	WORD $0x3dc002a2 // ldr q2, [x21, #0]
	WORD $0x3dc006a3 // ldr q3, [x21, #16]
	WORD $0x2e222024 // usubl v4.8h, v1.8b, v2.8b
	WORD $0x2e232025 // usubl v5.8h, v1.8b, v3.8b
	WORD $0x6e222026 // usubl2 v6.8h, v1.16b, v2.16b
	WORD $0x6e232027 // usubl2 v7.8h, v1.16b, v3.16b
	WORD $0x4e46188c // uzp1 v12.8h, v4.8h, v6.8h
	WORD $0x4e4718ad // uzp1 v13.8h, v5.8h, v7.8h
	WORD $0x4e46588e // uzp2 v14.8h, v4.8h, v6.8h
	WORD $0x4e4758a6 // uzp2 v6.8h, v5.8h, v7.8h
	WORD $0x4e4e1987 // uzp1 v7.8h, v12.8h, v14.8h
	WORD $0x4e4619af // uzp1 v15.8h, v13.8h, v6.8h
	WORD $0x4e4c59d0 // uzp2 v16.8h, v14.8h, v12.8h
	WORD $0x4e4d58ce // uzp2 v14.8h, v6.8h, v13.8h
	WORD $0x4e7084f1 // add v17.8h, v7.8h, v16.8h
	WORD $0x4e6e85f2 // add v18.8h, v15.8h, v14.8h
	WORD $0x6e7084f3 // sub v19.8h, v7.8h, v16.8h
	WORD $0x6e6e85f0 // sub v16.8h, v15.8h, v14.8h
	WORD $0x6e114234 // ext v20.16b, v17.16b, v17.16b, #8
	WORD $0x6e124255 // ext v21.16b, v18.16b, v18.16b, #8
	WORD $0x4e748636 // add v22.8h, v17.8h, v20.8h
	WORD $0x4e758657 // add v23.8h, v18.8h, v21.8h
	WORD $0x6e748638 // sub v24.8h, v17.8h, v20.8h
	WORD $0x6e758651 // sub v17.8h, v18.8h, v21.8h
	WORD $0x3dc00a32 // ldr q18, [x17, #32]
	WORD $0x3dc00a39 // ldr q25, [x17, #32]
	WORD $0x4f402272 // smlal2 v18.4s, v19.8h, v0.h[0]
	WORD $0x4f402219 // smlal2 v25.4s, v16.8h, v0.h[0]
	WORD $0x0f502272 // smlal v18.4s, v19.4h, v0.h[1]
	WORD $0x0f502219 // smlal v25.4s, v16.4h, v0.h[1]
	WORD $0x3dc00e3a // ldr q26, [x17, #48]
	WORD $0x3dc00e3b // ldr q27, [x17, #48]
	WORD $0x0f40227a // smlal v26.4s, v19.4h, v0.h[0]
	WORD $0x0f40221b // smlal v27.4s, v16.4h, v0.h[0]
	WORD $0x4f50627a // smlsl2 v26.4s, v19.8h, v0.h[1]
	WORD $0x4f50621b // smlsl2 v27.4s, v16.8h, v0.h[1]
	WORD $0x4f1356dc // shl v28.8h, v22.8h, #3
	WORD $0x4f1356fd // shl v29.8h, v23.8h, #3
	WORD $0x4f17865c // shrn2 v28.8h, v18.4s, #9
	WORD $0x4f17873d // shrn2 v29.8h, v25.4s, #9
	WORD $0x4f135719 // shl v25.8h, v24.8h, #3
	WORD $0x4f13563e // shl v30.8h, v17.8h, #3
	WORD $0x4f178759 // shrn2 v25.8h, v26.4s, #9
	WORD $0x4f17877e // shrn2 v30.8h, v27.4s, #9
	WORD $0x4e591b9f // uzp1 v31.8h, v28.8h, v25.8h
	WORD $0x4e5e1ba2 // uzp1 v2.8h, v29.8h, v30.8h
	WORD $0x4e595b83 // uzp2 v3.8h, v28.8h, v25.8h
	WORD $0x4e5e5ba4 // uzp2 v4.8h, v29.8h, v30.8h
	WORD $0x4e431be5 // uzp1 v5.8h, v31.8h, v3.8h
	WORD $0x4e44184c // uzp1 v12.8h, v2.8h, v4.8h
	WORD $0x4e5f5866 // uzp2 v6.8h, v3.8h, v31.8h
	WORD $0x4e425887 // uzp2 v7.8h, v4.8h, v2.8h
	WORD $0x4e6684ad // add v13.8h, v5.8h, v6.8h
	WORD $0x4e67858f // add v15.8h, v12.8h, v7.8h
	WORD $0x6e6684ae // sub v14.8h, v5.8h, v6.8h
	WORD $0x6e678594 // sub v20.8h, v12.8h, v7.8h
	WORD $0x3dc0062c // ldr q12, [x17, #16]
	WORD $0x4e6c85b5 // add v21.8h, v13.8h, v12.8h
	WORD $0x4e6c85f3 // add v19.8h, v15.8h, v12.8h
	WORD $0x6e1542ac // ext v12.16b, v21.16b, v21.16b, #8
	WORD $0x6e134270 // ext v16.16b, v19.16b, v19.16b, #8
	WORD $0x4e6d8596 // add v22.8h, v12.8h, v13.8h
	WORD $0x4e6f8615 // add v21.8h, v16.8h, v15.8h
	WORD $0x6e6d8597 // sub v23.8h, v12.8h, v13.8h
	WORD $0x6e6f8612 // sub v18.8h, v16.8h, v15.8h
	WORD $0x4ed77ad3 // zip2 v19.2d, v22.2d, v23.2d
	WORD $0x4ed27ab8 // zip2 v24.2d, v21.2d, v18.2d
	WORD $0x4f1c0673 // sshr v19.8h, v19.8h, #4
	WORD $0x4f1c0718 // sshr v24.8h, v24.8h, #4
	WORD $0x3dc01231 // ldr q17, [x17, #64]
	WORD $0x3dc01232 // ldr q18, [x17, #64]
	WORD $0x0f5021d1 // smlal v17.4s, v14.4h, v0.h[1]
	WORD $0x0f502292 // smlal v18.4s, v20.4h, v0.h[1]
	WORD $0x4f4021d1 // smlal2 v17.4s, v14.8h, v0.h[0]
	WORD $0x4f402292 // smlal2 v18.4s, v20.8h, v0.h[0]
	WORD $0x3dc0163a // ldr q26, [x17, #80]
	WORD $0x3dc01639 // ldr q25, [x17, #80]
	WORD $0x0f4021da // smlal v26.4s, v14.4h, v0.h[0]
	WORD $0x0f402299 // smlal v25.4s, v20.4h, v0.h[0]
	WORD $0x4f5061da // smlsl2 v26.4s, v14.8h, v0.h[1]
	WORD $0x4f506299 // smlsl2 v25.4s, v20.8h, v0.h[1]
	WORD $0x0f10863b // shrn v27.4h, v17.4s, #16
	WORD $0x0f10865c // shrn v28.4h, v18.4s, #16
	WORD $0x4f10875b // shrn2 v27.8h, v26.4s, #16
	WORD $0x4f10873c // shrn2 v28.8h, v25.4s, #16
	WORD $0x0e6e8ddd // cmtst v29.4h, v14.4h, v14.4h
	WORD $0x0e748e9e // cmtst v30.4h, v20.4h, v20.4h
	WORD $0x6e7d877b // sub v27.8h, v27.8h, v29.8h
	WORD $0x6e7e879c // sub v28.8h, v28.8h, v30.8h
	WORD $0x4edb3a63 // zip1 v3.2d, v19.2d, v27.2d
	WORD $0x4edc3b02 // zip1 v2.2d, v24.2d, v28.2d
	WORD $0x4edb7a7f // zip2 v31.2d, v19.2d, v27.2d
	WORD $0x4edc7b04 // zip2 v4.2d, v24.2d, v28.2d
	WORD $0x4e60b865 // abs v5.8h, v3.8h
	WORD $0x4e60b846 // abs v6.8h, v2.8h
	WORD $0x4e60bbe7 // abs v7.8h, v31.8h
	WORD $0x4e60b88c // abs v12.8h, v4.8h
	WORD $0x3dc0032d // ldr q13, [x25, #0]
	WORD $0x4e6d84a5 // add v5.8h, v5.8h, v13.8h
	WORD $0x4e6d84c6 // add v6.8h, v6.8h, v13.8h
	WORD $0x3dc00730 // ldr q16, [x25, #16]
	WORD $0x4e7084e7 // add v7.8h, v7.8h, v16.8h
	WORD $0x4e70858c // add v12.8h, v12.8h, v16.8h
	WORD $0x2e68c0af // umull v15.4s, v5.4h, v8.4h
	WORD $0x2e68c0d6 // umull v22.4s, v6.4h, v8.4h
	WORD $0x6f70a0b7 // umull2 v23.4s, v5.8h, v0.h[3]
	WORD $0x6f70a0c5 // umull2 v5.4s, v6.8h, v0.h[3]
	WORD $0x2f70a0e6 // umull v6.4s, v7.4h, v0.h[3]
	WORD $0x2f70a195 // umull v21.4s, v12.4h, v0.h[3]
	WORD $0x6f70a0f1 // umull2 v17.4s, v7.8h, v0.h[3]
	WORD $0x6f70a190 // umull2 v16.4s, v12.8h, v0.h[3]
	WORD $0x0e6a41ec // addhn v12.4h, v15.4s, v10.4s
	WORD $0x0e6a42ce // addhn v14.4h, v22.4s, v10.4s
	WORD $0x4e6b42ec // addhn2 v12.8h, v23.4s, v11.4s
	WORD $0x4e6b40ae // addhn2 v14.8h, v5.4s, v11.4s
	WORD $0x0e6b40cd // addhn v13.4h, v6.4s, v11.4s
	WORD $0x0e6b42af // addhn v15.4h, v21.4s, v11.4s
	WORD $0x4e6b422d // addhn2 v13.8h, v17.4s, v11.4s
	WORD $0x4e6b420f // addhn2 v15.8h, v16.4s, v11.4s
	WORD $0x6f1f058c // ushr v12.8h, v12.8h, #1
	WORD $0x6f1f05ce // ushr v14.8h, v14.8h, #1
	WORD $0x6f1f05ad // ushr v13.8h, v13.8h, #1
	WORD $0x6f1f05ef // ushr v15.8h, v15.8h, #1
	WORD $0x4e60a872 // cmlt v18.8h, v3.8h, #0
	WORD $0x4e60a843 // cmlt v3.8h, v2.8h, #0
	WORD $0x4e60abe2 // cmlt v2.8h, v31.8h, #0
	WORD $0x4e60a89a // cmlt v26.8h, v4.8h, #0
	WORD $0x6e321d8c // eor v12.16b, v12.16b, v18.16b
	WORD $0x6e231dce // eor v14.16b, v14.16b, v3.16b
	WORD $0x6e221dad // eor v13.16b, v13.16b, v2.16b
	WORD $0x6e3a1def // eor v15.16b, v15.16b, v26.16b
	WORD $0x6e72858c // sub v12.8h, v12.8h, v18.8h
	WORD $0x6e6385ce // sub v14.8h, v14.8h, v3.8h
	WORD $0x6e6285ad // sub v13.8h, v13.8h, v2.8h
	WORD $0x6e7a85ef // sub v15.8h, v15.8h, v26.8h
	WORD $0x4e699d9a // mul v26.8h, v12.8h, v9.8h
	WORD $0x4e699ddc // mul v28.8h, v14.8h, v9.8h
	WORD $0x4f6081bb // mul v27.8h, v13.8h, v0.h[2]
	WORD $0x4f6081fd // mul v29.8h, v15.8h, v0.h[2]
	WORD $0x3dc01e39 // ldr q25, [x17, #112]
	WORD $0x4e192194 // tbl v20.16b, {v12.16b, v13.16b}, v25.16b
	WORD $0x4e1921de // tbl v30.16b, {v14.16b, v15.16b}, v25.16b
	WORD $0x3dc0223f // ldr q31, [x17, #128]
	WORD $0x4e1f2193 // tbl v19.16b, {v12.16b, v13.16b}, v31.16b
	WORD $0x4e1f21d8 // tbl v24.16b, {v14.16b, v15.16b}, v31.16b
	WORD $0xad004c54 // stp q20, q19, [x2, #0]
	WORD $0xad01605e // stp q30, q24, [x2, #32]
	WORD $0x4e748e99 // cmtst v25.8h, v20.8h, v20.8h
	WORD $0x4e7e8fdf // cmtst v31.8h, v30.8h, v30.8h
	WORD $0x4e738e7e // cmtst v30.8h, v19.8h, v19.8h
	WORD $0x4e788f07 // cmtst v7.8h, v24.8h, v24.8h
	WORD $0x4e1e1b36 // uzp1 v22.16b, v25.16b, v30.16b
	WORD $0x4e071bfe // uzp1 v30.16b, v31.16b, v7.16b
	WORD $0x0f0c86df // shrn v31.8b, v22.8h, #4
	WORD $0x0f0c87d7 // shrn v23.8b, v30.8h, #4
	WORD $0x9e6603e3 // fmov x3, d31
	WORD $0x9e6602e4 // fmov x4, d23
	WORD $0x4e31badf // addv b31, v22.16b
	WORD $0x4e31bbc5 // addv b5, v30.16b
	WORD $0x0e012fe5 // smov w5, v31.b[0]
	WORD $0x0e012ca6 // smov w6, v5.b[0]
	WORD $0x12000067 // and w7, w3, #1
	WORD $0x12000088 // and w8, w4, #1
	WORD $0x0b0700a5 // add w5, w5, w7
	WORD $0x0b0800c6 // add w6, w6, w8
	WORD $0x4b0503e1 // neg w1, w5
	WORD $0x4b0603f3 // neg w19, w6
	WORD $0xdac01069 // clz x9, x3
	WORD $0xdac0108a // clz x10, x4
	WORD $0x4b490ae0 // sub w0, w23, w9, lsr #2
	WORD $0x4b4a0aed // sub w13, w23, w10, lsr #2
	JMP loop
tail:
	WORD $0x4edc3b42 // zip1 v2.2d, v26.2d, v28.2d
	WORD $0x4edc7b43 // zip2 v3.2d, v26.2d, v28.2d
	WORD $0x4edd3b7c // zip1 v28.2d, v27.2d, v29.2d
	WORD $0x4edd7b64 // zip2 v4.2d, v27.2d, v29.2d
	WORD $0x4e7c8445 // add v5.8h, v2.8h, v28.8h
	WORD $0x6e7c8446 // sub v6.8h, v2.8h, v28.8h
	WORD $0x4f40c867 // sqdmulh v7.8h, v3.8h, v0.h[4]
	WORD $0x4f40c882 // sqdmulh v2.8h, v4.8h, v0.h[4]
	WORD $0x4f50c86c // sqdmulh v12.8h, v3.8h, v0.h[5]
	WORD $0x4f50c88d // sqdmulh v13.8h, v4.8h, v0.h[5]
	WORD $0x4e62846e // add v14.8h, v3.8h, v2.8h
	WORD $0x6e67848f // sub v15.8h, v4.8h, v7.8h
	WORD $0x4f1f158e // ssra v14.8h, v12.8h, #1
	WORD $0x4f1f15af // ssra v15.8h, v13.8h, #1
	WORD $0x4e6e84a7 // add v7.8h, v5.8h, v14.8h
	WORD $0x6e6f84d0 // sub v16.8h, v6.8h, v15.8h
	WORD $0x4e6f84d1 // add v17.8h, v6.8h, v15.8h
	WORD $0x6e6e84a6 // sub v6.8h, v5.8h, v14.8h
	WORD $0x4e5028f2 // trn1 v18.8h, v7.8h, v16.8h
	WORD $0x4e5068f3 // trn2 v19.8h, v7.8h, v16.8h
	WORD $0x4e462a27 // trn1 v7.8h, v17.8h, v6.8h
	WORD $0x4e466a34 // trn2 v20.8h, v17.8h, v6.8h
	WORD $0x4e872a55 // trn1 v21.4s, v18.4s, v7.4s
	WORD $0x4e876a56 // trn2 v22.4s, v18.4s, v7.4s
	WORD $0x4e942a77 // trn1 v23.4s, v19.4s, v20.4s
	WORD $0x4e946a78 // trn2 v24.4s, v19.4s, v20.4s
	WORD $0x4e7686b9 // add v25.8h, v21.8h, v22.8h
	WORD $0x6e7686be // sub v30.8h, v21.8h, v22.8h
	WORD $0x4f40caff // sqdmulh v31.8h, v23.8h, v0.h[4]
	WORD $0x4f40cb1a // sqdmulh v26.8h, v24.8h, v0.h[4]
	WORD $0x4f50cafb // sqdmulh v27.8h, v23.8h, v0.h[5]
	WORD $0x4f50cb1d // sqdmulh v29.8h, v24.8h, v0.h[5]
	WORD $0x4e7a86fc // add v28.8h, v23.8h, v26.8h
	WORD $0x6e7f871a // sub v26.8h, v24.8h, v31.8h
	WORD $0x4f1f177c // ssra v28.8h, v27.8h, #1
	WORD $0x4f1f17ba // ssra v26.8h, v29.8h, #1
	WORD $0x4e7c873f // add v31.8h, v25.8h, v28.8h
	WORD $0x6e7a87db // sub v27.8h, v30.8h, v26.8h
	WORD $0x4e7a87dd // add v29.8h, v30.8h, v26.8h
	WORD $0x6e7c873e // sub v30.8h, v25.8h, v28.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4f1d277b // srshr v27.8h, v27.8h, #3
	WORD $0x4f1d27bd // srshr v29.8h, v29.8h, #3
	WORD $0x4f1d27de // srshr v30.8h, v30.8h, #3
	WORD $0x4edb3bfc // zip1 v28.2d, v31.2d, v27.2d
	WORD $0x4edb7be3 // zip2 v3.2d, v31.2d, v27.2d
	WORD $0x4ede3ba2 // zip1 v2.2d, v29.2d, v30.2d
	WORD $0x4ede7bbf // zip2 v31.2d, v29.2d, v30.2d
	WORD $0x3dc002a4 // ldr q4, [x21, #0]
	WORD $0x3dc006ac // ldr q12, [x21, #16]
	WORD $0x3dc0262d // ldr q13, [x17, #144]
	WORD $0x4e0d008f // tbl v15.16b, {v4.16b}, v13.16b
	WORD $0x4e0d018e // tbl v14.16b, {v12.16b}, v13.16b
	WORD $0x2e2f139c // uaddw v28.8h, v28.8h, v15.8b
	WORD $0x2e2e1063 // uaddw v3.8h, v3.8h, v14.8b
	WORD $0x6e2f1042 // uaddw2 v2.8h, v2.8h, v15.16b
	WORD $0x6e2e13ff // uaddw2 v31.8h, v31.8h, v14.16b
	WORD $0x2e212b8c // sqxtun v12.8b, v28.8h
	WORD $0x2e212865 // sqxtun v5.8b, v3.8h
	WORD $0x6e21284c // sqxtun2 v12.16b, v2.8h
	WORD $0x6e212be5 // sqxtun2 v5.16b, v31.8h
	WORD $0x4e0d0190 // tbl v16.16b, {v12.16b}, v13.16b
	WORD $0x4e0d00ac // tbl v12.16b, {v5.16b}, v13.16b
	WORD $0x3d800310 // str q16, [x24, #0]
	WORD $0x3d80070c // str q12, [x24, #16]
	WORD $0x6e217611 // uabd v17.16b, v16.16b, v1.16b
	WORD $0x6e217586 // uabd v6.16b, v12.16b, v1.16b
	WORD $0x3dc01a27 // ldr q7, [x17, #96]
	WORD $0x3dc01a32 // ldr q18, [x17, #96]
	WORD $0x6e919627 // udot v7.4s, v17.16b, v17.16b
	WORD $0x6e8694d2 // udot v18.4s, v6.16b, v6.16b
	WORD $0x4eb1b8e7 // addv s7, v7.4s
	WORD $0x4eb1ba52 // addv s18, v18.4s
	WORD $0x1e2600e3 // fmov w3, s7
	WORD $0x1e260244 // fmov w4, s18
	WORD $0x2f08a613 // uxtl v19.8h, v16.8b
	WORD $0x2f08a594 // uxtl v20.8h, v12.8b
	WORD $0x6f08a615 // uxtl2 v21.8h, v16.16b
	WORD $0x6f08a596 // uxtl2 v22.8h, v12.16b
	WORD $0x4e758677 // add v23.8h, v19.8h, v21.8h
	WORD $0x4e768698 // add v24.8h, v20.8h, v22.8h
	WORD $0x6e75867a // sub v26.8h, v19.8h, v21.8h
	WORD $0x6e768699 // sub v25.8h, v20.8h, v22.8h
	WORD $0x4eda2afb // trn1 v27.2d, v23.2d, v26.2d
	WORD $0x4ed92b1d // trn1 v29.2d, v24.2d, v25.2d
	WORD $0x4eda6afc // trn2 v28.2d, v23.2d, v26.2d
	WORD $0x4ed96b1a // trn2 v26.2d, v24.2d, v25.2d
	WORD $0x4e7c877e // add v30.8h, v27.8h, v28.8h
	WORD $0x4e7a87bf // add v31.8h, v29.8h, v26.8h
	WORD $0x6e7c8764 // sub v4.8h, v27.8h, v28.8h
	WORD $0x6e7a87af // sub v15.8h, v29.8h, v26.8h
	WORD $0x4e441bce // uzp1 v14.8h, v30.8h, v4.8h
	WORD $0x4e4f1be3 // uzp1 v3.8h, v31.8h, v15.8h
	WORD $0x4e445bc2 // uzp2 v2.8h, v30.8h, v4.8h
	WORD $0x4e4f5be5 // uzp2 v5.8h, v31.8h, v15.8h
	WORD $0x4e6285c4 // add v4.8h, v14.8h, v2.8h
	WORD $0x4e65846d // add v13.8h, v3.8h, v5.8h
	WORD $0x6e6285cc // sub v12.8h, v14.8h, v2.8h
	WORD $0x6e658462 // sub v2.8h, v3.8h, v5.8h
	WORD $0x4e4c1883 // uzp1 v3.8h, v4.8h, v12.8h
	WORD $0x4e4219b1 // uzp1 v17.8h, v13.8h, v2.8h
	WORD $0x4e4c5886 // uzp2 v6.8h, v4.8h, v12.8h
	WORD $0x4e4259a7 // uzp2 v7.8h, v13.8h, v2.8h
	WORD $0x4e668462 // add v2.8h, v3.8h, v6.8h
	WORD $0x4e678632 // add v18.8h, v17.8h, v7.8h
	WORD $0x4e667470 // sabd v16.8h, v3.8h, v6.8h
	WORD $0x4e677633 // sabd v19.8h, v17.8h, v7.8h
	WORD $0x4e60b842 // abs v2.8h, v2.8h
	WORD $0x4e60ba52 // abs v18.8h, v18.8h
	WORD $0x3dc02a31 // ldr q17, [x17, #160]
	WORD $0x2e71c055 // umull v21.4s, v2.4h, v17.4h
	WORD $0x2e71c254 // umull v20.4s, v18.4h, v17.4h
	WORD $0x6e718055 // umlal2 v21.4s, v2.8h, v17.8h
	WORD $0x6e718254 // umlal2 v20.4s, v18.8h, v17.8h
	WORD $0x3dc02e36 // ldr q22, [x17, #176]
	WORD $0x2e768215 // umlal v21.4s, v16.4h, v22.4h
	WORD $0x2e768274 // umlal v20.4s, v19.4h, v22.4h
	WORD $0x6e768215 // umlal2 v21.4s, v16.8h, v22.8h
	WORD $0x6e768274 // umlal2 v20.4s, v19.8h, v22.8h
	WORD $0x4eb1bab5 // addv s21, v21.4s
	WORD $0x4eb1ba94 // addv s20, v20.4s
	WORD $0x1e2602a5 // fmov w5, s21
	WORD $0x1e260286 // fmov w6, s20
	WORD $0x29000e80 // stp w0, w3, [x20, #0]
	WORD $0x2902128d // stp w13, w4, [x20, #16]
	WORD $0x29010685 // stp w5, w1, [x20, #8]
	WORD $0x29034e86 // stp w6, w19, [x20, #24]
	RET
loop:
	WORD $0x3dc00aa2 // ldr q2, [x21, #32]
	WORD $0x4edc3b43 // zip1 v3.2d, v26.2d, v28.2d
	WORD $0x3dc00ea4 // ldr q4, [x21, #48]
	WORD $0x4edc7b45 // zip2 v5.2d, v26.2d, v28.2d
	WORD $0x2e22203c // usubl v28.8h, v1.8b, v2.8b
	WORD $0x4edd3b66 // zip1 v6.2d, v27.2d, v29.2d
	WORD $0x2e242027 // usubl v7.8h, v1.8b, v4.8b
	WORD $0x4edd7b6c // zip2 v12.2d, v27.2d, v29.2d
	WORD $0x6e22202d // usubl2 v13.8h, v1.16b, v2.16b
	WORD $0x4e668462 // add v2.8h, v3.8h, v6.8h
	WORD $0x6e24202e // usubl2 v14.8h, v1.16b, v4.16b
	WORD $0x6e668464 // sub v4.8h, v3.8h, v6.8h
	WORD $0x4e4d1b83 // uzp1 v3.8h, v28.8h, v13.8h
	WORD $0x4f40c8a6 // sqdmulh v6.8h, v5.8h, v0.h[4]
	WORD $0x4e4e18ef // uzp1 v15.8h, v7.8h, v14.8h
	WORD $0x4f40c990 // sqdmulh v16.8h, v12.8h, v0.h[4]
	WORD $0x4e4d5b91 // uzp2 v17.8h, v28.8h, v13.8h
	WORD $0x4f50c8ad // sqdmulh v13.8h, v5.8h, v0.h[5]
	WORD $0x4e4e58f2 // uzp2 v18.8h, v7.8h, v14.8h
	WORD $0x4f50c987 // sqdmulh v7.8h, v12.8h, v0.h[5]
	WORD $0x4e51186e // uzp1 v14.8h, v3.8h, v17.8h
	WORD $0x4e7084b3 // add v19.8h, v5.8h, v16.8h
	WORD $0x4e5219e5 // uzp1 v5.8h, v15.8h, v18.8h
	WORD $0x6e668590 // sub v16.8h, v12.8h, v6.8h
	WORD $0x4e435a2c // uzp2 v12.8h, v17.8h, v3.8h
	WORD $0x4f1f15b3 // ssra v19.8h, v13.8h, #1
	WORD $0x4e4f5a46 // uzp2 v6.8h, v18.8h, v15.8h
	WORD $0x4f1f14f0 // ssra v16.8h, v7.8h, #1
	WORD $0x4e6c85c3 // add v3.8h, v14.8h, v12.8h
	WORD $0x4e73844d // add v13.8h, v2.8h, v19.8h
	WORD $0x4e6684af // add v15.8h, v5.8h, v6.8h
	WORD $0x6e708487 // sub v7.8h, v4.8h, v16.8h
	WORD $0x6e6c85d4 // sub v20.8h, v14.8h, v12.8h
	WORD $0x4e70848e // add v14.8h, v4.8h, v16.8h
	WORD $0x6e6684ac // sub v12.8h, v5.8h, v6.8h
	WORD $0x6e738455 // sub v21.8h, v2.8h, v19.8h
	WORD $0x6e034062 // ext v2.16b, v3.16b, v3.16b, #8
	WORD $0x4e4729b6 // trn1 v22.8h, v13.8h, v7.8h
	WORD $0x6e0f41f7 // ext v23.16b, v15.16b, v15.16b, #8
	WORD $0x4e4769b8 // trn2 v24.8h, v13.8h, v7.8h
	WORD $0x4e62846d // add v13.8h, v3.8h, v2.8h
	WORD $0x4e5529d9 // trn1 v25.8h, v14.8h, v21.8h
	WORD $0x4e7785fe // add v30.8h, v15.8h, v23.8h
	WORD $0x4e5569df // trn2 v31.8h, v14.8h, v21.8h
	WORD $0x6e62846e // sub v14.8h, v3.8h, v2.8h
	WORD $0x4e992ad5 // trn1 v21.4s, v22.4s, v25.4s
	WORD $0x6e7785fa // sub v26.8h, v15.8h, v23.8h
	WORD $0x4e996ad7 // trn2 v23.4s, v22.4s, v25.4s
	WORD $0x3dc00a36 // ldr q22, [x17, #32]
	WORD $0x4e9f2b19 // trn1 v25.4s, v24.4s, v31.4s
	WORD $0x3dc00a3b // ldr q27, [x17, #32]
	WORD $0x4e9f6b1d // trn2 v29.4s, v24.4s, v31.4s
	WORD $0x4f402296 // smlal2 v22.4s, v20.8h, v0.h[0]
	WORD $0x4e7786bc // add v28.8h, v21.8h, v23.8h
	WORD $0x4f40219b // smlal2 v27.4s, v12.8h, v0.h[0]
	WORD $0x6e7786b8 // sub v24.8h, v21.8h, v23.8h
	WORD $0x0f502296 // smlal v22.4s, v20.4h, v0.h[1]
	WORD $0x4f40cb3f // sqdmulh v31.8h, v25.8h, v0.h[4]
	WORD $0x0f50219b // smlal v27.4s, v12.4h, v0.h[1]
	WORD $0x4f40cbb5 // sqdmulh v21.8h, v29.8h, v0.h[4]
	WORD $0x3dc00e37 // ldr q23, [x17, #48]
	WORD $0x4f50cb31 // sqdmulh v17.8h, v25.8h, v0.h[5]
	WORD $0x3dc00e32 // ldr q18, [x17, #48]
	WORD $0x4f50cbb3 // sqdmulh v19.8h, v29.8h, v0.h[5]
	WORD $0x0f402297 // smlal v23.4s, v20.4h, v0.h[0]
	WORD $0x4e758724 // add v4.8h, v25.8h, v21.8h
	WORD $0x0f402192 // smlal v18.4s, v12.4h, v0.h[0]
	WORD $0x6e7f87b9 // sub v25.8h, v29.8h, v31.8h
	WORD $0x4f506297 // smlsl2 v23.4s, v20.8h, v0.h[1]
	WORD $0x4f1f1624 // ssra v4.8h, v17.8h, #1
	WORD $0x4f506192 // smlsl2 v18.4s, v12.8h, v0.h[1]
	WORD $0x4f1f1679 // ssra v25.8h, v19.8h, #1
	WORD $0x4f1355bd // shl v29.8h, v13.8h, #3
	WORD $0x4e64879f // add v31.8h, v28.8h, v4.8h
	WORD $0x4f1357d0 // shl v16.8h, v30.8h, #3
	WORD $0x6e79870f // sub v15.8h, v24.8h, v25.8h
	WORD $0x4f1786dd // shrn2 v29.8h, v22.4s, #9
	WORD $0x4e798711 // add v17.8h, v24.8h, v25.8h
	WORD $0x4f178770 // shrn2 v16.8h, v27.4s, #9
	WORD $0x6e64879e // sub v30.8h, v28.8h, v4.8h
	WORD $0x4f1355c5 // shl v5.8h, v14.8h, #3
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4f135746 // shl v6.8h, v26.8h, #3
	WORD $0x4f1d25ef // srshr v15.8h, v15.8h, #3
	WORD $0x4f1786e5 // shrn2 v5.8h, v23.4s, #9
	WORD $0x4f1d2631 // srshr v17.8h, v17.8h, #3
	WORD $0x4f178646 // shrn2 v6.8h, v18.4s, #9
	WORD $0x4f1d27de // srshr v30.8h, v30.8h, #3
	WORD $0x4e451ba7 // uzp1 v7.8h, v29.8h, v5.8h
	WORD $0x4ecf3be3 // zip1 v3.2d, v31.2d, v15.2d
	WORD $0x4e461a02 // uzp1 v2.8h, v16.8h, v6.8h
	WORD $0x4ecf7be4 // zip2 v4.2d, v31.2d, v15.2d
	WORD $0x4e455bbf // uzp2 v31.8h, v29.8h, v5.8h
	WORD $0x4ede3a25 // zip1 v5.2d, v17.2d, v30.2d
	WORD $0x4e465a15 // uzp2 v21.8h, v16.8h, v6.8h
	WORD $0x4ede7a26 // zip2 v6.2d, v17.2d, v30.2d
	WORD $0x4e5f18f4 // uzp1 v20.8h, v7.8h, v31.8h
	WORD $0x3dc002ac // ldr q12, [x21, #0]
	WORD $0x4e551853 // uzp1 v19.8h, v2.8h, v21.8h
	WORD $0x3dc006ad // ldr q13, [x21, #16]
	WORD $0x4e475bf6 // uzp2 v22.8h, v31.8h, v7.8h
	WORD $0x3dc02627 // ldr q7, [x17, #144]
	WORD $0x4e070198 // tbl v24.16b, {v12.16b}, v7.16b
	WORD $0x4e425ab7 // uzp2 v23.8h, v21.8h, v2.8h
	WORD $0x4e0701ac // tbl v12.16b, {v13.16b}, v7.16b
	WORD $0x4e768695 // add v21.8h, v20.8h, v22.8h
	WORD $0x2e381063 // uaddw v3.8h, v3.8h, v24.8b
	WORD $0x4e778662 // add v2.8h, v19.8h, v23.8h
	WORD $0x2e2c1084 // uaddw v4.8h, v4.8h, v12.8b
	WORD $0x6e768699 // sub v25.8h, v20.8h, v22.8h
	WORD $0x6e3810a5 // uaddw2 v5.8h, v5.8h, v24.16b
	WORD $0x6e778674 // sub v20.8h, v19.8h, v23.8h
	WORD $0x6e2c10c6 // uaddw2 v6.8h, v6.8h, v12.16b
	WORD $0x3dc0063b // ldr q27, [x17, #16]
	WORD $0x4e7b86ba // add v26.8h, v21.8h, v27.8h
	WORD $0x2e21287c // sqxtun v28.8b, v3.8h
	WORD $0x4e7b8443 // add v3.8h, v2.8h, v27.8h
	WORD $0x2e21289b // sqxtun v27.8b, v4.8h
	WORD $0x6e1a4344 // ext v4.16b, v26.16b, v26.16b, #8
	WORD $0x6e2128bc // sqxtun2 v28.16b, v5.8h
	WORD $0x6e03407a // ext v26.16b, v3.16b, v3.16b, #8
	WORD $0x6e2128db // sqxtun2 v27.16b, v6.8h
	WORD $0x4e758483 // add v3.8h, v4.8h, v21.8h
	WORD $0x4e07038e // tbl v14.16b, {v28.16b}, v7.16b
	WORD $0x4e628752 // add v18.8h, v26.8h, v2.8h
	WORD $0x4e070373 // tbl v19.16b, {v27.16b}, v7.16b
	WORD $0x6e75848f // sub v15.8h, v4.8h, v21.8h
	WORD $0x3d80030e // str q14, [x24, #0]
	WORD $0x6e62875d // sub v29.8h, v26.8h, v2.8h
	WORD $0x3d800713 // str q19, [x24, #16]
	WORD $0x4ecf7862 // zip2 v2.2d, v3.2d, v15.2d
	WORD $0x6e2175d0 // uabd v16.16b, v14.16b, v1.16b
	WORD $0x4edd7a51 // zip2 v17.2d, v18.2d, v29.2d
	WORD $0x6e21766f // uabd v15.16b, v19.16b, v1.16b
	WORD $0x4f1c0442 // sshr v2.8h, v2.8h, #4
	WORD $0x3dc01a32 // ldr q18, [x17, #96]
	WORD $0x4f1c0631 // sshr v17.8h, v17.8h, #4
	WORD $0x3dc01a3e // ldr q30, [x17, #96]
	WORD $0x3dc0123f // ldr q31, [x17, #64]
	WORD $0x6e909612 // udot v18.4s, v16.16b, v16.16b
	WORD $0x3dc01230 // ldr q16, [x17, #64]
	WORD $0x6e8f95fe // udot v30.4s, v15.16b, v15.16b
	WORD $0x0f50233f // smlal v31.4s, v25.4h, v0.h[1]
	WORD $0x4eb1ba52 // addv s18, v18.4s
	WORD $0x0f502290 // smlal v16.4s, v20.4h, v0.h[1]
	WORD $0x4eb1bbde // addv s30, v30.4s
	WORD $0x4f40233f // smlal2 v31.4s, v25.8h, v0.h[0]
	WORD $0x1e260243 // fmov w3, s18
	WORD $0x4f402290 // smlal2 v16.4s, v20.8h, v0.h[0]
	WORD $0x1e2603c4 // fmov w4, s30
	WORD $0x3dc0162f // ldr q15, [x17, #80]
	WORD $0x2f08a5d2 // uxtl v18.8h, v14.8b
	WORD $0x3dc0162d // ldr q13, [x17, #80]
	WORD $0x2f08a66c // uxtl v12.8h, v19.8b
	WORD $0x0f40232f // smlal v15.4s, v25.4h, v0.h[0]
	WORD $0x6f08a5d6 // uxtl2 v22.8h, v14.16b
	WORD $0x0f40228d // smlal v13.4s, v20.4h, v0.h[0]
	WORD $0x6f08a675 // uxtl2 v21.8h, v19.16b
	WORD $0x4f50632f // smlsl2 v15.4s, v25.8h, v0.h[1]
	WORD $0x4e76864e // add v14.8h, v18.8h, v22.8h
	WORD $0x4f50628d // smlsl2 v13.4s, v20.8h, v0.h[1]
	WORD $0x4e758593 // add v19.8h, v12.8h, v21.8h
	WORD $0x0f1087f8 // shrn v24.4h, v31.4s, #16
	WORD $0x6e768657 // sub v23.8h, v18.8h, v22.8h
	WORD $0x0f108612 // shrn v18.4h, v16.4s, #16
	WORD $0x6e758596 // sub v22.8h, v12.8h, v21.8h
	WORD $0x4f1085f8 // shrn2 v24.8h, v15.4s, #16
	WORD $0x4ed729cc // trn1 v12.2d, v14.2d, v23.2d
	WORD $0x4f1085b2 // shrn2 v18.8h, v13.4s, #16
	WORD $0x4ed62a75 // trn1 v21.2d, v19.2d, v22.2d
	WORD $0x0e798f2d // cmtst v13.4h, v25.4h, v25.4h
	WORD $0x4ed769c5 // trn2 v5.2d, v14.2d, v23.2d
	WORD $0x0e748e97 // cmtst v23.4h, v20.4h, v20.4h
	WORD $0x4ed66a74 // trn2 v20.2d, v19.2d, v22.2d
	WORD $0x6e6d8718 // sub v24.8h, v24.8h, v13.8h
	WORD $0x4e658593 // add v19.8h, v12.8h, v5.8h
	WORD $0x6e778652 // sub v18.8h, v18.8h, v23.8h
	WORD $0x4e7486a6 // add v6.8h, v21.8h, v20.8h
	WORD $0x4ed83847 // zip1 v7.2d, v2.2d, v24.2d
	WORD $0x6e65859c // sub v28.8h, v12.8h, v5.8h
	WORD $0x4ed23a3b // zip1 v27.2d, v17.2d, v18.2d
	WORD $0x6e7486a4 // sub v4.8h, v21.8h, v20.8h
	WORD $0x4ed87843 // zip2 v3.2d, v2.2d, v24.2d
	WORD $0x4e5c1a65 // uzp1 v5.8h, v19.8h, v28.8h
	WORD $0x4ed27a22 // zip2 v2.2d, v17.2d, v18.2d
	WORD $0x4e4418da // uzp1 v26.8h, v6.8h, v4.8h
	WORD $0x4e60b8fd // abs v29.8h, v7.8h
	WORD $0x4e5c5a7e // uzp2 v30.8h, v19.8h, v28.8h
	WORD $0x4e60bb7f // abs v31.8h, v27.8h
	WORD $0x4e4458dc // uzp2 v28.8h, v6.8h, v4.8h
	WORD $0x4e60b866 // abs v6.8h, v3.8h
	WORD $0x4e7e84a4 // add v4.8h, v5.8h, v30.8h
	WORD $0x4e60b850 // abs v16.8h, v2.8h
	WORD $0x4e7c874f // add v15.8h, v26.8h, v28.8h
	WORD $0x3dc00339 // ldr q25, [x25, #0]
	WORD $0x4e7987bd // add v29.8h, v29.8h, v25.8h
	WORD $0x6e7e84ae // sub v14.8h, v5.8h, v30.8h
	WORD $0x4e7987ff // add v31.8h, v31.8h, v25.8h
	WORD $0x6e7c8745 // sub v5.8h, v26.8h, v28.8h
	WORD $0x3dc0073e // ldr q30, [x25, #16]
	WORD $0x4e7e84c6 // add v6.8h, v6.8h, v30.8h
	WORD $0x4e4e189c // uzp1 v28.8h, v4.8h, v14.8h
	WORD $0x4e7e8610 // add v16.8h, v16.8h, v30.8h
	WORD $0x4e4519fe // uzp1 v30.8h, v15.8h, v5.8h
	WORD $0x2e68c3b6 // umull v22.4s, v29.4h, v8.4h
	WORD $0x4e4e588d // uzp2 v13.8h, v4.8h, v14.8h
	WORD $0x2e68c3ec // umull v12.4s, v31.4h, v8.4h
	WORD $0x4e4559e4 // uzp2 v4.8h, v15.8h, v5.8h
	WORD $0x6f70a3a5 // umull2 v5.4s, v29.8h, v0.h[3]
	WORD $0x4e6d879d // add v29.8h, v28.8h, v13.8h
	WORD $0x6f70a3f7 // umull2 v23.4s, v31.8h, v0.h[3]
	WORD $0x4e6487df // add v31.8h, v30.8h, v4.8h
	WORD $0x2f70a0d5 // umull v21.4s, v6.4h, v0.h[3]
	WORD $0x4e6d7794 // sabd v20.8h, v28.8h, v13.8h
	WORD $0x2f70a21c // umull v28.4s, v16.4h, v0.h[3]
	WORD $0x4e6477d8 // sabd v24.8h, v30.8h, v4.8h
	WORD $0x6f70a0de // umull2 v30.4s, v6.8h, v0.h[3]
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x6f70a204 // umull2 v4.4s, v16.8h, v0.h[3]
	WORD $0x4e60bbff // abs v31.8h, v31.8h
	WORD $0x0e6a42cd // addhn v13.4h, v22.4s, v10.4s
	WORD $0x3dc02a26 // ldr q6, [x17, #160]
	WORD $0x2e66c3b6 // umull v22.4s, v29.4h, v6.4h
	WORD $0x0e6a418f // addhn v15.4h, v12.4s, v10.4s
	WORD $0x2e66c3ec // umull v12.4s, v31.4h, v6.4h
	WORD $0x4e6b40ad // addhn2 v13.8h, v5.4s, v11.4s
	WORD $0x6e6683b6 // umlal2 v22.4s, v29.8h, v6.8h
	WORD $0x4e6b42ef // addhn2 v15.8h, v23.4s, v11.4s
	WORD $0x6e6683ec // umlal2 v12.4s, v31.8h, v6.8h
	WORD $0x0e6b42ae // addhn v14.4h, v21.4s, v11.4s
	WORD $0x3dc02e3d // ldr q29, [x17, #176]
	WORD $0x2e7d8296 // umlal v22.4s, v20.4h, v29.4h
	WORD $0x0e6b4390 // addhn v16.4h, v28.4s, v11.4s
	WORD $0x2e7d830c // umlal v12.4s, v24.4h, v29.4h
	WORD $0x4e6b43ce // addhn2 v14.8h, v30.4s, v11.4s
	WORD $0x6e7d8296 // umlal2 v22.4s, v20.8h, v29.8h
	WORD $0x4e6b4090 // addhn2 v16.8h, v4.4s, v11.4s
	WORD $0x6e7d830c // umlal2 v12.4s, v24.8h, v29.8h
	WORD $0x6f1f05ad // ushr v13.8h, v13.8h, #1
	WORD $0x4eb1bad6 // addv s22, v22.4s
	WORD $0x6f1f05ef // ushr v15.8h, v15.8h, #1
	WORD $0x4eb1b98c // addv s12, v12.4s
	WORD $0x6f1f05ce // ushr v14.8h, v14.8h, #1
	WORD $0x1e2602c5 // fmov w5, s22
	WORD $0x6f1f0610 // ushr v16.8h, v16.8h, #1
	WORD $0x1e260186 // fmov w6, s12
	WORD $0x4e60a8ec // cmlt v12.8h, v7.8h, #0
	WORD $0x29000e80 // stp w0, w3, [x20, #0]
	WORD $0x4e60ab71 // cmlt v17.8h, v27.8h, #0
	WORD $0x2902128d // stp w13, w4, [x20, #16]
	WORD $0x4e60a872 // cmlt v18.8h, v3.8h, #0
	WORD $0x29010685 // stp w5, w1, [x20, #8]
	WORD $0x4e60a853 // cmlt v19.8h, v2.8h, #0
	WORD $0x29034e86 // stp w6, w19, [x20, #24]
	WORD $0x6e2c1dad // eor v13.16b, v13.16b, v12.16b
	WORD $0x6e311def // eor v15.16b, v15.16b, v17.16b
	WORD $0x6e321dce // eor v14.16b, v14.16b, v18.16b
	WORD $0x6e331e10 // eor v16.16b, v16.16b, v19.16b
	WORD $0x6e6c85ad // sub v13.8h, v13.8h, v12.8h
	WORD $0x6e7185ef // sub v15.8h, v15.8h, v17.8h
	WORD $0x6e7285ce // sub v14.8h, v14.8h, v18.8h
	WORD $0x6e738610 // sub v16.8h, v16.8h, v19.8h
	WORD $0x4e699dba // mul v26.8h, v13.8h, v9.8h
	WORD $0x4e699dfc // mul v28.8h, v15.8h, v9.8h
	WORD $0x4f6081db // mul v27.8h, v14.8h, v0.h[2]
	WORD $0x4f60821d // mul v29.8h, v16.8h, v0.h[2]
	WORD $0x3dc01e2c // ldr q12, [x17, #112]
	WORD $0x4e0c21b9 // tbl v25.16b, {v13.16b, v14.16b}, v12.16b
	WORD $0x4e0c21e5 // tbl v5.16b, {v15.16b, v16.16b}, v12.16b
	WORD $0x3dc0222c // ldr q12, [x17, #128]
	WORD $0x4e0c21b7 // tbl v23.16b, {v13.16b, v14.16b}, v12.16b
	WORD $0x4e0c21f8 // tbl v24.16b, {v15.16b, v16.16b}, v12.16b
	WORD $0xad025c59 // stp q25, q23, [x2, #64]
	WORD $0xad036045 // stp q5, q24, [x2, #96]
	WORD $0x4e798f3f // cmtst v31.8h, v25.8h, v25.8h
	WORD $0x4e658cbe // cmtst v30.8h, v5.8h, v5.8h
	WORD $0x4e778ef9 // cmtst v25.8h, v23.8h, v23.8h
	WORD $0x4e788f06 // cmtst v6.8h, v24.8h, v24.8h
	WORD $0x4e191be7 // uzp1 v7.16b, v31.16b, v25.16b
	WORD $0x4e061bdf // uzp1 v31.16b, v30.16b, v6.16b
	WORD $0x0f0c84fe // shrn v30.8b, v7.8h, #4
	WORD $0x0f0c87f5 // shrn v21.8b, v31.8h, #4
	WORD $0x9e6603c7 // fmov x7, d30
	WORD $0x9e6602a8 // fmov x8, d21
	WORD $0x4e31b8fe // addv b30, v7.16b
	WORD $0x4e31bbf4 // addv b20, v31.16b
	WORD $0x0e012fc9 // smov w9, v30.b[0]
	WORD $0x0e012e8a // smov w10, v20.b[0]
	WORD $0x120000eb // and w11, w7, #1
	WORD $0x1200010c // and w12, w8, #1
	WORD $0x0b0b0129 // add w9, w9, w11
	WORD $0x0b0c014a // add w10, w10, w12
	WORD $0x4b0903e1 // neg w1, w9
	WORD $0x4b0a03f3 // neg w19, w10
	WORD $0xdac010ee // clz x14, x7
	WORD $0xdac0110f // clz x15, x8
	WORD $0x4b4e0ae0 // sub w0, w23, w14, lsr #2
	WORD $0x4b4f0aed // sub w13, w23, w15, lsr #2
	WORD $0x910082b5 // add x21, x21, #32
	WORD $0x91010042 // add x2, x2, #64
	WORD $0x91008318 // add x24, x24, #32
	WORD $0x91008294 // add x20, x20, #32
	WORD $0xf10006d6 // subs x22, x22, #1
	BNE loop
	JMP tail

// 256*trellisWeight in zigzag order.
DATA trellisWZ<>+0(SB)/8, $7680
DATA trellisWZ<>+8(SB)/8, $6912
DATA trellisWZ<>+16(SB)/8, $6912
DATA trellisWZ<>+24(SB)/8, $4864
DATA trellisWZ<>+32(SB)/8, $6144
DATA trellisWZ<>+40(SB)/8, $4864
DATA trellisWZ<>+48(SB)/8, $2816
DATA trellisWZ<>+56(SB)/8, $4352
DATA trellisWZ<>+64(SB)/8, $4352
DATA trellisWZ<>+72(SB)/8, $2816
DATA trellisWZ<>+80(SB)/8, $2560
DATA trellisWZ<>+88(SB)/8, $3072
DATA trellisWZ<>+96(SB)/8, $2560
DATA trellisWZ<>+104(SB)/8, $2048
DATA trellisWZ<>+112(SB)/8, $2048
DATA trellisWZ<>+120(SB)/8, $1536
GLOBL trellisWZ<>(SB), RODATA|NOPTR, $128

// func trellisAsm(s *costTabs, m *quantMatrix, in, out *[16]int16, ctx0, typ, lambda int) bool
// in is fdct output, so never -32768.
// The frame holds {c0, l0, tl, 0} uint16 per zigzag position.
// R8/R9 node scores, R10/R11 their cost rows, R12 best score,
// R13 path end n+16*d or -1, R15 prev bits at n+16*d.
TEXT ·trellisAsm(SB), NOSPLIT, $136-57
	MOVD s+0(FP), R3
	MOVD m+8(FP), R2
	MOVD in+16(FP), R0
	MOVD out+24(FP), R1
	MOVD ctx0+32(FP), R17
	MOVD typ+40(FP), R6
	MOVD lambda+48(FP), R5
	CMP  $0, R6
	CSET EQ, R7

	// p0 = probs[typ*264+(first*3+ctx0)*11]
	MOVD  $264, R8
	MUL   R8, R6, R8
	ADD   R7<<1, R7, R9
	ADD   R17, R9, R9
	MOVD  $11, R10
	MADD  R10, R8, R9, R9
	ADD   $costTabs_probs, R3, R10
	MOVBU (R10)(R9), R9

	ADD  $costTabs_eob, R3, R4
	ADD  R6<<7, R4, R4
	ADD  R7<<3, R4, R4
	ADD  $costTabs_cost, R3, R3
	ADD  R6<<14, R3, R3
	ADD  R7<<10, R3, R10
	ADD  $1024, R10, R24
	ADD  R17<<8, R10, R10
	MOVD R10, R11

	MOVD $·bitCost(SB), R16
	MOVD (R16)(R9<<3), R12
	MUL  R5, R12, R12
	ADD  $2048, R16, R16
	MOVD (R16)(R9<<3), R8
	MUL  R5, R8, R8
	CMP  $0, R17
	CSEL EQ, R8, ZR, R8
	MOVD R8, R9

	// v in zigzag order.
	VLD1    (R0), [V0.H8, V1.H8]
	VMOVQ   $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ   $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	VTBL    V28.B16, [V0.B16, V1.B16], V2.B16
	VTBL    V29.B16, [V0.B16, V1.B16], V3.B16

	// last = max(min(1+highest n with v*v > q1*q1/4, 15), first).
	MOVWU   quantMatrix_q+4(R2), R16
	MULW    R16, R16, R16
	LSRW    $2, R16, R16
	VDUP    R16, V16.S4
	VSMULL  V2.H4, V2.H4, V4.S4
	VSMULL2 V2.H8, V2.H8, V5.S4
	VSMULL  V3.H4, V3.H4, V6.S4
	VSMULL2 V3.H8, V3.H8, V7.S4
	VCMGT   V16.S4, V4.S4, V4.S4
	VCMGT   V16.S4, V5.S4, V5.S4
	VCMGT   V16.S4, V6.S4, V6.S4
	VCMGT   V16.S4, V7.S4, V7.S4
	VUZP1   V5.H8, V4.H8, V4.H8
	VUZP1   V7.H8, V6.H8, V6.H8
	VMOVQ   $0x0004000300020001, $0x0008000700060005, V30
	VMOVQ   $0x000c000b000a0009, $0x0010000f000e000d, V31
	VAND    V30.B16, V4.B16, V4.B16
	VAND    V31.B16, V6.B16, V6.B16
	VUMAX   V6.H8, V4.H8, V4.H8
	VUMAXV  V4.H8, V4
	VMOV    V4.H[0], R19
	MOVD    $15, R26
	CMP     R26, R19
	CSEL    HI, R26, R19, R19
	CMP     R7, R19
	CSEL    LT, R7, R19, R19

	// c0 = |v| + sharpen, l0 = min(c0*iq >> 17, 2047),
	// tl likewise with +1<<16; l0 = 0 where tl == 0.
	ADD     $quantMatrix_sharpen, R2, R16
	VLD1    (R16), [V16.S4, V17.S4, V18.S4, V19.S4]
	VUZP1   V17.H8, V16.H8, V16.H8
	VUZP1   V19.H8, V18.H8, V17.H8
	VTBL    V28.B16, [V16.B16, V17.B16], V18.B16
	VTBL    V29.B16, [V16.B16, V17.B16], V19.B16
	VABS    V2.H8, V24.H8
	VABS    V3.H8, V28.H8
	VADD    V18.H8, V24.H8, V24.H8
	VADD    V19.H8, V28.H8, V28.H8
	MOVWU   quantMatrix_iq+4(R2), R16
	MOVWU   quantMatrix_iq(R2), R17
	VDUP    R16, V22.H8
	VMOV    V22.B16, V23.B16
	VMOV    R17, V23.H[0]
	VUMULL  V23.H4, V24.H4, V4.S4
	VUMULL2 V23.H8, V24.H8, V5.S4
	VUMULL  V22.H4, V28.H4, V6.S4
	VUMULL2 V22.H8, V28.H8, V7.S4
	MOVD    $0x10000, R16
	VDUP    R16, V16.S4
	VADD    V16.S4, V4.S4, V8.S4
	VADD    V16.S4, V5.S4, V9.S4
	VADD    V16.S4, V6.S4, V10.S4
	VADD    V16.S4, V7.S4, V11.S4
	VSHRN   $16, V4.S4, V25.H4
	VSHRN2  $16, V5.S4, V25.H8
	VSHRN   $16, V6.S4, V29.H4
	VSHRN2  $16, V7.S4, V29.H8
	VSHRN   $16, V8.S4, V26.H4
	VSHRN2  $16, V9.S4, V26.H8
	VSHRN   $16, V10.S4, V30.H4
	VSHRN2  $16, V11.S4, V30.H8
	VUSHR   $1, V25.H8, V25.H8
	VUSHR   $1, V29.H8, V29.H8
	VUSHR   $1, V26.H8, V26.H8
	VUSHR   $1, V30.H8, V30.H8
	MOVD    $2047, R16
	VDUP    R16, V16.H8
	VUMIN   V16.H8, V25.H8, V25.H8
	VUMIN   V16.H8, V29.H8, V29.H8
	VUMIN   V16.H8, V26.H8, V26.H8
	VUMIN   V16.H8, V30.H8, V30.H8
	VCMTST  V26.H8, V26.H8, V17.H8
	VCMTST  V30.H8, V30.H8, V18.H8
	VAND    V17.B16, V25.B16, V25.B16
	VAND    V18.B16, V29.B16, V29.B16
	VEOR    V27.B16, V27.B16, V27.B16
	VEOR    V31.B16, V31.B16, V31.B16
	ADD     $8, RSP, R3
	VST4    [V24.H8, V25.H8, V26.H8, V27.H8], (R3)
	ADD     $64, R3, R16
	VST4    [V28.H8, V29.H8, V30.H8, V31.H8], (R16)

	ADD      R7<<3, R3, R3
	MOVD     R7, R6
	MOVD     R19, R7
	MOVWU    quantMatrix_q(R2), R25
	MOVWU    quantMatrix_q+4(R2), R2
	CMP      $0, R6
	CSEL     EQ, R25, R2, R25
	MOVD     $trellisWZ<>(SB), R19
	MOVD     $0, R15
	MOVD     $-1, R13

tloop:
	MOVHU 4(R3), R23
	CBNZ  R23, tfull

	// tl == 0: one level-0 node that cannot end the block.
	MOVHU (R10), R14
	MOVHU (R11), R16
	MADD  R5, R8, R14, R14
	MADD  R5, R9, R16, R16
	CMP   R14, R16
	CSEL  LT, R16, R14, R8
	CSET  LT, R26
	LSL   R6, R26, R26
	ORR   R26, R15, R15
	MOVD  $0x7fffffffffffff, R9
	MOVD  R24, R10
	MOVD  R24, R11
	B     tnext

tfull:
	MOVHU (R3), R21
	MOVHU 2(R3), R22
	MOVD  (R19)(R6<<3), R20
	MUL   R21, R21, R17

	// Node 0 always lives: tl >= l0.
	CMP   $67, R22
	BHI   tbig0
	MOVHU (R10)(R22<<1), R16
	MOVHU (R11)(R22<<1), R26

tcost0:
	MADD  R5, R8, R16, R16
	MADD  R5, R9, R26, R26
	CMP   R16, R26
	CSEL  LT, R26, R16, R16
	CSET  LT, R26
	LSL   R6, R26, R26
	ORR   R26, R15, R15
	MSUB  R25, R21, R22, R14
	MUL   R14, R14, R14
	SUB   R17, R14, R14
	MADD  R20, R16, R14, R14
	CBZ   R22, tnode1
	MOVD  $2, R26
	CMP   R26, R22
	CSEL  HI, R26, R22, R16
	MOVHU (R4)(R16<<1), R16
	MADD  R5, R14, R16, R16
	CMP   R12, R16
	CSEL  LT, R16, R12, R12
	CSEL  LT, R6, R13, R13

tnode1:
	ADD   $1, R22, R22
	CMP   R23, R22
	BHI   tno1
	CMP   $67, R22
	BHI   tbig1
	MOVHU (R10)(R22<<1), R16
	MOVHU (R11)(R22<<1), R26

tcost1:
	MADD  R5, R8, R16, R16
	MADD  R5, R9, R26, R26
	CMP   R16, R26
	CSEL  LT, R26, R16, R16
	CSET  LT, R26
	LSL   R6, R26, R26
	ORR   R26<<16, R15, R15
	MSUB  R25, R21, R22, R23
	MUL   R23, R23, R23
	SUB   R17, R23, R23
	MADD  R20, R16, R23, R9
	MOVD  $2, R26
	CMP   R26, R22
	CSEL  HI, R26, R22, R16
	MOVHU (R4)(R16<<1), R16
	MADD  R5, R9, R16, R16
	CMP   R12, R16
	CSEL  LT, R16, R12, R12
	ADD   $16, R6, R26
	CSEL  LT, R26, R13, R13
	B     trows

tno1:
	MOVD $0x7fffffffffffff, R9

trows:
	MOVD R14, R8
	MOVD $2, R26
	SUB  $1, R22, R16
	CMP  R26, R16
	CSEL HI, R26, R16, R16
	ADD  R16<<8, R24, R10
	CMP  R26, R22
	CSEL HI, R26, R22, R22
	ADD  R22<<8, R24, R11

tnext:
	ADD  $8, R3
	ADD  $8, R4
	ADD  $1024, R24
	MOVD R2, R25
	ADD  $1, R6
	CMP  R7, R6
	BLE  tloop

	// Backtrack: bit n of R20 picks node 1 at n on the path;
	// then levels in NEON, zero past pathN.
	MOVD  typ+40(FP), R16
	CMP   $0, R16
	CSET  EQ, R16
	MOVH  (R0), R24 // in[0] and out[0] stay when first == 1
	MOVH  (R1), R25
	MOVD  $0, R20
	AND   $15, R13, R14
	LSR   $4, R13, R26
	CMP   $0, R13
	CSINV GE, R14, ZR, R14
	BLT   tvec
	MOVD  R14, R6

tpath:
	LSL  R6, R26, R21
	ORR  R21, R20, R20
	ADD  R26<<4, R6, R21
	LSR  R21, R15, R26
	AND  $1, R26, R26
	SUB  $1, R6
	CMP  R16, R6
	BGE  tpath

tvec:
	ADD    $8, RSP, R3
	VLD4   (R3), [V4.H8, V5.H8, V6.H8, V7.H8]
	ADD    $64, R3, R3
	VLD4   (R3), [V8.H8, V9.H8, V10.H8, V11.H8]
	VDUP   R20, V12.H8
	VMOVQ  $0x0008000400020001, $0x0080004000200010, V13
	VMOVQ  $0x0800040002000100, $0x8000400020001000, V14
	VCMTST V13.H8, V12.H8, V15.H8
	VCMTST V14.H8, V12.H8, V16.H8
	VSUB   V15.H8, V5.H8, V5.H8
	VSUB   V16.H8, V9.H8, V9.H8
	VDUP   R14, V12.H8
	VMOVQ  $0x0003000200010000, $0x0007000600050004, V13
	VMOVQ  $0x000b000a00090008, $0x000f000e000d000c, V14
	VCMGE  V13.H8, V12.H8, V15.H8
	VCMGE  V14.H8, V12.H8, V16.H8
	VAND   V15.B16, V5.B16, V5.B16
	VAND   V16.B16, V9.B16, V6.B16
	VLD1   (R0), [V0.H8, V1.H8]
	VMOVQ  $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ  $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	VTBL   V28.B16, [V0.B16, V1.B16], V2.B16
	VTBL   V29.B16, [V0.B16, V1.B16], V3.B16
	VSSHR  $15, V2.H8, V2.H8
	VSSHR  $15, V3.H8, V3.H8
	VEOR   V2.B16, V5.B16, V5.B16
	VEOR   V3.B16, V6.B16, V6.B16
	VSUB   V2.H8, V5.H8, V5.H8
	VSUB   V3.H8, V6.H8, V6.H8
	VST1   [V5.H8, V6.H8], (R1)
	MOVD   m+8(FP), R22
	MOVWU  quantMatrix_q(R22), R22
	VDUP   R2, V13.H8
	VMOV   V13.B16, V12.B16
	VMOV   R22, V12.H[0]
	VMUL   V12.H8, V5.H8, V5.H8
	VMUL   V13.H8, V6.H8, V6.H8
	VMOVQ  $0x0d0c0b0a03020100, $0x19180f0e09080504, V30
	VMOVQ  $0x1b1a171611100706, $0x1f1e1d1c15141312, V31
	VTBL   V30.B16, [V5.B16, V6.B16], V0.B16
	VTBL   V31.B16, [V5.B16, V6.B16], V1.B16
	VST1   [V0.H8, V1.H8], (R0)
	CBZ    R16, tret
	MOVH   R24, (R0)
	MOVH   R25, (R1)

tret:
	CMP  $0, R13
	CSET GE, R3
	MOVB R3, ret+56(FP)
	RET

	// Levels above 67: row 67 plus fixedCost[l] - fixedCost[67].
tbig0:
	MOVD  $·fixedCost(SB), R14
	MOVD  (R14)(R22<<3), R26
	MOVD  536(R14), R14
	SUB   R14, R26, R14
	MOVHU 134(R10), R16
	MOVHU 134(R11), R26
	ADD   R14, R16, R16
	ADD   R14, R26, R26
	B     tcost0

	// R14 holds cs0 here; R23 is free.
tbig1:
	MOVD  $·fixedCost(SB), R23
	MOVD  (R23)(R22<<3), R26
	MOVD  536(R23), R23
	SUB   R23, R26, R23
	MOVHU 134(R10), R16
	MOVHU 134(R11), R26
	ADD   R23, R16, R16
	ADD   R23, R26, R26
	B     tcost1

// RCSUM4 adds to acc the costs at the 4 indices packed in r.
#define RCSUM4(r, acc) \
	UBFX  $0, r, $16, R24 \
	UBFX  $16, r, $16, R25 \
	UBFX  $32, r, $16, R26 \
	LSR   $48, r, r \
	MOVHU (R15)(R24<<1), R24 \
	MOVHU (R15)(R25<<1), R25 \
	MOVHU (R15)(R26<<1), R26 \
	MOVHU (R15)(r<<1), r \
	ADD   R24, acc, acc \
	ADD   R25, R26, R26 \
	ADD   r, acc, acc \
	ADD   R26, acc, acc

// func resCostAsm(s *costTabs, typ, ctx, first int, lv *[16]int16) (c, last int)
// first is 0 or 1, so bands[first] == first.
TEXT ·resCostAsm(SB), NOSPLIT, $0-56
	MOVD s+0(FP), R0
	MOVD typ+8(FP), R1
	MOVD ctx+16(FP), R2
	MOVD first+24(FP), R3
	MOVD lv+32(FP), R5

	// last = highest nonzero n >= first, else -1.
	VLD1    (R5), [V0.H8, V1.H8]
	VMOVQ   $0x0004000300020001, $0x0008000700060005, V30
	VMOVQ   $0x000c000b000a0009, $0x0010000f000e000d, V31
	VCMTST  V0.H8, V0.H8, V2.H8
	VCMTST  V1.H8, V1.H8, V3.H8
	VAND    V30.B16, V2.B16, V2.B16
	VAND    V31.B16, V3.B16, V3.B16
	VUMAX   V3.H8, V2.H8, V2.H8
	VUMAXV  V2.H8, V2
	VMOV    V2.H[0], R4
	SUB     $1, R4
	MOVD    $-1, R6
	CMP     R3, R4
	CSEL    LT, R6, R4, R4
	MOVD    R4, last+48(FP)

	// p0 = probs[typ*264+(first*3+ctx)*11]
	MOVD  $264, R6
	MUL   R6, R1, R6
	ADD   R3<<1, R3, R7
	ADD   R2, R7, R7
	MOVD  $11, R8
	MADD  R8, R6, R7, R7
	ADD   $costTabs_probs, R0, R9
	MOVBU (R9)(R7), R7
	MOVD  $·bitCost(SB), R10
	TBNZ  $63, R4, rcnone
	ADD   $2048, R10, R11
	MOVD  (R11)(R7<<3), R12
	CMP   $0, R2
	CSEL  EQ, R12, ZR, R12

	// Row index n*512 + ctx*128 + min(|v|, 67), where ctx is
	// min(|v|, 2) of the previous position.
	VABS  V0.H8, V0.H8
	VABS  V1.H8, V1.H8
	MOVD  $2, R13
	VDUP  R13, V2.H8
	VUMIN V2.H8, V0.H8, V3.H8
	VUMIN V2.H8, V1.H8, V4.H8
	VEXT  $14, V3.B16, V3.B16, V5.B16
	VEXT  $14, V4.B16, V3.B16, V6.B16
	CBNZ  R3, rcfirst1
	VMOV  R2, V5.H[0]
	B     rcidx

rcfirst1:
	VMOV R2, V5.H[1]

rcidx:
	MOVD  $67, R13
	VDUP  R13, V2.H8
	VUMIN V2.H8, V0.H8, V7.H8
	VUMIN V2.H8, V1.H8, V8.H8
	VSHL  $7, V5.H8, V5.H8
	VSHL  $7, V6.H8, V6.H8
	VADD  V7.H8, V5.H8, V5.H8
	VADD  V8.H8, V6.H8, V6.H8
	VMOVQ $0x0600040002000000, $0x0e000c000a000800, V9
	VMOVQ $0x1600140012001000, $0x1e001c001a001800, V10
	VADD  V9.H8, V5.H8, V5.H8
	VADD  V10.H8, V6.H8, V6.H8
	// Positions outside [first, last] read cost[0][0][127], never set.
	VMOVQ $0x0003000200010000, $0x0007000600050004, V11
	VMOVQ $0x000b000a00090008, $0x000f000e000d000c, V12
	VDUP  R3, V13.H8
	VDUP  R4, V14.H8
	MOVD  $127, R13
	VDUP  R13, V17.H8
	VCMHS V13.H8, V11.H8, V15.H8
	VCMHS V11.H8, V14.H8, V16.H8
	VAND  V16.B16, V15.B16, V15.B16
	VCMHS V13.H8, V12.H8, V18.H8
	VCMHS V12.H8, V14.H8, V16.H8
	VAND  V16.B16, V18.B16, V18.B16
	VBSL  V17.B16, V5.B16, V15.B16
	VBSL  V17.B16, V6.B16, V18.B16
	ADD   $costTabs_cost, R0, R15
	ADD   R1<<14, R15, R15
	VMOV  V15.D[0], R16
	VMOV  V15.D[1], R17
	VMOV  V18.D[0], R19
	VMOV  V18.D[1], R20
	MOVD  $0, R21
	MOVD  $0, R22
	MOVD  $0, R23
	RCSUM4(R16, R12)
	RCSUM4(R17, R21)
	RCSUM4(R19, R22)
	RCSUM4(R20, R23)
	ADD   R21, R12, R12
	ADD   R23, R22, R22
	ADD   R22, R12, R12

	VUMAX  V1.H8, V0.H8, V9.H8
	VUMAXV V9.H8, V9
	VMOV   V9.H[0], R16
	CMP    $67, R16
	BHI    rcbig

rctail:
	CMP   $15, R4
	BEQ   rcdone
	MOVH  (R5)(R4<<1), R16
	CMP   $0, R16
	CNEG  LT, R16, R16
	MOVD  $2, R17
	CMP   R17, R16
	CSEL  HI, R17, R16, R16
	MOVD  $·bands(SB), R17
	ADD   $1, R4, R8
	MOVBU (R17)(R8), R17
	ADD   R17<<1, R17, R17
	ADD   R16, R17, R17
	MOVD  $11, R8
	MADD  R8, R6, R17, R17
	MOVBU (R9)(R17), R17
	MOVD  (R10)(R17<<3), R17
	ADD   R17, R12, R12

rcdone:
	MOVD R12, c+40(FP)
	RET

rcnone:
	MOVD (R10)(R7<<3), R12
	MOVD R12, c+40(FP)
	RET

	// Levels above 67 add fixedCost[v] - fixedCost[67].
rcbig:
	MOVD first+24(FP), R3
	MOVD $·fixedCost(SB), R17
	MOVD 536(R17), R8

rcbloop:
	MOVH (R5)(R3<<1), R16
	CMP  $0, R16
	CNEG LT, R16, R16
	CMP  $67, R16
	BLS  rcbnext
	AND  $2047, R16
	MOVD (R17)(R16<<3), R16
	SUB  R8, R16, R16
	ADD  R16, R12, R12

rcbnext:
	ADD $1, R3
	CMP R4, R3
	BLE rcbloop
	B   rctail

// func resCostLastAsm(s *lwState, typ, ctx, first, last int, lv *[16]int16) (c int)
// resCostAsm with last from the caller: -1, or the highest nonzero
// n >= first. The TBNZ on the argument keeps -1 for resCostLast.
TEXT ·resCostLastAsm(SB), NOSPLIT, $0-56
	MOVD s+0(FP), R0
	MOVD typ+8(FP), R1
	MOVD ctx+16(FP), R2
	MOVD first+24(FP), R3
	MOVD last+32(FP), R4
	MOVD lv+40(FP), R5
	VLD1 (R5), [V0.H8, V1.H8]

	MOVD  $264, R6
	MUL   R6, R1, R6
	ADD   R3<<1, R3, R7
	ADD   R2, R7, R7
	MOVD  $11, R8
	MADD  R8, R6, R7, R7
	ADD   $costTabs_probs, R0, R9
	MOVBU (R9)(R7), R7
	MOVD  $·bitCost(SB), R10
	TBNZ  $63, R4, rlnone
	ADD   $2048, R10, R11
	MOVD  (R11)(R7<<3), R12
	CMP   $0, R2
	CSEL  EQ, R12, ZR, R12

	VABS  V0.H8, V0.H8
	VABS  V1.H8, V1.H8
	MOVD  $2, R13
	VDUP  R13, V2.H8
	VUMIN V2.H8, V0.H8, V3.H8
	VUMIN V2.H8, V1.H8, V4.H8
	VEXT  $14, V3.B16, V3.B16, V5.B16
	VEXT  $14, V4.B16, V3.B16, V6.B16
	CBNZ  R3, rlfirst1
	VMOV  R2, V5.H[0]
	B     rlidx

rlfirst1:
	VMOV R2, V5.H[1]

rlidx:
	MOVD  $67, R13
	VDUP  R13, V2.H8
	VUMIN V2.H8, V0.H8, V7.H8
	VUMIN V2.H8, V1.H8, V8.H8
	VSHL  $7, V5.H8, V5.H8
	VSHL  $7, V6.H8, V6.H8
	VADD  V7.H8, V5.H8, V5.H8
	VADD  V8.H8, V6.H8, V6.H8
	VMOVQ $0x0600040002000000, $0x0e000c000a000800, V9
	VMOVQ $0x1600140012001000, $0x1e001c001a001800, V10
	VADD  V9.H8, V5.H8, V5.H8
	VADD  V10.H8, V6.H8, V6.H8
	VMOVQ $0x0003000200010000, $0x0007000600050004, V11
	VMOVQ $0x000b000a00090008, $0x000f000e000d000c, V12
	VDUP  R3, V13.H8
	VDUP  R4, V14.H8
	MOVD  $127, R13
	VDUP  R13, V17.H8
	VCMHS V13.H8, V11.H8, V15.H8
	VCMHS V11.H8, V14.H8, V16.H8
	VAND  V16.B16, V15.B16, V15.B16
	VCMHS V13.H8, V12.H8, V18.H8
	VCMHS V12.H8, V14.H8, V16.H8
	VAND  V16.B16, V18.B16, V18.B16
	VBSL  V17.B16, V5.B16, V15.B16
	VBSL  V17.B16, V6.B16, V18.B16
	ADD   $costTabs_cost, R0, R15
	ADD   R1<<14, R15, R15
	VMOV  V15.D[0], R16
	VMOV  V15.D[1], R17
	VMOV  V18.D[0], R19
	VMOV  V18.D[1], R20
	MOVD  $0, R21
	MOVD  $0, R22
	MOVD  $0, R23
	RCSUM4(R16, R12)
	RCSUM4(R17, R21)
	RCSUM4(R19, R22)
	RCSUM4(R20, R23)
	ADD   R21, R12, R12
	ADD   R23, R22, R22
	ADD   R22, R12, R12

	VUMAX  V1.H8, V0.H8, V9.H8
	VUMAXV V9.H8, V9
	VMOV   V9.H[0], R16
	CMP    $67, R16
	BHI    rlbig

rltail:
	CMP   $15, R4
	BEQ   rldone
	MOVH  (R5)(R4<<1), R16
	CMP   $0, R16
	CNEG  LT, R16, R16
	MOVD  $2, R17
	CMP   R17, R16
	CSEL  HI, R17, R16, R16
	MOVD  $·bands(SB), R17
	ADD   $1, R4, R8
	MOVBU (R17)(R8), R17
	ADD   R17<<1, R17, R17
	ADD   R16, R17, R17
	MOVD  $11, R8
	MADD  R8, R6, R17, R17
	MOVBU (R9)(R17), R17
	MOVD  (R10)(R17<<3), R17
	ADD   R17, R12, R12

rldone:
	MOVD R12, c+48(FP)
	RET

rlnone:
	MOVD (R10)(R7<<3), R12
	MOVD R12, c+48(FP)
	RET

rlbig:
	MOVD first+24(FP), R3
	MOVD $·fixedCost(SB), R17
	MOVD 536(R17), R8

rlbloop:
	MOVH (R5)(R3<<1), R16
	CMP  $0, R16
	CNEG LT, R16, R16
	CMP  $67, R16
	BLS  rlbnext
	AND  $2047, R16
	MOVD (R17)(R16<<3), R16
	SUB  R8, R16, R16
	ADD  R16, R12, R12

rlbnext:
	ADD $1, R3
	CMP R4, R3
	BLE rlbloop
	B   rltail

// func quantNAsm(m *quantMatrix, in, out *[16]int16, n int) (nz int)
// quantAsm over n blocks; bit i of nz is set when block i has a level.
TEXT ·quantNAsm(SB), NOSPLIT, $0-40
	MOVD m+0(FP), R0
	MOVD in+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R8
	QUANTK
	VMOVQ $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	MOVD  $0, R9
	MOVD  $1, R10

	// QUANTB without SSE and last.
qnblk:
	VLD1 (R1), [V0.H8, V1.H8]
	VABS    V0.H8, V2.H8
	VABS    V1.H8, V3.H8
	VADD    V16.H8, V2.H8, V2.H8
	VADD    V17.H8, V3.H8, V3.H8
	VUMULL  V23.H4, V2.H4, V4.S4
	VUMULL2 V23.H8, V2.H8, V5.S4
	VUMULL  V22.H4, V3.H4, V6.S4
	VUMULL2 V22.H8, V3.H8, V7.S4
	WORD $0x0e794088 // addhn v8.4h, v4.4s, v25.4s
	WORD $0x4e7840a8 // addhn2 v8.8h, v5.4s, v24.4s
	WORD $0x0e7840c9 // addhn v9.4h, v6.4s, v24.4s
	WORD $0x4e7840e9 // addhn2 v9.8h, v7.4s, v24.4s
	VUSHR   $1, V8.H8, V8.H8
	VUSHR   $1, V9.H8, V9.H8
	// cmlt, not VSSHR: no shift pipe.
	WORD $0x4e60a80a // cmlt v10.8h, v0.8h, #0
	WORD $0x4e60a82b // cmlt v11.8h, v1.8h, #0
	VEOR  V10.B16, V8.B16, V8.B16
	VEOR  V11.B16, V9.B16, V9.B16
	VSUB  V10.H8, V8.H8, V8.H8
	VSUB  V11.H8, V9.H8, V9.H8
	VMUL V21.H8, V8.H8, V12.H8
	VMUL V20.H8, V9.H8, V13.H8
	VST1.P [V12.H8, V13.H8], 32(R1)
	VTBL V28.B16, [V8.B16, V9.B16], V10.B16
	VTBL V29.B16, [V8.B16, V9.B16], V11.B16
	VST1.P [V10.H8, V11.H8], 32(R2)
	VORR V9.B16, V8.B16, V14.B16
	VMOV V14.D[0], R6
	VMOV V14.D[1], R7
	ORR  R7, R6, R6
	// Branchless: zero blocks mispredict.
	CMP  $0, R6
	CSEL NE, R10, ZR, R6
	ORR  R6, R9, R9
	LSL  $1, R10
	SUBS $1, R8
	BNE  qnblk
	MOVD R9, nz+32(FP)
	RET

// func quantI16ACAsm(m *quantMatrix, dq, lv *[16][16]int16, dc *[16]int16) (nz int)
// quantNAsm over 16 blocks with lane 0 zeroed in the register and
// dc[n] stored there in the dequantized block, so no narrow store
// meets a wide load.
TEXT ·quantI16ACAsm(SB), NOSPLIT, $0-40
	MOVD m+0(FP), R0
	MOVD dq+8(FP), R1
	MOVD lv+16(FP), R2
	MOVD dc+24(FP), R11
	MOVD $16, R8
	QUANTK
	VMOVQ $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	MOVD  $0, R9
	MOVD  $1, R10

	// QUANTB without SSE and last.
qiblk:
	VLD1 (R1), [V0.H8, V1.H8]
	VMOV    ZR, V0.H[0]
	VABS    V0.H8, V2.H8
	VABS    V1.H8, V3.H8
	VADD    V16.H8, V2.H8, V2.H8
	VADD    V17.H8, V3.H8, V3.H8
	VUMULL  V23.H4, V2.H4, V4.S4
	VUMULL2 V23.H8, V2.H8, V5.S4
	VUMULL  V22.H4, V3.H4, V6.S4
	VUMULL2 V22.H8, V3.H8, V7.S4
	WORD $0x0e794088 // addhn v8.4h, v4.4s, v25.4s
	WORD $0x4e7840a8 // addhn2 v8.8h, v5.4s, v24.4s
	WORD $0x0e7840c9 // addhn v9.4h, v6.4s, v24.4s
	WORD $0x4e7840e9 // addhn2 v9.8h, v7.4s, v24.4s
	VUSHR   $1, V8.H8, V8.H8
	VUSHR   $1, V9.H8, V9.H8
	// cmlt, not VSSHR: no shift pipe.
	WORD $0x4e60a80a // cmlt v10.8h, v0.8h, #0
	WORD $0x4e60a82b // cmlt v11.8h, v1.8h, #0
	VEOR  V10.B16, V8.B16, V8.B16
	VEOR  V11.B16, V9.B16, V9.B16
	VSUB  V10.H8, V8.H8, V8.H8
	VSUB  V11.H8, V9.H8, V9.H8
	VMUL V21.H8, V8.H8, V12.H8
	VMUL V20.H8, V9.H8, V13.H8
	MOVHU.P 2(R11), R6
	VMOV R6, V12.H[0]
	VST1.P [V12.H8, V13.H8], 32(R1)
	VTBL V28.B16, [V8.B16, V9.B16], V10.B16
	VTBL V29.B16, [V8.B16, V9.B16], V11.B16
	VST1.P [V10.H8, V11.H8], 32(R2)
	VORR V9.B16, V8.B16, V14.B16
	VMOV V14.D[0], R6
	VMOV V14.D[1], R7
	ORR  R7, R6, R6
	// Branchless: zero blocks mispredict.
	CMP  $0, R6
	CSEL NE, R10, ZR, R6
	ORR  R6, R9, R9
	LSL  $1, R10
	SUBS $1, R8
	BNE  qiblk
	MOVD R9, nz+32(FP)
	RET

// func quantNDCAsm(m *quantMatrix, in, out *[16]int16, n int, dc *[8]int16) (nz int)
// quantNAsm with lane 0 of block i taken from dc[i] after the load.
TEXT ·quantNDCAsm(SB), NOSPLIT, $0-48
	MOVD m+0(FP), R0
	MOVD in+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R8
	MOVD dc+32(FP), R11
	QUANTK
	VMOVQ $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	MOVD  $0, R9
	MOVD  $1, R10

	// QUANTB without SSE and last.
qdblk:
	VLD1 (R1), [V0.H8, V1.H8]
	MOVHU.P 2(R11), R6
	VMOV    R6, V0.H[0]
	VABS    V0.H8, V2.H8
	VABS    V1.H8, V3.H8
	VADD    V16.H8, V2.H8, V2.H8
	VADD    V17.H8, V3.H8, V3.H8
	VUMULL  V23.H4, V2.H4, V4.S4
	VUMULL2 V23.H8, V2.H8, V5.S4
	VUMULL  V22.H4, V3.H4, V6.S4
	VUMULL2 V22.H8, V3.H8, V7.S4
	WORD $0x0e794088 // addhn v8.4h, v4.4s, v25.4s
	WORD $0x4e7840a8 // addhn2 v8.8h, v5.4s, v24.4s
	WORD $0x0e7840c9 // addhn v9.4h, v6.4s, v24.4s
	WORD $0x4e7840e9 // addhn2 v9.8h, v7.4s, v24.4s
	VUSHR   $1, V8.H8, V8.H8
	VUSHR   $1, V9.H8, V9.H8
	// cmlt, not VSSHR: no shift pipe.
	WORD $0x4e60a80a // cmlt v10.8h, v0.8h, #0
	WORD $0x4e60a82b // cmlt v11.8h, v1.8h, #0
	VEOR  V10.B16, V8.B16, V8.B16
	VEOR  V11.B16, V9.B16, V9.B16
	VSUB  V10.H8, V8.H8, V8.H8
	VSUB  V11.H8, V9.H8, V9.H8
	VMUL V21.H8, V8.H8, V12.H8
	VMUL V20.H8, V9.H8, V13.H8
	VST1.P [V12.H8, V13.H8], 32(R1)
	VTBL V28.B16, [V8.B16, V9.B16], V10.B16
	VTBL V29.B16, [V8.B16, V9.B16], V11.B16
	VST1.P [V10.H8, V11.H8], 32(R2)
	VORR V9.B16, V8.B16, V14.B16
	VMOV V14.D[0], R6
	VMOV V14.D[1], R7
	ORR  R7, R6, R6
	// Branchless: zero blocks mispredict.
	CMP  $0, R6
	CSEL NE, R10, ZR, R6
	ORR  R6, R9, R9
	LSL  $1, R10
	SUBS $1, R8
	BNE  qdblk
	MOVD R9, nz+40(FP)
	RET

// func quantY16Asm(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) (d int)
// quantAsm over 16 blocks with lane 0 moved to dc and zeroed; d sums
// quantAsm's d per block, so ADDV stays per block.
TEXT ·quantY16Asm(SB), NOSPLIT, $0-40
	MOVD m+0(FP), R0
	MOVD dq+8(FP), R1
	MOVD lv+16(FP), R2
	MOVD dc+24(FP), R11
	QUANTK
	VMOVQ $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	MOVD  $16, R8
	MOVD  $0, R9
	MOVD  $0, R10

	// QUANTB without last.
qyblk:
	VLD1    (R1), [V0.H8, V1.H8]
	VMOV    V0.H[0], R6
	MOVH.P  R6, 2(R11)
	VMOV    R10, V0.H[0]
	VABS    V0.H8, V2.H8
	VABS    V1.H8, V3.H8
	VADD    V16.H8, V2.H8, V2.H8
	VADD    V17.H8, V3.H8, V3.H8
	VUMULL  V23.H4, V2.H4, V4.S4
	VUMULL2 V23.H8, V2.H8, V5.S4
	VUMULL  V22.H4, V3.H4, V6.S4
	VUMULL2 V22.H8, V3.H8, V7.S4
	VADD    V25.S4, V4.S4, V4.S4
	VADD    V24.S4, V5.S4, V5.S4
	VADD    V24.S4, V6.S4, V6.S4
	VADD    V24.S4, V7.S4, V7.S4
	VSHRN   $16, V4.S4, V8.H4
	VSHRN2  $16, V5.S4, V8.H8
	VSHRN   $16, V6.S4, V9.H4
	VSHRN2  $16, V7.S4, V9.H8
	VUSHR   $1, V8.H8, V8.H8
	VUSHR   $1, V9.H8, V9.H8
	VSSHR   $15, V0.H8, V10.H8
	VSSHR   $15, V1.H8, V11.H8
	VEOR    V10.B16, V8.B16, V8.B16
	VEOR    V11.B16, V9.B16, V9.B16
	VSUB    V10.H8, V8.H8, V8.H8
	VSUB    V11.H8, V9.H8, V9.H8
	VMUL    V21.H8, V8.H8, V12.H8
	VMUL    V20.H8, V9.H8, V13.H8
	VST1.P  [V12.H8, V13.H8], 32(R1)
	VSUB    V12.H8, V0.H8, V14.H8
	VSUB    V13.H8, V1.H8, V15.H8
	VSMULL  V14.H4, V14.H4, V26.S4
	VSMULL2 V14.H8, V14.H8, V27.S4
	VSMLAL  V15.H4, V15.H4, V26.S4
	VSMLAL2 V15.H8, V15.H8, V27.S4
	VADD    V27.S4, V26.S4, V26.S4
	VADDV   V26.S4, V26
	VMOV    V26.S[0], R6
	ADD     R6, R9
	VTBL    V28.B16, [V8.B16, V9.B16], V10.B16
	VTBL    V29.B16, [V8.B16, V9.B16], V11.B16
	VST1.P  [V10.H8, V11.H8], 32(R2)
	SUBS    $1, R8
	BNE     qyblk
	MOVD    R9, d+32(FP)
	RET

// TR4 transposes the 4x4 words in each 64-bit half of x0-x3.
#define TR4(x0, x1, x2, x3, t0, t1, t2, t3) \
	VTRN1 x1.H8, x0.H8, t0.H8 \
	VTRN2 x1.H8, x0.H8, t1.H8 \
	VTRN1 x3.H8, x2.H8, t2.H8 \
	VTRN2 x3.H8, x2.H8, t3.H8 \
	VTRN1 t2.S4, t0.S4, x0.S4 \
	VTRN1 t3.S4, t1.S4, x1.S4 \
	VTRN2 t2.S4, t0.S4, x2.S4 \
	VTRN2 t3.S4, t1.S4, x3.S4

// ROWP is the fdct row pass on columns 0-3 in x0-x3.
#define ROWP(x0, x1, x2, x3, t0, t1, t2, t3, p0, p1, p2, p3) \
	VADD    x3.H8, x0.H8, t0.H8 \
	VADD    x2.H8, x1.H8, t1.H8 \
	VSUB    x2.H8, x1.H8, t2.H8 \
	VSUB    x3.H8, x0.H8, t3.H8 \
	VADD    t1.H8, t0.H8, x0.H8 \
	VSUB    t1.H8, t0.H8, x2.H8 \
	VSHL    $3, x0.H8, x0.H8 \
	VSHL    $3, x2.H8, x2.H8 \
	VSMULL  V28.H4, t2.H4, p0.S4 \
	VSMULL2 V28.H8, t2.H8, p1.S4 \
	VSMLAL  V29.H4, t3.H4, p0.S4 \
	VSMLAL2 V29.H8, t3.H8, p1.S4 \
	VSMULL  V28.H4, t3.H4, p2.S4 \
	VSMULL2 V28.H8, t3.H8, p3.S4 \
	VSMLSL  V29.H4, t2.H4, p2.S4 \
	VSMLSL2 V29.H8, t2.H8, p3.S4 \
	VADD    V30.S4, p0.S4, p0.S4 \
	VADD    V30.S4, p1.S4, p1.S4 \
	VADD    V31.S4, p2.S4, p2.S4 \
	VADD    V31.S4, p3.S4, p3.S4 \
	VSHRN   $9, p0.S4, x1.H4 \
	VSHRN2  $9, p1.S4, x1.H8 \
	VSHRN   $9, p2.S4, x3.H4 \
	VSHRN2  $9, p3.S4, x3.H8

// COLP is the fdct column pass on rows 0-3 in x0-x3.
#define COLP(x0, x1, x2, x3, t0, t1, t2, t3, p0, p1, p2, p3) \
	VADD    x3.H8, x0.H8, t0.H8 \
	VADD    x2.H8, x1.H8, t1.H8 \
	VSUB    x2.H8, x1.H8, t2.H8 \
	VSUB    x3.H8, x0.H8, t3.H8 \
	VADD    V24.H8, t0.H8, t0.H8 \
	VADD    t1.H8, t0.H8, x0.H8 \
	VSUB    t1.H8, t0.H8, x2.H8 \
	VSSHR   $4, x0.H8, x0.H8 \
	VSSHR   $4, x2.H8, x2.H8 \
	VSMULL  V28.H4, t2.H4, p0.S4 \
	VSMULL2 V28.H8, t2.H8, p1.S4 \
	VSMLAL  V29.H4, t3.H4, p0.S4 \
	VSMLAL2 V29.H8, t3.H8, p1.S4 \
	VSMULL  V28.H4, t3.H4, p2.S4 \
	VSMULL2 V28.H8, t3.H8, p3.S4 \
	VSMLSL  V29.H4, t2.H4, p2.S4 \
	VSMLSL2 V29.H8, t2.H8, p3.S4 \
	VADD    V25.S4, p0.S4, p0.S4 \
	VADD    V25.S4, p1.S4, p1.S4 \
	VADD    V26.S4, p2.S4, p2.S4 \
	VADD    V26.S4, p3.S4, p3.S4 \
	VSHRN   $16, p0.S4, x1.H4 \
	VSHRN2  $16, p1.S4, x1.H8 \
	VSHRN   $16, p2.S4, x3.H4 \
	VSHRN2  $16, p3.S4, x3.H8 \
	VCMTST  t3.H8, t3.H8, t3.H8 \
	VSUB    t3.H8, x1.H8, x1.H8

// Residual fdct of four blocks: two halves of two blocks, rows 0-3
// in V12-V15 and V16-V19, the left block in the low 4 lanes. Each
// half ends as raster rows 0-3 of its two blocks, for st4 .2d.
// The halves interleave; FDCT34K constants.
#define FDCT4 \
	TR4(V12, V13, V14, V15, V0, V1, V2, V3) \
	TR4(V16, V17, V18, V19, V8, V9, V10, V11) \
	ROWP(V12, V13, V14, V15, V0, V1, V2, V3, V4, V5, V6, V7) \
	ROWP(V16, V17, V18, V19, V8, V9, V10, V11, V20, V21, V22, V23) \
	TR4(V12, V13, V14, V15, V0, V1, V2, V3) \
	TR4(V16, V17, V18, V19, V8, V9, V10, V11) \
	COLP(V12, V13, V14, V15, V0, V1, V2, V3, V4, V5, V6, V7) \
	COLP(V16, V17, V18, V19, V8, V9, V10, V11, V20, V21, V22, V23)

// func residualsYAsm(pred, src *uint8, stride int, out *[16][16]int16)
// residualAsm over the 16 luma blocks, one block row per pass.
TEXT ·residualsYAsm(SB), NOSPLIT, $0-32
	MOVD pred+0(FP), R11
	MOVD src+8(FP), R12
	MOVD stride+16(FP), R13
	MOVD out+24(FP), R2
	MOVD $4, R4
	FDCT34K

rsyrow:
	VLD1   (R12), [V0.B16]
	ADD    R13, R12
	VLD1   (R12), [V1.B16]
	ADD    R13, R12
	VLD1   (R12), [V2.B16]
	ADD    R13, R12
	VLD1   (R12), [V3.B16]
	ADD    R13, R12
	VLD1.P 64(R11), [V4.B16, V5.B16, V6.B16, V7.B16]
	VLD1.P 64(R11), [V8.B16, V9.B16, V10.B16, V11.B16]
	WORD $0x2e24200c // usubl v12.8h, v0.8b, v4.8b
	WORD $0x2e26202d // usubl v13.8h, v1.8b, v6.8b
	WORD $0x2e28204e // usubl v14.8h, v2.8b, v8.8b
	WORD $0x2e2a206f // usubl v15.8h, v3.8b, v10.8b
	WORD $0x6e242010 // usubl2 v16.8h, v0.16b, v4.16b
	WORD $0x6e262031 // usubl2 v17.8h, v1.16b, v6.16b
	WORD $0x6e282052 // usubl2 v18.8h, v2.16b, v8.16b
	WORD $0x6e2a2073 // usubl2 v19.8h, v3.16b, v10.16b
	FDCT4
	WORD $0x4c9f0c4c // st4 {v12.2d, v13.2d, v14.2d, v15.2d}, [x2], #64
	WORD $0x4c9f0c50 // st4 {v16.2d, v17.2d, v18.2d, v19.2d}, [x2], #64
	SUBS $1, R4
	BNE  rsyrow
	RET

// func residualsUVAsm(pred, u, v *uint8, stride int, out *[8][16]int16)
// residualAsm over the 8 chroma blocks, U at pred, V at pred+16.
// 8-byte pred loads keep the last row inside ybrBuf.
TEXT ·residualsUVAsm(SB), NOSPLIT, $0-40
	MOVD pred+0(FP), R11
	MOVD u+8(FP), R12
	MOVD v+16(FP), R14
	MOVD stride+24(FP), R13
	MOVD out+32(FP), R2
	ADD  $128, R2, R3
	MOVD $2, R4
	FDCT34K

rsuvrow:
	VLD1   (R12), [V0.B8]
	ADD    R13, R12
	VLD1   (R12), [V1.B8]
	ADD    R13, R12
	VLD1   (R12), [V2.B8]
	ADD    R13, R12
	VLD1   (R12), [V3.B8]
	ADD    R13, R12
	VLD1   (R14), [V20.B8]
	ADD    R13, R14
	VLD1   (R14), [V21.B8]
	ADD    R13, R14
	VLD1   (R14), [V22.B8]
	ADD    R13, R14
	VLD1   (R14), [V23.B8]
	ADD    R13, R14
	FMOVD  (R11), F4
	FMOVD  16(R11), F5
	FMOVD  32(R11), F6
	FMOVD  48(R11), F7
	FMOVD  64(R11), F8
	FMOVD  80(R11), F9
	FMOVD  96(R11), F10
	FMOVD  112(R11), F11
	ADD    $128, R11
	WORD $0x2e24200c // usubl v12.8h, v0.8b, v4.8b
	WORD $0x2e26202d // usubl v13.8h, v1.8b, v6.8b
	WORD $0x2e28204e // usubl v14.8h, v2.8b, v8.8b
	WORD $0x2e2a206f // usubl v15.8h, v3.8b, v10.8b
	WORD $0x2e252290 // usubl v16.8h, v20.8b, v5.8b
	WORD $0x2e2722b1 // usubl v17.8h, v21.8b, v7.8b
	WORD $0x2e2922d2 // usubl v18.8h, v22.8b, v9.8b
	WORD $0x2e2b22f3 // usubl v19.8h, v23.8b, v11.8b
	FDCT4
	WORD $0x4c9f0c4c // st4 {v12.2d, v13.2d, v14.2d, v15.2d}, [x2], #64
	WORD $0x4c9f0c70 // st4 {v16.2d, v17.2d, v18.2d, v19.2d}, [x3], #64
	SUBS $1, R4
	BNE  rsuvrow
	RET

// func idctAddNAsm(c *[16]int16, dst *uint8, rows, cols int)
// idctAddAsm over a rows x cols grid of blocks, skipping zero blocks.
TEXT ·idctAddNAsm(SB), NOSPLIT, $0-32
	MOVD c+0(FP), R0
	MOVD dst+8(FP), R19
	MOVD rows+16(FP), R21

inrow:
	MOVD cols+24(FP), R22
	MOVD R19, R1

inblk:
	LDP (R0), (R6, R7)
	LDP 16(R0), (R8, R9)
	ORR R7, R6, R6
	ORR R9, R8, R8
	ORR R8, R6, R6
	CBZ R6, inskip
	IDCTADD

inskip:
	ADD  $32, R0
	ADD  $4, R1
	SUBS $1, R22
	BNE  inblk
	ADD  $128, R19
	SUBS $1, R21
	BNE  inrow
	RET

// func costGridAsm(s *costTabs, typ, first int, lv *[16]int16, rows, cols, lbits, tbits int) (c int)
// resCostAsm's body over a rows x cols block grid, rows*cols <= 16.
// Block (y, x) has ctx lbits bit y plus tbits bit x; then both take
// its nz. Pass 1 prices every block's p0 bit with no branch on nz and
// lists the nonempty blocks; pass 2 sums their level costs.
// Locals: 8 list of {lv, last | ctx<<8}.
TEXT ·costGridAsm(SB), NOSPLIT, $256-72
	MOVD s+0(FP), R0
	MOVD typ+8(FP), R1
	MOVD first+16(FP), R3
	MOVD lv+24(FP), R5
	MOVD rows+32(FP), R19
	MOVD cols+40(FP), R20
	MOVD lbits+48(FP), R21
	MOVD tbits+56(FP), R12

	// R9 = probs[typ*264:], R7 = R9[first*33:], R15 = cost[typ],
	// R14 = nibble mask without n < first.
	ADD  $costTabs_probs, R0, R9
	MOVD $264, R6
	MADD R6, R9, R1, R9
	MOVD $33, R6
	MADD R6, R9, R3, R7
	MOVD $·bitCost(SB), R10
	ADD  $costTabs_cost, R0, R15
	ADD  R1<<14, R15, R15
	MOVD $-1, R14
	LSL  $2, R3, R6
	LSL  R6, R14, R14
	MOVD $15, R24

	// R11 sum, R16 x, R17 y, R13 left bit, R12 top bits, R8 list end.
	MOVD ZR, R11
	MOVD ZR, R16
	MOVD ZR, R17
	AND  $1, R21, R13
	ADD  $8, RSP, R8

cg1blk:
	// last = highest nonzero n >= first, else -1, from a nibble per n.
	VLD1   (R5), [V0.H8, V1.H8]
	VCMTST V0.H8, V0.H8, V0.H8
	VCMTST V1.H8, V1.H8, V1.H8
	VUZP1  V1.B16, V0.B16, V0.B16
	VSHRN  $4, V0.H8, V0.B8
	VMOV   V0.D[0], R4
	ANDS   R14, R4, R4
	CSET   NE, R22
	CLZ    R4, R4
	SUB    R4>>2, R24, R4

	// p0 = probs[typ*264+(first*3+ctx)*11]. Empty adds bitCost[0][p0],
	// else bitCost[1][p0] when ctx is 0.
	LSR   R16, R12, R2
	AND   $1, R2, R2
	ADD   R13, R2, R2
	ADD   R2<<1, R2, R6
	ADD   R2<<3, R6, R6
	MOVBU (R7)(R6), R6
	ADD   R6<<3, R10, R6
	MOVD  (R6), R23
	MOVD  2048(R6), R6
	CMP   $0, R2
	CSEL  EQ, R6, ZR, R6
	CMP   $0, R22
	CSEL  EQ, R23, R6, R6
	ADD   R6, R11, R11

	// Every block is stored; only a nonempty one advances R8.
	ORR R2<<8, R4, R6
	STP (R5, R6), (R8)
	ADD R22<<4, R8, R8

	// The left bit and top bit x take nz.
	MOVD R22, R13
	MOVD $1, R6
	LSL  R16, R6, R6
	BIC  R6, R12, R12
	LSL  R16, R22, R6
	ORR  R6, R12, R12

	ADD  $32, R5
	ADD  $1, R16
	CMP  R20, R16
	BLT  cg1blk
	MOVD ZR, R16
	ADD  $1, R17
	CMP  R19, R17
	BGE  cg2
	LSR  R17, R21, R13
	AND  $1, R13, R13
	B    cg1blk

cg2:
	// Lanes: n*512, n-first, 2 but 0 below first, 67, 127, n == first.
	VMOVQ $0x0600040002000000, $0x0e000c000a000800, V9
	VMOVQ $0x1600140012001000, $0x1e001c001a001800, V10
	VMOVQ $0x0003000200010000, $0x0007000600050004, V11
	VMOVQ $0x000b000a00090008, $0x000f000e000d000c, V12
	MOVD  $2, R7
	VDUP  R7, V20.H8
	VDUP  R3, V14.H8
	VCMHS V14.H8, V11.H8, V13.H8
	VAND  V20.B16, V13.B16, V13.B16
	VSUB  V14.H8, V11.H8, V11.H8
	VSUB  V14.H8, V12.H8, V12.H8
	VEOR  V28.B16, V28.B16, V28.B16
	VCMHS V11.H8, V28.H8, V23.H8
	MOVD  $67, R6
	VDUP  R6, V21.H8
	MOVD  $127, R6
	VDUP  R6, V22.H8

	// R0 = bands[1:], R1 = fixedCost, R14 = fixedCost[67], R13 list
	// cursor; R11 and R21-R23 accumulate.
	MOVD $·bands(SB), R0
	ADD  $1, R0
	MOVD $·fixedCost(SB), R1
	MOVD 536(R1), R14
	MOVD $11, R12
	MOVD ZR, R21
	MOVD ZR, R22
	MOVD ZR, R23
	ADD  $8, RSP, R13
	CMP  R8, R13
	BEQ  cgdone

cg2blk:
	LDP  (R13), (R5, R6)
	ADD  $16, R13
	VLD1 (R5), [V0.H8, V1.H8]
	AND  $255, R6, R4
	LSR  $8, R6, R2

	// Lane first adds ctx*128; n is in [first, last] when
	// n-first <= last-first.
	LSL   $7, R2, R2
	VDUP  R2, V24.H8
	VAND  V23.B16, V24.B16, V24.B16
	VADD  V9.H8, V24.H8, V24.H8
	SUB   R3, R4, R6
	VDUP  R6, V25.H8
	VCMHS V11.H8, V25.H8, V26.H8
	VCMHS V12.H8, V25.H8, V27.H8

	// Row index n*512 + ctx*128 + min(|v|, 67), where ctx is
	// min(|v|, 2) of the previous position.
	VABS  V0.H8, V0.H8
	VABS  V1.H8, V1.H8
	VUMIN V13.H8, V0.H8, V3.H8
	VUMIN V20.H8, V1.H8, V4.H8
	VEXT  $14, V3.B16, V28.B16, V5.B16
	VEXT  $14, V4.B16, V3.B16, V6.B16
	VUMIN V21.H8, V0.H8, V7.H8
	VUMIN V21.H8, V1.H8, V8.H8
	VADD  V24.H8, V7.H8, V7.H8
	VADD  V10.H8, V8.H8, V8.H8
	VSHL  $7, V5.H8, V5.H8
	VSHL  $7, V6.H8, V6.H8
	VADD  V7.H8, V5.H8, V5.H8
	VADD  V8.H8, V6.H8, V6.H8
	// Positions outside [first, last] read cost[0][0][127], never set.
	VBSL  V22.B16, V5.B16, V26.B16
	VBSL  V22.B16, V6.B16, V27.B16
	VMOV  V26.D[0], R16
	VMOV  V26.D[1], R17
	VMOV  V27.D[0], R19
	VMOV  V27.D[1], R20
	RCSUM4(R16, R11)
	RCSUM4(R17, R21)
	RCSUM4(R19, R22)
	RCSUM4(R20, R23)

	// End bit unless last is 15; bands[16] keeps that read in range.
	MOVH  (R5)(R4<<1), R6
	CMP   $0, R6
	CNEG  LT, R6, R6
	CMP   R7, R6
	CSEL  HI, R7, R6, R6
	MOVBU (R0)(R4), R2
	ADD   R2<<1, R2, R2
	ADD   R6, R2, R2
	MUL   R12, R2, R2
	MOVBU (R9)(R2), R2
	MOVD  (R10)(R2<<3), R2
	CMP   $15, R4
	CSEL  EQ, ZR, R2, R2
	ADD   R2, R22, R22

	VUMAX  V1.H8, V0.H8, V29.H8
	VUMAXV V29.H8, V29
	VMOV   V29.H[0], R6
	CMP    $67, R6
	BHI    cgbig

cg2next:
	CMP R8, R13
	BNE cg2blk

cgdone:
	ADD  R21, R11, R11
	ADD  R23, R22, R22
	ADD  R22, R11, R11
	MOVD R11, c+64(FP)
	RET

	// Levels above 67 add fixedCost[v] - fixedCost[67].
cgbig:
	MOVD R3, R2

cgbloop:
	MOVH (R5)(R2<<1), R6
	CMP  $0, R6
	CNEG LT, R6, R6
	CMP  $67, R6
	BLS  cgbnext
	AND  $2047, R6
	MOVD (R1)(R6<<3), R6
	SUB  R14, R6, R6
	ADD  R6, R11, R11

cgbnext:
	ADD $1, R2
	CMP R4, R2
	BLE cgbloop
	B   cg2next

// DCCOL: one DC column into lane; quantI16ACAsm zeroes the DCs.
#define DCCOL(off, lane) \
	MOVHU off(R8), R3 \
	MOVHU off+128(R8), R4 \
	MOVHU off+256(R8), R5 \
	MOVHU off+384(R8), R6 \
	ORR   R4<<16, R3, R3 \
	ORR   R5<<32, R3, R3 \
	ORR   R6<<48, R3, R3 \
	VMOV  R3, lane

// WHT4: p0+p2, p1+p3 butterfly into V12-V15.
#define WHT4(p0, p1, p2, p3) \
	VADD p2.S4, p0.S4, V6.S4 \
	VSUB p2.S4, p0.S4, V9.S4 \
	VADD p3.S4, p1.S4, V7.S4 \
	VSUB p3.S4, p1.S4, V8.S4 \
	VADD V7.S4, V6.S4, V12.S4 \
	VADD V8.S4, V9.S4, V13.S4 \
	VSUB V8.S4, V9.S4, V14.S4 \
	VSUB V7.S4, V6.S4, V15.S4

// TRANS4: rows a-d, transposed, into V2-V5.
#define TRANS4(a, b, c, d) \
	VTRN1 b.S4, a.S4, V16.S4 \
	VTRN2 b.S4, a.S4, V17.S4 \
	VTRN1 d.S4, c.S4, V18.S4 \
	VTRN2 d.S4, c.S4, V19.S4 \
	VTRN1 V18.D2, V16.D2, V2.D2 \
	VTRN1 V19.D2, V17.D2, V3.D2 \
	VTRN2 V18.D2, V16.D2, V4.D2 \
	VTRN2 V19.D2, V17.D2, V5.D2

// IWHT is iwht of V0, V1 in place, raster order; uses R3.
// V14, V15 hold rows 3, 2.
#define IWHT \
	VSXTL  V0.H4, V2.S4 \
	VSXTL2 V0.H8, V3.S4 \
	VSXTL  V1.H4, V4.S4 \
	VSXTL2 V1.H8, V5.S4 \
	WHT4(V2, V3, V5, V4) \
	TRANS4(V12, V13, V15, V14) \
	MOVW   $3, R3 \
	VDUP   R3, V29.S4 \
	VADD   V29.S4, V2.S4, V2.S4 \
	WHT4(V2, V3, V5, V4) \
	TRANS4(V12, V13, V15, V14) \
	VSHRN  $3, V2.S4, V0.H4 \
	VSHRN2 $3, V3.S4, V0.H8 \
	VSHRN  $3, V4.S4, V1.H4 \
	VSHRN2 $3, V5.S4, V1.H8

// func i16DCAsm(m *quantMatrix, dq *[16][16]int16, lv, dc *[16]int16) (nz bool)
TEXT ·i16DCAsm(SB), NOSPLIT, $0-33
	MOVD m+0(FP), R0
	MOVD dq+8(FP), R8
	MOVD lv+16(FP), R2
	MOVD dc+24(FP), R1

	// V0 = columns 0|1, V1 = 2|3.
	DCCOL(0, V0.D[0])
	DCCOL(32, V0.D[1])
	DCCOL(64, V1.D[0])
	DCCOL(96, V1.D[1])

	// fwhtPlain; SHRN truncates like int16().
	VSXTL  V0.H4, V2.S4
	VSXTL2 V0.H8, V3.S4
	VSXTL  V1.H4, V4.S4
	VSXTL2 V1.H8, V5.S4
	WHT4(V2, V3, V4, V5)
	TRANS4(V12, V13, V14, V15)
	WHT4(V2, V3, V4, V5)
	VSHRN  $1, V12.S4, V0.H4
	VSHRN2 $1, V13.S4, V0.H8
	VSHRN  $1, V14.S4, V1.H4
	VSHRN2 $1, V15.S4, V1.H8

	// QUANTB without SSE or last; dequantized in V0, V1.
	QUANTK
	VABS    V0.H8, V2.H8
	VABS    V1.H8, V3.H8
	VADD    V16.H8, V2.H8, V2.H8
	VADD    V17.H8, V3.H8, V3.H8
	VUMULL  V23.H4, V2.H4, V4.S4
	VUMULL2 V23.H8, V2.H8, V5.S4
	VUMULL  V22.H4, V3.H4, V6.S4
	VUMULL2 V22.H8, V3.H8, V7.S4
	VADD    V25.S4, V4.S4, V4.S4
	VADD    V24.S4, V5.S4, V5.S4
	VADD    V24.S4, V6.S4, V6.S4
	VADD    V24.S4, V7.S4, V7.S4
	VSHRN   $16, V4.S4, V8.H4
	VSHRN2  $16, V5.S4, V8.H8
	VSHRN   $16, V6.S4, V9.H4
	VSHRN2  $16, V7.S4, V9.H8
	VUSHR   $1, V8.H8, V8.H8
	VUSHR   $1, V9.H8, V9.H8
	VSSHR   $15, V0.H8, V10.H8
	VSSHR   $15, V1.H8, V11.H8
	VEOR    V10.B16, V8.B16, V8.B16
	VEOR    V11.B16, V9.B16, V9.B16
	VSUB    V10.H8, V8.H8, V8.H8
	VSUB    V11.H8, V9.H8, V9.H8
	VMUL    V21.H8, V8.H8, V0.H8
	VMUL    V20.H8, V9.H8, V1.H8
	VMOVQ   $0x1110090803020100, $0x0d0c070605040b0a, V28
	VMOVQ   $0x15141b1a19181312, $0x1f1e1d1c17160f0e, V29
	VTBL    V28.B16, [V8.B16, V9.B16], V10.B16
	VTBL    V29.B16, [V8.B16, V9.B16], V11.B16
	VST1    [V10.H8, V11.H8], (R2)
	VORR    V9.B16, V8.B16, V8.B16
	VUMAXV  V8.H8, V8
	VMOV    V8.H[0], R7
	CMP     $0, R7
	CSET    NE, R7
	MOVB    R7, nz+32(FP)

	IWHT
	VST1   [V0.H8, V1.H8], (R1)
	RET

// func iwhtAsm(c, o *[16]int16)
TEXT ·iwhtAsm(SB), NOSPLIT, $0-16
	MOVD c+0(FP), R0
	MOVD o+8(FP), R1
	VLD1 (R0), [V0.H8, V1.H8]
	IWHT
	VST1 [V0.H8, V1.H8], (R1)
	RET

// CHQ subtracts 64 from the pixel indices and ORs in the TBL of the next
// lo gammaToLinear quarter; a lane is in range in one quarter
// only and TBL gives 0 elsewhere, so this equals TBX (cheaper on V1).
// V3, V7 and V29 are free here.
#define CHQ \
	VSUB   V30.B16, V0.B16, V0.B16 \
	VSUB   V30.B16, V1.B16, V1.B16 \
	VSUB   V30.B16, V2.B16, V2.B16 \
	VSUB   V30.B16, V4.B16, V4.B16 \
	VSUB   V30.B16, V5.B16, V5.B16 \
	VSUB   V30.B16, V6.B16, V6.B16 \
	VLD1.P 64(R6), [V20.B16, V21.B16, V22.B16, V23.B16] \
	VTBL   V0.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V3.B16 \
	VTBL   V1.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V7.B16 \
	VTBL   V2.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V29.B16 \
	VORR   V3.B16, V8.B16, V8.B16 \
	VORR   V7.B16, V9.B16, V9.B16 \
	VORR   V29.B16, V10.B16, V10.B16 \
	VTBL   V4.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V3.B16 \
	VTBL   V5.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V7.B16 \
	VTBL   V6.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V29.B16 \
	VORR   V3.B16, V11.B16, V11.B16 \
	VORR   V7.B16, V12.B16, V12.B16 \
	VORR   V29.B16, V13.B16, V13.B16

// CHSUM sums one channel's 2x2 blocks: hi bytes of both rows (h+v>>4,
// sum <= 30) zip over row 0's lo bytes, then row 1's lo bytes and the
// pixel pairs add in uint16 (<= 16380).
#define CHSUM(lo0, lo1, hi0, hi1, t0, t1, s) \
	VUSHR   $4, hi0.B16, hi0.B16 \
	VUSRA   $4, hi1.B16, hi0.B16 \
	VZIP1   hi0.B16, lo0.B16, t0.B16 \
	VZIP2   hi0.B16, lo0.B16, t1.B16 \
	VUADDW  lo1.B8, t0.H8, t0.H8 \
	VUADDW2 lo1.B16, t1.H8, t1.H8 \
	VADDP   t1.H8, t0.H8, s.H8

// func chromaAsm(t *chromaTabs, cb, cr, p0, p1 *uint8, n int) (and uint8)
// n > 0 and n%8 == 0. Eight blocks per pass; linToGamma is
// 4*L[p] + (D[p]*x+64)>>7, D[p]*x <= 5110.
TEXT ·chromaAsm(SB), NOSPLIT, $0-49
	MOVD  t+0(FP), R5
	MOVD  cb+8(FP), R0
	MOVD  cr+16(FP), R1
	MOVD  p0+24(FP), R2
	MOVD  p1+32(FP), R3
	MOVD  n+40(FP), R4
	VMOVI $64, V30.B16
	VMOVI $255, V31.B16
	MOVD  $511, R7
	VDUP  R7, V28.H8
chloop:
	MOVD    R5, R6
	VLD4.P  64(R2), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD4.P  64(R3), [V4.B16, V5.B16, V6.B16, V7.B16]
	VAND    V3.B16, V31.B16, V31.B16
	VAND    V7.B16, V31.B16, V31.B16
	VLD1.P  16(R6), [V24.B16]
	VLD1.P  64(R6), [V20.B16, V21.B16, V22.B16, V23.B16]
	VUSHR   $4, V0.B16, V14.B16
	VUSHR   $4, V1.B16, V15.B16
	VUSHR   $4, V2.B16, V16.B16
	VUSHR   $4, V4.B16, V17.B16
	VUSHR   $4, V5.B16, V18.B16
	VUSHR   $4, V6.B16, V19.B16
	VTBL    V14.B16, [V24.B16], V14.B16
	VTBL    V15.B16, [V24.B16], V15.B16
	VTBL    V16.B16, [V24.B16], V16.B16
	VTBL    V17.B16, [V24.B16], V17.B16
	VTBL    V18.B16, [V24.B16], V18.B16
	VTBL    V19.B16, [V24.B16], V19.B16
	VADD    V0.B16, V14.B16, V14.B16
	VADD    V1.B16, V15.B16, V15.B16
	VADD    V2.B16, V16.B16, V16.B16
	VADD    V4.B16, V17.B16, V17.B16
	VADD    V5.B16, V18.B16, V18.B16
	VADD    V6.B16, V19.B16, V19.B16
	VTBL    V0.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V8.B16
	VTBL    V1.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V9.B16
	VTBL    V2.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V10.B16
	VTBL    V4.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V11.B16
	VTBL    V5.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V12.B16
	VTBL    V6.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V13.B16
	CHQ
	CHQ
	CHQ
	CHSUM(V8, V11, V14, V17, V0, V1, V20)
	CHSUM(V9, V12, V15, V18, V2, V3, V21)
	CHSUM(V10, V13, V16, V19, V4, V5, V22)

	// p = s>>9 by bytes: s>>8 <= 63 fits.
	VLD1.P  64(R6), [V24.B16, V25.B16, V26.B16, V27.B16]
	VSHRN   $8, V20.H8, V0.B8
	VSHRN2  $8, V21.H8, V0.B16
	VSHRN   $8, V22.H8, V1.B8
	VUSHR   $1, V0.B16, V0.B16
	VUSHR   $1, V1.B8, V1.B8
	VTBL    V0.B16, [V24.B16, V25.B16], V2.B16
	VTBL    V0.B16, [V26.B16, V27.B16], V3.B16
	VTBL    V1.B8, [V24.B16, V25.B16], V4.B8
	VTBL    V1.B8, [V26.B16, V27.B16], V5.B8
	VAND    V28.B16, V20.B16, V20.B16
	VAND    V28.B16, V21.B16, V21.B16
	VAND    V28.B16, V22.B16, V22.B16
	VUXTL   V3.B8, V6.H8
	VUXTL2  V3.B16, V7.H8
	VUXTL   V5.B8, V8.H8
	VMUL    V6.H8, V20.H8, V20.H8
	VMUL    V7.H8, V21.H8, V21.H8
	VMUL    V8.H8, V22.H8, V22.H8
	VSRSHR  $7, V20.H8, V20.H8
	VSRSHR  $7, V21.H8, V21.H8
	VSRSHR  $7, V22.H8, V22.H8
	VUSHLL  $2, V2.B8, V6.H8
	VUSHLL2 $2, V2.B16, V7.H8
	VUSHLL  $2, V4.B8, V8.H8
	VADD    V6.H8, V20.H8, V20.H8
	VADD    V7.H8, V21.H8, V21.H8
	VADD    V8.H8, V22.H8, V22.H8

	// s <= 1020, so int16 products; bias+dot is in [2^22, 2^26).
	VLD1.P  64(R6), [V8.B16, V9.B16, V10.B16, V11.B16]
	VLD1    (R6), [V12.B16, V13.B16]
	VMOV    V13.B16, V0.B16
	VMOV    V13.B16, V1.B16
	VMOV    V13.B16, V2.B16
	VMOV    V13.B16, V3.B16
	VSMLAL  V8.H4, V20.H4, V0.S4
	VSMLAL2 V8.H8, V20.H8, V1.S4
	VSMLAL  V9.H4, V21.H4, V0.S4
	VSMLAL2 V9.H8, V21.H8, V1.S4
	VSMLAL  V10.H4, V22.H4, V0.S4
	VSMLAL2 V10.H8, V22.H8, V1.S4
	VSMLAL  V10.H4, V20.H4, V2.S4
	VSMLAL2 V10.H8, V20.H8, V3.S4
	VSMLAL  V11.H4, V21.H4, V2.S4
	VSMLAL2 V11.H8, V21.H8, V3.S4
	VSMLAL  V12.H4, V22.H4, V2.S4
	VSMLAL2 V12.H8, V22.H8, V3.S4
	VSHRN   $16, V0.S4, V4.H4
	VSHRN2  $16, V1.S4, V4.H8
	VSHRN   $2, V4.H8, V5.B8
	VSHRN   $16, V2.S4, V6.H4
	VSHRN2  $16, V3.S4, V6.H8
	VSHRN   $2, V6.H8, V7.B8
	VST1.P  [V5.B8], 8(R0)
	VST1.P  [V7.B8], 8(R1)
	SUBS    $8, R4, R4
	BNE     chloop
	VMOV    V31.D[0], R7
	VMOV    V31.D[1], R8
	AND     R8, R7, R7
	AND     R7>>32, R7, R7
	AND     R7>>16, R7, R7
	AND     R7>>8, R7, R7
	MOVB    R7, and+48(FP)
	RET
