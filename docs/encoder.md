# Encoder

The default path is the Go encoder's own mode search (one partition for the first macroblock
header and two for the tokens, with skip flags). `match_libwebp` at methods 3 to 6 is libwebp's
RD_OPT_BASIC, with its decisions and one token partition. `src/fmath.rs` is Go's `math.Pow` as
amd64 with FMA computes it, so the quality curve, the segment quantisers and the gamma tables are
bit for bit what the Go encoder uses on every machine (`f64::mul_add` is the explicit FMA, and
plain Rust products are never fused). The encoder tests pin cwebp's output by SHA-256 (the
`match_libwebp` goldens in `tests/encode.rs`) and check the reconstruction against the decoder.

Ported from Go's fast paths: `lwDedup` (equal predictions share a rate-distortion trial in
`match_libwebp`; a test compares it with the dedup off), the asm kernels and their batch forms
(`quantAsm`, `residualAsm`, `idctAddAsm`, `resCostAsm`, `costGridAsm`, `cost4Asm`, the batched
i4 trials) as the SIMD tiers, and the `Encoder`'s kept buffers. `anaDedup` (skipping the TM
histogram when TM predicts DC's bytes) is not ported: it is output neutral in Go, so the full
computation is the reference.

The macroblock pass runs as a wavefront over the macroblock rows, on as many threads as
`max_threads` gives: a row's macroblock x starts when the row above has finished macroblock x + 1,
so every macroblock reads the reconstruction and contexts a serial pass gives it, and the bytes do
not depend on the thread count. Each worker records tokens into its own buffer, coded afterwards in
row order. On the `match_libwebp` path the wavefront stops at each of libwebp's probability
refreshes, where the statistics are recorded in raster order and the tables rebuilt, so cwebp's
output is kept. [threading.md](threading.md) has the details.
