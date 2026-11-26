# Safety and SIMD

`unsafe` is only in the SIMD load/store helpers and tier dispatch (`src/simd/`); everything else is
`#![deny(unsafe_code)]`.

`src/simd/` has a scalar reference and, beside it, AVX2 and AVX-512 (x86_64) and NEON (aarch64) kernels for the
encoder: the RGBA to luma row and the sum of squared differences of a prediction, and the block
kernels of Go's assembly, in groups (`enc_i4` i4 mode trials, `enc_pred` predictor SSE, `enc_block`
fdct, quantizer and IDCT, `enc_collect` analysis bins, `enc_chroma` chroma rows, `enc_cost` rate costs). The tier is
picked once at first use by runtime detection; `WEBP_SIMD=scalar` forces the scalar one and
`WEBP_SIMD=avx2` rules out AVX-512 (so the bench can compare the two); `WEBP_SIMD=avx512` requires it
and panics on a CPU without AVX-512 F, BW and VL, for CI under an emulator. The
kernels are safe `#[target_feature]` functions with no `unsafe` inside. `unsafe` is in two small
places per tier: the load/store helpers (`src/simd/avx2/mem.rs`, `avx512/mem.rs`, `neon/mem.rs`, whose
array-reference parameter types prove the bounds) and the dispatch next to the CPU detection, which
calls the kernels after the feature check. The tests run every detected tier against scalar, bit
for bit, on random and extreme inputs and every edge length, and assert that AVX2 (x86_64) or NEON
(aarch64) was detected. CI runs the suite with both tiers on x86_64 and, on a native arm64
runner, on aarch64, and lints the aarch64 target. The AVX-512 kernels run natively when the hosted
runner has AVX-512, and always under Intel SDE (job `avx512-sde`: the unit tests on `-spr` with
`WEBP_SIMD=avx512`). The `simd` fuzz target checks every kernel the same way.

## The AVX-512 tier

`src/simd/avx512.rs` and `src/simd/avx512/` follow the AVX2 layout: `avx512/mem.rs` holds the
512-bit (and the few 128 and 256-bit) load, store and gather helpers, the only `unsafe` besides the
dispatch, and the kernels are safe `#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]`
fns. The tier needs those four features (`is_x86_feature_detected!`, in `available()`); it uses no
VNNI or VBMI. It is not a full copy of the AVX2 tier: it holds the kernels that do more work in
512-bit registers, and each group's `detect` returns the AVX2 `Kernels` struct with those entries
replaced, so a kernel that loses is dropped by pointing its entry back at the AVX2 fn. The kernels:

- `luma_row` (16 pixels a step) and `sse_row` (64 bytes a step), tails through zero-padded copies.
- `quantize_n`: one block in 16 i32 lanes, the zigzag reorder (`vpermw`) and the nonzero test
  (`vpcmpw` mask) in registers.
- `residuals_y` and `idct_adds_y`: the four blocks of a block row in the four 128-bit lanes, so
  the per-lane unpacks of the AVX2 transposes carry over unchanged.
- `sse4x10` (five predictions a register).
- `chroma`: 16 blocks a pass, the gamma table through `vpermw` on two 32-entry u16 tables (the
  AVX2 form's four `pshufb`s) and 16-lane gathers for `to_linear`.

Not done, because one 256-bit register already holds the work of a call: `fdct4`, `quantize_sse`,
`idct_add4`, `collect16`, `collect_uv`, `res_cost` and `cost_grid`; and not done for effort:
`i4_trials`, `rec_metrics16`, `tt_mb`, `edge_sse`, `residual_quant3`, `pred4_all` and the chroma
(uv) batches. They run the AVX2 fns on an AVX-512 CPU.

The cross-tier tests and the fuzz groups list `avx512` next to `avx2` and skip it on a CPU without
the features (only AVX2 and NEON are asserted present). `examples/kbench.rs` times every kernel on
each tier (through `webp::kbench_api`, a `#[doc(hidden)]` module that exposes `run(name, tier,
iters)` and no internals); `bench.yml` appends its table to the job summary.

## NEON rate costs

`neon/enc_cost.rs` has `res_cost` and `cost_grid`. As in the AVX2 form there is no gather: the 16
table indices come out of one vector computation (`vabsq_s16`, `vminq_u16`, shifts and adds, a
`vextq_u16` for the shifted context), are stored to arrays and read with sixteen scalar loads;
`cost_grid` gets the nonzero lanes as a bit mask from `vceqzq_s16` and `vaddvq_u16`. The
cross-tier tests and the `enc_cost` fuzz group cover it, and the arm64 CI stage runs them on a
native arm64 runner. Its speed is not measured: the bench runs on x86_64 only. Go's
arm64 assembly has the same kernels (`resCostAsm`, `costGridAsm`), which is why it is kept.

## Kernels measured and dropped

From the `kbench` table of a `bench` run (AMD Ryzen 7 9800X3D; ns a call,
best of three). A kernel at 1.00 or more over AVX2 was dropped: its AVX-512 code is deleted and
the group's entry stays AVX2.

| kernel | avx2 ns | avx512 ns | avx512/avx2 | verdict |
|---|---:|---:|---:|---|
| `luma_row` | 74.2 | 29.3 | 0.40 | kept |
| `sse_row` | 15.6 | 9.9 | 0.63 | kept |
| `quantize_n` | 38.8 | 17.9 | 0.46 | kept |
| `residuals_y` | 88.6 | 22.3 | 0.25 | kept |
| `idct_adds_y` | 90.5 | 24.5 | 0.27 | kept |
| `sse4x10` | 3.6 | 3.0 | 0.83 | kept |
| `chroma` | 999.5 | 814.0 | 0.81 | kept |
| `sse_block` | 7.7 | 10.2 | 1.33 | dropped |
| `bin_stats` | 17.4 | 22.3 | 1.28 | dropped |

