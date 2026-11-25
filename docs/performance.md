# Performance

How to build and call this crate for speed. The Go and Rust timings are produced by
`.github/workflows/bench.yml` (`tests/parity/bench.sh`, see [parity.md](parity.md)).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. This crate's own `[profile.release]` is `codegen-units = 1` and `lto = "fat"`. For a
worker that encodes or decodes images:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the kernels and the loops that call them inline across
  modules and across the crate boundary. A dependent that only wants the one-unit build of this
  crate can set `[profile.release.package.webp] codegen-units = 1`.
- `panic = "abort"` removes the unwinding tables. It is not set in this crate's `Cargo.toml`, since a
  library cannot choose it for its consumer. It changes one thing here: a panic on a pool thread
  is caught and returned as `Error::Internal` ([threading.md](threading.md)), and with `abort` the
  process stops at the panic instead. The crate does not panic on any input (clippy denies
  `unwrap`, `expect` and `panic` in its non-test code), so this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 everywhere, not only in the kernels.
  **Use it only when every machine that runs the binary has AVX-512.** On a CPU without it the
  program dies with `SIGILL` (illegal instruction) at the first such instruction, with no message.
  The default build needs no flag and runs everywhere. Do not set it for an aarch64 build.
- The default build still picks its SIMD tier at run time (AVX-512, AVX2 or NEON, `WEBP_SIMD=scalar` forces
  the scalar one; [safety.md](safety.md)), so it needs no flag to use the kernels.
- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` is part of the v4 recommendation. On Zen 4
  and Zen 5 `tzcnt`, `blsr`, `blsi` and `blsmsk` wait for the last write to their output register;
  LLVM breaks that wait only under `znver4`/`znver5` tuning, not under `x86-64-v4`, so a v4 build
  without the two features can be slower than the default build (a sibling codec port measured 30 to 50% on
  its entropy coder). rustc warns that it does not know the two features and passes them to LLVM,
  which does. The output bytes are the same in every build.

## The v4 columns

`bench.yml` also builds `bench-v4` (the flags above, its own target cache `webp-target-v4`) and the
Go oracle with `GOAMD64=v4` (`oracle-v4`); `tests/parity/bench.sh` times them like the default
builds (best of five runs) and adds the columns `Go v4 ms/Mpixel`, `Rust v4 ms/Mpixel` and
`Rust v4 / Go v4` to both tables. The `ci` stage runs `cargo test --locked --release --test
encode` with the same flags, so the v4 build must give the same bytes as libwebp's.

A `bench` run on an AMD Ryzen 7 9800X3D, `Rust v4 / Go v4`:

| class | default threads | 1 thread |
|---|---|---|
| all | 0.39 | 0.88 |
| encode m0 | 0.40 | 1.01 |
| encode m4 | 0.40 | 1.01 |
| encode m6 | 0.24 | 0.78 |
| encode match_libwebp m4 | 0.41 | 0.98 |
| encode match_libwebp m6 | 0.36 | 0.96 |

For Rust the v4 build is about 1 to 5% faster than the default build.

## Reusing an `Encoder` and a `Decoder`

Keep one `Encoder` and one `Decoder` for each worker thread and handle every image on them.
`Encoder::encode` takes the same arguments as `webp::encode` and gives the same bytes; the
difference is what it keeps between calls (`src/encode.rs`, `Bufs` in `src/vp8enc/mod.rs`): the
padded Y, Cb and Cr source planes, the per-macroblock array, the top-context and row arrays, the
token buffers, the three partition buffers, an alpha and a strip buffer, the ALPH encoder's
scratch, and the libwebp-matching state (its cost tables) for `match_libwebp` at method 3 and
above. `Decoder` keeps the pixels, the canvas, the ALPH plane, the planes and the VP8 and VP8L
tables ([usage.md](usage.md)).

- `Encoder::encode_into` writes into a `Vec` the caller keeps, so the output is not allocated
  either; `Encoder::encode` returns a new `Vec` on every call.
- A later call of a similar size does not allocate and zero those buffers again (stale contents are
  always overwritten before they are read). The buffers sized to the image (the planes and the
  per-macroblock arrays) are dropped and allocated again when they are more than 2x too big for
  the call; a larger image grows them. The token buffers, the partitions and the ALPH scratch
  keep their largest capacity. The libwebp-matching state has a fixed size and is always kept
  and reset.
- With `max_threads` 1, `tests/reuse_alloc.rs` counts the allocator calls of the second call
  (the third for an image with alpha, as the ALPH encoder swaps its reference buffers) and finds
  none, apart from the 1.6.0 ALPH writer, which makes its scratch in each call. Above 1 thread
  small bookkeeping is allocated in each call: the ALPH join task, the wavefront workers and the
  analysis bands.
- Every charge still goes through the ledger: a kept buffer is charged to the call at its
  capacity when the call uses it, so `max_alloc` limits the peak of a call as before; if the kept
  buffers make a call exceed it they are dropped and the call runs again from scratch.
- Memory stays with the `Encoder` or `Decoder` until it drops: about the working set of the
  largest image it has seen.
- Both take `&mut self`; give each worker thread its own. The bench of both sides (`oracle -bench`,
  `examples/bench.rs`) reuses one encoder for all the encodes.

## Threads and SIMD

With `max_threads` above 1 the encoder runs its macroblock rows as a wavefront over the rows, the
segment analysis in bands, and the ALPH chunk beside them; see [threading.md](threading.md). The
output is the same for every thread count.

The SIMD tier is picked at run time: AVX-512 where the CPU has the four features it needs, else
AVX2 on x86_64, NEON on aarch64, with `WEBP_SIMD=scalar` or `avx2` to force a lower tier. The
AVX-512 kernels are only those that beat AVX2 in `examples/kbench.rs`; see
[safety.md](safety.md).

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
worker that links this crate); the library does not change. `pgo.yml` (dispatch only) measures how
much they gain on the bench example: it builds the example with `cargo-pgo`, trains it on the
benchmark inputs (`oracle -genbench tests/data /bench`) twice, single-threaded
(`cargo pgo run -- --locked --example bench -- /bench 40 1`) and with the default threads
(`... -- /bench 40`), then times the default, the PGO and the PGO plus BOLT binaries with
`tests/parity/bench.sh` through `tests/parity/pgo.sh`, and prints, for each class, the Rust
ms/Mpixel and the change against the default build, for the default threads and for one thread,
in the job summary. BOLT is optional there: if the package is not available the job reports PGO
only.

A `pgo.yml` run on an AMD Ryzen 7 9800X3D, Rust ms/Mpixel change
against the default build:

| class | PGO (default threads) | PGO + BOLT (default threads) | PGO (1 thread) | PGO + BOLT (1 thread) |
|---|---|---|---|---|
| all | -5.1% | -5.7% | -1.8% | -2.9% |
| decode anim | -2.9% | -3.5% | -3.4% | -4.0% |
| decode ll | -10.8% | -10.4% | -9.9% | -9.5% |
| decode lossy | -4.8% | -4.2% | -4.5% | -4.1% |
| decode lossya | -3.0% | -4.3% | -3.5% | -4.3% |
| encode alpha m4 | -2.9% | -1.8% | -1.3% | -2.3% |
| encode alpha match_libwebp m4 alpha16 | -9.7% | -9.4% | -7.1% | -7.9% |
| encode alpha match_libwebp m4 | -8.3% | -6.8% | -5.7% | -5.1% |
| encode m0 | +4.2% | +0.6% | +4.0% | +3.0% |
| encode m4 | +5.4% | +2.0% | +3.6% | +2.6% |
| encode m6 | -5.0% | -8.0% | -1.0% | -1.8% |
| encode match_libwebp m4 | -2.0% | -3.3% | +1.5% | -2.4% |
| encode match_libwebp m6 | -4.0% | -4.1% | -1.2% | -2.7% |

Against PGO alone, BOLT ranges from 1.5 points slower (encode alpha match_libwebp m4, default
threads) to 3.9 points faster (encode match_libwebp m4, one thread); on all it is 0.6 points
faster with the default threads and 1.1 with one. It is a little slower than PGO alone on decode
ll and lossy and on encode alpha match_libwebp m4, and on encode alpha m4 with the default threads. The plain m0 and m4 encodes get slower, probably because the profile is trained on
all classes, and the m6 and libwebp paths dominate its counts, so the code layout favours them. A
worker that encodes only at m4 trains on its own workload.

The gain is an upper bound for another workload: the training input is the benchmark's own, so the
profile fits it more closely than it would fit a real image.
