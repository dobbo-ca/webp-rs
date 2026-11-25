# Status

The port is complete and checked against Go: the decoder (RIFF walk, VP8 key frame decoder, VP8L
decoder, ALPH reader, animated files as first frame, planar decode, metadata), the encoder (the
VP8 lossy key frame encoder in both its paths, the default one and `match_libwebp`, with its
ALPH writers: the simple one, libwebp 1.2.2's and libwebp 1.6.0's) and the SIMD tiers (see [safety.md](safety.md)).
Four CI jobs gate `main`: `test`, `arm64`, `parity` and `parity-arm64`.

- `tests/parity/` runs webp-go (a snapshot) and this crate on the same inputs (fixture
  mutations, generated files, the encoder tests' images under many options) and diffs the output,
  on x86_64 and, on a native arm64 runner, aarch64: see [parity.md](parity.md).
- `tests/fuzz/` has targets for the decoders and the header and metadata parsers (`decode`, `metadata`),
  the encoder (`encode`, `encode_limits`), the reused `Encoder` (`encode_reuse`: three calls of
  one `Encoder`, each equal to a fresh `encode`; `decode` also checks a reused `Decoder`, warmed on
  other files, against the free functions) and the SIMD kernels (`simd`: the colour
  conversion and SSE rows, and every kernel of the `enc_i4`, `enc_pred`, `enc_block`,
  `enc_collect`, `enc_chroma` and `enc_cost` groups, each tier against its scalar reference, bit
  for bit); `.github/workflows/fuzz.yml` runs them weekly, on x86_64 and the NEON kernels on a
  native arm64 runner. A unit test (`reused_encoder_equals_fresh`) also compares reused output with fresh output.
- `.github/workflows/bench.yml` times Go and Rust on the same inputs: see [parity.md](parity.md).
  Build profile, `Encoder` reuse, the v4 columns and the PGO/BOLT measurement (`pgo.yml`): see [performance.md](performance.md).
- `bench.yml` also runs `examples/kbench.rs`, the per-kernel timings of the scalar, AVX2 and AVX-512 tiers;
  the AVX-512 kernels that lose to AVX2 there are dropped ([safety.md](safety.md)).
