# Parity with Go

`tests/parity/go/upstream/` is the non-test source of dobbo-ca/webp-go: `webp.go`, `internal/**`
(the `.go` and `.s` files, `internal/vp8l/LICENSE.libwebp`), `LICENSE` and `NOTICE`. Its `go.mod`
is the upstream one without the `require golang.org/x/image` line: only Go's tests import it, and
the snapshot has none, so the oracle builds with no network and no `go.sum`. `tests/parity/go/oracle/` is a Go
program with its own `go.mod` that `replace`s the module with `../upstream`. The CI job `parity`
(`.github/ci/Dockerfile`, stages `go-oracle`, `go-run`, `parity`) runs:

1. `oracle -gen tests/data /inputs` writes the inputs.
2. `oracle /inputs /go-out` runs them through webp-go (its amd64 assembly kernels included).
3. `cargo run --release --example oracle -- /inputs /rust-out` runs them through this crate.
4. `diff -rq /go-out /rust-out` must be empty. A difference is a bug in the Rust port, never in
   the oracle. The job fails if the Go tree has fewer than 50,000 outputs, 5,000 `.dec.out` or 500
   `.enc.out`, so two empty trees cannot pass. Then the subset that `parity-arm64` uses runs once
   more with `WEBP_SIMD=scalar`, and its diff must be empty too.

`parity-arm64` runs `examples/oracle.rs` on a native arm64 runner (`ubuntu-24.04-arm`) on a subset:
every `.webp` that is not a mutation (no `~` in its name), every 20th mutation, and every 4th
`.enc` under 200 kB; Go runs the subset again, and `diff -r` must be empty, once with NEON and
once with `WEBP_SIMD=scalar`. The job fails if the subset has fewer than 5,000 outputs or 50
`.enc.out`.

Both oracles bound their memory (they run inside BuildKit): every decode and encode has
`max_pixels` 1 << 22, inputs over 16 MiB are skipped, and Go's `SetMemoryLimit` is 0 (no
admission queue) with `GOMEMLIMIT`-style `debug.SetMemoryLimit(1 << 30)` for the collector.

The Rust oracle takes its `Limits` from `examples/common` (`max_pixels` as above, the default
`max_alloc` and the default `max_threads`), so it decodes lossy+ALPH files and encodes images with
alpha on two threads (images of at least 64K pixels; smaller ones stay on the calling thread);
Go runs each call on one. The output is byte identical for every
`max_threads` (`tests/threads.rs` checks 0, 1, 2 and the default), so the diff stays empty.
`decode_config` takes a `Limits` but enforces none, as Go's `DecodeConfig` has no limits.

## Inputs

`.webp` files are decoded, `.enc` files are encoded.

**Decode inputs** (`fx-<fixture>.webp`, `fx-<fixture>~<tag>.webp`, `junk-*.webp`, `ge-*.webp`):

- every `.webp` fixture of `tests/data/` (the 70 of webp-go's `internal/webp/testdata`: lossless,
  lossy, lossy with ALPH, animated, malformed) and its mutations: every cut of a file under 1,200
  bytes, else 64 header cuts, 48 spread cuts and the last 1, 2, 3 and 9 (`t<cut>`); 48 spread cuts
  with the RIFF size fixed (`u<cut>`); 64 bit flips from a fixed xorshift seed (`f<bit>`); 32 byte
  sets (`s<pos>_<value>`); 8 RIFF size edits (`r<k>`); the size field of every chunk set to 0,
  -1, +1, +2 and 0xffffffff (`z<chunk>_<k>`); three values (0, 0xff, one bit flipped) at each byte
  of the RIFF header and of the first 20 bytes of every chunk (`h<pos>_<k>`); every value of the
  first ALPH byte (`p<value>`); appended zeros with and without the RIFF size fixed (`a<k>`,
  `A<k>`);
- `junk-<nnn>.webp`: 200 random chunk soups; `junk-vp8-<nnn>.webp`, `junk-vp8l-<nnn>.webp`: 150
  VP8 and 150 VP8L chunks with valid-looking headers and random bodies, which the decoders parse
  deep into before failing; an empty file and a bare `RIFF`;
- `ge-<case>-<options>.webp`: files webp-go itself encodes (VP8X with the 1.2.2, 1.6.0 and legacy
  ALPH writers, plain VP8, gray, premultiplied), each with the light set of mutations (40 header
  cuts, 16 spread cuts, 24 flips, 12 byte sets), so the decoder runs on the encoder's output
  structure too.

**Encode inputs** (`enc-<case>-q<quality>_m<method>_f<flags>_s<segments>.enc`): header, then
pixels (the format is in `tests/parity/go/oracle/main.go`, `encFormat`): kind (NRGBA, premultiplied
RGBA, Gray, YCbCr 4:2:0), method (any i8), flags (`match_libwebp`, `alpha_libwebp16`,
`legacy_alpha`, nil Options), segments (any i8), quality (f32), width, height. The cases:

- the four fixtures of `libwebp_test.go`'s `lwCases` (`fl_768` as Gray and as NRGBA, `he_768`,
  `kodim05`, `kodim23`) and its noise pyramid (PCG 21, 22, halved to 7 levels): under the three
  options pinned from cwebp (`-q 90 -m 4`, `-q 50 -m 6`, `-q 90 -m 4 -segments 1`, all
  `match_libwebp`) and the default path at several methods and qualities; `kodim05`, `kodim23`
  and the 200 by 150 level under the full grid (methods 0 to 6, `match_libwebp` on and off,
  quality 0, 30, 75, 100, segments 0 and 1: 115 option sets each);
- `alpha_test.go`: `holes` (PCG seeded with the width) at 1x1, 7x5, 33x17, 101x250, 770x770;
  `TestMatchLibwebpAlphaALPH` (101x250, seed 5, methods 0 to 6 with every combination of
  `match_libwebp`, `alpha_libwebp16`, `legacy_alpha`; the small sizes with methods 3 to 5 and all
  eight flag combinations); `TestSegmentsAlphaAndErrors` (segments 0, 1, 4, -1, 2, 3, 5, with and
  without `match_libwebp`); `TestEncodeSolidAlpha` (alpha 0, 1, 128, 254; 770x770 for 0 and 128);
  `TestEncodeAlphaKinds` (premultiplied RGBA with nil options, opaque NRGBA);
- `encode_test.go`: `flatSizes` as Gray (1x1, 16x16, 17x33, 768x768, 1000x1), `TestEncodeBadSize`
  (0x0, 16384x1, and 0x5, 1x16384), `TestEncodeYCbCr` (37x21) and the in-gamut planes of
  `TestEncodeYCbCrOddOrigin` (at the origin; see below), with several options;
- `nanq-solid`, `nanq-textured` and `nanq-alpha`: a NaN quality (a one-colour image ignores it, any other encodes as at quality 100) under the default path and `match_libwebp` at method 4, segments 0 and 1;
- 300 random images (1 to 96 a side, a third at sizes around multiples of 16; noise under a mask,
  gradients, checkerboards, bumps, flat colours; a third with alpha holes; all four kinds) under
  random options (quality 0 to 100, fractional, and -10 and 150; method 0 to 6, and -1 and 9;
  every flag; segments 0, 1, 4 and, one case in 40, an invalid value).

Go supplies the expected bytes or error class for all of them; the Rust side is only given the
inputs.

### Go tests: counts and what is not ported

| Go package | Go tests | Rust |
|---|---|---|
| `internal/webp` | 68: the encode, alpha, decode, planar, metadata, golden and libwebp tests | ported in `tests/*.rs` (the `cwebp` and `dwebp` ones against pinned hashes), and generated here as above |
| `internal/webp` | the memory limiter (`limiter_test.go`, `limit_test.go`: slots, budget, GC accounting, cgroups) | not ported: Go's process-wide admission queue is a runtime device ([limits.md](limits.md)) |
| `internal/vp8enc` (118), `internal/vp8l` (20), `internal/riff`, `internal/vp8ldec` (1 each), `internal/fmath` (2) | unit tests of the encoder, the ALPH encoders and the helpers | the ones that test the scalar algorithms are in `src/**/tests.rs`; the asm kernel and purego tests (`kernel_*_test.go`) are ported as the cross-tier tests of `src/simd` (every tier against the scalar reference) and the `simd` fuzz target; the `lwDedup` test is ported (`NO_DEDUP` in `src/vp8enc/libwebp.rs`), and the `anaDedup` one is not, because that shortcut is not ported |

The Rust crate has 108 `#[test]` functions (`grep -rn '#\[test\]' src tests`: 70 in `src/`, 38 in `tests/`); the generators above add the
Go-produced expectations: the encoder is compared with Go on roughly a thousand inputs and the
decoder on tens of thousands of mutated and generated files (the job prints the counts).

Not generated: Go's `*image.RGBA` with a `Pix` that omits the last alpha byte, 4:4:4 YCbCr and a
sub-image with an odd or negative origin (`TestEncodeYCbCrShortPlanes`, `TestEncodeRGBAShortPix`,
the odd origins): the Rust `Source` cannot represent them ([go-differences.md](go-differences.md)). The Go tests
that need an external `cwebp` or `dwebp` are covered by their pinned hashes.

## Operations and output files

Per `.webp` the oracle runs, and writes `<input>.<op>.out` with the output or `<input>.<op>.err`
with the error class and a newline:

| op | what | output |
|---|---|---|
| `cfg` | `DecodeConfig` | `width height has_alpha lossless frames_dropped` |
| `dec` | `Decode`, `MaxPixels` 1 << 22 | width and height as u32 LE, then the RGBA pixels |
| `planar` | `DecodePlanar`, then `RGB` of all rows | width, height, the cropped Y, Cb and Cr planes, a flag byte and the alpha plane if there is one, then the RGB rows |
| `meta` | `Metadata`, default limit | one line per chunk: `exif\|xmp index truncated fnv64 length` |
| `decl-1`, `decl+0` | `Decode` with `MaxPixels` one under and equal to the image's pixels (not mutations) | digest |
| `metal33`, `metal100` | `Metadata` with `MaxMetadataBytes` 33 and 100 (not mutations) | as `meta` |

Per `.enc` it runs `enc` (`Encode`, `EncodeGray` or `EncodeYCbCr`; the output file). A mutation
(`~` in the name) writes the FNV-1a 64 and length of each output instead of the bytes, to keep the
trees small.

**Partial outputs.** Go returns no output together with an error in any entry point: `Decode`
and `DecodePlanar` return nil, `Metadata` an empty `Meta`, and `Encode` writes nothing to the
`io.Writer` when it fails. So there is no `.partial.out`, and the Rust API has no `*_into`
variant. The error classes are compared for every one.

## Error classes

| class | Go | Rust |
|---|---|---|
| `format` | `ErrFormat`, wrapped or not | `Error::Format` |
| `truncated` | `ErrTruncated` | `Error::Truncated` |
| `animated` | `ErrAnimated` | `Error::Animated` |
| `not-lossy` | `ErrNotLossy` | `Error::NotLossy` |
| `limit` | `*LimitError` (`ErrLimit`), `ErrMemoryLimit` | `Error::MemoryBudget`, `MaxPixels`, `Width`, `Height`, `MetadataBytes`, `OutOfMemory` |
| `vp8-header`, `vp8-truncated` | `vp8dec`'s `errHeader`, `errTruncated`, returned unwrapped | `Error::Vp8Header`, `Vp8Truncated` |
| `vp8l-header`, `vp8l-truncated`, `vp8l-bitstream`, `prefix-code` | `vp8ldec`'s `errSig`, `errTruncated`, `errBitstream`, `errCode` | `Error::Vp8lHeader`, `Vp8lTruncated`, `Vp8lBitstream`, `PrefixCode` |
| `argument` | the `errors.New` of `Encode`: "dimensions out of range", "Segments must be 0, 1 or 4", "YCbCr planes shorter than bounds" | `Error::InvalidArgument` |
| `unsupported` | "first partition too large", "token partition too large" | `Error::Unsupported` |
| `internal` | none (the Go oracle has no panics to catch) | `Error::Internal`, a panic on a pool thread |
| `bad-input` | a `.enc` file that is short, has a wrong pixel count or an unknown kind | the same |

`Error::Cancelled` has no Go counterpart in these runs (the oracles never cancel); anything else
prints as `other:<message>` and shows up as a diff.

## Benchmark

`bench.sh` and `.github/workflows/bench.yml`: `oracle -genbench tests/data /bench` writes every
fixture as `dec-<kind>-<name>.webp` (kinds `lossy`, `lossya` for lossy with alpha, `ll`
lossless, `anim`) and four images (`kodim05`, `kodim23`, `he_768`, `fl_768`) as `enc-*.enc`
under methods 0, 4 and 6 and `match_libwebp` at methods 4 and 6, plus a 512 by 512 alpha tile;
`oracle -bench N /bench` and `examples/bench.rs` time each, as many times as fit in about 200 ms
(at most N); `bench.sh` joins the best of five runs per side into a table of ms per Mpixel per
class. Both sides reuse one `Encoder` (Go's `webp.Encoder`, Rust's `webp::Encoder`) for all the
`.enc` inputs. `bench.sh` fails with a message if the two sides time different sets of files (one
skipped a file that the other ran, or a side has no output), so the `files` counts of both are
the same by check, not by construction.

### Hosted runner baseline

Bench run 45734302297 (2026-11-25) on a GitHub-hosted `ubuntu-24.04` runner, 2 CPUs, AVX-512 and
VNNI present (the v4 columns are real v4 builds). It is the baseline for later hosted runs; do not
compare it with the Ryzen 7 9800X3D numbers in [performance.md](performance.md) as a regression.
Rust / Go, ms per Mpixel, 92 files:

| run | all | decode classes | encode classes | worst class |
|---|---|---|---|---|
| default threads | 0.73 | 0.58 to 0.86 | 0.63 to 0.87 | 0.87 (encode alpha m4) |
| single-threaded | 0.90 | 0.59 to 0.91 | 0.82 to 1.10 | 1.10 (encode m0, encode m4) |
| v4, default threads | 0.69 | 0.57 to 0.87 | 0.56 to 0.86 | 0.87 (decode ll) |
| v4, single-threaded | 0.87 | 0.58 to 0.92 | 0.76 to 1.04 | 1.04 (encode m4) |

The Rust v4 build is faster than the default build on most classes (all, single-threaded: 42.07
against 43.68 ms/Mpixel, about 4%; encode m6 about 7 to 10%). This is in line with the v4 build
advice in [performance.md](performance.md).
