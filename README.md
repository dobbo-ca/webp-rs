# webp-rs

WebP in Rust: a decoder for still WebP (lossless, and lossy with or without an ALPH chunk, bit
exact with libwebp's `dwebp`), and a lossy VP8 key frame encoder. A port of
[dobbo-ca/webp-go](https://github.com/dobbo-ca/webp-go), which has the design. Pure
Rust, one dependency (`rayon`, for the worker pool: [docs/dependencies.md](docs/dependencies.md));
`unsafe` only in the SIMD load/store helpers and tier dispatch ([docs/safety.md](docs/safety.md)).
Not on crates.io: `webp = { git = "https://github.com/dobbo-ca/webp-rs", rev = "<sha>" }`.

## Quick start

```rust
use std::sync::atomic::AtomicBool;
use webp::{Limits, Options, Source};

// Every call takes its limits. This caller has one CPU and 64 MiB to give.
let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
let cancel = AtomicBool::new(false);

let cfg = webp::decode_config(data, &limits)?;   // size and kind, no pixels
let img = webp::decode(data, &limits, &cancel)?; // RGBA, 4 bytes a pixel
let meta = webp::metadata(data, &limits)?;       // EXIF and XMP chunks, aliasing data

// Lossy VP8, quality 75 method 4 by default; the source rows are `stride` bytes apart.
let src = Source::Nrgba { pix: &pix, stride: 4 * width };
let file = webp::encode(width, height, &src, &Options::default(), &limits, &cancel)?;
```

Planar decode, `cwebp`-equal output and reusing an `Encoder` and a `Decoder`: [docs/usage.md](docs/usage.md).

## Resource limits

Every entry point takes a `Limits`; there is no unlimited call.

| Field | Unit | Default | Over it |
|---|---|---|---|
| `max_pixels` | pixels (width x height) | `1 << 26`; sides are also capped at 16383 | `Error::MaxPixels`; the side cap: [docs/limits.md](docs/limits.md) |
| `max_alloc` | bytes | `1 << 30` (1 GiB) | `Error::MemoryBudget` |
| `max_threads` | threads, the calling one included | `std::thread::available_parallelism()`, 1 if unknown | none: the call uses fewer threads |
| `pool` | `Option<ThreadPool>` | `None`: the crate's shared pool | none |
| `max_metadata_bytes` | bytes | `1 << 24` | `Error::MetadataBytes` |

- `max_alloc` is the enforced peak working memory of one call, all its threads together. The
  crate has no dependency on another port, so there are no nested calls to share it with.
- `max_threads: 1` spawns no thread and never touches a pool. The output is the same for every
  `max_threads` and pool. A caller's `Limits::pool` always wins; without one, a call above 1
  thread uses the crate's shared pool, built lazily with at most `available_parallelism` threads.
- `cancel: &AtomicBool` is checked by every thread of the call; once set, the call returns
  `Error::Cancelled`.

Details: [docs/limits.md](docs/limits.md), [docs/threading.md](docs/threading.md).

## Docs

[docs/status.md](docs/status.md) (what is ported, how it is checked), [docs/go-differences.md](docs/go-differences.md),
[docs/encoder.md](docs/encoder.md), [docs/safety.md](docs/safety.md) (`unsafe`, SIMD), [docs/parity.md](docs/parity.md)
(Go parity, benchmark) and [docs/performance.md](docs/performance.md) (build profile, reuse, v4, PGO, BOLT).

## Third-party code

`LICENSE` is Apache-2.0. Parts of the decoders and most of the encoder port libwebp
(BSD-3-Clause, `LICENSE.libwebp`), and some tables and helpers follow golang.org/x/image and Go's
standard library; `NOTICE` lists what and in which Rust file.
