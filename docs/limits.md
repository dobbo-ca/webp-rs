# Resource limits

Every entry point takes a `Limits`; there is no unlimited call.

| Field | Unit | Default | Over it |
|---|---|---|---|
| `max_pixels` | pixels (width x height) | `1 << 26`; sides are also capped at 16383 | `Error::MaxPixels`. The side cap is `Error::Width` or `Error::Height` from the decoders and `Error::InvalidArgument` from `encode` |
| `max_alloc` | bytes | `1 << 30` (1 GiB) | `Error::MemoryBudget` |
| `max_threads` | threads, the calling one included | `std::thread::available_parallelism()`, 1 if unknown | none: the call uses fewer threads |
| `pool` | `Option<ThreadPool>` | `None`: the crate's shared pool | none |
| `max_metadata_bytes` | bytes | `1 << 24` | `Error::MetadataBytes` |

- `max_alloc` is the enforced peak working memory of one call, all its threads together: every
  allocation that scales with the input is charged before it is made and released when it is
  freed. The image a call returns stays charged until it returns. Fixed-size tables and thread
  stacks are not charged. A call that runs two things at once ([threading.md](threading.md)) needs both working sets at
  once, so it can fit `max_alloc` with `max_threads: 1` and not with more. The ALPH encoder is
  charged allocation by allocation like the rest. With `max_threads: 1` the charges are the same
  on every run, so whether a call fits is deterministic. With 2 or more, the VP8 frame and the
  ALPH chunk charge the one budget at the same time, and the encoder's macroblock rows (the
  wavefront, each worker with its own token buffer) do too, for an opaque image as well: a call
  that fits at 1 can fail with `Error::MemoryBudget` at 2, and a call whose peak is within a few buffers of `max_alloc` may
  return `Ok` on one run and the error on another, depending on how the threads interleave. The
  bytes of an `Ok` result do not depend on it. The crate has no dependency on another port, so
  there are no nested calls to share the budget with.
- Threads and pools: see [threading.md](threading.md).
- `Encoder` is `encode` with the working buffers kept between calls (Go's `webp.Encoder`), for
  encoding many images: `let mut enc = Encoder::new(); enc.encode(w, h, &src, &o, &limits,
  &cancel)?`, or `enc.encode_into(.., &mut out)` to write into a `Vec` that keeps its capacity.
  A kept buffer (the planes, the per-macroblock arrays, the token buffers, the partitions, the
  libwebp-matching state and the ALPH encoders' buffers) is charged at its capacity when the call
  uses it; one the call does not use is not charged. The buffers that are sized to the image (the
  planes and per-macroblock arrays) are dropped and allocated again when they are more than 2x
  too big for the call; the token buffers, the partitions and the ALPH scratch keep their largest
  capacity and are charged whole, and a kept partition too small for the call is charged at its
  new size while it grows. The libwebp-matching state has a fixed size: it is always kept, reset
  and charged at that size. If the encoder held any
  buffer and the call fails with `Error::MemoryBudget`, it drops them all and runs again from
  scratch, so with `max_threads` 1 a call fits whenever `encode` would. What the encoder holds
  between calls is not charged and is the caller's choice (drop it to free it). The output is the
  same as `encode`'s.
- `Decoder` is `decode` and `decode_planar` with the buffers kept between calls. At the start of
  each call every buffer it holds, the ones the call does not use too, is charged at its capacity;
  a buffer that has to grow is charged at its new size and the old size is given back. If the kept
  buffers make a call fail with `Error::MemoryBudget`, they are dropped and the call runs again
  from scratch. Free `decode` is a fresh `Decoder` too, so it does not give back the VP8L tables,
  the planes or the ALPH plane when their stage ends either: they stay in the `Decoder`, charged
  at their capacity until the call returns. See [usage.md](usage.md).
- `cancel: &AtomicBool` is checked once a pixel row, a macroblock row and a transform row by every
  thread of the call, and by the ALPH encoder every 1024 pixels, row or pass; when it is set the
  call returns `Error::Cancelled`. `decode_config` and `metadata` use no thread and have no flag.
- `decode_config` takes a `Limits` like every call but enforces none of it, as Go's
  `DecodeConfig`: it reports the size of a file `decode` would refuse.
- `Error::OutOfMemory` is the allocator refusing a request that passed the limits, or (from
  `ThreadPool::new`) the OS refusing to start the threads. It is in the same class (`limit`) as
  the five limit errors above.
- `metadata` returns at most `max_metadata_bytes` of chunks (32 bytes more for each) and charges
  the chunk entries it holds against `max_alloc`.
- The tests measure both peaks: `tests/peak.rs` runs the calls under a counting global allocator
  for several `max_alloc` values, and the crate's unit tests count live threads for
  `max_threads` 0, 1, 2, 3 and 64, with and without a caller's pool; `tests/pool.rs` checks
  that the shared pool is built only when a call needs it.

`Source` is RGBA (premultiplied), NRGBA, Gray or 4:2:0 planes (what `Planar` holds). An animated file decodes to its first frame on a transparent canvas of the file's size;
`Config::frames_dropped` counts the rest, and `decode_planar` returns `Error::Animated`.

Go's package also has a process-wide admission queue (`SetMemoryLimit`, `GOMEMLIMIT`, at most
`GOMAXPROCS` encodes at once). That is a Go runtime device and is not ported: a Rust caller
bounds its own concurrency and passes each call a `max_alloc` and a `max_threads`.
