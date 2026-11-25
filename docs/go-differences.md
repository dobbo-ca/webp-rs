# Differences from the Go API

| Go | Rust |
|---|---|
| `Decode(data, *Limits)`, `DecodeContext(ctx, ...)` | `decode(data, &Limits, &AtomicBool)` |
| `Limits{MaxPixels, MaxMetadataBytes}`, zero means default | the same plus `max_alloc` and `max_threads`; no zero magic |
| `*LimitError`, `ErrLimit`, `ErrMemoryLimit` | `Error::MaxPixels`, `Width`, `Height`, `MetadataBytes`, `MemoryBudget`; the budget is per call, so there is no waiting |
| `Quality` NaN | the same: one-colour images ignore it, any other encodes as at quality 100 (no argument error) |
| `ErrFormat`, `ErrTruncated`, `ErrAnimated`, `ErrNotLossy` | `Error::Format`, `Truncated`, `Animated`, `NotLossy` |
| ctx cancelled | `Error::Cancelled` |
| `Encode(w, image.Image, *Options)`, `EncodeGray`, `EncodeYCbCr` | `encode(width, height, &Source, &Options, &Limits, &AtomicBool) -> Vec<u8>` |
| `Options` nil means Quality 75 Method 4, a zero one Method 0 | `Options::default()` is Quality 75 Method 4 |
| `Encoder` that keeps its buffers | `Encoder::new()` and `Encoder::encode`, same arguments as `encode`; each call charges the kept buffers it uses against `max_alloc` |
| (none: Go has no reusable decoder) | `Decoder::new()`, `Decoder::decode` and `Decoder::decode_planar` return `&Image` and `&Planar` valid until the next call; `Encoder::encode_into` writes into a caller's `Vec` |
| `Reset`, `MaxEncodeCharge`, `MaxEncodedSize`, `SetMemoryLimit` | not ported: drop the `Encoder` to free its buffers; the charge is per call |
| any `image.Image` | the four `Source` kinds; convert other types to NRGBA first. Planes shorter than the image are `Error::InvalidArgument`, as is Go's `Pix` that omits the last alpha byte |
| 4:4:4 YCbCr, a sub-image with an odd origin | not representable: `Source::Ycbcr420` always starts at the origin of its planes |
| `*image.NRGBA` | `Image { width, height, pix }` |

Errors Go returns unwrapped from the VP8 and VP8L decoders keep their own variants (`Vp8Header`,
`Vp8Truncated`, `Vp8lHeader`, `Vp8lTruncated`, `Vp8lBitstream`, `PrefixCode`); those it wraps in
`ErrFormat` (header errors, anything from an ALPH chunk) are `Error::Format`.
