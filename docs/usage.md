# Usage

More of the API, beside the quick start in the README.

```rust
// Lossy only: Y, Cb and Cr planes (1.5 bytes a pixel), RGB converted a band of rows at a time.
let p = webp::decode_planar(data, &limits, &cancel)?;
let mut band = vec![0u8; 3 * p.width * 8];
p.rgb(&mut band, 0, 8, &limits, &cancel)?;

// Output equal to `cwebp -q 90 -m 4` byte for byte.
let o = Options { quality: 90.0, method: 4, match_libwebp: true, ..Options::default() };
```

## Reusing an Encoder and a Decoder

To handle many images, keep one `Encoder` and one `Decoder`: they keep their buffers between calls,
so a call after one of the same size does not allocate, clear and page in the planes, the tables
and the output again.

```rust
use webp::{Decoder, Encoder};

let mut enc = Encoder::new();
let mut file = Vec::new();
for frame in frames {
    // Clears `file` and writes the WebP into it, keeping its capacity.
    enc.encode_into(w, h, &frame.src(), &o, &limits, &cancel, &mut file)?;
    send(&file);
}

let mut dec = Decoder::new();
for data in files {
    let img = dec.decode(&data, &limits, &cancel)?; // &Image, valid until the next call
    show(img.width, img.height, &img.pix);
    // dec.decode_planar(..) gives a &Planar the same way.
}
```

- With `max_threads: 1` the second call of the same size allocates nothing for `Decoder::decode`
  and `decode_planar` (the pixels, the canvas of an animated file, the ALPH plane, and the VP8 and
  VP8L tables) and for `Encoder::encode_into` of an opaque image (the VP8 planes, tokens and
  partitions and the libwebp-matching state). For an image with alpha it is the third call: the
  ALPH encoder swaps the roles of some of its reference buffers while it works, so their
  capacities settle only after two calls. `tests/reuse_alloc.rs` counts exactly these calls. The
  exceptions are the 1.6.0 ALPH writer (`alpha_libwebp16`), which makes its
  scratch in each call, and `max_threads` above 1, where each call allocates a little bookkeeping:
  the ALPH join task, the wavefront workers and the analysis bands. `Encoder::encode` also allocates the `Vec` it returns.
- The output of a call does not depend on the earlier calls: it equals `encode`'s, `decode`'s and
  `decode_planar`'s. `tests/reuse_alloc.rs` counts the allocations of the counted call (the
  second; the third for an encode with alpha) under a counting global allocator and compares the
  output.
- A `Decoder` charges every buffer it holds against `max_alloc` at its capacity at the start of
  each call, whether the call uses it or not, so a decoder that held a big image counts it
  against a small call. An `Encoder` charges each kept buffer at its capacity when the call uses
  it. In both, if the kept buffers make a call fail with `Error::MemoryBudget`, they are dropped
  and the call runs again as a fresh one (for the `Encoder`, if any buffer holds capacity), so
  with `max_threads` 1 a call fits whenever `encode` or `decode` would. See [limits.md](limits.md).
- What a `Decoder` returns is valid until its next call, which reuses the pixels. After an error
  the `Encoder` and the `Decoder` are still usable; the `Decoder` may have dropped some buffers
  and allocates them again.
- Drop the `Encoder` or the `Decoder` to free what it holds.

`Source` is RGBA (premultiplied), NRGBA, Gray or 4:2:0 planes; see [limits.md](limits.md) and
[go-differences.md](go-differences.md).
