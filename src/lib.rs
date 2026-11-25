//! WebP: a decoder for still WebP (lossless, and lossy with or without an ALPH chunk, bit exact
//! with libwebp's dwebp) and a lossy VP8 key frame encoder. A port of
//! `github.com/dobbo-ca/webp-go`.
//!
//! # Quick start
//!
//! ```
//! use std::sync::atomic::AtomicBool;
//! use webp::{Limits, Options, Source};
//!
//! # fn main() -> Result<(), webp::Error> {
//! let (w, h) = (16, 16);
//! let pix = vec![200u8; 4 * w * h];
//! let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
//! let cancel = AtomicBool::new(false);
//!
//! let src = Source::Nrgba { pix: &pix, stride: 4 * w };
//! let file = webp::encode(w, h, &src, &Options::default(), &limits, &cancel)?;
//! let cfg = webp::decode_config(&file, &limits)?;
//! let img = webp::decode(&file, &limits, &cancel)?;
//! assert_eq!((cfg.width, img.width, img.pix.len()), (w, w, 4 * w * h));
//! # Ok(())
//! # }
//! ```
//!
//! # Resource limits
//!
//! Every entry point takes a [`Limits`]; there is no unlimited call. Its fields, with the error
//! each returns:
//!
//! | Field | Unit | Default | Over it |
//! |---|---|---|---|
//! | `max_pixels` | pixels | `1 << 26` (sides are also capped at 16383) | `Error::MaxPixels` |
//! | `max_alloc` | bytes | `1 << 30` | `Error::MemoryBudget` |
//! | `max_threads` | threads | `std::thread::available_parallelism()`, else 1 | none: the call uses fewer threads |
//! | `pool` | [`ThreadPool`] | `None`: the crate's shared pool | none |
//! | `max_metadata_bytes` | bytes | `1 << 24` | `Error::MetadataBytes` |
//!
//! The side cap of 16383 is `Error::Width` or `Error::Height` from the decoders and
//! `Error::InvalidArgument` from `encode`.
//!
//! `max_alloc` is the enforced peak working memory of one call, all its threads together: every
//! allocation that scales with the input is charged before it is made and released when it is
//! freed, so a caller can stay under a container's memory limit. The image the call returns stays
//! charged until it returns. `max_threads` is the most OS threads one call uses, the calling
//! thread included; with `max_threads: 1` (or 0) no other thread is used, and a tiny image (under
//! 4K pixels) stays on the calling thread. The other threads are the workers of a [`ThreadPool`]:
//! the one in `Limits::pool` if the caller made one, else the crate's shared pool, built on the
//! first call with `max_threads` above 1 and never larger than
//! `std::thread::available_parallelism()`. The output is the same for every `max_threads`, but
//! whether a call fits `max_alloc` is not: a call that fits at `max_threads: 1` can fail with
//! `Error::MemoryBudget` at 2, where the VP8 and ALPH work run at once (and, for an encode, the
//! macroblock rows of the wavefront, opaque images too) and charge the budget together, and with 2 or more
//! threads the outcome of a call whose peak is near `max_alloc` depends on how the threads
//! interleave (see [`Limits::max_alloc`]). A caller limited to 2 CPUs and 1 GiB sets `max_threads: 2`
//! and `max_alloc: 768 << 20`, and may pass its own pool.
//!
//! `Encoder` keeps its working buffers between calls to encode many images faster; each call still
//! charges what the encoder holds against its `max_alloc`, and the memory it holds between calls
//! is the caller's choice.
//!
//! Decode and encode take `cancel: &AtomicBool`, checked once a pixel, macroblock or transform
//! row by every thread of the call, the ALPH encoder's included; once it is set the call returns
//! `Error::Cancelled`.
//!
//! ```
//! use std::sync::atomic::AtomicBool;
//! use webp::{Error, Limits, Options, Source, ThreadPool};
//!
//! # fn main() -> Result<(), Error> {
//! let (w, h) = (32, 32);
//! let pix: Vec<u8> = (0..4 * w * h).map(|i| (i * 7) as u8).collect();
//! // Two threads for a call, taken from a pool of the caller's own (created once and kept).
//! let pool = ThreadPool::new(2)?;
//! let limits = Limits { max_threads: 2, max_alloc: 64 << 20, pool: Some(pool), ..Limits::default() };
//! let src = Source::Nrgba { pix: &pix, stride: 4 * w };
//! let file = webp::encode(w, h, &src, &Options::default(), &limits, &AtomicBool::new(false))?;
//!
//! // A flag that is set (here from the start; in real use by another thread) stops the call.
//! let stop = AtomicBool::new(true);
//! assert_eq!(webp::decode(&file, &limits, &stop).err(), Some(Error::Cancelled));
//! // A max_alloc under the pixels' 4 bytes each is refused before anything is allocated.
//! let tiny = Limits { max_alloc: 1000, ..limits };
//! let r = webp::decode(&file, &tiny, &AtomicBool::new(false));
//! assert!(matches!(r, Err(Error::MemoryBudget { .. })));
//! # Ok(())
//! # }
//! ```
//!
//! Go's process-wide admission queue (`SetMemoryLimit`, `GOMEMLIMIT`) has no equivalent: Rust
//! callers bound concurrency themselves and give each call its own `max_alloc`.
#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]
// The codec loops are ported index for index from libwebp, with the same shapes.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod decode;
mod encode;
mod error;
mod fmath;
mod gosort;
mod limits;
mod parallel;
mod riff;
mod simd;
mod tables;
mod vp8dec;
mod vp8enc;
mod vp8l;

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod testutil;
mod vp8ldec;

pub use decode::{Config, Decoder, Image, Planar, decode, decode_config, decode_planar, metadata};
pub use encode::{Encoder, Options, Source, encode};
pub use error::Error;
pub use limits::Limits;
pub use parallel::ThreadPool;
#[doc(hidden)]
pub use parallel::shared_pool_started;
pub use riff::{Meta, MetaChunk};

/// The kernel micro-bench of `examples/kbench.rs`: times one SIMD kernel of one tier. Not API.
#[doc(hidden)]
pub mod kbench_api {
    /// The kernel names `run` knows.
    pub const KERNELS: &[&str] = crate::simd::kbench::KERNELS;

    /// The tiers this CPU can run: `scalar`, then `avx2`, `avx512` or `neon` as present.
    pub fn tiers() -> Vec<&'static str> {
        crate::simd::kbench::tier_names()
    }

    /// The mean ns a call of kernel `name` takes on `tier`, over `iters` calls on deterministic
    /// inputs of a realistic size; `None` for an unknown kernel or a tier this CPU lacks.
    pub fn run(name: &str, tier: &str, iters: u32) -> Option<f64> {
        crate::simd::kbench::run(name, tier, iters)
    }
}

/// The SIMD tiers, for the fuzz targets (`--cfg fuzzing`) to compare with the scalar definition.
/// Not API.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_api {
    /// A kernel writing luma into `dst` from RGBA `src` of exactly `4 * dst.len()` bytes.
    pub type LumaRow = fn(&mut [u8], &[u8]);
    /// A kernel summing squared differences of equal-length slices.
    pub type SseRow = fn(&[u8], &[u8]) -> u64;

    /// Every tier this CPU has, scalar included: name, `luma_row`, `sse_row`.
    pub fn kernels() -> Vec<(&'static str, LumaRow, SseRow)> {
        let scalar = crate::simd::SCALAR;
        std::iter::once(("scalar", scalar.luma_row, scalar.sse_row))
            .chain(
                crate::simd::tiers()
                    .into_iter()
                    .map(|(n, k)| (n, k.luma_row, k.sse_row)),
            )
            .collect()
    }

    /// Checks one kernel of group `group % 6` (`enc_i4`, `enc_pred`, `enc_block`, `enc_collect`,
    /// `enc_chroma`, `enc_cost`) on inputs decoded from `data`: every tier against the scalar reference.
    /// Panics on a difference.
    pub fn kernel(group: u8, data: &[u8]) {
        crate::simd::fuzz(group, data);
    }
}
