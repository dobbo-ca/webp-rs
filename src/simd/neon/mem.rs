//! Load and store helpers for the NEON kernels: the only `unsafe` besides the dispatch in
//! `super`. Each takes a reference to an array of exactly the size the intrinsic touches, so the
//! type proves the bounds, and none needs alignment. They are
//! `#[target_feature(enable = "neon")]`, callable without `unsafe` from a NEON kernel. Not every
//! helper is used yet: the encoder kernels to come (quantize, FDCT/IDCT, block SSE) load and store
//! through them.

use std::arch::aarch64::*;

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load128(a: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1q_u8` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1q_u8(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store128(a: &mut [u8; 16], v: uint8x16_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1q_u8` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1q_u8(a.as_mut_ptr(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load64(a: &[u8; 8]) -> uint8x8_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1_u8` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1_u8(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store64(a: &mut [u8; 8], v: uint8x8_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1_u8` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1_u8(a.as_mut_ptr(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load128_s16(a: &[i16; 8]) -> int16x8_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1q_s16` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1q_s16(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store128_s16(a: &mut [i16; 8], v: int16x8_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1q_s16` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1q_s16(a.as_mut_ptr(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load128_s32(a: &[i32; 4]) -> int32x4_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1q_s32` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1q_s32(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store128_s32(a: &mut [i32; 4], v: int32x4_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1q_s32` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1q_s32(a.as_mut_ptr(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load128_u16(a: &[u16; 8]) -> uint16x8_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1q_u16` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1q_u16(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store128_u16(a: &mut [u16; 8], v: uint16x8_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1q_u16` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1q_u16(a.as_mut_ptr(), v) }
}

/// Loads 16 RGBA pixels, de-interleaved into R, G, B and A vectors.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load4x16(a: &[u8; 64]) -> uint8x16x4_t {
    // SAFETY: `a` is exactly the 64 bytes `vld4q_u8` reads, so the load is in bounds.
    unsafe { vld4q_u8(a.as_ptr()) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn load128_u32(a: &[u32; 4]) -> uint32x4_t {
    // SAFETY: `a` is a reference to exactly the bytes `vld1q_u32` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { vld1q_u32(a.as_ptr()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn store128_u32(a: &mut [u32; 4], v: uint32x4_t) {
    // SAFETY: `a` is a reference to exactly the bytes `vst1q_u32` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { vst1q_u32(a.as_mut_ptr(), v) }
}
