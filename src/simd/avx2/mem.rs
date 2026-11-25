//! Load and store helpers for the AVX2 kernels: the only `unsafe` besides the dispatch in
//! `super`. Each takes a reference to an array of exactly the size the intrinsic touches, so the
//! type proves the bounds, and none needs alignment (`loadu`/`storeu`). The 256-bit ones are
//! `#[target_feature(enable = "avx2")]`, callable without `unsafe` from an AVX2 kernel; the
//! 128-bit and 64-bit ones need SSE2 only, which every x86_64 CPU has. Not every helper is used
//! yet: the encoder kernels to come (quantize, FDCT/IDCT, block SSE) load and store through them.

use std::arch::x86_64::*;

/// Loads `a` unaligned.
#[inline]
pub(super) fn load128(a: &[u8; 16]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadu_si128` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[inline]
pub(super) fn store128(a: &mut [u8; 16], v: __m128i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_storeu_si128` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn load256(a: &[u8; 32]) -> __m256i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_loadu_si256` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn store256(a: &mut [u8; 32], v: __m256i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_storeu_si256` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[inline]
pub(super) fn loadl64(a: &[u8; 8]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadl_epi64` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadl_epi64(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[inline]
pub(super) fn storel64(a: &mut [u8; 8], v: __m128i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_storel_epi64` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[inline]
pub(super) fn load128_i16(a: &[i16; 8]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadu_si128` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[inline]
pub(super) fn store128_i16(a: &mut [i16; 8], v: __m128i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_storeu_si128` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn load256_i16(a: &[i16; 16]) -> __m256i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_loadu_si256` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn store256_i16(a: &mut [i16; 16], v: __m256i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_storeu_si256` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[inline]
pub(super) fn load128_i32(a: &[i32; 4]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadu_si128` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[inline]
pub(super) fn store128_i32(a: &mut [i32; 4], v: __m128i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_storeu_si128` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn load256_i32(a: &[i32; 8]) -> __m256i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_loadu_si256` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn store256_i32(a: &mut [i32; 8], v: __m256i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm256_storeu_si256` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[inline]
pub(super) fn load128_u32(a: &[u32; 4]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadu_si128` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[inline]
pub(super) fn store128_u32(a: &mut [u32; 4], v: __m128i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_storeu_si128` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

/// Gathers `t[idx & 255]` for the eight i32 lanes of `idx`.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn gather256_i32(t: &[i32; 256], idx: __m256i) -> __m256i {
    let idx = _mm256_and_si256(idx, _mm256_set1_epi32(255));
    // SAFETY: every lane of `idx` is at most 255 after the AND, so each 4-byte read at `t + 4 * lane` stays inside the 256 entries of `t`; scale 4 is the i32 size.
    unsafe { _mm256_i32gather_epi32::<4>(t.as_ptr(), idx) }
}
