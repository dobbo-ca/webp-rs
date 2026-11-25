//! Load and store helpers for the AVX-512 kernels: the only `unsafe` besides the dispatch in
//! `super`. As in `avx2/mem.rs`, each takes a reference to an array of exactly the size the
//! intrinsic touches, so the type proves the bounds, and none needs alignment (`loadu`/`storeu`).
//! The 512-bit ones are `#[target_feature(enable = "avx512f")]`, the 256-bit ones `"avx2"`;
//! both are callable without `unsafe` from an AVX-512 kernel. The 128-bit and 64-bit ones need
//! SSE2 only, which every x86_64 CPU has.

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
#[inline]
pub(super) fn load128_u32(a: &[u32; 4]) -> __m128i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm_loadu_si128` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
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
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn load512(a: &[u8; 64]) -> __m512i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_loadu_si512` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn store512(a: &mut [u8; 64], v: __m512i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_storeu_si512` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn load512_i16(a: &[i16; 32]) -> __m512i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_loadu_si512` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn store512_i16(a: &mut [i16; 32], v: __m512i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_storeu_si512` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn load512_i32(a: &[i32; 16]) -> __m512i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_loadu_si512` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

/// Stores `v` to `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn store512_i32(a: &mut [i32; 16], v: __m512i) {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_storeu_si512` writes, so the store is in bounds, and it has no alignment requirement.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}

/// Loads `a` unaligned.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn load512_u32(a: &[u32; 16]) -> __m512i {
    // SAFETY: `a` is a reference to exactly the bytes `_mm512_loadu_si512` reads, so the load is in bounds, and it has no alignment requirement.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

/// Gathers `t[idx & 255]` for the sixteen i32 lanes of `idx`.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn gather512_i32(t: &[i32; 256], idx: __m512i) -> __m512i {
    let idx = _mm512_and_si512(idx, _mm512_set1_epi32(255));
    // SAFETY: every lane of `idx` is at most 255 after the AND, so each 4-byte read at `t + 4 * lane` stays inside the 256 entries of `t`; scale 4 is the i32 size.
    unsafe { _mm512_i32gather_epi32::<4>(idx, t.as_ptr().cast()) }
}
