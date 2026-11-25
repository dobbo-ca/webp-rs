//! The reference kernels: the oracle of the cross-tier tests and the fallback.

pub(super) fn luma_row(dst: &mut [u8], src: &[u8]) {
    for (d, p) in dst.iter_mut().zip(src.as_chunks::<4>().0.iter()) {
        let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
        *d = ((16839 * r + 33059 * g + 6420 * b + (1 << 15) + (16 << 16)) >> 16) as u8;
    }
}

pub(super) fn sse_row(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let k = i32::from(x) - i32::from(y);
            (k * k) as u64
        })
        .sum()
}
