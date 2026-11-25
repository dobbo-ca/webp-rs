//! Go's math/rand/v2 PCG (DXSM) and the `Rand` methods the ported tests use, so the generated
//! inputs equal the Go tests'.
#![allow(dead_code)]

pub struct Pcg {
    state: u128,
}

impl Pcg {
    pub fn new(seed1: u64, seed2: u64) -> Pcg {
        Pcg {
            state: (u128::from(seed1) << 64) | u128::from(seed2),
        }
    }

    pub fn uint64(&mut self) -> u64 {
        const MUL: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;
        const INC: u128 = (6364136223846793005u128 << 64) | 1442695040888963407u128;
        self.state = self.state.wrapping_mul(MUL).wrapping_add(INC);
        let (mut hi, lo) = ((self.state >> 64) as u64, self.state as u64);
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(0xda942042e4dd58b5);
        hi ^= hi >> 48;
        hi.wrapping_mul(lo | 1)
    }

    pub fn uint32(&mut self) -> u32 {
        (self.uint64() >> 32) as u32
    }

    fn uint64n(&mut self, n: u64) -> u64 {
        if n & (n - 1) == 0 {
            return self.uint64() & (n - 1);
        }
        let m = u128::from(self.uint64()) * u128::from(n);
        let (mut hi, mut lo) = ((m >> 64) as u64, m as u64);
        if lo < n {
            let thresh = n.wrapping_neg() % n;
            while lo < thresh {
                let m = u128::from(self.uint64()) * u128::from(n);
                (hi, lo) = ((m >> 64) as u64, m as u64);
            }
        }
        hi
    }

    pub fn int_n(&mut self, n: usize) -> usize {
        self.uint64n(n as u64) as usize
    }

    pub fn float64(&mut self) -> f64 {
        ((self.uint64() << 11) >> 11) as f64 / (1u64 << 53) as f64
    }
}
