/// An LSB-first bit reader over a byte slice; reading past the end sets `over`.
pub(super) struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
    pub(super) val: u64,
    pub(super) n: u32,
    pub(super) over: bool,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(b: &'a [u8]) -> BitReader<'a> {
        BitReader {
            b,
            pos: 0,
            val: 0,
            n: 0,
            over: false,
        }
    }

    pub(super) fn refill(&mut self) {
        if self.pos + 8 <= self.b.len() {
            let mut a = [0u8; 8];
            a.copy_from_slice(&self.b[self.pos..self.pos + 8]);
            self.val |= u64::from_le_bytes(a) << self.n;
            let k = (63 - self.n) >> 3;
            self.pos += k as usize;
            self.n += k << 3;
            return;
        }
        while self.n <= 56 && self.pos < self.b.len() {
            self.val |= u64::from(self.b[self.pos]) << self.n;
            self.pos += 1;
            self.n += 8;
        }
    }

    /// Drops k bits.
    pub(super) fn skip(&mut self, k: u32) {
        if k > self.n {
            self.over = true;
            self.val = 0;
            self.n = 0;
            return;
        }
        self.val >>= k;
        self.n -= k;
    }

    pub(super) fn bits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.refill();
        }
        let v = (self.val & ((1u64 << k) - 1)) as u32;
        self.skip(k);
        v
    }

    pub(super) fn bit(&mut self) -> bool {
        self.bits(1) != 0
    }
}
