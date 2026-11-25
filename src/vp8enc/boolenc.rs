//! VP8 boolean entropy encoder, ported from libwebp src/utils/bit_writer_utils.c (BSD-3-Clause).

#[derive(Default)]
pub(crate) struct BoolEncoder {
    pub buf: Vec<u8>,
    pub(super) rng: i32, // range-1, 127..254 between calls
    pub(super) value: i64,
    pub(super) run: usize, // pending 0xff bytes
    pub(super) nb_bits: i32,
}

impl BoolEncoder {
    pub(crate) fn reset(&mut self) {
        self.buf.clear();
        self.rng = 254;
        self.value = 0;
        self.run = 0;
        self.nb_bits = -8;
    }

    pub(crate) fn put_bit(&mut self, bit: bool, prob: u8) -> bool {
        let split = (self.rng * i32::from(prob)) >> 8;
        let m: i32 = if bit { -1 } else { 0 };
        self.value += i64::from((split + 1) & m);
        let r = split + ((self.rng - 2 * split - 1) & m) + 1;
        let sh = r.leading_zeros() as i32 - 24; // 0 when r >= 128
        self.rng = (r << sh) - 1;
        self.value <<= sh;
        self.nb_bits += sh;
        if self.nb_bits > 0 {
            self.flush();
        }
        bit
    }

    pub(crate) fn put_literal(&mut self, v: u32, n: i32) {
        for i in (0..n).rev() {
            self.put_bit((v >> i) & 1 == 1, 128);
        }
    }

    pub(super) fn flush(&mut self) {
        let s = 8 + self.nb_bits;
        let b = self.value >> s;
        self.value -= b << s;
        self.nb_bits -= 8;
        if b & 0xff == 0xff {
            self.run += 1;
            return;
        }
        let mut fill = 0xffu8;
        if b & 0x100 != 0 {
            fill = 0;
            if let Some(l) = self.buf.last_mut() {
                *l = l.wrapping_add(1);
            }
        }
        for _ in 0..self.run {
            self.buf.push(fill);
        }
        self.run = 0;
        self.buf.push(b as u8);
    }

    pub(crate) fn finish(&mut self) -> &[u8] {
        self.put_literal(0, 9 - self.nb_bits);
        self.nb_bits = 0;
        self.flush();
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::BoolEncoder;
    use crate::testutil::sha256;

    #[test]
    fn bool_golden() {
        let mut e = BoolEncoder::default();
        e.reset();
        let mut s = 1u32;
        for i in 0..100_000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let (mut prob, mut bit) = ((s & 0xff) as u8, (s >> 8) & 0xff >= s & 0xff);
            if (50_000..60_000).contains(&i) {
                (prob, bit) = (255, false);
            } else if (60_000..61_000).contains(&i) {
                (prob, bit) = (1, true);
            }
            e.put_bit(bit, prob);
        }
        let buf = e.finish();
        assert_eq!(buf.len(), 8029);
        assert_eq!(
            sha256(buf),
            "80cf5c7d2caa0f8aa19c183508dfa79bc6e7895582f4484c3325b2fb5c968a64"
        );
    }

    #[test]
    fn bool_empty() {
        let mut e = BoolEncoder::default();
        e.reset();
        assert_eq!(e.finish(), [0, 0]);
    }
}
