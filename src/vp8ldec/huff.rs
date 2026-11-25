//! Canonical prefix codes, packed MSB first like deflate's.

use super::bits::BitReader;
use crate::Error;
use crate::limits::{Budget, cap_bytes, fit_zeroed};

/// Bounds the memory of a code besides its symbols (a test checks it).
pub(super) const CODE_BYTES: u64 = 576;

pub(super) struct Code {
    single: bool,
    sym: u16,
    /// `sym << 4 | len` for codes of at most 8 bits, else 0.
    root: [u16; 256],
    count: [u16; 16],
    /// By length, then value.
    syms: Vec<u16>,
}

impl Default for Code {
    fn default() -> Code {
        Code::new()
    }
}

impl Code {
    /// The bytes the symbol table holds, at capacity.
    pub(super) fn bytes(&self) -> u64 {
        cap_bytes(&self.syms)
    }

    pub(super) fn new() -> Code {
        Code {
            single: false,
            sym: 0,
            root: [0; 256],
            count: [0; 16],
            syms: Vec::new(),
        }
    }

    /// Makes the code of `lens`. One used symbol codes in zero bits; any other code must be
    /// complete; with `keep` false it is only checked. The symbol table keeps its capacity from
    /// the code's earlier use; if it must grow, the new size is charged to `budget` first.
    pub(super) fn build(
        &mut self,
        lens: &[u8],
        budget: Option<&Budget>,
        keep: bool,
    ) -> Result<(), Error> {
        let syms = std::mem::take(&mut self.syms);
        *self = Code::new();
        self.syms = syms;
        for &l in lens {
            self.count[usize::from(l)] += 1;
        }
        let n = lens.len() - usize::from(self.count[0]);
        self.count[0] = 0;
        if n == 0 {
            return Err(Error::PrefixCode);
        }
        if n == 1 {
            for (s, &l) in lens.iter().enumerate() {
                if l != 0 {
                    self.single = true;
                    self.sym = s as u16;
                }
            }
            return Ok(());
        }
        let mut left = 1i32;
        for l in 1..16 {
            left = (left << 1) - i32::from(self.count[l]);
            if left < 0 {
                return Err(Error::PrefixCode);
            }
        }
        if left != 0 {
            return Err(Error::PrefixCode);
        }
        if !keep {
            return Ok(()); // only checked
        }
        if let Some(b) = budget {
            fit_zeroed(b, &mut self.syms, n)?;
        } else {
            self.syms.clear();
            self.syms
                .try_reserve_exact(n)
                .map_err(|_| Error::OutOfMemory)?;
            self.syms.resize(n, 0);
        }
        let mut off = [0u16; 16];
        for l in 1..15 {
            off[l + 1] = off[l] + self.count[l];
        }
        let mut next = [0u32; 16]; // next code of each length
        let mut cd = 0u32;
        for l in 1..16 {
            cd = (cd + u32::from(self.count[l - 1])) << 1;
            next[l] = cd;
        }
        for (s, &l) in lens.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = usize::from(l);
            self.syms[usize::from(off[l])] = s as u16;
            off[l] += 1;
            let cd = next[l];
            next[l] += 1;
            if l <= 8 {
                let r = reverse(cd, l as u32) as usize;
                let mut j = r;
                while j < 256 {
                    self.root[j] = (s as u16) << 4 | l as u16;
                    j += 1 << l;
                }
            }
        }
        Ok(())
    }

    /// Decodes one symbol; a stream that ends sets `r.over`.
    pub(super) fn read(&self, r: &mut BitReader<'_>) -> usize {
        if self.single {
            return usize::from(self.sym);
        }
        if r.n < 15 {
            r.refill();
        }
        let v = r.val as u32;
        let e = self.root[(v & 255) as usize];
        if e != 0 {
            r.skip(u32::from(e & 15));
            return usize::from(e >> 4);
        }
        let (mut cd, mut first, mut index) = (0i32, 0i32, 0i32);
        for l in 1..16u32 {
            cd |= ((v >> (l - 1)) & 1) as i32;
            let cnt = i32::from(self.count[l as usize]);
            if cd - cnt < first {
                r.skip(l);
                return self
                    .syms
                    .get((index + cd - first) as usize)
                    .map_or(0, |&s| usize::from(s));
            }
            index += cnt;
            first = (first + cnt) << 1;
            cd <<= 1;
        }
        r.over = true; // not reached: the code is complete
        0
    }
}

fn reverse(mut v: u32, n: u32) -> u32 {
    let mut r = 0;
    for _ in 0..n {
        r = r << 1 | (v & 1);
        v >>= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_bytes_bounds_the_struct() {
        assert!(std::mem::size_of::<Code>() as u64 <= CODE_BYTES);
    }
}
