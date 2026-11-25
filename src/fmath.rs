//! `Pow` as Go's `math.Pow` computes it on amd64 with FMA, bit for bit (port of webp-go's
//! internal/fmath, which follows the Go 1.27 pow.go, exp_amd64.s and log_amd64.s).
//!
//! Go fuses `x*y+z` only on some architectures; `f64::mul_add` is the explicit FMA where the amd64
//! assembly fuses, and every other product is a plain Rust multiply, which is never fused.
//!
//! Copyright 2009 The Go Authors. Use of this source code is governed by a BSD-style license, which
//! NOTICE carries.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

pub(crate) fn frexp(f: f64) -> (f64, i32) {
    if f == 0.0 || f.is_infinite() || f.is_nan() {
        return (f, 0);
    }
    let (f, mut exp) = normalize(f);
    let mut x = f.to_bits();
    exp += ((x >> 52) & 0x7ff) as i32 - 1023 + 1;
    x &= !(0x7ffu64 << 52);
    x |= 1022u64 << 52;
    (f64::from_bits(x), exp)
}

fn normalize(x: f64) -> (f64, i32) {
    if x.abs() < 2.2250738585072014e-308 {
        return (x * (1u64 << 52) as f64, -52);
    }
    (x, 0)
}

fn ldexp(frac: f64, exp: i32) -> f64 {
    if frac == 0.0 || frac.is_infinite() || frac.is_nan() {
        return frac;
    }
    let (frac, e) = normalize(frac);
    let mut exp = exp + e;
    let mut x = frac.to_bits();
    exp += ((x >> 52) & 0x7ff) as i32 - 1023;
    if exp < -1075 {
        return 0.0f64.copysign(frac);
    }
    if exp > 1023 {
        return if frac < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let mut m = 1.0;
    if exp < -1022 {
        exp += 53;
        m = 1.0 / (1u64 << 53) as f64;
    }
    x &= !(0x7ffu64 << 52);
    x |= ((exp + 1023) as u64) << 52;
    m * f64::from_bits(x)
}

fn is_odd_int(x: f64) -> bool {
    if x.abs() >= (1u64 << 53) as f64 {
        return false;
    }
    let xi = x.trunc();
    x - xi == 0.0 && (xi as i64) & 1 == 1
}

/// `math.Pow` on amd64 with FMA.
pub(crate) fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if y == 1.0 {
        return x;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        if y < 0.0 {
            if x.is_sign_negative() && is_odd_int(y) {
                return f64::NEG_INFINITY;
            }
            return f64::INFINITY;
        }
        if y > 0.0 {
            if x.is_sign_negative() && is_odd_int(y) {
                return x;
            }
            return 0.0;
        }
    }
    if y.is_infinite() {
        if x == -1.0 {
            return 1.0;
        }
        if (x.abs() < 1.0) == (y > 0.0) {
            return 0.0;
        }
        return f64::INFINITY;
    }
    if x.is_infinite() {
        if x < 0.0 {
            return pow(1.0 / x, -y);
        }
        if y < 0.0 {
            return 0.0;
        }
        if y > 0.0 {
            return f64::INFINITY;
        }
    }
    if y == 0.5 {
        return x.sqrt();
    }
    if y == -0.5 {
        return 1.0 / x.sqrt();
    }

    let ya = y.abs();
    let mut yi = ya.trunc();
    let mut yf = ya - yi;
    if yf != 0.0 && x < 0.0 {
        return f64::NAN;
    }
    if yi >= 9.223372036854775808e18 {
        if x == -1.0 {
            return 1.0;
        }
        if (x.abs() < 1.0) == (y > 0.0) {
            return 0.0;
        }
        return f64::INFINITY;
    }

    let mut a1 = 1.0f64;
    let mut ae = 0i64;
    if yf != 0.0 {
        if yf > 0.5 {
            yf -= 1.0;
            yi += 1.0;
        }
        a1 = exp(yf * log(x));
    }
    let (mut x1, xe) = frexp(x);
    let mut xe = i64::from(xe);
    let mut i = yi as i64;
    while i != 0 {
        if !(-(1 << 12)..=(1 << 12)).contains(&xe) {
            ae += xe;
            break;
        }
        if i & 1 == 1 {
            a1 *= x1;
            ae += xe;
        }
        x1 *= x1;
        xe <<= 1;
        if x1 < 0.5 {
            x1 += x1;
            xe -= 1;
        }
        i >>= 1;
    }
    if y < 0.0 {
        a1 = 1.0 / a1;
        ae = -ae;
    }
    ldexp(
        a1,
        ae.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
    )
}

/// `exp_amd64.s`'s AVX+FMA path.
fn exp(x: f64) -> f64 {
    const LOG2E: f64 = 1.4426950408889634073599246810018920;
    const LN2U: f64 = 0.69314718055966295651160180568695068359375;
    const LN2L: f64 = 0.28235290563031577122588448175013436025525412068e-12;
    const OVERFLOW: f64 = 7.09782712893384e+02;
    let b = x.to_bits();
    if b & !(1u64 << 63) >= 0x7FF0_0000_0000_0000 {
        if b == 0xFFF0_0000_0000_0000 {
            return 0.0;
        }
        return x; // NaN or +Inf
    }
    if x > OVERFLOW {
        return f64::INFINITY;
    }
    let k = cvtsd2sl(LOG2E * x);
    let kf = f64::from(k);
    let mut x = (-kf).mul_add(LN2U, x);
    x = (-kf).mul_add(LN2L, x);
    x *= 0.0625;
    let mut p: f64 = 2.4801587301587301587e-5;
    for c in [
        1.9841269841269841270e-4,
        1.3888888888888888889e-3,
        8.3333333333333333333e-3,
        4.1666666666666666667e-2,
        1.6666666666666666667e-1,
        0.5,
        1.0,
    ] {
        p = p.mul_add(x, c);
    }
    x *= p;
    for _ in 0..3 {
        p = x + 2.0;
        x *= p;
    }
    p = x + 2.0;
    x = x.mul_add(p, 1.0);
    // fr * 2**k, as the asm's ldexp
    let bx = i64::from(k) + 0x3FF;
    if bx <= 0 {
        if bx < -52 {
            return 0.0;
        }
        x *= f64::from_bits(((bx + 0x3FE) as u64) << 52);
        return x * f64::from_bits(1u64 << 52);
    }
    if bx >= 0x7FF {
        return f64::INFINITY;
    }
    x * f64::from_bits((bx as u64) << 52)
}

/// CVTSD2SL: round to nearest even, `i32::MIN` if out of range.
fn cvtsd2sl(t: f64) -> i32 {
    let r = t.round_ties_even();
    if !(r >= f64::from(i32::MIN) && r <= f64::from(i32::MAX)) {
        return i32::MIN;
    }
    r as i32
}

/// `log_amd64.s`, its Frexp from the exponent bits included.
fn log(x: f64) -> f64 {
    const HSQRT2: f64 = 7.07106781186547524401e-01;
    const LN2HI: f64 = 6.93147180369123816490e-01;
    const LN2LO: f64 = 1.90821492927058770002e-10;
    const L1: f64 = 6.666666666666735130e-01;
    const L2: f64 = 3.999999999940941908e-01;
    const L3: f64 = 2.857142874366239149e-01;
    const L4: f64 = 2.222219843214978396e-01;
    const L5: f64 = 1.818357216161805012e-01;
    const L6: f64 = 1.531383769920937332e-01;
    const L7: f64 = 1.479819860511658591e-01;
    let b = x.to_bits();
    if b & !(1u64 << 63) == 0 {
        return f64::NEG_INFINITY;
    }
    if (b as i64) < 0 {
        return f64::from_bits(0x7FF8_0000_0000_0001);
    }
    if b >= 0x7FF0_0000_0000_0000 {
        return x; // +Inf or NaN
    }
    let mut f1 = f64::from_bits(b & 0x000F_FFFF_FFFF_FFFF | 0x3FE0_0000_0000_0000);
    let mut k = f64::from(((b >> 52 & 0x7FF) as i32) - 0x3FE);
    if HSQRT2 >= f1 {
        k -= 1.0;
        f1 *= 2.0;
    }
    let f = f1 - 1.0;
    let s = f / (2.0 + f);
    let s2 = s * s;
    let s4 = s2 * s2;
    let mut t1 = L7 * s4 + L5;
    t1 = t1 * s4 + L3;
    t1 = t1 * s4 + L1;
    t1 *= s2;
    let mut t2 = L6 * s4 + L4;
    t2 = t2 * s4 + L2;
    t2 *= s4;
    let r = t1 + t2;
    let hfsq = (0.5 * f) * f;
    let u = s * (hfsq + r) + LN2LO * k;
    k * LN2HI - ((hfsq - u) - f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{pcg::Pcg, sha256};

    /// From amd64 with FMA (c7i); every arch must match.
    const WANT_HASH: &str = "5f676f1a9394002cba89923b2aba4898c12b453ffa682db1e2de08d192eca188";

    /// webp-go's Pow arguments (quality curve, segment exponent, gamma tables), a wide random
    /// spread, and edge cases.
    fn inputs() -> (Vec<f64>, Vec<f64>) {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let mut add = |x: f64, y: f64| {
            xs.push(x);
            ys.push(y);
        };
        let amp = 9.0 / 2560.0; // 0.9 * 50 / 100 / 128
        for q in 0..=100 * 64 {
            // quality steps of 1/64
            let c = f64::from(q) / 6400.0;
            let lin = if c < 0.75 {
                c * (2.0 / 3.0)
            } else {
                2.0 * c - 1.0
            };
            add(lin, 1.0 / 3.0);
            let c0 = pow(lin, 1.0 / 3.0);
            for a in -127..=127 {
                add(c0, 1.0 - amp * f64::from(a));
            }
        }
        for v in 0..256 {
            add(f64::from(v) * (1.0 / 255.0), 0.8);
        }
        for v in 0..33 {
            add(f64::from(v) * (128.0 / 4095.0), 1.25);
        }
        let mut rng = Pcg::new(5, 6);
        for _ in 0..300_000 {
            let f1 = rng.float64();
            let e1 = rng.int_n(60) as i32 - 30;
            let f2 = rng.float64();
            let e2 = rng.int_n(8) as i32 - 2;
            add(f1 * ldexp(1.0, e1), (f2 * 2.0 - 1.0) * ldexp(1.0, e2));
        }
        for k in -60..=60 {
            // log's f1 == Sqrt2/2 branch
            add(ldexp(7.07106781186547524401e-01, k), 1.5);
        }
        for x in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            1e-310,
            1e308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            for y in [
                0.0,
                1.0,
                -1.0,
                0.5,
                -0.5,
                2.0,
                3.0,
                -3.0,
                1.0 / 3.0,
                0.8,
                1e10,
                -1e10,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                add(x, y);
            }
        }
        (xs, ys)
    }

    #[test]
    fn arch_hash() {
        let (xs, ys) = inputs();
        let mut buf = Vec::with_capacity(24 * xs.len());
        let mut put = |v: f64| {
            // NaN payloads differ by arch; Go's math.NaN() is 0x7FF8000000000001.
            let b = if v.is_nan() {
                0x7FF8_0000_0000_0001
            } else {
                v.to_bits()
            };
            buf.extend_from_slice(&b.to_le_bytes());
        };
        for (i, &x) in xs.iter().enumerate() {
            put(pow(x, ys[i]));
            put(exp(ys[i] * 20.0));
            put(log(x));
        }
        assert_eq!(sha256(&buf), WANT_HASH);
    }
}
