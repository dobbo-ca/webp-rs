//! Ports of webp-go's internal/vp8l tests: the libwebp ALPH oracle rows (tests/data/alph122.tsv) and
//! the round trip of the simple writer.

use super::*;
use crate::limits::{Limits, with_ledger};
use crate::testutil::{pcg::Pcg, read_lines, sha256};
use legacy::Legacy;

/// A synthetic alpha plane for the libwebp ALPH oracle.
struct Plane {
    name: String,
    w: usize,
    h: usize,
    data: Vec<u8>,
}

/// xorshift64, fixed so the oracle inputs never drift.
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn n(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const TIE_SEEDS: [u64; 31] = [
    11, 94, 112, 124, 126, 140, 160, 180, 192, 193, 250, 253, 255, 263, 264, 294, 306, 322, 542,
    579, 981, 1002, 1060, 1082, 1127, 3556, 4537, 4836, 4869, 4908, 4930,
];

/// Plane tie-<seed> of the tie search: the seed picks the size (4..64 a side), colours (2..255)
/// and family.
fn tie_plane(seed: u64) -> Plane {
    let mut r = Rand((seed + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    r.next();
    let w = 4 + r.n(61);
    let h = 4 + r.n(61);
    let mut k = 2 + r.n(15); // half take the palette path
    if r.n(2) == 0 {
        k = 17 + r.n(239);
    }
    let fam = r.n(3);
    let gx = r.n(8);
    let gy = r.n(8);
    let a = r.n(k);
    let nz = a / (1 + r.n(16));
    let base: Vec<usize> = (0..64).map(|_| r.n(k)).collect();
    let mut d = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let j = match fam {
                0 => r.n(k),
                1 => (x * gx + y * gy) / 4 + r.n(nz + 1),
                _ => base[[x, y][gx & 1]] + r.n(nz + 1),
            };
            d[y * w + x] = (j % k * 254 / (k - 1)) as u8; // never 255
        }
    }
    Plane {
        name: format!("tie-{seed}"),
        w,
        h,
        data: d,
    }
}

/// The oracle planes, covering the branches of the port plan.
fn planes() -> Vec<Plane> {
    let mut r = Rand(0x9e37_79b9_7f4a_7c15);
    let mut ps = Vec::new();
    fn add(
        ps: &mut Vec<Plane>,
        name: String,
        w: usize,
        h: usize,
        mut f: impl FnMut(usize, usize) -> usize,
    ) {
        let mut d = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                d[y * w + x] = f(x, y) as u8;
            }
        }
        ps.push(Plane {
            name,
            w,
            h,
            data: d,
        });
    }
    let pal = |n: usize| -> Vec<usize> {
        (0..n)
            .map(|i| i * 254 / n.saturating_sub(1).max(1))
            .collect() // never all 255
    };
    let rows =
        |r: &mut Rand, n: usize, k: usize| -> Vec<usize> { (0..n).map(|_| r.n(k)).collect() };

    // <= 16 colours: palette path
    for n in [1usize, 2, 3, 4, 5, 16] {
        let p = pal(n);
        add(&mut ps, format!("pal{n}"), 128, 96, |_, _| p[r.n(n)]);
        add(&mut ps, format!("pal{n}-blocks"), 200, 150, |x, y| {
            p[(x / 13 + y / 7 * 3) % n]
        });
    }
    add(&mut ps, "pal-0-1-255".into(), 96, 96, |_, _| {
        [0, 1, 255][r.n(3)]
    });
    add(&mut ps, "pal-runs".into(), 256, 128, |x, y| {
        let i = y * 256 + x;
        if i < 9000 || (20000..25000).contains(&i) {
            return 0;
        }
        [0, 200][r.n(2)]
    });

    // 17-192 colours: filter estimate and trials
    let rv = rows(&mut r, 256, 100);
    let cv = rows(&mut r, 256, 100);
    add(&mut ps, "h-wins".into(), 256, 256, |_, y| rv[y]);
    add(&mut ps, "v-wins".into(), 256, 256, |x, _| cv[x]);
    add(&mut ps, "g-wins".into(), 256, 256, |x, y| {
        rv[y] * 9 / 10 + cv[x] * 9 / 10
    });
    add(&mut ps, "h-noisy".into(), 200, 200, |_, y| rv[y] + r.n(3));
    add(&mut ps, "v-noisy".into(), 200, 200, |x, _| cv[x] + r.n(3));
    add(&mut ps, "g-noisy".into(), 200, 200, |x, y| {
        rv[y] * 9 / 10 + cv[x] * 9 / 10 + r.n(2)
    });
    for s in [[4usize, 4], [5, 4], [8, 3], [6, 6], [17, 1], [9, 2]] {
        add(
            &mut ps,
            format!("tie-{}x{}", s[0], s[1]),
            s[0],
            s[1],
            |x, y| 100 + (x + y * s[0]) % 17,
        );
    }
    add(&mut ps, "w3".into(), 3, 200, |_, _| 50 + r.n(60));
    add(&mut ps, "h2".into(), 200, 2, |_, _| 50 + r.n(60));
    add(&mut ps, "c100-noise".into(), 160, 160, |_, _| r.n(100));

    // > 192 colours: spatial predictors and histogram clustering
    add(&mut ps, "noise256".into(), 256, 256, |_, _| r.n(256));
    add(&mut ps, "grad".into(), 512, 512, |x, y| {
        ((x + 2 * y) / 3) & 255
    });
    add(&mut ps, "grad-noisy".into(), 512, 384, |x, y| {
        ((x + y) / 4 + r.n(5)) & 255
    });
    let tb = rows(&mut r, 32 * 32, 256);
    add(&mut ps, "many-histo".into(), 1024, 1024, |x, y| {
        let t = tb[y / 32 * 32 + x / 32];
        (t + r.n(1 + t % 64)) & 255
    });
    add(&mut ps, "few-histo".into(), 512, 512, |x, _| {
        if x < 256 {
            return r.n(256);
        }
        120 + r.n(8)
    });
    add(&mut ps, "big1700".into(), 1700, 1700, |x, y| {
        ((x * 3 + y * 5) / 16 + r.n(4)) & 255
    });
    add(&mut ps, "dominant".into(), 400, 400, |_, _| {
        if r.n(100) < 85 {
            return 0;
        }
        r.n(256)
    });

    // raw wins (header 0x00, filtered raw 0x04/0x08/0x0c)
    add(&mut ps, "raw1x1".into(), 1, 1, |_, _| 7);
    add(&mut ps, "raw1x2".into(), 1, 2, |_, y| 7 + 9 * y); // odd ALPH length
    add(&mut ps, "raw16".into(), 16, 16, |_, _| r.n(256));
    add(&mut ps, "raw3x5".into(), 3, 5, |_, _| r.n(256));
    let rh = rows(&mut r, 64, 120);
    for s in [[16usize, 16], [24, 12], [32, 8]] {
        add(
            &mut ps,
            format!("fraw-h{}x{}", s[0], s[1]),
            s[0],
            s[1],
            |_, y| rh[y] + r.n(60),
        );
        add(
            &mut ps,
            format!("fraw-v{}x{}", s[0], s[1]),
            s[0],
            s[1],
            |x, _| rh[x] + r.n(60),
        );
    }

    // size ties for the strict picks
    for s in TIE_SEEDS {
        ps.push(tie_plane(s));
    }
    ps
}

/// Compares `alpha` with the rows of a libwebp ALPH oracle file: name w h m hdr sha256 len.
fn oracle(file: &str, alpha: impl Fn(&[u8], usize, usize, usize) -> Vec<u8>) {
    let planes = planes();
    let mut rows = 0;
    for line in read_lines(file) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() || f[0] == "name" {
            continue;
        }
        assert_eq!(f.len(), 7, "bad row {line:?}");
        let (w, h, m): (usize, usize, usize) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        );
        let hdr = u8::from_str_radix(f[4].trim_start_matches("0x"), 16).unwrap();
        let n: usize = f[6].parse().unwrap();
        let p = planes
            .iter()
            .find(|p| p.name == f[0] && p.w == w && p.h == h)
            .unwrap_or_else(|| panic!("{}: no {w}x{h} plane", f[0]));
        let got = alpha(&p.data, w, h, m);
        assert!(
            got[0] == hdr && got.len() == n && sha256(&got) == f[5],
            "{} m{m}: hdr {:#04x} len {}, want {hdr:#04x} len {n} (sha {})",
            f[0],
            got[0],
            got.len(),
            f[5]
        );
        rows += 1;
    }
    assert!(rows > 0);
}

#[test]
fn lw_alpha_oracle() {
    oracle("alph122.tsv", |a, w, h, m| {
        with_ledger(|cx| Lw::new(false, cx).alpha(Vec::new(), a, w, h, m).unwrap())
    });
}

#[test]
fn lw16_alpha_oracle() {
    oracle("alph160.tsv", |a, w, h, m| {
        with_ledger(|cx| Lw::new(true, cx).alpha(Vec::new(), a, w, h, m).unwrap())
    });
}

/// The simple writer's stream, wrapped in a VP8L file, decodes to the same plane.
#[test]
fn alpha_round_trip() {
    let mut r = Pcg::new(1, 2);
    let sizes = [
        (1, 1),
        (1, 9),
        (9, 1),
        (2, 2),
        (16, 16),
        (77, 31),
        (770, 770),
        (1, 5000),
        (5000, 1),
    ];
    let budget = crate::limits::Budget::new(&Limits::default());
    let never = std::sync::atomic::AtomicBool::new(false);
    let cx = Ledger::new(&budget, &never);
    let mut e = Legacy::new(&cx);
    for kind in ["opaque", "clear", "noise", "skew", "holes", "ramp"] {
        for &(w, h) in &sizes {
            let mut a = vec![0u8; w * h];
            match kind {
                "opaque" => a.fill(255),
                "clear" => {}
                "noise" => a.iter_mut().for_each(|v| *v = r.uint32() as u8),
                // Geometric values skew counts past 15-bit codes.
                "skew" => {
                    for v in a.iter_mut() {
                        let mut n = 0u8;
                        while n < 40 && r.int_n(2) == 0 {
                            n += 1;
                        }
                        *v = n;
                    }
                }
                "holes" => {
                    for y in 0..h {
                        for x in 0..w {
                            a[y * w + x] = 255;
                            if (x / 13 + y / 7) % 3 == 0 {
                                a[y * w + x] = 0;
                            }
                            if (x + y) % 97 == 0 {
                                a[y * w + x] = x as u8;
                            }
                        }
                    }
                }
                _ => a
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = (i / 3) as u8),
            }
            let s = e.alpha(Vec::new(), &a, w).unwrap();
            // The VP8L header: signature, size, no alpha, version 0.
            let v = (w as u32 - 1) | ((h as u32 - 1) << 14);
            let mut file = vec![
                0x2f,
                v as u8,
                (v >> 8) as u8,
                (v >> 16) as u8,
                (v >> 24) as u8,
            ];
            file.extend_from_slice(&s);
            let mut riff = Vec::new();
            riff.extend_from_slice(b"RIFF");
            riff.extend_from_slice(&((4 + 8 + file.len() + (file.len() & 1)) as u32).to_le_bytes());
            riff.extend_from_slice(b"WEBPVP8L");
            riff.extend_from_slice(&(file.len() as u32).to_le_bytes());
            riff.extend_from_slice(&file);
            if file.len() & 1 == 1 {
                riff.push(0);
            }
            let img = crate::decode(
                &riff,
                &Limits::default(),
                &std::sync::atomic::AtomicBool::new(false),
            )
            .unwrap_or_else(|e| panic!("{kind} {w}x{h}: {e}"));
            let g: Vec<u8> = (0..w * h).map(|i| img.pix[4 * i + 1]).collect();
            assert_eq!(g, a, "{kind} {w}x{h}");
        }
    }
}

#[test]
fn prefix_bits_codes() {
    for (v, want) in [
        (1usize, (0, 0, 0)),
        (4, (3, 0, 0)),
        (5, (4, 1, 0)),
        (6, (4, 1, 1)),
        (7, (5, 1, 0)),
    ] {
        assert_eq!(prefix_bits(v), want, "{v}");
    }
}

/// A cancel flag set from another thread once an encoder has started stops it inside a row or a
/// pass, not at the end of the plane: planes that take seconds to encode return `Cancelled` soon
/// after, and the charges are all given back.
#[test]
fn alph_encoders_check_cancel() {
    use crate::limits::Budget;
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    let noise = |w: usize, h: usize| -> Vec<u8> {
        let mut r = Rand(0x1234_5678_9abc_def1);
        (0..w * h).map(|_| (r.next() >> 33) as u8).collect()
    };
    // The simple writer is some 4x faster, so its plane is bigger.
    let (lw_plane, legacy_plane) = (
        (2048, 1024, noise(2048, 1024)),
        (16384, 2048, noise(16384, 2048)),
    );
    for (name, (w, h, a)) in [
        ("lw 1.2.2", &lw_plane),
        ("lw 1.6.0", &lw_plane),
        ("legacy", &legacy_plane),
    ] {
        let budget = Budget::new(&Limits::default());
        let cancel = AtomicBool::new(false);
        let (res, took) = std::thread::scope(|s| {
            s.spawn(|| {
                // Once the encoder has charged its first buffer it is under way.
                while budget.peak() == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                cancel.store(true, Relaxed);
            });
            let cx = Ledger::new(&budget, &cancel);
            let t = Instant::now();
            let res = match name {
                "lw 1.2.2" => Lw::new(false, &cx).alpha(Vec::new(), a, *w, *h, 5),
                "lw 1.6.0" => Lw::new(true, &cx).alpha(Vec::new(), a, *w, *h, 5),
                _ => Legacy::new(&cx).alpha(vec![1], a, *w),
            };
            (res, t.elapsed())
        });
        assert_eq!(res.err(), Some(crate::Error::Cancelled), "{name}");
        assert!(
            took < Duration::from_secs(3),
            "{name}: took {took:?} to stop"
        );
        assert_eq!(budget.held(), 0, "{name}: charges left after the stop");
    }
}

/// The `Lw` scratch an `AlphaScratch` keeps stays charged after the ledger drops: it stays
/// allocated until the call ends, while other threads of the call may still be charging.
#[test]
fn kept_alpha_scratch_stays_charged() {
    use crate::limits::Budget;
    let budget = Budget::new(&Limits::default());
    let never = std::sync::atomic::AtomicBool::new(false);
    let mut r = Rand(0x9e37_79b9_7f4a_7c15);
    let a: Vec<u8> = (0..96 * 64).map(|_| (r.next() >> 60) as u8 * 17).collect();
    let mut scr = AlphaScratch::default();
    {
        let cx = Ledger::new(&budget, &never);
        scr.encode(&cx, true, false, &a, 96, 64, 4).unwrap();
    }
    let kept = scr.lw.bytes();
    assert!(kept > 0);
    assert!(budget.held() >= kept, "{} held, {kept} kept", budget.held());
}
