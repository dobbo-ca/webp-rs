//! Ports of webp-go's internal/vp8enc tests that run without cwebp or dwebp: the encoder's
//! reconstruction against the decoder, segments, solid frames, the gamma tables.

use super::*;
use crate::decode_planar;
use crate::testutil::{pcg::Pcg, png::decode_png, read};

static NO_CANCEL: AtomicBool = AtomicBool::new(false);

/// An encoder over a budget made from `limits`, which the test leaks.
fn new_enc<'a>(
    w: usize,
    h: usize,
    opts: &Options,
    limits: &Limits,
    cancel: &'a AtomicBool,
) -> Enc<'a> {
    let budget: &'static Budget = Box::leak(Box::new(Budget::new(limits)));
    Enc::new(w, h, opts, budget, cancel)
}

impl Enc<'_> {
    /// `run` into a new `Vec`.
    fn run_vec(&mut self, src: &Source) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        self.run(src, &mut out)?;
        Ok(out)
    }
}

/// `super::encode` into a new `Vec`.
fn encode(
    w: usize,
    h: usize,
    src: &Source,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    bufs: &mut Bufs,
) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    super::encode(w, h, src, opts, limits, cancel, bufs, &mut out)?;
    Ok(out)
}

fn opaque(mut p: Vec<u8>) -> Vec<u8> {
    for a in p.iter_mut().skip(3).step_by(4) {
        *a = 255;
    }
    p
}

/// randNRGBA: random bytes under a random nonzero mask, so the image is not solid (opaque; the
/// ALPH writer is not in this stage).
fn rand_nrgba(rng: &mut Pcg, w: usize, h: usize) -> Vec<u8> {
    let mask = [0xffu8, 0xf0, 0xc0, 0x80][rng.int_n(4)];
    opaque((0..4 * w * h).map(|_| rng.uint32() as u8 & mask).collect())
}

fn pattern(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut m = vec![255u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            let v = f(x, y);
            m[4 * (y * w + x)..4 * (y * w + x) + 3].fill(v);
        }
    }
    m
}

fn load_png_rgba(name: &str) -> (usize, usize, Vec<u8>) {
    let p = decode_png(&read(name));
    assert_eq!(p.channels, 3);
    let mut v = Vec::new();
    for c in p.pix.as_chunks::<3>().0 {
        v.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    (p.width, p.height, v)
}

/// Asserts the decoded planes equal the encoder's reconstruction.
fn assert_recon(e: &Enc, out: &[u8], what: &str) {
    let p = decode_planar(out, &Limits::default(), &NO_CANCEL).unwrap();
    assert_eq!((p.width, p.height), (e.w, e.h), "{what}");
    for y in 0..e.h {
        for x in 0..e.w {
            let (a, b) = (p.y()[y * p.y_stride() + x], e.img.y[y * e.img.ys + x]);
            assert_eq!(a, b, "{what}: Y({x},{y}) decoded {a}, recon {b}");
        }
    }
    for y in 0..e.h.div_ceil(2) {
        for x in 0..e.w.div_ceil(2) {
            let o = y * p.c_stride() + x;
            let r = y * e.img.cs + x;
            assert_eq!(p.cb()[o], e.img.cb[r], "{what}: Cb({x},{y})");
            assert_eq!(p.cr()[o], e.img.cr[r], "{what}: Cr({x},{y})");
        }
    }
}

fn enc_frame<'a>(
    w: usize,
    h: usize,
    pix: &[u8],
    s: &mut Segs,
    limits: &Limits,
    cancel: &'a AtomicBool,
    set_segs: bool,
) -> (Enc<'a>, Vec<u8>) {
    let mut e = new_enc(w, h, &Options::default(), limits, cancel);
    e.load(&Source::Nrgba { pix, stride: 4 * w }).unwrap();
    if set_segs {
        for mby in 0..e.mbh {
            for mbx in 0..e.mbw {
                e.mbs[mby * e.mbw + mbx].seg = ((mbx + mby) & 3) as u8;
            }
        }
        s.set_probs(&mut e.mbs);
    }
    let parts = e.encode_frame(s).unwrap();
    let mut out = Vec::new();
    e.write_riff(&parts, &[], &mut out).unwrap();
    (e, out)
}

#[test]
fn recon_matches_decoder() {
    let mut rng = Pcg::new(1, 2);
    struct Input {
        name: &'static str,
        w: usize,
        h: usize,
        pix: Vec<u8>,
        qi: i32,
    }
    let mut inputs = Vec::new();
    let (w5, h5, k5) = load_png_rgba("kodim05_192.png");
    let (w23, h23, k23) = load_png_rgba("kodim23_192.png");
    // A 17x33 crop at (3, 5).
    let mut crop = Vec::new();
    for y in 5..38 {
        crop.extend_from_slice(&k5[4 * (y * w5 + 3)..4 * (y * w5 + 20)]);
    }
    for qi in [0, 9, 38, 90, 127] {
        inputs.push(Input {
            name: "kodim05",
            w: w5,
            h: h5,
            pix: k5.clone(),
            qi,
        });
        inputs.push(Input {
            name: "kodim23",
            w: w23,
            h: h23,
            pix: k23.clone(),
            qi,
        });
        inputs.push(Input {
            name: "crop17x33",
            w: 17,
            h: 33,
            pix: crop.clone(),
            qi,
        });
    }
    for _ in 0..120 {
        let (w, h) = (1 + rng.int_n(70), 1 + rng.int_n(70));
        let qi = rng.int_n(128) as i32;
        inputs.push(Input {
            name: "random",
            w,
            h,
            pix: rand_nrgba(&mut rng, w, h),
            qi,
        });
    }
    for qi in [0, 40, 127] {
        inputs.push(Input {
            name: "flat",
            w: 40,
            h: 40,
            pix: pattern(40, 40, |_, _| 77),
            qi,
        });
        inputs.push(Input {
            name: "gradient",
            w: 48,
            h: 40,
            pix: pattern(48, 40, |x, y| (3 * x + 2 * y) as u8),
            qi,
        });
        inputs.push(Input {
            name: "hstripes",
            w: 40,
            h: 48,
            pix: pattern(40, 48, |_, y| (y % 3 * 100) as u8),
            qi,
        });
        inputs.push(Input {
            name: "vstripes",
            w: 40,
            h: 40,
            pix: pattern(40, 40, |x, _| (x % 5 * 40) as u8),
            qi,
        });
    }

    let (mut n_i4, mut n_i16, mut n_skip) = (0, 0, 0);
    let limits = Limits::default();
    for i in &inputs {
        let mut s = Segs {
            n: 1,
            quant: [i.qi; 4],
            dq_uv_ac: -4,
            ..Segs::default()
        };
        let (e, out) = enc_frame(i.w, i.h, &i.pix, &mut s, &limits, &NO_CANCEL, false);
        assert_recon(&e, &out, &format!("{} qi {} {}x{}", i.name, i.qi, i.w, i.h));
        for mb in &e.mbs {
            n_skip += usize::from(mb.skip);
            if mb.i16 {
                n_i16 += 1;
            } else {
                n_i4 += 1;
            }
        }
    }
    assert!(
        n_i4 > 0 && n_i16 > 0 && n_skip > 0,
        "i4 {n_i4}, i16 {n_i16}, skip {n_skip} macroblocks; want all nonzero"
    );
}

#[test]
fn segments_decode() {
    let mut rng = Pcg::new(3, 4);
    let pix = rand_nrgba(&mut rng, 64, 48);
    for level in [[0; 4], [0, 10, 20, 63]] {
        let mut s = Segs {
            n: 4,
            quant: [5, 40, 80, 127],
            level,
            dq_uv_ac: -4,
            ..Segs::default()
        };
        let (e, out) = enc_frame(64, 48, &pix, &mut s, &Limits::default(), &NO_CANCEL, true);
        assert!(s.update_map);
        if level == [0; 4] {
            assert_recon(&e, &out, "segments");
        } else {
            let p = decode_planar(&out, &Limits::default(), &NO_CANCEL).unwrap();
            assert_eq!((p.width, p.height), (64, 48));
        }
    }
}

#[test]
fn two_partitions() {
    let mut rng = Pcg::new(5, 6);
    for h in [16, 32, 48] {
        let pix = rand_nrgba(&mut rng, 40, h);
        let mut e = new_enc(40, h, &Options::default(), &Limits::default(), &NO_CANCEL);
        let out = e
            .run_vec(&Source::Nrgba {
                pix: &pix,
                stride: 160,
            })
            .unwrap();
        assert_recon(&e, &out, &format!("mbh {}", h / 16));
    }
}

#[test]
fn recon_matches_decoder_methods() {
    let (w, h, pix) = load_png_rgba("kodim23_192.png");
    for method in 0..=6 {
        let o = Options {
            quality: 70.0,
            method,
            ..Options::default()
        };
        let mut e = new_enc(w, h, &o, &Limits::default(), &NO_CANCEL);
        let out = e
            .run_vec(&Source::Nrgba {
                pix: &pix,
                stride: 4 * w,
            })
            .unwrap();
        // Segment filters are on, so only the size is compared.
        let p = decode_planar(&out, &Limits::default(), &NO_CANCEL).unwrap();
        assert_eq!((p.width, p.height), (w, h), "method {method}");
    }
}

/// A solid frame encodes to its flat colour exactly, whatever the source type.
#[test]
fn solid_exact() {
    let colors: [[u8; 3]; 9] = [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [1, 2, 3],
        [254, 253, 252],
        [12, 200, 99],
        [128, 128, 128],
    ];
    for (w, h) in [
        (1, 1),
        (2, 1),
        (1, 2),
        (15, 17),
        (16, 16),
        (17, 33),
        (265, 770),
    ] {
        // Every colour opaque; at 17x33 also with alpha, as NRGBA and as RGBA.
        let alphas: &[u8] = if (w, h) == (17, 33) {
            &[255, 0, 1, 128, 254]
        } else {
            &[255]
        };
        for c in colors {
            for &a in alphas {
                for premultiplied in [false, true] {
                    let mut pix = Vec::with_capacity(4 * w * h);
                    for _ in 0..w * h {
                        pix.extend_from_slice(&[c[0], c[1], c[2], a]);
                    }
                    let src = if premultiplied {
                        Source::Rgba {
                            pix: &pix,
                            stride: 4 * w,
                        }
                    } else {
                        Source::Nrgba {
                            pix: &pix,
                            stride: 4 * w,
                        }
                    };
                    let mut r = new_enc(w, h, &Options::default(), &Limits::default(), &NO_CANCEL);
                    r.load(&src).unwrap();
                    let want = (r.img.y[0], r.img.cb[0], r.img.cr[0]);

                    let out = encode(
                        w,
                        h,
                        &src,
                        &Options::default(),
                        &Limits::default(),
                        &NO_CANCEL,
                        &mut Bufs::default(),
                    )
                    .unwrap();
                    let p = decode_planar(&out, &Limits::default(), &NO_CANCEL).unwrap();
                    let ctx = format!("{w}x{h} {c:?} a{a} premultiplied {premultiplied}");
                    for y in 0..h {
                        for x in 0..w {
                            assert_eq!(p.y()[y * p.y_stride() + x], want.0, "{ctx}: Y");
                        }
                    }
                    for y in 0..h.div_ceil(2) {
                        for x in 0..w.div_ceil(2) {
                            assert_eq!(p.cb()[y * p.c_stride() + x], want.1, "{ctx}: Cb");
                            assert_eq!(p.cr()[y * p.c_stride() + x], want.2, "{ctx}: Cr");
                        }
                    }
                    if a != 255 {
                        let al = p.alpha.as_ref().unwrap();
                        assert!(al.iter().all(|&v| v == a), "{ctx}: alpha");
                    }
                }
            }
        }
    }
}

/// An NRGBA tile with transparent holes and soft alpha edges.
fn holes(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut r = Pcg::new(seed, 1);
    let mut m = vec![0u8; 4 * w * h];
    for y in 0..h {
        for x in 0..w {
            let a = match (x / 20 + y / 13) % 5 {
                0 => 0,
                1 => (x * 7) as u8,
                _ => 255,
            };
            let o = 4 * (y * w + x);
            m[o..o + 4].copy_from_slice(&[r.uint32() as u8, x as u8, y as u8, a]);
        }
    }
    m
}

/// The default path keeps webp-go's original writer; match_libwebp at methods 3 to 5 uses libwebp's, as
/// 1.6.0 with alpha_libwebp16.
#[test]
fn match_libwebp_alpha_alph() {
    use crate::limits::with_ledger;
    use crate::vp8l::{Lw, legacy::Legacy};
    let (w, h) = (101, 250);
    let src = holes(w, h, 5);
    let a: Vec<u8> = (0..w * h).map(|i| src[4 * i + 3]).collect();
    let old = with_ledger(|cx| Legacy::new(cx).alpha(vec![1], &a, w).unwrap());
    for m in 0..=6 {
        for match_libwebp in [false, true] {
            for alpha_libwebp16 in [false, true] {
                for legacy_alpha in [false, true] {
                    let o = Options {
                        quality: 90.0,
                        method: m,
                        match_libwebp,
                        alpha_libwebp16,
                        legacy_alpha,
                        ..Options::default()
                    };
                    let b = encode(
                        w,
                        h,
                        &Source::Nrgba {
                            pix: &src,
                            stride: 4 * w,
                        },
                        &o,
                        &Limits::default(),
                        &NO_CANCEL,
                        &mut Bufs::default(),
                    )
                    .unwrap();
                    let want = if match_libwebp && (3..=5).contains(&m) && !legacy_alpha {
                        with_ledger(|cx| {
                            Lw::new(alpha_libwebp16, cx)
                                .alpha(Vec::new(), &a, w, h, m as usize)
                                .unwrap()
                        })
                    } else {
                        old.clone()
                    };
                    let n = u32::from_le_bytes([b[34], b[35], b[36], b[37]]) as usize;
                    assert_eq!(
                        &b[38..38 + n],
                        &want[..],
                        "m{m} match {match_libwebp} a16 {alpha_libwebp16} legacy {legacy_alpha}"
                    );
                }
            }
        }
    }
}

/// Quality and method only affect the slow path, so a solid frame is byte-identical either way.
#[test]
fn solid_ignores_options() {
    let pix = vec![255u8; 4 * 770 * 770];
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * 770,
    };
    let run = |quality, method| {
        let o = Options {
            quality,
            method,
            ..Options::default()
        };
        encode(
            770,
            770,
            &src,
            &o,
            &Limits::default(),
            &NO_CANCEL,
            &mut Bufs::default(),
        )
        .unwrap()
    };
    assert_eq!(run(10.0, 0), run(100.0, 6));
}

#[test]
fn gamma_tables() {
    let g = transform::gamma();
    assert_eq!(g.to_linear.iter().sum::<i32>(), 582170);
    assert_eq!(g.to_linear[1], 49);
    assert_eq!(g.to_linear[255], 4095);
    assert_eq!(g.to_gamma.iter().sum::<i32>(), 3756);
    assert_eq!(g.to_gamma[1], 3);
    assert_eq!(g.to_gamma[32], 255);
}

#[test]
fn filter_level_values() {
    for (qi, beta, want) in [
        (127, 0, 63),
        (38, 0, 11),
        (26, 0, 8),
        (9, 0, 3),
        (0, 0, 0),
        (12, 0, 4),
        (10, 100, 2),
        (7, 0, 2),
        (127, 255, 36),
    ] {
        assert_eq!(
            quant::filter_level(qi, beta),
            want,
            "filter_level({qi}, {beta})"
        );
    }
}

#[test]
fn frame_tag_bytes() {
    assert_eq!(
        header::frame_tag(768, 768, 996),
        [0x90, 0x7c, 0x00, 0x9d, 0x01, 0x2a, 0x00, 0x03, 0x00, 0x03]
    );
}

/// The ALPH chunk encodes on a second thread when `max_threads` has one and the image has at least
/// `MIN_JOIN_PIXELS` pixels (this one has), the MB rows on the threads left, never more than
/// `max_threads` threads are live, and the file is the same for every count, on the shared pool or
/// a caller's.
#[test]
fn alpha_thread_within_max_threads() {
    const _: () = assert!(300 * 230 >= crate::limits::MIN_JOIN_PIXELS);
    let (w, h) = (300, 230);
    let src = holes(w, h, 9);
    let lw = Options {
        match_libwebp: true,
        method: 4,
        ..Options::default()
    };
    for (name, o) in [("default", Options::default()), ("libwebp", lw)] {
        let mut want: Option<Vec<u8>> = None;
        for (n, own_pool) in [0usize, 1, 2, 64]
            .into_iter()
            .flat_map(|n| [(n, false), (n, true)])
        {
            let limits = Limits {
                max_threads: n,
                pool: own_pool.then(|| crate::ThreadPool::new(3).unwrap()),
                ..Limits::default()
            };
            let mut e = new_enc(w, h, &o, &limits, &NO_CANCEL);
            let out = e
                .run_vec(&Source::Nrgba {
                    pix: &src,
                    stride: 4 * w,
                })
                .unwrap();
            assert!(e.has_alpha);
            let peak = e.budget.peak_threads();
            assert!(
                peak <= n.max(1) && (n < 2 || peak >= 2),
                "{name}, max_threads {n}, own pool {own_pool}: {peak} threads"
            );
            match &want {
                Some(w0) => assert!(
                    &out == w0,
                    "{name}, max_threads {n}, own pool {own_pool}: output differs"
                ),
                None => want = Some(out),
            }
        }
    }
}

/// With one thread the ledger's peak is the smallest `max_alloc` that passes.
#[test]
fn encode_ledger_peak_is_the_smallest_budget() {
    let (w, h) = (101, 70);
    let src = holes(w, h, 9);
    let run = |max_alloc: usize, o: &Options| {
        let limits = Limits {
            max_alloc,
            max_threads: 1,
            ..Limits::default()
        };
        let mut e = new_enc(w, h, o, &limits, &NO_CANCEL);
        let r = e.run_vec(&Source::Nrgba {
            pix: &src,
            stride: 4 * w,
        });
        (r, e.budget.peak() as usize)
    };
    let lw = Options {
        match_libwebp: true,
        method: 4,
        ..Options::default()
    };
    for (name, o) in [("default", Options::default()), ("libwebp", lw)] {
        let (r, peak) = run(1 << 30, &o);
        r.unwrap();
        assert!(run(peak, &o).0.is_ok(), "{name}: peak {peak} does not pass");
        assert!(
            matches!(run(peak - 1, &o).0, Err(Error::MemoryBudget { .. })),
            "{name}: peak {peak} - 1 passes"
        );
    }
}

/// One input of the reuse tests.
struct Case {
    what: &'static str,
    w: usize,
    h: usize,
    kind: u8, // 0 NRGBA, 1 RGBA, 2 Gray, 3 YCbCr
    pix: Vec<u8>,
    o: Options,
}

impl Case {
    fn src(&self) -> Source<'_> {
        let (w, h) = (self.w, self.h);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        match self.kind {
            0 => Source::Nrgba {
                pix: &self.pix,
                stride: 4 * w,
            },
            1 => Source::Rgba {
                pix: &self.pix,
                stride: 4 * w,
            },
            2 => Source::Gray {
                pix: &self.pix,
                stride: w,
            },
            _ => Source::Ycbcr420 {
                y: &self.pix[..w * h],
                cb: &self.pix[w * h..w * h + cw * ch],
                cr: &self.pix[w * h + cw * ch..],
                y_stride: w,
                c_stride: cw,
            },
        }
    }
}

fn reuse_cases() -> Vec<Case> {
    let lw = |method, f: fn(&mut Options)| {
        let mut o = Options {
            match_libwebp: true,
            method,
            ..Options::default()
        };
        f(&mut o);
        o
    };
    let m = |method| Options {
        method,
        ..Options::default()
    };
    let mut r = Pcg::new(5, 3);
    let c = |what, w, h, kind, pix, o| Case {
        what,
        w,
        h,
        kind,
        pix,
        o,
    };
    let gray = |w: usize, h: usize| (0..w * h).map(|i| (i * 31 % 251) as u8).collect::<Vec<_>>();
    vec![
        c("768 m4", 768, 768, 0, opaque(holes(768, 768, 1)), m(4)),
        c("192 alpha m4", 192, 192, 0, holes(192, 192, 2), m(4)),
        c("1x1 m0", 1, 1, 0, vec![9, 8, 7, 255], m(0)),
        c(
            "300x230 alpha lw m4",
            300,
            230,
            0,
            holes(300, 230, 3),
            lw(4, |_| {}),
        ),
        c(
            "768 lw m6",
            768,
            768,
            0,
            rand_nrgba(&mut r, 768, 768),
            lw(6, |_| {}),
        ),
        c("100 premul RGBA m4", 100, 90, 1, holes(100, 90, 4), m(4)),
        c(
            "alpha16 lw m4",
            300,
            230,
            0,
            holes(300, 230, 5),
            lw(4, |o| o.alpha_libwebp16 = true),
        ),
        c(
            "legacy alpha lw m4",
            300,
            230,
            0,
            holes(300, 230, 6),
            lw(4, |o| o.legacy_alpha = true),
        ),
        c("alpha m6", 200, 120, 0, holes(200, 120, 7), m(6)),
        c("gray m4", 130, 70, 2, gray(130, 70), m(4)),
        c(
            "ycbcr m2",
            131,
            71,
            3,
            gray(131 * 71 + 2 * 66 * 36, 1),
            m(2),
        ),
        c(
            "solid",
            200,
            100,
            0,
            [40u8, 90, 200, 255].repeat(200 * 100),
            m(4),
        ),
        c(
            "segments 1 lw m5",
            150,
            150,
            0,
            rand_nrgba(&mut r, 150, 150),
            lw(5, |o| o.segments = 1),
        ),
        c(
            "768 m4 again",
            768,
            768,
            0,
            opaque(holes(768, 768, 8)),
            m(4),
        ),
        c(
            "alpha lw m3 small",
            20,
            20,
            0,
            holes(20, 20, 9),
            lw(3, |_| {}),
        ),
    ]
}

/// One `Encoder` run over big, small and big images, every method, alpha writer and source kind
/// gives the bytes a fresh `encode` does, in both orders and for every `max_threads`.
#[test]
fn reused_encoder_equals_fresh() {
    let cases = reuse_cases();
    for threads in [0usize, 1, 2] {
        let limits = Limits {
            max_threads: threads,
            ..Limits::default()
        };
        let mut enc = crate::Encoder::new();
        for pass in 0..2 {
            for c in &cases {
                let fresh = crate::encode(c.w, c.h, &c.src(), &c.o, &limits, &NO_CANCEL).unwrap();
                let got = enc
                    .encode(c.w, c.h, &c.src(), &c.o, &limits, &NO_CANCEL)
                    .unwrap();
                assert!(
                    got == fresh,
                    "{}: threads {threads}, pass {pass}: output differs",
                    c.what
                );
            }
        }
    }
}

/// A reused call stays inside `max_alloc` and an encoder that holds more than a small `max_alloc`
/// still encodes what a fresh call encodes under it; an error leaves the encoder usable.
#[test]
fn reused_encoder_respects_max_alloc() {
    let (w, h) = (101, 70);
    let pix = holes(w, h, 9);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let big = opaque(holes(512, 512, 1));
    let bsrc = Source::Nrgba {
        pix: &big,
        stride: 4 * 512,
    };
    let o = Options::default();
    let with = |max_alloc| Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    };
    // The smallest budget a fresh call passes.
    let mut e = new_enc(w, h, &o, &with(1 << 30), &NO_CANCEL);
    let want = e.run_vec(&src).unwrap();
    let small = e.budget.peak() as usize;
    let mut enc = crate::Encoder::new();
    enc.encode(512, 512, &bsrc, &o, &with(1 << 30), &NO_CANCEL)
        .unwrap();
    // Holding a 512x512 working set does not stop a call that fits fresh.
    let got = enc
        .encode(w, h, &src, &o, &with(small), &NO_CANCEL)
        .unwrap();
    assert!(got == want);
    let r = enc.encode(w, h, &src, &o, &with(small - 1), &NO_CANCEL);
    assert!(matches!(r, Err(Error::MemoryBudget { .. })));
    assert!(
        enc.encode(w, h, &src, &o, &with(small), &NO_CANCEL)
            .unwrap()
            == want
    );
    // A reused call at its own size fits the budget a fresh one needs.
    enc.encode(512, 512, &bsrc, &o, &with(1 << 30), &NO_CANCEL)
        .unwrap();
    let mut fresh = new_enc(512, 512, &o, &with(1 << 30), &NO_CANCEL);
    fresh.run_vec(&bsrc).unwrap();
    let need = fresh.budget.peak() as usize;
    assert!(
        enc.encode(512, 512, &bsrc, &o, &with(need), &NO_CANCEL)
            .is_ok()
    );
}

/// Partitions kept from an earlier call are charged at their capacity, as the `Encoder` docs say:
/// a kept partition much larger than the call needs is refused under a small `max_alloc` and
/// accepted under one that covers it, and the bytes are the fresh call's.
#[test]
fn kept_partitions_are_charged() {
    let (w, h) = (64, 48);
    let pix = holes(w, h, 9);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let o = Options::default();
    let with = |max_alloc| Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    };
    let mut e = new_enc(w, h, &o, &with(1 << 30), &NO_CANCEL);
    let want = e.run_vec(&src).unwrap();
    let need = e.budget.peak() as usize;
    let kept = 8 * need;
    for part in 0..3 {
        let mut bufs = Bufs::default();
        bufs.parts[part] = Vec::with_capacity(kept);
        let r = encode(w, h, &src, &o, &with(2 * need), &NO_CANCEL, &mut bufs);
        assert!(
            matches!(r, Err(Error::MemoryBudget { .. })),
            "partition {part}: {:?}",
            r.err()
        );
        let mut bufs = Bufs::default();
        bufs.parts[part] = Vec::with_capacity(kept);
        let r = encode(w, h, &src, &o, &with(need + kept), &NO_CANCEL, &mut bufs).unwrap();
        assert!(r == want, "partition {part}: output differs");
    }
}

/// The libwebp state is charged at the capacity of its buffers, a kept one's too, so a call that
/// fits fresh fits with it kept.
#[test]
fn lw_state_is_charged_at_capacity() {
    use super::libwebp::LwState;
    let s = LwState::new(100, 7, 75.0);
    assert_eq!(LwState::bytes(Some(&s)), LwState::bytes(None));

    let (w, h) = (101, 70);
    let pix = holes(w, h, 9);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let big = opaque(holes(512, 512, 1));
    let bsrc = Source::Nrgba {
        pix: &big,
        stride: 4 * 512,
    };
    let o = Options {
        match_libwebp: true,
        method: 4,
        ..Options::default()
    };
    let with = |max_alloc| Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    };
    let mut fresh = new_enc(w, h, &o, &with(1 << 30), &NO_CANCEL);
    let want = fresh.run_vec(&src).unwrap();
    let small = fresh.budget.peak() as usize;
    // Each call gets an encoder warmed by the large image: after a call the encoder keeps that
    // call's buffers, and a second call of the same size grows none of them, so it peaks below a
    // fresh call (no old and new block live at once) and is not what this tests.
    let warm = || {
        let mut enc = crate::Encoder::new();
        enc.encode(512, 512, &bsrc, &o, &with(1 << 30), &NO_CANCEL)
            .unwrap();
        enc
    };
    let got = warm()
        .encode(w, h, &src, &o, &with(small), &NO_CANCEL)
        .unwrap();
    assert!(got == want);
    let r = warm().encode(w, h, &src, &o, &with(small - 1), &NO_CANCEL);
    assert!(matches!(r, Err(Error::MemoryBudget { .. })));
}

/// The limits of the retry tests: `max_alloc` and one thread.
fn one_thread(max_alloc: usize) -> Limits {
    Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    }
}

/// A noisy w by h NRGBA image, its fresh output and the fresh call's peak.
fn fresh_peak(w: usize, h: usize, seed: u64, o: &Options) -> (Vec<u8>, Vec<u8>, usize) {
    let pix = rand_nrgba(&mut Pcg::new(seed, 1), w, h);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let mut fresh = new_enc(w, h, o, &one_thread(1 << 30), &NO_CANCEL);
    let want = fresh.run_vec(&src).unwrap();
    let peak = fresh.budget.peak() as usize;
    (pix, want, peak)
}

/// A mid-grey solid frame records no tokens and keeps no planes, but keeps the per-macroblock
/// arrays and the partitions, which the next call charges: that call still fits whenever a fresh
/// encode would.
#[test]
fn retry_after_a_mid_grey_solid_frame() {
    let o = Options::default();
    let (w, h) = (100, 70);
    let (pix, want, peak) = fresh_peak(w, h, 7, &o);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let grey = vec![130u8; 1024 * 1024];
    let gsrc = Source::Gray {
        pix: &grey,
        stride: 1024,
    };
    let mut enc = crate::Encoder::new();
    enc.encode(1024, 1024, &gsrc, &o, &one_thread(1 << 30), &NO_CANCEL)
        .unwrap();
    let got = enc
        .encode(w, h, &src, &o, &one_thread(peak), &NO_CANCEL)
        .unwrap();
    assert!(got == want);
}

/// A call that fails at the planes keeps the per-macroblock arrays it got; the next call still
/// fits whenever a fresh encode would.
#[test]
fn retry_after_a_failure_at_the_planes() {
    let o = Options::default();
    let (w, h) = (256, 256);
    let pix = rand_nrgba(&mut Pcg::new(8, 1), w, h);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    // Room for the per-macroblock arrays and not for the luma plane.
    let n = (w / 16) * (h / 16);
    let mbs = n * size_of::<MbInfo>() + (h / 16 + n / 96 + 2) * size_of::<Piece>();
    let mut enc = crate::Encoder::new();
    let r = enc.encode(w, h, &src, &o, &one_thread(mbs), &NO_CANCEL);
    assert!(matches!(r, Err(Error::MemoryBudget { .. })));

    // 192 MBs to the kept 256: the arrays are kept and charged at their capacity.
    let (w, h) = (256, 192);
    let (pix, want, peak) = fresh_peak(w, h, 9, &o);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let got = enc
        .encode(w, h, &src, &o, &one_thread(peak), &NO_CANCEL)
        .unwrap();
    assert!(got == want);
}

/// Partitions kept from a large solid frame (grey and not) are bigger than a small noisy image
/// needs in partition 0 and smaller in the token partitions, which grow. Either way each
/// partition is charged at what it holds: after the call the budget holds the same less the
/// partitions as a fresh call does. An `Encoder` so warmed fits the fresh peak with the fresh
/// bytes.
#[test]
fn kept_partitions_that_grow_are_charged() {
    let o = Options::default();
    let (w, h) = (256, 256);
    let (pix, want, peak) = fresh_peak(w, h, 11, &o);
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    // The bytes held after a call with `parts` kept, and the partitions' capacities then.
    let held = |parts: [Vec<u8>; 3]| {
        let mut bufs = Bufs {
            parts,
            ..Bufs::default()
        };
        let mut e = new_enc(w, h, &o, &one_thread(1 << 30), &NO_CANCEL);
        e.take(&mut bufs);
        assert!(e.run_vec(&src).unwrap() == want);
        let caps = e.pbufs.each_ref().map(Vec::capacity);
        (e.budget.held(), caps)
    };
    let (fresh, fcaps) = held(Default::default());
    let fsum: usize = fcaps.iter().sum();
    let solid = [200u8, 50, 50, 255].repeat(1024 * 1024);
    let grey = vec![130u8; 1024 * 1024];
    let warm = [
        Source::Nrgba {
            pix: &solid,
            stride: 4 * 1024,
        },
        Source::Gray {
            pix: &grey,
            stride: 1024,
        },
    ];
    for (i, wsrc) in warm.iter().enumerate() {
        let mut bufs = Bufs::default();
        encode(
            1024,
            1024,
            wsrc,
            &o,
            &one_thread(1 << 30),
            &NO_CANCEL,
            &mut bufs,
        )
        .unwrap();
        let kept = std::mem::take(&mut bufs.parts);
        let kcaps = kept.each_ref().map(Vec::capacity);
        assert!(kcaps[0] > fcaps[0], "warm {i}: partition 0 is not larger");
        assert!(kcaps[1] < fcaps[1], "warm {i}: partition 1 does not grow");
        let (got, caps) = held(kept);
        let sum: usize = caps.iter().sum();
        assert_eq!(
            got + fsum as u64,
            fresh + sum as u64,
            "warm {i}: the partitions are not charged at what they hold"
        );

        let mut enc = crate::Encoder::new();
        enc.encode(1024, 1024, wsrc, &o, &one_thread(1 << 30), &NO_CANCEL)
            .unwrap();
        let got = enc
            .encode(w, h, &src, &o, &one_thread(peak), &NO_CANCEL)
            .unwrap();
        assert!(got == want, "warm {i}: output differs");
    }
}

/// The thread configurations of the wavefront tests: `max_threads` 1, 2, 3, 4 and 8 on the shared
/// pool, and 8 on a caller's pool of 1 and of 3 threads.
fn thread_configs() -> Vec<Limits> {
    let mut v: Vec<Limits> = [1, 2, 3, 4, 8]
        .into_iter()
        .map(|n| Limits {
            max_threads: n,
            ..Limits::default()
        })
        .collect();
    for p in [1, 3] {
        v.push(Limits {
            max_threads: 8,
            pool: Some(crate::ThreadPool::new(p).unwrap()),
            ..Limits::default()
        });
    }
    v
}

/// The MB rows run as a wavefront with the same bytes for every thread count and pool, never on
/// more than `max_threads` threads: every method, both paths, one and four segments, with and
/// without alpha, from one MB to 768x512. The sizes take in widths of one to three MBs (where a
/// row reuses the ring slot of the row two above while that row is still running), more workers
/// than rows, and libwebp refreshes inside a row (200x136: every 96 MBs, 768x512: every 193).
/// Under `cfg(test)` every frame of two MB rows or more gets a worker per row.
#[test]
fn wavefront_is_the_same_for_every_thread_count() {
    let configs = thread_configs();
    let opts = |method, match_libwebp, segments| Options {
        quality: 80.0,
        method,
        match_libwebp,
        segments,
        ..Options::default()
    };
    let mut cases = Vec::new();
    for (w, h) in [
        (1, 1),
        (16, 16),
        (17, 33),
        (16, 70),
        (30, 70),
        (40, 50),
        (200, 136),
    ] {
        for method in 0..=6 {
            for lw in [false, true] {
                for segments in [1, 4] {
                    for alpha in [false, true] {
                        cases.push((w, h, alpha, opts(method, lw, segments)));
                    }
                }
            }
        }
    }
    for method in 0..=6 {
        for lw in [false, true] {
            cases.push((768, 512, false, opts(method, lw, 4)));
        }
    }
    cases.push((768, 512, true, opts(4, true, 1)));
    for (w, h, alpha, o) in &cases {
        let (w, h) = (*w, *h);
        let pix = if *alpha {
            holes(w, h, 3)
        } else {
            opaque(holes(w, h, 3))
        };
        let src = Source::Nrgba {
            pix: &pix,
            stride: 4 * w,
        };
        let mut want: Option<Vec<u8>> = None;
        for l in &configs {
            let mut e = new_enc(w, h, o, l, &NO_CANCEL);
            let out = e.run_vec(&src).unwrap();
            let what = format!(
                "{w}x{h} alpha {alpha} m{} lw {} segments {}, max_threads {}, pool {:?}",
                o.method,
                o.match_libwebp,
                o.segments,
                l.max_threads,
                l.pool.as_ref().map(|p| p.threads())
            );
            let peak = e.budget.peak_threads();
            assert!(peak <= l.max_threads, "{what}: {peak} threads");
            if l.max_threads >= 2 && h >= 512 {
                assert!(peak >= 2, "{what}: the rows ran on one thread");
            }
            match &want {
                Some(w0) => assert!(out == *w0, "{what}: output differs"),
                None => want = Some(out),
            }
        }
    }
}

/// Garbage in a worker's prediction workspace, levels, quantizer and libwebp scratch at the start
/// of every row changes no byte: nothing an earlier row left there is read, so a row may run on
/// any worker after any other row. Where a libwebp segment resumes a row (200x136 refreshes every
/// 96 MBs, inside a row), all but the workspace the carry restores is poisoned too.
#[test]
fn stale_worker_scratch_is_never_read() {
    for (w, h) in [(17, 33), (40, 50), (200, 136)] {
        let pix = opaque(holes(w, h, 4));
        let src = Source::Nrgba {
            pix: &pix,
            stride: 4 * w,
        };
        for method in 0..=6 {
            for match_libwebp in [false, true] {
                for segments in [1, 4] {
                    let o = Options {
                        quality: 60.0,
                        method,
                        match_libwebp,
                        segments,
                        ..Options::default()
                    };
                    let run = |max_threads: usize, poison: bool| {
                        wave::POISON.set(poison);
                        let limits = Limits {
                            max_threads,
                            ..Limits::default()
                        };
                        let r = encode(w, h, &src, &o, &limits, &NO_CANCEL, &mut Bufs::default());
                        wave::POISON.set(false);
                        r.unwrap()
                    };
                    let want = run(1, false);
                    for threads in [1, 4] {
                        assert!(
                            run(threads, true) == want,
                            "{w}x{h} m{method} lw {match_libwebp} segments {segments}, \
                             {threads} threads: poisoned scratch changes the output"
                        );
                    }
                }
            }
        }
    }
}

/// A cancel flag set before the MB pass ends it with `Error::Cancelled` on every thread count,
/// with no worker left waiting.
#[test]
fn wavefront_cancel() {
    let (w, h) = (200, 136);
    let pix = opaque(holes(w, h, 5));
    let src = Source::Nrgba {
        pix: &pix,
        stride: 4 * w,
    };
    let stop = AtomicBool::new(true);
    for l in thread_configs() {
        for match_libwebp in [false, true] {
            let o = Options {
                match_libwebp,
                ..Options::default()
            };
            let mut e = new_enc(w, h, &o, &l, &stop);
            e.load(&src).unwrap();
            let mut s = e.one_segment(75.0);
            assert!(matches!(e.encode_frame(&mut s), Err(Error::Cancelled)));
        }
    }
}
