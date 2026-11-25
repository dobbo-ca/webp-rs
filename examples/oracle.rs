//! The Rust side of the parity check: `oracle <inputs> <outputs>` runs every input like
//! `tests/parity/go/oracle` and writes `<name>.<op>.out` (the bytes) or `<name>.<op>.err` (the error
//! class). The ops and the class table are in `docs/parity.md`.

mod common;

use common::{class, digest, encode, limits, parse_enc};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::{env, fs, process};
use webp::Limits;

const MAX_FILE: usize = 16 << 20;

fn must<T, E: std::fmt::Display>(r: Result<T, E>) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("oracle: {e}");
        process::exit(1)
    })
}

type Res = Result<Vec<u8>, String>;

/// Writes the results of one input. A mutation (`~` in the name) writes the digest of each
/// output, not the bytes.
struct Emit {
    dir: PathBuf,
    name: String,
    mutation: bool,
}

impl Emit {
    fn put(&self, op: &str, r: Res) {
        let m = self.mutation;
        self.write(op, r.map(|b| if m { digest(&b) } else { b }));
    }

    /// `put` with the digest always.
    fn putd(&self, op: &str, r: Res) {
        self.write(op, r.map(|b| digest(&b)));
    }

    fn write(&self, op: &str, r: Res) {
        let base = self.dir.join(format!("{}.{op}", self.name));
        match r {
            Ok(b) => must(fs::write(format!("{}.out", base.display()), b)),
            Err(c) => must(fs::write(
                format!("{}.err", base.display()),
                format!("{c}\n"),
            )),
        }
    }
}

fn pix_out(i: &webp::Image) -> Vec<u8> {
    let mut b = (i.width as u32).to_le_bytes().to_vec();
    b.extend((i.height as u32).to_le_bytes());
    b.extend_from_slice(&i.pix);
    b
}

fn planar_out(p: &webp::Planar, cancel: &AtomicBool) -> Res {
    let (w, h) = (p.width, p.height);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut b = (w as u32).to_le_bytes().to_vec();
    b.extend((h as u32).to_le_bytes());
    for y in 0..h {
        b.extend_from_slice(&p.y()[y * p.y_stride()..][..w]);
    }
    for pl in [p.cb(), p.cr()] {
        for y in 0..ch {
            b.extend_from_slice(&pl[y * p.c_stride()..][..cw]);
        }
    }
    match &p.alpha {
        Some(a) => {
            b.push(1);
            b.extend_from_slice(a);
        }
        None => b.push(0),
    }
    let mut rgb = vec![0u8; 3 * w * h];
    p.rgb(&mut rgb, 0, h, &limits(), cancel).map_err(class)?;
    b.extend_from_slice(&rgb);
    Ok(b)
}

/// One line per chunk: kind, index, truncated, and the digest of its bytes.
fn meta_out(m: webp::Meta) -> Vec<u8> {
    let mut b = Vec::new();
    for (kind, chunks) in [("exif", &m.exif), ("xmp", &m.xmp)] {
        for (i, c) in chunks.iter().enumerate() {
            b.extend(format!("{kind} {i} {} ", c.truncated).into_bytes());
            b.extend(digest(c.data));
        }
    }
    b
}

fn webp_ops(e: &Emit, data: &[u8]) {
    let cancel = AtomicBool::new(false);
    let cfg = webp::decode_config(data, &limits());
    e.put(
        "cfg",
        cfg.clone()
            .map(|c| {
                format!(
                    "{} {} {} {} {}\n",
                    c.width, c.height, c.has_alpha, c.lossless, c.frames_dropped
                )
                .into_bytes()
            })
            .map_err(class),
    );
    e.put(
        "dec",
        webp::decode(data, &limits(), &cancel)
            .map(|i| pix_out(&i))
            .map_err(class),
    );
    e.put(
        "planar",
        match webp::decode_planar(data, &limits(), &cancel) {
            Ok(p) => planar_out(&p, &cancel),
            Err(err) => Err(class(err)),
        },
    );
    e.put(
        "meta",
        webp::metadata(data, &Limits::default())
            .map(meta_out)
            .map_err(class),
    );
    if e.mutation {
        return;
    }
    // The limits: one pixel under the image fails, exactly the image passes.
    if let Ok(c) = cfg {
        let n = (c.width * c.height) as i64;
        for d in [-1i64, 0] {
            if n + d < 1 {
                continue;
            }
            let l = Limits {
                max_pixels: (n + d) as u64,
                ..Limits::default()
            };
            e.putd(
                &format!("decl{d:+}"),
                webp::decode(data, &l, &cancel)
                    .map(|i| pix_out(&i))
                    .map_err(class),
            );
        }
    }
    for n in [33u64, 100] {
        let l = Limits {
            max_metadata_bytes: n,
            ..Limits::default()
        };
        e.put(
            &format!("metal{n}"),
            webp::metadata(data, &l).map(meta_out).map_err(class),
        );
    }
}

fn enc_ops(e: &Emit, data: &[u8]) {
    let r = match parse_enc(data) {
        Some(c) => encode(&c, &limits()),
        None => Err("bad-input".to_string()),
    };
    e.put("enc", r);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let [_, inputs, outputs] = args.as_slice() else {
        eprintln!("usage: oracle <inputs> <outputs>");
        process::exit(2);
    };
    must(fs::create_dir_all(outputs));
    let mut names: Vec<String> = must(fs::read_dir(inputs))
        .map(|e| must(e).file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let is_webp = name.ends_with(".webp");
        if !is_webp && !name.ends_with(".enc") {
            continue;
        }
        let data = must(fs::read(format!("{inputs}/{name}")));
        if data.len() > MAX_FILE {
            continue;
        }
        let e = Emit {
            dir: PathBuf::from(outputs),
            mutation: name.contains('~'),
            name,
        };
        if is_webp {
            webp_ops(&e, &data);
        } else {
            enc_ops(&e, &data);
        }
    }
}
