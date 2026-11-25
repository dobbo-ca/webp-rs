//! `bench <dir> <n>`: prints name, pixels and ns per operation for each input of dir, as
//! `oracle -bench` does for Go: a decode for a `.webp` that decodes, an encode for a `.enc` that
//! encodes with one `Encoder` kept for all inputs. Each runs once to time it, then as many times as
//! fit in about 200 ms, between 1 and n. The optional third argument is `max_threads` for the Limits; absent, the default, with
//! the crate's shared pool (the counterpart of Go's default parallelism), and 1 runs everything
//! on the calling thread. See `tests/parity/bench.sh`.

mod common;

use common::{encode_with, limits, parse_enc};
use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use std::{env, fs, io};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (dir, n, threads) = match args.as_slice() {
        [_, dir, n] => (dir, n, None),
        [_, dir, n, t] => (dir, n, t.parse::<usize>().ok()),
        _ => {
            eprintln!("usage: bench <dir> <n> [threads]");
            std::process::exit(2);
        }
    };
    let n: u32 = n.parse().unwrap_or(1).max(1);
    let mut limits = limits();
    if let Some(t) = threads {
        limits.max_threads = t;
    }
    let cancel = AtomicBool::new(false);
    // One Encoder for all the .enc inputs, as Go's oracle -bench reuses one webp.Encoder.
    let encoder = RefCell::new(webp::Encoder::new());
    let mut names: Vec<String> = fs::read_dir(dir)?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<io::Result<_>>()?;
    names.sort();
    for name in names {
        let data = fs::read(format!("{dir}/{name}"))?;
        let (px, op): (usize, Box<dyn Fn() -> bool + '_>) = if name.ends_with(".webp") {
            let Ok(c) = webp::decode_config(&data, &limits) else {
                continue;
            };
            (
                c.width * c.height,
                Box::new(|| webp::decode(&data, &limits, &cancel).is_ok()),
            )
        } else if name.ends_with(".enc") {
            let Some(c) = parse_enc(&data) else {
                continue;
            };
            let (l, enc) = (&limits, &encoder);
            (
                c.width * c.height,
                Box::new(move || encode_with(&mut enc.borrow_mut(), &c, l).is_ok()),
            )
        } else {
            continue;
        };
        let t = Instant::now();
        if !op() {
            continue;
        }
        let one = t.elapsed();
        let iters = if one.is_zero() {
            n
        } else {
            (Duration::from_millis(200).as_nanos() / one.as_nanos()).clamp(1, u128::from(n)) as u32
        };
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(op());
        }
        println!(
            "{name}\t{px}\t{}",
            t.elapsed().as_nanos() / u128::from(iters)
        );
    }
    Ok(())
}
