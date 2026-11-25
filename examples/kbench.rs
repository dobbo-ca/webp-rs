//! `kbench [iters]`: times every SIMD kernel on each tier this CPU runs (scalar, AVX2, AVX-512,
//! NEON) on deterministic xorshift inputs of a realistic size (768 pixel rows, one macroblock of
//! 4x4 blocks) and prints a markdown table of the ns a call takes, the best of three runs of
//! `iters` calls (default 20000), and the ratio avx512 / avx2 (below 1 is a win for AVX-512). A
//! kernel without an AVX-512 form runs the AVX2 one in that column. See docs/safety.md.

fn main() {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000);
    let tiers = webp::kbench_api::tiers();
    let mut cols = vec!["scalar", "avx2", "avx512"];
    if tiers.contains(&"neon") {
        cols.push("neon");
    }
    let mut head = String::from("| kernel |");
    let mut rule = String::from("|---|");
    for c in &cols {
        head += &format!(" {c} ns |");
        rule += "---:|";
    }
    println!("{head} avx512/avx2 |\n{rule}---:|");
    for name in webp::kbench_api::KERNELS {
        let ns: Vec<Option<f64>> = cols
            .iter()
            .map(|t| {
                (0..3)
                    .filter_map(|_| webp::kbench_api::run(name, t, iters))
                    .reduce(f64::min)
            })
            .collect();
        let mut row = format!("| {name} |");
        for v in &ns {
            match v {
                Some(v) => row += &format!(" {v:.1} |"),
                None => row += " - |",
            }
        }
        match (ns.get(1).copied().flatten(), ns.get(2).copied().flatten()) {
            (Some(a), Some(b)) if a > 0.0 => row += &format!(" {:.2} |", b / a),
            _ => row += " - |",
        }
        println!("{row}");
    }
}
