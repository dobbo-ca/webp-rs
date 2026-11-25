#!/bin/sh
# Time of the decodes and encodes of /bench, Go and Rust on the same inputs, as a markdown table:
# tests/parity/bench.sh [N]. Needs `oracle` (Go) and `bench` (Rust) on PATH and the inputs in /bench
# (`oracle -genbench tests/data /bench`). Each side runs an operation as many times as fit in about
# 200 ms, at most N, and keeps the best of five runs: the machine is shared. Two tables: the
# default parallelism of both (Go GOMAXPROCS, Rust max_threads), then both single-threaded. When
# `oracle-v4` (GOAMD64=v4) and `bench-v4` (x86-64-v4) are on PATH too, they are timed the same way
# and add three columns. $BENCH is the Rust binary (default bench; pgo.sh sets it) and $BENCH_V4 the
# v4 one (default bench-v4; set empty, it is not timed).
set -eu
export LC_ALL=C
n=${1:-50}
d=${BENCH_DIR:-/bench}
bench=${BENCH:-bench}
bench4=${BENCH_V4-bench-v4}
v4=
if [ -n "$bench4" ] && command -v oracle-v4 >/dev/null 2>&1 && command -v "$bench4" >/dev/null 2>&1; then v4=1; fi
tab=$(printf '\t')
grep -m1 'model name' /proc/cpuinfo >&2 || true
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|bmi2|avx512(f|bw|vl|_vnni))$' | tr '\n' ' ' >&2 || true
echo >&2
best() {
  awk -F'\t' '!($1 in m) || $3 < m[$1] { m[$1] = $3; b[$1] = $2 } END { for (k in m) printf "%s\t%s\t%s\n", k, b[k], m[k] }' "$@" | sort
}
# run LABEL GOMAXPROCS THREADS: Go with GOMAXPROCS=$2 (empty: its default), Rust with threads $3
# (empty: its default).
run() {
  for i in 1 2 3 4 5; do
    GOMAXPROCS=$2 oracle -bench "$n" $d >/tmp/go$i.tsv
    $bench $d "$n" $3 >/tmp/rs$i.tsv
    if [ -n "$v4" ]; then
      GOMAXPROCS=$2 oracle-v4 -bench "$n" $d >/tmp/v4go$i.tsv
      $bench4 $d "$n" $3 >/tmp/v4rs$i.tsv
    fi
  done
  best /tmp/go?.tsv >/tmp/go.tsv
  best /tmp/rs?.tsv >/tmp/rs.tsv
  # Both sides must have timed the same files: join drops a file one side skipped, which would
  # make the "files" counts equal by construction.
  cut -f1 /tmp/go.tsv >/tmp/go.names
  cut -f1 /tmp/rs.tsv >/tmp/rs.names
  if [ ! -s /tmp/go.names ] || [ ! -s /tmp/rs.names ] || ! cmp -s /tmp/go.names /tmp/rs.names; then
    echo "bench.sh: Go and Rust timed different files (a side skipped one or has none); only in Go, then only in Rust:" >&2
    comm -23 /tmp/go.names /tmp/rs.names >&2
    echo "--" >&2
    comm -13 /tmp/go.names /tmp/rs.names >&2
    exit 1
  fi
  if [ -n "$v4" ]; then
    best /tmp/v4go?.tsv >/tmp/v4go.tsv
    best /tmp/v4rs?.tsv >/tmp/v4rs.tsv
    cut -f1 /tmp/v4go.tsv >/tmp/v4go.names
    cut -f1 /tmp/v4rs.tsv >/tmp/v4rs.names
    if ! cmp -s /tmp/go.names /tmp/v4go.names || ! cmp -s /tmp/go.names /tmp/v4rs.names; then
      echo "bench.sh: the v4 builds timed different files than the default builds" >&2
      exit 1
    fi
  fi
  echo "### $1"
  # Columns: name, Go pixels, Go ns, Rust pixels, Rust ns, then with v4 Go v4 pixels and ns, Rust v4 pixels and ns.
  join -t "$tab" /tmp/go.tsv /tmp/rs.tsv >/tmp/joined.tsv
  if [ -n "$v4" ]; then
    join -t "$tab" /tmp/joined.tsv /tmp/v4go.tsv | join -t "$tab" - /tmp/v4rs.tsv >/tmp/joined4.tsv
    mv /tmp/joined4.tsv /tmp/joined.tsv
  fi
  awk -F'\t' -v v4="$v4" '
  function cls(n,   a, m, f, k) {
    if (n ~ /^dec-/) { split(n, a, "-"); return "decode " a[2] }
    k = (n ~ /^enc-alpha512/) ? "encode alpha " : "encode "
    if (match(n, /_m[0-9]_f[0-9]+_/)) {
      m = substr(n, RSTART + 2, 1); f = substr(n, RSTART + 5) + 0
      return k (f % 2 ? "match_libwebp " : "") "m" m (f >= 2 ? " alpha16" : "")
    }
    return k "other"
  }
  function add(k) { p[k] += $2; g[k] += $3; r[k] += $5; c[k]++; if (v4) { gv[k] += $7; rv[k] += $9 } }
  { add(cls($1)); add("all") }
  END {
    h = "| input class | files | Mpixel | Go ms/Mpixel | Rust ms/Mpixel | Rust / Go |"; u = "|---|---|---|---|---|---|"
    if (v4) { h = h " Go v4 ms/Mpixel | Rust v4 ms/Mpixel | Rust v4 / Go v4 |"; u = u "---|---|---|" }
    print h
    print u
    for (k in p) {
      l = sprintf("| %s | %d | %.2f | %.2f | %.2f | %.2f |", k, c[k], p[k] / 1e6, g[k] / p[k], r[k] / p[k], r[k] / g[k])
      if (v4) l = l sprintf(" %.2f | %.2f | %.2f |", gv[k] / p[k], rv[k] / p[k], rv[k] / gv[k])
      print l | "sort"
    }
  }' /tmp/joined.tsv
  # Per file (the default run only): the Go and Rust ns and the ratio, the worst first (stderr, for
  # reading the log).
  if [ -z "$2" ]; then
    awk -F'\t' '$3 > 0 { printf "%s\t%d\t%d\t%d\t%.2f\n", $1, $2, $3, $5, $5 / $3 }' /tmp/joined.tsv | sort -t"$(printf '\t')" -k5 -nr | head -40 >&2
  fi
}
run "default threads" "" ""
run "single-threaded" 1 1
