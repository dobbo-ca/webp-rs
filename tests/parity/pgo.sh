#!/bin/sh
# Times the bench example built three ways against Go: tests/parity/pgo.sh [N]. Needs `oracle` (Go)
# and `bench` (the default build), `bench-pgo` (cargo-pgo, trained on /bench) and, when BOLT was
# available, `bench-bolt` (PGO then BOLT) on PATH, and the inputs in /bench; run from the repo root
# (stage `pgo` of .github/ci/Dockerfile, workflow pgo.yml). Each build runs tests/parity/bench.sh
# (its Rust binary is $BENCH; the v4 columns are off); the tables give the Rust ms/Mpixel of each
# per class, for the default threads and for one thread, and the change against the default build.
set -eu
export LC_ALL=C
n=${1:-40}
t=$(mktemp -d)
have=
labels=
for v in bench:default bench-pgo:PGO bench-bolt:"PGO + BOLT"; do
  b=${v%%:*}
  if ! command -v "$b" >/dev/null 2>&1; then
    echo >&2 "no $b: skipped"
    continue
  fi
  BENCH=$b BENCH_V4= sh tests/parity/bench.sh "$n" >"$t/$b.md"
  have="$have $t/$b.md"
  labels="$labels,${v#*:}"
done
[ -n "$have" ] || { echo >&2 "no bench binary on PATH"; exit 1; }
# Rust ms/Mpixel is column 6 of a class row; a pass starts at its "### " line of bench.sh.
# shellcheck disable=SC2086
awk -F'|' -v labels="${labels#,}" '
FNR == 1 { nf++; p = 0 }
/^### / { p++; pass[p] = substr($0, 5) }
/^[|] [a-z0-9_ -]+ [|] [0-9]+ [|]/ {
  name = $2; sub(/^ +/, "", name); sub(/ +$/, "", name)
  v[nf, p, name] = $6 + 0
  if (!((p, name) in seen)) { seen[p, name] = 1; names[p, ++cnt[p]] = name }
}
END {
  split(labels, lab, ",")
  for (p = 1; p <= 2; p++) {
    print "#### Rust ms/Mpixel, " pass[p] " (lower is better; the change is against the default build)"
    print ""
    s = "| class |"; r = "|---|"
    for (f = 1; f <= nf; f++) { s = s " " lab[f] " |"; r = r "---|"; if (f > 1) { s = s " change |"; r = r "---|" } }
    print s; print r
    for (i = 1; i <= cnt[p]; i++) {
      k = names[p, i]
      s = "| " k " |"
      for (f = 1; f <= nf; f++) {
        s = s sprintf(" %.3f |", v[f, p, k])
        if (f > 1 && v[1, p, k] > 0) s = s sprintf(" %+.1f%% |", (v[f, p, k] / v[1, p, k] - 1) * 100)
      }
      print s
    }
    print ""
  }
}' $have
for f in $have; do
  echo "<details><summary>$(basename "$f" .md): the full bench.sh tables</summary>"
  echo
  cat "$f"
  echo
  echo "</details>"
done
rm -rf "$t"
