# CI image: every check runs inside this image under BuildKit (docker/setup-buildx-action on a GitHub-hosted runner).
# Dependencies (rayon) are fetched in the toolchain stage, before the source is copied, and every later
# stage builds offline from that layer (so the cargo registry is not a cache mount).
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu
WORKDIR /src
# The manifests and lockfiles of the crate and of the fuzz crate, with stub targets, so the fetch is
# a layer that only a change to them rebuilds.
COPY Cargo.toml Cargo.lock ./
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in $(sed -n 's#^path = "fuzz_targets/\(.*\)\.rs"$#\1#p' tests/fuzz/Cargo.toml); do echo 'fn main() {}' > tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo fetch --locked && cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml
ENV CARGO_NET_OFFLINE=true

FROM toolchain AS src
COPY . .

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The target dirs are cache mounts, so they persist across builds.
FROM src AS ci
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo fmt --check
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo clippy --locked --all-targets -- -D warnings
# Lint the aarch64 target too: the NEON modules are not compiled for x86_64.
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo test --locked
RUN --mount=type=cache,id=webp-target,target=/src/target \
    WEBP_SIMD=scalar cargo test --locked
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo build --locked --release --target aarch64-unknown-linux-gnu
# The encoder under the v4 build of the bench (same flags as stage `bench`): its output must be the
# same bytes, which tests/encode.rs checks against libwebp's. Its own target cache, since RUSTFLAGS
# changes every artifact; the sources are touched first, as the mount is shared by every branch.
RUN --mount=type=cache,id=webp-target-v4,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && if grep -qw avx512bw /proc/cpuinfo; then RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo test --locked --release --test encode; else echo "no AVX-512 on this runner: v4 encode test skipped"; fi

# aarch64 tests on a native arm64 runner, both SIMD tiers (NEON, then WEBP_SIMD=scalar): only the
# tests that the target changes. The `simd::` unit tests hold the NEON dispatch test and the NEON
# against scalar kernel test; the `fmath::` tests include arch_hash, which checks that pow/exp/log
# give the same bits on every arch (fused multiply-add on aarch64); tests/encode.rs runs the
# encoder, which calls the kernels, and checks its output bytes against libwebp's. The other unit
# tests and the alpha and decoder tests reach the kernels only through the encoder, which
# tests/encode.rs already covers. The full suite runs on x86_64 in `test`.
FROM src AS arm64
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo test --locked --lib -- simd:: fmath:: \
    && cargo test --locked --test encode
RUN --mount=type=cache,id=webp-target,target=/src/target \
    WEBP_SIMD=scalar cargo test --locked --lib -- simd:: fmath:: \
    && WEBP_SIMD=scalar cargo test --locked --test encode

# Go parity: the Go oracle writes the inputs and runs them; the Rust example runs the same inputs;
# the two output trees must be equal.
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
RUN --mount=type=cache,id=webp-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle .
# The same oracle built for x86-64-v4 (the bench's v4 column); the parity stages use /out/oracle.
RUN --mount=type=cache,id=webp-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && GOAMD64=v4 go build -o /out/oracle-v4 .

FROM go-oracle AS go-run
COPY tests/data tests/data
# Inputs: the fixtures and their mutations, junk files, the encoder cases (docs/parity.md).
RUN /out/oracle -gen tests/data /inputs && ls /inputs | wc -l && du -sh /inputs
RUN /out/oracle /inputs /go-out && ls /go-out | wc -l && du -sh /go-out
# The subset the aarch64 oracle runs (stage parity-arm64): every .webp that is not a
# mutation (`~` in its name) and every 20th mutation, and every 4th .enc under 200 kB. Go runs it
# again, so the two trees have the same names.
RUN mkdir /sub && ls /inputs | awk '/\.webp$/ { if ($0 !~ /~/ || ++m % 20 == 0) print } /\.enc$/ { if (++e % 4 == 0) print }' \
    | (cd /inputs && xargs cp -t /sub) \
    && find /sub -name '*.enc' -size +200k -delete \
    && /out/oracle /sub /go-sub && ls /sub | wc -l && ls /go-sub | wc -l

FROM src AS parity
# The guards fail the job if the trees are empty or short of the kinds of output, so a diff of
# two empty trees cannot pass. The scalar tier (WEBP_SIMD=scalar) runs the subset of parity-arm64.
# The Go trees are rw bind mounts, not COPY --from: BuildKit keys a rw mount by the source stage's
# cache key, not by a hash of its content, and the Rust trees are deleted in the same RUN.
RUN --mount=type=cache,id=webp-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs,rw --mount=type=bind,from=go-run,source=/go-out,target=/go-out,rw \
    --mount=type=bind,from=go-run,source=/sub,target=/sub,rw --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub,rw \
    cargo run --locked --release --example oracle -- /inputs /rust-out \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs" \
    && [ "$(ls /go-out | wc -l)" -ge 50000 ] \
    && [ "$(ls /go-out | grep -c '\.dec\.out$')" -ge 5000 ] \
    && [ "$(ls /go-out | grep -c '\.enc\.out$')" -ge 500 ] \
    && { diff -rq /go-out /rust-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && WEBP_SIMD=scalar cargo run --locked --release --example oracle -- /sub /rust-out-scalar \
    && { diff -rq /go-sub /rust-out-scalar >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && rm -rf /rust-out /rust-out-scalar

# aarch64 against Go: the oracle example run on a native arm64 runner (the NEON kernels) on the
# subset, once per SIMD tier; the output must equal Go's.
FROM src AS parity-arm64
RUN --mount=type=cache,id=webp-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/sub,target=/sub,rw --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub,rw \
    cargo run --locked --release --example oracle -- /sub /arm-out \
    && echo "$(ls /sub | wc -l) inputs, $(ls /go-sub | wc -l) outputs" \
    && [ "$(ls /go-sub | wc -l)" -ge 5000 ] \
    && [ "$(ls /go-sub | grep -c '\.enc\.out$')" -ge 50 ] \
    && { diff -rq /go-sub /arm-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && WEBP_SIMD=scalar cargo run --locked --release --example oracle -- /sub /arm-out-scalar \
    && { diff -rq /go-sub /arm-out-scalar >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && rm -rf /arm-out /arm-out-scalar

# Fuzzing (fuzz.yml): nightly and cargo-fuzz are installed before `COPY . .`, so a new commit does not
# install them again; the targets are built into the image. The workflow runs it with `docker run`,
# so a BuildKit cache hit cannot skip the fuzzing.
# cargo-fuzz is not in the fetched layer, so this one install may use the network.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && CARGO_NET_OFFLINE=false cargo install cargo-fuzz --locked --version 0.13.1

FROM fuzz-tools AS fuzz
COPY . .
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz

# Benchmark (bench.yml): the Go oracle, its inputs and the Rust bench example in one image.
FROM go-oracle AS go-bench
COPY tests/data tests/data
RUN /out/oracle -genbench tests/data /bench && ls /bench | wc -l

FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
COPY --from=go-bench /bench /bench
RUN --mount=type=cache,id=webp-target,target=/src/target \
    cargo build --locked --release --example bench --example kbench \
    && cp target/release/examples/bench target/release/examples/kbench /usr/local/bin/
# Its own cache mount (RUSTFLAGS changes every artifact); the sources are touched first, as in `ci`,
# since the mount is shared by every branch.
RUN --mount=type=cache,id=webp-target-v4,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --example bench \
    && cp target/release/examples/bench /usr/local/bin/bench-v4

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on /bench (single-threaded, then with the default threads), rebuilds it
# with the profile, and, when BOLT is installed, once more through BOLT. The tools are installed
# before `COPY . .`, so a new commit does not install them again. BOLT comes from apt.llvm.org
# (bookworm has none of its own); if the repository or the package is missing the image still
# builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && CARGO_NET_OFFLINE=false cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle and /bench; tests/parity/pgo.sh times them. The profiles of an
# earlier run stay in the cache mount, so they are deleted first.
FROM pgo-tools AS pgo
COPY . .
COPY --from=go-bench /bench /bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
RUN --mount=type=cache,id=webp-target-pgo,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- /bench 40 1 \
    && cargo pgo run -- --locked --example bench -- /bench 40 \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" /bench 40 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi
