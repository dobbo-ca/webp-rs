# Third-party dependencies

One crate, `rayon`, with what it pulls in (`rayon-core`, `crossbeam-deque`, `crossbeam-epoch`,
`crossbeam-utils`, `either`; exact versions in `Cargo.lock`). All are pure Rust: no C, no `-sys`
crate, no build script that compiles anything. It is here only for `ThreadPool`: safe Rust cannot
reuse worker threads for jobs that borrow short-lived buffers (`std::thread::scope` must start
and join its threads on every call, tens of microseconds each, which is more than the work of a
small image), and rayon's scoped pool does. It backs the caller's `Limits::pool` and the
crate's lazy shared pool (see [limits.md](limits.md)).
