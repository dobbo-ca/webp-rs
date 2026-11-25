# Threading

From the `Limits` rules ([limits.md](limits.md)):

- `max_threads: 1` runs everything on the calling thread and never touches a pool (0 is treated
  as 1). Above 1, `decode` and `decode_planar` read the ALPH chunk of a lossy image while the VP8
  frame decodes, `decode` and `Planar::rgb` convert bands of a large image to RGB on several
  threads, and `encode` of an image with alpha encodes the ALPH chunk while the VP8 frame is
  encoded (libwebp does the same). `encode` also runs the segment analysis in bands of
  macroblock rows and the macroblock pass as a wavefront over the rows (see below). The output
  is byte for byte the same for every `max_threads` and every pool. A caller limited to 2 CPUs and 1 GiB sets `max_threads: 2` and
  `max_alloc: 768 << 20`.
- The other threads of a call are the workers of a `ThreadPool`, which are started once and kept,
  so handing them a task costs microseconds, not the tens of microseconds that starting a thread
  does. Small inputs therefore use fewer workers instead of none: the ALPH chunk gets a worker
  from 4K pixels (64 by 64), a band of RGB is at least 64 KiB of output (tens of
  microseconds of work), and the encoder's macroblock rows get workers from 64 macroblocks and two
  rows; a smaller image stays on the calling thread. A call takes at most
  `max_threads - 1` workers besides the calling thread, for all its stages together, and no more
  than the pool has.
  - `Limits::pool` is a pool the caller made with `ThreadPool::new(threads)` and keeps; it always
    wins. Clones of a `ThreadPool` share the threads.
  - Without one, a call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock`
    inside the crate, built on the first such call with `available_parallelism` threads and never
    more, and holding no other state. A call with `max_threads` 1 never builds or touches it, and
    neither does a call with a caller's pool. If the shared pool cannot be built (the OS refuses
    the threads), the call runs on the calling thread alone, and so does every later call
    without a pool of its own: the failure is kept for the life of the process. The shared pool is
    built at the first stage that has work for a second thread, so `metadata`, a lossless decode,
    a small opaque encode or a small image never builds it.
  - A panic on a pool thread is caught and returned as `Error::Internal`.

## The encoder's wavefront

The VP8 encoder encodes each macroblock row on one worker, left to right. Macroblock (x, y)
starts once row y - 1 has finished macroblock x + 1 (all of it in the last column): that is all
it reads of the row above, through the reconstructed pixels above and to the top right, the
nonzero contexts and the i4 modes of the macroblock above, and on the libwebp path the chroma DC
errors diffused down. The rows pass these on through a ring of two slots per column, and each
row publishes how far it has got in an atomic counter (Release after the slot is written, Acquire
before it is read). So a macroblock sees the same inputs as in a serial pass, whatever thread runs
it and whenever. One thread runs the same code, row after row, with no pool.

- Workers take the rows in order, so a row is only ever held by a running worker and the lowest
  unfinished row never waits: a waiting worker spins briefly, then yields. A worker never waits on
  a pool task. An error, a cancel or a panic in one row stops the others, which leave their
  waits; the error of the lowest row is returned, as a serial pass would. `cancel` is checked at
  the start of each row.
- A frame takes at most one worker per row and per two macroblocks of width (the rows in flight
  are two macroblocks apart), and only from 64 macroblocks and two rows.
- Each worker records its tokens into its own buffer, charged against `max_alloc` as it grows, in
  pieces of a row each. The pieces are coded in row order, which is the serial order; the
  probabilities come from counts, which do not depend on the order.
- The libwebp path refreshes its probabilities and costs every so many macroblocks (an eighth of
  the frame, at least 96) from the statistics of all the macroblocks before. It runs as a series
  of wavefronts, one per such segment; between two, the calling thread records the statistics in
  raster order (the counts are halved as they fill, so the order matters) and refreshes the
  tables. A row the segment ends inside goes on in the next one with the state it had.
- The segment analysis reads only the source, so its rows are split among the workers, each with
  its own counts, which are summed.
