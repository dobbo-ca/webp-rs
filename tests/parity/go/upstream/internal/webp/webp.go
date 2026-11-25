// Package webp implements github.com/dobbo-ca/webp-go, which re-exports its API.
package webp

import (
	"context"
	"errors"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/dobbo-ca/webp-go/internal/vp8enc"
)

// Options controls the encoder. Method is 0..6 like cwebp -m; nil
// means Quality 75 Method 4, a zero Options is Method 0.
//
// MatchLibwebp makes Methods 3 to 6 decide and write exactly as
// libwebp 1.6.0 (and 1.2.2) does with default settings, so the output
// bytes equal cwebp -q Quality -m Method. It costs speed: every intra
// mode gets a full rate-distortion trial. Methods 0 to 2 ignore it.
// ALPH matches libwebp 1.2.2, or 1.6.0 with AlphaLibwebp16.
type Options struct {
	Quality      float32
	Method       int
	MatchLibwebp bool
	// AlphaLibwebp16, with MatchLibwebp at Methods 3 to 5, writes ALPH
	// as libwebp 1.6.0 does; the default is 1.2.2.
	AlphaLibwebp16 bool
	// LegacyAlpha, with MatchLibwebp, writes ALPH with webp-go's original
	// writer: same decoded alpha, other bytes, about 4x faster.
	LegacyAlpha bool
	// Segments 1 is one segment and no analysis pass, cwebp -segments 1
	// with MatchLibwebp at Methods 3 to 6. 0 and 4 are the default 4;
	// any other value is an error.
	Segments int
}

// Encoder keeps its buffers between calls.
type Encoder struct {
	e   vp8enc.Encoder
	r   resv
	w   io.Writer
	out outBuf // coded file, written to w after release
}

// outBuf collects the coded file, so the encode's buffers can go to the
// next encode before the Writer runs. Its bytes come from outs when the
// file is written, at the end of the encode, and go back after the
// Writer, so a queued encode reuses the last one's.
type outBuf struct {
	b    []byte
	have bool
}

// outs are idle output buffers, at most one per slot. sync.Pool drops
// items at random under -race, which the zero-alloc tests would see.
var outs struct {
	mu   sync.Mutex
	free [][]byte
}

func (o *outBuf) Write(p []byte) (int, error) {
	if !o.have {
		outs.mu.Lock()
		if k := len(outs.free); k > 0 {
			o.b, outs.free[k-1] = outs.free[k-1][:0], nil
			outs.free = outs.free[:k-1]
		}
		outs.mu.Unlock()
		o.have = true
	}
	o.b = append(o.b, p...)
	return len(p), nil
}

func (o *outBuf) put() {
	if !o.have {
		return
	}
	outs.mu.Lock()
	if len(outs.free) < runtime.GOMAXPROCS(0) {
		outs.free = append(outs.free, o.b)
	}
	outs.mu.Unlock()
	o.b, o.have = nil, false
}

// Pooled so one-shot calls reuse buffers.
var pool = sync.Pool{New: func() any { return new(Encoder) }}

func Encode(w io.Writer, m image.Image, o *Options) error {
	return EncodeContext(context.Background(), w, m, o)
}

// EncodeContext is Encode that gives up, returning ctx.Err(), when ctx is
// done while it waits for a slot or budget or while it encodes. Nothing
// is written then.
func EncodeContext(ctx context.Context, w io.Writer, m image.Image, o *Options) error {
	// Wait before Get so waiters hold no buffers.
	r, err := admit(ctx.Done(), m.Bounds(), nil)
	if err != nil {
		return ctx.Err()
	}
	e := pool.Get().(*Encoder)
	e.Reset(w)
	err = e.run(ctx, m, o, r)
	e.Reset(nil)
	pool.Put(e)
	return err
}

// EncodeGray matches Encode on the same image.
func EncodeGray(w io.Writer, m *image.Gray, o *Options) error {
	return Encode(w, m, o)
}

func EncodeYCbCr(w io.Writer, m *image.YCbCr, o *Options) error {
	if m.SubsampleRatio != image.YCbCrSubsampleRatio420 {
		return errors.New("webp: only 4:2:0 YCbCr is supported")
	}
	return Encode(w, m, o)
}

func (e *Encoder) Reset(w io.Writer) { e.w = w }

func (e *Encoder) Encode(m image.Image, o *Options) error {
	return e.EncodeContext(context.Background(), m, o)
}

// EncodeContext is Encode with EncodeContext's cancellation.
func (e *Encoder) EncodeContext(ctx context.Context, m image.Image, o *Options) error {
	r, err := admit(ctx.Done(), m.Bounds(), &e.e)
	if err != nil {
		return ctx.Err()
	}
	return e.run(ctx, m, o, r)
}

// run encodes under r, restarting when the gate refuses, then releases r
// and writes.
func (e *Encoder) run(ctx context.Context, m image.Image, o *Options, r resv) error {
	done := ctx.Done()
	e.e.Reset(&e.out)
	e.e.Cancel(done)
	var pages *vp8enc.Encoder // non-nil when pages go back to lim
	defer func() { e.finish(pages != nil) }()
	if r.seq == 0 {
		if err := e.encode(m, o); err != nil {
			return e.fail(ctx, err)
		}
		return e.write()
	}
	b := m.Bounds()
	e.r = r
	e.r.done = done
	held := int64(-1)
	released := false
	defer func() {
		// seq 0: a canceled retry already gave r back.
		if !released && e.r.seq != 0 { // an error or a panic
			release(&e.r, b, held, pages)
		}
	}()
	for {
		pages = nil
		e.e.Limit(nil)
		if budget() != math.MaxInt64 {
			pages = &e.e
			adopt(pages, e.r.need)
			e.e.Limit(&e.r)
		}
		err := e.encode(m, o)
		if err == vp8enc.ErrRetry && ctx.Err() == nil {
			if err = retry(&e.r, &e.e, b); err == nil {
				continue
			}
		}
		if err != nil {
			return e.fail(ctx, err)
		}
		held = e.e.Held() + int64(len(e.out.b)) // reuse slack is not this encode's
		released = true
		release(&e.r, b, held, pages)
		return e.write()
	}
}

// keep bounds the partitions an idle Encoder holds outside the budget,
// with its alpha state after an opaque encode, and each pooled output
// buffer.
const keep = 2 << 20

// finish pools the output buffer; under a limit it first drops one over
// keep, which would sit outside the budget. release trimmed the rest.
func (e *Encoder) finish(limited bool) {
	if limited {
		lim.mu.Lock()
		if c := cap(e.out.b); c > keep {
			drop(int64(c))
			e.out.b, e.out.have = nil, false
		}
		collect()
		lim.mu.Unlock()
	}
	e.out.put()
}

// fail maps a stop for ctx to ctx.Err().
func (e *Encoder) fail(ctx context.Context, err error) error {
	if err == vp8enc.ErrCanceled || err == vp8enc.ErrRetry || err == errCanceled {
		if c := ctx.Err(); c != nil {
			return c
		}
	}
	return err
}

func (e *Encoder) write() error {
	_, err := e.w.Write(e.out.b)
	return err
}

func (e *Encoder) encode(m image.Image, o *Options) error {
	q, method, lw, a16, la, sg := float32(75), 4, false, false, false, 0
	if o != nil {
		q, method, lw, a16, la, sg = o.Quality, o.Method, o.MatchLibwebp, o.AlphaLibwebp16, o.LegacyAlpha, o.Segments
	}
	if sg != 0 && sg != 1 && sg != 4 {
		return errors.New("webp: Segments must be 0, 1 or 4")
	}
	e.e.Segments(sg)
	e.e.MatchLibwebp(lw)
	e.e.AlphaLibwebp16(a16)
	e.e.LegacyAlpha(la)
	return e.e.Encode(m, q, method)
}

// Measured in TestChargeCoversAlloc and docs/design/admission.md.
const (
	fixedBytes = 256 << 10
	pageCharge = 64 << 10 * 5 / 4
	priorRate  = 2 // B/px before any encode finishes
	decay      = 32
)

func padded(b image.Rectangle) int64 {
	return int64((b.Dx()+15)&^15) * int64((b.Dy()+15)&^15)
}

// MaxEncodeCharge is the most budget one opaque w x h encode can charge,
// for an Encoder last used on images no bigger. keep covers the alpha
// state an opaque encode's Trim may leave.
func MaxEncodeCharge(w, h int) int64 { return charge(vp8enc.MaxCharge(w, h) + keep) }

// MaxEncodedSize is the most bytes one opaque w x h encode writes.
func MaxEncodedSize(w, h int) int64 { return vp8enc.MaxOutput(w, h) }

// charge covers garbage beyond held bytes.
func charge(held int64) int64 { return (held-fixedBytes)*5/4 + fixedBytes }

var lim struct {
	mu       sync.Mutex
	n        int      // encodes in flight
	reserved int64    // sum of their reservations
	used     int64    // sum of their charges
	rate     float64  // learned reserved bytes per pixel
	seq      uint64   // last ticket
	live     []uint64 // tickets in flight, oldest first
	queue    []*waiter
	dq       []*decodeWaiter // decodes waiting for budget, oldest first
	oldest   *waiter         // oldest in flight, waiting to grow
	free     [][]uint16
	spare    []vp8enc.Encoder // parked buffers for the next encode
	spared   int64            // their charges
	retries  int
	limitSet bool  // SetMemoryLimit was called
	limit    int64 // its budget, math.MaxInt64 for none
	garbage  []gcBytes
	gcing    bool // a forced GC is running
	cycles   []metrics.Sample
}

// gcBytes are n bytes dropped when cycle GC cycles had ended; they stay
// counted until a cycle that started after has ended, as in jp2k-go.
type gcBytes struct {
	cycle uint64
	n     int64
}

var errCanceled = errors.New("webp: canceled")

// gcCycles counts ended GC cycles. Caller holds lim.mu.
func gcCycles() uint64 {
	if lim.cycles == nil {
		lim.cycles = []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	}
	metrics.Read(lim.cycles)
	return lim.cycles[0].Value.Uint64()
}

// garbageNow drops collected garbage and returns the rest.
// Caller holds lim.mu.
func garbageNow() int64 {
	if len(lim.garbage) == 0 {
		return 0
	}
	c := gcCycles()
	k, g := 0, int64(0)
	for _, x := range lim.garbage {
		if x.cycle+2 <= c {
			continue
		}
		lim.garbage[k] = x
		k++
		g += x.n
	}
	lim.garbage = lim.garbage[:k]
	return g
}

// drop counts n bytes just made garbage. Caller holds lim.mu.
func drop(n int64) {
	if n > 0 {
		lim.garbage = append(lim.garbage, gcBytes{gcCycles(), n})
	}
}

// collect runs one shared GC for waiters that only garbage blocks, then
// wakes them. Caller holds lim.mu.
func collect() {
	if lim.gcing || garbageNow() == 0 || len(lim.queue) == 0 && lim.oldest == nil && len(lim.dq) == 0 {
		return
	}
	lim.gcing = true
	go func() {
		runtime.GC()
		lim.mu.Lock()
		lim.gcing = false
		wake()
		lim.mu.Unlock()
	}()
}

type waiter struct {
	b     image.Rectangle
	min   int64 // charge reached before restarting
	need  int64
	seq   uint64
	ready chan struct{}
}

// resv is one encode's share; Page makes it a vp8enc.Limiter.
type resv struct {
	need, used, min int64
	seq             uint64
	done            <-chan struct{} // the encode's ctx.Done()
}

// Caller holds lim.mu.
func estimate(b image.Rectangle) int64 {
	r := lim.rate
	if r == 0 {
		r = priorRate
	}
	return int64(r*float64(padded(b))) + fixedBytes
}

// SetMemoryLimit sets how many bytes of heap the encodes in flight may
// hold together, and returns the previous limit. n <= 0 means no limit.
// Until it is called, the limit is half of GOMEMLIMIT (see
// runtime/debug.SetMemoryLimit) when one is set, else, on Linux, half of
// the cgroup memory limit (v2 memory.max for this process's cgroup, else
// the root's, else v1 memory.limit_in_bytes) read once, else none. Lowering the limit can
// make an encode already running restart, as in docs/memory.md; a waiting
// encode is woken to recheck it either way.
func SetMemoryLimit(n int64) int64 {
	if n <= 0 {
		n = math.MaxInt64
	}
	lim.mu.Lock()
	prev := budget()
	lim.limitSet, lim.limit = true, n
	wake()
	lim.mu.Unlock()
	return prev
}

// Caller holds lim.mu.
func budget() int64 { return budgetFor("") }

// budgetFor is budget, with the cgroup limit read from cgroupRoot
// instead of the default path; "" uses cgroupMax. Caller holds lim.mu.
func budgetFor(cgroupRoot string) int64 {
	if lim.limitSet {
		return lim.limit
	}
	if l := debug.SetMemoryLimit(-1); l < math.MaxInt64 {
		return l / 2
	}
	if runtime.GOOS == "linux" {
		l := cgroupMax()
		if cgroupRoot != "" {
			l = cgroupLimit(cgroupRoot)
		}
		if l > 0 {
			return l / 2
		}
	}
	return math.MaxInt64
}

// cgroupMax is the cgroup memory limit, read once: a file read per
// encode costs more than a small encode. Same rules as jp2k-go.
var cgroupMax = sync.OnceValue(func() int64 { return cgroupMaxFrom(cgroupRoot(), "/sys/fs/cgroup") })

// cgroupMaxFrom falls back on limitless roots.
func cgroupMaxFrom(root, fallback string) int64 {
	if l := cgroupLimit(root); l > 0 {
		return l
	}
	return cgroupLimit(fallback)
}

// cgroupRoot is this process's cgroup v2 directory, so a pod sharing the
// host's namespace reads its own limit, not the host's.
func cgroupRoot() string {
	if b, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		if p, ok := cgroupV2Path(b); ok {
			return filepath.Join("/sys/fs/cgroup", p)
		}
	}
	return "/sys/fs/cgroup"
}

// cgroupV2Path reads the "0::" line's path.
func cgroupV2Path(procSelfCgroup []byte) (string, bool) {
	for _, line := range strings.Split(string(procSelfCgroup), "\n") {
		if p, ok := strings.CutPrefix(line, "0::"); ok {
			return p, true
		}
	}
	return "", false
}

// cgroupLimit reads the cgroup v2 or v1 memory limit under root, or 0.
func cgroupLimit(root string) int64 {
	for _, f := range []string{"memory.max", "memory/memory.limit_in_bytes"} {
		b, err := os.ReadFile(filepath.Join(root, f))
		if err != nil {
			continue
		}
		// v2 says "max" for none, v1 a number near 2^63.
		n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err == nil && n > 0 && n < 1<<50 {
			return n
		}
	}
	return 0
}

// Caller holds lim.mu.
func admits(need int64) bool {
	if lim.n >= runtime.GOMAXPROCS(0) || lim.oldest != nil {
		return false
	}
	return lim.n == 0 || lim.reserved+need+garbageNow() <= budget()
}

// Caller holds lim.mu.
func enter(need int64) uint64 {
	lim.n++
	lim.reserved += need
	lim.seq++
	lim.live = append(lim.live, lim.seq)
	return lim.seq
}

// admit acquires a slot for b; e parks its buffers to wait. It gives up
// with errCanceled when done closes first.
func admit(done <-chan struct{}, b image.Rectangle, e *vp8enc.Encoder) (resv, error) {
	if w, h := b.Dx(), b.Dy(); w < 1 || h < 1 || w > 16383 || h > 16383 {
		return resv{}, nil // vp8enc fails before allocating
	}
	lim.mu.Lock()
	if need := estimate(b); len(lim.queue) == 0 && admits(need) {
		s := enter(need)
		lim.mu.Unlock()
		return resv{need: need, seq: s}, nil
	}
	if e != nil && budget() != math.MaxInt64 {
		park(e) // waiters sit outside the budget
	}
	// FIFO so oversized encodes are not starved.
	w := &waiter{b: b, ready: make(chan struct{})}
	lim.queue = append(lim.queue, w)
	collect()
	lim.mu.Unlock()
	return await(done, w)
}

// await waits for w's slot, or takes w out when done closes first.
func await(done <-chan struct{}, w *waiter) (resv, error) {
	select {
	case <-w.ready:
		return resv{need: w.need, seq: w.seq}, nil
	case <-done:
	}
	lim.mu.Lock()
	defer lim.mu.Unlock()
	if i := slices.Index(lim.queue, w); i >= 0 {
		lim.queue = slices.Delete(lim.queue, i, i+1)
		wake() // w may have blocked the head
		return resv{}, errCanceled
	}
	// Admitted as done closed: give the slot straight back.
	r := resv{need: w.need, seq: w.seq}
	leave(&r, nil)
	wake()
	return resv{}, errCanceled
}

// Page charges held bytes, then hands out a free page.
func (r *resv) Page(held int64) ([]uint16, bool) {
	c := charge(held)
	lim.mu.Lock()
	if more := c - r.need; more > 0 {
		switch {
		case lim.oldest == nil && lim.reserved+more+garbageNow() <= budget(), lim.n == 1:
			lim.reserved += more
		case r.seq != lim.live[0]:
			r.min = c
			lim.mu.Unlock()
			return nil, false
		default:
			// The oldest waits so it always finishes.
			w := &waiter{need: more, ready: make(chan struct{})}
			lim.oldest = w
			collect()
			lim.mu.Unlock()
			select {
			case <-w.ready:
				lim.mu.Lock()
			case <-r.done:
				lim.mu.Lock()
				if lim.oldest == w {
					lim.oldest = nil
					wake()
				} else {
					r.need = c // granted as done closed
				}
				lim.mu.Unlock()
				return nil, false
			}
		}
		r.need = c
	}
	lim.used += c - r.used
	r.used = c
	var p []uint16
	if k := len(lim.free); k > 0 {
		p, lim.free[k-1] = lim.free[k-1], nil
		lim.free = lim.free[:k-1]
	}
	trimFree()
	lim.mu.Unlock()
	return p, true
}

// Caller holds lim.mu.
func trimFree() {
	b := budget()
	limited := b != math.MaxInt64
	if !limited {
		b = 0 // unused without a limit
	}
	for k := len(lim.free); k > 0 && lim.used+lim.spared+int64(k)*pageCharge > b; k-- {
		lim.free[k-1] = nil
		lim.free = lim.free[:k-1]
		if limited {
			drop(pageCharge)
		}
	}
	for k := len(lim.spare); k > 0 && lim.used+lim.spared > b; k-- {
		c := spareCharge(&lim.spare[k-1])
		lim.spared -= c
		lim.spare[k-1] = vp8enc.Encoder{}
		lim.spare = lim.spare[:k-1]
		if limited {
			drop(c)
		}
	}
}

func spareCharge(e *vp8enc.Encoder) int64 {
	if h := e.Held(); h > 0 {
		return charge(h)
	}
	return 0
}

// park moves e's buffers to lim so they are reused, not garbage, less
// what Trim sheds. Caller holds lim.mu.
func park(e *vp8enc.Encoder) {
	drop(e.Trim(keep))
	lim.free = e.Pages(lim.free)
	if e.Held() > 0 {
		lim.spare = append(lim.spare, vp8enc.Encoder{})
		e.Swap(&lim.spare[len(lim.spare)-1])
		lim.spared += spareCharge(&lim.spare[len(lim.spare)-1])
	}
	trimFree()
}

// adopt moves e's pages to lim and takes a bigger parked set that fits need.
func adopt(e *vp8enc.Encoder, need int64) {
	lim.mu.Lock()
	lim.free = e.Pages(lim.free)
	h := e.Held()
	for i := len(lim.spare) - 1; i >= 0; i-- {
		s := &lim.spare[i]
		if c := spareCharge(s); c <= need && s.Held() > h {
			e.Swap(s)
			lim.spared += spareCharge(s) - c
			if s.Held() == 0 {
				lim.spare = slices.Delete(lim.spare, i, i+1)
			}
			break
		}
	}
	trimFree()
	lim.mu.Unlock()
}

// leave drops r from the in-flight set; caller holds lim.mu.
func leave(r *resv, e *vp8enc.Encoder) {
	lim.n--
	lim.reserved -= r.need
	lim.used -= r.used
	r.used = 0
	lim.live = slices.DeleteFunc(lim.live, func(s uint64) bool { return s == r.seq })
	if e != nil {
		lim.free = e.Pages(lim.free)
	}
}

// wake serves the waiting oldest encode, then the queue.
// Caller holds lim.mu.
func wake() {
	if w := lim.oldest; w != nil {
		if lim.reserved+w.need+garbageNow() > budget() && lim.n > 1 {
			trimFree()
			collect()
			return
		}
		lim.reserved += w.need
		lim.oldest = nil
		close(w.ready)
	}
	for len(lim.queue) > 0 {
		w := lim.queue[0]
		need := max(estimate(w.b), w.min)
		if !admits(need) {
			break
		}
		lim.queue = lim.queue[1:]
		w.need, w.seq = need, enter(need)
		close(w.ready)
	}
	for len(lim.dq) > 0 {
		w := lim.dq[0]
		if lim.reserved+w.need+garbageNow() > budget() {
			break
		}
		lim.dq = lim.dq[1:]
		lim.reserved += w.need
		close(w.ready)
	}
	trimFree()
	collect()
}

// retry gives back r and requeues it at the head. On errCanceled r
// holds nothing.
func retry(r *resv, e *vp8enc.Encoder, b image.Rectangle) error {
	lim.mu.Lock()
	lim.retries++
	leave(r, e)
	park(e) // waiters sit outside the budget
	w := &waiter{b: b, min: r.min, ready: make(chan struct{})}
	lim.queue = slices.Insert(lim.queue, 0, w)
	wake()
	lim.mu.Unlock()
	done := r.done
	nr, err := await(done, w)
	*r = nr
	r.done = done
	return err
}

// release learns from held bytes, held < 0 if the encode failed.
func release(r *resv, b image.Rectangle, held int64, e *vp8enc.Encoder) {
	lim.mu.Lock()
	if held >= 0 {
		// Jump up at once, decay down slowly.
		// Floored: tiny images sit under fixedBytes. Page charges count
		// partition bounds, so learn them too or every encode grows.
		x := float64(max(charge(held), r.used, fixedBytes)-fixedBytes) / float64(padded(b))
		if x > lim.rate {
			lim.rate = x
		} else {
			lim.rate += float64((x - lim.rate) / decay) // no arm64 fusion
		}
	}
	leave(r, e)
	// Trim before the Writer runs: out has the bytes.
	if e != nil && len(lim.queue) > 0 {
		park(e) // hand buffers to the next encode
	} else if e != nil {
		drop(e.Trim(keep))
	}
	wake()
	lim.mu.Unlock()
}
