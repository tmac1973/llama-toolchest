package generation

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"
)

// Queue limits.
const (
	// MaxWaitingPerGPU is how many jobs may wait for one GPU.
	MaxWaitingPerGPU = 16
	// historyPerGPU is how many finished jobs each GPU remembers.
	historyPerGPU = 100
)

// ErrQueueFull is returned when a GPU already has MaxWaitingPerGPU jobs
// waiting.
var ErrQueueFull = errors.New("too many generation jobs are waiting for this GPU")

// Queue lets one image or music job run on each GPU at a time, the rest
// waiting in order. Jobs on one GPU slow each other down a lot (measured on
// one R9700: images 2.3x, songs 2.8x, chat 2.2-4x), so they take turns;
// jobs on different GPUs run side by side. LLM chat never waits here, but
// LLM measurements (benchmarks, autotune, Autoconfigure's test load) take
// their turn too, through Hold: their numbers would be wrong with an image
// job running beside them.
type Queue struct {
	mu     sync.Mutex
	nextID int64
	gpus   map[int]*gpuQueue
	// changed is closed and replaced whenever a job finishes or a hold
	// ends, waking Hold callers waiting for a GPU to be free.
	changed chan struct{}
}

type gpuQueue struct {
	running *Ticket
	waiting []*Ticket
	history []TicketInfo
	// held is the hold that has the GPU (nil if none); claims counts
	// Hold calls waiting for or holding it. While claims > 0 no new job
	// starts, so a hold is not kept waiting by jobs arriving after it.
	held   *TicketInfo
	claims int
}

// KindLLM marks a hold in a GPU's history: an LLM measurement, not an
// image or music job.
const KindLLM BackendKind = "llm"

// Ticket is one job's place in a GPU's queue.
type Ticket struct {
	q     *Queue
	info  TicketInfo
	ready chan struct{} // closed when the job may run
}

// TicketInfo describes a job for snapshots and history.
type TicketInfo struct {
	ID         int64       `json:"id"`
	GPU        int         `json:"gpu"`
	InstanceID string      `json:"instance"`
	Kind       BackendKind `json:"kind"`
	Submitted  time.Time   `json:"-"`
	Started    time.Time   `json:"-"`
	Finished   time.Time   `json:"-"`
}

// NewQueue returns an empty queue.
func NewQueue() *Queue {
	return &Queue{gpus: map[int]*gpuQueue{}, changed: make(chan struct{})}
}

// notifyLocked wakes Hold callers waiting for a GPU.
func (q *Queue) notifyLocked() {
	close(q.changed)
	q.changed = make(chan struct{})
}

// Hold takes GPUs for an LLM measurement named label (e.g. "a benchmark").
// It waits until no image or music job is running on any of them, then
// keeps them: jobs submitted meanwhile wait, as behind any other job, and
// start when release is called. Jobs already waiting keep their places
// and run after it. When ctx ends while waiting, nothing is held and
// ctx's error is returned.
func (q *Queue) Hold(ctx context.Context, gpus []int, label string) (release func(), err error) {
	gpus = slices.Compact(slices.Sorted(slices.Values(gpus)))
	q.mu.Lock()
	for _, i := range gpus {
		q.gpu(i).claims++
	}
	for {
		free := true
		for _, i := range gpus {
			if g := q.gpu(i); g.running != nil || g.held != nil {
				free = false
			}
		}
		if free {
			break
		}
		changed := q.changed
		q.mu.Unlock()
		select {
		case <-changed:
			q.mu.Lock()
		case <-ctx.Done():
			q.mu.Lock()
			for _, i := range gpus {
				g := q.gpu(i)
				g.claims--
				q.startNextLocked(g)
			}
			q.mu.Unlock()
			return nil, ctx.Err()
		}
	}
	held := make(map[int]*TicketInfo, len(gpus))
	for _, i := range gpus {
		q.nextID++
		info := &TicketInfo{ID: q.nextID, GPU: i, InstanceID: label, Kind: KindLLM, Submitted: time.Now(), Started: time.Now()}
		q.gpu(i).held = info
		held[i] = info
	}
	q.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			q.mu.Lock()
			defer q.mu.Unlock()
			for i, info := range held {
				g := q.gpu(i)
				info.Finished = time.Now()
				q.recordLocked(g, *info)
				g.held = nil
				g.claims--
				q.startNextLocked(g)
			}
			q.notifyLocked()
		})
	}, nil
}

// startNextLocked starts the first waiting job when nothing runs on the
// GPU and no hold has or wants it.
func (q *Queue) startNextLocked(g *gpuQueue) {
	if g.running != nil || g.held != nil || g.claims > 0 || len(g.waiting) == 0 {
		return
	}
	next := g.waiting[0]
	g.waiting = g.waiting[1:]
	q.startLocked(g, next)
}

func (q *Queue) recordLocked(g *gpuQueue, info TicketInfo) {
	g.history = append(g.history, info)
	if len(g.history) > historyPerGPU {
		g.history = g.history[len(g.history)-historyPerGPU:]
	}
}

func (q *Queue) gpu(i int) *gpuQueue {
	g := q.gpus[i]
	if g == nil {
		g = &gpuQueue{}
		q.gpus[i] = g
	}
	return g
}

// Submit queues a job on a GPU. It runs at once when the GPU is free;
// otherwise it waits behind the others, up to MaxWaitingPerGPU.
func (q *Queue) Submit(gpu int, instanceID string, kind BackendKind) (*Ticket, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	g := q.gpu(gpu)
	if (g.running != nil || g.held != nil || g.claims > 0) && len(g.waiting) >= MaxWaitingPerGPU {
		return nil, ErrQueueFull
	}
	q.nextID++
	t := &Ticket{q: q, ready: make(chan struct{}), info: TicketInfo{
		ID: q.nextID, GPU: gpu, InstanceID: instanceID, Kind: kind, Submitted: time.Now(),
	}}
	if g.running == nil && g.held == nil && g.claims == 0 {
		q.startLocked(g, t)
	} else {
		g.waiting = append(g.waiting, t)
	}
	return t, nil
}

func (q *Queue) startLocked(g *gpuQueue, t *Ticket) {
	t.info.Started = time.Now()
	g.running = t
	close(t.ready)
}

// Wait blocks until the job may run. When ctx ends first (the client went
// away), the job leaves the queue and ctx's error is returned.
func (t *Ticket) Wait(ctx context.Context) error {
	select {
	case <-t.ready:
		return nil
	case <-ctx.Done():
	}
	q := t.q
	q.mu.Lock()
	g := q.gpu(t.info.GPU)
	if i := slices.Index(g.waiting, t); i >= 0 {
		g.waiting = slices.Delete(g.waiting, i, i+1)
		q.mu.Unlock()
		return ctx.Err()
	}
	q.mu.Unlock()
	// It was started just as ctx ended: give the GPU to the next job.
	t.Done()
	return ctx.Err()
}

// Done marks a running job finished, records it, and starts the next one
// waiting for the GPU. Calling it on a job that is not running is a no-op.
func (t *Ticket) Done() {
	q := t.q
	q.mu.Lock()
	defer q.mu.Unlock()
	g := q.gpu(t.info.GPU)
	if g.running != t {
		return
	}
	t.info.Finished = time.Now()
	q.recordLocked(g, t.info)
	g.running = nil
	q.startNextLocked(g)
	q.notifyLocked()
}

// Position is how many jobs are ahead of this one: 0 once it runs.
func (t *Ticket) Position() int {
	q := t.q
	q.mu.Lock()
	defer q.mu.Unlock()
	g := q.gpu(t.info.GPU)
	if i := slices.Index(g.waiting, t); i >= 0 {
		return i + 1
	}
	return 0
}

// Info returns the ticket's details.
func (t *Ticket) Info() TicketInfo {
	t.q.mu.Lock()
	defer t.q.mu.Unlock()
	return t.info
}

// GPUSnapshot is a GPU's queue right now.
type GPUSnapshot struct {
	GPU     int          `json:"gpu"`
	Running *TicketInfo  `json:"running,omitempty"`
	Waiting []TicketInfo `json:"waiting"`
	// HeldBy names the LLM measurement holding the GPU, if one is.
	HeldBy string `json:"held_by,omitempty"`
}

// Busy reports whether a job is running or waiting, or the GPU is held.
func (s GPUSnapshot) Busy() bool { return s.Running != nil || len(s.Waiting) > 0 || s.HeldBy != "" }

// Snapshot returns one GPU's queue.
func (q *Queue) Snapshot(gpu int) GPUSnapshot {
	q.mu.Lock()
	defer q.mu.Unlock()
	g := q.gpu(gpu)
	snap := GPUSnapshot{GPU: gpu, Waiting: []TicketInfo{}}
	if g.running != nil {
		info := g.running.info
		snap.Running = &info
	}
	if g.held != nil {
		snap.HeldBy = g.held.InstanceID
	}
	for _, t := range g.waiting {
		snap.Waiting = append(snap.Waiting, t.info)
	}
	return snap
}

// Load is how many jobs are running or waiting on a GPU.
func (q *Queue) Load(gpu int) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	g := q.gpu(gpu)
	n := len(g.waiting)
	if g.running != nil || g.held != nil {
		n++
	}
	return n
}

// History returns a GPU's finished jobs, oldest first.
func (q *Queue) History(gpu int) []TicketInfo {
	q.mu.Lock()
	defer q.mu.Unlock()
	return slices.Clone(q.gpu(gpu).history)
}
