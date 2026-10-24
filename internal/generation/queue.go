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
// jobs on different GPUs run side by side. LLM chat never waits here.
type Queue struct {
	mu     sync.Mutex
	nextID int64
	gpus   map[int]*gpuQueue
}

type gpuQueue struct {
	running *Ticket
	waiting []*Ticket
	history []TicketInfo
}

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
	return &Queue{gpus: map[int]*gpuQueue{}}
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
	if g.running != nil && len(g.waiting) >= MaxWaitingPerGPU {
		return nil, ErrQueueFull
	}
	q.nextID++
	t := &Ticket{q: q, ready: make(chan struct{}), info: TicketInfo{
		ID: q.nextID, GPU: gpu, InstanceID: instanceID, Kind: kind, Submitted: time.Now(),
	}}
	if g.running == nil {
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
	g.history = append(g.history, t.info)
	if len(g.history) > historyPerGPU {
		g.history = g.history[len(g.history)-historyPerGPU:]
	}
	g.running = nil
	if len(g.waiting) > 0 {
		next := g.waiting[0]
		g.waiting = g.waiting[1:]
		q.startLocked(g, next)
	}
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
}

// Busy reports whether a job is running or waiting.
func (s GPUSnapshot) Busy() bool { return s.Running != nil || len(s.Waiting) > 0 }

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
	if g.running != nil {
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
