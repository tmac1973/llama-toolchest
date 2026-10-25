package api

// The in-memory job store that image and music jobs share. Jobs live for
// the run of the server only; finished ones are pruned after jobKeep.

import (
	"crypto/rand"
	"encoding/hex"
	"os"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// Job limits.
const (
	jobKeep       = 24 * time.Hour
	jobMax        = 200
	jobPruneEvery = 10 * time.Minute
)

// Job statuses.
const (
	jobQueued    = "queued"
	jobRunning   = "running"
	jobSucceeded = "succeeded"
	jobFailed    = "failed"
	jobCancelled = "cancelled"
)

// jobState is what the store reads from a job.
type jobState struct {
	id       string
	status   string
	ticket   *generation.Ticket
	finished *time.Time
	// files are the job's output files, deleted when it is pruned.
	files []string
}

// jobRecord is a job type the store can hold.
type jobRecord[T any] interface {
	state() jobState
	// withPosition returns a copy of the job with its queue position set.
	withPosition(int) T
}

// jobStore holds one kind of job, in memory.
type jobStore[T jobRecord[T]] struct {
	mu    sync.Mutex
	jobs  map[string]*T
	order []string // oldest first
	// onUpdate, when set, sees a copy of each job after every update.
	// Tests use it to record the phases a job goes through.
	onUpdate func(T)
}

func (m *jobStore[T]) add(j *T) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.jobs == nil {
		m.jobs = map[string]*T{}
	}
	id := (*j).state().id
	m.jobs[id] = j
	m.order = append(m.order, id)
}

// view returns a copy of a job for responses, with its queue position.
func (m *jobStore[T]) view(id string) (T, bool) {
	m.mu.Lock()
	j, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		var zero T
		return zero, false
	}
	out := *j
	m.mu.Unlock()
	pos := 0
	if st := out.state(); st.status == jobQueued && st.ticket != nil {
		pos = st.ticket.Position()
	}
	return out.withPosition(pos), true
}

func (m *jobStore[T]) update(id string, fn func(*T)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if j, ok := m.jobs[id]; ok {
		fn(j)
		if m.onUpdate != nil {
			m.onUpdate(*j)
		}
	}
}

// prune drops finished jobs (and their output files) older than jobKeep,
// and the oldest finished ones beyond jobMax.
func (m *jobStore[T]) prune(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	keep := m.order[:0:0]
	excess := len(m.order) - jobMax
	for _, id := range m.order {
		st := (*m.jobs[id]).state()
		if st.finished != nil && (now.Sub(*st.finished) > jobKeep || excess > 0) {
			for _, f := range st.files {
				if f != "" {
					os.Remove(f)
				}
			}
			delete(m.jobs, id)
			excess--
			continue
		}
		keep = append(keep, id)
	}
	m.order = keep
}

// watchGenJobs prunes old image and music jobs for the life of the server.
func (s *Server) watchGenJobs() {
	for range time.Tick(jobPruneEvery) {
		now := time.Now()
		s.genMusic.prune(now)
		s.genImageJobs.prune(now)
	}
}

func newGenJobID() string {
	var b [8]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
