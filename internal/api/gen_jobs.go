package api

// The job store that image, video and music jobs share. Each job is kept in
// memory and, once the store has a directory, in a record file there, so a
// restart does not lose it. Finished jobs are pruned after jobKeep.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
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

// jobRestartedError is the error of a job that was queued or running when
// the server stopped.
const jobRestartedError = "the server restarted before the job finished"

// jobState is what the store reads from a job.
type jobState struct {
	id        string
	status    string
	ticket    *generation.Ticket
	submitted time.Time
	finished  *time.Time
	// files are the job's output files, deleted when it is pruned.
	files []string
}

// jobRecord is a job type the store can hold.
type jobRecord[T any] interface {
	state() jobState
	// withPosition returns a copy of the job with its queue position set.
	withPosition(int) T
	// private is the part of the job that clients never see but that a
	// restart must keep, such as its output paths. It is saved as JSON.
	private() any
	// restored returns a job read back from its record, with its private
	// part decoded from raw. A job that had not finished is returned
	// failed at now, since its backend stopped with the server.
	restored(raw json.RawMessage, now time.Time) (T, error)
}

// jobFile is a job's record file.
type jobFile[T any] struct {
	Job     T               `json:"job"`
	Private json.RawMessage `json:"private,omitempty"`
}

// jobStore holds one kind of job.
type jobStore[T jobRecord[T]] struct {
	mu    sync.Mutex
	jobs  map[string]*T
	order []string // oldest first
	// onUpdate, when set, sees a copy of each job after every update.
	// Tests use it to record the phases a job goes through.
	onUpdate func(T)
	// max is how many finished jobs are kept; zero means jobMax.
	max int

	// dir holds a record file per job; empty keeps jobs in memory only.
	dir string
	// outDir is where the jobs' output files are written. Files there
	// that no job owns are removed once they are older than jobKeep.
	outDir string
	// saveMu keeps record writes in the order of the updates they save.
	saveMu sync.Mutex
}

// open sets the store's directories and loads the jobs saved there. Jobs
// that had not finished are saved again as failed.
func (m *jobStore[T]) open(dir, outDir string, now time.Time) error {
	m.mu.Lock()
	m.dir, m.outDir = dir, outDir
	m.mu.Unlock()
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var loaded, failed []T
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		file := filepath.Join(dir, e.Name())
		j, wasFinished, err := readJobFile[T](file, now)
		if err != nil {
			slog.Warn("generation job record not loaded", "file", file, "error", err)
			continue
		}
		loaded = append(loaded, j)
		if !wasFinished {
			failed = append(failed, j)
		}
	}
	slices.SortFunc(loaded, func(a, b T) int { return a.state().submitted.Compare(b.state().submitted) })
	m.mu.Lock()
	if m.jobs == nil {
		m.jobs = map[string]*T{}
	}
	for _, j := range loaded {
		id := j.state().id
		if _, dup := m.jobs[id]; dup {
			continue
		}
		m.jobs[id] = &j
		m.order = append(m.order, id)
	}
	m.mu.Unlock()
	for _, j := range failed {
		m.mu.Lock()
		m.saveLocked(j)
	}
	return nil
}

// readJobFile reads one record and reports whether the job had finished
// before the restart.
func readJobFile[T jobRecord[T]](file string, now time.Time) (T, bool, error) {
	var rec jobFile[T]
	raw, err := os.ReadFile(file)
	if err != nil {
		return rec.Job, false, err
	}
	if err := json.Unmarshal(raw, &rec); err != nil {
		return rec.Job, false, err
	}
	wasFinished := rec.Job.state().finished != nil
	j, err := rec.Job.restored(rec.Private, now)
	if err != nil {
		return j, false, err
	}
	if j.state().id == "" {
		return j, false, errors.New("the record has no job ID")
	}
	return j, wasFinished, nil
}

func (m *jobStore[T]) recordPath(id string) string {
	return filepath.Join(m.dir, id+".json")
}

// encodeJob returns a job's record file contents, or nil if it cannot be
// encoded.
func encodeJob[T jobRecord[T]](j T) []byte {
	priv, err := json.Marshal(j.private())
	if err == nil {
		var raw []byte
		if raw, err = json.Marshal(jobFile[T]{Job: j, Private: priv}); err == nil {
			return raw
		}
	}
	slog.Warn("generation job not saved", "job", j.state().id, "error", err)
	return nil
}

// saveLocked writes a job's record. The caller holds m.mu, which this
// releases: the file is written outside it, so a slow disk does not hold
// up readers, but under saveMu, so writes land in order.
func (m *jobStore[T]) saveLocked(j T) {
	if m.dir == "" {
		m.mu.Unlock()
		return
	}
	id, raw := j.state().id, encodeJob(j)
	m.saveMu.Lock()
	m.mu.Unlock()
	defer m.saveMu.Unlock()
	if raw == nil {
		return
	}
	if err := atomicfile.Write(m.recordPath(id), raw); err != nil {
		slog.Warn("generation job not saved", "job", id, "error", err)
	}
}

func (m *jobStore[T]) add(j *T) {
	m.mu.Lock()
	if m.jobs == nil {
		m.jobs = map[string]*T{}
	}
	id := (*j).state().id
	m.jobs[id] = j
	m.order = append(m.order, id)
	m.saveLocked(*j)
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
	return withQueuePosition(out), true
}

func withQueuePosition[T jobRecord[T]](j T) T {
	pos := 0
	if st := j.state(); st.status == jobQueued && st.ticket != nil {
		pos = st.ticket.Position()
	}
	return j.withPosition(pos)
}

// list returns copies of the jobs, newest first, with their queue
// positions. A non-empty statuses keeps only jobs in one of them.
func (m *jobStore[T]) list(statuses []string) []T {
	m.mu.Lock()
	out := make([]T, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		j := *m.jobs[m.order[i]]
		if len(statuses) == 0 || slices.Contains(statuses, j.state().status) {
			out = append(out, j)
		}
	}
	m.mu.Unlock()
	for i := range out {
		out[i] = withQueuePosition(out[i])
	}
	return out
}

// update changes a job in place and reports whether the store has it. A
// change of status is saved to the job's record; other changes, such as
// phase and progress, are not, since a restart fails an unfinished job.
func (m *jobStore[T]) update(id string, fn func(*T)) bool {
	m.mu.Lock()
	j, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return false
	}
	before := (*j).state().status
	fn(j)
	if m.onUpdate != nil {
		m.onUpdate(*j)
	}
	if (*j).state().status == before {
		m.mu.Unlock()
		return true
	}
	m.saveLocked(*j)
	return true
}

// removeLocked drops a job, its output files and its record. The caller
// holds m.mu and drops the ID from m.order.
func (m *jobStore[T]) removeLocked(id string) {
	for _, f := range (*m.jobs[id]).state().files {
		if f != "" {
			os.Remove(f)
		}
	}
	if m.dir != "" {
		os.Remove(m.recordPath(id))
	}
	delete(m.jobs, id)
}

// remove deletes a finished job and its files. It reports whether the job
// was there and finished; an unfinished job is left alone.
func (m *jobStore[T]) remove(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok || (*j).state().finished == nil {
		return false
	}
	m.removeLocked(id)
	m.order = slices.DeleteFunc(m.order, func(o string) bool { return o == id })
	return true
}

// prune drops finished jobs (and their output files) older than jobKeep,
// and the oldest finished ones beyond the store's maximum. It then removes
// output files older than jobKeep that no job owns: files from jobs that
// were lost before records were kept, or that were saved after their job
// was deleted.
func (m *jobStore[T]) prune(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	limit := m.max
	if limit == 0 {
		limit = jobMax
	}
	keep := m.order[:0:0]
	excess := len(m.order) - limit
	for _, id := range m.order {
		st := (*m.jobs[id]).state()
		if st.finished != nil && (now.Sub(*st.finished) > jobKeep || excess > 0) {
			m.removeLocked(id)
			excess--
			continue
		}
		keep = append(keep, id)
	}
	m.order = keep
	m.pruneOrphansLocked(now)
}

func (m *jobStore[T]) pruneOrphansLocked(now time.Time) {
	if m.outDir == "" {
		return
	}
	entries, err := os.ReadDir(m.outDir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if _, owned := m.jobs[outputJobID(e.Name())]; owned {
			continue
		}
		info, err := e.Info()
		if err != nil || now.Sub(info.ModTime()) <= jobKeep {
			continue
		}
		os.Remove(filepath.Join(m.outDir, e.Name()))
	}
}

// outputJobID is the job ID an output file name starts with: images are
// <id>-<n>.png, songs and videos <id>.<format>.
func outputJobID(name string) string {
	if i := strings.IndexAny(name, "-."); i >= 0 {
		return name[:i]
	}
	return name
}

// openGenJobs loads the image, video and music jobs saved by earlier runs.
func (s *Server) openGenJobs() {
	now := time.Now()
	dir := filepath.Join(s.cfg.DataDir, "gen-output", "jobs")
	for _, st := range []struct {
		name   string
		open   func(dir, outDir string, now time.Time) error
		outDir string
	}{
		{"music", s.genMusic.open, s.musicDir()},
		{"images", s.genImageJobs.open, s.imageDir()},
		{"videos", s.genVideoJobs.open, s.videoDir()},
	} {
		if err := st.open(filepath.Join(dir, st.name), st.outDir, now); err != nil {
			slog.Error("generation jobs not loaded", "kind", st.name, "error", err)
		}
	}
}

// watchGenJobs prunes old image, video and music jobs for the life of the
// server, starting at once so a restart clears what is already old.
func (s *Server) watchGenJobs() {
	prune := func(now time.Time) {
		s.genMusic.prune(now)
		s.genImageJobs.prune(now)
		s.genVideoJobs.prune(now)
	}
	prune(time.Now())
	for now := range time.Tick(jobPruneEvery) {
		prune(now)
	}
}

// jobStatusFilterError is a list route's answer to an unknown ?status=.
const jobStatusFilterError = "status must be a comma-separated list of queued, running, succeeded, failed and cancelled"

// jobListResponse is a list route's answer.
type jobListResponse[T any] struct {
	Object string `json:"object"`
	Data   []T    `json:"data"`
}

func jobList[T any](jobs []T) jobListResponse[T] {
	return jobListResponse[T]{Object: "list", Data: jobs}
}

// jobStatusFilter reads a list route's ?status=, a comma-separated list of
// job statuses. It reports false for a status that does not exist.
func jobStatusFilter(q string) ([]string, bool) {
	if q == "" {
		return nil, true
	}
	var out []string
	for _, st := range strings.Split(q, ",") {
		st = strings.TrimSpace(st)
		switch st {
		case jobQueued, jobRunning, jobSucceeded, jobFailed, jobCancelled:
			out = append(out, st)
		default:
			return nil, false
		}
	}
	return out, true
}

func newGenJobID() string {
	var b [8]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
