package autotune

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newRec(id, model string, created time.Time) *Autotune {
	return &Autotune{ID: id, ModelID: model, BaseProfile: "Autoconfig",
		UseCase: UseCode, Status: StatusPlanned, CreatedAt: created}
}

func TestStoreSaveGetList(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	older := newRec("at-1", "m1", time.Now().Add(-time.Hour))
	newer := newRec("at-2", "m1", time.Now())
	for _, r := range []*Autotune{older, newer} {
		if err := s.Save(r); err != nil {
			t.Fatal(err)
		}
	}
	if got, ok := s.Get("at-1"); !ok || got.ModelID != "m1" {
		t.Fatalf("Get = %v, %v", got, ok)
	}
	if list := s.List(); len(list) != 2 || list[0].ID != "at-2" {
		t.Errorf("List = %v, want newest first", ids(s.List()))
	}
	latest, ok := s.LatestForModel("m1")
	if !ok || latest.ID != "at-2" {
		t.Errorf("LatestForModel = %v", latest)
	}
	if _, ok := s.LatestForModel("other"); ok {
		t.Error("a model with no runs reported one")
	}

	// It survives a restart, and an unfinished run is marked so it can
	// be resumed rather than looking like it is still going.
	newer.Status = StatusRunning
	if err := s.Save(newer); err != nil {
		t.Fatal(err)
	}
	again := NewStore(dir)
	got, _ := again.Get("at-2")
	if got == nil || got.Status != StatusInterrupted {
		t.Errorf("status after a restart = %v, want interrupted", got)
	}
}

func TestStoreActive(t *testing.T) {
	s := NewStore(t.TempDir())
	if _, ok := s.Active(); ok {
		t.Error("an empty store reported an active run")
	}
	rec := newRec("at-1", "m", time.Now())
	rec.Status = StatusRunning
	s.Save(rec)
	if got, ok := s.Active(); !ok || got.ID != "at-1" {
		t.Errorf("Active = %v, %v", got, ok)
	}
	rec.Status = StatusDone
	s.Save(rec)
	if _, ok := s.Active(); ok {
		t.Error("a finished run is still reported as active")
	}
}

// The same gate as the other stores: a file this build cannot read is
// never written over.
func TestStoreReadOnlyGate(t *testing.T) {
	for name, contents := range map[string]string{
		"corrupt": `{"schema_version": 1, "runs": [`,
		"newer":   `{"schema_version": 99, "runs": [{"id": "at-1", "model_id": "m"}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "config", "autotune.json")
			os.MkdirAll(filepath.Dir(path), 0o755)
			os.WriteFile(path, []byte(contents), 0o644)

			s := NewStore(dir)
			if s.ReadOnlyReason() == "" {
				t.Fatal("the store is writable")
			}
			if err := s.Save(newRec("at-2", "m", time.Now())); !errors.Is(err, ErrStoreReadOnly) {
				t.Errorf("Save err = %v, want ErrStoreReadOnly", err)
			}
			after, _ := os.ReadFile(path)
			if string(after) != contents {
				t.Errorf("the file was changed:\n%s", after)
			}
		})
	}
}

func TestStageHelpers(t *testing.T) {
	rec := newRec("at-1", "m", time.Now())
	rec.SetStage(StageRecord{Key: StageSpec, Status: StatusDone,
		Finalists: []Candidate{{Values: map[string]string{"spec_type": "draft-mtp"}}}})
	rec.SetStage(StageRecord{Key: StageBatch, Status: StatusDone})
	if len(rec.Stages) != 2 || rec.Stages[0].Key != StageBatch {
		t.Errorf("stages are out of order: %v", stageKeys(rec))
	}
	if rec.Stages[0].Title != "Batch sizes and attention" {
		t.Errorf("stage title = %q", rec.Stages[0].Title)
	}
	rec.SetStage(StageRecord{Key: StageBatch, Status: StatusFailed})
	if len(rec.Stages) != 2 {
		t.Errorf("re-recording a stage added one: %v", stageKeys(rec))
	}
	if st, ok := rec.Stage(StageBatch); !ok || st.Status != StatusFailed {
		t.Errorf("stage = %v, %v", st, ok)
	}
	if got := rec.Finalists(StageSpec); len(got) != 1 {
		t.Errorf("finalists = %v", got)
	}
	if got := rec.Finalists(StageConfirm); got != nil {
		t.Errorf("a stage that has not run has finalists: %v", got)
	}
}

func TestStoreKeepsHistoryBounded(t *testing.T) {
	s := NewStore(t.TempDir())
	for i := 0; i < maxStoredRuns+10; i++ {
		if err := s.Save(newRec("at-"+strings.Repeat("x", i%3)+string(rune('a'+i%26))+string(rune('0'+i/26)), "m",
			time.Now().Add(time.Duration(i)*time.Minute))); err != nil {
			t.Fatal(err)
		}
	}
	if n := len(s.List()); n > maxStoredRuns {
		t.Errorf("stored %d runs, want at most %d", n, maxStoredRuns)
	}
}

func ids(runs []*Autotune) []string {
	var out []string
	for _, r := range runs {
		out = append(out, r.ID)
	}
	return out
}

func stageKeys(rec *Autotune) []string {
	var out []string
	for _, s := range rec.Stages {
		out = append(out, s.Key)
	}
	return out
}
