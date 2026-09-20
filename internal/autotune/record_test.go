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

// The store and the runner must not share a record: the runner mutates
// its copy for minutes at a time while the screens read what is stored.
func TestStoreHandsOutCopies(t *testing.T) {
	s := NewStore(t.TempDir())
	rec := newRec("at-1", "m", time.Now())
	rec.SetStage(StageRecord{Key: StageBatch, Status: StatusDone})
	if err := s.Save(rec); err != nil {
		t.Fatal(err)
	}

	// Mutating the record that was saved must not change what is stored.
	rec.Status = StatusRunning
	rec.Stages[0].Status = StatusRunning
	rec.Skipped = append(rec.Skipped, "later")
	stored, _ := s.Get("at-1")
	if stored.Status != StatusPlanned || stored.Stages[0].Status != StatusDone || len(stored.Skipped) != 0 {
		t.Errorf("the store shares its record with the caller: %+v", stored)
	}

	// And a copy handed out cannot be changed from under a later reader.
	stored.Status = StatusFailed
	again, _ := s.Get("at-1")
	if again.Status != StatusPlanned {
		t.Errorf("two readers share one record: %s", again.Status)
	}
}

// The model card has to show that a run finished. Without it a run that
// ended while the user was on another page left no trace: the button
// still said "Autotune", and the results were only reachable by pressing
// it again on the chance that something was behind it.
func TestCardStateSaysWhatHappened(t *testing.T) {
	saved := Outcome{Saved: true, ProfileName: "Autotune – fastest generation and response"}
	cases := []struct {
		name       string
		rec        *Autotune
		wantButton string
		wantIn     string
	}{
		{"never measured", nil, "Autotune", ""},
		{"running", &Autotune{Status: StatusRunning}, "Autotune running", "keeps going"},
		{"finished with a profile", &Autotune{Status: StatusDone, Results: map[Goal]Outcome{
			GoalGeneration: saved, GoalResponse: saved,
		}}, "Autotune results", "saved 1 faster profile"},
		{"finished with nothing to change", &Autotune{Status: StatusDone, Results: map[Goal]Outcome{
			GoalGeneration: {Saved: false},
		}}, "Autotune results", "already the fastest"},
		{"failed", &Autotune{Status: StatusFailed}, "Resume", "see why"},
		{"cancelled", &Autotune{Status: StatusCancelled}, "Resume", "continue"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.rec.Card()
			if got.ButtonLabel != tc.wantButton {
				t.Errorf("ButtonLabel = %q, want %q", got.ButtonLabel, tc.wantButton)
			}
			if tc.wantIn == "" {
				if got.Summary != "" {
					t.Errorf("Summary = %q, want none", got.Summary)
				}
				return
			}
			if !strings.Contains(got.Summary, tc.wantIn) {
				t.Errorf("Summary = %q, want it to mention %q", got.Summary, tc.wantIn)
			}
		})
	}
}

// Two goals won by the same profile count as one saved profile, not two.
func TestCardStateCountsProfilesNotGoals(t *testing.T) {
	one := Outcome{Saved: true, ProfileName: "Autotune – fastest generation and response"}
	rec := &Autotune{Status: StatusDone, Results: map[Goal]Outcome{
		GoalGeneration: one,
		GoalResponse:   one,
		GoalPrompt:     {Saved: true, ProfileName: "Autotune – fastest prompt"},
	}}
	if got := rec.Card().Saved; got != 2 {
		t.Errorf("Saved = %d, want 2 distinct profiles", got)
	}
}
