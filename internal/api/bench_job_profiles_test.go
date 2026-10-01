package api

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func profileJobRequest(starts map[string][]string) jobCreateRequest {
	return jobCreateRequest{
		Name:     "compare",
		ModelIDs: []string{profTestID},
		BuildIDs: []string{"b1"},
		Presets:  []string{"internal-quick"},
		Starts:   starts,
	}
}

func TestValidateStarts(t *testing.T) {
	ub := 512
	cases := []struct {
		name    string
		req     jobCreateRequest
		wantErr string
	}{
		{"current settings only", profileJobRequest(map[string][]string{profTestID: {""}}), ""},
		{"profiles", profileJobRequest(map[string][]string{profTestID: {"", "Fast"}}), ""},
		{"no starts", profileJobRequest(nil), ""},
		{"model not chosen", profileJobRequest(map[string][]string{"other": {"Fast"}}), "not one of the job's models"},
		{"empty list", profileJobRequest(map[string][]string{profTestID: {}}), "no starting point"},
		{"listed twice", profileJobRequest(map[string][]string{profTestID: {"Fast", "Fast"}}), "twice"},
	}
	withSweep := profileJobRequest(map[string][]string{profTestID: {"Fast"}})
	withSweep.Sweeps = []benchmark.SweepAxis{{Field: "ubatch_size", Values: []string{"256", "512"}}}
	cases = append(cases, struct {
		name    string
		req     jobCreateRequest
		wantErr string
	}{"profile with a sweep", withSweep, "parameters cannot be set or swept"})
	withFixed := profileJobRequest(map[string][]string{profTestID: {"Fast"}})
	withFixed.Overrides = &benchmark.ConfigOverrides{UBatchSize: &ub}
	cases = append(cases, struct {
		name    string
		req     jobCreateRequest
		wantErr string
	}{"profile with a fixed value", withFixed, "parameters cannot be set or swept"})
	// Current settings alone keep the parameter section.
	currentWithSweep := profileJobRequest(map[string][]string{profTestID: {""}})
	currentWithSweep.Sweeps = withSweep.Sweeps
	cases = append(cases, struct {
		name    string
		req     jobCreateRequest
		wantErr string
	}{"current settings with a sweep", currentWithSweep, ""})

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateStarts(tc.req)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Errorf("unexpected error: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("error = %v, want one containing %q", err, tc.wantErr)
			}
		})
	}
}

// The cell limit counts starting points, not models.
func TestCellLimitCountsStartingPoints(t *testing.T) {
	req := profileJobRequest(map[string][]string{profTestID: {"", "A", "B"}})
	if got := startCount(req); got != 3 {
		t.Errorf("startCount = %d, want 3", got)
	}
	req.Starts = nil
	if got := startCount(req); got != 1 {
		t.Errorf("startCount without starts = %d, want 1", got)
	}
}

func TestCopyJobProfiles(t *testing.T) {
	s := newProfileServer(t, "b1")
	if _, err := s.registry.SaveProfile(profTestID, "Fast", models.ProfileSourceUser, "b0"); err != nil {
		t.Fatal(err)
	}

	req := profileJobRequest(map[string][]string{profTestID: {"", "fast"}})
	copies, err := s.copyJobProfiles(&req)
	if err != nil {
		t.Fatal(err)
	}
	if len(copies) != 1 || copies[0].Name != "Fast" || copies[0].ModelID != profTestID || copies[0].BuildID != "b0" {
		t.Fatalf("copies = %+v, want one copy of Fast saved on b0", copies)
	}
	if copies[0].CopiedAt.IsZero() {
		t.Error("the copy has no copy time")
	}
	// The name is rewritten to the stored spelling, so the cells and the
	// copy match.
	if req.Starts[profTestID][1] != "Fast" {
		t.Errorf("start name = %q, want the stored spelling \"Fast\"", req.Starts[profTestID][1])
	}

	missing := profileJobRequest(map[string][]string{profTestID: {"Gone"}})
	if _, err := s.copyJobProfiles(&missing); err == nil || !strings.Contains(err.Error(), "Gone") {
		t.Errorf("err = %v, want a refusal naming the missing profile", err)
	}
}

// The form offers each model's saved profiles under it, and labels the
// current settings with the profile they came from.
func TestJobFormListsProfiles(t *testing.T) {
	s := newProfileServer(t, "b1")
	cfg, err := s.registry.GetConfig(profTestID)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Enabled = true
	if err := s.registry.SetConfig(profTestID, cfg); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"Fast", "Long context"} {
		if _, err := s.registry.SaveProfile(profTestID, name, models.ProfileSourceUser, "b1"); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.registry.ApplyProfile(profTestID, "Fast"); err != nil {
		t.Fatal(err)
	}

	rec := httptest.NewRecorder()
	s.handleJobForm(rec, httptest.NewRequest("GET", "/api/benchmark-jobs/form", nil))
	out := rec.Body.String()
	for _, want := range []string{
		`name="start" value=""`,
		`value="Fast"`,
		`value="Long context"`,
		"Current settings (Fast)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("form is missing %q", want)
		}
	}
}

// A job comparing saved profiles gains a Profile column and a line naming
// the copies it measures; any other job has neither.
func TestJobDetailProfileColumn(t *testing.T) {
	s := benchListServer(t)
	copied := time.Date(2026, 9, 30, 14, 5, 0, 0, time.UTC)
	job := &benchmark.BenchmarkJob{
		ID: "job-p",
		Profiles: []benchmark.JobProfile{
			{ModelID: "m4", Name: "Fast", CopiedAt: copied},
		},
		Cells: []benchmark.JobCell{
			{ModelID: "m4", BuildID: "b1", Preset: "internal-quick", Status: benchmark.CellStatusPending},
			{ModelID: "m4", BuildID: "b1", Preset: "internal-quick", Profile: "Fast", Status: benchmark.CellStatusPending},
		},
	}
	rec := httptest.NewRecorder()
	s.renderJobDetail(rec, job)
	out := rec.Body.String()
	for _, want := range []string{">Profile</th>", "<small>Current settings</small>", "<small>Fast</small>",
		"Compares saved profiles:", "copied Sep 30 14:05"} {
		if !strings.Contains(out, want) {
			t.Errorf("job detail is missing %q\n%s", want, out)
		}
	}

	plain := &benchmark.BenchmarkJob{ID: "job-q", Cells: job.Cells[:1]}
	rec = httptest.NewRecorder()
	s.renderJobDetail(rec, plain)
	if strings.Contains(rec.Body.String(), ">Profile</th>") {
		t.Error("a job without saved profiles shows the Profile column")
	}
}
