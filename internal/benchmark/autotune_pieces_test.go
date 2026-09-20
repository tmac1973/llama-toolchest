package benchmark

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The code workload sends the source file to edit, and pads with a second
// file rather than the same one twice, so the edit stays unambiguous.
func TestCodePromptCarriesTheFile(t *testing.T) {
	p := buildPromptFor("nonce", 1536, 1, PromptStyleCode)
	for _, want := range []string{"Rename the method `add_item`", "class Inventory", "def add_item"} {
		if !strings.Contains(p, want) {
			t.Errorf("code prompt missing %q", want)
		}
	}
	if len(p) < 1536*BenchPromptCharsPerToken/2 {
		t.Errorf("code prompt is %d characters, far short of the target", len(p))
	}
	long := buildPromptFor("nonce", 4096, 1, PromptStyleCode)
	if !strings.Contains(long, "inventory_v2.py") {
		t.Error("a longer code prompt should add a second file, not repeat the first")
	}
	if strings.Contains(p, BenchPromptText[:40]) {
		t.Error("the code prompt is padded with prose")
	}
}

// Autotune's presets are comparable across its own runs, and stay out of
// the pickers.
func TestAutotunePresetsAreHidden(t *testing.T) {
	for _, name := range []string{"autotune-chat", "autotune-code"} {
		p := GetPreset(name)
		if p.Name != name {
			t.Fatalf("%s is not registered; GetPreset fell back to %q", name, p.Name)
		}
		if !p.Hidden {
			t.Errorf("%s is offered in the pickers", name)
		}
		if p.Repetitions < 3 {
			t.Errorf("%s has %d repetitions; the noise rule needs a spread", name, p.Repetitions)
		}
	}
	if GetPreset("autotune-code").PromptStyle != PromptStyleCode {
		t.Error("autotune-code does not use the code workload")
	}
	for _, p := range VisiblePresets() {
		if strings.HasPrefix(p.Name, "autotune-") {
			t.Errorf("%s is in the visible list", p.Name)
		}
	}
}

// A spec value can name the draft file it loads, and the value survives a
// round trip through the parser and the encoder.
func TestSpecValueCarriesADraftModel(t *testing.T) {
	const id = "unsloth--Qwen3.5-0.8B-GGUF--Qwen3.5-0.8B-Q8_0"
	value := EncodeSpecValue("draft", "ngram-mod", map[string]string{
		SpecDraftModelKey: id, "draft_max": "16", "assist_n_max": "",
	})
	if value != "draft+ngram-mod:draft_max=16,draft_model="+id {
		t.Fatalf("encoded %q", value)
	}
	if got := canonicalSpecValue("ngram-mod+draft:draft_model=" + id + ",draft_max=16"); got != value {
		t.Errorf("canonical form = %q, want %q", got, value)
	}

	var ov ConfigOverrides
	if err := applySpecValue(&ov, value); err != nil {
		t.Fatal(err)
	}
	if ov.DraftModelPath == nil || *ov.DraftModelPath != id {
		t.Errorf("draft model = %v, want the unresolved ID", ov.DraftModelPath)
	}

	// A value naming a file for a slot it has no draft method for is a
	// mistake worth catching.
	if err := applySpecValue(&ConfigOverrides{}, "ngram-mod:draft_model="+id); err == nil {
		t.Error("draft_model accepted for an n-gram-only value")
	}
}

// The draft file is resolved before a cell runs: an ID becomes a path,
// and it lands on the field the method actually reads.
func TestResolveDraftFile(t *testing.T) {
	resolve := func(id string) (string, error) { return "/models/" + id + ".gguf", nil }

	cfg := models.ModelConfig{SpecType: "draft", DraftModelPath: "small-model"}
	if err := ResolveDraftFile(&cfg, resolve); err != nil {
		t.Fatal(err)
	}
	if cfg.DraftModelPath != "/models/small-model.gguf" || cfg.MtpPath != "" {
		t.Errorf("draft method: %+v", cfg)
	}

	mtp := models.ModelConfig{SpecType: "draft-mtp", DraftModelPath: "head-model"}
	if err := ResolveDraftFile(&mtp, resolve); err != nil {
		t.Fatal(err)
	}
	if mtp.MtpPath != "/models/head-model.gguf" || mtp.DraftModelPath != "" {
		t.Errorf("draft-mtp loads a head through MtpPath: %+v", mtp)
	}

	abs := models.ModelConfig{SpecType: "draft", DraftModelPath: "/models/given.gguf"}
	if err := ResolveDraftFile(&abs, nil); err != nil || abs.DraftModelPath != "/models/given.gguf" {
		t.Errorf("a path should be used as it is: %+v, %v", abs, err)
	}

	missing := models.ModelConfig{SpecType: "draft", DraftModelPath: "gone"}
	err := ResolveDraftFile(&missing, func(string) (string, error) { return "", errNotFound })
	if err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Errorf("err = %v, want a not-installed error", err)
	}
}

var errNotFound = errorString("not found")

type errorString string

func (e errorString) Error() string { return string(e) }

// ConfigForValues is what a cell ran, which is what a profile saving that
// cell's settings has to hold.
func TestConfigForValues(t *testing.T) {
	base := models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8, UBatchSize: 512}
	got, err := ConfigForValues(base, map[string]string{
		"ubatch_size": "1024",
		"spec_type":   "draft-mtp+ngram-mod:draft_max=3",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.UBatchSize != 1024 || got.SpecType != "draft-mtp" || got.SpecAssist != "ngram-mod" || got.DraftMax != 3 {
		t.Errorf("config = %+v", got)
	}
	if got.ContextSize != 8192 || !got.Enabled {
		t.Errorf("the settings a sweep did not touch changed: %+v", got)
	}
}

// Every snapshot field with a matching config field reaches the config a
// cell launches.
func TestApplySnapshotToConfigCoversEveryField(t *testing.T) {
	skip := map[string]bool{"ProfileName": true, "ProfileEdited": true, "GPUAssign": true, "TensorSplit": true}
	snapType := reflect.TypeOf(ConfigSnapshot{})
	cfgType := reflect.TypeOf(models.ModelConfig{})
	for i := 0; i < snapType.NumField(); i++ {
		f := snapType.Field(i)
		if skip[f.Name] {
			continue
		}
		if _, ok := cfgType.FieldByName(f.Name); !ok {
			continue
		}
		var snap ConfigSnapshot
		v := reflect.ValueOf(&snap).Elem().Field(i)
		switch v.Kind() {
		case reflect.Bool:
			v.SetBool(true)
		case reflect.Int:
			v.SetInt(7)
		case reflect.String:
			v.SetString("x")
		default:
			t.Fatalf("%s: unhandled kind %s", f.Name, v.Kind())
		}
		out := ApplySnapshotToConfig(models.ModelConfig{}, snap)
		if reflect.ValueOf(out).FieldByName(f.Name).IsZero() {
			t.Errorf("ApplySnapshotToConfig drops %s", f.Name)
		}
	}
}

// Wait returns the finished job without polling, and gives up when its
// caller does.
func TestJobQueueWait(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8}}
	store := NewStore(t.TempDir(), nil)
	q := NewJobQueue(store, env)

	if err := q.Submit(oneCellJob(nil)); err != nil {
		t.Fatal(err)
	}
	job, err := q.Wait(context.Background(), "job-1")
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != JobStatusCompleted {
		t.Errorf("status = %s, want completed", job.Status)
	}

	// A job this process never ran answers from the store.
	if _, err := q.Wait(context.Background(), "job-1"); err != nil {
		t.Errorf("waiting on a finished job: %v", err)
	}
	if _, err := q.Wait(context.Background(), "never-submitted"); err == nil {
		t.Error("waiting on an unknown job should report it is unknown")
	}

	// A cancelled wait returns, leaving the job alone.
	router2 := newFakeRouter(t)
	env2 := &fakeEnv{routerURL: router2.URL, saved: env.saved}
	q2 := NewJobQueue(NewStore(t.TempDir(), nil), env2)
	slow := oneCellJob(nil)
	slow.ID = "job-slow"
	if err := q2.Submit(slow); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := q2.Wait(ctx, "job-slow"); err == nil {
		t.Error("a cancelled wait should return its context's error")
	}
	// Giving up on the wait does not stop the job: let it finish before
	// the test's directories go away under it.
	if _, err := q2.Wait(context.Background(), "job-slow"); err != nil {
		t.Errorf("waiting for the job to finish: %v", err)
	}
}

// A job measuring a saved profile runs that profile's settings, sends its
// sampling values, and leaves the live config alone.
func TestJobRunsAgainstABaseProfile(t *testing.T) {
	router := newFakeRouter(t)
	live := ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8, UBatchSize: 512}
	env := &fakeEnv{routerURL: router.URL, saved: live}

	temp := 0.6
	profile := models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8,
		UBatchSize: 1024, Jinja: true, Temperature: &temp}
	job := oneCellJob(nil)
	job.BaseProfile = &BaseProfile{Name: "Fast", Config: profile}

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("status = %s", done.Status)
	}
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("runs = %d", len(runs))
	}
	if runs[0].Config.UBatchSize != 1024 {
		t.Errorf("recorded ubatch = %d, want the profile's 1024", runs[0].Config.UBatchSize)
	}
	if runs[0].Config.ProfileName != "Fast" || runs[0].Config.ProfileEdited {
		t.Errorf("recorded profile = %q edited=%v", runs[0].Config.ProfileName, runs[0].Config.ProfileEdited)
	}
	// The first request is the warm-up, which deliberately carries no
	// sampling; the measured ones do.
	bodies := router.completionBodies()
	if len(bodies) < 2 {
		t.Fatalf("recorded %d completion requests", len(bodies))
	}
	last := bodies[len(bodies)-1]
	if got, ok := last["temperature"].(float64); !ok || got != 0.6 {
		t.Errorf("temperature sent = %v, want the profile's 0.6", last["temperature"])
	}
	if env.appliedBase == nil || env.appliedBase.UBatchSize != 1024 {
		t.Errorf("the ephemeral config was not built from the profile: %+v", env.appliedBase)
	}
}

// A cell that only switches the speculative method must not move the
// model's own draft model into the MTP slot: draft-mtp loads a head, and
// the profile's draft model is not one.
func TestDraftFileOnlyMovesWhenTheCellNamedIt(t *testing.T) {
	base := models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192,
		SpecType: "draft", DraftModelPath: "/models/small-draft.gguf", MtpPath: "/models/head.gguf"}

	// The cell switches to MTP without naming a file.
	got, err := ConfigForValues(base, map[string]string{"spec_type": "draft-mtp:draft_max=6"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.MtpPath != "/models/head.gguf" {
		t.Errorf("MtpPath = %q, want the profile's own head", got.MtpPath)
	}
	if got.DraftModelPath != "/models/small-draft.gguf" {
		t.Errorf("DraftModelPath = %q, want the profile's own draft model", got.DraftModelPath)
	}

	// A cell that does name one still resolves it into the right slot.
	got, err = ConfigForValues(base, map[string]string{"spec_type": "draft-mtp:draft_model=other-head"},
		func(id string) (string, error) { return "/models/" + id + ".gguf", nil })
	if err != nil {
		t.Fatal(err)
	}
	if got.MtpPath != "/models/other-head.gguf" {
		t.Errorf("MtpPath = %q, want the head the cell named", got.MtpPath)
	}
}
