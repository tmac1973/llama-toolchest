package api

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// readPreset returns the contents of a preset file in the server's
// config directory.
func readPreset(t *testing.T, s *Server, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(s.cfg.DataDir, "config", name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(data)
}

// A benchmark cell runs under a substitute config, but the user's saved
// config is never touched: not in memory, not in models.json, not in
// preset.ini. When the job ends, the router goes back to the saved
// config. This goes through the production jobEnv and a real
// process.Manager running a fake llama-server.
func TestEphemeralConfigKeepsTheSavedConfig(t *testing.T) {
	s, proc := evalStopServer(t)
	const id = "m1"
	if err := s.registry.Add(&models.Model{
		ID: id, ModelID: "org/M-GGUF", Quant: "Q4_K_M", Filename: "m.gguf",
		FilePath: filepath.Join(t.TempDir(), "m.gguf"),
	}); err != nil {
		t.Fatal(err)
	}
	saved := models.DefaultConfig()
	saved.ContextSize = 8192
	if err := s.registry.SetConfig(id, &saved); err != nil {
		t.Fatal(err)
	}
	if err := s.startRouter(); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)

	env := newJobEnv(s)
	snap := benchmark.SnapshotFromConfig(saved, "", false)
	snap.ContextSize = 4096
	if err := env.ApplyEphemeralConfig(context.Background(), id, snap, nil); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if !env.routerOwnedByJob() {
		t.Error("the job does not own the router after applying a config")
	}
	if live, ok := s.runningConfigFor(id); !ok || live.ContextSize != 4096 {
		t.Errorf("router is not running the substitute config: %+v", live)
	}
	if !strings.Contains(readPreset(t, s, models.BenchPresetFileName), "ctx-size = 4096") {
		t.Error("benchmark preset does not carry the substitute context size")
	}

	// The saved config, in memory and on disk, is the user's.
	checkSaved := func(when string) {
		t.Helper()
		if cfg, _ := s.registry.GetConfig(id); cfg.ContextSize != 8192 {
			t.Errorf("%s: saved context size = %d, want 8192", when, cfg.ContextSize)
		}
		onDisk := models.NewRegistry(s.cfg.DataDir, "/models")
		if cfg, err := onDisk.GetConfig(id); err != nil || cfg.ContextSize != 8192 {
			t.Errorf("%s: models.json context size = %v (err %v), want 8192", when, cfg, err)
		}
		preset := readPreset(t, s, models.PresetFileName)
		if !strings.Contains(preset, "ctx-size = 8192") || strings.Contains(preset, "ctx-size = 4096") {
			t.Errorf("%s: preset.ini does not hold the saved config:\n%s", when, preset)
		}
	}
	checkSaved("during the job")

	if err := env.ClearEphemeralConfig(context.Background()); err != nil {
		t.Fatalf("clear: %v", err)
	}
	waitRouterRunning(t, proc)
	if env.routerOwnedByJob() {
		t.Error("the job still owns the router after cleanup")
	}
	if live, ok := s.runningConfigFor(id); !ok || live.ContextSize != 8192 {
		t.Errorf("router is not back on the saved config: %+v", live)
	}
	checkSaved("after the job")
}

// configDiff is the log line that answers "did my override reach
// llama-server". It must name each changed field with both values, in a
// stable order, and say "none" rather than log an empty list.
func TestConfigDiff(t *testing.T) {
	base := models.ModelConfig{ContextSize: 8192, FlashAttention: true, TensorSplit: ""}
	if got := configDiff(base, base); !reflect.DeepEqual(got, []string{"none"}) {
		t.Errorf("no change: %v, want [none]", got)
	}

	merged := base
	merged.ContextSize = 4096
	merged.FlashAttention = false
	merged.TensorSplit = "1,1"
	merged.DraftPMin = "0.75"
	want := []string{
		"ctx-size 8192→4096",
		"flash-attn true→false",
		"tensor-split →1,1",
		"draft-p-min →0.75",
	}
	if got := configDiff(base, merged); !reflect.DeepEqual(got, want) {
		t.Errorf("configDiff = %q, want %q", got, want)
	}
}

// A job records the build each run used. A known build is copied in full;
// an unknown id (a build deleted since the job was queued) gives an empty
// record rather than a failure or another build's details.
func TestJobEnvResolveBuild(t *testing.T) {
	s := newTestServer(t)
	b := builder.BuildResult{
		ID: "b1", Tag: "fast", Profile: "rocm", GitSHA: "abc123", GitRef: "b5000",
		Status: builder.BuildStatusSuccess, BinaryPath: "/builds/b1/llama-server",
		CMakeFlags: map[string]string{"GGML_HIP": "ON"},
	}
	s.builder = testBuilder(t, b)
	env := newJobEnv(s)

	want := benchmark.BuildSnapshot{
		ID: "b1", Tag: "fast", Profile: "rocm", Vendor: "rocm", GitSHA: "abc123",
		GitRef: "b5000", CMakeFlags: map[string]string{"GGML_HIP": "ON"},
		BinaryPath: "/builds/b1/llama-server",
	}
	if got := env.ResolveBuild("b1"); !reflect.DeepEqual(got, want) {
		t.Errorf("ResolveBuild(b1) = %+v, want %+v", got, want)
	}
	if got := env.ResolveBuild("gone"); !reflect.DeepEqual(got, benchmark.BuildSnapshot{}) {
		t.Errorf("ResolveBuild(gone) = %+v, want an empty snapshot", got)
	}
}

// The backend decides how GPU placement flags are written for an
// evaluation. It comes from the build's profile; a profile name that is
// not a built-in one is used as written, and an unknown build gives ""
// (the generic placement).
func TestJobEnvBuildBackend(t *testing.T) {
	s := newTestServer(t)
	s.builder = testBuilder(t,
		builder.BuildResult{ID: "r", Profile: "rocm", Status: builder.BuildStatusSuccess},
		builder.BuildResult{ID: "c", Profile: "cuda", Status: builder.BuildStatusSuccess},
		builder.BuildResult{ID: "x", Profile: "my-custom", Status: builder.BuildStatusSuccess},
	)
	env := newJobEnv(s)
	cases := map[string]string{"r": "rocm", "c": "cuda", "x": "my-custom", "gone": ""}
	for id, want := range cases {
		if got := env.buildBackend(id); got != want {
			t.Errorf("buildBackend(%q) = %q, want %q", id, got, want)
		}
	}
}
