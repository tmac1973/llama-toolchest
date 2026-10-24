package generation

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

func mingSet() ModelSet {
	return ModelSet{
		ID: "ming-image", Kind: KindSD, Recipe: "ming-image", Complete: true,
		Components: []Component{
			{Slot: "diffusion_model", Path: "/m/dit.gguf"},
			{Slot: "llm", Path: "/m/ling.gguf"},
			{Slot: "tokenizer", Path: "/m/tokenizer.json"},
			{Slot: "vae", Path: "/m/vae.safetensors"},
		},
		LaunchArgs: []string{"--steps", "12"},
	}
}

func envValue(env []string, key string) (string, int) {
	var v string
	n := 0
	for _, kv := range env {
		if val, ok := strings.CutPrefix(kv, key+"="); ok {
			v, n = val, n+1
		}
	}
	return v, n
}

func TestLaunchSpecSD(t *testing.T) {
	t.Setenv("HIP_VISIBLE_DEVICES", "0,1") // inherited; must be replaced
	data := t.TempDir()
	inst := Instance{ID: "ming", Port: 8081, GPU: 1, ReservedMiB: 10240, ExtraArgs: []string{"--offload-to-cpu"}}
	build := builder.BuildResult{ID: "sd-x-rocm", Profile: "rocm", BinaryPath: "/b/sd-x-rocm/sd-server"}

	cfg, err := LaunchSpec(inst, mingSet(), build, "0.0.0.0", data, LaunchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--listen-ip", "0.0.0.0", "--listen-port", "8081",
		"--diffusion-model", "/m/dit.gguf", "--llm", "/m/ling.gguf", "--tokenizer", "/m/tokenizer.json", "--vae", "/m/vae.safetensors",
		"--steps", "12", "--offload-to-cpu", "--max-vram", "10.0"}
	if !slices.Equal(cfg.Args, want) {
		t.Errorf("args =\n%q\nwant\n%q", cfg.Args, want)
	}
	if v, n := envValue(cfg.Env, "HIP_VISIBLE_DEVICES"); v != "1" || n != 1 {
		t.Errorf("HIP_VISIBLE_DEVICES = %q (%d entries), want exactly 1", v, n)
	}
	if cfg.HealthURL != "http://127.0.0.1:8081/v1/models" || cfg.BinaryPath != build.BinaryPath {
		t.Errorf("health %q binary %q", cfg.HealthURL, cfg.BinaryPath)
	}
	if !strings.Contains(strings.Join(cfg.Env, " "), "/b/sd-x-rocm") {
		t.Error("the build directory is not on the library path")
	}

	cfg, _ = LaunchSpec(inst, mingSet(), build, "0.0.0.0", data, LaunchOptions{OmitMaxVRAM: true})
	if slices.Contains(cfg.Args, "--max-vram") {
		t.Error("OmitMaxVRAM kept --max-vram")
	}
}

func TestLaunchSpecDeviceVariablePerProfile(t *testing.T) {
	for profile, want := range map[string]string{"cuda": "CUDA_VISIBLE_DEVICES", "vulkan": "GGML_VK_VISIBLE_DEVICES"} {
		cfg, err := LaunchSpec(Instance{ID: "x", Port: 8082, GPU: 2}, mingSet(),
			builder.BuildResult{ID: "b", Profile: profile, BinaryPath: "/b/sd-server"}, "", t.TempDir(), LaunchOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if v, _ := envValue(cfg.Env, want); v != "2" {
			t.Errorf("%s: %s = %q, want 2", profile, want, v)
		}
	}
	if _, err := LaunchSpec(Instance{ID: "x", Port: 8082}, mingSet(),
		builder.BuildResult{ID: "b", Profile: "cpu"}, "", t.TempDir(), LaunchOptions{}); err == nil {
		t.Error("a cpu build was accepted")
	}
}

func TestLaunchSpecAceLinksItsFiles(t *testing.T) {
	files := t.TempDir()
	set := ModelSet{ID: "ace", Kind: KindAce, Complete: true}
	for _, slot := range AceSlots {
		p := filepath.Join(files, slot+".gguf")
		if err := os.WriteFile(p, []byte(slot), 0o644); err != nil {
			t.Fatal(err)
		}
		set.Components = append(set.Components, Component{Slot: slot, Path: p})
	}
	// Another GGUF beside them must not reach ace-server.
	os.WriteFile(filepath.Join(files, "other.gguf"), []byte("x"), 0o644)
	data := t.TempDir()

	cfg, err := LaunchSpec(Instance{ID: "music", Port: 8083, ExtraArgs: []string{"--no-fa"}}, set,
		builder.BuildResult{ID: "ace-b", Profile: "rocm", BinaryPath: "/b/ace-server"}, "0.0.0.0", data, LaunchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(InstanceDir(data, "music"), "models")
	want := []string{"--host", "0.0.0.0", "--port", "8083", "--models", dir, "--max-batch", "1", "--no-fa"}
	if !slices.Equal(cfg.Args, want) {
		t.Errorf("args = %q, want %q", cfg.Args, want)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 4 {
		t.Errorf("models dir has %d entries, want the set's 4", len(entries))
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "dit.gguf")); string(got) != "dit" {
		t.Errorf("dit link reads %q", got)
	}
	if cfg.HealthURL != "http://127.0.0.1:8083/health" {
		t.Errorf("health = %q", cfg.HealthURL)
	}
}

func TestLaunchSpecRefusals(t *testing.T) {
	incomplete := mingSet()
	incomplete.Complete = false
	build := builder.BuildResult{ID: "b", Profile: "rocm", BinaryPath: "/b/sd-server"}
	if _, err := LaunchSpec(Instance{ID: "x", Port: 8081}, incomplete, build, "", t.TempDir(), LaunchOptions{}); err == nil {
		t.Error("an incomplete set was accepted")
	}
	if _, err := LaunchSpec(Instance{ID: "x"}, mingSet(), build, "", t.TempDir(), LaunchOptions{}); err == nil {
		t.Error("an instance without a port was accepted")
	}
}

func TestMaxVRAMGiB(t *testing.T) {
	for mib, want := range map[int]string{10240: "10.0", 11776: "11.5", 8191: "7.9", 512: "0.5"} {
		if got := maxVRAMGiB(mib); got != want {
			t.Errorf("maxVRAMGiB(%d) = %s, want %s", mib, got, want)
		}
	}
}

func TestCheckGPUVendor(t *testing.T) {
	if err := CheckGPUVendor("rocm", "nvidia"); err == nil || !strings.Contains(err.Error(), "needs AMD GPUs") {
		t.Errorf("rocm on nvidia: %v", err)
	}
	if err := CheckGPUVendor("cuda", "rocm"); err == nil {
		t.Error("cuda on AMD accepted")
	}
	for _, ok := range [][2]string{{"rocm", "rocm"}, {"cuda", "nvidia"}, {"vulkan", "nvidia"}, {"vulkan", "rocm"}, {"rocm", ""}} {
		if err := CheckGPUVendor(ok[0], ok[1]); err != nil {
			t.Errorf("%s on %q: %v", ok[0], ok[1], err)
		}
	}
}

// The manager runs, logs and stops instances; StopAll stops them all.
func TestManagerLifecycle(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake")
	}
	health := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer health.Close()
	bin := filepath.Join(t.TempDir(), "fake-sd-server")
	os.WriteFile(bin, []byte("#!/bin/sh\necho fake backend up\nexec sleep 30\n"), 0o755)
	cfg := func() process.SupervisorConfig {
		return process.SupervisorConfig{Name: "fake", BinaryPath: bin, Env: os.Environ(), HealthURL: health.URL,
			HealthInterval: 20 * time.Millisecond, StopTimeout: 5 * time.Second}
	}

	m := NewManager()
	logs := m.Subscribe("a")
	if logs != nil {
		t.Error("Subscribe before the first run should return nil")
	}
	for _, id := range []string{"a", "b"} {
		if err := m.Start(id, KindSD, cfg()); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.WaitRunning(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if !m.IsActive("a") || m.PID("a") == 0 {
		t.Error("instance a not active after start")
	}
	if logs = m.Subscribe("a"); logs == nil {
		t.Fatal("no log stream for a running instance")
	}
	select {
	case line := <-logs:
		if line != "fake backend up" {
			t.Errorf("first log line %q", line)
		}
	case <-time.After(5 * time.Second):
		t.Error("no log line")
	}
	m.Unsubscribe("a", logs)

	m.StopAll()
	for _, id := range []string{"a", "b"} {
		if m.IsActive(id) {
			t.Errorf("%s still active after StopAll", id)
		}
	}
	if err := m.Stop("a"); err != nil {
		t.Errorf("stopping a stopped instance: %v", err)
	}
	if st := m.Status("never"); st.State != process.StateStopped {
		t.Errorf("unknown instance state %q", st.State)
	}
}
