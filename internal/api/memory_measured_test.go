package api

import (
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// oneLoad is a single instance's report, trimmed to one line per kind:
// 20 GiB of weights and 1 GiB of them host-mapped, a 2 GiB KV cache, a
// 1 GiB compute buffer and the small output buffer that llama.cpp puts
// in host memory.
const oneLoad = `0.00.100.000 I srv          load: spawning server instance with name=m1 on port 46231
[46231] 0.14.275.753 I load_tensors:        ROCm0 model buffer size = 20480.00 MiB
[46231] 0.14.275.751 I load_tensors:   CPU_Mapped model buffer size =  1024.00 MiB
[46231] 0.16.162.940 I llama_kv_cache:      ROCm0 KV buffer size =  2048.00 MiB
[46231] 0.16.260.662 I sched_reserve:       ROCm0 compute buffer size =  1024.00 MiB
[46231] 0.17.752.569 I llama_context:   ROCm_Host output buffer size =     4.00 MiB`

func memTestServer(t *testing.T, verbosity string) *Server {
	t.Helper()
	env := map[string]string{}
	if verbosity != "" {
		env["LLAMA_ARG_LOG_VERBOSITY"] = verbosity
	}
	return &Server{
		cfg:      &config.Config{RuntimeEnv: env},
		registry: models.NewRegistry(t.TempDir(), t.TempDir()),
		monitor:  monitor.New(time.Hour), // never polled: no GPUs
		memory:   memreport.NewCollector(),
	}
}

func memTestModel() *models.Model {
	return &models.Model{
		ID:            "m1",
		ModelID:       "unsloth/Test-GGUF",
		Filename:      "test-Q4_K_M.gguf",
		SizeBytes:     21 << 30,
		Arch:          "qwen3",
		NLayers:       48,
		ContextLength: 32768,
	}
}

func feedLoad(s *Server, log string, note loadNote) {
	for _, line := range strings.Split(log, "\n") {
		if ev := s.memory.Add(line); ev.Kind == memreport.EventSpawned {
			s.memory.Annotate(ev.Port, note)
		}
	}
}

func TestMemoryTooltipReportsWhatWasAllocated(t *testing.T) {
	s := memTestServer(t, "4")
	cfg := &models.ModelConfig{ContextSize: 8192, GPULayers: 99}
	feedLoad(s, oneLoad, loadNote{modelID: "m1", cards: 1, fingerprint: memoryFingerprint(cfg)})

	got := s.memoryTooltip(memTestModel(), cfg, "loaded")

	// 20480 weights + 2048 KV + 1024 compute on the card; the host-mapped
	// weights and the output buffer are not on it.
	for _, want := range []string{
		"Measured while loading: 23.0 GiB of GPU memory",
		"20.0 weights",
		"2.0 KV cache",
		"1.0 working buffers",
		"Per GPU: ROCm0 23.0 GiB",
		"Another 1.0 GiB sits in system memory",
		"Estimated for the current settings:",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("tooltip missing %q; got:\n%s", want, got)
		}
	}
}

func TestMemoryTooltipFlagsAConfigChangedSinceTheLoad(t *testing.T) {
	s := memTestServer(t, "4")
	loaded := &models.ModelConfig{ContextSize: 8192}
	feedLoad(s, oneLoad, loadNote{modelID: "m1", cards: 1, fingerprint: memoryFingerprint(loaded)})

	// The user has since doubled the context, which the running instance
	// knows nothing about: the measurement describes the old setting.
	now := &models.ModelConfig{ContextSize: 16384}
	got := s.memoryTooltip(memTestModel(), now, "loaded")

	if !strings.Contains(got, "settings that have since changed") {
		t.Errorf("a measurement from another configuration must say so; got:\n%s", got)
	}
	if !strings.Contains(got, "23.0 GiB of GPU memory") {
		t.Errorf("the old measurement is still worth showing; got:\n%s", got)
	}
}

// Below verbosity 4 llama.cpp prints no buffer report at all, so there is
// nothing to measure. Saying why beats showing an estimate that looks
// like a measurement.
func TestMemoryTooltipExplainsTheMissingSetting(t *testing.T) {
	s := memTestServer(t, "3")
	got := s.memoryTooltip(memTestModel(), &models.ModelConfig{ContextSize: 8192}, "")

	if !strings.Contains(got, "Model loading detail") {
		t.Errorf("tooltip must name the setting that enables measurement; got:\n%s", got)
	}
	if !strings.Contains(got, "Estimated for the current settings:") {
		t.Errorf("tooltip must still offer the estimate; got:\n%s", got)
	}
}

func TestMemoryTooltipWaitsForAModelThatIsStillLoading(t *testing.T) {
	s := memTestServer(t, "4")
	got := s.memoryTooltip(memTestModel(), &models.ModelConfig{ContextSize: 8192}, "loading")
	if !strings.Contains(got, "Loading.") {
		t.Errorf("a loading model must not read as unmeasurable; got:\n%s", got)
	}
}

func TestMemoryTooltipWithVerbosityOnButNothingLoadedYet(t *testing.T) {
	s := memTestServer(t, "4")
	got := s.memoryTooltip(memTestModel(), &models.ModelConfig{ContextSize: 8192}, "")
	if !strings.Contains(got, "Not measured yet") {
		t.Errorf("got:\n%s", got)
	}
	if strings.Contains(got, "Model loading detail") {
		t.Errorf("the setting is already right; don't tell the user to change it; got:\n%s", got)
	}
}

// A tensor-parallel load reports per card. Reading those as totals would
// quarter a model spread over four GPUs.
func TestMemoryTooltipScalesATensorParallelLoad(t *testing.T) {
	const split = `0.00.100.000 I srv          load: spawning server instance with name=m1 on port 46231
[46231] 0.14.275.753 I load_tensors:       Meta() model buffer size = 5120.00 MiB
[46231] 0.16.162.940 I llama_kv_cache:     Meta() KV buffer size = 1024.00 MiB`

	s := memTestServer(t, "4")
	cfg := &models.ModelConfig{ContextSize: 8192, GPUAssign: "all", TensorSplit: "1,1,1,1"}
	feedLoad(s, split, loadNote{modelID: "m1", cards: 4, fingerprint: memoryFingerprint(cfg)})

	got := s.memoryTooltip(memTestModel(), cfg, "loaded")
	if !strings.Contains(got, "24.0 GiB of GPU memory") {
		t.Errorf("4 x (5120 + 1024) MiB = 24 GiB; got:\n%s", got)
	}
	if !strings.Contains(got, "Spread evenly across 4 GPUs") {
		t.Errorf("an aggregate report cannot name the cards, and should say so; got:\n%s", got)
	}
}

func TestMemoryFingerprintTracksTheFieldsThatCostMemory(t *testing.T) {
	base := models.ModelConfig{ContextSize: 8192, GPULayers: 99, KVCacheQuant: "q8_0"}

	changes := map[string]func(*models.ModelConfig){
		"context size":     func(c *models.ModelConfig) { c.ContextSize = 16384 },
		"parallel slots":   func(c *models.ModelConfig) { c.Parallel = 4 },
		"micro-batch":      func(c *models.ModelConfig) { c.UBatchSize = 2048 },
		"KV cache quant":   func(c *models.ModelConfig) { c.KVCacheQuant = "q4_0" },
		"flash attention":  func(c *models.ModelConfig) { c.FlashAttention = true },
		"GPU layers":       func(c *models.ModelConfig) { c.GPULayers = 20 },
		"tensor split":     func(c *models.ModelConfig) { c.TensorSplit = "1,0" },
		"vision projector": func(c *models.ModelConfig) { c.MmprojPath = "/models/mmproj.gguf" },
		"speculation mode": func(c *models.ModelConfig) { c.SpecType = "draft-mtp" },
		"MTP drafter head": func(c *models.ModelConfig) { c.MtpPath = "/models/MTP/head.gguf" },
		"draft GPU layers": func(c *models.ModelConfig) { c.DraftGPULayers = 40 },
		"per-layer embeds": func(c *models.ModelConfig) { c.PLEMode = "off" },
		"extra flags":      func(c *models.ModelConfig) { c.ExtraFlags = "--n-cpu-moe 20" },
	}
	for what, change := range changes {
		cfg := base
		change(&cfg)
		if memoryFingerprint(&cfg) == memoryFingerprint(&base) {
			t.Errorf("%s changes how much memory a load takes, but the fingerprint didn't move", what)
		}
	}

	// Sampling does not allocate anything, so a preset change must not
	// invalidate a perfectly good measurement.
	same := base
	temp := 0.7
	same.Temperature = &temp
	same.SamplingPreset = "thinking"
	same.Aliases = []string{"fast"}
	if memoryFingerprint(&same) != memoryFingerprint(&base) {
		t.Error("sampling settings must not invalidate a measurement")
	}
}

func TestLogVerbosityReadsTheEffectiveValue(t *testing.T) {
	s := memTestServer(t, "")
	if got := s.logVerbosity(); got != 3 {
		t.Errorf("unset = %d; want llama.cpp's own default of 3", got)
	}

	s = memTestServer(t, "4")
	if got := s.logVerbosity(); got != 4 {
		t.Errorf("curated = %d; want 4", got)
	}

	// The free-form block overrides the curated dropdown, and the
	// tooltip has to agree with what the router will really run with.
	s = memTestServer(t, "3")
	s.cfg.RuntimeEnvExtra = "LLAMA_ARG_LOG_VERBOSITY=5"
	if got := s.logVerbosity(); got != 5 {
		t.Errorf("extra env = %d; want 5", got)
	}
}

// The unit tests above feed the collector directly. This one drives the
// whole path a real load takes: a router process writing to stdout, the
// process manager broadcasting those lines, and watchRouterMemory
// subscribing to them — the wiring no amount of parser testing covers.
func TestWatchRouterMemoryCollectsFromTheLiveLogStream(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skipf("no sh available: %v", err)
	}
	binDir := t.TempDir()
	binary := filepath.Join(binDir, "llama-server")
	script := "#!/bin/sh\n"
	for _, line := range strings.Split(oneLoad, "\n") {
		script += "echo '" + line + "'\n"
	}
	script += "while :; do sleep 1; done\n"
	if err := os.WriteFile(binary, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	s := memTestServer(t, "4")
	s.process = process.NewManager()
	t.Cleanup(func() {
		if s.process.IsRunning() {
			s.process.Stop()
		}
	})
	go s.watchRouterMemory()

	// A free port, so the manager's health poll knocks on a door nobody
	// is behind rather than on whatever holds the default.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()

	if err := s.process.Start(process.RouterConfig{
		BinaryPath: binary, Host: "127.0.0.1", Port: port,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}

	deadline := time.Now().Add(5 * time.Second)
	for {
		if meas, ok := s.memory.Latest("m1"); ok && len(meas.Report.Entries) == 5 {
			if got := meas.Report.GPUMiB(true); got != 23552 {
				t.Errorf("GPU = %v MiB; want 23552", got)
			}
			return
		}
		if time.Now().After(deadline) {
			meas, ok := s.memory.Latest("m1")
			t.Fatalf("nothing collected from the log stream: found=%v entries=%d", ok, len(meas.Report.Entries))
		}
		time.Sleep(20 * time.Millisecond)
	}
}
