package api

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// testBuilder constructs a Builder whose persisted state (builds.json
// under the temp data dir, the format the builder itself writes) holds
// the given builds, so Find() resolves them.
func testBuilder(t *testing.T, builds ...builder.BuildResult) *builder.Builder {
	t.Helper()
	dataDir := t.TempDir()
	data, err := json.MarshalIndent(builds, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dataDir, "config", "builds.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return builder.NewBuilder(dataDir)
}

func testBuild(id, binDir string) builder.BuildResult {
	return builder.BuildResult{
		ID: id, Profile: "cpu", GitRef: "b1",
		Status: builder.BuildStatusSuccess, BinaryPath: filepath.Join(binDir, "llama-server"),
	}
}

// capJobReq builds a minimal capability job request for the pure
// validator: one model, one build, the given presets and sweeps.
func capJobReq(presets []string, sweeps ...benchmark.SweepAxis) jobCreateRequest {
	return jobCreateRequest{
		Name: "cap", ModelIDs: []string{"m"}, BuildIDs: []string{"b"}, Presets: presets,
		Sweeps: sweeps,
	}
}

// The four-condition refusal: capability-only presets + sweeps + no
// eval-reaching axis + a single model×build.
func TestValidateCapabilitySweepsRefusesInertSweep(t *testing.T) {
	cases := []jobCreateRequest{
		capJobReq([]string{"perplexity-quick"},
			benchmark.SweepAxis{Field: "temperature", Values: []string{"0.7", "0.8"}}),
		capJobReq([]string{"hellaswag-quick"},
			benchmark.SweepAxis{Field: "top_p", Values: []string{"0.9", "1.0"}}),
		// Multiple inert axes are still inert.
		capJobReq([]string{"kl-divergence-quick"},
			benchmark.SweepAxis{Field: "temperature", Values: []string{"0.7", "0.8"}},
			benchmark.SweepAxis{Field: "top_k", Values: []string{"40", "50"}}),
		// Two capability presets do not make the sweep meaningful.
		capJobReq([]string{"perplexity-quick", "hellaswag-quick"},
			benchmark.SweepAxis{Field: "min_p", Values: []string{"0", "0.1"}}),
	}
	for i, req := range cases {
		if err := validateJobRequest(req); err == nil {
			t.Errorf("case %d: inert-sweep capability job was accepted: %+v", i, req)
		} else if !strings.Contains(err.Error(), "do not affect capability evaluations") {
			t.Errorf("case %d: refusal message = %q", i, err)
		}
	}
}

// The refusals the plan says are NOT refusals.
func TestValidateCapabilitySweepsNotRefused(t *testing.T) {
	okCases := []struct {
		name string
		req  jobCreateRequest
	}{
		{
			// The flagship: four quants, zero sweeps.
			"flagship-no-sweeps",
			jobCreateRequest{
				Name: "cap", ModelIDs: []string{"m4", "m5", "m8", "mq"}, BuildIDs: []string{"b"},
				Presets: []string{"perplexity-quick", "kl-divergence-quick", "hellaswag-quick", "winogrande-quick"},
			},
		},
		{
			// Multi-model job with an inert sweep: after collapse the
			// cells still vary by model.
			"multi-model-inert-sweep",
			jobCreateRequest{
				Name: "cap", ModelIDs: []string{"m4", "m8"}, BuildIDs: []string{"b"},
				Presets: []string{"perplexity-quick"},
				Sweeps:  []benchmark.SweepAxis{{Field: "temperature", Values: []string{"0.7", "0.8"}}},
			},
		},
		{
			// Multi-build job with an inert sweep: cells vary by build.
			"multi-build-inert-sweep",
			jobCreateRequest{
				Name: "cap", ModelIDs: []string{"m"}, BuildIDs: []string{"b1", "b2"},
				Presets: []string{"perplexity-quick"},
				Sweeps:  []benchmark.SweepAxis{{Field: "top_p", Values: []string{"0.9", "1.0"}}},
			},
		},
		{
			// An eval-reaching axis: the cells genuinely differ.
			"eval-reaching-sweep",
			capJobReq([]string{"perplexity-quick"},
				benchmark.SweepAxis{Field: "kv_cache_quant", Values: []string{"f16", "q8_0"}}),
		},
		{
			// Placement is eval-reaching (it becomes --device/--tensor-split).
			"gpu-assign-sweep",
			capJobReq([]string{"perplexity-quick"},
				benchmark.SweepAxis{Field: "gpu_assign", Values: []string{"all", "0"}}),
		},
		{
			// A mixed job is not capability-only.
			"mixed-presets",
			capJobReq([]string{"perplexity-quick", "internal-quick"},
				benchmark.SweepAxis{Field: "temperature", Values: []string{"0.7", "0.8"}}),
		},
	}
	for _, c := range okCases {
		if err := validateJobRequest(c.req); err != nil {
			t.Errorf("%s: refused a valid job: %v", c.name, err)
		}
	}
}

// The benchy refusal still fires first for a benchy + capability +
// sampling-axis mix (unchanged behavior).
func TestBenchyRefusalStillFirstForMixedJob(t *testing.T) {
	req := capJobReq([]string{"benchy-standard", "perplexity-quick"},
		benchmark.SweepAxis{Field: "temperature", Values: []string{"0.7", "0.8"}})
	if err := validateJobRequest(req); err == nil {
		t.Fatal("benchy + sampling axis must be refused")
	} else if strings.Contains(err.Error(), "do not affect capability evaluations") {
		t.Errorf("the capability refusal shadowed the benchy refusal: %v", err)
	}
}

// The matrix count applies the collapse: inert axes do not multiply the
// capability presets' cells, and eval-reaching axes do.
func TestValidateJobRequestCellCountAppliesCollapse(t *testing.T) {
	// 1 model × 1 build × (2 capability presets collapsed to 1 + 1
	// performance preset at 2 temperature points) = 4 cells.
	req := capJobReq([]string{"perplexity-quick", "hellaswag-quick", "internal-quick"},
		benchmark.SweepAxis{Field: "temperature", Values: []string{"0.7", "0.8"}})
	if err := validateJobRequest(req); err != nil {
		t.Fatalf("valid collapsed matrix refused: %v", err)
	}

	// The same axes without the collapse would be 6 cells — the count
	// must be 4, so push the same shape over the limit: 256 models × 1
	// build × 4 cells = 1024 > 500.
	big := req
	big.ModelIDs = make([]string, 256)
	for i := range big.ModelIDs {
		big.ModelIDs[i] = "m" + string(rune('a'+i/26)) + string(rune('a'+i%26))
	}
	if err := validateJobRequest(big); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Errorf("collapsed matrix over the cap must be refused, got: %v", err)
	}

	// And the non-collapsed shape (256 × 1 × 6 = 1536) is over too — the
	// refusal exists either way, so the distinguishing check is the
	// boundary: 128 models × 4 collapsed cells = 512 > 500 while
	// 128 × 6 = 768. Use the boundary from below: 125 × 4 = 500 is
	// exactly the cap and must pass.
	big.ModelIDs = big.ModelIDs[:125]
	if err := validateJobRequest(big); err != nil {
		t.Errorf("exactly-at-cap collapsed matrix refused: %v", err)
	}
}

// KL validation: every cell self-referential → refuse, and the message
// names what to add.
func TestValidateKLJobRefusesAllSelfReference(t *testing.T) {
	// Single model, the only installed quant of its repo, no override.
	s := batchMatrixServer(t, models.ModelConfig{Enabled: true})
	req := jobCreateRequest{
		Name: "kl", ModelIDs: []string{"m"}, BuildIDs: []string{"b"},
		Presets: []string{"kl-divergence-quick"},
	}
	err := s.validateKLJob(req)
	if err == nil {
		t.Fatal("a KL job whose only model is its own only quant must be refused")
	}
	if !strings.Contains(err.Error(), "reference") {
		t.Errorf("refusal does not mention the reference: %v", err)
	}

	// Explicit reference naming the only model.
	req.KLReference = "m"
	if err := s.validateKLJob(req); err == nil {
		t.Fatal("an explicit self-reference must be refused")
	}

	// A non-KL job is never KL-validated.
	req.Presets = []string{"perplexity-quick"}
	if err := s.validateKLJob(req); err != nil {
		t.Errorf("non-KL job tripped the KL check: %v", err)
	}
}

// The primary flow: the reference is among the models, so only its own
// cell skips and the job is accepted.
func TestValidateKLJobAllowsReferenceAmongModels(t *testing.T) {
	reg := models.NewRegistry(t.TempDir(), "/models")
	addModel := func(id, quant string, bytes int64) {
		m := &models.Model{ID: id, ModelID: "u/M", Quant: quant, SizeBytes: bytes, FilePath: "/models/" + id + ".gguf"}
		if err := reg.Add(m); err != nil {
			t.Fatalf("add %s: %v", id, err)
		}
		if err := reg.SetConfig(id, &models.ModelConfig{Enabled: true}); err != nil {
			t.Fatalf("config %s: %v", id, err)
		}
	}
	addModel("m4", "Q4_K_XL", 2<<30)
	addModel("m8", "Q8_0", 4<<30) // the automatic reference

	s := &Server{registry: reg}
	req := jobCreateRequest{
		Name: "kl", ModelIDs: []string{"m4", "m8"}, BuildIDs: []string{"b"},
		Presets: []string{"kl-divergence-quick"},
	}
	if err := s.validateKLJob(req); err != nil {
		t.Fatalf("the flagship all-quants flow was refused: %v", err)
	}

	// An explicit reference among the models is also fine.
	req.KLReference = "m8"
	if err := s.validateKLJob(req); err != nil {
		t.Fatalf("explicit reference among the models was refused: %v", err)
	}
}

// An unknown explicit reference is a bad request, not an all-skip
// refusal.
func TestValidateKLJobUnknownReferenceIsBadRequest(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{Enabled: true})
	req := jobCreateRequest{
		Name: "kl", ModelIDs: []string{"m", "m2"}, BuildIDs: []string{"b"},
		Presets:   []string{"kl-divergence-quick"},
		KLReference: "ghost",
	}
	err := s.validateKLJob(req)
	if err == nil || !strings.Contains(err.Error(), "not an installed model") {
		t.Fatalf("unknown reference must be a bad request, got: %v", err)
	}
}

// EvalBinary: the binary is expected next to the build's llama-server;
// a missing one produces the rebuild message.
func TestEvalBinaryFindsAndMisses(t *testing.T) {
	withBin := t.TempDir()
	perplexity := filepath.Join(withBin, "llama-perplexity")
	for _, p := range []string{filepath.Join(withBin, "llama-server"), perplexity} {
		if err := os.WriteFile(p, []byte("x"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	withoutBin := t.TempDir()
	if err := os.WriteFile(filepath.Join(withoutBin, "llama-server"), []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	s := &Server{
		cfg:     &config.Config{DataDir: t.TempDir()},
		builder: testBuilder(t, testBuild("b1", withBin), testBuild("b2", withoutBin)),
	}
	env := newJobEnv(s)

	got, err := env.EvalBinary("b1")
	if err != nil {
		t.Fatalf("EvalBinary: %v", err)
	}
	if got != perplexity {
		t.Errorf("EvalBinary = %s, want %s", got, perplexity)
	}

	// A build predating the install: the message names the fix.
	if _, err := env.EvalBinary("b2"); err == nil || !strings.Contains(err.Error(), "rebuild") {
		t.Errorf("missing binary error = %v, want the rebuild message", err)
	}

	if _, err := env.EvalBinary("nope"); err == nil {
		t.Error("unknown build must error")
	}
}

// ResolveKLReference: the override wins; otherwise the largest quant of
// the same repo; error when the model is its repo's only quant.
func TestResolveKLReferencePolicy(t *testing.T) {
	reg := models.NewRegistry(t.TempDir(), "/models")
	addModel := func(id, repo, quant string, bytes int64) {
		m := &models.Model{ID: id, ModelID: repo, Quant: quant, SizeBytes: bytes, FilePath: "/models/" + id + ".gguf"}
		if err := reg.Add(m); err != nil {
			t.Fatalf("add %s: %v", id, err)
		}
		if err := reg.SetConfig(id, &models.ModelConfig{Enabled: true, GPULayers: 999}); err != nil {
			t.Fatalf("config %s: %v", id, err)
		}
	}
	// u/M has two quants (m8 larger); u/L is a different repo with one.
	addModel("m4", "u/M", "Q4_K_XL", 2<<30)
	addModel("m8", "u/M", "Q8_0", 4<<30)
	addModel("l4", "u/L", "Q4_K_M", 5<<30)

	s := &Server{registry: reg, cfg: &config.Config{DataDir: t.TempDir()}}
	env := newJobEnv(s)

	// Automatic: the largest quant of the SAME repo, not the largest
	// model overall (l4 is bigger but a different repo).
	ref, err := env.ResolveKLReference("m4", "")
	if err != nil {
		t.Fatalf("automatic resolution: %v", err)
	}
	if ref.ID != "m8" {
		t.Errorf("reference = %s, want m8 (largest quant of u/M)", ref.ID)
	}
	if ref.Config.GPULayers != 999 {
		t.Errorf("reference config GPULayers = %d, want the reference's saved 999", ref.Config.GPULayers)
	}
	if ref.FilePath != "/models/m8.gguf" || ref.Quant != "Q8_0" || ref.SizeBytes != 4<<30 {
		t.Errorf("reference info incomplete: %+v", ref)
	}

	// The override wins even when it is a different repo.
	ref, err = env.ResolveKLReference("m4", "l4")
	if err != nil || ref.ID != "l4" {
		t.Errorf("override resolution = %+v, %v", ref, err)
	}

	// No distinct quant in the repo: error naming the model.
	if _, err := env.ResolveKLReference("l4", ""); err == nil || !strings.Contains(err.Error(), "only installed quant") {
		t.Errorf("single-quant repo error = %v", err)
	}

	// An unknown override is an error.
	if _, err := env.ResolveKLReference("m4", "ghost"); err == nil {
		t.Error("unknown override must error")
	}
}

// EvalFlags: the complete flag list for a cell config — the mapped
// fields plus the placement flags, and the named batch validation error.
func TestEvalFlagsCompleteAndValidated(t *testing.T) {
	reg := models.NewRegistry(t.TempDir(), "/models")
	m := &models.Model{ID: "m", ModelID: "u/M", Quant: "Q8_0", SizeBytes: 4 << 30, FilePath: "/models/m.gguf"}
	if err := reg.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := reg.SetConfig("m", &models.ModelConfig{
		Enabled: true, GPULayers: 999, Threads: 8,
		BatchSize: 2048, UBatchSize: 512, KVCacheQuant: "q8_0", FlashAttention: true, DirectIO: true,
	}); err != nil {
		t.Fatal(err)
	}
	s := &Server{registry: reg, cfg: &config.Config{DataDir: t.TempDir()}, monitor: monitor.New(time.Hour)}
	s.builder = testBuilder(t)
	env := newJobEnv(s)

	snap := benchmark.ConfigSnapshot{
		GPULayers: 40, Threads: 16, BatchSize: 1024, UBatchSize: 256,
		KVCacheQuant: "f16", DirectIO: true,
	}
	flags, err := env.EvalFlags("m", snap, "b1")
	if err != nil {
		t.Fatalf("EvalFlags: %v", err)
	}
	joined := strings.Join(flags, " ")
	for _, want := range []string{
		"--n-gpu-layers 40",
		"--threads 16",
		"--batch-size 1024",
		"--ubatch-size 256",
		"--cache-type-k f16",
		"--cache-type-v f16",
		"--flash-attn off", // the cell's snapshot does not enable it
		"--direct-io",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("flags missing %q: %v", want, flags)
		}
	}
	// The excluded fields must NOT be mapped: context size is fixed by
	// the evaluator, speculative decoding is excluded from evaluations.
	for _, absent := range []string{"--ctx-size", "--draft", "-sm"} {
		if strings.Contains(joined, absent) {
			t.Errorf("flags contain excluded %q: %v", absent, flags)
		}
	}

	// A -ub > -b pair is the named validation error, not a loader error.
	bad := snap
	bad.UBatchSize = 4096
	bad.BatchSize = 512
	if _, err := env.EvalFlags("m", bad, "b1"); err == nil ||
		!strings.Contains(err.Error(), "micro-batch 4096 exceeds batch size 512") {
		t.Errorf("bad batch pair error = %v", err)
	}
}

// evalStopServer wires a real process.Manager to a fake llama-server:
// the "binary" is a script that stays alive and ignores the router's
// arguments, and an httptest server on the configured llama port serves
// /health (one failed poll, then 200) so IsRunning() goes false → true
// on a realistic timescale. The router can start, report healthy, be
// stopped, and be restarted — the real lifecycle the eval-stop tests
// exercise.
func evalStopServer(t *testing.T) (*Server, *process.Manager) {
	t.Helper()
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skipf("no sh available: %v", err)
	}
	binDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(binDir, "llama-server"),
		[]byte("#!/bin/sh\nwhile :; do sleep 1; done\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	health := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		health++
		if health == 1 {
			// One failed poll first, so the state machine actually
			// traverses starting → running.
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		io.WriteString(w, `{"status":"ok"}`)
	}))
	t.Cleanup(ts.Close)
	port := ts.Listener.Addr().(*net.TCPAddr).Port

	dataDir := t.TempDir()
	cfg := &config.Config{DataDir: dataDir, ModelsMax: 1, LlamaPort: port}
	reg := models.NewRegistry(dataDir, "/models")
	bld := testBuilder(t, testBuild("b1", binDir))
	proc := process.NewManager()
	t.Cleanup(func() {
		if proc.IsRunning() {
			proc.Stop()
		}
	})
	return &Server{cfg: cfg, registry: reg, builder: bld, process: proc}, proc
}

// StopRouterForEval: stopping a RUNNING router records ownership; an
// already-stopped router records nothing (cleanup must not start a
// server the user turned off). Idempotent.
func TestStopRouterForEvalRecordsOwnership(t *testing.T) {
	s, proc := evalStopServer(t)
	if err := proc.Start(process.RouterConfig{
		BinaryPath: filepath.Join(s.builderDataDirForTest(t), "llama-server"),
		Host:       "127.0.0.1", Port: s.cfg.LlamaPort, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)
	env := newJobEnv(s)

	if err := env.StopRouterForEval(context.Background()); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if proc.IsRunning() {
		t.Fatal("the router is still running after StopRouterForEval")
	}
	env.mu.Lock()
	owns, stopped := env.ownsRouter, env.stoppedForEval
	env.mu.Unlock()
	if !owns || !stopped {
		t.Errorf("flags after stopping a running router: owns=%v stoppedForEval=%v, want both true", owns, stopped)
	}

	// Idempotent: a second call changes nothing and does not fail on
	// an already-stopped process.
	if err := env.StopRouterForEval(context.Background()); err != nil {
		t.Fatalf("second stop: %v", err)
	}
	env.mu.Lock()
	still := env.stoppedForEval
	env.mu.Unlock()
	if !still {
		t.Error("second call cleared the eval-stop flag")
	}

	// User-stopped router: nothing recorded, no stop attempted.
	s2, _ := evalStopServer(t)
	env2 := newJobEnv(s2)
	if err := env2.StopRouterForEval(context.Background()); err != nil {
		t.Fatalf("stop on a stopped router: %v", err)
	}
	env2.mu.Lock()
	owns2, stopped2 := env2.ownsRouter, env2.stoppedForEval
	env2.mu.Unlock()
	if owns2 || stopped2 {
		t.Errorf("flags after an already-stopped router: owns=%v stoppedForEval=%v, want both false", owns2, stopped2)
	}
}

// ClearEphemeralConfig: the eval-stop case restarts the router (this
// job stopped it), while the user-stop case leaves it stopped.
func TestClearEphemeralConfigEvalStopRestartsUserStopDoesNot(t *testing.T) {
	s, proc := evalStopServer(t)
	// Bring the router up and have the job stop it for an eval.
	if err := proc.Start(process.RouterConfig{
		BinaryPath: filepath.Join(s.builderDataDirForTest(t), "llama-server"),
		Host:       "127.0.0.1", Port: s.cfg.LlamaPort, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)
	env := newJobEnv(s)
	if err := env.StopRouterForEval(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := env.ClearEphemeralConfig(context.Background()); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	waitRouterRunning(t, proc) // the eval-stop case restarts
	env.mu.Lock()
	owns := env.ownsRouter
	stopped := env.stoppedForEval
	env.mu.Unlock()
	if owns || stopped {
		t.Errorf("cleanup left flags set: owns=%v stoppedForEval=%v", owns, stopped)
	}

	// User-stop case: nothing owned → nothing restarted.
	s2, proc2 := evalStopServer(t)
	env2 := newJobEnv(s2)
	if err := env2.ClearEphemeralConfig(context.Background()); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	if proc2.IsRunning() {
		t.Error("a user-stopped router must stay stopped")
	}
}

// builderDataDirForTest returns the binary directory of the test build
// ("b1") so the tests can pass its llama-server to the real Manager.
func (s *Server) builderDataDirForTest(t *testing.T) string {
	t.Helper()
	b, ok := s.builder.Find("b1")
	if !ok {
		t.Fatal("test build b1 not found")
	}
	return filepath.Dir(b.BinaryPath)
}

func waitRouterRunning(t *testing.T, m *process.Manager) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if m.IsRunning() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("router did not report running within 15s")
}
