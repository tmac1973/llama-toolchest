package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// forkInstanceServer is a test server with forks on, two 24 GiB GPUs,
// Avalon's model set marked complete, and one successful Avalon ROCm
// build whose binary is a script that sleeps (it ignores llama-server's
// arguments). Health checks go to a fake that answers 200, at the port
// returned; give an instance that port to have it come up.
func forkInstanceServer(t *testing.T) (*Server, int) {
	t.Helper()
	s := newTestServer(t)
	s.cfg.ForksEnabled = true
	dir := s.cfg.DataDir
	store, err := forks.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	s.forks = store
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, Name: "RX 7900 XTX", VRAMTotalMiB: 24560},
		{Index: 1, Name: "RX 7900 XTX", VRAMTotalMiB: 24560},
	}}
	setDir := filepath.Join(dir, "models", "fork-models", "avalon")
	os.MkdirAll(setDir, 0o755)
	if err := store.PutModelSet(forks.ModelSet{ForkID: "avalon", Dir: setDir, Complete: true}); err != nil {
		t.Fatal(err)
	}

	bin := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 300\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	records := fmt.Sprintf(`[{"id":"avalon-ad2c6e7-rocm","project":"avalon","profile":"rocm","git_ref":"x","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","binary_path":%q,"commit_count":1}]`, bin)
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)

	health := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(health.Close)
	_, port, _ := net.SplitHostPort(strings.TrimPrefix(health.URL, "http://"))
	var p int
	fmt.Sscan(port, &p)
	t.Cleanup(s.forkMgr.StopAll)
	return s, p
}

// withIDAction adds the {id} and {action} chi URL parameters.
func withIDAction(r *http.Request, id, action string) *http.Request {
	rc := chi.NewRouteContext()
	rc.URLParams.Add("id", id)
	rc.URLParams.Add("action", action)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

func forkForm(method, target string, form url.Values) *http.Request {
	r := httptest.NewRequest(method, target, strings.NewReader(form.Encode()))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return r
}

func TestForkInstanceRoutesHiddenWhileOff(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.cfg.ForksEnabled = false
	rec := httptest.NewRecorder()
	s.handleForkInstances(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("list with forks off = %d", rec.Code)
	}
	if _, err := s.startForkInstance("x"); err == nil {
		t.Error("start with forks off")
	}
}

func TestForkInstanceSaveValidates(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.cfg.GenerationEnabled = true
	gen, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = gen
	gen.AddModelSet(generation.ModelSet{ID: "set", Name: "Set", Kind: generation.KindSD, Complete: true})
	if err := gen.AddInstance(generation.Instance{ID: "flux", Name: "flux", ModelSetID: "set", Port: 8081}); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.Add(&models.Model{ID: "qwen-chat", ModelID: "org/qwen", Filename: "q.gguf"}); err != nil {
		t.Fatal(err)
	}

	add := func(form url.Values) *httptest.ResponseRecorder {
		base := url.Values{"id": {"avalon-qwen"}, "fork": {"avalon"}, "gpu": {"1"}}
		for k, v := range form {
			base[k] = v
		}
		rec := httptest.NewRecorder()
		s.handleForkAddInstance(rec, forkForm("POST", "/", base))
		return rec
	}
	for name, c := range map[string]struct {
		form url.Values
		code int
	}{
		"bad name":          {url.Values{"id": {"Avalon Qwen"}}, http.StatusBadRequest},
		"model name":        {url.Values{"id": {"qwen-chat"}}, http.StatusConflict},
		"generation name":   {url.Values{"id": {"flux"}}, http.StatusConflict},
		"unknown fork":      {url.Values{"fork": {"nope"}}, http.StatusBadRequest},
		"no GPU":            {url.Values{"gpu": {""}}, http.StatusBadRequest},
		"bad env":           {url.Values{"extra_env": {"lower=1"}}, http.StatusBadRequest},
		"port in args":      {url.Values{"extra_args": {"--port 9000"}}, http.StatusBadRequest},
		"model in args":     {url.Values{"extra_args": {"-m other.gguf"}}, http.StatusBadRequest},
		"port out of range": {url.Values{"port": {"9999"}}, http.StatusBadRequest},
		"generation's port": {url.Values{"port": {"8081"}}, http.StatusConflict},
		"build of no fork":  {url.Values{"build": {"b9000-rocm"}}, http.StatusBadRequest},
	} {
		if rec := add(c.form); rec.Code != c.code {
			t.Errorf("%s: %d %s, want %d", name, rec.Code, strings.TrimSpace(rec.Body.String()), c.code)
		}
	}

	rec := add(url.Values{"extra_env": {"GGML_CUDA_MOE_PIN=0\n\nA=1"}, "extra_args": {"-c 32768"}, "auto_start": {"on"}})
	if rec.Code != http.StatusOK {
		t.Fatalf("valid instance: %d %s", rec.Code, rec.Body)
	}
	inst, err := s.forks.Instance("avalon-qwen")
	if err != nil {
		t.Fatal(err)
	}
	// The generation instance holds 8081, so the fork gets the next port.
	if inst.Port != 8082 || !inst.AutoStart || len(inst.ExtraEnv) != 2 || len(inst.ExtraArgs) != 2 {
		t.Errorf("saved %+v", inst)
	}
	if rec := add(nil); rec.Code != http.StatusConflict {
		t.Errorf("duplicate name: %d", rec.Code)
	}

	// A new generation instance does not take the fork's port either.
	genInst := generation.Instance{ID: "flux2"}
	if code, err := s.genPortFromForm(forkForm("POST", "/", url.Values{"port": {"8082"}}), gen, &genInst); err == nil || code != http.StatusConflict {
		t.Errorf("generation instance on the fork's port: %d %v", code, err)
	}
	genInst = generation.Instance{ID: "flux3"}
	if _, err := s.genPortFromForm(forkForm("POST", "/", url.Values{}), gen, &genInst); err != nil || genInst.Port != 8083 {
		t.Errorf("generation instance got port %d, %v; want 8083", genInst.Port, err)
	}
}

func TestForkInstanceStartChecks(t *testing.T) {
	s, port := forkInstanceServer(t)
	inst := forks.Instance{ID: "avalon-qwen", ForkID: "avalon", GPU: 1, Port: port}
	if err := s.forks.AddInstance(inst); err != nil {
		t.Fatal(err)
	}

	// The model set is not complete.
	set, _ := s.forks.ModelSet("avalon")
	set.Complete = false
	s.forks.PutModelSet(set)
	if _, err := s.startForkInstance(inst.ID); err == nil || !strings.Contains(err.Error(), "not fully downloaded") {
		t.Errorf("incomplete set: %v", err)
	}
	set.Complete = true
	s.forks.PutModelSet(set)

	// Something else on the GPU: the start is refused and names it.
	s.cfg.GenerationEnabled = true
	gen, _ := generation.Load(s.cfg.DataDir)
	s.gen = gen
	gen.AddModelSet(generation.ModelSet{ID: "set", Name: "Set", Kind: generation.KindSD, Complete: true})
	gen.AddInstance(generation.Instance{ID: "flux", Name: "flux", ModelSetID: "set", GPU: 1, Port: 8081, Enabled: true})
	_, err := s.startForkInstance(inst.ID)
	var busy *forkGPUBusyError
	if !errors.As(err, &busy) || !strings.Contains(err.Error(), "GPU 1 holds flux (generation)") {
		t.Errorf("busy GPU: %v", err)
	}
	rec := httptest.NewRecorder()
	s.handleForkInstanceAction(rec, withIDAction(forkForm("POST", "/", nil), inst.ID, "start"))
	if rec.Code != http.StatusConflict {
		t.Errorf("busy GPU over HTTP: %d", rec.Code)
	}
	if s.forkMgr.IsActive(inst.ID) {
		t.Fatal("started on a busy GPU")
	}

	// No build of the fork.
	gen.DeleteInstance("flux")
	os.WriteFile(filepath.Join(s.cfg.DataDir, "config", "builds.json"), []byte("[]"), 0o644)
	good := s.builder
	s.builder = builder.NewBuilder(s.cfg.DataDir)
	if _, err := s.startForkInstance(inst.ID); err == nil || !strings.Contains(err.Error(), "no successful Avalon build") {
		t.Errorf("no build: %v", err)
	}
	s.builder = good
}

// A start launches the fork's llama-server with the recipe, the
// instance's environment on top, and the model set's folder as working
// directory; while it runs, its GPU is reserved whole.
func TestForkInstanceRunsAndReservesItsGPU(t *testing.T) {
	s, port := forkInstanceServer(t)
	inst := forks.Instance{ID: "avalon-qwen", ForkID: "avalon", GPU: 1, Port: port,
		ExtraEnv: []string{"GGML_CUDA_MOE_PIN=0"}}
	if err := s.forks.AddInstance(inst); err != nil {
		t.Fatal(err)
	}
	if _, err := s.startForkInstance(inst.ID); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := s.forkMgr.WaitRunning(ctx, inst.ID); err != nil || !s.forkMgr.IsRunning(inst.ID) {
		t.Fatalf("not running: %v %+v", err, s.forkMgr.Status(inst.ID))
	}
	pid := s.forkMgr.PID(inst.ID)
	environ, _ := os.ReadFile(fmt.Sprintf("/proc/%d/environ", pid))
	for _, want := range []string{"HIP_VISIBLE_DEVICES=1", "GGML_CUDA_MOE_PIN=0"} {
		if !strings.Contains(string(environ), want+"\x00") {
			t.Errorf("process environment lacks %s", want)
		}
	}
	cwd, _ := os.Readlink(fmt.Sprintf("/proc/%d/cwd", pid))
	if set, _ := s.forks.ModelSet("avalon"); cwd != set.Dir {
		t.Errorf("working directory %s, want %s", cwd, set.Dir)
	}

	// The GPU is reserved whole: an LLM planned on it is refused, one on
	// the other GPU is not, and no generation instance fits there.
	hw := s.hardware()
	if err := s.reservationFitError(hw, map[int]int{1: 1024}, false); err == nil || !strings.Contains(err.Error(), "avalon-qwen (fork, whole GPU)") {
		t.Errorf("LLM on the fork's GPU: %v", err)
	}
	if err := s.reservationFitError(hw, map[int]int{0: 1024}, false); err != nil {
		t.Errorf("LLM on the other GPU: %v", err)
	}
	if err := s.genFitCheck(generation.Instance{ID: "flux", Name: "flux", GPU: 1}, 1024); err == nil {
		t.Error("a generation instance fits on the fork's GPU")
	}
	// Auto-evict never takes it back.
	s.cfg.GenEvict = true
	if v := s.genVRAM("", true)[1]; v == nil || v.Evictable != 0 || v.Reserved < 24560 {
		t.Errorf("fork reservation = %+v", v)
	}

	// Running: no edit, no delete.
	rec := httptest.NewRecorder()
	s.handleForkEditInstance(rec, withID(forkForm("PUT", "/", url.Values{"fork": {"avalon"}, "gpu": {"0"}}), inst.ID))
	if rec.Code != http.StatusConflict {
		t.Errorf("edit while running: %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	s.handleForkDeleteInstance(rec, withID(httptest.NewRequest("DELETE", "/", nil), inst.ID))
	if rec.Code != http.StatusConflict {
		t.Errorf("delete while running: %d", rec.Code)
	}

	// Turning forks off stops it, and the reservation goes.
	s.forksSwitched(true, false)
	deadline := time.Now().Add(10 * time.Second)
	for s.forkMgr.IsActive(inst.ID) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if s.forkMgr.IsActive(inst.ID) {
		t.Fatal("still running after forks were switched off")
	}
	s.cfg.ForksEnabled = true
	if v := s.genVRAM("", false)[1]; v != nil {
		t.Errorf("reservation kept after stop: %+v", v)
	}
}

func TestForkSwitchOnAutoStarts(t *testing.T) {
	s, _ := forkInstanceServer(t)
	started := make(chan struct{}, 1)
	s.forkAutoStartFn = func() { started <- struct{}{} }
	s.forksSwitched(false, true)
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Error("switching forks on did not auto-start")
	}
}

func TestAutoStartForkInstancesSkipsAndLogs(t *testing.T) {
	s, port := forkInstanceServer(t)
	s.forks.AddInstance(forks.Instance{ID: "on", ForkID: "avalon", GPU: 0, Port: port, AutoStart: true})
	s.forks.AddInstance(forks.Instance{ID: "off", ForkID: "avalon", GPU: 1, Port: port + 1})
	s.autoStartForkInstances(false)
	if !s.forkMgr.IsActive("on") || s.forkMgr.IsActive("off") {
		t.Errorf("auto-start: on %v, off %v", s.forkMgr.IsActive("on"), s.forkMgr.IsActive("off"))
	}
}

func TestForkInstanceListJSON(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.forks.AddInstance(forks.Instance{ID: "q", ForkID: "avalon", Port: 8081})
	rec := httptest.NewRecorder()
	s.handleForkInstances(rec, httptest.NewRequest("GET", "/", nil))
	var got []forkInstanceView
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil || len(got) != 1 || got[0].State != "stopped" {
		t.Errorf("list = %s", rec.Body)
	}
}

// With auto-evict on, a stopped generation instance holds its GPU only
// when it is locked; with it off, every enabled one does.
func TestForkBusyCheckFollowsAutoEvict(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.cfg.GenerationEnabled = true
	gen, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = gen
	gen.AddModelSet(generation.ModelSet{ID: "set", Name: "Set", Kind: generation.KindSD, Complete: true})
	gen.AddInstance(generation.Instance{ID: "free", Name: "free", ModelSetID: "set", GPU: 1, Port: 8081, Enabled: true})
	gen.AddInstance(generation.Instance{ID: "locked", Name: "locked", ModelSetID: "set", GPU: 1, Port: 8082, Enabled: true, KeepRunning: true})
	gen.AddInstance(generation.Instance{ID: "off", Name: "off", ModelSetID: "set", GPU: 1, Port: 8083})

	s.cfg.GenEvict = false
	if got := strings.Join(s.gpuOccupants(1, ""), ", "); got != "free (generation), locked (generation)" {
		t.Errorf("auto-evict off: %q", got)
	}
	s.cfg.GenEvict = true
	if got := strings.Join(s.gpuOccupants(1, ""), ", "); got != "locked (generation)" {
		t.Errorf("auto-evict on: %q", got)
	}
	gen.EditInstance("locked", func(inst *generation.Instance) { inst.KeepRunning = false })
	if got := s.gpuOccupants(1, ""); len(got) != 0 {
		t.Errorf("auto-evict on, nothing locked: %v", got)
	}
}
