package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// startFakeGen runs script as generation instance inst, with the engine
// notes hooks a real launch gets, and a health check that passes at once.
func startFakeGen(t *testing.T, s *Server, inst generation.Instance, kind generation.BackendKind, script string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fakes need a unix shell")
	}
	bin := filepath.Join(t.TempDir(), "fake-backend")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	health := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(health.Close)
	cfg := process.SupervisorConfig{
		Name: "fake", BinaryPath: bin, Env: os.Environ(),
		HealthURL: health.URL, HealthInterval: 20 * time.Millisecond, StopTimeout: 5 * time.Second,
	}
	cfg.OnReady, cfg.OnExit = s.genNotesHooks(inst, kind, "sd-build-1")
	if err := s.genMgr.Start(inst.ID, kind, cfg); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.genMgr.Stop(inst.ID) })
}

// The whole path of an sd-server crash: the instance starts, passes its
// health check, prints a CUDA fault and aborts. Engine notes record the
// start and the crash under sd-server, with the GPU, the build, the error
// lines and the cause, and the instance is no longer open.
func TestGenInstanceCrashIsRecorded(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	inst := generation.Instance{ID: "wan", Name: "Wan video", GPU: 2, Port: 8190}
	// Waits for the health check before it fails, as a crash during a
	// generation does.
	startFakeGen(t, s, inst, generation.KindVideo, `sleep 0.5
echo "[I] sampling completed --- video.cpp:1776"
printf '  |===>   |  1/15\033[K\r[E] ggml: ggml_cuda_compute_forward: CONCAT failed --- ggml.cpp:1\n'
echo "[E] ggml: CUDA error: an illegal memory access was encountered"
kill -ABRT $$`)

	testutil.Eventually(t, func() bool { return allNotes(s).Totals.Crashes == 1 }, "no crash recorded")
	sum := allNotes(s)
	if sum.Totals.Loads != 1 {
		t.Errorf("loads = %d, want the start before the crash", sum.Totals.Loads)
	}
	f := sum.Failures[0]
	if f.Source != enginenotes.SourceSDServer || f.ModelID != "wan" || f.ModelName != "Wan video" ||
		f.GPUs != "gpu:2" || f.Port != "8190" || f.Build != "sd-build-1" || f.ExitStatus != 134 || f.JobID != "" {
		t.Errorf("crash = %+v", f)
	}
	if f.Cause != enginenotes.CauseGPUFault {
		t.Errorf("cause = %q, want a GPU fault", f.Cause)
	}
	// The progress bar and sd.cpp's source suffix are cleaned off.
	if !slices.Contains(f.ErrorLines, "[E] ggml: ggml_cuda_compute_forward: CONCAT failed") {
		t.Errorf("error lines = %q", f.ErrorLines)
	}
	if len(sum.Models) != 1 || sum.Models[0].Source != enginenotes.SourceSDServer || sum.Models[0].Crashes != 1 {
		t.Errorf("rows = %+v", sum.Models)
	}
	if open := s.genOpenInstances(); len(open) != 0 {
		t.Errorf("open after the crash = %+v", open)
	}
}

// A stop that was asked for is an unload, even though the backend exits
// with a signal status; an exit before the health check passes is a
// failed start.
func TestGenInstanceStopAndFailedStart(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	song := generation.Instance{ID: "song", Name: "Song", GPU: 0}
	startFakeGen(t, s, song, generation.KindAce, "exec sleep 30")
	testutil.Eventually(t, func() bool { return len(s.genOpenInstances()) == 1 }, "the instance never counted as running")
	if open := s.genOpenInstances()[0]; open.Source != enginenotes.SourceAceServer || open.Name != "Song" {
		t.Errorf("open = %+v", open)
	}
	if err := s.genMgr.Stop("song"); err != nil {
		t.Fatal(err)
	}
	testutil.Eventually(t, func() bool { return allNotes(s).Totals.Unloads == 1 }, "no unload recorded")

	img := generation.Instance{ID: "img", Name: "Image", GPU: 1}
	startFakeGen(t, s, img, generation.KindSD, `echo "[W] model manager cannot make enough memory available on CUDA0: need 9000 MB device / 8000 MB budget"
exit 1`)
	testutil.Eventually(t, func() bool { return allNotes(s).Totals.FailedLoads == 1 }, "no failed start recorded")
	sum := allNotes(s)
	if sum.Totals.Crashes != 0 || len(sum.Failures) != 1 {
		t.Fatalf("totals = %+v", sum.Totals)
	}
	if f := sum.Failures[0]; f.Source != enginenotes.SourceSDServer || f.Cause != enginenotes.CauseOutOfVRAM || f.ExitStatus != 1 {
		t.Errorf("failed start = %+v", f)
	}
}

// sd-server given an argument it does not know prints the error, then
// hundreds of lines of help, then exits. The error is far above the last
// lines and the help has "error" in it: the failed start keeps the error,
// not the help, and the cause is a setting that is not supported.
func TestGenInstanceFailedStartAfterUsage(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	inst := generation.Instance{ID: "img", Name: "Image", GPU: 0}
	startFakeGen(t, s, inst, generation.KindSD, `echo "[E] error: unknown argument: --no-such-flag --- common.cpp:354"
echo "stable-diffusion.cpp version master-1028-37bc88b, commit 37bc88b"
echo "Usage: /data/builds/sd/sd-server [options]"
i=0; while [ $i -lt 120 ]; do echo "  --option-$i                 some help"; i=$((i+1)); done
echo "  --log-level                   minimum log level, one of [debug, verbose, info, warn, error] (default: info)"
exit 1`)

	testutil.Eventually(t, func() bool { return allNotes(s).Totals.FailedLoads == 1 }, "no failed start recorded")
	f := allNotes(s).Failures[0]
	if len(f.ErrorLines) == 0 || f.ErrorLines[0] != "[E] error: unknown argument: --no-such-flag" {
		t.Errorf("error lines = %q, want the unknown argument first", f.ErrorLines)
	}
	for _, l := range append(slices.Clone(f.ErrorLines), f.Tail...) {
		if strings.Contains(l, "--option-") || strings.Contains(l, "--log-level") {
			t.Errorf("help text kept: %q", l)
		}
	}
	if f.Cause != enginenotes.CauseUnsupported {
		t.Errorf("cause = %q, want a setting that is not supported", f.Cause)
	}
}

// A start made while Measure holds the instance carries Measure as its
// job, so running out of memory at a size Measure tries is expected.
func TestGenInstanceMeasureIsTheJob(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	s.genMgr.SetMeasuring("m", true)
	defer s.genMgr.SetMeasuring("m", false)
	startFakeGen(t, s, generation.Instance{ID: "m", Name: "M"}, generation.KindVideo, `sleep 0.5
echo "[W] VAE decode ran out of memory; retrying with temporal tiling --- backend_fit.cpp:518"
echo "[E] ggml: CUDA error: out of memory"
exit 1`)
	testutil.Eventually(t, func() bool { return allNotes(s).Totals.Crashes == 1 }, "no crash recorded")
	f := allNotes(s).Failures[0]
	if f.JobName != "Measure" || f.Cause != enginenotes.CauseOutOfVRAM || !f.Expected {
		t.Errorf("crash during Measure = %+v", f)
	}
}

// The dialog keeps generation instances out of the Models table and names
// the program on each failure; the strip names it for a generation one.
func TestNotesViewShowsGenerationSources(t *testing.T) {
	s := newTestServer(t)
	crash := withCause(enginenotes.Event{Kind: enginenotes.KindCrash, Time: viewNow.Add(-time.Hour),
		Source: enginenotes.SourceSDServer, ModelID: "wan", ModelName: "Wan video", ExitStatus: 134,
		Tail: []string{"last sd line"}}, enginenotes.CauseGPUFault)
	failed := withCause(enginenotes.Event{Kind: enginenotes.KindLoadFailed, Time: viewNow.Add(-2 * time.Hour),
		Source: enginenotes.SourceAceServer, ModelID: "song", ModelName: "Song"}, enginenotes.CauseOther)
	sum := enginenotes.Summary{
		CountingSince: viewNow.Add(-48 * time.Hour), From: viewNow.Add(-24 * time.Hour), To: viewNow,
		Totals: enginenotes.Totals{Loads: 2, Crashes: 1, FailedLoads: 1},
		Models: []enginenotes.ModelRow{
			{Source: enginenotes.SourceLlamaServer, ModelID: "q", Name: "Qwen", Loads: 1},
			{Source: enginenotes.SourceSDServer, ModelID: "wan", Name: "Wan video", Loads: 1, Crashes: 1},
		},
		Failures: []enginenotes.Event{crash, failed},
	}
	v := newNotesBodyView("24h", sum)
	if len(v.Models) != 1 || v.Models[0].Name != "Qwen" || len(v.GenInstances) != 1 || v.GenInstances[0].Source != "sd-server" {
		t.Fatalf("models = %+v, instances = %+v", v.Models, v.GenInstances)
	}
	rec := httptest.NewRecorder()
	s.renderPartial(rec, "engine_notes_body", v)
	html := rec.Body.String()
	for _, want := range []string{
		"<h4>Generation instances</h4>", "<td>Wan video</td>\n        <td>sd-server</td>",
		`<span class="engine-notes-source" title="The program that failed.">sd-server</span> · Wan video · crash`,
		`ace-server</span> · Song · failed to start`,
		"What sd-server wrote",
	} {
		if !strings.Contains(html, want) {
			t.Errorf("body is missing %q", want)
		}
	}
	if i, j := strings.Index(html, "<h4>Models</h4>"), strings.Index(html, "<h4>Generation instances</h4>"); i < 0 || j < i ||
		strings.Contains(html[i:j], "Wan video") {
		t.Error("the generation instance is in the Models table")
	}

	strip := newNotesStripView(sum)
	if !strings.Contains(strip.Text, "last: Wan video (sd-server, GPU fault)") {
		t.Errorf("strip = %q", strip.Text)
	}
}
