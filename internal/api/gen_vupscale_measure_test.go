package api

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/videofile"
)

func relaySet() generation.ModelSet {
	return generation.ModelSet{ID: "relay2", Kind: generation.KindVUpscale, Recipe: "relayvsr", Scale: 2,
		Components: []generation.Component{{Slot: "dit", Path: "/m/dit.safetensors"}, {Slot: "vae", Path: "/m/vae.safetensors"}}}
}

func TestMeasureClip(t *testing.T) {
	b, err := measureClip(64, 48, 5)
	if err != nil {
		t.Fatal(err)
	}
	info, err := videofile.Probe(bytes.NewReader(b), int64(len(b)))
	want := videofile.Info{Format: "avi", Width: 64, Height: 48, Frames: 5, FPSNum: 16, FPSDen: 1}
	if err != nil || info != want {
		t.Errorf("probe = %+v, %v", info, err)
	}
}

func TestVUpscaleMeasureSize(t *testing.T) {
	form := func(v url.Values) *http.Request {
		r := httptest.NewRequest("POST", "/", strings.NewReader(v.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		return r
	}
	got, err := vupscaleMeasureSize(form(url.Values{}), relaySet())
	if err != nil || got != (generation.VideoSize{Frames: 81, Width: 832, Height: 480, Window: 16}) {
		t.Errorf("defaults = %+v, %v", got, err)
	}
	seed := relaySet()
	seed.Recipe = "seedvr2"
	for name, tc := range map[string]struct {
		set  generation.ModelSet
		v    url.Values
		want string
	}{
		"width 100":   {relaySet(), url.Values{"width": {"100"}}, "multiples of 16"},
		"one frame":   {relaySet(), url.Values{"frames": {"1"}}, "from 2 to 1024"},
		"seed win 4":  {seed, url.Values{"window_size": {"4"}}, "window_size must be 1, 5, 9"},
		"over 256MiB": {relaySet(), url.Values{"width": {"1280"}, "height": {"720"}, "frames": {"200"}}, "sd.cpp reads at most 256 MiB"},
		"not number":  {relaySet(), url.Values{"frames": {"many"}}, "whole number"},
	} {
		if _, err := vupscaleMeasureSize(form(tc.v), tc.set); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want %q", name, err, tc.want)
		}
	}
}

func TestRunMeasureVUpscale(t *testing.T) {
	fastPoll(t)
	f := newFakeDGL(t)
	s := instanceServer(t)
	f.workDir = generation.InstanceDir(s.cfg.DataDir, "vup")
	u, _ := url.Parse(f.srv.URL)
	inst := generation.Instance{ID: "vup", Port: atoiPort(t, u.Port())}
	size := generation.VideoSize{Frames: 9, Width: 64, Height: 48, Window: 8}
	if err := s.runMeasureVUpscale(context.Background(), inst, relaySet(), size); err != nil {
		t.Fatal(err)
	}
	src := f.sources[0]
	for _, want := range []string{`input_path = "input/measure.avi"`, `output_path = "output/measure.webm"`, "window_size = 8", "seed = 1"} {
		if !strings.Contains(src, want) {
			t.Errorf("script lacks %q", want)
		}
	}
	for _, p := range []string{"input/measure.avi", "output/measure.webm"} {
		if _, err := os.Stat(filepath.Join(f.workDir, p)); err == nil {
			t.Errorf("%s was left behind", p)
		}
	}

	f2 := newFakeDGL(t)
	f2.fail = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: cudaMalloc failed: out of memory"
	u, _ = url.Parse(f2.srv.URL)
	inst.Port = atoiPort(t, u.Port())
	err := s.runMeasureVUpscale(context.Background(), inst, relaySet(), size)
	if err == nil || !strings.Contains(err.Error(), "ran out of GPU memory at this size") {
		t.Errorf("OOM: err = %v", err)
	}
}

func TestApplyMeasurementVUpscale(t *testing.T) {
	s := vramServer(t)
	inst := generation.Instance{ID: "v", GPU: 0, Enabled: true, ReservedMiB: 4096, MeasuredPeakMiB: 9000}
	size := generation.VideoSize{Frames: 81, Width: 832, Height: 480, Window: 16}
	got, err := s.applyMeasurement(inst, relaySet(), size, 0, 28000, true)
	if err != nil || got.ReservedMiB != generation.SuggestReservation(9000) {
		t.Fatalf("got %+v, %v", got, err)
	}
	if got.MeasuredFrames != 81 || got.MeasuredWidth != 832 || got.MeasuredHeight != 480 || got.MeasuredWindow != 16 {
		t.Errorf("measured clip not stored: %+v", got)
	}
	over := generation.Instance{ID: "v", GPU: 0, Enabled: true, MeasuredPeakMiB: 20000}
	if _, err := s.applyMeasurement(over, relaySet(), size, 0, 15000, true); err == nil {
		t.Error("a peak over the room was granted")
	}
}

func TestVUpscaleCeiling(t *testing.T) {
	inst := generation.Instance{MeasuredWidth: 832, MeasuredHeight: 480, MeasuredFrames: 81, MeasuredWindow: 16}
	info := func(w, h, f int) videofile.Info { return videofile.Info{Width: w, Height: h, Frames: f} }
	if e := checkVUpscaleCeiling(inst, info(832, 480, 81), 16); e != nil {
		t.Errorf("at the measured size: %v", e)
	}
	for name, tc := range map[string]struct {
		info   videofile.Info
		window int
	}{
		"wider":    {info(848, 480, 81), 16},
		"portrait": {info(480, 832, 81), 16},
		"longer":   {info(832, 480, 97), 16},
		"window":   {info(832, 480, 81), 17},
	} {
		e := checkVUpscaleCeiling(inst, tc.info, tc.window)
		if e == nil || !strings.Contains(e.msg, "measured with 832x480, 81 frames and window 16") {
			t.Errorf("%s: %v", name, e)
		}
	}
	if e := checkVUpscaleCeiling(generation.Instance{}, info(4000, 4000, 900), 32); e != nil {
		t.Errorf("unmeasured: %v", e)
	}
	if n := vupscaleDefaultWindow(relaySet(), generation.Instance{MeasuredWindow: 8}); n != 8 {
		t.Errorf("default window under a measured 8 = %d", n)
	}
	if n := vupscaleDefaultWindow(relaySet(), generation.Instance{}); n != 16 {
		t.Errorf("default window unmeasured = %d", n)
	}
}

func TestVUpscaleJobOverMeasuredSize(t *testing.T) {
	f := newFakeDGL(t)
	s := vupscaleServer(t, f, func(i *generation.Instance) {
		i.MeasuredWidth, i.MeasuredHeight, i.MeasuredFrames, i.MeasuredWindow = 64, 48, 5, 8
	})
	rec := serve(s, multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "vp8-key-64x48-9f.webm")))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "measured with 64x48, 5 frames and window 8") {
		t.Errorf("HTTP %d %s", rec.Code, rec.Body)
	}
	if len(f.sources) != 0 {
		t.Error("the refused clip reached sd-server")
	}
}

func TestEditClearsMeasuredWindow(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true, ReservedMiB: 20000,
		MeasuredPeakMiB: 18000, MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 832, MeasuredHeight: 480, MeasuredWindow: 16}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenEditInstance(rec, instanceReq("PUT", "/", "m", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "extra_args": {"--offload-to-cpu"}}))
	if inst, _ := s.gen.Instance("m"); inst.MeasuredWindow != 0 {
		t.Errorf("after edit: window %d, want cleared", inst.MeasuredWindow)
	}
}

func atoiPort(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatal(err)
	}
	return n
}
