package api

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// a14bLog is sd-server's log for a one-step-per-expert test of a Wan2.2
// I2V A14B pair (sd.cpp master-1015, R9700, 2026-10-28), progress bars
// and all.
var a14bLog = []string{
	"[I] encode_first_stage completed, taking 1504 ms --- video.cpp:1033",
	"[I] get_learned_condition completed, taking 1.97s --- video.cpp:1259",
	"\r  |==================================================| 1/1 -  46.00s/it\x1b[K\r[I] sampling(high noise) completed, taking 46.03s --- video.cpp:1735",
	"[I] sampling completed, taking 45.93s --- video.cpp:1776",
	"[I] generating latent video completed, taking 92.10s --- video.cpp:1920",
	"[I] decode_first_stage completed, taking 15.21s --- video.cpp:1290",
}

func TestMeasureLogStages(t *testing.T) {
	l := &measureLog{}
	if got := l.stage(); got != "during sampling" {
		t.Errorf("stage before any line = %q", got)
	}
	want := []string{"during sampling", "during sampling", "during sampling", "during the decode", "during the decode", "after the decode"}
	for i, line := range a14bLog {
		l.add(line, time.Unix(int64(i), 0))
		if got := l.stage(); got != want[i] {
			t.Errorf("stage after line %d = %q, want %q", i, got, want[i])
		}
	}
	if l.decodeFallback() || l.spatialFallback() {
		t.Errorf("fallback %v spatial %v, want neither", l.decodeFallback(), l.spatialFallback())
	}
	if l.lastLine() != time.Unix(int64(len(a14bLog)-1), 0) {
		t.Errorf("lastLine = %v", l.lastLine())
	}
}

func TestMeasureLogFallbacks(t *testing.T) {
	l := &measureLog{}
	l.add("[W] VAE decode ran out of memory; retrying with temporal tiling --- backend_fit.cpp:518", time.Now())
	if !l.decodeFallback() || l.spatialFallback() {
		t.Errorf("temporal retry: fallback %v spatial %v", l.decodeFallback(), l.spatialFallback())
	}
	l = &measureLog{}
	l.add("[W] Reducing VAE decode tiles from 832x480 to 416x240 image pixels --- backend_fit.cpp:513", time.Now())
	l.add("[W] VAE decode ran out of memory; retrying with spatial+temporal tiling --- backend_fit.cpp:518", time.Now())
	if !l.decodeFallback() || !l.spatialFallback() {
		t.Errorf("spatial retry: fallback %v spatial %v", l.decodeFallback(), l.spatialFallback())
	}
}

func TestMeasureVideoRunsOneStep(t *testing.T) {
	fastVidPoll(t)
	steps := func(body map[string]any, key string) any {
		p, _ := body[key].(map[string]any)
		return p["sample_steps"]
	}
	f := newFakeVidServer(t, 0, "completed")
	ti2v := generation.ModelSet{Kind: generation.KindVideo, Capabilities: []string{generation.CapText}, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 24,
		Components: []generation.Component{{Slot: "diffusion_model"}}}
	if err := runMeasureVideo(context.Background(), f.srv.URL, ti2v, generation.VideoSize{}); err != nil {
		t.Fatal(err)
	}
	if steps(f.body, "sample_params") != float64(1) || f.body["high_noise_sample_params"] != nil {
		t.Errorf("single model: body %v, want one step and no high-noise steps", f.body)
	}
	f = newFakeVidServer(t, 0, "completed")
	a14b := ti2v
	a14b.Components = []generation.Component{{Slot: "diffusion_model"}, {Slot: "high_noise_diffusion_model"}}
	if err := runMeasureVideo(context.Background(), f.srv.URL, a14b, generation.VideoSize{}); err != nil {
		t.Fatal(err)
	}
	if steps(f.body, "sample_params") != float64(1) || steps(f.body, "high_noise_sample_params") != float64(1) {
		t.Errorf("pair: body %v, want one step on each expert", f.body)
	}
}

func TestVidGenBodyKeepsExtraSampleParams(t *testing.T) {
	r := vidGenRequest{Prompt: "x", Steps: 1, Extra: map[string]any{"sample_params": map[string]any{"scheduler": "simple"}}}
	raw, err := r.body()
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	json.Unmarshal(raw, &m)
	p, _ := m["sample_params"].(map[string]any)
	if p["sample_steps"] != float64(1) || p["scheduler"] != "simple" {
		t.Errorf("sample_params = %v, want steps and the extra scheduler", m["sample_params"])
	}
	raw, _ = vidGenRequest{Prompt: "x"}.body()
	if strings.Contains(string(raw), "sample_params") {
		t.Errorf("a job without Steps sent sample_params: %s", raw)
	}
}

func TestDurationText(t *testing.T) {
	for sec, want := range map[int]string{0: "", 45: "about 45 s", 119: "about 119 s", 659: "about 11 min", 7260: "about 2 h 1 min"} {
		if got := durationText(sec); got != want {
			t.Errorf("durationText(%d) = %q, want %q", sec, got, want)
		}
	}
}

func TestWithPeak(t *testing.T) {
	base := errors.New("the test generation failed")
	if got := withPeak(base, 0); got != base {
		t.Errorf("withPeak with no peak changed the error: %v", got)
	}
	got := withPeak(base, 30720)
	if !errors.Is(got, base) || !strings.Contains(got.Error(), "it reached 30.0 GiB") {
		t.Errorf("withPeak = %v", got)
	}
}

func TestMeasureNoteMentionsFallback(t *testing.T) {
	inst := generation.Instance{MeasuredPeakMiB: 20000, ReservedMiB: 22016, MeasuredSuggestedMiB: 22016, MeasuredFrames: 81, MeasuredDecodeFallback: true}
	if note := measureNote(0, inst, false, false); !strings.Contains(note, "a few frames at a time") {
		t.Errorf("note = %q", note)
	}
}

func TestMeasureAbort(t *testing.T) {
	base := measureRunState{peak: 20000, room: 28000, sawBusy: true, quiet: time.Minute, size: "81 frames at 1280x704", stage: "during the decode"}
	if err := measureAbort(base); err != nil {
		t.Errorf("a run within its room was stopped: %v", err)
	}
	over := base
	over.peak = 28000 + measureOverheadMiB + 1
	if err := measureAbort(over); err == nil || !strings.Contains(err.Error(), "81 frames at 1280x704 needed more than") || !strings.Contains(err.Error(), "during the decode") {
		t.Errorf("over the room: %v", err)
	}
	allowed := base
	allowed.peak = 28000 + measureOverheadMiB
	if err := measureAbort(allowed); err != nil {
		t.Errorf("a peak within the allowance was stopped: %v", err)
	}
	noRoom := over
	noRoom.room = 0
	if err := measureAbort(noRoom); err != nil {
		t.Errorf("a run with no room to keep to was stopped for its peak: %v", err)
	}
	spatial := base
	spatial.spatial = true
	if err := measureAbort(spatial); err != nil {
		t.Errorf("a spatial fallback off ROCm was stopped: %v", err)
	}
	spatial.rocm = true
	if err := measureAbort(spatial); err == nil || !strings.Contains(err.Error(), "Radeon") {
		t.Errorf("a spatial fallback on ROCm: %v", err)
	}
	stuck := base
	stuck.quiet = measureStuckAfter + time.Second
	if err := measureAbort(stuck); err == nil || !strings.Contains(err.Error(), "stopped making progress") {
		t.Errorf("a stuck run: %v", err)
	}
	stuck.sawBusy = false
	if err := measureAbort(stuck); err != nil {
		t.Errorf("a card with no use reading was taken for stuck: %v", err)
	}
}
