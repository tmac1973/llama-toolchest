package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// ti2vSet is the TI2V 5B catalog set as downloaded.
func ti2vSet() generation.ModelSet {
	e, _ := generation.LookupCatalog("wan22-ti2v-5b")
	return generation.ModelSet{ID: e.ID, Name: e.Name, Kind: e.Kind, CatalogID: e.ID, Recipe: e.Recipe, Capabilities: e.Capabilities,
		DefaultWidth: e.DefaultWidth, DefaultHeight: e.DefaultHeight, DefaultFrames: e.DefaultFrames, DefaultFPS: e.DefaultFPS,
		MaxFrames: e.MaxFrames, MaxWidth: e.MaxWidth, MaxHeight: e.MaxHeight, Complete: true}
}

func intp(n int) *int { return &n }

// A request with more video than the instance was measured for is refused
// with a message naming both sizes and the preset to measure; any shape
// with less video is fine; the model's limit is still checked first.
func TestBuildVideoRequestMeasuredLimit(t *testing.T) {
	set := ti2vSet()
	medium := generation.Instance{ID: "ti2v", Name: "ti2v", ReservedMiB: 20000, MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704}
	_, _, verr := buildVideoRequest(videoInput{Prompt: "p", Size: "1280x704", Frames: intp(121)}, medium, set)
	if verr == nil || verr.code != "over_measured_size" {
		t.Fatalf("over the measured size: %v", verr)
	}
	for _, want := range []string{"up to 81 frames at 1280x704 (Medium)", "121 frames at 1280x704", "Measure, then Full"} {
		if !strings.Contains(verr.msg, want) {
			t.Errorf("message %q lacks %q", verr.msg, want)
		}
	}
	if _, _, verr := buildVideoRequest(videoInput{Prompt: "p", Size: "832x480", Frames: intp(121)}, medium, set); verr != nil {
		t.Errorf("5 s at 832×480 after measuring Medium: %v", verr)
	}
	if _, _, verr := buildVideoRequest(videoInput{Prompt: "p", Frames: intp(122)}, medium, set); verr == nil || verr.code != "over_limit" {
		t.Errorf("past the model's limit: %v, want over_limit first", verr)
	}
	never := generation.Instance{ID: "ti2v", Name: "ti2v", ReservedMiB: 26000}
	_, _, verr = buildVideoRequest(videoInput{Prompt: "p", Frames: intp(81)}, never, set)
	if verr == nil || verr.code != "over_measured_size" || !strings.Contains(verr.msg, "only allowed its test size") {
		t.Errorf("never measured: %v", verr)
	}
}

func TestVideoMeasureSize(t *testing.T) {
	set := ti2vSet()
	form := func(f, w, h string) *http.Request {
		r := httptest.NewRequest("POST", "/", strings.NewReader(url.Values{"frames": {f}, "width": {w}, "height": {h}}.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		return r
	}
	if got, err := videoMeasureSize(form("81", "1280", "704"), set); err != nil || got.Frames != 81 || got.Width != 1280 {
		t.Errorf("Medium = %+v, %v", got, err)
	}
	if _, err := videoMeasureSize(form("121", "704", "1280"), set); err != nil {
		t.Errorf("portrait at the limit: %v", err)
	}
	for name, tc := range map[string][3]string{
		"missing":     {"", "1280", "704"},
		"too long":    {"122", "1280", "704"},
		"too large":   {"33", "1920", "1080"},
		"not mult. 8": {"33", "836", "476"},
	} {
		if _, err := videoMeasureSize(form(tc[0], tc[1], tc[2]), set); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The measurement's decision: capped at the room, a video measurement
// always applies with its size, an image re-measurement leaves the
// reservation, and music is never capped.
func TestApplyMeasurement(t *testing.T) {
	s := vramServer(t)
	video := ti2vSet()
	full := generation.VideoSize{Frames: 121, Width: 1280, Height: 704}

	// Capped: a peak just above the room (within the runtime allowance)
	// gives a reservation equal to the room, and the instance stays enabled.
	inst := generation.Instance{ID: "v", GPU: 0, Enabled: true, MeasuredPeakMiB: 14500}
	got, err := s.applyMeasurement(inst, video, full, 0, 14000, true)
	if err != nil || got.ReservedMiB != 14000 || got.MeasuredSuggestedMiB != 14000 || !got.Enabled {
		t.Errorf("capped: %+v, %v", got, err)
	}
	if got.MeasuredFrames != 121 || got.MeasuredWidth != 1280 {
		t.Errorf("measured size not stored: %+v", got)
	}

	// A video re-measurement applies over an existing reservation.
	inst = generation.Instance{ID: "v", GPU: 0, Enabled: true, ReservedMiB: 12288, MeasuredPeakMiB: 20000}
	got, _ = s.applyMeasurement(inst, video, full, 0, 28000, true)
	if want := generation.SuggestReservation(20000); got.ReservedMiB != want {
		t.Errorf("video re-measure: reservation %d, want %d", got.ReservedMiB, want)
	}
	if l, m := generation.VideoLimit(got, video); !m || l.Label != "Full" {
		t.Errorf("limit after measuring Full = %+v, %v", l, m)
	}

	// An image re-measurement keeps the reservation but records the capped
	// suggestion for "Use suggested".
	image := generation.ModelSet{Kind: generation.KindSD}
	inst = generation.Instance{ID: "i", GPU: 0, Enabled: true, ReservedMiB: 10240, MeasuredPeakMiB: 15641}
	got, _ = s.applyMeasurement(inst, image, generation.VideoSize{}, 0, 14000, true)
	if got.ReservedMiB != 10240 || got.MeasuredSuggestedMiB != 14000 {
		t.Errorf("image re-measure: %+v", got)
	}

	// Music is never capped and records no suggestion of its own.
	music := generation.ModelSet{Kind: generation.KindAce}
	inst = generation.Instance{ID: "a", GPU: 0, Enabled: true, MeasuredPeakMiB: 7300}
	got, _ = s.applyMeasurement(inst, music, generation.VideoSize{}, 0, 0, false)
	if got.ReservedMiB != generation.SuggestReservation(7300) || got.MeasuredSuggestedMiB != 0 {
		t.Errorf("music: %+v", got)
	}
}

// The room is the card's budget less the other instances' reservations.
func TestGenRoom(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "other", Port: 8081, Enabled: true, ReservedMiB: 10240})
	inst := generation.Instance{ID: "v", GPU: 0}
	room, ok := s.genRoomMiB(inst)
	spec := s.hardware().GPUs[0]
	spec.ReservedMiB = 10240
	if !ok || room != models.CardBudgetMiB(spec) {
		t.Errorf("room = %d, %v; want the budget less the other reservation (%d)", room, ok, models.CardBudgetMiB(spec))
	}
}

// Measure makes the test video at the chosen size.
func TestMeasureVideoUsesChosenSize(t *testing.T) {
	fastVidPoll(t)
	f := newFakeVidServer(t, 0, "completed")
	if err := runMeasureVideo(context.Background(), f.srv.URL, ti2vSet(), generation.VideoSize{Frames: 81, Width: 1280, Height: 704}); err != nil {
		t.Fatal(err)
	}
	if f.body["video_frames"] != float64(81) || f.body["width"] != float64(1280) || f.body["height"] != float64(704) {
		t.Errorf("sd-server got %v, want 81 frames at 1280×704", f.body)
	}
}

func TestEditClearsMeasuredSize(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true, ReservedMiB: 20000,
		MeasuredPeakMiB: 18000, MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenEditInstance(rec, instanceReq("PUT", "/", "m", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "extra_args": {"--offload-to-cpu"}}))
	inst, _ := s.gen.Instance("m")
	if inst.MeasuredSuggestedMiB != 0 || inst.MeasuredFrames != 0 || inst.MeasuredWidth != 0 || inst.MeasuredHeight != 0 {
		t.Errorf("after edit: %+v, want the measured size cleared", inst)
	}
}

func TestV1ModelsVideoMeasuredMeta(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddModelSet(ti2vSet()); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "ti2v", ModelSetID: "wan22-ti2v-5b", Port: 8090, Enabled: true, ReservedMiB: 20000,
		MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704})
	rec := serve(s, httptest.NewRequest("GET", "/v1/models/ti2v", nil))
	var got struct {
		Meta struct {
			Video map[string]any `json:"video"`
		} `json:"meta"`
	}
	json.Unmarshal(rec.Body.Bytes(), &got)
	v := got.Meta.Video
	if v["measured_frames"] != float64(81) || v["measured_size"] != "1280x704" || v["measured_pixel_frames"] != float64(81*1280*704) || v["max_frames"] != float64(121) {
		t.Errorf("meta.video = %v", v)
	}
}

// The instance row shows what a video instance was measured for, and its
// Measure button carries the presets for the dialog.
func TestInstanceRowVideoMeasure(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddModelSet(ti2vSet()); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "ti2v", ModelSetID: "wan22-ti2v-5b", Port: 8090, Enabled: true, ReservedMiB: 20000,
		MeasuredPeakMiB: 18000, MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704})
	rows := s.genInstanceRows(s.gen)
	var buf bytes.Buffer
	if err := s.pages["server.html"].ExecuteTemplate(&buf, "gen_instances_table", genInstancesData{Rows: rows}); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"measured for up to 81 frames at 1280×704 (Medium)", "openGenMeasureVideo(this)", "Full", "measuring Short took under a minute"} {
		if !strings.Contains(out, want) {
			t.Errorf("row lacks %q", want)
		}
	}
	if rows[0].SuggestedMiB != 20000 {
		t.Errorf("suggested = %d, want the stored suggestion", rows[0].SuggestedMiB)
	}
}

// A video measurement that does not fit leaves the size the instance was
// last measured for, and its reservation, as they were.
func TestApplyMeasurementKeepsSizeWhenItDoesNotFit(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "other", Port: 8081, Enabled: true, ReservedMiB: 20000})
	video := ti2vSet()
	medium := generation.Instance{ID: "v", GPU: 0, Enabled: true, ReservedMiB: 8192, MeasuredSuggestedMiB: 8192,
		MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704, MeasuredPeakMiB: 30000}
	// No room cap (no telemetry case), so the suggestion is far over what fits.
	got, err := s.applyMeasurement(medium, video, generation.VideoSize{Frames: 121, Width: 1280, Height: 704}, 0, 0, false)
	if err == nil {
		t.Fatal("a suggestion that does not fit was applied")
	}
	if got.ReservedMiB != 8192 || got.MeasuredFrames != 81 || got.MeasuredSuggestedMiB != 8192 {
		t.Errorf("after a failed Full measurement: %+v, want the Medium state kept", got)
	}
	if l, m := generation.VideoLimit(got, video); !m || l.Label != "Medium" {
		t.Errorf("limit = %+v, %v; want Medium still", l, m)
	}
}

// A video size whose measured peak is over the room (plus the runtime
// allowance) is not granted, whatever the cap would allow: sd-server went
// past --max-vram decoding such videos. Within the allowance it is.
func TestApplyMeasurementRefusesVideoOverRoom(t *testing.T) {
	s := vramServer(t)
	video := ti2vSet()
	short := generation.Instance{ID: "v", GPU: 0, Enabled: true, ReservedMiB: 26624, MeasuredSuggestedMiB: 26624,
		MeasuredFrames: 33, MeasuredWidth: 832, MeasuredHeight: 480}
	// The R9700 Medium run: peak 30.6 GiB held to a 28.2 GiB room.
	over := short
	over.MeasuredPeakMiB = 31302
	got, err := s.applyMeasurement(over, video, generation.VideoSize{Label: "Medium", Frames: 81, Width: 1280, Height: 704}, 0, 28846, true)
	if err == nil || !strings.Contains(err.Error(), "measure a smaller size") {
		t.Fatalf("err = %v, want the size refused", err)
	}
	if got.ReservedMiB != 26624 || got.MeasuredFrames != 33 || !got.Enabled {
		t.Errorf("after the refused size: %+v, want the Short state kept", got)
	}
	// The A4000 Short run: peak 15.3 GiB held to 14.7 GiB, within the
	// allowance.
	a4000 := generation.Instance{ID: "v", GPU: 0, Enabled: true, MeasuredPeakMiB: 15641}
	got, err = s.applyMeasurement(a4000, video, generation.VideoSize{Frames: 33, Width: 832, Height: 480}, 0, 15062, true)
	if err != nil || got.ReservedMiB != 15062 || got.MeasuredFrames != 33 {
		t.Errorf("A4000 Short: %+v, %v; want granted at the room", got, err)
	}
	// Image sets are still capped and granted.
	image := generation.Instance{ID: "i", GPU: 0, Enabled: true, MeasuredPeakMiB: 31302}
	if got, err := s.applyMeasurement(image, generation.ModelSet{Kind: generation.KindSD}, generation.VideoSize{}, 0, 20000, true); err != nil || got.ReservedMiB != 20000 {
		t.Errorf("image over the room: %+v, %v; want capped and granted", got, err)
	}
}

// A capped reservation leaves up to 0.5 GiB of the room free, but never
// goes below the measured peak: the figures are the R9700 Lightning run
// (refused at its start when capped at the room) and the A4000 TI2V run.
func TestCapReservation(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		suggest, peak, room, want int
	}{
		{"fits with slack", 20000, 18000, 28000, 20000},
		{"R9700 Lightning: slack left", generation.SuggestReservation(29240), 29240, 29931, 29931 - reserveSlackMiB},
		{"slack would cut the peak", generation.SuggestReservation(29600), 29600, 29931, 29600},
		{"A4000 TI2V: peak over the room", generation.SuggestReservation(15641), 15641, 15062, 15062},
		{"just inside the room", 21900, 20000, 22000, 22000 - reserveSlackMiB},
	} {
		if got := capReservation(tc.suggest, tc.peak, tc.room); got != tc.want {
			t.Errorf("%s: capReservation(%d, %d, %d) = %d, want %d", tc.name, tc.suggest, tc.peak, tc.room, got, tc.want)
		}
	}
}

// A backend that stops during Measure is blamed on memory unless its log
// shows a crash without running out of memory.
func TestBackendStoppedError(t *testing.T) {
	compute2 := []string{
		"[I] sampling completed, taking 101.96s --- video.cpp:1776",
		"[E] ggml: ggml_cuda_compute_forward: CONCAT failed",
		"[E] ggml: CUDA error: unspecified launch failure",
		"/data/stable-diffusion.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:111: CUDA error",
		"==> sd-server a14b-fast exited with error: signal: aborted (core dumped)",
	}
	err := backendStoppedError(compute2)
	if !strings.Contains(err.Error(), "crashed") || !strings.Contains(err.Error(), "(ggml: ggml_cuda_compute_forward: CONCAT failed)") {
		t.Errorf("CUDA crash: %v", err)
	}
	// The LTX-2.3 crash on the A4000 came while the progress bar showed.
	bar := []string{"  |===>   |  1/15 -   1.13s/it - ETA 00:15\x1b[K\r  |======>   |  7/15 -   7.58it/s - ETA 00:02\x1b[K[E] ggml: ggml_cuda_compute_forward: CONCAT failed"}
	if err := backendStoppedError(bar); !strings.Contains(err.Error(), "generation (ggml: ggml_cuda_compute_forward: CONCAT failed)") {
		t.Errorf("crash after a progress bar: %v", err)
	}
	if got := cleanLogLine("[E] ggml: CUDA error\r\n"); got != "ggml: CUDA error" {
		t.Errorf("line ending in CRLF: %q", got)
	}
	oom := []string{
		"[W] model manager cannot make enough memory available on ROCm0: need 8092.58 MB device / 7580.58 MB budget --- model_manager.cpp:2013",
		"[E] wan_vae segment 1/1 (graph) failed during weight preparation --- ggml_runner.cpp:898",
		"==> sd-server x exited with error: signal: aborted (core dumped)",
	}
	if err := backendStoppedError(oom); !strings.Contains(err.Error(), "ran out of GPU memory") {
		t.Errorf("out of memory: %v", err)
	}
	// TI2V 5B at 81 frames on the A4000 (2026-10-30): the whole decode ran
	// out of memory, sd.cpp retried twice, then crashed in the small-piece
	// decode. The memory warnings before the last retry were recovered
	// from; the crash after it is what stopped the backend.
	retried := []string{
		"[W] model manager cannot make enough memory available on CUDA0: need 25114.96 MB device / 24602.96 MB budget --- model_manager.cpp:2013",
		"[E] wan_vae segment 1/1 (graph) failed during weight preparation --- ggml_runner.cpp:898",
		"[W] VAE decode ran out of memory; retrying with temporal tiling --- backend_fit.cpp:518",
		"[W] model manager cannot make enough memory available on CUDA0: need 28560.03 MB device / 28048.03 MB budget --- model_manager.cpp:2013",
		"[W] VAE decode ran out of memory; retrying with spatial+temporal tiling --- backend_fit.cpp:518",
		"[E] ggml: ggml_cuda_compute_forward: CONCAT failed",
		"[E] ggml: CUDA error: unspecified launch failure",
		"==> sd-server ti2v-log exited with error: signal: aborted (core dumped)",
	}
	if err := backendStoppedError(retried); !strings.Contains(err.Error(), "crashed during the test generation (ggml: ggml_cuda_compute_forward: CONCAT failed), while decoding the video in smaller pieces") {
		t.Errorf("crash after a decode retry: %v", err)
	}
	// Running out of memory again after a retry is still memory.
	again := append(slices.Clone(retried[:5]), "[E] ggml: CUDA error: out of memory", "==> sd-server x exited with error: signal: aborted (core dumped)")
	if err := backendStoppedError(again); !strings.Contains(err.Error(), "ran out of GPU memory") {
		t.Errorf("out of memory after a retry: %v", err)
	}
	if err := backendStoppedError([]string{"[I] sampling", "==> sd-server x exited with error: signal: killed"}); !strings.Contains(err.Error(), "ran out of GPU memory") {
		t.Errorf("no crash in the log: %v", err)
	}
	if err := backendStoppedError(nil); !strings.Contains(err.Error(), "ran out of GPU memory") {
		t.Errorf("no log: %v", err)
	}
}
