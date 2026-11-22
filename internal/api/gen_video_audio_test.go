package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"math"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/audiofile"
	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// s2vSet is a speech-to-video set on the wan22-s2v recipe, whose
// snapshot recipe has a fixed 16 fps, and an instance measured at its limit.
func s2vSet() (generation.Instance, generation.ModelSet) {
	inst := generation.Instance{ID: "s2v", ReservedMiB: 1, MeasuredSuggestedMiB: 1, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 720}
	set := generation.ModelSet{ID: "s2v", Name: "Wan2.2 S2V 14B", Kind: generation.KindVideo, Recipe: "wan22-s2v",
		Capabilities: []string{generation.CapStartImage, generation.CapAudio},
		DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 16, MaxFrames: 81, MaxWidth: 1280, MaxHeight: 720,
		Components: []generation.Component{{Slot: "diffusion_model", Path: "/m/s2v.gguf"}}}
	return inst, set
}

// s2vWAV is seconds of a quiet 220 Hz sine as a 16 kHz mono WAV, base64.
func s2vWAV(seconds float64) string {
	n := int(seconds * 16000)
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(9000 * math.Sin(2*math.Pi*220*float64(i)/16000))
	}
	return base64.StdEncoding.EncodeToString(audiofile.WriteWAV(s, 16000, 1))
}

func testdataAudio(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("../audiofile/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(b)
}

func floatp(f float64) *float64 { return &f }

func TestAudioFrames(t *testing.T) {
	for _, tc := range []struct {
		seconds float64
		want    int
	}{{3.0, 49}, {2.9, 49}, {1.0, 17}, {0.01, 1}, {5.0, 81}, {6.0, 97}} {
		if got := audioFrames(tc.seconds, 16); got != tc.want {
			t.Errorf("audioFrames(%v, 16) = %d, want %d", tc.seconds, got, tc.want)
		}
	}
}

func TestBuildVideoRequestAudio(t *testing.T) {
	inst, set := s2vSet()
	img := "UE5H"
	build := func(in videoInput) (videoJobRequest, vidGenRequest, *requestError) {
		in.Prompt = "a person talking"
		return buildVideoRequest(in, inst, set)
	}

	t.Run("length from audio, fixed fps", func(t *testing.T) {
		shown, vid, verr := build(videoInput{StartImage: img, Audio: s2vWAV(3), FPS: intp(24)})
		if verr != nil {
			t.Fatal(verr)
		}
		if vid.Frames != 49 || vid.FPS != 16 || shown.FPS != 16 || !shown.Audio || shown.AudioSeconds != 3 {
			t.Errorf("frames %d fps %d (shown %+v), want 49 at 16 fps", vid.Frames, vid.FPS, shown)
		}
		if len(vid.RefAudios) != 1 {
			t.Fatalf("ref_audios = %d tracks", len(vid.RefAudios))
		}
		if videoInputs(vid) != "a start image and audio" {
			t.Errorf("inputs %q", videoInputs(vid))
		}
	})
	t.Run("seconds and frames win", func(t *testing.T) {
		_, vid, verr := build(videoInput{StartImage: img, Audio: s2vWAV(3), Seconds: floatp(2)})
		if verr != nil || vid.Frames != 32 {
			t.Errorf("seconds 2: %v, %d frames, want 32", verr, vid.Frames)
		}
		_, vid, verr = build(videoInput{StartImage: img, Audio: s2vWAV(3), Frames: intp(33)})
		if verr != nil || vid.Frames != 33 {
			t.Errorf("frames 33: %v, %d frames", verr, vid.Frames)
		}
	})
	t.Run("audio over the limit", func(t *testing.T) {
		_, _, verr := build(videoInput{StartImage: img, Audio: s2vWAV(6)})
		if verr == nil || verr.code != "over_limit" || !strings.Contains(verr.msg, "6.0 s") || !strings.Contains(verr.msg, "5.1 s") {
			t.Errorf("6 s with no length: %+v", verr)
		}
		if _, vid, verr := build(videoInput{StartImage: img, Audio: s2vWAV(6), Frames: intp(81)}); verr != nil || vid.Frames != 81 {
			t.Errorf("6 s cut to 81 frames: %+v", verr)
		}
	})
	t.Run("MP3 and FLAC go out as WAV", func(t *testing.T) {
		// The MP3 decodes about 0.07 s long (encoder delay and padding),
		// so its 1 s tone needs 18 frames, rounded up to 21.
		for name, frames := range map[string]int{"tone.mp3": 21, "tone.flac": 17} {
			_, vid, verr := build(videoInput{StartImage: img, Audio: testdataAudio(t, name)})
			if verr != nil {
				t.Fatalf("%s: %v", name, verr)
			}
			raw, _ := base64.StdEncoding.DecodeString(vid.RefAudios[0])
			if !bytes.HasPrefix(raw, []byte("RIFF")) {
				t.Errorf("%s sent as %q...", name, raw[:4])
			}
			if vid.Frames != frames {
				t.Errorf("%s: %d frames, want %d", name, vid.Frames, frames)
			}
		}
	})
	t.Run("refusals", func(t *testing.T) {
		ti2v := generation.ModelSet{ID: "ti2v", Name: "Wan2.2 TI2V 5B", Kind: generation.KindVideo, Recipe: "wan22-ti2v-5b",
			Capabilities: []string{generation.CapText, generation.CapStartImage}, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 24}
		if _, _, verr := buildVideoRequest(videoInput{Prompt: "p", Audio: s2vWAV(1)}, inst, ti2v); verr == nil || verr.code != "input_not_supported" {
			t.Errorf("audio on a set without audio: %+v", verr)
		}
		for name, tc := range map[string]struct {
			in   videoInput
			code string
		}{
			"no audio":       {videoInput{StartImage: img}, "audio_required"},
			"no start image": {videoInput{Audio: s2vWAV(1)}, "start_image_required"},
			"ogg":            {videoInput{StartImage: img, Audio: testdataAudio(t, "tone.ogg")}, "bad_audio"},
			"broken mp3":     {videoInput{StartImage: img, Audio: testdataAudio(t, "broken.mp3")}, "bad_audio"},
			"bad base64":     {videoInput{StartImage: img, Audio: "!!not base64!!"}, "bad_value"},
			"extra":          {videoInput{StartImage: img, Audio: s2vWAV(1), Extra: map[string]any{"ref_audios": []any{}}}, "bad_value"},
		} {
			if _, _, verr := build(tc.in); verr == nil || verr.code != tc.code {
				t.Errorf("%s: %+v, want %s", name, verr, tc.code)
			} else if tc.code == "bad_audio" && !strings.HasPrefix(verr.msg, "audio must be WAV, MP3 or FLAC") {
				t.Errorf("%s: message %q", name, verr.msg)
			}
		}
	})
}

// A recipe with a fixed frame rate other than S2V's: MiniMax-H3 FL2VA
// makes only 24 fps, which sd-server forces anyway.
func TestBuildVideoRequestFixedFPS(t *testing.T) {
	inst := generation.Instance{ID: "h3", ReservedMiB: 1, MeasuredSuggestedMiB: 1, MeasuredFrames: 241, MeasuredWidth: 1280, MeasuredHeight: 720}
	set := generation.ModelSet{ID: "h3", Name: "H3", Kind: generation.KindVideo, Recipe: "minimax-h3-fl2va",
		Capabilities: []string{generation.CapStartImage}, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 30}
	_, vid, verr := buildVideoRequest(videoInput{Prompt: "p", StartImage: "UE5H", FPS: intp(30), Seconds: floatp(2)}, inst, set)
	if verr != nil || vid.FPS != 24 || vid.Frames != 48 {
		t.Errorf("%+v: fps %d, %d frames, want 24 and 48", verr, vid.FPS, vid.Frames)
	}
}

func TestVideoJobMultipartAudio(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f, generation.CapText, generation.CapStartImage, generation.CapAudio)
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("prompt", "a person talking")
	part, _ := mw.CreateFormFile("start_image", "face.png")
	part.Write([]byte("PNGDATA"))
	part, _ = mw.CreateFormFile("audio", "speech.flac")
	raw, _ := base64.StdEncoding.DecodeString(testdataAudio(t, "tone.flac"))
	part.Write(raw)
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/videos/jobs", &buf)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	j := waitVideoJob(t, s, createVideoJobID(t, s, r))
	if j.Status != jobSucceeded || !j.Request.Audio || j.Request.AudioSeconds != 1 {
		t.Fatalf("job = %+v", j)
	}
	// The ti2v set's recipe has no fixed fps: 1 s at its 24 fps is 24,
	// rounded up to 25.
	tracks, _ := f.body["ref_audios"].([]any)
	if len(tracks) != 1 || f.body["video_frames"] != float64(25) {
		t.Fatalf("sd-server got %d tracks, %v frames", len(tracks), f.body["video_frames"])
	}
	wav, _ := base64.StdEncoding.DecodeString(tracks[0].(string))
	if !bytes.HasPrefix(wav, []byte("RIFF")) {
		t.Errorf("track starts %q", wav[:4])
	}
}

func TestV1ModelsFixedFPS(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f)
	get := func() map[string]any {
		rec := serve(s, httptest.NewRequest("GET", "/v1/models/ti2v", nil))
		var got struct {
			Meta struct {
				Video map[string]any `json:"video"`
			} `json:"meta"`
		}
		json.Unmarshal(rec.Body.Bytes(), &got)
		return got.Meta.Video
	}
	if v, ok := get()["fixed_fps"]; ok {
		t.Errorf("wan22-ti2v-5b has fixed_fps %v", v)
	}
	set, _ := s.gen.ModelSet("ti2v")
	set.Recipe = "wan22-s2v"
	if err := s.gen.UpdateModelSet(set); err != nil {
		t.Fatal(err)
	}
	if v := get()["fixed_fps"]; v != float64(16) {
		t.Errorf("wan22-s2v fixed_fps = %v, want 16", v)
	}
}
