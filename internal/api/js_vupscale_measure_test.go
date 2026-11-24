package api

import (
	"encoding/json"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/videofile"
)

// TestVUpscaleMeasureJS runs the video upscaler Measure dialog's
// vupscaleClipLimitText from server.html on the cases the server's
// checkVUpscaleClip decides, and expects the same answers.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestVUpscaleMeasureJS(t *testing.T) {
	extracted, missing := extractFunctions(templateScripts(t, "server.html"), []string{"vupscaleClipLimitText"})
	if len(missing) > 0 {
		t.Fatalf("functions not found in server.html (renamed or removed?): %v", missing)
	}
	type tc struct {
		Script  string `json:"script"`
		Scale   int    `json:"scale"`
		W       int    `json:"w"`
		H       int    `json:"h"`
		Frames  int    `json:"frames"`
		Window  int    `json:"window"`
		Refused bool   `json:"refused"`
	}
	cases := []tc{
		{Script: "relayvsr", Scale: 2, W: 832, H: 480, Frames: 81, Window: 16},
		{Script: "relayvsr", Scale: 2, W: 1280, H: 720, Frames: 120, Window: 16},
		{Script: "relayvsr", Scale: 4, W: 1280, H: 720, Frames: 20, Window: 32},
		{Script: "relayvsr", Scale: 4, W: 1280, H: 720, Frames: 20, Window: 24},
		{Script: "relayvsr", Scale: 5, W: 640, H: 360, Frames: 300, Window: 2},
		{Script: "seedvr2", Scale: 2, W: 1280, H: 720, Frames: 20, Window: 9},
		{Script: "seedvr2", Scale: 2, W: 1280, H: 720, Frames: 20, Window: 5},
		{Script: "seedvr2", Scale: 3, W: 640, H: 360, Frames: 81, Window: 1},
	}
	refused := 0
	for i, c := range cases {
		info := videofile.Info{Width: c.W, Height: c.H, Frames: c.Frames}
		cases[i].Refused = checkVUpscaleClip(c.Script, c.Scale, info, c.Window) != nil
		if cases[i].Refused {
			refused++
		}
	}
	if refused == 0 || refused == len(cases) {
		t.Fatalf("the cases must include both answers; %d of %d refused", refused, len(cases))
	}
	b, _ := json.Marshal(cases)
	runNodeTest(t, map[string]string{
		"vupscale.js": extracted,
		"cases.json":  string(b),
		"run.js":      jsTestFile(t, "vupscale_measure_test.js"),
	})
}
