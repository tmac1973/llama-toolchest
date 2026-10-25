package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// The sidebar's map shows only what is on the GPUs now: a loaded LLM on
// each card it spans, a running backend, and a stopped backend's
// reservation, but not an enabled LLM that is not loaded. A GPU with
// nothing on it gets no bar.
func TestSidebarMapShowsOnlyWhatIsOnTheGPUs(t *testing.T) {
	gpus := []monitor.GPUInfo{
		{Index: 0, VRAMTotalMB: 16384},
		{Index: 1, VRAMTotalMB: 16384},
		{Index: 2, VRAMTotalMB: 16384},
	}
	allocs := []models.GPUAllocation{
		{ModelID: "qwen", ModelName: "Qwen3-30B", GPUs: []int{0, 1}, PerGPUGB: 8, TotalGB: 16, Color: "#111111"},
		{ModelID: "gemma", ModelName: "gemma-3", GPUs: []int{2}, PerGPUGB: 4, TotalGB: 4, Color: "#222222"},
		{ModelID: "gen:sd", ModelName: "sd (reserved)", GPUs: []int{1}, PerGPUGB: 4, TotalGB: 4, Color: "#333333", Reserved: true},
		{ModelID: "gen:ace", ModelName: "ace (reserved)", GPUs: []int{0}, PerGPUGB: 2, TotalGB: 2, Color: "#444444", Reserved: true},
	}
	got := buildSidebarMap(gpus, allocs, map[string]string{"qwen": "loaded"}, map[string]bool{"gen:sd": true})

	if len(got.GPUs) != 2 || got.GPUs[0].Index != 0 || got.GPUs[1].Index != 1 {
		t.Fatalf("bars = %+v, want GPU 0 and 1 only (gemma on GPU 2 is not loaded)", got.GPUs)
	}
	g0 := got.GPUs[0].Segments
	if len(g0) != 2 || g0[0].Name != "Qwen3-30B" || g0[0].Kind != sidebarLLM || g0[0].WidthPct != 50 ||
		g0[1].Name != "ace" || g0[1].Kind != sidebarStopped {
		t.Errorf("GPU 0 = %+v, want Qwen at 50%% and ace's reservation outlined", g0)
	}
	g1 := got.GPUs[1].Segments
	if len(g1) != 2 || g1[1].Name != "sd" || g1[1].Kind != sidebarGen || g1[1].WidthPct != 25 {
		t.Errorf("GPU 1 = %+v, want Qwen and the running sd at 25%%", g1)
	}
	if !strings.Contains(g1[1].Title, "running") || !strings.Contains(g0[1].Title, "stopped") {
		t.Errorf("tooltips = %q, %q; want each to say whether the backend runs", g1[1].Title, g0[1].Title)
	}
	var names []string
	for _, e := range got.Legend {
		names = append(names, e.Name)
	}
	if strings.Join(names, ",") != "Qwen3-30B,sd,ace" {
		t.Errorf("legend = %v, want the loaded LLM and both backends", names)
	}

	if m := buildSidebarMap(gpus, allocs[:2], nil, nil); len(m.GPUs) != 0 || len(m.Legend) != 0 {
		t.Errorf("nothing loaded gave %+v, want an empty map", m)
	}
}

// The template draws each kind with its own class and keeps the color in
// the style, so the outline of a stopped backend has its color too.
func TestSidebarMapRenders(t *testing.T) {
	s := newTestServer(t)
	data := monitorBar{Map: buildSidebarMap(
		[]monitor.GPUInfo{{Index: 0, VRAMTotalMB: 16384}},
		[]models.GPUAllocation{
			{ModelID: "qwen", ModelName: "Qwen3-30B", GPUs: []int{0}, PerGPUGB: 8, TotalGB: 8, Color: "#4e79a7"},
			{ModelID: "gen:ace", ModelName: "ace (reserved)", GPUs: []int{0}, PerGPUGB: 2, TotalGB: 2, Color: "#f28e2b", Reserved: true},
		},
		map[string]string{"qwen": "loading"}, nil)}
	rec := httptest.NewRecorder()
	s.renderPartial(rec, "monitor_bar", data)
	out := rec.Body.String()
	for _, want := range []string{
		"On the GPUs",
		`style="width:50.0%;background-color:#4e79a7;--side-map-color:#4e79a7;"`,
		`class="side-map-stopped"`,
		"Qwen3-30B: about 8.0 GiB on GPU 0 (LLM, loading)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}

	rec = httptest.NewRecorder()
	s.renderPartial(rec, "monitor_bar", monitorBar{})
	if strings.Contains(rec.Body.String(), "On the GPUs") {
		t.Error("an empty map still drew its heading")
	}
}
