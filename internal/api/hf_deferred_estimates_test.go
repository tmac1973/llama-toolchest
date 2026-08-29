package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/web"
)

func renderFiles(t *testing.T, name string, view hfModelView) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, name, view); err != nil {
		t.Fatalf("execute %s: %v", name, err)
	}
	return buf.String()
}

const gibT = int64(1) << 30

// The point of the change: a file whose figure is still being measured
// shows a placeholder, and the file list is on screen without waiting for
// the network reads that measurement needs.
func TestPendingFileShowsPlaceholderAndAsksForTheAnswer(t *testing.T) {
	view := hfModelView{
		ID: "unsloth/Big-GGUF", Source: modelsource.SourceHuggingFace,
		Files: []hfFileView{{
			ModelFile:       modelsource.File{Filename: "big.gguf", Size: 68 * gibT, Quant: "Q4_K_M"},
			FitsOnDisk:      true,
			EstimatePending: true,
		}},
		AvailableBytes: 1 << 46,
		AnyPending:     true,
	}
	out := renderFiles(t, "hf_files", view)

	if !strings.Contains(out, "Calculating…") {
		t.Errorf("no placeholder for a pending estimate; output=\n%s", out)
	}
	// The filename must be there already — that is the whole point.
	if !strings.Contains(out, "big.gguf") {
		t.Error("the file list is not rendered alongside the placeholder")
	}
	// And exactly one deferred request, not one per file.
	if n := strings.Count(out, "/api/hf/model/estimates"); n != 1 {
		t.Errorf("%d deferred requests, want exactly 1", n)
	}
}

// A listing with nothing pending must not issue the deferred request at
// all: a repeat visit is served from the probe cache, and a round trip
// that changes nothing is just latency.
func TestNothingPendingIssuesNoRequest(t *testing.T) {
	view := hfModelView{
		ID: "unsloth/Small-GGUF", Source: modelsource.SourceHuggingFace,
		Files: []hfFileView{{
			ModelFile:  modelsource.File{Filename: "small.gguf", Size: 3 * gibT, VRAMEstGB: 3.4},
			FitsOnDisk: true,
		}},
		AvailableBytes: 1 << 46,
	}
	out := renderFiles(t, "hf_files", view)

	if strings.Contains(out, "/api/hf/model/estimates") {
		t.Error("issued a deferred request with nothing pending")
	}
	if strings.Contains(out, "Calculating…") {
		t.Error("showed a placeholder for a figure that is already final")
	}
	if !strings.Contains(out, "3.4 GiB") {
		t.Errorf("final estimate not shown; output=\n%s", out)
	}
}

// The fill targets only the two cells per file. Replacing whole rows would
// discard a download progress row that appeared while the user waited.
func TestFillTargetsOnlyTheEstimateCells(t *testing.T) {
	view := hfModelView{
		ID: "unsloth/Big-GGUF", Source: modelsource.SourceHuggingFace,
		Files: []hfFileView{{ModelFile: modelsource.File{
			Filename: "big.gguf", Size: 68 * gibT,
			StreamedBytes: 27 * gibT, StreamProbed: true,
			VRAMEstGB: modelsource.EstimateVRAM(41 * gibT),
		}}},
	}
	out := renderFiles(t, "hf_file_estimates", view)

	if n := strings.Count(out, `hx-swap-oob="true"`); n != 2 {
		t.Errorf("%d out-of-band swaps, want 2 (the VRAM cell and the fit cell)", n)
	}
	for _, want := range []string{"vram-", "fit-", "45.1 GiB", "held in system memory"} {
		if !strings.Contains(out, want) {
			t.Errorf("fill is missing %q; output=\n%s", want, out)
		}
	}
	// Nothing structural: no rows, no download buttons.
	for _, unwanted := range []string{"<tr", "<button", "Download"} {
		if strings.Contains(out, unwanted) {
			t.Errorf("fill contains %q — it should carry only the two cells", unwanted)
		}
	}
}

// Both renders must agree, or the number flickers to something else when
// the answer lands.
func TestFirstRenderAndFillAgree(t *testing.T) {
	f := modelsource.File{
		Filename: "big.gguf", Size: 68 * gibT,
		StreamedBytes: 27 * gibT, StreamProbed: true,
		VRAMEstGB: modelsource.EstimateVRAM(41 * gibT),
	}
	settled := renderFiles(t, "hf_files", hfModelView{
		ID: "r", Files: []hfFileView{{ModelFile: f, FitsOnDisk: true}}, AvailableBytes: 1 << 46,
	})
	filled := renderFiles(t, "hf_file_estimates", hfModelView{
		ID: "r", Files: []hfFileView{{ModelFile: f}},
	})
	for _, want := range []string{"45.1 GiB", "held in system memory"} {
		if !strings.Contains(settled, want) || !strings.Contains(filled, want) {
			t.Errorf("%q appears in only one of the two renders", want)
		}
	}
}

// A file the probe would never inspect must not be marked pending: its
// estimate is final already, and a placeholder resolving to the same
// number misrepresents what is happening.
func TestOnlyProbableFilesArePending(t *testing.T) {
	for _, tt := range []struct {
		name   string
		size   int64
		shards int
		want   bool
	}{
		{"small single file", 3 * gibT, 1, false},
		{"large single file", 68 * gibT, 1, false}, // unsplit files are not probed
		{"large split file", 68 * gibT, 4, true},
		{"small split file", 3 * gibT, 4, false},
	} {
		if got := modelsource.ProbeApplies(tt.size, tt.shards); got != tt.want {
			t.Errorf("%s: ProbeApplies = %v, want %v", tt.name, got, tt.want)
		}
	}
}
