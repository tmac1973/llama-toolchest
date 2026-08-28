package api

import (
	"bytes"
	"context"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/modelscope"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/web"
)

// An empty or unrecognized source must resolve to HuggingFace: every
// download record and registry entry written before ModelScope existed
// has an empty source, and links and re-downloads have to keep working.
func TestSourceClientDefaultsToHuggingFace(t *testing.T) {
	s := &Server{}
	for _, id := range []string{"", "hf", "nonsense", "HUGGINGFACE"} {
		got := s.sourceClient(id).ModelURL("unsloth/Qwen3-8B-GGUF")
		if !strings.HasPrefix(got, "https://huggingface.co/") {
			t.Errorf("sourceClient(%q).ModelURL = %q, want a huggingface.co URL", id, got)
		}
	}
	got := s.sourceClient(modelsource.SourceModelScope).ModelURL("unsloth/Qwen3-8B-GGUF")
	if !strings.HasPrefix(got, "https://modelscope.cn/") {
		t.Errorf("modelscope source gave %q, want a modelscope.cn URL", got)
	}
}

// The link next to a search result must point at the host the result came
// from — sending someone to a HuggingFace URL for a ModelScope-only repo
// is a 404 with no explanation.
func TestSearchResultsLinkToTheirOwnSource(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	results := []modelsource.SearchResult{{ID: "unsloth/Qwen3-8B-GGUF", Downloads: 5, Likes: 1}}

	for _, tc := range []struct{ source, wantHost, wantName string }{
		{modelsource.SourceHuggingFace, "https://huggingface.co/unsloth/Qwen3-8B-GGUF", "HuggingFace"},
		{modelsource.SourceModelScope, "https://modelscope.cn/models/unsloth/Qwen3-8B-GGUF", "ModelScope"},
	} {
		var buf bytes.Buffer
		data := struct {
			Results any
			Source  string
		}{Results: results, Source: tc.source}
		if err := base.ExecuteTemplate(&buf, "hf_results", data); err != nil {
			t.Fatalf("%s: execute: %v", tc.source, err)
		}
		out := buf.String()
		if !strings.Contains(out, tc.wantHost) {
			t.Errorf("%s: results do not link to %s; output=\n%s", tc.source, tc.wantHost, out)
		}
		if !strings.Contains(out, "Open on "+tc.wantName) {
			t.Errorf("%s: link title does not name %s", tc.source, tc.wantName)
		}
		// The Files button has to carry the source, or expanding a
		// ModelScope result would list files from HuggingFace.
		if !strings.Contains(out, "source="+tc.source) {
			t.Errorf("%s: Files request does not carry the source", tc.source)
		}
	}
}

// The download button posts the source back, so the bytes are fetched
// from the host the user was actually browsing.
func TestFileListCarriesSourceIntoDownload(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	view := hfModelView{
		ID:     "unsloth/Qwen3-8B-GGUF",
		Source: modelsource.SourceModelScope,
		Files: []hfFileView{{
			ModelFile:  modelsource.File{Filename: "Qwen3-8B-Q4_K_M.gguf", Size: 100, Quant: "Q4_K_M"},
			FitsOnDisk: true,
		}},
		AvailableBytes: 1 << 40,
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "hf_files", view); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(buf.String(), `"source":"modelscope"`) {
		t.Errorf("download button does not post the source; output=\n%s", buf.String())
	}
}

// The browse page's only new control is the source radio, and its
// initial position comes from the configured default — the whole point
// of the setting.
func TestBrowsePageRadioFollowsConfiguredDefault(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/models_browse.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	render := func(def string) string {
		var buf bytes.Buffer
		data := struct {
			pageData
			DefaultSource string
		}{DefaultSource: def}
		if err := base.ExecuteTemplate(&buf, "content", data); err != nil {
			t.Fatalf("execute (default=%q): %v", def, err)
		}
		return buf.String()
	}

	// checkedValue reports which radio carries the checked attribute.
	checkedValue := func(out string) string {
		for _, v := range []string{"hf", "modelscope"} {
			i := strings.Index(out, `value="`+v+`"`)
			if i < 0 {
				t.Fatalf("radio %q missing from the page", v)
			}
			rest := out[i:]
			if end := strings.Index(rest, ">"); end > 0 && strings.Contains(rest[:end], "checked") {
				return v
			}
		}
		return ""
	}

	for _, tc := range []struct{ configured, want string }{
		{"", "hf"},                   // unset falls back to HuggingFace
		{"hf", "hf"},                 //
		{"modelscope", "modelscope"}, // the setting takes effect
		{"nonsense", "hf"},           // an unrecognized value must not leave nothing selected
	} {
		if got := checkedValue(render(tc.configured)); got != tc.want {
			t.Errorf("default %q: %q is checked, want %q", tc.configured, got, tc.want)
		}
	}
}

// The ModelScope client must satisfy the same contract the API layer
// programs against, or none of the above holds together.
func TestModelScopeClientIsASource(t *testing.T) {
	var c modelsource.Client = modelscope.NewClient("")
	if _, err := c.GetModel(context.Background(), "not-an-id"); err == nil {
		t.Error("GetModel should reject an id that isn't owner/name")
	}
}

// The configured default answers "where should this new request go", not
// "where did this model come from". A model downloaded before ModelScope
// existed has an empty stored source and came from HuggingFace; setting
// the default to ModelScope must not repoint its link at a host it was
// never on.
func TestDefaultSourceDoesNotRewriteStoredRecords(t *testing.T) {
	s := &Server{cfg: &config.Config{DefaultModelSource: modelsource.SourceModelScope}}

	// A new request with no explicit source follows the preference...
	if got := s.requestSource(""); got != modelsource.SourceModelScope {
		t.Errorf("requestSource(\"\") = %q, want the configured default", got)
	}
	// ...while an explicit one still wins.
	if got := s.requestSource(modelsource.SourceHuggingFace); got != modelsource.SourceHuggingFace {
		t.Errorf("requestSource(hf) = %q, want hf", got)
	}
	// But a stored record with no source is still HuggingFace, whatever
	// the preference says — this is what the model-card link uses.
	if got := modelsource.NormalizeSource(""); got != modelsource.SourceHuggingFace {
		t.Errorf("NormalizeSource(\"\") = %q, want hf regardless of preference", got)
	}
	url := s.sourceClient(modelsource.NormalizeSource("")).ModelURL("unsloth/Qwen3-8B-GGUF")
	if !strings.HasPrefix(url, "https://huggingface.co/") {
		t.Errorf("pre-existing model links to %q, want huggingface.co", url)
	}
}

// An unset or unrecognized preference must resolve to HuggingFace rather
// than leaving the browse page with nothing selected.
func TestDefaultModelSourceFallback(t *testing.T) {
	for _, tc := range []struct{ configured, want string }{
		{"", modelsource.SourceHuggingFace},
		{"hf", modelsource.SourceHuggingFace},
		{"nonsense", modelsource.SourceHuggingFace},
		{"modelscope", modelsource.SourceModelScope},
	} {
		s := &Server{cfg: &config.Config{DefaultModelSource: tc.configured}}
		if got := s.defaultModelSource(); got != tc.want {
			t.Errorf("configured %q: defaultModelSource = %q, want %q", tc.configured, got, tc.want)
		}
	}
	// A nil config must not panic — the render tests construct bare Servers.
	if got := (&Server{}).defaultModelSource(); got != modelsource.SourceHuggingFace {
		t.Errorf("zero Server: %q, want hf", got)
	}
}

// A streamed table must be excluded from the VRAM estimate and shown as
// such, or the fit verdict stays wrong in exactly the case the probe
// exists to fix.
func TestFileListShowsStreamedPortion(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	const gib = int64(1) << 30
	view := hfModelView{
		ID:     "unsloth/Qwen3.8-Flash-Next-GGUF",
		Source: modelsource.SourceModelScope,
		Files: []hfFileView{{
			ModelFile: modelsource.File{
				Filename: "q.gguf", Size: 68 * gib, Quant: "UD_IQ1_S",
				StreamedBytes: 27 * gib, StreamProbed: true,
				VRAMEstGB: modelsource.EstimateVRAM(41 * gib),
			},
			FitsOnDisk: true,
		}},
		AvailableBytes: 1 << 46,
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "hf_files", view); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "streamed from disk") {
		t.Errorf("streamed portion not surfaced; output=\n%s", out)
	}
	// The full download is still what lands on disk, so Size must not be
	// quietly reduced to the resident part.
	if !strings.Contains(out, "68.0 GiB") {
		t.Error("Size column should still show the whole download")
	}
	if !strings.Contains(out, "45.1 GiB") {
		t.Errorf("VRAM estimate should cover only the resident part; output=\n%s", out)
	}
}

// applyProbe is what turns a measurement into a corrected estimate, and
// its edge cases decide whether a bad probe can make things worse.
func TestApplyProbe(t *testing.T) {
	const gib = int64(1) << 30
	base := modelsource.File{Size: 68 * gib, VRAMEstGB: modelsource.EstimateVRAM(68 * gib)}

	// A probe that did not run leaves the estimate exactly as it was.
	f := base
	applyProbe(&f, modelsource.ProbeResult{})
	if f.VRAMEstGB != base.VRAMEstGB || f.StreamProbed {
		t.Errorf("unprobed file was modified: %+v", f)
	}

	// A completed probe finding no table records the fact but changes
	// nothing, so the listing does not re-probe it.
	f = base
	applyProbe(&f, modelsource.ProbeResult{Probed: true})
	if !f.StreamProbed || f.VRAMEstGB != base.VRAMEstGB {
		t.Errorf("no-table probe should record and not adjust: %+v", f)
	}

	// A real table is subtracted.
	f = base
	applyProbe(&f, modelsource.ProbeResult{Probed: true, StreamedBytes: 27 * gib})
	if want := modelsource.EstimateVRAM(41 * gib); f.VRAMEstGB != want {
		t.Errorf("VRAMEstGB = %.2f, want %.2f", f.VRAMEstGB, want)
	}

	// A nonsensical measurement larger than the file must not produce a
	// negative estimate.
	f = base
	applyProbe(&f, modelsource.ProbeResult{Probed: true, StreamedBytes: 999 * gib})
	if f.VRAMEstGB != base.VRAMEstGB {
		t.Errorf("oversized measurement changed the estimate to %.2f", f.VRAMEstGB)
	}
}
