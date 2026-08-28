package api

import (
	"bytes"
	"context"
	"html/template"
	"strings"
	"testing"

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

// The browse page's only new control is the source radio; both options
// must be present and HuggingFace preselected.
func TestBrowsePageHasSourceRadio(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/models_browse.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "content", nil); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	for _, want := range []string{`name="source" value="hf" checked`, `name="source" value="modelscope"`} {
		if !strings.Contains(out, want) {
			t.Errorf("browse page missing %q; output=\n%s", want, out)
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
