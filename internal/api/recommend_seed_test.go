package api

import (
	"bytes"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/presets"
)

// seedGGUF writes a header-only GGUF describing a model, where the
// downloader would have put it.
func seedGGUF(t *testing.T, s *Server, repo, file string, keys map[string]uint32) {
	t.Helper()
	var b bytes.Buffer
	ws := func(v string) {
		binary.Write(&b, binary.LittleEndian, uint64(len(v)))
		b.WriteString(v)
	}
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0))
	binary.Write(&b, binary.LittleEndian, uint64(1+len(keys)))
	ws("general.architecture")
	binary.Write(&b, binary.LittleEndian, uint32(8))
	ws("llama")
	for k, v := range keys {
		ws("llama." + k)
		binary.Write(&b, binary.LittleEndian, uint32(4))
		binary.Write(&b, binary.LittleEndian, v)
	}
	dir := filepath.Join(s.cfg.ModelsPath(), huggingface.SafeModelID(repo))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

var denseKeys = map[string]uint32{
	"block_count": 36, "embedding_length": 4096, "attention.head_count": 32,
	"attention.head_count_kv": 8, "context_length": 131072,
}

func seedServer(t *testing.T, gpuGiB ...int) *Server {
	t.Helper()
	s := newHelperServer(t)
	hw := models.Hardware{LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	for i, g := range gpuGiB {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: i, Name: "GPU", VRAMTotalMiB: g * 1024})
	}
	s.testHardware = &hw
	// A finished download fetches sampling presets in the background;
	// here they come from a stub that has none.
	stub := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(stub.Close)
	s.presets = presets.NewFetcher(t.TempDir(), "")
	s.presets.HFBase, s.presets.DocsBase = stub.URL, stub.URL
	return s
}

// download finishes a download of repo/file the way the downloader does,
// with the context class the file panel passed, if any.
func download(t *testing.T, s *Server, repo, file string, size int64, class models.ContextClass) *models.Model {
	t.Helper()
	seedGGUF(t, s, repo, file, denseKeys)
	id := "dl-" + file
	if class != "" {
		s.rememberSeed(id, class)
	}
	s.onDownloadComplete(modelsource.SourceHuggingFace, id, repo, file, size)
	modelID := huggingface.SafeModelID(repo) + "--" + huggingface.SafeFileID(file)
	// Wait for the background preset fetch, so it is not still writing
	// the registry when the test reads the model or removes its folder.
	// Asked under the registry's lock: the model itself is shared.
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if !slices.Contains(s.registry.ListNeedingPresetFetch(), modelID) {
			break
		}
	}
	m, err := s.registry.Get(modelID)
	if err != nil {
		t.Fatalf("not registered: %v", err)
	}
	return m
}

func TestADownloadFromARecommendationIsSeeded(t *testing.T) {
	s := seedServer(t, 24)
	m := download(t, s, "org/Model-GGUF", "m-Q4_K_M.gguf", 5<<30, models.ContextLong)

	cfg, _ := s.registry.GetConfig(m.ID)
	if cfg.ContextSize != 131072 {
		t.Errorf("context %d, want the 128K the card showed", cfg.ContextSize)
	}
	if m.Seeded == nil || m.Seeded.Requested != 131072 || len(m.Seeded.Notes) == 0 {
		t.Fatalf("mark = %+v", m.Seeded)
	}

	data, err := s.configPanelData(m.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(data.SeededText, "downloaded from the recommendations (128K context)") ||
		!strings.Contains(data.SeededTip, "Why each setting was chosen") {
		t.Errorf("panel note = %q / %q", data.SeededText, data.SeededTip)
	}
}

func TestAPlainDownloadIsNotSeeded(t *testing.T) {
	s := seedServer(t, 24)
	m := download(t, s, "org/Model-GGUF", "m-Q4_K_M.gguf", 5<<30, "")
	cfg, _ := s.registry.GetConfig(m.ID)
	if m.Seeded != nil || !models.ProfileEqual(*cfg, models.DefaultConfig()) {
		t.Errorf("a plain download was changed: %+v, %+v", m.Seeded, cfg)
	}
}

// Asked for more than the machine holds now: seeded with what fits, and
// the note says so.
func TestASeedThatFallsShort(t *testing.T) {
	s := seedServer(t, 8)
	m := download(t, s, "org/Model-GGUF", "m-Q4_K_M.gguf", 5<<30, models.ContextLong)
	cfg, _ := s.registry.GetConfig(m.ID)
	if m.Seeded == nil || cfg.ContextSize >= 131072 || cfg.ContextSize != m.Seeded.Context {
		t.Fatalf("context %d, mark %+v", cfg.ContextSize, m.Seeded)
	}
	text, _ := seededNote(m.Seeded)
	if !strings.Contains(text, "did not fit any more") {
		t.Errorf("note = %q", text)
	}
}

func TestNotSeededWhenNothingFits(t *testing.T) {
	s := seedServer(t, 2)
	s.testHardware.RAMTotalMiB = 4 * 1024
	m := download(t, s, "org/Model-GGUF", "m-Q8_0.gguf", 60<<30, models.ContextMedium)
	if m.Seeded != nil {
		t.Error("a model that does not fit was seeded")
	}
}

// The helper model has settings of its own.
func TestTheHelperIsNotSeeded(t *testing.T) {
	s := seedServer(t, 24)
	s.cfg.PendingHelper = "org/Helper-GGUF|h-Q4_K_M.gguf"
	m := download(t, s, "org/Helper-GGUF", "h-Q4_K_M.gguf", 3<<30, models.ContextLong)
	if m.Seeded != nil {
		t.Error("the helper was seeded")
	}
}

// The first change anyone makes clears the mark; the panel's own autosave
// response no longer shows the note.
func TestAChangeClearsTheMark(t *testing.T) {
	s := seedServer(t, 24)
	m := download(t, s, "org/Model-GGUF", "m-Q4_K_M.gguf", 5<<30, models.ContextMedium)
	if m.Seeded == nil {
		t.Fatal("not seeded")
	}
	form := url.Values{"enabled": {"true"}, "gpu_layers": {"999"}, "context_size": {"16384"}, "threads": {"8"}}
	code, body := s.doProfileRequest(t, "PUT", "/api/models/"+m.ID+"/config", form)
	if code != 200 {
		t.Fatalf("PUT = %d: %s", code, body)
	}
	if strings.Contains(body, "downloaded from the recommendations") {
		t.Error("the autosave response still shows the note")
	}
	if got, _ := s.registry.Get(m.ID); got.Seeded != nil {
		t.Error("a config change did not clear the mark")
	}
}

func TestSeedsAreTakenOnce(t *testing.T) {
	s := &Server{}
	s.rememberSeed("a", models.ContextLong)
	if c, ok := s.takeSeed("a"); !ok || c != models.ContextLong {
		t.Errorf("take = %v, %v", c, ok)
	}
	if _, ok := s.takeSeed("a"); ok {
		t.Error("a seed was taken twice")
	}
}
