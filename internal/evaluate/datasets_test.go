package evaluate

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------- helpers ----------

// swapDataset replaces the table entry for name for the duration of the
// test (restored on cleanup) so EnsureDataset can be pointed at a test
// server without touching the pinned URLs.
func swapDataset(t *testing.T, name string, mutate func(*Dataset)) {
	t.Helper()
	for i := range pinnedDatasets {
		if pinnedDatasets[i].Name == name {
			orig := pinnedDatasets[i]
			mutate(&pinnedDatasets[i])
			t.Cleanup(func() { pinnedDatasets[i] = orig })
			return
		}
	}
	t.Fatalf("dataset %q not in table", name)
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// serveFile returns an httptest server that hands out content for every
// request and counts the requests.
func serveFile(t *testing.T, content []byte) (*httptest.Server, *int) {
	t.Helper()
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Write(content)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

// ---------- pinned table ----------

func TestPinnedTableNamesMatchModes(t *testing.T) {
	// The engine's Mode.DatasetName() and the table share the names —
	// every mode's dataset must be pinned, and the pinned names must be
	// the distinct mode names.
	modeNames := map[string]bool{}
	for _, m := range []Mode{ModePerplexity, ModeKLDiv, ModeHellaSwag, ModeWinogrande} {
		modeNames[m.DatasetName()] = true
	}
	tableNames := map[string]bool{}
	for _, ds := range Datasets() {
		if !modeNames[ds.Name] {
			t.Errorf("table dataset %q is not a mode dataset name", ds.Name)
		}
		tableNames[ds.Name] = true
		if ds.URL == "" || ds.File == "" || ds.License == "" || ds.ApproxSize <= 0 {
			t.Errorf("dataset %q: URL/File/License/ApproxSize must all be set (license is rendered from the table)", ds.Name)
		}
		if len(ds.SHA256) != 64 {
			t.Errorf("dataset %q: SHA256 %q is not a 64-char hex string", ds.Name, ds.SHA256)
		}
	}
	for name := range modeNames {
		if !tableNames[name] {
			t.Errorf("mode dataset %q missing from the pinned table", name)
		}
	}
}

// The table is the single source for the license strings: lookups return
// the same entry.
func TestDatasetLookup(t *testing.T) {
	ds, ok := LookupDataset("hellaswag")
	if !ok {
		t.Fatal("hellaswag not in table")
	}
	if ds.License == "" || ds.URL == "" {
		t.Errorf("hellaswag entry incomplete: %+v", ds)
	}
	if _, ok := LookupDataset("nope"); ok {
		t.Error("unknown name must not resolve")
	}
}

// ---------- layout ----------

func TestLayoutPaths(t *testing.T) {
	if got := EvalDataRoot("/data"); got != filepath.Join("/data", "eval-data") {
		t.Errorf("EvalDataRoot = %q", got)
	}
	if got := DatasetsDir("/r"); got != filepath.Join("/r", "datasets") {
		t.Errorf("DatasetsDir = %q", got)
	}
	if got := LogitsDir("/r"); got != filepath.Join("/r", "logits") {
		t.Errorf("LogitsDir = %q", got)
	}
}

// ---------- EnsureDataset ----------

func TestEnsureDatasetDownloadAndCache(t *testing.T) {
	content := []byte("hello hellaswag\nsecond line\n")
	srv, hits := serveFile(t, content)

	swapDataset(t, "hellaswag", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = sha256Hex(content)
		d.ExtractFile = ""
	})

	root := t.TempDir()
	path, err := EnsureDataset(context.Background(), root, "hellaswag")
	if err != nil {
		t.Fatal(err)
	}
	if *hits != 1 {
		t.Fatalf("server hits after first call = %d, want 1", *hits)
	}
	wantPath := filepath.Join(root, "datasets", "hellaswag_val_full.txt")
	if path != wantPath {
		t.Fatalf("path = %q, want %q", path, wantPath)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, content) {
		t.Error("stored content does not match the served content")
	}

	// Second call: cached short-circuit — no server hit, same path.
	path2, err := EnsureDataset(context.Background(), root, "hellaswag")
	if err != nil {
		t.Fatal(err)
	}
	if path2 != path {
		t.Errorf("second path = %q, want %q", path2, path)
	}
	if *hits != 1 {
		t.Fatalf("server hits after cached call = %d, want 1", *hits)
	}
}

func TestEnsureDatasetHashMismatch(t *testing.T) {
	// Server serves content whose hash does not match the pinned one.
	served := []byte("corrupted body")
	srv, hits := serveFile(t, served)
	pinned := sha256Hex([]byte("the bytes the file should have held"))

	swapDataset(t, "hellaswag", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = pinned
		d.ExtractFile = ""
	})

	root := t.TempDir()
	_, err := EnsureDataset(context.Background(), root, "hellaswag")
	if err == nil {
		t.Fatal("want a hash-mismatch error, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, "expected") || !strings.Contains(msg, "got") {
		t.Fatalf("error must name expected vs got, got %q", msg)
	}
	if !strings.Contains(msg, pinned) || !strings.Contains(msg, sha256Hex(served)) {
		t.Fatalf("error must carry both hashes, got %q", msg)
	}
	if *hits != 1 {
		t.Fatalf("hash mismatch must not be retried: hits = %d, want 1", *hits)
	}

	// Nothing left behind: no final file, no temp file.
	entries, err := os.ReadDir(filepath.Join(root, "datasets"))
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	for _, e := range entries {
		t.Fatalf("leftover file after hash mismatch: %q", e.Name())
	}
}

func TestEnsureDatasetZipExtraction(t *testing.T) {
	member := []byte("wiki.test.raw contents")
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create("wikitext-2-raw/wiki.test.raw")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(member); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	artifact := buf.Bytes()

	srv, hits := serveFile(t, artifact)
	swapDataset(t, "wikitext-2", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = sha256Hex(artifact)
		d.ExtractFile = "wikitext-2-raw/wiki.test.raw"
	})

	root := t.TempDir()
	path, err := EnsureDataset(context.Background(), root, "wikitext-2")
	if err != nil {
		t.Fatal(err)
	}
	if *hits != 1 {
		t.Fatalf("hits = %d, want 1", *hits)
	}
	wantPath := filepath.Join(root, "datasets", "wikitext-2-raw-test.txt")
	if path != wantPath {
		t.Fatalf("path = %q, want %q", path, wantPath)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, member) {
		t.Error("stored file must be the extracted member, not the archive")
	}

	// Only the dataset file may remain — no zip, no temp files.
	entries, err := os.ReadDir(filepath.Join(root, "datasets"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "wikitext-2-raw-test.txt" {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("datasets dir = %v, want exactly [wikitext-2-raw-test.txt]", names)
	}
}

func TestEnsureDatasetMissingZipMember(t *testing.T) {
	// A valid zip (correct hash) that lacks the pinned member.
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	if _, err := zw.Create("wikitext-2-raw/wiki.valid.raw"); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	artifact := buf.Bytes()
	srv, _ := serveFile(t, artifact)

	swapDataset(t, "wikitext-2", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = sha256Hex(artifact)
		d.ExtractFile = "wikitext-2-raw/wiki.test.raw"
	})

	root := t.TempDir()
	if _, err := EnsureDataset(context.Background(), root, "wikitext-2"); err == nil {
		t.Fatal("want an error for a missing zip member")
	}
	entries, _ := os.ReadDir(filepath.Join(root, "datasets"))
	for _, e := range entries {
		t.Fatalf("leftover file after extraction failure: %q", e.Name())
	}
}

func TestEnsureDatasetTransientThenSuccess(t *testing.T) {
	// First request 404s (the raw.githubusercontent.com transient), the
	// second serves the right bytes.
	content := []byte("retry content")
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.NotFound(w, r)
			return
		}
		w.Write(content)
	}))
	t.Cleanup(srv.Close)

	swapDataset(t, "winogrande", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = sha256Hex(content)
		d.ExtractFile = ""
	})

	root := t.TempDir()
	path, err := EnsureDataset(context.Background(), root, "winogrande")
	if err != nil {
		t.Fatalf("transient 404 should be retried, got %v", err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestEnsureDatasetUnknownName(t *testing.T) {
	if _, err := EnsureDataset(context.Background(), t.TempDir(), "nope"); err == nil {
		t.Fatal("want an error for an unknown dataset name")
	}
}

// ---------- Verify ----------

func TestVerify(t *testing.T) {
	content := []byte("verified content")
	srv, _ := serveFile(t, content)
	swapDataset(t, "hellaswag", func(d *Dataset) {
		d.URL = srv.URL
		d.SHA256 = sha256Hex(content)
		d.ExtractFile = ""
	})

	root := t.TempDir()
	path, err := EnsureDataset(context.Background(), root, "hellaswag")
	if err != nil {
		t.Fatal(err)
	}

	status := statusByName(t, Verify(root), "hellaswag")
	if !status.Present || !status.Verified {
		t.Errorf("freshly downloaded dataset: present=%v verified=%v, want true/true", status.Present, status.Verified)
	}
	if status.Size != int64(len(content)) {
		t.Errorf("size = %d, want %d", status.Size, len(content))
	}
	if status.License == "" {
		t.Error("status must carry the license from the table")
	}

	status = statusByName(t, Verify(root), "wikitext-2")
	if status.Present || status.Verified {
		t.Errorf("absent dataset: present=%v verified=%v, want false/false", status.Present, status.Verified)
	}

	// Corrupt the file: still present, no longer verified.
	if err := os.WriteFile(path, []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	status = statusByName(t, Verify(root), "hellaswag")
	if !status.Present || status.Verified {
		t.Errorf("corrupted dataset: present=%v verified=%v, want true/false", status.Present, status.Verified)
	}
}

func statusByName(t *testing.T, statuses []DatasetStatus, name string) DatasetStatus {
	t.Helper()
	for _, st := range statuses {
		if st.Name == name {
			return st
		}
	}
	t.Fatalf("dataset %q missing from Verify result", name)
	return DatasetStatus{}
}
