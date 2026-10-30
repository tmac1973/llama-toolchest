package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ngramRecord is one record as common_ngram_cache_save writes it.
func ngramRecord(key [4]int32, counts map[int32]int32) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, key)
	binary.Write(&b, binary.LittleEndian, int32(len(counts)))
	for tok, c := range counts {
		binary.Write(&b, binary.LittleEndian, tok)
		binary.Write(&b, binary.LittleEndian, c)
	}
	return b.Bytes()
}

func TestCheckNgramCacheFile(t *testing.T) {
	good := append(ngramRecord([4]int32{1, 2, -1, -1}, map[int32]int32{5: 3}),
		ngramRecord([4]int32{7, 8, 9, -1}, map[int32]int32{10: 1, 11: 2})...)
	zero := ngramRecord([4]int32{1, 2, -1, -1}, map[int32]int32{5: 0})
	noTokens := ngramRecord([4]int32{1, 2, -1, -1}, map[int32]int32{})
	cases := []struct {
		name string
		data []byte
		ok   bool
	}{
		{"good", good, true},
		{"empty", nil, true},
		{"cut inside a record", good[:len(good)-3], false},
		{"cut inside the key", good[:10], false},
		{"a count of zero", zero, false},
		{"no following tokens", noTokens, false},
		{"random bytes", bytes.Repeat([]byte{0xff, 0x01, 0x7f}, 333), false},
	}
	dir := t.TempDir()
	for _, c := range cases {
		path := filepath.Join(dir, strings.ReplaceAll(c.name, " ", "-")+".bin")
		if err := os.WriteFile(path, c.data, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := CheckNgramCacheFile(path); (err == nil) != c.ok {
			t.Errorf("%s: err = %v, want ok %v", c.name, err, c.ok)
		}
	}
	if err := CheckNgramCacheFile(filepath.Join(dir, "missing.bin")); err != nil {
		t.Errorf("a missing file: %v", err)
	}
}

// A broken file is deleted so the model can load; a good one is kept,
// and the folder is made.
func TestPrepareNgramCacheFile(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cache", "ngram")
	good, broken := filepath.Join(dir, "good.bin"), filepath.Join(dir, "broken.bin")
	if err := PrepareNgramCacheFile(good); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("folder not made: %v", err)
	}
	rec := ngramRecord([4]int32{1, -1, -1, -1}, map[int32]int32{2: 1})
	os.WriteFile(good, rec, 0o644)
	os.WriteFile(broken, rec[:len(rec)-1], 0o644)
	for _, p := range []string{good, broken} {
		if err := PrepareNgramCacheFile(p); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := os.Stat(good); err != nil {
		t.Error("a good file was deleted")
	}
	if _, err := os.Stat(broken); err == nil {
		t.Error("a broken file was kept")
	}
}

// Only the user's preset names the file, only for N-gram Cache, only on
// builds that write it, and not when the extra flags name one already.
// Writing the preset also deletes a broken file.
func TestNgramCachePreset(t *testing.T) {
	r := benchRegistry(t)
	cfg, _ := r.GetConfig("a")
	cfg.SpecAssist = "ngram-cache"
	if err := r.SetConfig("a", cfg); err != nil {
		t.Fatal(err)
	}
	want := "lookup-cache-dynamic = " + filepath.Join(r.dataDir, "cache", "ngram", "a.bin")
	broken := filepath.Join(r.dataDir, "cache", "ngram", "a.bin")
	os.MkdirAll(filepath.Dir(broken), 0o755)
	os.WriteFile(broken, []byte{1, 2, 3}, 0o644)

	read := func(path string, err error) string {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	ini := read(r.WritePresetINI(Target{}))
	if !strings.Contains(sectionOf(t, ini, "a"), want) {
		t.Errorf("user preset lacks %q:\n%s", want, sectionOf(t, ini, "a"))
	}
	if strings.Contains(sectionOf(t, ini, "b"), "lookup-cache-dynamic") {
		t.Error("a model without N-gram Cache got a file")
	}
	if _, err := os.Stat(broken); err == nil {
		t.Error("writing the preset kept a broken file")
	}
	if ini := read(r.WriteBenchPresetINI(map[string]*ModelConfig{"b": cfg}, Target{})); strings.Contains(ini, "lookup-cache-dynamic") {
		t.Errorf("benchmark preset uses the file:\n%s", ini)
	}
	if ini := read(r.WritePresetINI(Target{Version: versionNgramCacheSave - 1})); strings.Contains(ini, "lookup-cache-dynamic") {
		t.Error("written for a build that does not save the file")
	}
	cfg.ExtraFlags = "--lookup-cache-dynamic /elsewhere.bin"
	r.SetConfig("a", cfg)
	if ini := read(r.WritePresetINI(Target{})); strings.Count(ini, "lookup-cache-dynamic") != 1 {
		t.Errorf("want only the extra flags' path:\n%s", sectionOf(t, ini, "a"))
	}
}
