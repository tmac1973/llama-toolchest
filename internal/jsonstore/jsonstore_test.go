package jsonstore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type file struct {
	SchemaVersion int      `json:"schema_version"`
	Items         []string `json:"items"`
}

// Save writes a 0600 file in a folder it creates, and Load reads it back.
func TestSaveAndLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config", "store.json")
	if err := Save(path, "test store", file{SchemaVersion: 1, Items: []string{"a"}}); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(path); err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("stat: %v, mode %v; want 0600", err, fi.Mode().Perm())
	}
	var got file
	if err := Load(path, 1, &got); err != nil || len(got.Items) != 1 || got.Items[0] != "a" {
		t.Fatalf("Load = %+v, %v", got, err)
	}
}

// A missing file leaves the value as it is: the empty store.
func TestLoadMissingFile(t *testing.T) {
	got := file{SchemaVersion: 1}
	if err := Load(filepath.Join(t.TempDir(), "none.json"), 1, &got); err != nil || got.SchemaVersion != 1 {
		t.Fatalf("Load of a missing file = %+v, %v", got, err)
	}
}

// Another schema version or a broken file is an error, and v is not
// filled from it.
func TestLoadRefusesOtherVersionsAndBrokenFiles(t *testing.T) {
	dir := t.TempDir()
	newer := filepath.Join(dir, "newer.json")
	os.WriteFile(newer, []byte(`{"schema_version": 2, "items": ["x"]}`), 0o600)
	var got file
	if err := Load(newer, 1, &got); err == nil || !strings.Contains(err.Error(), "schema version 2") {
		t.Errorf("newer file: %v, want a schema version error", err)
	}
	if len(got.Items) != 0 {
		t.Errorf("newer file filled v: %+v", got)
	}
	broken := filepath.Join(dir, "broken.json")
	os.WriteFile(broken, []byte(`{"schema_version": 1, `), 0o600)
	if err := Load(broken, 1, &got); err == nil {
		t.Error("broken file: no error")
	}
}
