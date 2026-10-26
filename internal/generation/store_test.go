package generation

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

func testSet(id string) ModelSet {
	return ModelSet{ID: id, Name: id, Kind: KindSD, Recipe: "ming-image",
		Components: []Component{{Slot: "diffusion_model", Path: "/m/dit.gguf"}}}
}

func TestStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.AddModelSet(testSet("ming-image")); err != nil {
		t.Fatal(err)
	}
	inst := Instance{ID: "ming", Name: "ming", ModelSetID: "ming-image", GPU: 1, Port: 8081,
		ReservedMiB: 19456, ExtraArgs: []string{"--offload-to-cpu"}, Enabled: true}
	if err := s.AddInstance(inst); err != nil {
		t.Fatal(err)
	}

	again, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := again.Instance("ming")
	if err != nil {
		t.Fatal(err)
	}
	if got.GPU != 1 || got.ReservedMiB != 19456 || !got.Enabled || got.ExtraArgs[0] != "--offload-to-cpu" {
		t.Errorf("reloaded instance = %+v, want the saved one", got)
	}
	if sets := again.ModelSets(); len(sets) != 1 || sets[0].Recipe != "ming-image" {
		t.Errorf("reloaded sets = %+v", sets)
	}
	if info, err := os.Stat(StorePath(dir)); err != nil || info.Mode().Perm() != 0o600 {
		t.Errorf("store file mode: %v %v, want 0600", info, err)
	}
}

// A store that is only read never creates a file, so installs that never
// use generation backends gain nothing on disk.
func TestStoreMissingFileIsEmptyAndNotCreated(t *testing.T) {
	dir := t.TempDir()
	s, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.ModelSets()) != 0 || len(s.Instances()) != 0 {
		t.Error("a missing file should load as an empty store")
	}
	if _, err := os.Stat(StorePath(dir)); !os.IsNotExist(err) {
		t.Errorf("Load created %s: %v", StorePath(dir), err)
	}
}

func TestStoreRejectsUnknownSchema(t *testing.T) {
	dir := t.TempDir()
	path := StorePath(dir)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"schema_version": 99}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Load(dir)
	if err == nil || !strings.Contains(err.Error(), "schema version 99") {
		t.Fatalf("Load: %v, want a schema version error", err)
	}
}

func TestStoreIDsAreUnique(t *testing.T) {
	s, _ := Load(t.TempDir())
	if err := s.AddModelSet(testSet("a")); err != nil {
		t.Fatal(err)
	}
	if err := s.AddModelSet(testSet("a")); !errors.Is(err, ErrIDTaken) {
		t.Errorf("second AddModelSet: %v, want ErrIDTaken", err)
	}
	if err := s.AddInstance(Instance{ID: "x", ModelSetID: "a"}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddInstance(Instance{ID: "x", ModelSetID: "a"}); !errors.Is(err, ErrIDTaken) {
		t.Errorf("second AddInstance: %v, want ErrIDTaken", err)
	}
	if err := s.AddInstance(Instance{ID: "y", ModelSetID: "missing"}); !errors.Is(err, ErrNotFound) {
		t.Errorf("AddInstance with an unknown set: %v, want ErrNotFound", err)
	}
}

func TestDeleteModelSetRefusedWhileUsed(t *testing.T) {
	s, _ := Load(t.TempDir())
	_ = s.AddModelSet(testSet("a"))
	_ = s.AddInstance(Instance{ID: "x", ModelSetID: "a"})

	if err := s.DeleteModelSet("a"); !errors.Is(err, ErrInUse) {
		t.Fatalf("DeleteModelSet: %v, want ErrInUse", err)
	}
	if err := s.DeleteInstance("x"); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteModelSet("a"); err != nil {
		t.Errorf("DeleteModelSet after the instance is gone: %v", err)
	}
}

func TestAssignPort(t *testing.T) {
	s, _ := Load(t.TempDir())
	_ = s.AddModelSet(testSet("a"))
	for _, p := range []int{8081, 8083} {
		_ = s.AddInstance(Instance{ID: string(rune('a' + p - 8081)), ModelSetID: "a", Port: p})
	}
	if p, err := s.AssignPort(8081, 4); err != nil || p != 8082 {
		t.Errorf("AssignPort = %d, %v; want 8082 (lowest free)", p, err)
	}
	if _, err := s.AssignPort(8081, 1); !errors.Is(err, ErrNoPort) {
		t.Errorf("AssignPort on a full range: %v, want ErrNoPort", err)
	}
}

func TestModelsDir(t *testing.T) {
	if got := ModelsDir(&config.Config{DataDir: "/data"}); got != "/data/models/gen-models" {
		t.Errorf("default = %q, want /data/models/gen-models", got)
	}
	if got := ModelsDir(&config.Config{DataDir: "/data", ModelsDir: "/mnt/llm/models/"}); got != "/mnt/llm/models/gen-models" {
		t.Errorf("with ModelsDir = %q, want /mnt/llm/models/gen-models", got)
	}
}
