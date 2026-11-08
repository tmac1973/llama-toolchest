package api

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/backup"
	"github.com/tmac1973/llama-toolchest/internal/config"
)

// A run of uses of one model writes once; another model writes again,
// and a new load reads back the last one.
func TestDefaultModelStateWritesOnChange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "default-model.json")
	s := loadDefaultModelState(path)
	if got := s.get(); got.ID != "" {
		t.Fatalf("missing file: %+v, want empty", got)
	}
	t0 := time.Date(2026, 11, 8, 12, 0, 0, 0, time.UTC)
	s.note("qwen", false, t0)
	s.note("qwen", false, t0.Add(time.Minute))
	if s.writes != 1 {
		t.Errorf("two uses of one model wrote %d times, want 1", s.writes)
	}
	if got := s.get(); !got.UsedAt.Equal(t0.Add(time.Minute)) {
		t.Errorf("in memory: used at %v, want the later use", got.UsedAt)
	}
	s.note("ik-1", true, t0.Add(2*time.Minute))
	if s.writes != 2 {
		t.Errorf("another model wrote %d times in all, want 2", s.writes)
	}

	got := loadDefaultModelState(path).get()
	if got.ID != "ik-1" || !got.Fork || !got.UsedAt.Equal(t0.Add(2*time.Minute)) {
		t.Errorf("loaded %+v", got)
	}
}

func TestDefaultModelStateCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "default-model.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadDefaultModelState(path).get(); got.ID != "" {
		t.Errorf("corrupt file: %+v, want empty", got)
	}
}

// A write that fails keeps the value in memory.
func TestDefaultModelStateWriteFails(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	s := loadDefaultModelState(filepath.Join(blocker, "default-model.json"))
	s.note("qwen", false, time.Now())
	if got := s.get(); got.ID != "qwen" {
		t.Errorf("after a failed write: %+v, want qwen", got)
	}
}

// A restore sets the virtual model name's switch, trimmed name and pin,
// and lists each one it changed.
func TestRestoreDefaultModel(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	changed, err := s.restoreDeps().ApplySettings(backup.Settings{
		DefaultModelOff: ptr(true), DefaultModelName: ptr(" chat "), DefaultModelPin: ptr("qwen"),
	})
	if err != nil {
		t.Fatal(err)
	}
	want := config.DefaultModelConfig{Off: true, Name: "chat", Pin: "qwen"}
	if s.cfg.DefaultModel != want {
		t.Errorf("after restore: %+v, want %+v", s.cfg.DefaultModel, want)
	}
	if got := strings.Join(changed, ","); got != "default_model_off,default_model_name,default_model_pin" {
		t.Errorf("changed = %s", got)
	}
	changed, err = s.restoreDeps().ApplySettings(backup.Settings{DefaultModelName: ptr("chat")})
	if err != nil || len(changed) != 0 {
		t.Errorf("same name again: changed %v, err %v", changed, err)
	}
}
