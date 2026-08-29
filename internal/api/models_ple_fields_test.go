package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The models list is an explicit allow-list, so a field is invisible
// until it is named there. Both PLE fields drive things a client can
// see — the config selector's visibility and the VRAM estimate — and
// their absence once made a populated registry look empty.
func TestModelListExposesPLEFields(t *testing.T) {
	out := withPublicNames([]*models.Model{{
		ID: "m", PLEBytes: 1937768448, PLEChecked: true,
	}})
	if len(out) != 1 {
		t.Fatalf("got %d entries", len(out))
	}
	if got := out[0]["ple_bytes"]; got != int64(1937768448) {
		t.Errorf("ple_bytes = %v (%T), want 1937768448", got, got)
	}
	if got := out[0]["ple_checked"]; got != true {
		t.Errorf("ple_checked = %v, want true", got)
	}
}

// A model with no table must report zero rather than omitting the key,
// so "no table" and "not in this payload" stay distinguishable.
func TestModelListReportsAbsentPLETable(t *testing.T) {
	out := withPublicNames([]*models.Model{{ID: "m"}})
	if _, ok := out[0]["ple_bytes"]; !ok {
		t.Error("ple_bytes missing for a model without a table; callers cannot tell it from an old payload")
	}
	if out[0]["ple_checked"] != false {
		t.Errorf("ple_checked = %v, want false", out[0]["ple_checked"])
	}
}
