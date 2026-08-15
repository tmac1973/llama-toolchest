package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// A pre-upgrade "all" on a box with an iGPU must migrate to the
// discrete-set spanning value: "all" no longer matches any dropdown
// option there, and at runtime it would place layers on an iGPU the
// default build has no kernels for.
func TestMigrateGPUAssignAllOnIGPUBox(t *testing.T) {
	// GPU 1 of 3 is the iGPU → spanning option value is "0,2".
	opts := models.GPUAssignOptions(3, []bool{false, true, false})
	cfg := &models.ModelConfig{GPUAssign: "all", SplitMode: "layer"}

	migrateGPUAssign(cfg, opts, 3)

	if cfg.GPUAssign != "0,2" {
		t.Fatalf("GPUAssign = %q, want the discrete set 0,2", cfg.GPUAssign)
	}
	// Split fields re-resolved so emission agrees with the new value.
	if cfg.TensorSplit != "1,0,1" || cfg.SplitMode != "layer" {
		t.Errorf("split fields not re-resolved: %q/%q", cfg.TensorSplit, cfg.SplitMode)
	}
	// And the migrated value must actually match a dropdown option.
	found := false
	for _, o := range opts {
		if o.Value == cfg.GPUAssign {
			found = true
		}
	}
	if !found {
		t.Error("migrated value matches no dropdown option")
	}
}

// On an iGPU-free box "all" is still a real option — migration must not
// touch it.
func TestMigrateGPUAssignAllUnchangedWithoutIGPU(t *testing.T) {
	opts := models.GPUAssignOptions(4, nil)
	cfg := &models.ModelConfig{GPUAssign: "all"}
	migrateGPUAssign(cfg, opts, 4)
	if cfg.GPUAssign != "all" || cfg.TensorSplit != "" {
		t.Errorf("no-iGPU box: config should be untouched, got %q/%q", cfg.GPUAssign, cfg.TensorSplit)
	}
}

// The original legacy migration (pre-dropdown configs) must keep working
// through the extracted helper.
func TestMigrateGPUAssignLegacy(t *testing.T) {
	opts := models.GPUAssignOptions(4, nil)

	tensor := &models.ModelConfig{SplitMode: "tensor", TensorSplit: "1,1,0,0"}
	migrateGPUAssign(tensor, opts, 4)
	if tensor.GPUAssign != "tensor-2" {
		t.Errorf("legacy tensor config: got %q, want tensor-2", tensor.GPUAssign)
	}

	custom := &models.ModelConfig{TensorSplit: "3,1,0,0"}
	migrateGPUAssign(custom, opts, 4)
	if custom.GPUAssign != "custom" {
		t.Errorf("legacy split config: got %q, want custom", custom.GPUAssign)
	}

	fresh := &models.ModelConfig{}
	migrateGPUAssign(fresh, opts, 4)
	if fresh.GPUAssign != "" {
		t.Errorf("fresh config should stay unset (template defaults it), got %q", fresh.GPUAssign)
	}
}
