package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeMoEGGUF builds a GGUF v3 file with the given uint32 metadata keys
// (architecture "moearch") and a tensor-info block of equal-sized
// tensors, each tensorBytes long, in the order given.
func writeMoEGGUF(t *testing.T, path string, keys map[string]uint32, tensors []string, tensorBytes int) string {
	t.Helper()
	var buf bytes.Buffer
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(&buf, binary.LittleEndian, uint64(1+len(keys)))
	writeStr("general.architecture")
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	writeStr("moearch")
	for k, v := range keys {
		writeStr(k)
		binary.Write(&buf, binary.LittleEndian, ggufTypeUint32)
		binary.Write(&buf, binary.LittleEndian, v)
	}
	for i, name := range tensors {
		writeStr(name)
		binary.Write(&buf, binary.LittleEndian, uint32(1))
		binary.Write(&buf, binary.LittleEndian, uint64(tensorBytes/4))
		binary.Write(&buf, binary.LittleEndian, uint32(0)) // F32
		binary.Write(&buf, binary.LittleEndian, uint64(i*tensorBytes))
	}
	for buf.Len()%32 != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, len(tensors)*tensorBytes))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// A model whose first layer is dense and whose layers 1-3 carry experts:
// the counts come from the metadata, the byte total and the layer span
// from the expert tensors only (shared experts and the router excluded).
func TestParseMoELayout(t *testing.T) {
	path := writeMoEGGUF(t, filepath.Join(t.TempDir(), "moe.gguf"), map[string]uint32{
		"moearch.block_count":       4,
		"moearch.expert_count":      128,
		"moearch.expert_used_count": 8,
	}, []string{
		"token_embd.weight",
		"blk.0.ffn_up.weight", // dense layer
		"blk.1.ffn_gate_inp.weight", "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_gate_shexp.weight",
		"blk.2.ffn_up_exps.weight", "blk.2.ffn_down_exps.weight",
		"blk.3.ffn_gate_up_exps.weight", "blk.3.ffn_down_exps.weight",
		"output.weight",
	}, 64)

	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ExpertCount != 128 || meta.ExpertUsedCount != 8 {
		t.Errorf("expert counts = %d/%d, want 128/8", meta.ExpertCount, meta.ExpertUsedCount)
	}
	if meta.ExpertBytes != 6*64 {
		t.Errorf("ExpertBytes = %d, want %d (six expert tensors)", meta.ExpertBytes, 6*64)
	}
	if meta.ExpertLayerFirst != 1 || meta.ExpertLayers != 3 {
		t.Errorf("expert layers = first %d, count %d; want 1, 3", meta.ExpertLayerFirst, meta.ExpertLayers)
	}

	var m Model
	meta.ApplyTo(&m)
	if m.ExpertCount != 128 || m.ExpertBytes != 6*64 || m.ExpertLayers != 3 {
		t.Errorf("ApplyTo did not copy the MoE layout: %+v", m)
	}
}

// Expert tensors of a split model are spread over every shard, so they
// are summed across all of them.
func TestParseMoELayoutAcrossShards(t *testing.T) {
	dir := t.TempDir()
	first := writeMoEGGUF(t, filepath.Join(dir, "big-00001-of-00002.gguf"), map[string]uint32{
		"moearch.block_count": 2, "moearch.expert_count": 64,
	}, []string{"token_embd.weight", "blk.0.ffn_up_exps.weight"}, 32)
	writeMoEGGUF(t, filepath.Join(dir, "big-00002-of-00002.gguf"), nil,
		[]string{"blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight", "output.weight"}, 32)

	meta, err := ParseGGUFMeta(first)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ExpertBytes != 3*32 || meta.ExpertLayerFirst != 0 || meta.ExpertLayers != 2 {
		t.Errorf("split MoE = %d bytes, first %d, layers %d; want 96, 0, 2",
			meta.ExpertBytes, meta.ExpertLayerFirst, meta.ExpertLayers)
	}
}

func TestApplyToRecordsBuiltInMTPOnlyForModels(t *testing.T) {
	var m Model
	(&GGUFMeta{NextNPredictLayers: 1, HasBlockTensors: true, HasTrunkBlock0: true}).ApplyTo(&m)
	if m.NextNLayers != 1 {
		t.Errorf("main model with a NextN layer: NextNLayers = %d, want 1", m.NextNLayers)
	}
	var head Model
	(&GGUFMeta{NextNPredictLayers: 1, HasBlockTensors: true, HasTrunkBlock0: false}).ApplyTo(&head)
	if head.NextNLayers != 0 {
		t.Errorf("standalone MTP head: NextNLayers = %d, want 0", head.NextNLayers)
	}
}

func TestCPUMoEFlagEmission(t *testing.T) {
	var b strings.Builder
	writeConfigParams(&b, &ModelConfig{CPUMoE: 10, GPULayers: 999}, false, "")
	if !strings.Contains(b.String(), "n-cpu-moe = 10\n") {
		t.Errorf("preset INI lacks n-cpu-moe:\n%s", b.String())
	}
	if flags := (&ModelConfig{CPUMoE: 10}).EffectiveFlagsFor(false, ""); !strings.Contains(flags, "--n-cpu-moe 10") {
		t.Errorf("effective flags lack --n-cpu-moe: %s", flags)
	}
	b.Reset()
	writeConfigParams(&b, &ModelConfig{GPULayers: 999}, false, "")
	if strings.Contains(b.String(), "n-cpu-moe") {
		t.Error("n-cpu-moe emitted for 0")
	}
}

// Moving expert layers off the GPU lowers the estimate by the expert
// weights of the layers that actually carry experts, and reports them as
// system memory. Layers before the first expert layer move nothing.
func TestVRAMWithCPUMoE(t *testing.T) {
	gib := int64(1 << 30)
	m := &Model{NLayers: 10, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 8192,
		SizeBytes: 20 * gib, ExpertBytes: 16 * gib, ExpertLayerFirst: 2, ExpertLayers: 8}
	base := &ModelConfig{GPULayers: 999, ContextSize: 4096}
	full := VRAMBreakdownForConfigOn(m, base, 1)

	cases := []struct {
		cpuMoE   int
		movedGiB float64
	}{
		{2, 0}, // only dense layers 0-1: nothing moves
		{4, 4}, // layers 2-3 carry experts: 2 × 2 GiB
		{10, 16},
		{50, 16}, // beyond the layer count: every expert layer
	}
	for _, c := range cases {
		cfg := *base
		cfg.CPUMoE = c.cpuMoE
		got := VRAMBreakdownForConfigOn(m, &cfg, 1)
		if d := full.Weights - got.Weights; d < c.movedGiB-0.01 || d > c.movedGiB+0.01 {
			t.Errorf("cpu_moe %d: weights dropped by %.2f GiB, want %.2f", c.cpuMoE, d, c.movedGiB)
		}
		if got.CPURAM < c.movedGiB-0.01 || got.CPURAM > c.movedGiB+0.01 {
			t.Errorf("cpu_moe %d: CPURAM = %.2f GiB, want %.2f", c.cpuMoE, got.CPURAM, c.movedGiB)
		}
	}
}

// Fewer GPU layers than the model has leaves the rest in system memory;
// zero is read as "not set", as it always was.
func TestVRAMWithPartialOffload(t *testing.T) {
	gib := int64(1 << 30)
	m := &Model{NLayers: 10, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 8192, SizeBytes: 10 * gib}
	full := VRAMBreakdownForConfigOn(m, &ModelConfig{GPULayers: 999, ContextSize: 4096}, 1)
	half := VRAMBreakdownForConfigOn(m, &ModelConfig{GPULayers: 5, ContextSize: 4096}, 1)
	if d := full.Weights - half.Weights; d < 4.99 || d > 5.01 {
		t.Errorf("half the layers on the GPU dropped the weights by %.2f GiB, want 5", d)
	}
	zero := VRAMBreakdownForConfigOn(m, &ModelConfig{ContextSize: 4096}, 1)
	if zero.Weights != full.Weights || zero.CPURAM != 0 {
		t.Errorf("gpu_layers 0 changed the estimate: %+v", zero)
	}
}
