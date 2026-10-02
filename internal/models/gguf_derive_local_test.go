//go:build local

package models

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestDeriveMatchesFullParse checks DerivedFor against full parses of real
// files: for every main model GGUF under $GGUF_DIR (laid out as the
// toolchest stores downloads, <org>--<repo>/<file>), it compares the
// derived tensor sizes and the VRAM estimate at 32K with the measured
// ones. The parameter count comes from HuggingFace, so it needs the
// network.
//
//	go test -tags local -run TestDeriveMatchesFullParse ./internal/models/
//
// Targets: the estimate within 3%, and never more than 0.5 GiB below the
// measured one (a low estimate is the harmful direction).
func TestDeriveMatchesFullParse(t *testing.T) {
	dir := os.Getenv("GGUF_DIR")
	if dir == "" {
		t.Skip("set GGUF_DIR to a folder of downloaded models")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*", "*.gguf"))
	sub, _ := filepath.Glob(filepath.Join(dir, "*", "*", "*.gguf"))
	files = append(files, sub...)

	params := map[string]int64{}
	for _, path := range files {
		name := filepath.Base(path)
		if IsMMProjFile(name) || strings.Contains(name, "-of-0") && !strings.Contains(name, "-00001-of-") {
			continue
		}
		full, err := ParseGGUFMeta(path)
		if err != nil || full.IsMTPHead() {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		only, err := ParseGGUFMetaOnly(f)
		f.Close()
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		repo := strings.Replace(filepath.Base(filepath.Dir(path)), "--", "/", 1)
		if _, ok := params[repo]; !ok {
			params[repo] = hfParamCount(repo)
		}
		size := totalSize(path)

		derived := only.DerivedFor(size, params[repo])
		mFull, mDerived := &Model{SizeBytes: size}, &Model{SizeBytes: size}
		full.ApplyTo(mFull)
		derived.ApplyTo(mDerived)

		cfg := &ModelConfig{ContextSize: 32768, GPULayers: 999, FlashAttention: true}
		eFull := VRAMEstimateForConfigOn(mFull, cfg, 1)
		eDerived := VRAMEstimateForConfigOn(mDerived, cfg, 1)
		line := fmt.Sprintf("%-50s est %6.2f / %6.2f GiB (%+5.1f%%)  embd %5d / %5d MiB  ple %5d / %5d MiB",
			name, eDerived, eFull, 100*(eDerived-eFull)/eFull,
			derived.TokenEmbdBytes>>20, full.TokenEmbdBytes>>20, derived.PLEBytes>>20, full.PLEBytes>>20)
		if full.ExpertCount > 0 {
			half := *cfg
			half.CPUMoE = full.ExpertLayerFirst + full.ExpertLayers/2
			hFull := VRAMEstimateForConfigOn(mFull, &half, 1)
			hDerived := VRAMEstimateForConfigOn(mDerived, &half, 1)
			line += fmt.Sprintf("\n%-50s experts %d+%d / %d+%d layers, %6d / %6d MiB; half in RAM: est %6.2f / %6.2f GiB (%+5.1f%%)",
				"", derived.ExpertLayerFirst, derived.ExpertLayers, full.ExpertLayerFirst, full.ExpertLayers,
				derived.ExpertBytes>>20, full.ExpertBytes>>20, hDerived, hFull, 100*(hDerived-hFull)/hFull)
			if derived.ExpertLayerFirst != full.ExpertLayerFirst || derived.ExpertLayers != full.ExpertLayers {
				t.Errorf("%s: expert layers %d+%d, measured %d+%d", name,
					derived.ExpertLayerFirst, derived.ExpertLayers, full.ExpertLayerFirst, full.ExpertLayers)
			}
			checkEstimate(t, name+" (half experts in RAM)", hDerived, hFull)
		}
		t.Log(line)
		checkEstimate(t, name, eDerived, eFull)
		if only.VocabSize != full.VocabSize || only.KVFullPerTok != full.KVFullPerTok || only.NLayers != full.NLayers {
			t.Errorf("%s: metadata differs from the full parse", name)
		}
	}
}

func checkEstimate(t *testing.T, name string, derived, full float64) {
	t.Helper()
	if d := (derived - full) / full; d > 0.03 || d < -0.03 || full-derived > 0.5 {
		t.Errorf("%s: derived estimate %.2f GiB, measured %.2f GiB", name, derived, full)
	}
}

func totalSize(path string) int64 {
	var n int64
	for _, s := range ExpandShards(path) {
		if fi, err := os.Stat(s); err == nil {
			n += fi.Size()
		}
	}
	if n == 0 {
		if fi, err := os.Stat(path); err == nil {
			n = fi.Size()
		}
	}
	return n
}

func hfParamCount(repo string) int64 {
	c := http.Client{Timeout: 20 * time.Second}
	resp, err := c.Get("https://huggingface.co/api/models/" + repo)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var info struct {
		GGUF struct {
			Total int64 `json:"total"`
		} `json:"gguf"`
	}
	json.NewDecoder(resp.Body).Decode(&info)
	return info.GGUF.Total
}
