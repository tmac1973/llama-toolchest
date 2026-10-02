package api

import (
	"bytes"
	"context"
	"fmt"
	"html/template"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/recommend"
	"github.com/tmac1973/llama-toolchest/web"
)

func renderAny(t *testing.T, name string, data any) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, name, data); err != nil {
		t.Fatalf("execute %s: %v", name, err)
	}
	return buf.String()
}

// feedHub is a small HuggingFace for render tests: n dense models of
// growing size, a mixture-of-experts model too large for one 16 GB card,
// and a model trained for only 32K.
type feedHub struct {
	repos []huggingface.ListedModel
	files map[string][]modelsource.File
	meta  map[string]*models.GGUFMeta
}

func newFeedHub(n int) *feedHub {
	h := &feedHub{files: map[string][]modelsource.File{}, meta: map[string]*models.GGUFMeta{}}
	add := func(id string, params int64, meta *models.GGUFMeta, mmproj bool) {
		r := huggingface.ListedModel{ID: id, Author: "unsloth", Downloads: 1000 + int(params/1e9), PipelineTag: "text-generation",
			SHA: "s", CreatedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
		r.GGUF = &struct {
			Total         int64  `json:"total"`
			Architecture  string `json:"architecture"`
			ContextLength int    `json:"context_length"`
		}{params, meta.Architecture, meta.ContextLength}
		h.repos = append(h.repos, r)
		for _, q := range []struct {
			name string
			bpw  float64
		}{{"Q8_0", 8.5}, {"Q4_K_M", 4.85}} {
			h.files[id] = append(h.files[id], modelsource.File{Filename: "m-" + q.name + ".gguf", Quant: q.name, Size: int64(float64(params) * q.bpw / 8)})
		}
		if mmproj {
			h.files[id] = append(h.files[id], modelsource.File{Filename: "mmproj-F16.gguf", Size: 900 << 20, IsMMProj: true})
		}
		h.meta[id] = meta
	}
	dense := func(ctx int) *models.GGUFMeta {
		return &models.GGUFMeta{Architecture: "llama", NLayers: 32, NEmbd: 4096, NHead: 32, NKVHead: 8,
			ContextLength: ctx, KVFullPerTok: 32 * 8 * 256, AttnLayers: 32, VocabSize: 128000, MetaOnly: true}
	}
	for i := range n {
		add(fmt.Sprintf("unsloth/Dense-%dB-GGUF", 2+i), int64(2+i)*1e9, dense(131072), i == 0)
	}
	moe := dense(131072)
	moe.Architecture, moe.NEmbd, moe.ExpertCount, moe.ExpertUsedCount, moe.ExpertFFLength = "qwen3moe", 2048, 128, 8, 768
	moe.NLayers, moe.AttnLayers, moe.KVFullPerTok = 48, 48, 48*4*256
	add("unsloth/Moe-30B-A3B-GGUF", 30e9, moe, false)
	add("unsloth/Short-4B-GGUF", 4e9, dense(32768), false)
	return h
}

func (h *feedHub) ListGGUF(ctx context.Context, q huggingface.ListQuery) ([]huggingface.ListedModel, error) {
	return h.repos, nil
}
func (h *feedHub) Files(ctx context.Context, repo string) ([]modelsource.File, error) {
	return h.files[repo], nil
}
func (h *feedHub) Meta(ctx context.Context, repo string, f modelsource.File) (*models.GGUFMeta, error) {
	return h.meta[repo], nil
}

func feedProfile(gpuGiB ...int) recommend.Profile {
	hw := models.Hardware{LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	for i, g := range gpuGiB {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: g * 1024})
	}
	return recommend.Profile{Hardware: hw, BuildID: "b6500", ArchsKnown: true, Archs: map[string]bool{"llama": true, "qwen3moe": true}}
}

func renderFeed(t *testing.T, h recommend.Hub, p recommend.Profile, intent recommend.Intent, ctxName string, now time.Time) string {
	t.Helper()
	e := &recommend.Engine{Hub: func() recommend.Hub { return h }, Now: func() time.Time { return now }}
	pool := e.Get(context.Background(), p, false)
	v := (&Server{}).recommendFeed(pool, intent, ctxName, recommendClasses[ctxName], now)
	return renderAny(t, "recommend_feed", v)
}

var now = time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)

func TestTheFeedRenders(t *testing.T) {
	out := renderFeed(t, newFeedHub(3), feedProfile(16), recommend.IntentQuality, "32k", now)
	for _, want := range []string{
		"Recommended for this machine", "RTX A4000 · 16 GB VRAM · 64 GB RAM · build b6500", "updated just now",
		"Best quality", "Fastest", "Longest context", "Newest", "Model maximum",
		"Dense-4B", "from unsloth", "Q8_0", "bits per weight", "All on GPU", "32K context",
		"Moe-30B-A3B", "Experts in system memory", "rec-slow", "vision",
		"/api/hf/model?", "suggest=", "ctx=32k",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("feed lacks %q", want)
		}
	}
	// The pressed buttons are the ones asked for.
	if !regexp.MustCompile(`aria-pressed="true"[^>]*intent=quality&amp;ctx=32k[^>]*>\s*32K`).MatchString(out) &&
		!strings.Contains(out, `aria-pressed="true"`) {
		t.Error("no button is pressed")
	}
}

// Every figure on a card says how to read it.
func TestEveryCardFigureHasATooltip(t *testing.T) {
	out := renderFeed(t, newFeedHub(2), feedProfile(16), recommend.IntentQuality, "32k", now)
	card := out[strings.Index(out, `class="rec-card"`):]
	card = card[:strings.Index(card, "Details</button>")]
	for _, figure := range []string{"bits per weight", "32K context", "KV cache", "GiB of"} {
		re := regexp.MustCompile(`<span title="[^"]+"[^>]*>[^<]*` + regexp.QuoteMeta(figure))
		if !re.MatchString(card) {
			t.Errorf("%q has no tooltip in:\n%s", figure, card)
		}
	}
}

func TestFewestGPUsOnlyWhenFewer(t *testing.T) {
	three := renderFeed(t, newFeedHub(1), feedProfile(16, 16, 16), recommend.IntentQuality, "8k", now)
	if !strings.Contains(three, "fits on 1 of 3 GPUs") {
		t.Error("a small model on three cards does not say it fits on one")
	}
	one := renderFeed(t, newFeedHub(1), feedProfile(16), recommend.IntentQuality, "8k", now)
	if strings.Contains(one, "fits on 1 of 1") {
		t.Error("a single card claims to need fewer of itself")
	}
}

func TestTheFeedFoldsALongList(t *testing.T) {
	out := renderFeed(t, newFeedHub(10), feedProfile(24, 24), recommend.IntentQuality, "8k", now)
	shown := strings.Count(out[:strings.Index(out, "rec-more")], `class="rec-card"`)
	if shown != recommendShown {
		t.Errorf("%d cards before the fold, want %d", shown, recommendShown)
	}
	if !regexp.MustCompile(`Show \d+ more`).MatchString(out) {
		t.Error("no Show more")
	}
}

func TestTheFeedsOtherStates(t *testing.T) {
	// A model trained for 32K is hidden at 128K, and counted.
	long := renderFeed(t, newFeedHub(2), feedProfile(24), recommend.IntentQuality, "128k", now)
	if !strings.Contains(long, "1 more model fits only with a shorter context.") {
		t.Errorf("hidden count missing")
	}

	// Nothing fits: a 2 GB card.
	empty := renderFeed(t, newFeedHub(1), feedProfile(2), recommend.IntentQuality, "128k", now)
	if !strings.Contains(empty, "Try a shorter context.") {
		t.Error("no empty-state sentence")
	}

	// No GPU at all.
	none := renderFeed(t, newFeedHub(1), feedProfile(), recommend.IntentQuality, "32k", now)
	if !strings.Contains(none, "No recommendations right now. No GPU was detected") || strings.Contains(none, `class="rec-card"`) {
		t.Error("no-GPU state wrong")
	}

	// Seven hours old.
	e := &recommend.Engine{Hub: func() recommend.Hub { return newFeedHub(1) }, Now: func() time.Time { return now }}
	pool := e.Get(context.Background(), feedProfile(16), false)
	v := (&Server{}).recommendFeed(pool, recommend.IntentQuality, "32k", models.ContextMedium, now.Add(7*time.Hour))
	if !strings.Contains(renderAny(t, "recommend_feed", v), "This list is from 7 hours ago.") {
		t.Error("no stale notice")
	}

	// Architecture list unknown: said in the profile tooltip.
	p := feedProfile(16)
	p.ArchsKnown, p.Archs = false, nil
	if !strings.Contains(renderFeed(t, newFeedHub(1), p, recommend.IntentQuality, "32k", now), "are not known") {
		t.Error("unknown architectures not mentioned")
	}
}

func TestTheCollapsedFeed(t *testing.T) {
	out := renderAny(t, "recommend_collapsed", nil)
	if !strings.Contains(out, "Find recommended models") || !strings.Contains(out, "/api/hf/recommend?intent=quality&ctx=32k") {
		t.Errorf("collapsed = %s", out)
	}
}

// Opened from a card, the file panel marks the suggested file, passes the
// context on to its download, and lists the other publishers.
func TestDetailsOpenedFromACard(t *testing.T) {
	view := hfModelView{
		ID: "unsloth/M-GGUF", Source: "hf", AvailableBytes: 1 << 46,
		Suggest: "m-Q4_K_M.gguf", SuggestLabel: "Suggested for 32K", Ctx: "32k",
		Alts: []string{"bartowski/M-GGUF"},
		Files: []hfFileView{
			{ModelFile: modelsource.File{Filename: "m-Q8_0.gguf", Size: 8 << 30}, FitsOnDisk: true},
			{ModelFile: modelsource.File{Filename: "m-Q4_K_M.gguf", Size: 5 << 30}, FitsOnDisk: true},
		},
	}
	out := renderFiles(t, "hf_files", view)
	if strings.Count(out, "rec-suggested") != 1 || !strings.Contains(out, "Suggested for 32K") {
		t.Error("the suggested file is not marked once")
	}
	if strings.Count(out, `&#34;ctx&#34;:&#34;32k&#34;`)+strings.Count(out, `"ctx":"32k"`) != 2 {
		t.Errorf("the context is not passed to both downloads:\n%s", out)
	}
	if !strings.Contains(out, "Also published by") || !strings.Contains(out, "bartowski/M-GGUF") {
		t.Error("no other publishers")
	}

	// From a plain search: none of that.
	view.Suggest, view.Ctx, view.Alts = "", "", nil
	plain := renderFiles(t, "hf_files", view)
	if strings.Contains(plain, "rec-suggested") || strings.Contains(plain, "recommend") || strings.Contains(plain, "Also published") {
		t.Error("a plain search panel carries recommendation details")
	}
}
