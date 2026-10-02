package recommend

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// --- a small market -------------------------------------------------------

type ggufSummary = struct {
	Total         int64  `json:"total"`
	Architecture  string `json:"architecture"`
	ContextLength int    `json:"context_length"`
}

func listed(id string, arch string, params int64, downloads int, opts ...func(*huggingface.ListedModel)) huggingface.ListedModel {
	r := huggingface.ListedModel{
		ID: id, Downloads: downloads, PipelineTag: "text-generation", SHA: "sha-" + id,
		CreatedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		GGUF:      &ggufSummary{Total: params, Architecture: arch, ContextLength: 131072},
	}
	if i := indexByte(id, '/'); i > 0 {
		r.Author = id[:i]
	}
	for _, o := range opts {
		o(&r)
	}
	return r
}

func indexByte(s string, c byte) int {
	for i := range len(s) {
		if s[i] == c {
			return i
		}
	}
	return -1
}

func base(b string) func(*huggingface.ListedModel) {
	return func(r *huggingface.ListedModel) { r.CardData.BaseModel = huggingface.BaseModels{b} }
}

func created(t time.Time) func(*huggingface.ListedModel) {
	return func(r *huggingface.ListedModel) { r.CreatedAt = t }
}

// quants are the files of a repo for a model of params parameters, one
// per bits-per-weight value given.
func quants(stem string, params int64, bpws ...float64) []modelsource.File {
	names := map[float64]string{16: "BF16", 8.5: "Q8_0", 6.56: "Q6_K", 5.69: "Q5_K_M", 4.85: "Q4_K_M", 3.9: "Q3_K_M", 3.2: "IQ3_XXS", 2.6: "IQ2_M"}
	var out []modelsource.File
	for _, b := range bpws {
		q := names[b]
		out = append(out, modelsource.File{
			Filename: fmt.Sprintf("%s-%s.gguf", stem, q), Quant: q,
			Size: int64(float64(params) * b / 8), OID: fmt.Sprintf("%s-%v", stem, b),
		})
	}
	return out
}

func denseMeta(layers, embd, heads, kvHeads, ctx int) *models.GGUFMeta {
	return &models.GGUFMeta{
		Architecture: "llama", NLayers: layers, NEmbd: embd, NHead: heads, NKVHead: kvHeads,
		ContextLength: ctx, KVFullPerTok: layers * kvHeads * 256, AttnLayers: layers,
		VocabSize: 128000, MetaOnly: true,
	}
}

func moeMeta(layers, embd, experts, used, ff, ctx int) *models.GGUFMeta {
	m := denseMeta(layers, embd, 32, 4, ctx)
	m.Architecture = "qwen3moe"
	m.ExpertCount, m.ExpertUsedCount, m.ExpertFFLength = experts, used, ff
	return m
}

type fakeHub struct {
	mu     sync.Mutex
	list   []huggingface.ListedModel
	files  map[string][]modelsource.File
	meta   map[string]*models.GGUFMeta
	failAt map[string]error // repo → error from Files

	lists, fileCalls, metaCalls atomic.Int32
	listErr                     error
	block                       chan struct{} // when set, ListGGUF waits on it
}

func (h *fakeHub) ListGGUF(ctx context.Context, q huggingface.ListQuery) ([]huggingface.ListedModel, error) {
	h.lists.Add(1)
	if h.block != nil {
		<-h.block
	}
	if h.listErr != nil {
		return nil, h.listErr
	}
	var out []huggingface.ListedModel
	for _, r := range h.list {
		if q.Author == "" || q.Author == r.Author {
			out = append(out, r)
		}
	}
	return out, nil
}

func (h *fakeHub) Files(ctx context.Context, repo string) ([]modelsource.File, error) {
	h.fileCalls.Add(1)
	if err := h.failAt[repo]; err != nil {
		return nil, err
	}
	return h.files[repo], nil
}

func (h *fakeHub) Meta(ctx context.Context, repo string, f modelsource.File) (*models.GGUFMeta, error) {
	h.metaCalls.Add(1)
	m, ok := h.meta[repo]
	if !ok {
		return nil, errors.New("unreadable")
	}
	return m, nil
}

func machine(gpuGiB ...int) Profile {
	hw := models.Hardware{LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	for i, g := range gpuGiB {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: i, Name: "GPU", VRAMTotalMiB: g * 1024})
	}
	return Profile{Hardware: hw}
}

// market is a small HuggingFace: a dense 8B and 32B, a 30B-A3B MoE, a
// model three publishers quantize, and things that must not appear.
func market() *fakeHub {
	h := &fakeHub{files: map[string][]modelsource.File{}, meta: map[string]*models.GGUFMeta{}, failAt: map[string]error{}}
	add := func(r huggingface.ListedModel, files []modelsource.File, meta *models.GGUFMeta) {
		h.list = append(h.list, r)
		h.files[r.ID] = files
		if meta != nil {
			h.meta[r.ID] = meta
		}
	}
	add(listed("unsloth/Small-8B-GGUF", "llama", 8e9, 90000, base("org/Small-8B")),
		quants("Small-8B", 8e9, 16, 8.5, 4.85), denseMeta(36, 4096, 32, 8, 131072))
	add(listed("bartowski/Small-8B-GGUF", "llama", 8e9, 50000, base("org/Small-8B")),
		quants("Small-8B", 8e9, 8.5, 4.85), denseMeta(36, 4096, 32, 8, 131072))
	add(listed("org/Small-8B-GGUF", "llama", 8e9, 1000, base("org/Small-8B")),
		quants("Small-8B", 8e9, 8.5), denseMeta(36, 4096, 32, 8, 131072))
	add(listed("unsloth/Dense-32B-GGUF", "llama", 32e9, 80000, base("org/Dense-32B"),
		created(time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC))),
		quants("Dense-32B", 32e9, 8.5, 6.56, 5.69, 4.85, 3.9, 3.2, 2.6), denseMeta(64, 5120, 40, 8, 131072))
	add(listed("unsloth/Moe-30B-A3B-GGUF", "qwen3moe", 30e9, 70000, base("org/Moe-30B-A3B"),
		created(time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC))),
		quants("Moe-30B-A3B", 30e9, 8.5, 5.69, 4.85, 3.2), moeMeta(48, 2048, 128, 8, 768, 262144))
	add(listed("unsloth/Short-14B-GGUF", "llama", 14e9, 60000, base("org/Short-14B")),
		quants("Short-14B", 14e9, 8.5, 4.85), denseMeta(40, 5120, 40, 8, 32768))
	// Things that must not appear.
	emb := listed("ggml-org/Embed-GGUF", "modern-bert", 4e9, 9000)
	emb.PipelineTag = "feature-extraction"
	h.list = append(h.list, emb,
		listed("someone/Tiny-0.5B-GGUF", "llama", 5e8, 99000),
		listed("someone/Small-8B-draft-GGUF", "llama", 8e9, 99000),
		listed("someone/Finetune-8B-GGUF", "llama", 8e9, 10),
		listed("someone/Huge-1T-GGUF", "llama", 1e12, 99000),
	)
	return h
}

func build(t *testing.T, h *fakeHub, p Profile) *Pool {
	t.Helper()
	e := &Engine{Hub: func() Hub { return h }, Dir: t.TempDir()}
	pool := e.Get(context.Background(), p, false)
	if pool == nil || pool.Unavailable != "" {
		t.Fatalf("pool = %+v", pool)
	}
	return pool
}

func group(pool *Pool, name string) *Group {
	for _, g := range pool.Groups {
		if g.Name == name {
			return g
		}
	}
	return nil
}

// --- the tests --------------------------------------------------------------

func TestTheEngine(t *testing.T) {
	h := market()
	pool := build(t, h, machine(24, 24))

	names := map[string]bool{}
	for _, g := range pool.Groups {
		names[g.Name] = true
	}
	for _, want := range []string{"Small-8B", "Dense-32B", "Moe-30B-A3B", "Short-14B"} {
		if !names[want] {
			t.Errorf("%s missing; have %v", want, names)
		}
	}
	for _, unwanted := range []string{"Embed", "Tiny-0.5B", "Small-8B-draft", "Finetune-8B", "Huge-1T"} {
		if names[unwanted] {
			t.Errorf("%s listed", unwanted)
		}
	}
	// The huge model never reached a file listing.
	if h.fileCalls.Load() != 4 {
		t.Errorf("%d file listings, want 4 (one per finalist group)", h.fileCalls.Load())
	}

	// Three publishers, one card, the trusted publisher first and the
	// base model's own author ahead of it.
	small := group(pool, "Small-8B")
	if small.Repo().ID != "org/Small-8B-GGUF" {
		t.Errorf("Small-8B names %s, want the base author's own repo", small.Repo().ID)
	}
	if alts := small.Alts(); len(alts) != 2 || alts[0] != "unsloth/Small-8B-GGUF" {
		t.Errorf("alts = %v", alts)
	}
}

func TestQuantPicks(t *testing.T) {
	pool := build(t, market(), machine(24, 24))

	// A dense 32B on 48 GB at 32K: a good quant, every layer on the GPUs,
	// and not the full-precision file just because it fits.
	p := group(pool, "Dense-32B").Picks[models.ContextMedium]
	if p == nil || p.BPW < goodBPW || p.Placement != models.PlacementGPU {
		t.Fatalf("Dense-32B @32K = %+v", p)
	}
	if small := group(pool, "Small-8B").Picks[models.ContextMedium]; small.Quant != "Q8_0" {
		t.Errorf("Small-8B @32K picked %s, want Q8_0 over the full-precision file", small.Quant)
	}
	// More context, a smaller quant or a smaller KV cache.
	long := group(pool, "Dense-32B").Picks[models.ContextLong]
	if long == nil || (long.SizeBytes >= p.SizeBytes && long.KVQuant == p.KVQuant) {
		t.Errorf("Dense-32B @128K = %+v, want less than @32K (%+v)", long, p)
	}
	// Trained for 32K: no pick at 128K, and counted as hidden there.
	if group(pool, "Short-14B").Picks[models.ContextLong] != nil {
		t.Error("a 32K model has a 128K pick")
	}
	if pool.Hidden(models.ContextLong) < 1 {
		t.Error("the 32K model is not counted as hidden at 128K")
	}
	// Never below 3 bits per weight.
	for _, g := range pool.Groups {
		for c, pk := range g.Picks {
			if pk.BPW < smallestBPW {
				t.Errorf("%s @%s picked %.1f bpw", g.Name, c, pk.BPW)
			}
		}
	}
}

// On a card too small for it, a mixture-of-experts model keeps experts in
// system memory at a Q4–Q5 quant rather than dropping to 3 bits, and a
// dense model that does not fit is never suggested with layers on the CPU.
func TestPicksOnASmallCard(t *testing.T) {
	pool := build(t, market(), machine(16))

	moe := group(pool, "Moe-30B-A3B").Picks[models.ContextMedium]
	if moe == nil || moe.Placement != models.PlacementExperts || moe.BPW < goodBPW || moe.BPW >= offloadMaxBPW {
		t.Errorf("Moe-30B-A3B on 16 GB = %+v, want a Q4–Q5 quant with experts in system memory", moe)
	}
	for _, g := range pool.Groups {
		for c, pk := range g.Picks {
			if pk.Placement == models.PlacementPartial || pk.Placement == models.PlacementNone {
				t.Errorf("%s @%s placed %s", g.Name, c, pk.Placement)
			}
		}
	}
	if d := group(pool, "Dense-32B"); d != nil {
		if pk := d.Picks[models.ContextMedium]; pk != nil && pk.BPW >= goodBPW {
			t.Errorf("Dense-32B fits 16 GB at %.1f bpw?", pk.BPW)
		}
	}
}

func TestFewestGPUs(t *testing.T) {
	pool := build(t, market(), machine(24, 24, 24))
	p := group(pool, "Small-8B").Picks[models.ContextShort]
	if p.Cards != 3 || p.FewestGPUs != 1 {
		t.Errorf("Small-8B @8K: fewest %d of %d, want 1 of 3", p.FewestGPUs, p.Cards)
	}
}

func TestOrders(t *testing.T) {
	pool := build(t, market(), machine(16))
	class := models.ContextMedium

	newest := pool.Order(IntentNewest, class)
	for i := 1; i < len(newest); i++ {
		if newest[i].Released.After(newest[i-1].Released) {
			t.Errorf("newest = %v is not newest first", names(newest))
		}
	}
	// Every order holds the same models.
	n := len(pool.Order(IntentQuality, class))
	for _, in := range Intents {
		if got := len(pool.Order(in, class)); got != n || n == 0 {
			t.Errorf("%s has %d models, quality %d", in, got, n)
		}
	}
}

// Experts in system memory are read several times slower than from the
// GPU, so the more of them a plan moves there, the slower it is.
func TestReadCost(t *testing.T) {
	const g = 1 << 30
	dense := &Pick{SizeBytes: 8 * g}
	moe := &Pick{SizeBytes: 20 * g, ExpertBytes: 18 * g, ExpertUsedCount: 8, ExpertCount: 128}
	offloaded := *moe
	offloaded.CPURAMGiB = 6
	more := *moe
	more.CPURAMGiB = 12

	if c := readCost(dense); c != 8 {
		t.Errorf("dense read cost %.2f, want its size", c)
	}
	if !(readCost(moe) < readCost(&offloaded) && readCost(&offloaded) < readCost(&more)) {
		t.Errorf("read costs %.2f, %.2f, %.2f do not grow with experts in system memory",
			readCost(moe), readCost(&offloaded), readCost(&more))
	}
	// All on the GPU, an MoE using 8 of 128 experts reads its 2 GiB of
	// other weights and a sixteenth of the experts: 3.1 GiB of 20.
	if c := readCost(moe); c < 3.1 || c > 3.2 {
		t.Errorf("MoE read cost %.2f GiB", readCost(moe))
	}
}

func names(gs []*Group) []string {
	var out []string
	for _, g := range gs {
		out = append(out, g.Name)
	}
	return out
}

func TestUnverified(t *testing.T) {
	h := market()
	h.failAt["unsloth/Dense-32B-GGUF"] = errors.New("boom")
	delete(h.meta, "unsloth/Short-14B-GGUF")
	p := machine(24, 24)
	p.ArchsKnown, p.Archs = true, map[string]bool{"llama": true} // no qwen3moe

	pool := build(t, h, p)
	reasons := map[string]string{}
	for _, u := range pool.Unverified {
		reasons[u.Repo] = u.Reason
	}
	if reasons["unsloth/Dense-32B-GGUF"] != reasonFiles {
		t.Errorf("Dense-32B: %q", reasons["unsloth/Dense-32B-GGUF"])
	}
	if reasons["unsloth/Short-14B-GGUF"] != reasonMeta {
		t.Errorf("Short-14B: %q", reasons["unsloth/Short-14B-GGUF"])
	}
	if reasons["unsloth/Moe-30B-A3B-GGUF"] != reasonUnsupported("qwen3moe") {
		t.Errorf("Moe: %q", reasons["unsloth/Moe-30B-A3B-GGUF"])
	}
}

// HuggingFace's summary sometimes describes the image reader: the
// architecture is "clip" and the count is the reader's. The header and
// the largest file are believed instead.
func TestAWrongSummaryIsCorrected(t *testing.T) {
	h := market()
	r := listed("someone/Vision-12B-GGUF", clipArch, 4e8, 99000, base("org/Vision-12B"))
	files := append(quants("Vision-12B", 12e9, 8.5, 4.85), modelsource.File{Filename: "mmproj-F16.gguf", Size: 8e8, IsMMProj: true})
	h.list = append(h.list, r)
	h.files[r.ID] = files
	h.meta[r.ID] = denseMeta(48, 3840, 16, 8, 131072)

	pool := build(t, h, machine(24))
	g := group(pool, "Vision-12B")
	if g == nil {
		t.Fatal("the model was lost to its wrong summary")
	}
	if g.Arch != "llama" || g.Params < 11e9 || g.Params > 13e9 || !g.Vision {
		t.Errorf("arch %s, params %d, vision %v", g.Arch, g.Params, g.Vision)
	}
}

func TestTheFileListIsCachedPerRevision(t *testing.T) {
	h := market()
	dir := t.TempDir()
	e := &Engine{Hub: func() Hub { return h }, Dir: dir}
	e.Get(context.Background(), machine(24), false)
	first := h.fileCalls.Load()

	e2 := &Engine{Hub: func() Hub { return h }, Dir: dir}
	e2.Get(context.Background(), machine(24), true)
	if h.fileCalls.Load() != first {
		t.Errorf("a second build listed files again: %d → %d", first, h.fileCalls.Load())
	}

	// A new revision of a card's repo is listed afresh.
	h.list[2].SHA = "new"
	e2.Get(context.Background(), machine(24), true)
	if h.fileCalls.Load() != first+1 {
		t.Errorf("file listings %d, want %d", h.fileCalls.Load(), first+1)
	}
}

func TestThePool(t *testing.T) {
	h := market()
	now := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)
	e := &Engine{Hub: func() Hub { return h }, Now: func() time.Time { return now }}

	a := e.Get(context.Background(), machine(24), false)
	lists := h.lists.Load()
	if b := e.Get(context.Background(), machine(24), false); b != a || h.lists.Load() != lists {
		t.Error("the same machine rebuilt the pool")
	}
	if c := e.Get(context.Background(), machine(16), false); c == a {
		t.Error("a different machine reused the pool")
	}
	if a.Stale(now.Add(5*time.Hour)) || !a.Stale(now.Add(7*time.Hour)) {
		t.Error("staleness is not six hours")
	}
}

func TestOneBuildAtATime(t *testing.T) {
	h := market()
	h.block = make(chan struct{})
	e := &Engine{Hub: func() Hub { return h }}

	var wg sync.WaitGroup
	pools := make([]*Pool, 3)
	for i := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pools[i] = e.Get(context.Background(), machine(24), false)
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(h.block)
	wg.Wait()
	if pools[0] != pools[1] || pools[1] != pools[2] {
		t.Error("concurrent requests built separate pools")
	}
	if n := h.lists.Load(); n != int32(len(listQueries())) {
		t.Errorf("%d list queries, want %d (one build)", n, len(listQueries()))
	}
}

func TestUnavailable(t *testing.T) {
	e := &Engine{Hub: func() Hub { return market() }}
	if p := e.Get(context.Background(), machine(), false); p.Unavailable == "" {
		t.Error("no GPU, but a list")
	}
	h := market()
	h.listErr = errors.New("offline")
	e = &Engine{Hub: func() Hub { return h }}
	if p := e.Get(context.Background(), machine(24), false); p.Unavailable == "" {
		t.Error("every query failed, but a list")
	}
}

func TestJudgeCandidate(t *testing.T) {
	p := machine(24)
	trusted := listed("unsloth/New-8B-GGUF", "llama", 8e9, 3)
	if judgeCandidate(trusted, p) != candidateKeep {
		t.Error("a trusted publisher's new upload was dropped for its downloads")
	}
	mtp := listed("someone/Qwen-MTP-GGUF", "llama", 8e9, 5000)
	if judgeCandidate(mtp, p) != candidateKeep {
		t.Error("an -MTP- repo (a full model with built-in draft layers) was dropped")
	}
	head := listed("someone/Head-GGUF", "gemma4-assistant", 4e9, 5000)
	if judgeCandidate(head, p) != candidateDrop {
		t.Error("an MTP head was kept")
	}
	private := listed("someone/Private-GGUF", "llama", 8e9, 5000)
	private.Private = true
	if judgeCandidate(private, p) != candidateDrop {
		t.Error("a private repo was kept")
	}
}

func TestPercentiles(t *testing.T) {
	got := percentiles([]float64{10, 30, 20, 20})
	want := []float64{0, 1, 0.5, 0.5}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("percentiles = %v, want %v", got, want)
			break
		}
	}
	if one := percentiles([]float64{5}); one[0] != 1 {
		t.Errorf("a single value ranks %v", one[0])
	}
}

func TestParamsFromFile(t *testing.T) {
	if got := paramsFromFile(modelsource.File{Filename: "m-Q8_0.gguf", Quant: "Q8_0", Size: 8_500_000_000}); got != 8e9 {
		t.Errorf("Q8_0 = %d", got)
	}
	if got := paramsFromFile(modelsource.File{Filename: "m-weird.gguf", Quant: "unknown", Size: 1}); got != 0 {
		t.Errorf("unknown quant = %d", got)
	}
}

// A fine-tune names the model it was made from as its base, but it is a
// different model and gets its own card.
func TestGroupKey(t *testing.T) {
	for _, tt := range []struct{ id, base, want string }{
		{"bartowski/google_gemma-4-26B-A4B-it-GGUF", "google/gemma-4-26B-A4B-it", "google/gemma-4-26b-a4b-it"},
		{"unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen/Qwen3.6-35B-A3B", "qwen/qwen3.6-35b-a3b"},
		{"HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP", "google/gemma-4-26B-A4B-it", "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp"},
		{"FINAL-Bench/POCKET-26B-GGUF", "google/gemma-4-26B-A4B-it", "pocket-26b"},
		{"someone/Plain-7B-GGUF", "", "plain-7b"},
	} {
		r := listed(tt.id, "llama", 8e9, 100)
		if tt.base != "" {
			base(tt.base)(&r)
		}
		if got, _ := groupKey(r); got != tt.want {
			t.Errorf("groupKey(%s) = %q, want %q", tt.id, got, tt.want)
		}
	}
}
