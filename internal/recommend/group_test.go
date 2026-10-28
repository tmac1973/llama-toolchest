package recommend

import (
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// groupRepos puts every quant of one base model on one card. The base
// model's own author comes first, then the trusted publishers in their
// listed order, then everyone else by downloads. The card's numbers come
// from the first repo, except downloads and likes, which add up, and the
// release date, which is the earliest known upload.
func TestGroupReposOrdersAndSums(t *testing.T) {
	day := func(d int) time.Time { return time.Date(2026, 3, d, 0, 0, 0, 0, time.UTC) }
	likes := func(n int) func(*huggingface.ListedModel) {
		return func(r *huggingface.ListedModel) { r.Likes = n }
	}
	repos := []huggingface.ListedModel{
		listed("someone/Model-7B-GGUF", "llama", 7e9, 5000, base("org/Model-7B"), created(day(10)), likes(1)),
		listed("bartowski/Model-7B-GGUF", "llama", 7e9, 100, base("org/Model-7B"), created(day(3)), likes(2)),
		listed("unsloth/Model-7B-GGUF", "llama", 7e9, 50, base("org/Model-7B"), created(time.Time{}), likes(3)),
		listed("ORG/Model-7B-GGUF", "llama", 7.1e9, 10, base("org/Model-7B"), created(day(5)), likes(4)),
		listed("other/Model-7B-GGUF", "llama", 7e9, 9000, base("org/Model-7B"), created(day(20)), likes(5)),
		listed("someone/Plain-3B-GGUF", "mamba", 3e9, 70),
	}
	repos[3].GGUF.ContextLength = 65536
	repos[3].Gated = huggingface.Gated(true)

	groups := groupRepos(repos, Profile{})
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	g := groups[0]
	var ids []string
	for _, r := range g.Repos {
		ids = append(ids, r.ID)
	}
	want := []string{"ORG/Model-7B-GGUF", "unsloth/Model-7B-GGUF", "bartowski/Model-7B-GGUF", "other/Model-7B-GGUF", "someone/Model-7B-GGUF"}
	if !slices.Equal(ids, want) {
		t.Errorf("repo order = %v, want %v", ids, want)
	}
	if g.Repo().ID != want[0] || !slices.Equal(g.Alts(), want[1:]) {
		t.Errorf("Repo = %s, Alts = %v", g.Repo().ID, g.Alts())
	}
	if g.Key != "org/model-7b" || g.BaseRepo != "org/Model-7B" || g.Name != "Model-7B" {
		t.Errorf("key/base/name = %q/%q/%q", g.Key, g.BaseRepo, g.Name)
	}
	if g.Params != 7.1e9 || g.TrainedCtx != 65536 || !g.Gated || g.Arch != "llama" {
		t.Errorf("first repo's numbers not used: params %d, ctx %d, gated %v, arch %q", g.Params, g.TrainedCtx, g.Gated, g.Arch)
	}
	if g.Downloads != 5000+100+50+10+9000 || g.Likes != 15 {
		t.Errorf("downloads %d, likes %d", g.Downloads, g.Likes)
	}
	if !g.Released.Equal(day(3)) {
		t.Errorf("Released = %v, want the earliest known upload %v", g.Released, day(3))
	}
	if !g.paramsFromHub || g.unsupported {
		t.Errorf("paramsFromHub %v, unsupported %v", g.paramsFromHub, g.unsupported)
	}

	plain := groups[1]
	if plain.Key != "plain-3b" || plain.BaseRepo != "" || plain.Name != "Plain-3B" || len(plain.Repos) != 1 {
		t.Errorf("repo without a base model = %+v", plain)
	}
}

// Without a base model, repos are ordered by trusted publisher rank and
// then by downloads. Groups keep the order their first repo appeared in.
func TestGroupReposWithoutBaseAuthor(t *testing.T) {
	repos := []huggingface.ListedModel{
		listed("zzz/Second-GGUF", "llama", 1e9, 1),
		listed("someone/First-GGUF", "llama", 1e9, 900),
		listed("bartowski/First-GGUF", "llama", 1e9, 5),
		listed("unsloth/First-GGUF", "llama", 1e9, 1),
		listed("another/First-GGUF", "llama", 1e9, 1000),
	}
	groups := groupRepos(repos, Profile{})
	if len(groups) != 2 || groups[0].Key != "second" || groups[1].Key != "first" {
		t.Fatalf("groups = %v", groups)
	}
	var ids []string
	for _, r := range groups[1].Repos {
		ids = append(ids, r.ID)
	}
	if want := []string{"unsloth/First-GGUF", "bartowski/First-GGUF", "another/First-GGUF", "someone/First-GGUF"}; !slices.Equal(ids, want) {
		t.Errorf("repo order = %v, want %v", ids, want)
	}
	if groups[0].Alts() != nil {
		t.Errorf("a single repo has alternatives: %v", groups[0].Alts())
	}
}

// A group whose architecture the active build cannot load is marked
// unsupported, but only when the build's list is known. An image reader
// (clip) summary is never judged and its size is not the model's.
func TestGroupReposMarksUnsupported(t *testing.T) {
	repos := []huggingface.ListedModel{
		listed("a/Llama-GGUF", "llama", 8e9, 1),
		listed("a/Mamba-GGUF", "mamba", 8e9, 1),
		listed("a/Vision-GGUF", clipArch, 4e8, 1),
		listed("a/NoSize-GGUF", "llama", 0, 1),
	}
	known := Profile{ArchsKnown: true, Archs: map[string]bool{"llama": true}}
	byKey := map[string]*Group{}
	for _, g := range groupRepos(repos, known) {
		byKey[g.Key] = g
	}
	if byKey["llama"].unsupported || !byKey["mamba"].unsupported || byKey["vision"].unsupported {
		t.Errorf("unsupported: llama %v, mamba %v, vision %v",
			byKey["llama"].unsupported, byKey["mamba"].unsupported, byKey["vision"].unsupported)
	}
	if byKey["vision"].paramsFromHub || byKey["nosize"].paramsFromHub || !byKey["llama"].paramsFromHub {
		t.Errorf("paramsFromHub: vision %v, nosize %v, llama %v",
			byKey["vision"].paramsFromHub, byKey["nosize"].paramsFromHub, byKey["llama"].paramsFromHub)
	}

	for _, g := range groupRepos(repos, Profile{}) {
		if g.unsupported {
			t.Errorf("%s unsupported although the build's list is not known", g.Key)
		}
	}
	if got := groupRepos(nil, known); len(got) != 0 {
		t.Errorf("no repos = %v", got)
	}
}

// covParamsFor returns the parameter count whose typical 4-bit quant
// takes share of the given GPU budget.
func covParamsFor(share, vramGiB float64) int64 {
	return int64(share * vramGiB * (1 << 30) * 8 / typicalBPW)
}

// covGroup is a group with a known size and downloads, by an untrusted
// publisher.
func covGroup(key string, params int64, downloads int) *Group {
	return &Group{Key: key, Params: params, paramsFromHub: true, Downloads: downloads,
		Repos: []huggingface.ListedModel{{ID: "someone/" + key, Author: "someone"}}}
}

// covKeys lists the keys of groups.
func covKeys(gs []*Group) []string {
	var out []string
	for _, g := range gs {
		out = append(out, g.Key)
	}
	return out
}

// Places are reserved by size, so popular small models cannot take every
// finalist place: the best ten large models get in even with fewer
// downloads than fifty small ones. The list stops at finalistCount and
// is ordered by the coarse score.
func TestChooseFinalistsReservesPlacesBySize(t *testing.T) {
	hw := machine(24).Hardware
	vram, _ := models.FitBudgets(hw)
	if vram <= 0 {
		t.Fatalf("test machine has no GPU budget")
	}
	var groups []*Group
	for i := range 50 {
		groups = append(groups, covGroup(fmt.Sprintf("small-%02d", i), covParamsFor(0.1, vram), 10000+i))
	}
	for i := range 15 {
		groups = append(groups, covGroup(fmt.Sprintf("large-%02d", i), covParamsFor(1.0, vram), i))
	}

	out := chooseFinalists(groups, hw)
	if len(out) != finalistCount {
		t.Fatalf("%d finalists, want %d", len(out), finalistCount)
	}
	var large, small []string
	for _, g := range out {
		if strings.HasPrefix(g.Key, "large") {
			large = append(large, g.Key)
		} else {
			small = append(small, g.Key)
		}
	}
	if len(large) != bucketPlaces[bucketLarge] {
		t.Errorf("%d large finalists, want %d: %v", len(large), bucketPlaces[bucketLarge], large)
	}
	// The large ones that get in are the ones with the most downloads.
	for _, k := range large {
		var n int
		fmt.Sscanf(k, "large-%d", &n)
		if n < 5 {
			t.Errorf("%s got in ahead of a large model with more downloads", k)
		}
	}
	// The small ones that get in are the 30 most downloaded.
	if small[0] != "small-49" || small[len(small)-1] != "small-20" || len(small) != 30 {
		t.Errorf("small finalists = %v", small)
	}
	for i := 1; i < len(out); i++ {
		if out[i].coarse > out[i-1].coarse {
			t.Errorf("finalists not ordered by coarse score at %d: %v", i, covKeys(out))
			break
		}
	}
}

// A model too large even at the smallest quant, with all GPU and system
// memory, is dropped, but only when its size came from the hub summary;
// a model of unknown size is kept for the full check. With nothing left
// the result is nil.
func TestChooseFinalistsDropsModelsTooLarge(t *testing.T) {
	hw := machine(24).Hardware
	vram, ram := models.FitBudgets(hw)
	tooLarge := int64((vram + ram) * 1.1 * (1 << 30) * 8 / smallestBPW)
	fits := int64((vram + ram) * 0.9 * (1 << 30) * 8 / smallestBPW)

	unknown := covGroup("unknown-size", tooLarge, 1)
	unknown.paramsFromHub = false
	out := chooseFinalists([]*Group{
		covGroup("too-large", tooLarge, 1000),
		covGroup("just-fits", fits, 1),
		unknown,
	}, hw)
	got := covKeys(out)
	slices.Sort(got)
	if want := []string{"just-fits", "unknown-size"}; !slices.Equal(got, want) {
		t.Errorf("finalists = %v, want %v", got, want)
	}

	if out := chooseFinalists([]*Group{covGroup("too-large", tooLarge, 1)}, hw); out != nil {
		t.Errorf("only a too-large model = %v, want nil", covKeys(out))
	}
	if out := chooseFinalists(nil, hw); out != nil {
		t.Errorf("no groups = %v, want nil", out)
	}
}

// The coarse score favours downloads, then recent release, then trusted
// publishers, then likes. With the other numbers equal, a trusted
// publisher ranks first, and a newer model ranks above an older one.
func TestChooseFinalistsCoarseOrder(t *testing.T) {
	hw := machine(24).Hardware
	vram, _ := models.FitBudgets(hw)
	p := covParamsFor(0.5, vram)
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	plain := covGroup("plain", p, 100)
	trusted := covGroup("trusted", p, 100)
	trusted.Repos = append(trusted.Repos, huggingface.ListedModel{ID: "bartowski/trusted", Author: "bartowski"})
	popular := covGroup("popular", p, 1000)
	for _, g := range []*Group{plain, trusted, popular} {
		g.Released = t0
	}
	newer := covGroup("newer", p, 100)
	newer.Released = t0.AddDate(0, 6, 0)

	got := covKeys(chooseFinalists([]*Group{plain, trusted, newer, popular}, hw))
	if want := []string{"popular", "newer", "trusted", "plain"}; !slices.Equal(got, want) {
		t.Errorf("order = %v, want %v", got, want)
	}
}

// Without a GPU every model counts as medium sized, and the list still
// stops at finalistCount.
func TestChooseFinalistsWithoutGPU(t *testing.T) {
	hw := machine().Hardware
	if vram, _ := models.FitBudgets(hw); vram != 0 {
		t.Fatalf("machine without GPUs has a GPU budget of %v", vram)
	}
	var groups []*Group
	for i := range 50 {
		groups = append(groups, covGroup(fmt.Sprintf("m-%02d", i), 1e9, i))
	}
	out := chooseFinalists(groups, hw)
	if len(out) != finalistCount {
		t.Fatalf("%d finalists, want %d", len(out), finalistCount)
	}
	if out[0].Key != "m-49" || out[len(out)-1].Key != "m-10" {
		t.Errorf("finalists = %v, want the 40 most downloaded", covKeys(out))
	}
}
