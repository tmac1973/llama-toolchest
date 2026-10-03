package api

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/recommend"
)

// recommendShown is how many cards the feed shows before "Show N more":
// enough to choose from without pushing the search box off the page.
const recommendShown = 8

// feedButton is one of the feed's category or context buttons.
type feedButton struct {
	Value, Label, Tip string
	Pressed           bool
}

// recommendFeedView is everything the feed template shows. Every
// sentence is built here, so the template only lays them out.
type recommendFeedView struct {
	Profile, ProfileTip string
	Age                 string
	Stale               string
	Unavailable         string
	Intent, Ctx         string
	Intents, Contexts   []feedButton
	Cards, More         []recommendCardView
	Hidden              string
	Empty               string
	Unverified          []recommend.Unverified
}

// recommendCardView is one model's card.
type recommendCardView struct {
	ID                string // DOM id suffix
	Name, NameURL     string
	Publisher, Repo   string
	RepoURL           string
	Gated             bool
	Vision            string // tooltip; "" when the model reads no images
	Quant, Size, BPW  string
	Placement         string
	PlacementTip      string
	Slow              bool
	Context           string
	KV, KVTip         string
	Memory, MemoryTip string
	Fewest            string
	Next              string
	DetailsURL        string
}

var intentButtons = []feedButton{
	{Value: string(recommend.IntentQuality), Label: "Best quality",
		Tip: "Larger models first, counting the quant: a much larger model at 4 bits usually beats a smaller one at 8 bits. How widely a model is used counts a little too."},
	{Value: string(recommend.IntentFastest), Label: "Fastest",
		Tip: "Models that read the least data for each word they write come first. Mixture-of-experts models read only part of their weights, so they are often fast for their size."},
	{Value: string(recommend.IntentContext), Label: "Longest context",
		Tip: "Models that can hold the most text on this machine come first."},
	{Value: string(recommend.IntentNewest), Label: "Newest",
		Tip: "Models first published most recently come first."},
}

var contextButtons = []feedButton{
	{Value: "8k", Label: "8K"},
	{Value: "32k", Label: "32K"},
	{Value: "128k", Label: "128K"},
	{Value: "max", Label: "Model maximum", Tip: "The most each model was trained for, which differs from model to model."},
}

// contextTip explains the context selector.
const contextTip = "How much text the model can keep in mind at once: the conversation, any files, and its own answers. More context needs more memory, so a smaller quant may be suggested."

// nextClass is the context size after class, for a card's "At 128K" line.
var nextClass = map[models.ContextClass]models.ContextClass{
	models.ContextShort: models.ContextMedium, models.ContextMedium: models.ContextLong, models.ContextLong: models.ContextMax,
}

func (s *Server) recommendFeed(pool *recommend.Pool, intent recommend.Intent, ctxName string, class models.ContextClass, now time.Time) recommendFeedView {
	v := recommendFeedView{Intent: string(intent), Ctx: ctxName}
	for _, b := range intentButtons {
		b.Pressed = b.Value == string(intent)
		v.Intents = append(v.Intents, b)
	}
	for _, b := range contextButtons {
		b.Pressed = b.Value == ctxName
		if b.Tip == "" {
			b.Tip = contextTip
		} else {
			b.Tip = contextTip + " " + b.Tip
		}
		v.Contexts = append(v.Contexts, b)
	}
	if pool == nil {
		v.Unavailable = "The list is still being built. Try again in a moment."
		return v
	}
	v.Profile, v.ProfileTip = profileLine(pool.Profile)
	v.Age = "updated " + agoText(now.Sub(pool.Built))
	if pool.Stale(now) {
		v.Stale = fmt.Sprintf("This list is from %s. Refresh to check for newer models.", agoText(now.Sub(pool.Built)))
	}
	if pool.Unavailable != "" {
		v.Unavailable = pool.Unavailable
		return v
	}
	v.Unverified = pool.Unverified

	for i, g := range pool.Order(intent, class) {
		c := newRecommendCardView(g, class, ctxName, i)
		if i < recommendShown {
			v.Cards = append(v.Cards, c)
		} else {
			v.More = append(v.More, c)
		}
	}
	label := contextLabel(ctxName)
	if n := pool.Hidden(class); n > 0 && len(v.Cards) > 0 {
		v.Hidden = fmt.Sprintf("%d more %s only with a shorter context.", n, plural(n, "model fits", "models fit"))
	}
	if len(v.Cards) == 0 {
		v.Empty = fmt.Sprintf("Nothing among the models looked at runs well on this machine at %s. Try a shorter context.", label)
	}
	return v
}

func newRecommendCardView(g *recommend.Group, class models.ContextClass, ctxName string, i int) recommendCardView {
	p := g.Picks[class]
	repo := g.Repo()
	c := recommendCardView{
		ID:        fmt.Sprintf("%d", i),
		Name:      g.Name,
		Publisher: repo.Author,
		Repo:      repo.ID,
		RepoURL:   "https://huggingface.co/" + repo.ID,
		Gated:     g.Gated,
		Quant:     p.Quant,
		Size:      fmt.Sprintf("%.1f GiB", models.BytesToGiB(p.SizeBytes)),
		BPW:       fmt.Sprintf("%.1f bits per weight", p.BPW),
		Context:   tokensLabel(p.Context) + " context",
	}
	c.NameURL = c.RepoURL
	if g.BaseRepo != "" {
		c.NameURL = "https://huggingface.co/" + g.BaseRepo
	}
	if g.Vision {
		c.Vision = fmt.Sprintf("Can read images. The image reader needs about %.1f GiB more GPU memory when it is turned on.", models.BytesToGiB(g.MMProjBytes))
	}

	if p.Placement == models.PlacementExperts {
		c.Placement = "Experts in system memory"
		c.PlacementTip = fmt.Sprintf("Some of the model's expert weights (%.1f GiB) are kept in system memory so it fits. It runs, but generation is slower than with everything on the GPU.", p.CPURAMGiB)
		c.Slow = true
	} else {
		c.Placement = "All on GPU"
		c.PlacementTip = "Every layer runs on the GPU, the fastest way to run it."
	}
	if p.KVQuant == "q8_0" {
		c.KV = "8-bit KV cache"
	} else {
		c.KV = "full-precision KV cache"
	}
	c.KVTip = "The KV cache holds the conversation. 8-bit uses half the memory of full precision, with a very small effect on quality."
	c.Memory = fmt.Sprintf("%.1f GiB of %.1f GiB", p.EstimateGiB, p.BudgetGiB)
	c.MemoryTip = "Estimated GPU memory use against what is available after a safety margin on each card. Real use can differ by a few percent."
	if p.Placement == models.PlacementGPU && p.FewestGPUs < p.Cards {
		c.Fewest = fmt.Sprintf("fits on %d of %d GPUs", p.FewestGPUs, p.Cards)
	}

	if nc, ok := nextClass[class]; ok {
		next := g.Picks[nc]
		target := contextClassLabel(nc, g.TrainedCtx)
		switch {
		case nc == models.ContextMax && (g.TrainedCtx <= p.Context || target == ""):
			// The model holds no more than this anyway.
		case next == nil:
			c.Next = "At " + target + ": does not run well on this machine"
		case next.File == p.File && next.KVQuant == p.KVQuant && next.Placement == p.Placement:
			c.Next = "At " + target + ": the same"
		default:
			parts := []string{next.Quant}
			if next.KVQuant == "q8_0" {
				parts = append(parts, "8-bit KV cache")
			}
			if next.Placement == models.PlacementExperts {
				parts = append(parts, "experts in system memory")
			}
			c.Next = "At " + target + ": " + strings.Join(parts, ", ")
		}
	}

	q := url.Values{}
	q.Set("id", repo.ID)
	q.Set("source", "hf")
	q.Set("suggest", p.File)
	q.Set("ctx", ctxName)
	if alts := g.Alts(); len(alts) > 0 {
		q.Set("alts", strings.Join(alts, ","))
	}
	c.DetailsURL = "/api/hf/model?" + q.Encode()
	return c
}

// contextClassLabel names a class for one model: "128K", or the model's
// own limit for "max".
func contextClassLabel(class models.ContextClass, trained int) string {
	if class == models.ContextMax {
		if trained <= 0 {
			return ""
		}
		return tokensLabel(trained)
	}
	return tokensLabel(models.ContextClassTokens[class])
}

func contextLabel(ctxName string) string {
	for _, b := range contextButtons {
		if b.Value == ctxName {
			if ctxName == "max" {
				return "each model's maximum context"
			}
			return b.Label
		}
	}
	return ctxName
}

// profileLine names the machine a list was worked out for, e.g.
// "3× RTX A4000 · 48 GB VRAM · 64 GB RAM · build b6500".
func profileLine(p recommend.Profile) (line, tip string) {
	cards := models.PlanCards(p.Hardware)
	var parts []string
	other := 0.0
	if len(cards) > 0 {
		var vram float64
		names := map[string]int{}
		var order []string
		for _, c := range cards {
			vram += float64(c.VRAMTotalMiB) / 1024
			other += float64(c.OtherUsedMiB) / 1024
			if names[c.Name] == 0 {
				order = append(order, c.Name)
			}
			names[c.Name]++
		}
		var gpus []string
		for _, n := range order {
			if names[n] > 1 {
				gpus = append(gpus, fmt.Sprintf("%d× %s", names[n], n))
			} else {
				gpus = append(gpus, n)
			}
		}
		parts = append(parts, strings.Join(gpus, " + "), fmt.Sprintf("%.0f GB VRAM", vram))
		if other >= 0.5 {
			parts = append(parts, fmt.Sprintf("%.1f GB in use by other programs", other))
		}
	}
	if p.Hardware.RAMTotalMiB > 0 {
		parts = append(parts, fmt.Sprintf("%.0f GB RAM", float64(p.Hardware.RAMTotalMiB)/1024))
	}
	if p.BuildID != "" {
		parts = append(parts, "build "+p.BuildID)
	}
	tip = "What these recommendations were worked out for. If something looks wrong, check this first."
	if other >= 0.5 {
		tip += " GPU memory other programs are using (the desktop, another AI tool) is left to them, so a model that would fit on an empty card may be planned with less context or offered a smaller quant. The list is worked out again when that use changes by a gigabyte or more."
	}
	if !p.ArchsKnown {
		tip += " The model types this llama.cpp build can load are not known, so models are not checked against it."
	}
	return strings.Join(parts, " · "), tip
}

func agoText(d time.Duration) string {
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		n := int(d / time.Minute)
		return fmt.Sprintf("%d %s ago", n, plural(n, "minute", "minutes"))
	}
	n := int(d / time.Hour)
	return fmt.Sprintf("%d %s ago", n, plural(n, "hour", "hours"))
}
