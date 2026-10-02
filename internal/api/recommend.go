package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/recommend"
)

// recommendHub gives the recommendation engine HuggingFace through the
// server's own client and header cache, so a header read for the feed is
// one the Download Models table does not repeat.
type recommendHub struct {
	s      *Server
	client *huggingface.Client
}

func (h recommendHub) ListGGUF(ctx context.Context, q huggingface.ListQuery) ([]huggingface.ListedModel, error) {
	return h.client.ListGGUF(ctx, q)
}

// Files lists a repo's files with the same per-layer embedding probe the
// file table runs. Some models keep tens of GiB in such a table, which
// llama.cpp always holds in system memory; planned without it, a model
// looks that much larger on the GPU. Qwen3.8-Flash-Next's 26.8 GiB table
// was planned as GPU memory, and 46 layers of experts were moved to
// system memory to make room for it.
func (h recommendHub) Files(ctx context.Context, repo string) ([]modelsource.File, error) {
	d, err := h.client.GetModel(ctx, repo)
	if err != nil {
		return nil, err
	}
	h.s.probeStreamedBytes(ctx, modelsource.SourceHuggingFace, d)
	return d.Files, nil
}

func (h recommendHub) Meta(ctx context.Context, repo string, f modelsource.File) (*models.GGUFMeta, error) {
	meta := h.s.repoMeta(ctx, modelsource.SourceHuggingFace, repo, f, true)
	if meta == nil {
		return nil, errors.New("no model description")
	}
	return meta, nil
}

// newRecommendEngine wires the engine to this server.
func (s *Server) newRecommendEngine(dir string) *recommend.Engine {
	return &recommend.Engine{
		Hub: func() recommend.Hub {
			if s.hfClient == nil {
				return nil
			}
			return recommendHub{s: s, client: s.hfClient}
		},
		Dir: dir,
	}
}

// recommendProfile is the machine the recommendations are worked out for.
func (s *Server) recommendProfile() recommend.Profile {
	p := recommend.Profile{Hardware: s.hardware()}
	if b := s.resolveActiveBuild(); b != nil {
		p.BuildID = b.ID
	}
	p.Archs, p.ArchsKnown = s.supportedArchs()
	return p
}

// recommendClasses maps the feed's context choices to the planner's
// classes.
var recommendClasses = map[string]models.ContextClass{
	"8k": models.ContextShort, "32k": models.ContextMedium, "128k": models.ContextLong, "max": models.ContextMax,
}

func recommendQuery(r *http.Request) (recommend.Intent, string, models.ContextClass) {
	intent := recommend.Intent(r.URL.Query().Get("intent"))
	switch intent {
	case recommend.IntentQuality, recommend.IntentFastest, recommend.IntentContext, recommend.IntentNewest:
	default:
		intent = recommend.IntentQuality
	}
	ctxName := r.URL.Query().Get("ctx")
	class, ok := recommendClasses[ctxName]
	if !ok {
		ctxName, class = "32k", models.ContextMedium
	}
	return intent, ctxName, class
}

// recommendCard is one model in the JSON answer.
type recommendCard struct {
	Name      string          `json:"name"`
	BaseRepo  string          `json:"base_repo,omitempty"`
	Repo      string          `json:"repo"`
	Alts      []string        `json:"alts,omitempty"`
	Gated     bool            `json:"gated,omitempty"`
	Vision    bool            `json:"vision,omitempty"`
	Params    int64           `json:"params"`
	Arch      string          `json:"arch"`
	Released  time.Time       `json:"released"`
	Downloads int             `json:"downloads"`
	Pick      *recommend.Pick `json:"pick"`
}

type recommendAnswer struct {
	Intent      recommend.Intent       `json:"intent"`
	Context     string                 `json:"ctx"`
	BuiltAt     time.Time              `json:"built_at"`
	Stale       bool                   `json:"stale"`
	Unavailable string                 `json:"unavailable,omitempty"`
	Cards       []recommendCard        `json:"cards"`
	Hidden      int                    `json:"hidden"`
	Unverified  []recommend.Unverified `json:"unverified,omitempty"`
}

func (s *Server) recommendAnswer(pool *recommend.Pool, intent recommend.Intent, ctxName string, class models.ContextClass) recommendAnswer {
	a := recommendAnswer{Intent: intent, Context: ctxName, Cards: []recommendCard{}}
	if pool == nil {
		a.Unavailable = "The list is still being built. Try again in a moment."
		return a
	}
	a.BuiltAt, a.Stale, a.Unavailable = pool.Built, pool.Stale(time.Now()), pool.Unavailable
	a.Unverified = pool.Unverified
	if pool.Unavailable != "" {
		return a
	}
	a.Hidden = pool.Hidden(class)
	for _, g := range pool.Order(intent, class) {
		a.Cards = append(a.Cards, recommendCard{
			Name: g.Name, BaseRepo: g.BaseRepo, Repo: g.Repo().ID, Alts: g.Alts(), Gated: g.Gated, Vision: g.Vision,
			Params: g.Params, Arch: g.Arch, Released: g.Released, Downloads: g.Downloads, Pick: g.Picks[class],
		})
	}
	return a
}

// handleRecommend serves the recommendations: GET
// /api/hf/recommend?intent=quality|fastest|context|newest&ctx=8k|32k|128k|max.
// The first request builds the list; later ones read it. htmx requests
// get the feed, others JSON. Always 200: a failure is a sentence in the
// answer, so the page can show it. collapsed=1 returns the collapsed
// form, for the feed's Hide button.
func (s *Server) handleRecommend(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("collapsed") == "1" {
		respondHTML(w)
		s.renderPartial(w, "recommend_collapsed", nil)
		return
	}
	s.serveRecommend(w, r, false)
}

// handleRecommendRefresh rebuilds the list from HuggingFace.
func (s *Server) handleRecommendRefresh(w http.ResponseWriter, r *http.Request) {
	s.serveRecommend(w, r, true)
}

func (s *Server) serveRecommend(w http.ResponseWriter, r *http.Request, refresh bool) {
	intent, ctxName, class := recommendQuery(r)
	pool := s.recommend.Get(r.Context(), s.recommendProfile(), refresh)
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "recommend_feed", s.recommendFeed(pool, intent, ctxName, class, time.Now()))
		return
	}
	respondJSON(w, s.recommendAnswer(pool, intent, ctxName, class))
}
