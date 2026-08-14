// Package presets discovers recommended sampling parameters for a downloaded
// model from network sources: the Unsloth docs (thinking/instruct/coding
// variants), the base model's generation_config.json, and the legacy params
// file some GGUF repos ship. The GGUF-embedded defaults (general.sampling.*)
// are parsed locally at download time; this package layers additional
// variants and missing fields on top of them.
//
// Every source is best-effort: misses and network failures are logged at
// debug level and skipped, never surfaced to the download path.
package presets

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

const (
	defaultHFBase   = "https://huggingface.co"
	defaultDocsBase = "https://unsloth.ai/docs"
	userAgent       = "llama-toolchest/1.0 (+https://github.com/tmac1973/llama-toolchest)"
	docsCacheTTL    = 24 * time.Hour
)

// Fetcher runs the network preset source chain. Base URLs are overridable
// for tests; the zero values mean the real services.
type Fetcher struct {
	HFBase   string
	DocsBase string
	Token    string // HF bearer token, optional (higher rate limits, gated mirrors)
	CacheDir string // on-disk cache for docs fetches; "" disables caching
	Client   *http.Client
}

// NewFetcher returns a Fetcher caching docs fetches under cacheDir.
func NewFetcher(cacheDir, hfToken string) *Fetcher {
	return &Fetcher{
		Token:    hfToken,
		CacheDir: cacheDir,
		Client:   &http.Client{Timeout: 15 * time.Second},
	}
}

func (f *Fetcher) hfBase() string {
	if f.HFBase != "" {
		return f.HFBase
	}
	return defaultHFBase
}

func (f *Fetcher) docsBase() string {
	if f.DocsBase != "" {
		return f.DocsBase
	}
	return defaultDocsBase
}

// Fetch runs the source chain and returns the merged preset list, starting
// from the presets already attached to the model (the GGUF-embedded default,
// when present). Later sources never overwrite a field an earlier source
// set for the same variant — they fill nils and add new variants. A fetch
// that finds nothing returns the input presets unchanged.
func (f *Fetcher) Fetch(ctx context.Context, m *models.Model) []models.SamplingPreset {
	out := slices.Clone(m.SamplingPresets)

	if docs := f.fetchUnslothDocs(ctx, m.ModelID); len(docs) > 0 {
		out = mergePresets(out, docs)
	}

	base := m.BaseModelRepo
	if base == "" {
		base = f.resolveBaseRepo(ctx, m.ModelID)
	}
	if base != "" {
		if p := f.fetchGenerationConfig(ctx, base); p != nil {
			out = mergePresets(out, []models.SamplingPreset{*p})
		}
	}

	if p := f.fetchParamsFile(ctx, m.ModelID); p != nil {
		out = mergePresets(out, []models.SamplingPreset{*p})
	}

	return out
}

// mergePresets folds incoming variants into existing: unknown variant names
// are added, known ones only have their nil fields filled. Existing
// provenance (Source/SourceURL) is kept — it identifies whichever source set
// the variant's first values.
func mergePresets(existing, incoming []models.SamplingPreset) []models.SamplingPreset {
	for _, p := range incoming {
		idx := -1
		for i := range existing {
			if existing[i].Name == p.Name {
				idx = i
				break
			}
		}
		if idx < 0 {
			existing = models.UpsertSamplingPreset(existing, p)
			continue
		}
		e := &existing[idx]
		if e.Temperature == nil {
			e.Temperature = p.Temperature
		}
		if e.TopP == nil {
			e.TopP = p.TopP
		}
		if e.TopK == nil {
			e.TopK = p.TopK
		}
		if e.MinP == nil {
			e.MinP = p.MinP
		}
		if e.PresencePenalty == nil {
			e.PresencePenalty = p.PresencePenalty
		}
		if e.RepeatPenalty == nil {
			e.RepeatPenalty = p.RepeatPenalty
		}
	}
	return existing
}

// get fetches a URL with etiquette: UA header, optional HF bearer token, one
// retry on 429/network errors. Non-200 responses return an error (the caller
// treats them as a silent miss).
func (f *Fetcher) get(ctx context.Context, url string, withAuth bool) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(2 * time.Second):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", userAgent)
		if withAuth && f.Token != "" {
			req.Header.Set("Authorization", "Bearer "+f.Token)
		}
		resp, err := f.Client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("GET %s: 429", url)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("GET %s: %d", url, resp.StatusCode)
		}
		return body, nil
	}
	return nil, lastErr
}

// cachedGet is get with an on-disk cache (TTL docsCacheTTL). Used for docs
// fetches, which are shared across models; HF per-repo fetches skip caching.
func (f *Fetcher) cachedGet(ctx context.Context, url string) ([]byte, error) {
	if f.CacheDir == "" {
		return f.get(ctx, url, false)
	}
	sum := sha256.Sum256([]byte(url))
	path := filepath.Join(f.CacheDir, hex.EncodeToString(sum[:8])+".cache")
	if st, err := os.Stat(path); err == nil && time.Since(st.ModTime()) < docsCacheTTL {
		if body, err := os.ReadFile(path); err == nil {
			return body, nil
		}
	}
	body, err := f.get(ctx, url, false)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(f.CacheDir, 0o755); err == nil {
		tmp := path + ".tmp"
		if os.WriteFile(tmp, body, 0o644) == nil {
			os.Rename(tmp, path)
		}
	}
	return body, nil
}

// repoName returns the repo part of an "org/repo" HF ID.
func repoName(repoID string) string {
	if i := strings.IndexByte(repoID, '/'); i >= 0 {
		return repoID[i+1:]
	}
	return repoID
}

func debugMiss(what, target string, err error) {
	slog.Debug("preset source miss", "source", what, "target", target, "error", err)
}
