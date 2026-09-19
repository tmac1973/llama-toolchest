// Package autoconfig proposes a starting profile for a model: code rules
// make it fit the machine (models.PlanFit), and a small helper model reads
// the model's documentation for advice only the publisher knows, which
// code then checks before anything is proposed.
package autoconfig

import (
	"context"
	"regexp"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Fetcher reads a URL, from a cache when it can. withAuth sends the
// Hugging Face token, which gated repositories need.
type Fetcher interface {
	CachedGet(ctx context.Context, url string, withAuth bool) ([]byte, error)
	ResolveBaseRepo(ctx context.Context, repoID string) string
}

// Card is a model's documentation, trimmed to what matters for running it.
type Card struct {
	Text    string   // what the helper model reads
	Sources []string // the repositories it came from, GGUF repository first
}

// DefaultCardChars is the card budget when the caller has no better one:
// about 6,000 tokens, which with the instructions and a 2,048-token answer
// fits a 16,384-token context.
const DefaultCardChars = 24000

// cardKeywords select the sections worth reading. A section is kept when
// its heading or its text mentions any of them.
var cardKeywords = []string{
	"recommend", "sampling", "temperature", "top_p", "top-p", "top_k", "top-k", "min_p", "min-p",
	"presence", "repetition", "repeat", "llama.cpp", "llama-server", "llama-cli", "gguf",
	"context", "thinking", "reasoning", "speculative", "mtp", "multi-token", "draft", "eagle",
	"dflash", "jinja", "chat template", "tool", "best practice", "usage", "quickstart",
}

var (
	frontMatter = regexp.MustCompile(`(?s)\A---\n.*?\n---\n`)
	htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)
	htmlTag     = regexp.MustCompile(`</?[a-zA-Z][^>]*>`)
	imageLink   = regexp.MustCompile(`!\[[^\]]*\]\([^)]*\)`)
	linkOnly    = regexp.MustCompile(`^\s*(\[[^\]]*\]\([^)]*\)\s*)+$`)
	blankRuns   = regexp.MustCompile(`\n{3,}`)
	headingLine = regexp.MustCompile(`^#{1,6}\s`)
)

// FetchCard reads the model card of the GGUF repository and of the model
// it was made from, when that is a different repository. A missing card is
// not an error: the result is simply empty, and autoconfigure proceeds on
// the hardware fit alone.
func FetchCard(ctx context.Context, f Fetcher, hfBase string, m *models.Model, maxChars int) Card {
	if maxChars <= 0 {
		maxChars = DefaultCardChars
	}
	if m.Source != "" && m.Source != modelsource.SourceHuggingFace {
		return Card{} // model cards are read from Hugging Face only
	}
	repos := []string{m.ModelID}
	base := m.BaseModelRepo
	if base == "" {
		base = f.ResolveBaseRepo(ctx, m.ModelID)
	}
	if base != "" && !strings.EqualFold(base, m.ModelID) {
		repos = append(repos, base)
	}

	var card Card
	var parts []string
	for _, repo := range repos {
		body, err := f.CachedGet(ctx, strings.TrimRight(hfBase, "/")+"/"+repo+"/raw/main/README.md", true)
		if err != nil || len(body) == 0 {
			continue
		}
		if text := TrimCard(string(body)); text != "" {
			parts = append(parts, "## From the model card of "+repo+"\n\n"+text)
			card.Sources = append(card.Sources, repo)
		}
	}
	card.Text = capText(strings.Join(parts, "\n\n"), maxChars)
	return card
}

// TrimCard strips what a reader does not need — front matter, HTML,
// images, badge rows — and keeps only the sections about running the
// model. Sections keep their order.
func TrimCard(md string) string {
	md = strings.ReplaceAll(md, "\r\n", "\n")
	md = frontMatter.ReplaceAllString(md, "")
	md = htmlComment.ReplaceAllString(md, "")
	md = imageLink.ReplaceAllString(md, "")
	md = htmlTag.ReplaceAllString(md, "")

	var lines []string
	for _, l := range strings.Split(md, "\n") {
		if linkOnly.MatchString(l) {
			continue // badge and link rows
		}
		lines = append(lines, strings.TrimRight(l, " \t"))
	}

	// Split into sections at headings; text before the first heading is a
	// section of its own.
	var sections [][]string
	cur := []string{}
	for _, l := range lines {
		if headingLine.MatchString(l) && len(cur) > 0 {
			sections = append(sections, cur)
			cur = []string{}
		}
		cur = append(cur, l)
	}
	sections = append(sections, cur)

	var kept []string
	for _, sec := range sections {
		text := strings.TrimSpace(strings.Join(sec, "\n"))
		if text == "" || !mentionsKeyword(text) {
			continue
		}
		kept = append(kept, text)
	}
	return strings.TrimSpace(blankRuns.ReplaceAllString(strings.Join(kept, "\n\n"), "\n\n"))
}

func mentionsKeyword(text string) bool {
	lower := strings.ToLower(text)
	for _, k := range cardKeywords {
		if strings.Contains(lower, k) {
			return true
		}
	}
	return false
}

// capText cuts text at the last paragraph break within max characters.
func capText(text string, max int) string {
	if len(text) <= max {
		return text
	}
	cut := text[:max]
	if i := strings.LastIndex(cut, "\n\n"); i > max/2 {
		cut = cut[:i]
	}
	return cut + "\n\n[The rest of the model card was left out for length.]"
}

// CardCharsForContext is the card budget for a helper model with context
// ctx tokens: what is left after the answer and the instructions, at a
// conservative three characters per token.
func CardCharsForContext(ctx int) int {
	const answerTokens, instructionTokens, charsPerToken = 2048, 1000, 3
	n := (ctx - answerTokens - instructionTokens) * charsPerToken
	if n < 4000 {
		return 4000
	}
	return min(n, 48000)
}
