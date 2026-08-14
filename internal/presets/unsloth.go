package presets

import (
	"context"
	neturl "net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// fetchUnslothDocs locates the Unsloth "How to Run" docs page for the model's
// family via the GitBook llms.txt index and parses its Recommended Settings
// table into preset variants. GitBook serves a markdown version of every page
// by appending ".md" to the URL — no HTML scraping involved.
func (f *Fetcher) fetchUnslothDocs(ctx context.Context, repoID string) []models.SamplingPreset {
	index, err := f.cachedGet(ctx, f.docsBase()+"/llms.txt")
	if err != nil {
		debugMiss("unsloth-docs", "llms.txt", err)
		return nil
	}
	pageURL := matchDocsPage(string(index), f.docsBase(), repoID)
	if pageURL == "" {
		return nil
	}
	body, err := f.cachedGet(ctx, pageURL+".md")
	if err != nil {
		debugMiss("unsloth-docs", pageURL, err)
		return nil
	}
	return parseRecommendedSettings(string(body), pageURL)
}

var mdLinkRE = regexp.MustCompile(`\[([^\]]*)\]\(([^()\s]+)\)`)

// matchDocsPage scans llms.txt for model docs pages (paths under
// /docs/models/) and returns the URL whose slug or title matches the repo's
// family key. Exact key match wins; otherwise a prefix match is accepted only
// when the next character of the family key is a letter, so "qwen3" can match
// "qwen3scout"-style names but never swallow "qwen3.8". Returns "" when
// nothing matches — wrong-family numbers are worse than no preset.
func matchDocsPage(llmsTxt, docsBase, repoID string) string {
	family := familyKey(repoName(repoID))
	if family == "" {
		return ""
	}
	origin := docsBase
	if u, err := neturl.Parse(docsBase); err == nil && u.Scheme != "" {
		origin = u.Scheme + "://" + u.Host
	}
	bestURL, bestLen := "", 0
	for _, m := range mdLinkRE.FindAllStringSubmatch(llmsTxt, -1) {
		title, url := m[1], m[2]
		// Root-relative links resolve against the site origin, not the docs
		// base path.
		if strings.HasPrefix(url, "/") {
			url = origin + url
		}
		if !strings.Contains(url, "/models/") {
			continue
		}
		url = strings.TrimSuffix(url, ".md")
		slug := url[strings.LastIndexByte(url, '/')+1:]
		if i := strings.Index(slug, "-how-to"); i > 0 {
			slug = slug[:i]
		}
		// Titles read "Qwen 3.8: How to Run…" — the part before the colon
		// names the family.
		if i := strings.IndexByte(title, ':'); i > 0 {
			title = title[:i]
		}
		for _, key := range []string{familyKey(slug), familyKey(title)} {
			if key == "" {
				continue
			}
			if key == family {
				return url
			}
			if len(key) > bestLen && strings.HasPrefix(family, key) {
				next := family[len(key)]
				if next >= 'a' && next <= 'z' {
					bestURL, bestLen = url, len(key)
				}
			}
		}
	}
	return bestURL
}

var (
	// Tokens that vary per quant/file but not per family.
	sizeTokRE  = regexp.MustCompile(`^(?:\d+(?:\.\d+)?b|a\d+(?:\.\d+)?b|\d+x\d+b?|e\d+|\d+e|\d{4})$`)
	quantTokRE = regexp.MustCompile(`^(?:i?q\d\S*|ud\S*|f16|bf16|f32|fp\d+|gguf)$`)
	dropToks   = map[string]bool{"instruct": true, "it": true, "chat": true, "gguf": true}
)

// familyKey normalizes a repo name, docs slug, or page title down to the
// model-family identity: lowercase, size/quant/date/variant tokens dropped,
// separators removed. "Qwen3.8-27B-GGUF", "qwen3.8" and "Qwen 3.8" all
// produce "qwen3.8".
func familyKey(name string) string {
	name = strings.ToLower(name)
	name = strings.NewReplacer("_", "-", " ", "-").Replace(name)
	var keep []string
	for _, tok := range strings.Split(name, "-") {
		if tok == "" || dropToks[tok] || sizeTokRE.MatchString(tok) || quantTokRE.MatchString(tok) {
			continue
		}
		keep = append(keep, tok)
	}
	return strings.Join(keep, "")
}

// parseRecommendedSettings extracts sampling presets from the "Recommended
// Settings" section of an Unsloth docs page (markdown form). The canonical
// shape is a table with one column per mode:
//
//	| Parameter       | Thinking Mode | Instruct (non-thinking) Mode |
//	| `temperature`   | 1.0           | 0.7                          |
//
// Two-column tables produce a single "default" variant. Pages without a
// table fall back to inline "temperature = 0.7"-style lines within the
// section, all folded into "default".
func parseRecommendedSettings(md, pageURL string) []models.SamplingPreset {
	section := recommendedSection(md)
	if section == "" {
		return nil
	}
	if presets := parseSettingsTable(section, pageURL); len(presets) > 0 {
		return presets
	}
	return parseInlineSettings(section, pageURL)
}

var headingRE = regexp.MustCompile(`^#{1,6}\s`)

// recommendedSection returns the lines from a heading mentioning
// recommended/suggested settings/parameters up to the next heading.
func recommendedSection(md string) string {
	lines := strings.Split(md, "\n")
	start := -1
	for i, line := range lines {
		if !headingRE.MatchString(line) {
			continue
		}
		l := strings.ToLower(line)
		if start >= 0 {
			return strings.Join(lines[start:i], "\n")
		}
		if (strings.Contains(l, "recommended") || strings.Contains(l, "suggested")) &&
			(strings.Contains(l, "settings") || strings.Contains(l, "parameters") || strings.Contains(l, "inference")) {
			start = i + 1
		}
	}
	if start >= 0 {
		return strings.Join(lines[start:], "\n")
	}
	return ""
}

func parseSettingsTable(section, pageURL string) []models.SamplingPreset {
	var header []string
	var presets []models.SamplingPreset
	for _, line := range strings.Split(section, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "|") {
			continue
		}
		cells := splitTableRow(line)
		if len(cells) < 2 {
			continue
		}
		if header == nil {
			header = cells
			presets = make([]models.SamplingPreset, len(cells)-1)
			for i, h := range cells[1:] {
				name := variantForHeader(h, len(cells) == 2)
				presets[i] = models.SamplingPreset{
					Name:        name,
					Label:       labelForVariant(name),
					Description: "From Unsloth docs — recommended settings",
					Source:      "unsloth-docs",
					SourceURL:   pageURL,
				}
			}
			continue
		}
		if strings.Trim(cells[0], ":- ") == "" { // separator row
			continue
		}
		param := normalizeParam(cells[0])
		if param == "" {
			continue
		}
		for i, cell := range cells[1:] {
			if i >= len(presets) {
				break
			}
			if v, ok := firstFloat(cell); ok {
				assignParam(&presets[i], param, v)
			}
		}
	}
	var out []models.SamplingPreset
	for _, p := range presets {
		if !emptyPreset(p) {
			out = append(out, p)
		}
	}
	return dedupeByName(out)
}

var inlineParamRE = regexp.MustCompile("(?i)[`\"']?(temperature|temp|top[_ ]?p|top[_ ]?k|min[_ ]?p|presence[_ ]?penalty|repetition[_ ]?penalty|repeat[_ ]?penalty)[`\"']?\\s*[:=]?[^\\d\\n-]{0,6}([-+]?\\d*\\.?\\d+)")

func parseInlineSettings(section, pageURL string) []models.SamplingPreset {
	p := models.SamplingPreset{
		Name:        "default",
		Label:       labelForVariant("default"),
		Description: "From Unsloth docs — recommended settings",
		Source:      "unsloth-docs",
		SourceURL:   pageURL,
	}
	for _, m := range inlineParamRE.FindAllStringSubmatch(section, -1) {
		if v, err := strconv.ParseFloat(m[2], 64); err == nil {
			assignParam(&p, normalizeParam(m[1]), v)
		}
	}
	if emptyPreset(p) {
		return nil
	}
	return []models.SamplingPreset{p}
}

func splitTableRow(line string) []string {
	parts := strings.Split(strings.Trim(line, "|"), "|")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, strings.TrimSpace(p))
	}
	return out
}

// variantForHeader buckets a table column header into a variant name,
// mirroring the old scraper's naming so persisted configs stay familiar.
func variantForHeader(h string, single bool) string {
	l := strings.ToLower(h)
	suffix := ""
	if strings.Contains(l, "coding") || strings.Contains(l, "webdev") {
		suffix = "-coding"
	}
	switch {
	case strings.Contains(l, "non-thinking") || strings.Contains(l, "non thinking") || strings.Contains(l, "instruct"):
		return "non-thinking" + suffix
	case strings.Contains(l, "thinking"):
		return "thinking" + suffix
	case strings.Contains(l, "reasoning"):
		return "reasoning" + suffix
	default:
		if suffix != "" {
			return "default" + suffix
		}
		_ = single
		return "default"
	}
}

func labelForVariant(name string) string {
	base := strings.TrimSuffix(name, "-coding")
	label := map[string]string{
		"default":      "Unsloth recommended",
		"thinking":     "Thinking mode",
		"non-thinking": "Non-thinking mode",
		"reasoning":    "Reasoning",
	}[base]
	if label == "" {
		label = name
	}
	if strings.HasSuffix(name, "-coding") {
		label += " (coding)"
	}
	return label
}

func normalizeParam(s string) string {
	s = strings.ToLower(strings.Trim(s, "`*\" '"))
	s = strings.NewReplacer(" ", "_", "-", "_").Replace(s)
	switch s {
	case "temperature", "temp":
		return "temperature"
	case "top_p", "topp":
		return "top_p"
	case "top_k", "topk":
		return "top_k"
	case "min_p", "minp":
		return "min_p"
	case "presence_penalty":
		return "presence_penalty"
	case "repetition_penalty", "repeat_penalty":
		return "repeat_penalty"
	}
	return ""
}

var floatRE = regexp.MustCompile(`[-+]?\d*\.?\d+`)

func firstFloat(cell string) (float64, bool) {
	m := floatRE.FindString(cell)
	if m == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(m, 64)
	return v, err == nil
}

// assignParam sets a parsed value on the preset, enforcing the same sanity
// ranges the GGUF parser and the old scraper use.
func assignParam(p *models.SamplingPreset, param string, v float64) {
	switch param {
	case "temperature":
		if v >= 0 && v <= 4 {
			p.Temperature = &v
		}
	case "top_p":
		if v >= 0 && v <= 1 {
			p.TopP = &v
		}
	case "top_k":
		if k := int(v); float64(k) == v && k >= 1 && k <= 1000 {
			p.TopK = &k
		}
	case "min_p":
		if v >= 0 && v <= 1 {
			p.MinP = &v
		}
	case "presence_penalty":
		if v >= -2 && v <= 2 {
			p.PresencePenalty = &v
		}
	case "repeat_penalty":
		if v > 0 && v <= 3 {
			p.RepeatPenalty = &v
		}
	}
}

func emptyPreset(p models.SamplingPreset) bool {
	return p.Temperature == nil && p.TopP == nil && p.TopK == nil &&
		p.MinP == nil && p.PresencePenalty == nil && p.RepeatPenalty == nil
}

func dedupeByName(presets []models.SamplingPreset) []models.SamplingPreset {
	seen := map[string]bool{}
	out := presets[:0]
	for _, p := range presets {
		if !seen[p.Name] {
			seen[p.Name] = true
			out = append(out, p)
		}
	}
	return out
}
