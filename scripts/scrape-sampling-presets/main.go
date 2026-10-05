// Command scrape-sampling-presets pulls recommended sampling parameters
// from HuggingFace and emits internal/models/sampling_presets_data.go.
//
// Strategy per repo:
//  1. List Unsloth GGUF repos via the HF /api/models endpoint, sorted by
//     download count. The base-model name is read from each repo's tags.
//  2. Fetch generation_config.json on the base repo — structured, authoritative
//     for the default sampler values.
//  3. Fetch the GGUF repo's README and scan for explicit "Temperature=X" style
//     recommendations, including thinking / non-thinking variants that Qwen3
//     and similar models ship.
//  4. Merge into one or more named presets per repo.
//
// Models without any usable sampling data are skipped silently — coverage is
// uneven across model families and that's fine.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	hfBase     = "https://huggingface.co"
	apiList    = "/api/models"
	rawPath    = "/raw/main"
	httpUA     = "llama-toolchest-scraper/1.0"
	httpRetry  = 2
	outputDef  = "internal/models/sampling_presets_data.go"
	listLimit  = 200
	listSort   = "downloads"
	listDir    = "-1"
	httpExpire = 30 * time.Second
)

var (
	flagAuthor  = flag.String("author", "unsloth", "HuggingFace author to enumerate (comma-separated for multiple)")
	flagLimit   = flag.Int("limit", listLimit, "max repos to enumerate per author")
	flagOut     = flag.String("out", outputDef, "output Go file (relative to repo root)")
	flagToken   = flag.String("token", os.Getenv("HF_TOKEN"), "HuggingFace token (optional, for higher rate limits)")
	flagVerbose = flag.Bool("v", false, "verbose logging")
)

type hfModel struct {
	ID   string   `json:"id"`
	Tags []string `json:"tags"`
}

// genConfig matches the subset of HF generation_config.json fields we care
// about. The schema is loose in practice — fields may be missing or null.
type genConfig struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	MinP              *float64 `json:"min_p,omitempty"`
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
	DoSample          *bool    `json:"do_sample,omitempty"`
}

type preset struct {
	Name        string
	Label       string
	Description string
	Source      string
	SourceURL   string

	Temperature     *float64
	TopP            *float64
	TopK            *int
	MinP            *float64
	PresencePenalty *float64
	RepeatPenalty   *float64
}

func main() {
	flag.Parse()
	log.SetFlags(0)

	client := &http.Client{Timeout: httpExpire}

	authors := splitCSV(*flagAuthor)
	out := map[string][]preset{}

	for _, author := range authors {
		repos, err := listRepos(client, author, *flagLimit)
		if err != nil {
			log.Fatalf("list %s: %v", author, err)
		}
		log.Printf("[%s] %d repos enumerated", author, len(repos))

		for _, r := range repos {
			base := baseModelFromTags(r.Tags)
			if base == "" {
				if *flagVerbose {
					log.Printf("  skip %s — no base_model tag", r.ID)
				}
				continue
			}

			presets := buildPresets(client, r.ID, base)
			if len(presets) == 0 {
				if *flagVerbose {
					log.Printf("  no presets %s (base=%s)", r.ID, base)
				}
				continue
			}
			out[r.ID] = presets
			log.Printf("  + %s → %d preset(s)", r.ID, len(presets))
		}
	}

	if err := writeOutput(*flagOut, out); err != nil {
		log.Fatalf("write output: %v", err)
	}
	log.Printf("wrote %s — %d repos, %d total preset variants",
		*flagOut, len(out), countPresets(out))
}

// listRepos queries the HF API for GGUF repos under the given author.
func listRepos(client *http.Client, author string, limit int) ([]hfModel, error) {
	q := fmt.Sprintf("%s?author=%s&filter=gguf&limit=%d&sort=%s&direction=%s",
		apiList, author, limit, listSort, listDir)
	body, err := fetch(client, hfBase+q)
	if err != nil {
		return nil, err
	}
	var repos []hfModel
	if err := json.Unmarshal(body, &repos); err != nil {
		return nil, fmt.Errorf("decode list: %w", err)
	}
	return repos, nil
}

// baseModelFromTags pulls the upstream model ID from the HF "base_model:..."
// tag. We prefer the unprefixed form when present and fall back to the
// "quantized:" variant otherwise.
func baseModelFromTags(tags []string) string {
	const (
		plainPrefix = "base_model:"
		quantPrefix = "base_model:quantized:"
	)
	var plain, quant string
	for _, t := range tags {
		switch {
		case strings.HasPrefix(t, quantPrefix):
			quant = strings.TrimPrefix(t, quantPrefix)
		case strings.HasPrefix(t, plainPrefix):
			v := strings.TrimPrefix(t, plainPrefix)
			// The plain tag is sometimes the same as the quant tag with no
			// "quantized:" prefix; pick whichever isn't empty and skip values
			// that are themselves "quantized:..." (shouldn't happen but be
			// defensive).
			if !strings.HasPrefix(v, "quantized:") {
				plain = v
			}
		}
	}
	if plain != "" {
		return plain
	}
	return quant
}

// buildPresets merges generation_config.json (base repo) with README-extracted
// variants (GGUF repo) into a flat preset list.
func buildPresets(client *http.Client, ggufRepo, baseRepo string) []preset {
	var presets []preset

	// 1. generation_config.json from the base repo. Skip 404s silently — not
	//    every model carries one.
	if gc, url, ok := fetchGenConfig(client, baseRepo); ok {
		if p, populated := genConfigToPreset(gc, baseRepo, url); populated {
			presets = append(presets, p)
		}
	}

	// 2. README on the GGUF repo. The README is where Unsloth documents
	//    thinking-mode vs non-thinking-mode variants, so this is what gives
	//    us multi-preset entries for the Qwen3 family.
	if md, url, ok := fetchReadme(client, ggufRepo); ok {
		variants := parseReadmeVariants(md, url)
		presets = mergeVariants(presets, variants)
	}

	return dedupePresets(presets)
}

func fetchGenConfig(client *http.Client, repo string) (genConfig, string, bool) {
	url := hfBase + "/" + repo + rawPath + "/generation_config.json"
	body, err := fetch(client, url)
	if err != nil {
		return genConfig{}, "", false
	}
	var gc genConfig
	if err := json.Unmarshal(body, &gc); err != nil {
		return genConfig{}, "", false
	}
	return gc, url, true
}

func genConfigToPreset(gc genConfig, baseRepo, url string) (preset, bool) {
	p := preset{
		Name:        "default",
		Label:       "Upstream default",
		Source:      "generation_config.json",
		SourceURL:   url,
		Description: fmt.Sprintf("From %s/generation_config.json", baseRepo),

		Temperature:   gc.Temperature,
		TopP:          gc.TopP,
		TopK:          gc.TopK,
		MinP:          gc.MinP,
		RepeatPenalty: gc.RepetitionPenalty,
	}
	// A preset has to carry at least one value to be useful. generation_config
	// often contains only token IDs; skip those cases.
	populated := p.Temperature != nil || p.TopP != nil || p.TopK != nil ||
		p.MinP != nil || p.RepeatPenalty != nil
	return p, populated
}

func fetchReadme(client *http.Client, repo string) (string, string, bool) {
	url := hfBase + "/" + repo + rawPath + "/README.md"
	body, err := fetch(client, url)
	if err != nil {
		return "", "", false
	}
	return string(body), url, true
}

// Matches `Temperature=0.6`, `top_p: 0.95`, `TopK=20`, etc. The param token
// is intentionally loose to handle Unsloth's varied capitalization.
var paramRE = regexp.MustCompile(`(?i)\b(temperature|temp|top[_ ]?p|top[_ ]?k|min[_ ]?p|repetition[_ ]?penalty|repeat[_ ]?penalty|presence[_ ]?penalty)\b[^\d\n]{0,8}([+-]?\d*\.?\d+)`)

// parseReadmeVariants scans the README for sampling-parameter mentions and
// groups them into named variants based on context cues in the same line.
// Variant detection is heuristic — when nothing matches, the values land in
// the "default" bucket.
func parseReadmeVariants(md, url string) []preset {
	bucket := map[string]*preset{}
	get := func(name, label, desc string) *preset {
		if p, ok := bucket[name]; ok {
			return p
		}
		p := &preset{
			Name:        name,
			Label:       label,
			Description: desc,
			Source:      "readme",
			SourceURL:   url,
		}
		bucket[name] = p
		return p
	}

	for _, line := range strings.Split(md, "\n") {
		matches := paramRE.FindAllStringSubmatch(line, -1)
		if len(matches) == 0 {
			continue
		}
		// Filter false positives. A line with 2+ param=value pairs is
		// almost certainly a sampling recommendation (e.g. "temperature=0.6,
		// top_p=0.95, top_k=20"), regardless of surrounding prose — many
		// READMEs put the "we recommend" framing on the preceding line.
		// A line with only 1 match needs a guidance keyword to clear it,
		// to avoid catching random prose like "the temperature in Paris is 22°C".
		if len(matches) < 2 && !isGuidanceLine(line) {
			continue
		}

		name, label, desc := variantOfLine(line)
		p := get(name, label, desc)
		for _, m := range matches {
			assign(p, m[1], m[2])
		}
	}

	var out []preset
	for _, p := range bucket {
		if presetPopulated(*p) {
			out = append(out, *p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func isGuidanceLine(line string) bool {
	lc := strings.ToLower(line)
	for _, hint := range []string{
		"recommend", "suggest", "use ", "use`", "set ", "for thinking",
		"for non-thinking", "for non thinking", "default", "best practice",
	} {
		if strings.Contains(lc, hint) {
			return true
		}
	}
	return false
}

// codingRE matches the literal word "coding" / "webdev" as a whole word so
// substrings like "decoding" / "encoding" don't falsely flag a thinking line
// as the coding variant.
var codingRE = regexp.MustCompile(`(?i)\b(coding|webdev)\b`)

func variantOfLine(line string) (name, label, desc string) {
	lc := strings.ToLower(line)
	// "coding" tasks sometimes get their own variant within a mode (Qwen3.6
	// publishes both a general-purpose thinking preset and a coding one).
	codingSuffix := ""
	codingLabel := ""
	if codingRE.MatchString(line) {
		codingSuffix = "-coding"
		codingLabel = " (coding)"
	}

	switch {
	case strings.Contains(lc, "non-thinking") || strings.Contains(lc, "non thinking"):
		return "non-thinking" + codingSuffix,
			"Non-thinking mode" + codingLabel,
			"From README — for enable_thinking=False"
	case strings.Contains(lc, "thinking"):
		return "thinking" + codingSuffix,
			"Thinking mode" + codingLabel,
			"From README — for enable_thinking=True"
	case strings.Contains(lc, "reasoning"):
		return "reasoning" + codingSuffix,
			"Reasoning mode" + codingLabel,
			"From README — for reasoning/CoT prompts"
	}
	if codingSuffix != "" {
		return "coding", "Publisher recommendation (coding)", "From README"
	}
	return "default", "Publisher recommendation", "From README"
}

// assign sets the matching field on the preset. Unknown params are ignored.
func assign(p *preset, paramRaw, valueRaw string) {
	param := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(paramRaw, "_", ""), " ", ""))
	v, err := strconv.ParseFloat(valueRaw, 64)
	if err != nil {
		return
	}
	switch param {
	case "temperature", "temp":
		if v >= 0 && v <= 4 {
			p.Temperature = fptr(v)
		}
	case "topp":
		if v >= 0 && v <= 1 {
			p.TopP = fptr(v)
		}
	case "topk":
		// top_k is an int. Reject obvious nonsense (top_k=0 disables, treat as nil)
		if v >= 1 && v <= 1000 {
			n := int(v)
			p.TopK = &n
		}
	case "minp":
		if v >= 0 && v <= 1 {
			p.MinP = fptr(v)
		}
	case "presencepenalty":
		if v >= -2 && v <= 2 {
			p.PresencePenalty = fptr(v)
		}
	case "repetitionpenalty", "repeatpenalty":
		if v >= 0 && v <= 3 {
			p.RepeatPenalty = fptr(v)
		}
	}
}

// mergeVariants merges README-derived variants into a presets list that
// already contains the generation_config.json default. README values
// override the gen-config default for the same variant; new variants are
// appended.
func mergeVariants(base []preset, variants []preset) []preset {
	if len(variants) == 0 {
		return base
	}

	// Find the existing "default" preset (from gen-config) so we can use it
	// as a fill-in for variants that don't specify every field.
	var defaults *preset
	for i := range base {
		if base[i].Name == "default" {
			defaults = &base[i]
			break
		}
	}

	// If README contributes a "default" variant, prefer it over gen-config
	// (the README is curated by the publisher specifically for this release).
	out := append([]preset(nil), base...)
	for _, v := range variants {
		v := v // local copy
		if defaults != nil {
			fillFrom(&v, defaults)
		}

		replaced := false
		for i := range out {
			if out[i].Name == v.Name {
				out[i] = v
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, v)
		}
	}
	return out
}

func fillFrom(dst, src *preset) {
	if dst.Temperature == nil {
		dst.Temperature = src.Temperature
	}
	if dst.TopP == nil {
		dst.TopP = src.TopP
	}
	if dst.TopK == nil {
		dst.TopK = src.TopK
	}
	if dst.MinP == nil {
		dst.MinP = src.MinP
	}
	if dst.PresencePenalty == nil {
		dst.PresencePenalty = src.PresencePenalty
	}
	if dst.RepeatPenalty == nil {
		dst.RepeatPenalty = src.RepeatPenalty
	}
}

func dedupePresets(in []preset) []preset {
	seen := map[string]bool{}
	var out []preset
	for _, p := range in {
		if seen[p.Name] {
			continue
		}
		seen[p.Name] = true
		if presetPopulated(p) {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		// "default" first, then alphabetical
		if out[i].Name == "default" {
			return true
		}
		if out[j].Name == "default" {
			return false
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func presetPopulated(p preset) bool {
	return p.Temperature != nil || p.TopP != nil || p.TopK != nil ||
		p.MinP != nil || p.PresencePenalty != nil || p.RepeatPenalty != nil
}

// writeOutput emits the generated Go file. Entries are sorted for stable
// diffs, and the file's only purpose is to populate samplingPresetsByRepo
// — the SamplingPreset type itself lives in sampling_presets.go.
func writeOutput(path string, data map[string][]preset) error {
	var b strings.Builder
	b.WriteString("// Code generated by scripts/scrape-sampling-presets; DO NOT EDIT.\n")
	b.WriteString("// To refresh: make scrape-sampling-presets\n\n")
	b.WriteString("package models\n\n")
	b.WriteString("// samplingPresetsByRepo maps HuggingFace repo IDs to their preset variants.\n")
	b.WriteString("// Generated from generation_config.json + README scraping; see scripts/scrape-sampling-presets.\n")
	b.WriteString("var samplingPresetsByRepo = map[string][]SamplingPreset{\n")

	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		fmt.Fprintf(&b, "\t%q: {\n", k)
		for _, p := range data[k] {
			fmt.Fprintf(&b, "\t\t{\n")
			fmt.Fprintf(&b, "\t\t\tName: %q,\n", p.Name)
			fmt.Fprintf(&b, "\t\t\tLabel: %q,\n", p.Label)
			if p.Description != "" {
				fmt.Fprintf(&b, "\t\t\tDescription: %q,\n", p.Description)
			}
			fmt.Fprintf(&b, "\t\t\tSource: %q,\n", p.Source)
			if p.SourceURL != "" {
				fmt.Fprintf(&b, "\t\t\tSourceURL: %q,\n", p.SourceURL)
			}
			if p.Temperature != nil {
				fmt.Fprintf(&b, "\t\t\tTemperature: fptr(%s),\n", trimFloat(*p.Temperature))
			}
			if p.TopP != nil {
				fmt.Fprintf(&b, "\t\t\tTopP: fptr(%s),\n", trimFloat(*p.TopP))
			}
			if p.TopK != nil {
				fmt.Fprintf(&b, "\t\t\tTopK: iptr(%d),\n", *p.TopK)
			}
			if p.MinP != nil {
				fmt.Fprintf(&b, "\t\t\tMinP: fptr(%s),\n", trimFloat(*p.MinP))
			}
			if p.PresencePenalty != nil {
				fmt.Fprintf(&b, "\t\t\tPresencePenalty: fptr(%s),\n", trimFloat(*p.PresencePenalty))
			}
			if p.RepeatPenalty != nil {
				fmt.Fprintf(&b, "\t\t\tRepeatPenalty: fptr(%s),\n", trimFloat(*p.RepeatPenalty))
			}
			fmt.Fprintf(&b, "\t\t},\n")
		}
		fmt.Fprintf(&b, "\t},\n")
	}

	b.WriteString("}\n")
	// gofmt the output, so the checked-in file passes gofmt -l and a
	// refresh's diff shows only data changes.
	src, err := format.Source([]byte(b.String()))
	if err != nil {
		return fmt.Errorf("formatting generated code: %w", err)
	}
	return os.WriteFile(path, src, 0o644)
}

func trimFloat(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	// Force a decimal so Go infers float64 — `0` becomes `0.0`, `1` becomes `1.0`.
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

func countPresets(data map[string][]preset) int {
	n := 0
	for _, v := range data {
		n += len(v)
	}
	return n
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := parts[:0]
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func fptr(v float64) *float64 { return &v }

// fetch GETs a URL and returns the body. 404s and other non-2xx return an
// error so callers can decide to skip. Retries network errors httpRetry times.
func fetch(client *http.Client, url string) ([]byte, error) {
	var last error
	for attempt := 0; attempt <= httpRetry; attempt++ {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", httpUA)
		if *flagToken != "" {
			req.Header.Set("Authorization", "Bearer "+*flagToken)
		}

		resp, err := client.Do(req)
		if err != nil {
			last = err
			time.Sleep(time.Duration(attempt+1) * time.Second)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 200 {
			return body, nil
		}
		// 429 (rate-limited) deserves a retry; everything else is terminal.
		if resp.StatusCode == 429 && attempt < httpRetry {
			time.Sleep(time.Duration(attempt+1) * 2 * time.Second)
			last = fmt.Errorf("HTTP %d", resp.StatusCode)
			continue
		}
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return nil, last
}
