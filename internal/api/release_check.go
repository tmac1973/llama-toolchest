package api

import (
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// latestReleaseAPI is GitHub's newest published release of toolchest.
// Drafts and pre-releases are left out of it, so a release-please PR that
// has not merged never shows as an update.
const latestReleaseAPI = "https://api.github.com/repos/tmac1973/llama-toolchest/releases/latest"

// latestReleasePage is the badge's link when GitHub's answer has no
// usable page link of its own.
const latestReleasePage = "https://github.com/tmac1973/llama-toolchest/releases/latest"

const (
	// releaseCheckTTL is how long one answer from GitHub is kept. The
	// sidebar asks on every page load, and unauthenticated GitHub API
	// calls are limited to 60 an hour per address.
	releaseCheckTTL = 6 * time.Hour
	// releaseCheckRetry is how long a failed check is kept before asking
	// again, so a host with no internet access does not ask on every page.
	releaseCheckRetry = time.Hour
)

// latestRelease is the part of GitHub's release object the badge needs.
type latestRelease struct {
	Tag string `json:"tag_name"`
	URL string `json:"html_url"`
}

// releaseChecker asks GitHub for toolchest's newest release and keeps the
// answer. The zero value is ready to use.
type releaseChecker struct {
	apiURL string       // "" = latestReleaseAPI; tests point it at a fake
	client *http.Client // nil = a client with a short timeout

	mu      sync.Mutex
	checked time.Time
	ttl     time.Duration
	latest  latestRelease
}

// get returns the newest release, asking GitHub when the kept answer is
// older than its TTL. A failed check returns the last good answer, or a
// zero release when there is none.
func (c *releaseChecker) get(ctx context.Context) latestRelease {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checked.IsZero() && time.Since(c.checked) < c.ttl {
		return c.latest
	}
	rel, err := c.fetch(ctx)
	c.checked = time.Now()
	if err != nil {
		slog.Debug("toolchest release check failed", "error", err)
		c.ttl = releaseCheckRetry
		return c.latest
	}
	c.ttl = releaseCheckTTL
	c.latest = rel
	return rel
}

func (c *releaseChecker) fetch(ctx context.Context) (latestRelease, error) {
	url := c.apiURL
	if url == "" {
		url = latestReleaseAPI
	}
	client := c.client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return latestRelease{}, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := client.Do(req)
	if err != nil {
		return latestRelease{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return latestRelease{}, fmt.Errorf("github: %s", resp.Status)
	}
	var rel latestRelease
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return latestRelease{}, err
	}
	if rel.Tag == "" {
		return latestRelease{}, fmt.Errorf("github: release has no tag")
	}
	if !strings.HasPrefix(rel.URL, "https://github.com/") {
		rel.URL = latestReleasePage
	}
	return rel, nil
}

// parseSemver reads "v1.2.3" or "1.2.3" into its three numbers. ok is
// false for anything else, which covers dev builds ("dev-abc1234") and
// snapshots ("1.2.3-snapshot+abc1234"): those are never offered an update.
func parseSemver(v string) (parts [3]int, ok bool) {
	fields := strings.Split(strings.TrimPrefix(v, "v"), ".")
	if len(fields) != 3 {
		return parts, false
	}
	for i, f := range fields {
		n, err := strconv.Atoi(f)
		if err != nil || n < 0 {
			return parts, false
		}
		parts[i] = n
	}
	return parts, true
}

// newerRelease reports whether release is a newer version than current.
func newerRelease(current, release string) bool {
	cur, ok := parseSemver(current)
	if !ok {
		return false
	}
	rel, ok := parseSemver(release)
	if !ok {
		return false
	}
	for i := range cur {
		if rel[i] != cur[i] {
			return rel[i] > cur[i]
		}
	}
	return false
}

// handleReleaseCheck answers the sidebar's update badge: the badge's HTML
// when a newer release than this build is out, an empty body otherwise.
// Non-htmx callers get the result as JSON.
func (s *Server) handleReleaseCheck(w http.ResponseWriter, r *http.Request) {
	var rel latestRelease
	// Dev and snapshot builds have no version to compare, so they never
	// ask GitHub.
	if _, ok := parseSemver(s.version); ok {
		rel = s.releaseCheck.get(r.Context())
	}
	available := newerRelease(s.version, rel.Tag)

	if !isHTMX(r) {
		respondJSON(w, map[string]any{
			"current":          s.version,
			"latest":           rel.Tag,
			"url":              rel.URL,
			"update_available": available,
		})
		return
	}
	respondHTML(w)
	if !available {
		return
	}
	fmt.Fprintf(w, `<a class="update-badge" href="%s" target="_blank" rel="noopener" title="Open the release notes on GitHub">New version %s available</a>`,
		template.HTMLEscapeString(rel.URL), template.HTMLEscapeString(rel.Tag))
}
