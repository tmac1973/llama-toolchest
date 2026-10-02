package modelsource

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// countingTransport counts the requests and body bytes a probe costs.
type countingTransport struct {
	requests, bytes atomic.Int64
}

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := http.DefaultTransport.RoundTrip(r)
	if err == nil {
		c.requests.Add(1)
		if resp.ContentLength > 0 {
			c.bytes.Add(resp.ContentLength)
		}
	}
	return resp, err
}

// Measures what ProbeMeta costs on popular repositories: the first GGUF
// model file of each of the most downloaded GGUF repos. Opt-in, since it uses
// the network: HF_LIVE_TEST=1.
func TestLiveProbeMetaCost(t *testing.T) {
	if os.Getenv("HF_LIVE_TEST") != "1" {
		t.Skip("set HF_LIVE_TEST=1 to run against huggingface.co")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	var repos []struct {
		ID string `json:"id"`
	}
	getJSON(t, "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=12", &repos)

	for _, r := range repos {
		var tree []struct {
			Path string `json:"path"`
			Size int64  `json:"size"`
		}
		getJSON(t, "https://huggingface.co/api/models/"+r.ID+"/tree/main?recursive=true", &tree)
		var file string
		var size int64
		for _, e := range tree {
			// The largest file, as the browse tab picks: a draft model
			// or MTP head in the same repository is always smaller.
			if strings.HasSuffix(e.Path, ".gguf") && !strings.Contains(strings.ToLower(e.Path), "mmproj") &&
				(!strings.Contains(e.Path, "-of-0") || strings.Contains(e.Path, "-00001-of-")) && e.Size > size {
				file, size = e.Path, e.Size
			}
		}
		if file == "" {
			continue
		}
		ct := &countingTransport{}
		start := time.Now()
		meta, err := ProbeMeta(ctx, &http.Client{Transport: ct}, "", "https://huggingface.co/"+r.ID+"/resolve/main/"+file, size)
		if err != nil {
			t.Logf("%-55s %v", r.ID, err)
			continue
		}
		t.Logf("%-55s %-12s %3d layers  %d request(s), %6d bytes, %v",
			r.ID, meta.Architecture, meta.NLayers, ct.requests.Load(), ct.bytes.Load(), time.Since(start).Round(time.Millisecond))
	}
}

func getJSON(t *testing.T, url string, v any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}
