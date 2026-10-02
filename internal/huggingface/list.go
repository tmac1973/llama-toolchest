package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ListedModel is one repository from a GGUF list query, with the fields
// the recommendations rank and filter by.
type ListedModel struct {
	ID           string    `json:"id"`
	Author       string    `json:"author"`
	Downloads    int       `json:"downloads"`
	Likes        int       `json:"likes"`
	Tags         []string  `json:"tags"`
	Gated        Gated     `json:"gated"`
	Private      bool      `json:"private"`
	CreatedAt    time.Time `json:"createdAt"`
	LastModified time.Time `json:"lastModified"`
	PipelineTag  string    `json:"pipeline_tag"`
	SHA          string    `json:"sha"`
	CardData     struct {
		BaseModel BaseModels `json:"base_model"`
	} `json:"cardData"`
	// GGUF is HuggingFace's summary of one GGUF file in the repository.
	// Which file it describes is HuggingFace's choice and is sometimes the
	// image reader or a draft model, so treat it as a hint.
	GGUF *struct {
		Total         int64  `json:"total"`
		Architecture  string `json:"architecture"`
		ContextLength int    `json:"context_length"`
	} `json:"gguf"`
}

// listExpand names every field ListedModel reads. Once a query asks for
// any expand[] field, HuggingFace returns only the fields named, so a
// field missing from this list silently arrives empty.
var listExpand = []string{
	"author", "downloads", "likes", "tags", "gated", "private", "createdAt",
	"lastModified", "pipeline_tag", "sha", "cardData", "gguf",
}

// Gated is HuggingFace's gated flag: false, or the kind of approval the
// repository asks for ("auto", "manual").
type Gated bool

func (g *Gated) UnmarshalJSON(b []byte) error {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch x := v.(type) {
	case bool:
		*g = Gated(x)
	case string:
		*g = Gated(x != "" && x != "false")
	}
	return nil
}

// BaseModels is cardData.base_model: a repository name or a list of them.
type BaseModels []string

func (m *BaseModels) UnmarshalJSON(b []byte) error {
	var one string
	if json.Unmarshal(b, &one) == nil {
		if one != "" {
			*m = BaseModels{one}
		}
		return nil
	}
	var many []string
	if json.Unmarshal(b, &many) == nil {
		*m = many
	}
	return nil
}

// ListQuery is one GGUF list query: by pipeline tag or by author, in one
// sort order, newest or largest first.
type ListQuery struct {
	PipelineTag string
	Author      string
	Sort        string // "downloads", "trendingScore", "createdAt", ...
	Limit       int
}

func (q ListQuery) values() url.Values {
	v := url.Values{}
	v.Set("filter", "gguf")
	if q.PipelineTag != "" {
		v.Set("pipeline_tag", q.PipelineTag)
	}
	if q.Author != "" {
		v.Set("author", q.Author)
	}
	if q.Sort != "" {
		v.Set("sort", q.Sort)
		v.Set("direction", "-1")
	}
	if q.Limit > 0 {
		v.Set("limit", strconv.Itoa(q.Limit))
	}
	for _, f := range listExpand {
		v.Add("expand[]", f)
	}
	return v
}

// ListGGUF runs one GGUF list query.
func (c *Client) ListGGUF(ctx context.Context, q ListQuery) ([]ListedModel, error) {
	var out []ListedModel
	if err := c.getJSON(ctx, c.apiURL()+"/models?"+q.values().Encode(), &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *Client) apiURL() string {
	if c.apiBase != "" {
		return c.apiBase
	}
	return baseURL
}

// maxRetryAfter caps how long a rate-limited request waits before its one
// retry. A host asking for longer is better answered with an error the
// caller can show than with a request that seems to hang.
const maxRetryAfter = 10 * time.Second

// getJSON fetches and decodes one API response. A 429 is retried once,
// after the wait the host asks for (Retry-After, in seconds), capped at
// maxRetryAfter.
func (c *Client) getJSON(ctx context.Context, u string, v any) error {
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return err
		}
		c.setAuth(req)
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return err
		}
		if resp.StatusCode == http.StatusTooManyRequests && attempt == 0 {
			wait := retryAfter(resp.Header.Get("Retry-After"))
			resp.Body.Close()
			select {
			case <-time.After(wait):
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			if resp.StatusCode == http.StatusTooManyRequests {
				return fmt.Errorf("Hugging Face is limiting requests from this address (HTTP 429); try again in a few minutes")
			}
			return fmt.Errorf("HF API returned %d", resp.StatusCode)
		}
		return json.NewDecoder(resp.Body).Decode(v)
	}
}

func retryAfter(h string) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(h))
	if err != nil || secs < 0 {
		return time.Second
	}
	return min(time.Duration(secs)*time.Second, maxRetryAfter)
}
