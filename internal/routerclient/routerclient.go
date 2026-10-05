// Package routerclient talks to llama-server's router mode HTTP API:
// listing models and loading or unloading one by name. It is the single
// place that knows the wire format, so the process manager and the
// benchmark runner (which may not import process) cannot drift apart.
package routerclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// ModelStatus represents the state of a model in the router.
type ModelStatus struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Aliases []string `json:"aliases"`
	Status  struct {
		Value string `json:"value"` // "loaded", "loading", "unloaded"
	} `json:"status"`
}

// IsLoaded reports whether the model holds memory: loaded, or on its way.
func (m ModelStatus) IsLoaded() bool {
	return m.Status.Value == "loaded" || m.Status.Value == "loading"
}

// listClient bounds GET /models, which answers at once.
var listClient = &http.Client{Timeout: 5 * time.Second}

// List returns every model the router knows, with its status. The router
// answers {"data": [...], "object": "list"}.
func List(ctx context.Context, baseURL string) ([]ModelStatus, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := listClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody+1))
		return nil, fmt.Errorf("list models: HTTP %d: %s", resp.StatusCode, ErrorSnippet(body))
	}
	var result struct {
		Data []ModelStatus `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	return result.Data, nil
}

// Load asks the router to load a model. The router starts the load and
// answers at once, so success means the load has begun, not that the model
// is ready. A model that is already loaded or loading counts as success:
// the router answers that case with HTTP 400 "model is already running".
func Load(ctx context.Context, baseURL, name string) error {
	status, body, err := post(ctx, baseURL+"/models/load", name)
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		return nil
	}
	if status == http.StatusBadRequest {
		msg := strings.ToLower(body)
		if strings.Contains(msg, "already loaded") || strings.Contains(msg, "already running") {
			return nil
		}
	}
	return fmt.Errorf("HTTP %d: %s", status, ErrorSnippet([]byte(body)))
}

// Unload asks the router to unload a model.
//
// Load and Unload errors do not name the model; callers add that context.
func Unload(ctx context.Context, baseURL, name string) error {
	status, body, err := post(ctx, baseURL+"/models/unload", name)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", status, ErrorSnippet([]byte(body)))
	}
	return nil
}

// maxErrorBody is how much of a failed response's body goes into an error
// message. Errors are logged and shown in the UI, and an upstream error
// page can be large.
const maxErrorBody = 2048

// ErrorSnippet is body cut to maxErrorBody bytes for an error message,
// on a character boundary, with a note of how much was left out.
func ErrorSnippet(body []byte) string {
	if len(body) <= maxErrorBody {
		return string(body)
	}
	cut := maxErrorBody
	for cut > 0 && !utf8.RuneStart(body[cut]) {
		cut--
	}
	return fmt.Sprintf("%s… (%d more bytes)", body[:cut], len(body)-cut)
}

// post sends {"model": name} to url and returns the status and body.
func post(ctx context.Context, url, name string) (int, string, error) {
	payload, _ := json.Marshal(map[string]string{"model": name})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	// Load and unload answer with a short JSON object; the limit only
	// matters for an unexpected error page.
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	return resp.StatusCode, string(body), nil
}
