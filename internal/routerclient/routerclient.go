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
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("list models: HTTP %d: %s", resp.StatusCode, string(body))
	}
	var result struct {
		Data []ModelStatus `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	return result.Data, nil
}

// Load asks the router to load a model. The router blocks until the model
// is ready, which can take minutes, so the only time limit is ctx. A model
// that is already loaded counts as success: the router answers that case
// with HTTP 400 "model is already running".
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
	return fmt.Errorf("HTTP %d: %s", status, body)
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
		return fmt.Errorf("HTTP %d: %s", status, body)
	}
	return nil
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
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body), nil
}
