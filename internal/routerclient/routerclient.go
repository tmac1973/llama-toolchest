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
	"net/url"
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

// Slot is one of a loaded model's conversation slots, as GET /slots
// reports it. PromptTokens is how many tokens its prompt cache holds;
// zero for a slot that has not served a request yet.
type Slot struct {
	ID           int  `json:"id"`
	Processing   bool `json:"is_processing"`
	PromptTokens int  `json:"n_prompt_tokens"`
}

// slotClient bounds the slot calls. A save or restore writes or reads a
// file of up to several GB, and waits for a slot that is busy.
var slotClient = &http.Client{Timeout: 2 * time.Minute}

// Slots lists the conversation slots of a loaded model. The router
// passes GET /slots?model=NAME on to the model's own server.
func Slots(ctx context.Context, baseURL, name string) ([]Slot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/slots?model="+url.QueryEscape(name), nil)
	if err != nil {
		return nil, err
	}
	resp, err := slotClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list slots: HTTP %d: %s", resp.StatusCode, ErrorSnippet(body))
	}
	var slots []Slot
	if err := json.Unmarshal(body, &slots); err != nil {
		return nil, fmt.Errorf("list slots: %w", err)
	}
	return slots, nil
}

// SlotResult is the answer to a slot save or restore: how many tokens
// and bytes it moved.
type SlotResult struct {
	Tokens int
	Bytes  int
}

// SaveSlot writes a slot's prompt cache to filename in the model
// server's --slot-save-path directory.
func SaveSlot(ctx context.Context, baseURL, name string, id int, filename string) (SlotResult, error) {
	return slotAction(ctx, baseURL, name, id, "save", filename)
}

// RestoreSlot reads a slot's prompt cache back from filename.
func RestoreSlot(ctx context.Context, baseURL, name string, id int, filename string) (SlotResult, error) {
	return slotAction(ctx, baseURL, name, id, "restore", filename)
}

func slotAction(ctx context.Context, baseURL, name string, id int, action, filename string) (SlotResult, error) {
	payload, _ := json.Marshal(map[string]string{"model": name, "filename": filename})
	u := fmt.Sprintf("%s/slots/%d?action=%s", baseURL, id, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return SlotResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := slotClient.Do(req)
	if err != nil {
		return SlotResult{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode != http.StatusOK {
		return SlotResult{}, fmt.Errorf("slot %d %s: HTTP %d: %s", id, action, resp.StatusCode, ErrorSnippet(body))
	}
	var r struct {
		Saved    int `json:"n_saved"`
		Written  int `json:"n_written"`
		Restored int `json:"n_restored"`
		Read     int `json:"n_read"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return SlotResult{}, fmt.Errorf("slot %d %s: %w", id, action, err)
	}
	return SlotResult{Tokens: r.Saved + r.Restored, Bytes: r.Written + r.Read}, nil
}
