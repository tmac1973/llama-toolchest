// Package llmcall asks a locally served model a question and gets back
// JSON that matches a schema. It is how llama-toolchest uses a model for
// its own work — reading a model card for autoconfigure — as opposed to
// serving one to clients.
//
// The package does not know how models are registered or how the router
// is started; a Backend supplies that, so the client can be tested
// against a fake router.
package llmcall

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Message is one chat message.
type Message struct {
	Role    string `json:"role"` // "system" | "user" | "assistant"
	Content string `json:"content"`
}

// Thinking says how to turn a model's reasoning mode off, as detected from
// its chat template (models.ReasoningCapability): Toggle is
// "chat_template_kwargs", "reasoning_effort" or "none".
type Thinking struct {
	Toggle string
	Kwarg  string
}

// Target is a model ready to answer: the name the router serves it under,
// and how to turn its thinking off.
type Target struct {
	RouterName string
	Thinking   Thinking
}

// Backend is what the client needs from the server around it.
type Backend interface {
	// Busy returns a plain-language reason the GPU cannot be used for a
	// request now (a benchmark is running), or "".
	Busy() string
	// Prepare makes the model with registry ID modelID loaded and ready.
	Prepare(ctx context.Context, modelID string) (Target, error)
	// RouterURL is the base URL of the running router.
	RouterURL() string
}

// Client sends structured requests through a Backend.
type Client struct {
	Backend Backend
	HTTP    *http.Client
	// MaxTokens bounds the answer. Zero uses DefaultMaxTokens.
	MaxTokens int
}

// DefaultMaxTokens is the answer budget when Client.MaxTokens is zero.
const DefaultMaxTokens = 2048

// ErrBusy is returned (wrapped, with the reason) when Backend.Busy
// reports the GPU in use.
var ErrBusy = errors.New("the GPU is in use")

// JSON asks the model with registry ID modelID and decodes its answer into
// out. The answer is constrained to schema (a JSON Schema object) by
// llama-server's grammar support, the temperature is 0, and the model's
// thinking is turned off: this is extraction, not reasoning, and a
// reasoning pass would spend the answer budget before any JSON.
//
// If the answer still does not decode into out, the model is asked once
// more with the error, and then JSON gives up.
func (c *Client) JSON(ctx context.Context, modelID, schemaName string, schema any, messages []Message, out any) error {
	if reason := c.Backend.Busy(); reason != "" {
		return fmt.Errorf("%w: %s", ErrBusy, reason)
	}
	target, err := c.Backend.Prepare(ctx, modelID)
	if err != nil {
		return err
	}

	content, err := c.complete(ctx, target, schemaName, schema, messages)
	if err != nil {
		return err
	}
	decodeErr := decodeStrict(content, out)
	if decodeErr == nil {
		return nil
	}

	retry := append(append([]Message(nil), messages...),
		Message{Role: "assistant", Content: content},
		Message{Role: "user", Content: "That answer could not be read: " + decodeErr.Error() +
			". Reply again with only the JSON object, following the schema exactly."})
	content, err = c.complete(ctx, target, schemaName, schema, retry)
	if err != nil {
		return err
	}
	if err := decodeStrict(content, out); err != nil {
		return fmt.Errorf("the helper model did not return usable JSON after two tries: %w", err)
	}
	return nil
}

// decodeStrict decodes content into out, refusing fields the schema does
// not have: a stray key means the model was not following the schema.
func decodeStrict(content string, out any) error {
	dec := json.NewDecoder(strings.NewReader(strings.TrimSpace(content)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	if dec.More() {
		return errors.New("extra text after the JSON object")
	}
	return nil
}

func (c *Client) complete(ctx context.Context, target Target, schemaName string, schema any, messages []Message) (string, error) {
	maxTokens := c.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	payload := map[string]any{
		"model":       target.RouterName,
		"messages":    messages,
		"temperature": 0,
		"max_tokens":  maxTokens,
		"stream":      false,
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   schemaName,
				"strict": true,
				"schema": schema,
			},
		},
	}
	thinkingOff(target.Thinking, payload)

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(c.Backend.RouterURL(), "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("asking the helper model: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("the helper model answered HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("reading the helper model's answer: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("the helper model returned no answer")
	}
	ch := parsed.Choices[0]
	if ch.FinishReason == "length" {
		return "", fmt.Errorf("the helper model's answer was cut off at %d tokens", maxTokens)
	}
	return ch.Message.Content, nil
}

// thinkingOff adds whatever the model needs to skip its reasoning pass.
func thinkingOff(t Thinking, payload map[string]any) {
	switch t.Toggle {
	case "chat_template_kwargs":
		if t.Kwarg != "" {
			payload["chat_template_kwargs"] = map[string]any{t.Kwarg: false}
		}
	case "reasoning_effort":
		payload["reasoning_effort"] = "none"
	}
}
