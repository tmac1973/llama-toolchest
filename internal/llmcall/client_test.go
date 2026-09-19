package llmcall

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type fakeBackend struct {
	url      string
	busy     string
	prepared []string
}

func (f *fakeBackend) Busy() string      { return f.busy }
func (f *fakeBackend) RouterURL() string { return f.url }
func (f *fakeBackend) Prepare(_ context.Context, id string) (Target, error) {
	f.prepared = append(f.prepared, id)
	return Target{RouterName: "helper-router", Thinking: Thinking{Toggle: "chat_template_kwargs", Kwarg: "enable_thinking"}}, nil
}

// fakeRouter answers each chat request with the next canned content and
// records the request bodies.
type fakeRouter struct {
	mu      sync.Mutex
	answers []string
	bodies  []map[string]any
}

func (r *fakeRouter) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var body map[string]any
	json.NewDecoder(req.Body).Decode(&body)
	r.mu.Lock()
	r.bodies = append(r.bodies, body)
	answer := r.answers[0]
	if len(r.answers) > 1 {
		r.answers = r.answers[1:]
	}
	r.mu.Unlock()
	json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"content": answer}, "finish_reason": "stop"}},
	})
}

type answer struct {
	Temperature float64 `json:"temperature"`
	Reason      string  `json:"reason"`
}

var schema = map[string]any{"type": "object", "properties": map[string]any{
	"temperature": map[string]any{"type": "number"}, "reason": map[string]any{"type": "string"},
}}

func newClient(t *testing.T, answers ...string) (*Client, *fakeRouter, *fakeBackend) {
	t.Helper()
	router := &fakeRouter{answers: answers}
	srv := httptest.NewServer(router)
	t.Cleanup(srv.Close)
	b := &fakeBackend{url: srv.URL}
	return &Client{Backend: b}, router, b
}

func TestJSONSendsSchemaAndTurnsThinkingOff(t *testing.T) {
	c, router, b := newClient(t, `{"temperature": 0.6, "reason": "card says 0.6"}`)
	var got answer
	err := c.JSON(context.Background(), "helper-id", "advice", schema,
		[]Message{{Role: "user", Content: "read this"}}, &got)
	if err != nil {
		t.Fatal(err)
	}
	if got.Temperature != 0.6 || got.Reason != "card says 0.6" {
		t.Errorf("decoded %+v", got)
	}
	if len(b.prepared) != 1 || b.prepared[0] != "helper-id" {
		t.Errorf("Prepare calls = %v", b.prepared)
	}
	body := router.bodies[0]
	if body["model"] != "helper-router" || body["temperature"] != float64(0) {
		t.Errorf("model/temperature = %v/%v", body["model"], body["temperature"])
	}
	rf, _ := body["response_format"].(map[string]any)
	js, _ := rf["json_schema"].(map[string]any)
	if rf["type"] != "json_schema" || js["name"] != "advice" || js["strict"] != true || js["schema"] == nil {
		t.Errorf("response_format = %v", rf)
	}
	kw, _ := body["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != false {
		t.Errorf("thinking not turned off: %v", body["chat_template_kwargs"])
	}
}

func TestJSONRetriesOnceThenGivesUp(t *testing.T) {
	c, router, _ := newClient(t, `not json`, `{"temperature": 1, "reason": "ok"}`)
	var got answer
	if err := c.JSON(context.Background(), "h", "advice", schema, []Message{{Role: "user", Content: "q"}}, &got); err != nil {
		t.Fatalf("second answer was valid, got %v", err)
	}
	if len(router.bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(router.bodies))
	}
	msgs, _ := router.bodies[1]["messages"].([]any)
	last, _ := msgs[len(msgs)-1].(map[string]any)
	if !strings.Contains(last["content"].(string), "could not be read") {
		t.Errorf("retry does not explain the error: %v", last)
	}

	c, router, _ = newClient(t, `{"temperature": 1, "unexpected": true}`)
	if err := c.JSON(context.Background(), "h", "advice", schema, []Message{{Role: "user", Content: "q"}}, &got); err == nil {
		t.Error("an answer with a field outside the schema was accepted twice")
	}
	if len(router.bodies) != 2 {
		t.Errorf("requests = %d, want exactly 2", len(router.bodies))
	}
}

func TestJSONRefusesWhenBusy(t *testing.T) {
	c, router, b := newClient(t, `{}`)
	b.busy = "A benchmark is running."
	var got answer
	err := c.JSON(context.Background(), "h", "advice", schema, nil, &got)
	if !errors.Is(err, ErrBusy) || !strings.Contains(err.Error(), "benchmark") {
		t.Errorf("err = %v, want ErrBusy with the reason", err)
	}
	if len(router.bodies) != 0 || len(b.prepared) != 0 {
		t.Error("a busy backend still loaded or asked the model")
	}
}

func TestJSONReportsCutOffAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": `{"temp`}, "finish_reason": "length"}},
		})
	}))
	defer srv.Close()
	c := &Client{Backend: &fakeBackend{url: srv.URL}}
	var got answer
	if err := c.JSON(context.Background(), "h", "advice", schema, nil, &got); err == nil || !strings.Contains(err.Error(), "cut off") {
		t.Errorf("err = %v, want a cut-off error", err)
	}
}
