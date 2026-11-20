package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// trafficEvents collects what the viewer got, waiting a moment for the
// last events of a request.
func trafficEvents(t *testing.T, ch chan string) []trafficEvent {
	t.Helper()
	var out []trafficEvent
	for {
		select {
		case line := <-ch:
			var ev trafficEvent
			if err := json.Unmarshal([]byte(line), &ev); err != nil {
				t.Fatalf("event %q: %v", line, err)
			}
			out = append(out, ev)
		case <-time.After(100 * time.Millisecond):
			return out
		}
	}
}

func eventsOf(evs []trafficEvent, typ string) []trafficEvent {
	var out []trafficEvent
	for _, ev := range evs {
		if ev.Type == typ {
			out = append(out, ev)
		}
	}
	return out
}

// With no viewer open, nothing is recorded and the response writer is
// the client's own.
func TestTrafficIdleWithoutViewer(t *testing.T) {
	s, _ := runningFork(t)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if got, x := s.beginTraffic(w, r); got != w || x != nil {
		t.Fatalf("idle tap wrapped the writer: %T %v", got, x)
	}
	// A viewer that opens and closes leaves the tap idle again.
	_, done := s.traffic.subscribe()
	done()
	if s.traffic.active() {
		t.Fatal("tap still active after the viewer closed")
	}
}

// A streamed chat to a fork instance: the viewer gets the request as
// sent on (fork defaults added), the stream as it came, and the end, all
// under one exchange ID. The API key never reaches it.
func TestTrafficRecordsForkStream(t *testing.T) {
	s, _ := runningFork(t)
	ch, done := s.traffic.subscribe()
	defer done()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen-next","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-very-secret")
	rec := httptest.NewRecorder()
	s.newProxyHandler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("client got %d %q", rec.Code, rec.Body)
	}

	evs := trafficEvents(t, ch)
	for _, ev := range evs {
		if ev.ID != evs[0].ID {
			t.Fatalf("events from more than one exchange: %+v", evs)
		}
		if b, _ := json.Marshal(ev); strings.Contains(string(b), "sk-very-secret") {
			t.Fatalf("the API key reached the viewer: %s", b)
		}
	}
	if evs[0].Type != "start" || evs[0].Path != "/v1/chat/completions" || evs[0].Method != "POST" {
		t.Errorf("first event = %+v", evs[0])
	}
	reqs := eventsOf(evs, "request")
	if len(reqs) == 0 {
		t.Fatal("no request event")
	}
	last := reqs[len(reqs)-1]
	if last.Target != "fork:qwen-next" || last.Model != "qwen-next" || !strings.Contains(last.Body, `"top_k":20`) {
		t.Errorf("request = %+v; want the fork target and its sampling defaults", last)
	}
	if r := eventsOf(evs, "response"); len(r) != 1 || r[0].Status != 200 || !strings.Contains(r[0].ContentType, "event-stream") {
		t.Errorf("response events = %+v", r)
	}
	var streamed strings.Builder
	for _, c := range eventsOf(evs, "chunk") {
		streamed.WriteString(c.Data)
	}
	if streamed.String() != rec.Body.String() {
		t.Errorf("viewer got %q, client got %q", streamed.String(), rec.Body.String())
	}
	if e := evs[len(evs)-1]; e.Type != "end" || e.Status != 200 {
		t.Errorf("last event = %+v", e)
	}
}

// An answer from toolchest itself (the router is not running) is
// recorded too, with its status, and GETs are not recorded.
func TestTrafficRecordsProxyErrorsNotGets(t *testing.T) {
	s, _ := runningFork(t)
	ch, done := s.traffic.subscribe()
	defer done()

	rec := proxyPost(s, "/v1/chat/completions", `{"model":"some-llm","messages":[]}`)
	evs := trafficEvents(t, ch)
	if r := eventsOf(evs, "request"); len(r) == 0 || r[0].Target != "router" || r[0].Model != "some-llm" {
		t.Errorf("request events = %+v", r)
	}
	if e := eventsOf(evs, "end"); len(e) != 1 || e[0].Status != rec.Code || rec.Code < 400 {
		t.Errorf("end = %+v, client got %d", e, rec.Code)
	}

	get := httptest.NewRequest("GET", "/v1/slots", nil)
	s.newProxyHandler().ServeHTTP(httptest.NewRecorder(), get)
	if evs := trafficEvents(t, ch); len(evs) != 0 {
		t.Errorf("a GET was recorded: %+v", evs)
	}
}

// Responses past the limit are cut, once, with a cut event.
func TestTrafficCutsLongResponses(t *testing.T) {
	var tap trafficTap
	ch, done := tap.subscribe()
	defer done()
	x := &trafficExchange{tap: &tap, id: 1, start: time.Now()}
	x.chunk(make([]byte, trafficBodyLimit-10))
	x.chunk([]byte(strings.Repeat("a", 100)))
	x.chunk([]byte("more"))
	var sent, cuts int
	for _, ev := range trafficEvents(t, ch) {
		sent += len(ev.Data)
		if ev.Type == "cut" {
			cuts++
		}
	}
	if sent != trafficBodyLimit || cuts != 1 {
		t.Errorf("sent %d bytes and %d cut events; want %d and 1", sent, cuts, trafficBodyLimit)
	}
}

func TestTrafficBodyShortensBase64(t *testing.T) {
	img := strings.Repeat("iVBORw0KGgo", 100)
	audio := strings.Repeat("QUJD", 2000)
	body := fmt.Sprintf(`{"messages":[{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},`+
		`{"type":"input_audio","input_audio":{"data":"%s","format":"wav"}},{"type":"text","text":"what is this?"}]}]}`, img, audio)
	got, note := trafficBody([]byte(body))
	if !json.Valid([]byte(got)) {
		t.Fatalf("shortened body is not JSON: %s", got)
	}
	if strings.Contains(got, img) || strings.Contains(got, audio) {
		t.Errorf("base64 kept: %s", got)
	}
	for _, want := range []string{`"data:image/png;base64,<1100 bytes cut>"`, `"<base64, 8000 bytes cut>"`, "what is this?"} {
		if !strings.Contains(got, want) {
			t.Errorf("body lacks %q: %s", want, got)
		}
	}
	if note != "base64 data shortened" {
		t.Errorf("note = %q", note)
	}

	// Plain text, however long, is kept until the size limit.
	long := `{"prompt":"` + strings.Repeat("word ", trafficBodyLimit/5+10) + `"}`
	got, note = trafficBody([]byte(long))
	if len(got) != trafficBodyLimit || !strings.Contains(note, "cut at 1024 KiB") {
		t.Errorf("long body: %d bytes, note %q", len(got), note)
	}
}

func renderPost(s *Server, body string) (int, map[string]string) {
	req := httptest.NewRequest("POST", "/api/service/traffic/render", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleTrafficRender(rec, req)
	var out map[string]string
	json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestTrafficRenderAsksTheEngine(t *testing.T) {
	s, _ := runningFork(t)
	code, out := renderPost(s, `{"target":"fork:qwen-next","body":{"model":"qwen-next","stream":true,"messages":[{"role":"user","content":"hi"}]}}`)
	// The stream flag is dropped: /apply-template answers in one piece.
	if code != http.StatusOK || out["prompt"] != "<|hi|>" {
		t.Errorf("render = %d %v", code, out)
	}

	cases := []struct {
		name, body string
		code       int
	}{
		{"not a chat", `{"target":"fork:qwen-next","body":{"model":"qwen-next","prompt":"x"}}`, http.StatusBadRequest},
		{"no such fork", `{"target":"fork:gone","body":{"messages":[]}}`, http.StatusNotFound},
		{"router model not loaded", `{"target":"router","body":{"model":"some-llm","messages":[]}}`, http.StatusConflict},
		{"unknown engine", `{"target":"elsewhere","body":{"messages":[]}}`, http.StatusBadRequest},
	}
	for _, c := range cases {
		code, out := renderPost(s, c.body)
		if code != c.code || out["error"] == "" {
			t.Errorf("%s: %d %v; want %d with an error", c.name, code, out, c.code)
		}
	}
}

// An engine without /apply-template says so.
func TestTrafficRenderWithoutApplyTemplate(t *testing.T) {
	s, fake := runningFork(t)
	old := fake.srv.Config.Handler
	fake.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/apply-template" {
			http.NotFound(w, r)
			return
		}
		old.ServeHTTP(w, r)
	})
	code, out := renderPost(s, `{"target":"fork:qwen-next","body":{"messages":[{"role":"user","content":"hi"}]}}`)
	if code != http.StatusNotImplemented || !strings.Contains(out["error"], "/apply-template") {
		t.Errorf("render = %d %v", code, out)
	}
}
