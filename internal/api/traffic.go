package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// The live traffic viewer (Service page) shows the model requests that
// go through the proxy, and their responses as they arrive. It records
// nothing while no one has the viewer open: the tap is idle, nothing is
// kept and nothing is written to disk. Headers are never recorded, so an
// API key never reaches the viewer.

// trafficBodyLimit caps the request body and the response one exchange
// sends to the viewer. Past it, the viewer is told the rest was cut.
const trafficBodyLimit = 1 << 20

// trafficChanCap is how many events a viewer can fall behind by before
// it misses some. A streamed answer sends one event every 50 ms or so
// per request, so this is minutes of room.
const trafficChanCap = 4096

// trafficTap fans proxied requests out to the open viewers. The zero
// value is ready to use.
type trafficTap struct {
	once    sync.Once
	b       *broadcast.Broadcaster[string]
	viewers atomic.Int32
	nextID  atomic.Uint64
}

func (t *trafficTap) broadcaster() *broadcast.Broadcaster[string] {
	t.once.Do(func() { t.b = broadcast.New[string](0, trafficChanCap) })
	return t.b
}

// subscribe adds a viewer. The tap records while it has one or more.
func (t *trafficTap) subscribe() (chan string, func()) {
	ch := t.broadcaster().Subscribe()
	t.viewers.Add(1)
	return ch, func() {
		t.viewers.Add(-1)
		t.broadcaster().Unsubscribe(ch)
	}
}

func (t *trafficTap) active() bool { return t.viewers.Load() > 0 }

// trafficEvent is one message to the viewer. Every event names the
// exchange (request and response) it belongs to, so the viewer keeps
// parallel requests apart.
type trafficEvent struct {
	Type string `json:"type"` // start, request, response, chunk, cut, end
	ID   uint64 `json:"id"`

	// start
	Time   string `json:"time,omitempty"`
	Method string `json:"method,omitempty"`
	Path   string `json:"path,omitempty"`

	// request: where the request goes ("router" or "fork:<instance>"),
	// the model it names and its body as sent on. Note says what the body
	// leaves out.
	Target string `json:"target,omitempty"`
	Model  string `json:"model,omitempty"`
	Body   string `json:"body,omitempty"`
	Note   string `json:"note,omitempty"`

	// response and end
	Status      int    `json:"status,omitempty"`
	ContentType string `json:"content_type,omitempty"`
	MS          int64  `json:"ms,omitempty"`

	// chunk
	Data string `json:"data,omitempty"`
}

func (t *trafficTap) send(ev trafficEvent) {
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	t.broadcaster().Broadcast(string(b))
}

// trafficExchange records one proxied request. Its methods do nothing on
// a nil exchange, which is what beginTraffic returns while no viewer is
// open.
type trafficExchange struct {
	tap   *trafficTap
	id    uint64
	start time.Time

	mu     sync.Mutex
	status int
	sent   int
	cut    bool
	ended  bool
}

// beginTraffic starts recording r when a viewer is open. Only POSTs are
// recorded: they are the model requests. GETs (model lists, health
// checks, the realtime WebSocket) would bury them. The returned writer
// is w with the response teed to the viewer; use it for everything sent
// back, and call end when the handler is done.
func (s *Server) beginTraffic(w http.ResponseWriter, r *http.Request) (http.ResponseWriter, *trafficExchange) {
	if r.Method != http.MethodPost || !s.traffic.active() {
		return w, nil
	}
	x := &trafficExchange{tap: &s.traffic, id: s.traffic.nextID.Add(1), start: time.Now()}
	x.tap.send(trafficEvent{
		Type: "start", ID: x.id,
		Time:   x.start.Format(time.RFC3339Nano),
		Method: r.Method, Path: r.URL.Path,
	})
	return &trafficWriter{ResponseWriter: w, x: x}, x
}

// request records the body sent on and where it goes. Called again when
// the body changes (sampling defaults, system prompt); the viewer shows
// the last one.
func (x *trafficExchange) request(target string, body []byte) {
	if x == nil {
		return
	}
	shown, note := trafficBody(body)
	x.tap.send(trafficEvent{
		Type: "request", ID: x.id,
		Target: target, Model: bodyModel(body),
		Body: shown, Note: note,
	})
}

// requestNote records a request whose body is not shown (a multipart
// form), with the model it names.
func (x *trafficExchange) requestNote(target, model, note string) {
	if x == nil {
		return
	}
	x.tap.send(trafficEvent{Type: "request", ID: x.id, Target: target, Model: model, Note: note})
}

func (x *trafficExchange) response(status int, contentType string) {
	x.mu.Lock()
	if x.status != 0 {
		x.mu.Unlock()
		return
	}
	x.status = status
	x.mu.Unlock()
	x.tap.send(trafficEvent{Type: "response", ID: x.id, Status: status, ContentType: contentType})
}

func (x *trafficExchange) chunk(p []byte) {
	x.mu.Lock()
	if x.cut || len(p) == 0 {
		x.mu.Unlock()
		return
	}
	room := trafficBodyLimit - x.sent
	cut := len(p) > room
	if cut {
		p = p[:room]
		x.cut = true
	}
	x.sent += len(p)
	x.mu.Unlock()
	if len(p) > 0 {
		x.tap.send(trafficEvent{Type: "chunk", ID: x.id, Data: string(p)})
	}
	if cut {
		x.tap.send(trafficEvent{Type: "cut", ID: x.id})
	}
}

// end closes the exchange. A handler that wrote nothing answered 200.
func (x *trafficExchange) end() {
	if x == nil {
		return
	}
	x.mu.Lock()
	if x.ended {
		x.mu.Unlock()
		return
	}
	x.ended = true
	status := x.status
	x.mu.Unlock()
	if status == 0 {
		status = http.StatusOK
	}
	x.tap.send(trafficEvent{Type: "end", ID: x.id, Status: status, MS: time.Since(x.start).Milliseconds()})
}

// trafficWriter tees a response to its exchange. Unwrap lets the reverse
// proxy flush through it, so streamed tokens still reach the client as
// they come.
type trafficWriter struct {
	http.ResponseWriter
	x *trafficExchange
}

func (w *trafficWriter) WriteHeader(status int) {
	if status >= 200 {
		w.x.response(status, w.Header().Get("Content-Type"))
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *trafficWriter) Write(p []byte) (int, error) {
	w.x.response(http.StatusOK, w.Header().Get("Content-Type"))
	n, err := w.ResponseWriter.Write(p)
	w.x.chunk(p[:n])
	return n, err
}

func (w *trafficWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *trafficWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// Base64 data in a request (images, audio) is shortened before it goes
// to the viewer: it is large and says nothing to a person. A data URI
// keeps its "data:...;base64," prefix; any other string is shortened
// when it is long and all base64.
const (
	trafficDataURIMin = 256
	trafficBase64Min  = 4096
)

// trafficBody is a request body as the viewer shows it, and a note that
// says what was left out.
func trafficBody(body []byte) (string, string) {
	var notes []string
	short := shortenBase64(body)
	if len(short) != len(body) {
		notes = append(notes, "base64 data shortened")
	}
	if len(short) > trafficBodyLimit {
		short = short[:trafficBodyLimit]
		notes = append(notes, fmt.Sprintf("body cut at %d KiB", trafficBodyLimit>>10))
	}
	return string(short), strings.Join(notes, "; ")
}

// shortenBase64 replaces the base64 strings in a JSON body with a note of
// their size. Base64 has no characters JSON escapes, so a string with an
// escape in it is never one.
func shortenBase64(body []byte) []byte {
	var out []byte // nil until something is shortened
	last := 0
	for i := 0; i < len(body); i++ {
		if body[i] != '"' {
			continue
		}
		// The string runs to the next quote that is not escaped.
		j := i + 1
		escaped := false
		for j < len(body) && body[j] != '"' {
			if body[j] == '\\' {
				escaped = true
				j++
			}
			j++
		}
		if j >= len(body) {
			break
		}
		if !escaped {
			if repl := shortBase64String(body[i+1 : j]); repl != nil {
				out = append(out, body[last:i+1]...)
				out = append(out, repl...)
				last = j
			}
		}
		i = j
	}
	if out == nil {
		return body
	}
	return append(out, body[last:]...)
}

// shortBase64String is the shortened form of s, or nil to keep it.
func shortBase64String(s []byte) []byte {
	if bytes.HasPrefix(s, []byte("data:")) {
		k := bytes.Index(s, []byte(";base64,"))
		if k < 0 || k > 200 {
			return nil
		}
		prefix := k + len(";base64,")
		if data := s[prefix:]; len(data) >= trafficDataURIMin && isBase64(data) {
			return fmt.Appendf(nil, "%s<%d bytes cut>", s[:prefix], len(data))
		}
		return nil
	}
	if len(s) >= trafficBase64Min && isBase64(s) {
		return fmt.Appendf(nil, "<base64, %d bytes cut>", len(s))
	}
	return nil
}

func isBase64(s []byte) bool {
	for _, c := range s {
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '+', c == '/', c == '=', c == '-', c == '_':
		default:
			return false
		}
	}
	return true
}

// handleTrafficStream sends the proxied requests to the viewer as server
// sent events, one JSON object per event, for as long as it is open.
func (s *Server) handleTrafficStream(w http.ResponseWriter, r *http.Request) {
	ch, done := s.traffic.subscribe()
	defer done()
	StreamLines(w, r.Context(), ch, "")
}

// renderTimeout caps a "Show rendered prompt" request to the engine.
const renderTimeout = 30 * time.Second

// handleTrafficRender asks the engine a recorded request went to for the
// prompt its chat template makes of that request (llama-server's
// /apply-template). The router is asked only when the model is loaded:
// the router would load it otherwise, past toolchest's room checks.
func (s *Server) handleTrafficRender(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Target string          `json:"target"`
		Body   json.RawMessage `json:"body"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&in); err != nil {
		renderError(w, http.StatusBadRequest, "The request could not be read.")
		return
	}
	var req map[string]any
	if json.Unmarshal(in.Body, &req) != nil {
		renderError(w, http.StatusBadRequest, "The recorded request body is not JSON.")
		return
	}
	if _, ok := req["messages"]; !ok {
		renderError(w, http.StatusBadRequest, "Only chat requests (with messages) have a rendered prompt.")
		return
	}
	delete(req, "stream")
	delete(req, "stream_options")
	body, _ := json.Marshal(req)

	var base string
	switch {
	case in.Target == "router":
		id := s.requestModelID(body)
		if id == "" || !s.modelLoaded(id) {
			renderError(w, http.StatusConflict, "The model is not loaded. Load it first: the prompt comes from the loaded model.")
			return
		}
		base = s.routerBaseURL()
	case strings.HasPrefix(in.Target, "fork:"):
		inst, err := s.forks.Instance(strings.TrimPrefix(in.Target, "fork:"))
		if err != nil {
			renderError(w, http.StatusNotFound, "That fork instance no longer exists.")
			return
		}
		if s.forkMgr.Status(inst.ID).State != process.StateRunning {
			renderError(w, http.StatusConflict, "The fork instance is not running. Start it first.")
			return
		}
		base = inst.BaseURL()
	default:
		renderError(w, http.StatusBadRequest, "Unknown engine.")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), renderTimeout)
	defer cancel()
	up, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/apply-template", bytes.NewReader(body))
	if err != nil {
		renderError(w, http.StatusInternalServerError, err.Error())
		return
	}
	up.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(up)
	if err != nil {
		renderError(w, http.StatusBadGateway, "The engine did not answer: "+err.Error())
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	switch {
	case resp.StatusCode == http.StatusNotFound:
		renderError(w, http.StatusNotImplemented, "This engine does not have /apply-template, so it cannot show the rendered prompt.")
		return
	case resp.StatusCode != http.StatusOK:
		renderError(w, http.StatusBadGateway, fmt.Sprintf("The engine answered %d: %s", resp.StatusCode, strings.TrimSpace(string(raw))))
		return
	}
	var out struct {
		Prompt *string `json:"prompt"`
	}
	if json.Unmarshal(raw, &out) != nil || out.Prompt == nil {
		renderError(w, http.StatusBadGateway, "The engine's answer has no prompt.")
		return
	}
	respondJSONStatus(w, http.StatusOK, map[string]string{"prompt": *out.Prompt})
}

func renderError(w http.ResponseWriter, status int, msg string) {
	respondJSONStatus(w, status, map[string]string{"error": msg})
}
