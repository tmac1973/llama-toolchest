package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmac1973/llama-toolchest/internal/routerclient"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"testing/iotest"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// proxyModelID is the registry ID of the model the proxy tests register.
// Its public name is "org-Model.Q4_K_M".
const proxyModelID = "org--Model-GGUF--Q4_K_M"

// addProxyModel registers one model with the given config.
func addProxyModel(t *testing.T, s *Server, cfg models.ModelConfig) *models.Model {
	t.Helper()
	m := &models.Model{
		ID: proxyModelID, ModelID: "org/Model-GGUF", Quant: "Q4_K_M",
		Filename: "model.gguf", FilePath: filepath.Join(t.TempDir(), "model.gguf"),
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig(m.ID, &cfg); err != nil {
		t.Fatal(err)
	}
	return m
}

// The sampling settings saved for a model are defaults: they fill in what
// the client left out and never replace what the client sent. A client
// that asks for temperature 0.2 must get 0.2, not the saved 0.7.
func TestInjectSamplingDefaultsKeepsClientValues(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{
		Enabled: true, Temperature: ptr(0.7), TopK: ptr(20), MinP: ptr(0.05),
	})

	body := []byte(`{"model":"` + proxyModelID + `","temperature":0.2,"min_p":null,"messages":[{"role":"user","content":"hi"}]}`)
	out := s.injectSamplingDefaults(body)

	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, out)
	}
	if got["temperature"] != 0.2 {
		t.Errorf("temperature = %v, want the client's 0.2", got["temperature"])
	}
	if got["top_k"] != float64(20) {
		t.Errorf("top_k = %v, want the saved default 20", got["top_k"])
	}
	// A key the client sent as null was still sent: it is the client's
	// choice and is left alone.
	if v, ok := got["min_p"]; !ok || v != nil {
		t.Errorf("min_p = %v (present %v), want the client's null", v, ok)
	}
	if _, ok := got["top_p"]; ok {
		t.Error("top_p was added although the model has no saved top_p")
	}
	msgs, _ := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("messages were not passed through: %v", got["messages"])
	}
}

// Clients name a model by whatever /v1/models lists (the public name) or
// by an alias, not only by the registry ID. The lookup used to accept only
// the registry ID, so most clients never got the saved defaults from the
// proxy (preset.ini still applied them on the router side).
func TestInjectSamplingDefaultsFindsAnyName(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{Enabled: true, Temperature: ptr(0.7), Aliases: []string{"chat"}})

	for _, name := range []string{proxyModelID, "org-Model.Q4_K_M", "chat"} {
		var got map[string]any
		if err := json.Unmarshal(s.injectSamplingDefaults([]byte(`{"model":"`+name+`"}`)), &got); err != nil {
			t.Fatal(err)
		}
		if got["temperature"] != 0.7 {
			t.Errorf("model %q: temperature = %v, want the saved 0.7", name, got["temperature"])
		}
	}
}

// Bodies the function cannot act on go upstream byte for byte, so the
// router produces its own error for a malformed request rather than one
// this proxy invented.
func TestInjectSamplingDefaultsLeavesOtherBodiesUntouched(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{Enabled: true, Temperature: ptr(0.7)})

	noOverrides := newTestServer(t)
	addProxyModel(t, noOverrides, models.ModelConfig{Enabled: true})

	cases := []struct {
		name string
		s    *Server
		body string
	}{
		{"not JSON", s, `model=` + proxyModelID},
		{"no model field", s, `{"messages":[]}`},
		{"empty model", s, `{"model":""}`},
		{"unknown model", s, `{"model":"nobody"}`},
		{"model with no sampling settings", noOverrides, `{"model":"` + proxyModelID + `" , "x":1}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := c.s.injectSamplingDefaults([]byte(c.body))
			if !bytes.Equal(out, []byte(c.body)) {
				t.Errorf("body changed:\n got %s\nwant %s", out, c.body)
			}
		})
	}
}

// Only a JSON body is read and rewritten by the proxy; anything else
// (multipart audio uploads) streams through. The check must accept the
// usual variants of the JSON media type and nothing else.
func TestIsJSONContentType(t *testing.T) {
	cases := map[string]bool{
		"application/json":                  true,
		"application/json; charset=utf-8":   true,
		"Application/JSON":                  true,
		"  application/json  ;charset=utf8": true,
		"":                                  false,
		"text/plain":                        false,
		"multipart/form-data; boundary=x":   false,
		"application/jsonl":                 false,
		"application/json-patch+json":       false,
	}
	for ct, want := range cases {
		if got := isJSONContentType(ct); got != want {
			t.Errorf("isJSONContentType(%q) = %v, want %v", ct, got, want)
		}
	}
}

// sseStream is a chat completion stream as llama.cpp sends it: the model
// name on the first chunk, timings only on the last one, then [DONE].
// The second timings chunk must not be reported again.
const sseStream = "data: {\"model\":\"m1\",\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n" +
	"data:{\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n" +
	": keep-alive comment\n\n" +
	"data: {\"choices\":[],\"timings\":{\"predicted_n\":2,\"predicted_per_second\":40.5}}\n\n" +
	"data: {\"choices\":[],\"timings\":{\"predicted_n\":99}}\n\n" +
	"data: [DONE]\n\n"

type timingCall struct {
	model   string
	timings map[string]any
}

// readThroughCapture reads r through the timing capture and returns what
// the client received and the capture calls.
func readThroughCapture(t *testing.T, r io.Reader) (string, chan timingCall, *sseTimingCapture) {
	t.Helper()
	calls := make(chan timingCall, 4)
	c := newSSETimingCapture(io.NopCloser(r), func(model string, timings map[string]any) {
		calls <- timingCall{model, timings}
	}, nil)
	out, err := io.ReadAll(c)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	return string(out), calls, c
}

// The capture sits between llama-server and the client, so it must hand
// every byte on unchanged, and still find the final timings when the
// network splits an event across reads — including one byte at a time.
func TestSSETimingCaptureFindsTimingsAcrossSplitReads(t *testing.T) {
	readers := map[string]func() io.Reader{
		"whole stream": func() io.Reader { return strings.NewReader(sseStream) },
		"one byte per read": func() io.Reader {
			return iotest.OneByteReader(strings.NewReader(sseStream))
		},
		"seven bytes per read": func() io.Reader {
			return &chunkReader{data: []byte(sseStream), size: 7}
		},
	}
	for name, mk := range readers {
		t.Run(name, func(t *testing.T) {
			out, calls, c := readThroughCapture(t, mk())
			if out != sseStream {
				t.Errorf("client received a changed stream:\n%q", out)
			}
			select {
			case call := <-calls:
				// The model comes from the first chunk; the final chunk
				// does not repeat it.
				if call.model != "m1" {
					t.Errorf("model = %q, want m1", call.model)
				}
				if call.timings["predicted_n"] != float64(2) {
					t.Errorf("timings = %v, want the first timings chunk", call.timings)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("capture function was never called")
			}
			if !c.captured {
				t.Error("capture is not marked done")
			}
			// Every call is started inside Read, so all of them were
			// started before ReadAll returned; a second would arrive
			// almost at once.
			select {
			case extra := <-calls:
				t.Errorf("capture called a second time with %v", extra)
			case <-time.After(50 * time.Millisecond):
			}
		})
	}
}

// A stream without timings (an older llama.cpp, or a cut-off response)
// passes through and reports nothing rather than a half-filled record.
func TestSSETimingCaptureWithoutTimings(t *testing.T) {
	stream := "data: {\"model\":\"m1\",\"choices\":[]}\n\ndata: [DONE]\n\n"
	out, calls, _ := readThroughCapture(t, iotest.OneByteReader(strings.NewReader(stream)))
	if out != stream {
		t.Errorf("client received a changed stream: %q", out)
	}
	select {
	case call := <-calls:
		t.Errorf("capture called without timings: %v", call)
	case <-time.After(50 * time.Millisecond):
	}
}

// chunkReader returns data at most size bytes per Read.
type chunkReader struct {
	data []byte
	size int
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := min(r.size, len(p), len(r.data))
	copy(p, r.data[:n])
	r.data = r.data[n:]
	return n, nil
}

// fakeRouter is a llama-server router API the test controls: GET
// /models lists the models in status, POST /models/load records the name
// and marks it loading. onList, when set, runs on each list (with the
// lock held and the 1-based call count) so a test can move a model from
// loading to loaded, or make it disappear.
type fakeRouter struct {
	mu      sync.Mutex
	status  map[string]string
	aliases map[string][]string
	loads   []string
	lists   int
	onList  func(f *fakeRouter, n int)
	// slots is what GET /slots answers; slotCalls records each slot
	// action as "save 0 file.bin model".
	slots     []routerclient.Slot
	slotCalls []string
	// restoreGate, when set, holds each slot restore until it is closed,
	// with the lock released so the model list still answers.
	restoreGate chan struct{}
	// slotLists counts GET /slots: on a real router, one for a model that
	// is not loaded loads it.
	slotLists int
	// ttsBodies records the body of each POST /tts.
	ttsBodies [][]byte
	// realtimeProtocols records the Sec-WebSocket-Protocol header of
	// each realtime session.
	realtimeProtocols []string
	// props is what GET /props answers, and propsQueries records the
	// query of each call.
	props        string
	propsQueries []string
	// v1, when set, answers the /v1/ requests the cases below do not.
	v1 http.HandlerFunc
}

// reset replaces the router's state between subtests.
func (f *fakeRouter) reset(status map[string]string, onList func(*fakeRouter, int)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.status, f.aliases, f.loads, f.lists, f.onList = status, nil, nil, 0, onList
	f.ttsBodies = nil
}

func (f *fakeRouter) ttsCalls() [][]byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([][]byte(nil), f.ttsBodies...)
}

func (f *fakeRouter) loadCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.loads...)
}

func (f *fakeRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Before the lock: a session stays open while the test lists models.
	if r.URL.Path == realtimePath {
		f.serveRealtime(w, r)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/v1/") && f.v1 != nil {
		f.v1(w, r)
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.URL.Path == "/health":
		io.WriteString(w, `{"status":"ok"}`)
	case r.URL.Path == "/models" && r.Method == http.MethodGet:
		f.lists++
		if f.onList != nil {
			f.onList(f, f.lists)
		}
		var data []process.ModelStatus
		for name, st := range f.status {
			var ms process.ModelStatus
			ms.ID, ms.Model, ms.Aliases = name, name, f.aliases[name]
			ms.Status.Value = st
			data = append(data, ms)
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data, "object": "list"})
	case r.URL.Path == "/slots" && r.Method == http.MethodGet:
		f.slotLists++
		json.NewEncoder(w).Encode(f.slots)
	case strings.HasPrefix(r.URL.Path, "/slots/") && r.Method == http.MethodPost:
		var req struct{ Model, Filename string }
		json.NewDecoder(r.Body).Decode(&req)
		action := r.URL.Query().Get("action")
		f.slotCalls = append(f.slotCalls, fmt.Sprintf("%s %s %s %s", action, strings.TrimPrefix(r.URL.Path, "/slots/"), req.Filename, req.Model))
		if gate := f.restoreGate; action == "restore" && gate != nil {
			f.mu.Unlock()
			<-gate
			f.mu.Lock()
		}
		if action == "save" {
			fmt.Fprintf(w, `{"n_saved":2000,"n_written":1048576}`)
		} else {
			fmt.Fprintf(w, `{"n_restored":2000,"n_read":1048576}`)
		}
	case r.URL.Path == "/props" && r.Method == http.MethodGet:
		f.propsQueries = append(f.propsQueries, r.URL.RawQuery)
		if f.props == "" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		io.WriteString(w, f.props)
	case r.URL.Path == "/tts" && r.Method == http.MethodPost:
		body, _ := io.ReadAll(r.Body)
		f.ttsBodies = append(f.ttsBodies, body)
		w.Header().Set("Content-Type", "audio/wav")
		io.WriteString(w, "RIFF")
	case r.URL.Path == "/models/load" && r.Method == http.MethodPost:
		var req struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		f.loads = append(f.loads, req.Model)
		f.status[req.Model] = "loading"
		io.WriteString(w, `{"success":true}`)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

// startFakeRouter returns a server whose process.Manager runs a fake
// llama-server and reports running, with the router API answered by a
// fakeRouter. One model (proxyModelID) is registered.
func startFakeRouter(t *testing.T) (*Server, *fakeRouter, *models.Model) {
	t.Helper()
	binDir := fakeLlamaServerDir(t)
	f := &fakeRouter{status: map[string]string{}}
	ts := httptest.NewServer(f)
	t.Cleanup(ts.Close)

	s := newTestServer(t)
	s.cfg.LlamaPort = ts.Listener.Addr().(*net.TCPAddr).Port
	t.Cleanup(func() { s.process.Stop() })
	if err := s.process.Start(process.RouterConfig{
		BinaryPath: filepath.Join(binDir, "llama-server"),
		Host:       "127.0.0.1", Port: s.cfg.LlamaPort, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, s.process)
	m := addProxyModel(t, s, models.ModelConfig{Enabled: true})
	return s, f, m
}

// A chat request naming a model that is installed but not loaded must
// load it and wait, rather than let the router answer 400. Requests the
// proxy cannot act on are passed on unchanged so the router gives its own
// error.
func TestEnsureModelLoadedForRequest(t *testing.T) {
	s, f, m := startFakeRouter(t)
	reqFor := func(name string) []byte { return []byte(`{"model":"` + name + `"}`) }

	t.Run("bodies without a known model are ignored", func(t *testing.T) {
		f.reset(map[string]string{}, nil)
		for _, body := range []string{"", "not json", `{"messages":[]}`, `{"model":""}`, `{"model":"nobody"}`} {
			if err := s.ensureModelLoadedForRequest(context.Background(), []byte(body)); err != nil {
				t.Errorf("body %q: %v", body, err)
			}
		}
		if loads := f.loadCalls(); len(loads) != 0 {
			t.Errorf("load requested for an unknown model: %v", loads)
		}
	})

	t.Run("a loaded model is used as it is", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "loaded"}, nil)
		if err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.PublicName())); err != nil {
			t.Fatal(err)
		}
		if loads := f.loadCalls(); len(loads) != 0 {
			t.Errorf("a loaded model was loaded again: %v", loads)
		}
	})

	t.Run("the router may know the model by an alias", func(t *testing.T) {
		f.reset(map[string]string{"section": "loaded"}, nil)
		f.mu.Lock()
		f.aliases = map[string][]string{"section": {m.PublicName()}}
		f.mu.Unlock()
		if err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.ID)); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("a model missing from the router asks for a restart", func(t *testing.T) {
		f.reset(map[string]string{"someone-else": "loaded"}, nil)
		err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.ID))
		if err == nil || !strings.Contains(err.Error(), "restart the router") {
			t.Fatalf("err = %v, want the restart-the-router message", err)
		}
		if loads := f.loadCalls(); len(loads) != 0 {
			t.Errorf("load requested for a model the router does not have: %v", loads)
		}
	})

	t.Run("an unloaded model is loaded and waited for", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "unloaded"}, func(f *fakeRouter, n int) {
			if f.status[m.ID] == "loading" && n >= 3 {
				f.status[m.ID] = "loaded"
			}
		})
		if err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.PublicName())); err != nil {
			t.Fatal(err)
		}
		if loads := f.loadCalls(); len(loads) != 1 || loads[0] != s.registry.RouterName(m.ID) {
			t.Errorf("load calls = %v, want one for %s", loads, m.ID)
		}
	})

	t.Run("a model already loading is waited for, not loaded again", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "loading"}, func(f *fakeRouter, n int) {
			if n >= 3 {
				f.status[m.ID] = "loaded"
			}
		})
		if err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.ID)); err != nil {
			t.Fatal(err)
		}
		if loads := f.loadCalls(); len(loads) != 0 {
			t.Errorf("a loading model was loaded again: %v", loads)
		}
	})

	t.Run("a model that disappears during the load is an error", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "loading"}, func(f *fakeRouter, n int) {
			if n >= 3 {
				delete(f.status, m.ID)
			}
		})
		err := s.ensureModelLoadedForRequest(context.Background(), reqFor(m.ID))
		if err == nil || !strings.Contains(err.Error(), "disappeared") {
			t.Fatalf("err = %v, want the disappeared message", err)
		}
	})

	// A client that gives up must not keep the request waiting for the
	// rest of the 90 second load timeout.
	t.Run("the wait ends when the request is cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f.reset(map[string]string{m.ID: "loading"}, func(f *fakeRouter, n int) { cancel() })
		err := s.ensureModelLoadedForRequest(ctx, reqFor(m.ID))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	})
}

// A stopped router knows no models, so a chat request for an installed
// model used to get "not in the router preset; restart the router", which
// sends the user looking in the wrong place. It now says the router is
// not running.
func TestEnsureModelLoadedReportsAStoppedRouter(t *testing.T) {
	s := newTestServer(t) // its process manager has never started a router
	m := addProxyModel(t, s, models.ModelConfig{Enabled: true})

	err := s.ensureModelLoadedForRequest(context.Background(), []byte(`{"model":"`+m.PublicName()+`"}`))
	if err == nil || !strings.Contains(err.Error(), "router is not running") {
		t.Fatalf("err = %v, want the router-not-running message", err)
	}
}

// A request waiting for its model to load is not counted yet, so a
// generation job's unload does not wait for it. Counting it from before
// the load deadlocked compute2 for 15 minutes: two chats waited for a
// music job's turn on the GPU while the job waited for the chats to end.
func TestLoadAndCountRequestCountsOnlyOnceLoaded(t *testing.T) {
	s, f, m := startFakeRouter(t)
	release := make(chan struct{})
	f.reset(map[string]string{m.ID: "unloaded"}, func(f *fakeRouter, n int) {
		select {
		case <-release:
			if f.status[m.ID] == "loading" {
				f.status[m.ID] = "loaded"
			}
		default:
		}
	})
	type result struct {
		end func()
		err error
	}
	done := make(chan result, 1)
	go func() {
		end, err := s.loadAndCountRequest(context.Background(), []byte(`{"model":"`+m.ID+`"}`))
		done <- result{end, err}
	}()
	testutil.Eventually(t, func() bool { return len(f.loadCalls()) == 1 }, "the model was not loaded")

	// While the load is under way, an unload does not wait for the request.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	evicted, err := s.llmUse.evict(ctx, m.ID)
	if err != nil {
		t.Fatalf("an unload waited for a request that was still loading its model: %v", err)
	}
	evicted()

	close(release)
	r := <-done
	if r.err != nil {
		t.Fatal(r.err)
	}
	// Loaded and replying: now an unload waits for it.
	short, cancel2 := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel2()
	if _, err := s.llmUse.evict(short, m.ID); err == nil {
		t.Error("an unload did not wait for a request with its reply under way")
	}
	r.end()
}

// An unload that gets in between the load and the count leaves the model
// unloaded: the request loads it again rather than going to the router
// for a model that is not there.
func TestLoadAndCountRequestReloadsAfterAnUnload(t *testing.T) {
	s, f, m := startFakeRouter(t)
	f.reset(map[string]string{m.ID: "loaded"}, func(f *fakeRouter, n int) {
		if f.status[m.ID] == "loading" {
			f.status[m.ID] = "loaded"
		}
	})
	// An unload is under way: the request finds the model loaded, then
	// waits to be counted until the unload is done.
	evicted, err := s.llmUse.evict(context.Background(), m.ID)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		end, err := s.loadAndCountRequest(context.Background(), []byte(`{"model":"`+m.ID+`"}`))
		if err == nil {
			end()
		}
		done <- err
	}()
	time.Sleep(50 * time.Millisecond)
	f.mu.Lock()
	f.status[m.ID] = "unloaded"
	f.mu.Unlock()
	evicted()

	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the request did not finish")
	}
	if loads := f.loadCalls(); len(loads) != 1 {
		t.Errorf("load calls = %v, want the model loaded again once", loads)
	}
}
