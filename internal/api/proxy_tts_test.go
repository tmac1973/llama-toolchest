package api

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ttsForm builds a /tts multipart form: input, a speaker_ref file, then
// model, so the file comes before the field the proxy looks for.
func ttsForm(t *testing.T, model string) (body []byte, contentType string) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("input", "Hello, this is a test.")
	fw, err := mw.CreateFormFile("speaker_ref", "voice.wav")
	if err != nil {
		t.Fatal(err)
	}
	fw.Write(bytes.Repeat([]byte("RIFF"), 4096))
	if model != "" {
		mw.WriteField("model", model)
	}
	mw.Close()
	return buf.Bytes(), mw.FormDataContentType()
}

// The model field is found after a file part, and the body that goes on
// is the one the client sent, whether or not the field was found.
func TestMultipartModel(t *testing.T) {
	cases := []struct {
		name, model string
		limit       int64
		want        string
	}{
		{"model after a file", "org-Model.Q4_K_M", 1 << 20, "org-Model.Q4_K_M"},
		{"no model field", "", 1 << 20, ""},
		{"model past the limit", "org-Model.Q4_K_M", 1024, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body, ct := ttsForm(t, c.model)
			r := httptest.NewRequest(http.MethodPost, "/tts", bytes.NewReader(body))
			r.Header.Set("Content-Type", ct)
			if got := multipartModel(r, c.limit); got != c.want {
				t.Errorf("model = %q, want %q", got, c.want)
			}
			forwarded, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(forwarded, body) {
				t.Errorf("forwarded body changed: %d bytes, sent %d", len(forwarded), len(body))
			}
		})
	}

	r := httptest.NewRequest(http.MethodPost, "/tts", strings.NewReader("x"))
	r.Header.Set("Content-Type", "multipart/form-data")
	if got := multipartModel(r, 1024); got != "" {
		t.Errorf("without a boundary: %q", got)
	}
}

// POST /tts goes to llama-server's /tts behind the API key, and the
// model a multipart form names is loaded first, as for a JSON body.
func TestTTSRoute(t *testing.T) {
	s, f, m := startFakeRouter(t)
	s.cfg.APIKey = "secret"
	router := s.buildRouter()
	loadOnList := func(f *fakeRouter, n int) {
		if f.status[m.ID] == "loading" {
			f.status[m.ID] = "loaded"
		}
	}
	post := func(body []byte, ct, key string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, "/tts", bytes.NewReader(body))
		r.Header.Set("Content-Type", ct)
		if key != "" {
			r.Header.Set("Authorization", "Bearer "+key)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, r)
		return rec
	}

	f.reset(map[string]string{m.ID: "unloaded"}, loadOnList)
	if rec := post([]byte(`{"model":"x","input":"hi"}`), "application/json", ""); rec.Code != http.StatusUnauthorized {
		t.Errorf("without the key: status %d, want 401", rec.Code)
	}
	if len(f.ttsCalls()) != 0 || len(f.loadCalls()) != 0 {
		t.Error("a request without the key reached the router")
	}

	t.Run("json", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "unloaded"}, loadOnList)
		body := []byte(`{"model":"` + m.PublicName() + `","input":"hi"}`)
		rec := post(body, "application/json", "secret")
		if rec.Code != http.StatusOK || rec.Body.String() != "RIFF" {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		if loads := f.loadCalls(); len(loads) != 1 {
			t.Errorf("load calls = %v, want one", loads)
		}
		if got := f.ttsCalls(); len(got) != 1 || !bytes.Equal(got[0], body) {
			t.Errorf("router got %q", got)
		}
	})

	t.Run("multipart", func(t *testing.T) {
		f.reset(map[string]string{m.ID: "unloaded"}, loadOnList)
		body, ct := ttsForm(t, m.PublicName())
		rec := post(body, ct, "secret")
		if rec.Code != http.StatusOK || rec.Body.String() != "RIFF" {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		if loads := f.loadCalls(); len(loads) != 1 || loads[0] != s.registry.RouterName(m.ID) {
			t.Errorf("load calls = %v, want one for %s", loads, m.ID)
		}
		if got := f.ttsCalls(); len(got) != 1 || !bytes.Equal(got[0], body) {
			t.Error("the form reached the router changed")
		}
	})
}
