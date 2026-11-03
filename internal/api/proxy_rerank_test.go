package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// POST /rerank, at the root as llama-server has it, and /v1/rerank both
// reach llama-server behind the API key, with the model loaded first.
func TestRerankRoutes(t *testing.T) {
	s, f, m := startFakeRouter(t)
	s.cfg.APIKey = "secret"
	router := s.buildRouter()
	loadOnList := func(f *fakeRouter, n int) {
		if f.status[m.ID] == "loading" {
			f.status[m.ID] = "loaded"
		}
	}
	body := `{"model":"` + m.PublicName() + `","query":"q","documents":["a","b"]}`
	post := func(path, key string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		if key != "" {
			r.Header.Set("Authorization", "Bearer "+key)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, r)
		return rec
	}

	for _, path := range []string{"/rerank", "/v1/rerank"} {
		t.Run(path, func(t *testing.T) {
			f.reset(map[string]string{m.ID: "unloaded"}, loadOnList)
			if rec := post(path, ""); rec.Code != http.StatusUnauthorized {
				t.Errorf("without the key: status %d, want 401", rec.Code)
			}
			if len(f.rerankCalls()) != 0 {
				t.Error("a request without the key reached the router")
			}
			rec := post(path, "secret")
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "relevance_score") {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if got := f.rerankCalls(); len(got) != 1 || got[0] != path {
				t.Errorf("router got %v, want one call to %s", got, path)
			}
			if loads := f.loadCalls(); len(loads) != 1 {
				t.Errorf("load calls = %v, want one", loads)
			}
		})
	}
}
