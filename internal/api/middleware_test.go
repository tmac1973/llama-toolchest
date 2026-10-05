package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// A 401 from /v1 must be JSON: http.Error used to replace the JSON
// content type with text/plain, so OpenAI clients could not read it.
func TestAPIKeyAuthAnswersJSON(t *testing.T) {
	s := &Server{cfg: &config.Config{APIKey: "sk-right"}}
	h := s.apiKeyAuth(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))

	for _, tc := range []struct{ header, want string }{
		{"", "missing API key"},
		{"Bearer sk-wrong", "invalid API key"},
	} {
		req := httptest.NewRequest("GET", "/v1/models", nil)
		if tc.header != "" {
			req.Header.Set("Authorization", tc.header)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)

		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%q: status %d, want 401", tc.header, rec.Code)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("%q: Content-Type %q, want application/json", tc.header, ct)
		}
		var body struct {
			Error struct{ Message, Type string }
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || body.Error.Message != tc.want || body.Error.Type != "auth_error" {
			t.Errorf("%q: body %q, want message %q", tc.header, rec.Body.String(), tc.want)
		}
	}

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer sk-right")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("right key: status %d, want 200", rec.Code)
	}
}
