package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A malformed form body answers 400 instead of running the handler with
// empty values.
func TestParseFormRejectsMalformedBody(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("a=%zz"))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	if parseForm(w, r) {
		t.Fatal("parseForm accepted a malformed body")
	}
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}

	r = httptest.NewRequest(http.MethodPost, "/", strings.NewReader("a=1"))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w = httptest.NewRecorder()
	if !parseForm(w, r) || r.FormValue("a") != "1" {
		t.Fatal("parseForm rejected a valid body")
	}
}
