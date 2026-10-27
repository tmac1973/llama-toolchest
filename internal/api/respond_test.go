package api

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
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

// The 413 text follows maxImageRequestBytes instead of repeating it.
func TestTooLargeErrorNamesTheLimit(t *testing.T) {
	e := tooLargeError()
	if e.status != http.StatusRequestEntityTooLarge || e.msg != "the request is larger than 128 MiB" {
		t.Errorf("tooLargeError = %d %q", e.status, e.msg)
	}
}

// A full queue answers 429 with Retry-After; any other submit error is 500.
func TestWriteQueueError(t *testing.T) {
	w := httptest.NewRecorder()
	writeQueueError(w, generation.ErrQueueFull)
	if w.Code != http.StatusTooManyRequests || w.Header().Get("Retry-After") != "30" {
		t.Errorf("queue full: %d, Retry-After %q", w.Code, w.Header().Get("Retry-After"))
	}
	w = httptest.NewRecorder()
	writeQueueError(w, errors.New("boom"))
	if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "queue_error") {
		t.Errorf("other error: %d %s", w.Code, w.Body.String())
	}
}
