package routerclient

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"
)

// The router wraps the list in {"data": [...]}, as server-models.cpp does.
// A bare-array decoder silently read this as "nothing loaded".
func TestListDecodesRouterEnvelope(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[
			{"id":"a","status":{"value":"loaded"}},
			{"id":"b","status":{"value":"unloaded"}},
			{"id":"c","status":{"value":"loading"}}
		],"object":"list"}`)
	}))
	defer srv.Close()

	list, err := List(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	var loaded []string
	for _, m := range list {
		if m.IsLoaded() {
			loaded = append(loaded, m.ID)
		}
	}
	if len(list) != 3 || len(loaded) != 2 || loaded[0] != "a" || loaded[1] != "c" {
		t.Fatalf("got list %+v, loaded %v; want 3 models with a and c loaded", list, loaded)
	}
}

func TestLoadTreatsAlreadyRunningAsSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"model is already running"}}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	if err := Load(context.Background(), srv.URL, "m"); err != nil {
		t.Fatalf("Load: %v, want nil for an already running model", err)
	}
}

func TestLoadAndUnloadReportOtherFailures(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusBadRequest)
	}))
	defer srv.Close()

	if err := Load(context.Background(), srv.URL, "m"); err == nil {
		t.Fatal("Load: got nil, want an error")
	}
	if err := Unload(context.Background(), srv.URL, "m"); err == nil {
		t.Fatal("Unload: got nil, want an error")
	}
}

// Error messages are logged and shown in the UI, so a large error page is
// cut, on a character boundary, with a note of how much was left out.
func TestErrorSnippet(t *testing.T) {
	if got := ErrorSnippet([]byte("short")); got != "short" {
		t.Errorf("short body changed: %q", got)
	}
	long := []byte(strings.Repeat("a", maxErrorBody-1) + "é" + strings.Repeat("b", 100))
	got := ErrorSnippet(long)
	if !utf8.ValidString(got) {
		t.Errorf("snippet split a character: %q", got[len(got)-30:])
	}
	if !strings.HasSuffix(got, "… (102 more bytes)") {
		t.Errorf("snippet ends %q", got[len(got)-30:])
	}
}
