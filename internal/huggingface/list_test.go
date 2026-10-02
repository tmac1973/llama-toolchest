package huggingface

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Every field ListedModel reads must be named in expand[]: once any is,
// HuggingFace returns only those, and a missing one arrives empty with no
// error anywhere.
func TestListExpandsEveryFieldItUses(t *testing.T) {
	typ := reflect.TypeOf(ListedModel{})
	for i := range typ.NumField() {
		name := strings.Split(typ.Field(i).Tag.Get("json"), ",")[0]
		if name == "id" {
			continue // always returned
		}
		if !slices.Contains(listExpand, name) {
			t.Errorf("ListedModel.%s (%q) is not in listExpand", typ.Field(i).Name, name)
		}
	}
}

func TestListedModelDecodes(t *testing.T) {
	const body = `[
	 {"id":"a/one","gated":"manual","cardData":{"base_model":"org/Base"},
	  "gguf":{"total":8000000000,"architecture":"llama","context_length":131072},
	  "createdAt":"2026-07-31T10:27:38.000Z"},
	 {"id":"b/two","gated":false,"cardData":{"base_model":["org/A","org/B"]}},
	 {"id":"c/three","gated":"auto"}
	]`
	var got []ListedModel
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatal(err)
	}
	if !got[0].Gated || got[1].Gated || !got[2].Gated {
		t.Errorf("gated = %v %v %v", got[0].Gated, got[1].Gated, got[2].Gated)
	}
	if !slices.Equal(got[0].CardData.BaseModel, []string{"org/Base"}) ||
		!slices.Equal(got[1].CardData.BaseModel, []string{"org/A", "org/B"}) ||
		got[2].CardData.BaseModel != nil {
		t.Errorf("base models = %v %v %v", got[0].CardData.BaseModel, got[1].CardData.BaseModel, got[2].CardData.BaseModel)
	}
	if got[0].GGUF == nil || got[0].GGUF.Total != 8e9 || got[0].CreatedAt.Year() != 2026 {
		t.Errorf("first = %+v", got[0])
	}
}

func TestListGGUFQuery(t *testing.T) {
	var query string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.RawQuery
		w.Write([]byte(`[{"id":"a/b"}]`))
	}))
	defer srv.Close()
	c := NewClient("")
	c.apiBase = srv.URL

	got, err := c.ListGGUF(context.Background(), ListQuery{Author: "unsloth", Sort: "createdAt", Limit: 50})
	if err != nil || len(got) != 1 {
		t.Fatalf("got %v, %v", got, err)
	}
	for _, want := range []string{"filter=gguf", "author=unsloth", "sort=createdAt", "direction=-1", "limit=50", "expand%5B%5D=cardData"} {
		if !strings.Contains(query, want) {
			t.Errorf("query %q lacks %q", query, want)
		}
	}
}

// A 429 is retried once after the wait asked for; a second one is an
// error that says what happened.
func TestGetJSONRetriesOnceOnRateLimit(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 || r.URL.Path == "/always" {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Write([]byte(`[]`))
	}))
	defer srv.Close()
	c := NewClient("")

	var out []ListedModel
	if err := c.getJSON(context.Background(), srv.URL+"/once", &out); err != nil {
		t.Errorf("retry did not recover: %v", err)
	}
	err := c.getJSON(context.Background(), srv.URL+"/always", &out)
	if err == nil || !strings.Contains(err.Error(), "limiting requests") {
		t.Errorf("err = %v", err)
	}
}

func TestRetryAfter(t *testing.T) {
	for h, want := range map[string]time.Duration{"3": 3 * time.Second, "": time.Second, "600": maxRetryAfter, "soon": time.Second} {
		if got := retryAfter(h); got != want {
			t.Errorf("retryAfter(%q) = %v, want %v", h, got, want)
		}
	}
}

func TestNextLink(t *testing.T) {
	h := `<https://huggingface.co/api/models/a/b/tree/main?cursor=xyz>; rel="next"`
	if got := nextLink(h); got != "https://huggingface.co/api/models/a/b/tree/main?cursor=xyz" {
		t.Errorf("nextLink = %q", got)
	}
	if got := nextLink(""); got != "" {
		t.Errorf("nextLink(empty) = %q", got)
	}
}
