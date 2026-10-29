package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
)

// publicRoute reports whether a router path belongs to the public API
// that openapi.yaml documents.
func publicRoute(path string) bool {
	return path == "/release_task" || path == "/query_result" || path == "/tts" ||
		strings.HasPrefix(path, "/v1/")
}

// TestOpenAPISpecMatchesRouter fails when a public route is added,
// removed or given another method without the same change to
// web/apidocs/openapi.yaml. Paths that llama-server answers through the
// /v1 catch-all are not in the router; the spec marks those with
// x-proxied: true.
func TestOpenAPISpecMatchesRouter(t *testing.T) {
	spec, err := loadOpenAPISpec("")
	if err != nil {
		t.Fatal(err)
	}
	paths, _ := spec["paths"].(map[string]any)
	if len(paths) == 0 {
		t.Fatal("openapi.yaml has no paths")
	}

	routed := map[string]bool{} // "GET /v1/models"
	s := newTestServer(t)
	err = chi.Walk(s.buildRouter(), func(method, route string, _ http.Handler, _ ...func(http.Handler) http.Handler) error {
		route = strings.TrimSuffix(route, "/")
		if publicRoute(route) && !strings.HasSuffix(route, "/*") {
			routed[method+" "+route] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	documented := map[string]bool{}
	for path, item := range paths {
		ops, _ := item.(map[string]any)
		proxied, _ := ops["x-proxied"].(bool)
		for method := range ops {
			m := strings.ToUpper(method)
			switch m {
			case "GET", "POST", "PUT", "PATCH", "DELETE":
			default:
				continue
			}
			key := m + " " + path
			if proxied {
				if routed[key] {
					t.Errorf("%s is marked x-proxied in openapi.yaml but the router handles it", key)
				}
				continue
			}
			documented[key] = true
		}
	}

	var missing, stale []string
	for k := range routed {
		if !documented[k] {
			missing = append(missing, k)
		}
	}
	for k := range documented {
		if !routed[k] {
			stale = append(stale, k)
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	for _, k := range missing {
		t.Errorf("%s is routed but not in web/apidocs/openapi.yaml", k)
	}
	for _, k := range stale {
		t.Errorf("%s is in web/apidocs/openapi.yaml but not routed (mark it x-proxied if llama-server answers it)", k)
	}
}

// TestOpenAPISpecRefsResolve checks every $ref in the spec points at a
// component that exists, since a broken one only shows as a blank
// section in the viewer.
func TestOpenAPISpecRefsResolve(t *testing.T) {
	spec, err := loadOpenAPISpec("")
	if err != nil {
		t.Fatal(err)
	}
	var walk func(v any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if ref, ok := x["$ref"].(string); ok {
				if !resolveRef(spec, ref) {
					t.Errorf("$ref %q does not resolve", ref)
				}
			}
			for _, c := range x {
				walk(c)
			}
		case []any:
			for _, c := range x {
				walk(c)
			}
		}
	}
	walk(spec)
}

func resolveRef(spec map[string]any, ref string) bool {
	if !strings.HasPrefix(ref, "#/") {
		return false
	}
	var cur any = spec
	for _, part := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		m, ok := cur.(map[string]any)
		if !ok {
			return false
		}
		if cur, ok = m[part]; !ok {
			return false
		}
	}
	return true
}

func TestOpenAPIJSONHasBuildVersion(t *testing.T) {
	s := newTestServer(t)
	s.SetVersion("9.8.7")
	rec := httptest.NewRecorder()
	s.handleOpenAPIJSON(rec, httptest.NewRequest("GET", "/openapi.json", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got struct {
		OpenAPI string `json:"openapi"`
		Info    struct {
			Version string `json:"version"`
		} `json:"info"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Info.Version != "9.8.7" || !strings.HasPrefix(got.OpenAPI, "3.1") {
		t.Errorf("openapi %q, info.version %q", got.OpenAPI, got.Info.Version)
	}
}
