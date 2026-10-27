package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"gopkg.in/yaml.v3"

	"github.com/tmac1973/llama-toolchest/web"
)

// The API reference: web/apidocs/openapi.yaml describes the public API
// (/v1 and the ACE-Step routes) by hand, and /docs/api shows it with the
// Scalar viewer. TestOpenAPISpecMatchesRouter keeps the file and the
// router in step.

// loadOpenAPISpec parses the embedded spec and sets info.version to this
// build's version, so the page shows which release it describes.
func loadOpenAPISpec(version string) (map[string]any, error) {
	var spec map[string]any
	if err := yaml.Unmarshal(web.OpenAPISpec, &spec); err != nil {
		return nil, fmt.Errorf("parse openapi.yaml: %w", err)
	}
	if info, ok := spec["info"].(map[string]any); ok && version != "" {
		info["version"] = version
	}
	return spec, nil
}

func (s *Server) handleAPIDocsPage(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	w.Write(web.APIDocsPage)
}

func (s *Server) handleOpenAPIJSON(w http.ResponseWriter, r *http.Request) {
	spec, err := loadOpenAPISpec(s.version)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(spec)
}

func (s *Server) handleOpenAPIYAML(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	w.Write(web.OpenAPISpec)
}
