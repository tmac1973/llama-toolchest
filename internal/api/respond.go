package api

import (
	"encoding/json"
	"net/http"
)

// isHTMX returns true if the request was made by htmx.
func isHTMX(r *http.Request) bool {
	return r.Header.Get("HX-Request") == "true"
}

// respondJSON writes v as JSON with the appropriate Content-Type header.
func respondJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// respondHTML sets the Content-Type header for HTML responses.
func respondHTML(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
}
