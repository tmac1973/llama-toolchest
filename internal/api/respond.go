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

// toFloat64 converts int or float types to float64 for template math.
func toFloat64(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case int32:
		return float64(n)
	default:
		return 0
	}
}
