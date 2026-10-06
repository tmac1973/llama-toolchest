package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Where server-rendered HTML lives:
//
//   - A fragment with structure (several elements, a loop, a condition)
//     goes in a partial under web/templates/partials and is written with
//     s.renderPartial. html/template escapes every value there.
//   - A one-line message or a single element may stay in Go, written with
//     fmt.Fprintf, but every value in it must go through
//     html.EscapeString (for example `<p>Settings saved.</p>` or
//     `<small>%s</small>`).

// isHTMX returns true if the request was made by htmx.
func isHTMX(r *http.Request) bool {
	return r.Header.Get("HX-Request") == "true"
}

// isJSONRequest reports whether the request body is JSON. Handlers that
// accept both JSON and form posts branch on this; it allows a charset
// suffix, so "application/json; charset=utf-8" is not read as a form.
func isJSONRequest(r *http.Request) bool {
	return isJSONContentType(r.Header.Get("Content-Type"))
}

// decodeJSONBody decodes the request body into v. On failure it answers
// 400 and returns false, and the handler should return.
func decodeJSONBody(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return false
	}
	return true
}

// respondJSON writes v as JSON with the appropriate Content-Type header.
func respondJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// respondJSONStatus writes v as JSON with an explicit HTTP status. The
// Content-Type header must be set before WriteHeader, so callers can't
// compose this from respondJSON themselves.
func respondJSONStatus(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// respondHTML sets the Content-Type header for HTML responses.
func respondHTML(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
}

// toFloat64 converts int or float types to float64 for template math.
func toFloat64(v any) float64 {
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

// hxVals builds the JSON for an hx-vals attribute from key/value pairs,
// in the order given: hxVals "model_id" .ModelID "size" .Size. Values are
// written as JSON strings (htmx posts them as form values). Building the
// JSON by hand in a template broke on a value with a quote in it, such as
// a profile the user named with one; here every value is JSON-encoded,
// and html/template then escapes the result for the attribute.
func hxVals(pairs ...any) (string, error) {
	if len(pairs)%2 != 0 {
		return "", fmt.Errorf("hxVals: odd number of arguments")
	}
	var b strings.Builder
	b.WriteByte('{')
	for i := 0; i < len(pairs); i += 2 {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(fmt.Sprint(pairs[i]))
		v, _ := json.Marshal(fmt.Sprint(pairs[i+1]))
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	return b.String(), nil
}
