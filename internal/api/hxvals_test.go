package api

import (
	"bytes"
	"encoding/json"
	"html"
	"regexp"
	"testing"
)

// hx-vals JSON used to be written by hand in the templates, so a value
// with a quote in it (a profile the user named `My "fast" one`) ended the
// JSON string early and the button posted nothing useful. hxVals encodes
// each value; html/template escapes the result for the attribute, and the
// browser decodes it back to the JSON.
func TestHXValsSurvivesQuotes(t *testing.T) {
	name := `My "fast" one's profile <b>`
	tmpl := testTemplates(t)
	tmpl, err := tmpl.New("probe").Parse(`<button hx-vals='{{hxVals "profile" .}}'></button>`)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, name); err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`hx-vals='([^']*)'`).FindStringSubmatch(buf.String())
	if m == nil {
		t.Fatalf("no single-quoted hx-vals in %s", buf.String())
	}
	var vals map[string]string
	if err := json.Unmarshal([]byte(html.UnescapeString(m[1])), &vals); err != nil {
		t.Fatalf("hx-vals is not valid JSON once decoded: %v\n%s", err, m[1])
	}
	if vals["profile"] != name {
		t.Errorf("profile = %q, want %q", vals["profile"], name)
	}

	if _, err := hxVals("odd"); err == nil {
		t.Error("an odd number of arguments should be an error")
	}
}
