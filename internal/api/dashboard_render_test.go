package api

import (
	"bytes"
	"html"
	"regexp"
	"strings"
	"testing"
)

func renderDashboardCards(t *testing.T, data dashboardCardsData) string {
	t.Helper()
	var buf bytes.Buffer
	if err := testTemplates(t).ExecuteTemplate(&buf, "dashboard_cards", data); err != nil {
		t.Fatal(err)
	}
	return buf.String()
}

// The copy buttons swap their icon for a check mark and back from their
// data-icon attributes. Those hold the icons as escaped markup written
// out by hand in the template, so they must stay the same as the icons
// drawn on the button.
func TestDashboardCopyButtonIcons(t *testing.T) {
	out := renderDashboardCards(t, dashboardCardsData{APIURL: "http://h/v1"})
	tmpl := testTemplates(t)
	icon := func(name string) string {
		var b bytes.Buffer
		if err := tmpl.ExecuteTemplate(&b, name, nil); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	attr := func(name string) string {
		m := regexp.MustCompile(` ` + name + `="([^"]*)"`).FindStringSubmatch(out)
		if m == nil {
			t.Fatalf("no %s attribute in:\n%s", name, out)
		}
		return html.UnescapeString(m[1])
	}
	if got, want := attr("data-icon"), icon("icon_copy"); got != want {
		t.Errorf("data-icon = %q, want icon_copy %q", got, want)
	}
	if got, want := attr("data-icon-check"), icon("icon_check"); got != want {
		t.Errorf("data-icon-check = %q, want icon_check %q", got, want)
	}
	if !strings.Contains(out, `data-copy="http://h/v1" onclick="copyFromButton(this)">`+icon("icon_copy")+`</button>`) {
		t.Errorf("endpoint copy button does not carry the URL and the copy icon:\n%s", out)
	}
}

// The chat link appears only when the external URL names a host, and
// the model rows show the load button only for an idle model. The
// buttons and the name are inline, so no whitespace may come between
// them: it would show as a gap.
func TestDashboardCardsRows(t *testing.T) {
	out := renderDashboardCards(t, dashboardCardsData{APIURL: "/v1"})
	if strings.Contains(out, "Open Chat UI") {
		t.Error("chat link shown with no chat URL")
	}
	if !strings.Contains(out, `<div class="available-models-scroll"><p>None</p></div>`) {
		t.Errorf("no models should read None:\n%s", out)
	}

	out = renderDashboardCards(t, dashboardCardsData{
		APIURL:  "http://h:8080/v1",
		ChatURL: "http://h:5000",
		Available: []availableModelRow{
			{ID: `a&"b`, PublicName: "idle-model", Tooltip: "t"},
			{ID: "c", PublicName: "loaded-model", State: "loaded"},
		},
	})
	if !strings.Contains(out, `<p><a href="http://h:5000" target="_blank">Open Chat UI →</a></p>`) {
		t.Errorf("chat link missing:\n%s", out)
	}
	if !strings.Contains(out, `hx-put="/api/models/a&amp;&#34;b/activate"`) {
		t.Errorf("load button missing or its model ID not escaped once:\n%s", out)
	}
	if strings.Count(out, `title="Load into VRAM"`) != 1 {
		t.Error("only the idle model should offer the load button")
	}
	for _, want := range []string{
		`<div class="available-model-row" title="t"><button type="button" class="action-icon" title="Load into VRAM"`,
		`</button><code>idle-model</code></div>`,
		`<span class="action-icon-placeholder">&nbsp;</span><button`,
		`</button><code>loaded-model</code> <mark style="padding:0 0.3rem;font-size:0.65rem;">loaded</mark></div>`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}
