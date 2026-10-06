package api

import (
	"bytes"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// The badge sits inside a line of text, so it must render with no
// whitespace of its own around it, and a failure reason is escaped once.
func TestServiceBadge(t *testing.T) {
	tmpl := testTemplates(t)
	for _, c := range []struct {
		status process.Status
		want   string
	}{
		{process.Status{State: process.StateRunning, Uptime: "5m0s"}, `<ins>Running</ins> <small>(5m0s)</small>`},
		{process.Status{State: process.StateStarting}, `<mark>Starting...</mark>`},
		{process.Status{State: process.StateFailed, Error: `exit <1> & "x"`},
			`<del>Failed</del> <small style="color:var(--pico-del-color)">exit &lt;1&gt; &amp; &#34;x&#34;</small>`},
		{process.Status{State: process.StateStopped}, `Stopped`},
	} {
		var buf bytes.Buffer
		if err := tmpl.ExecuteTemplate(&buf, "service_badge", c.status); err != nil {
			t.Fatal(err)
		}
		if got := buf.String(); got != c.want {
			t.Errorf("%s: got %q, want %q", c.status.State, got, c.want)
		}
	}
}
