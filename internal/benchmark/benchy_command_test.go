package benchmark

import (
	"strings"
	"testing"
)

// The command string is logged and stored with the run, so a real API key
// must not appear in it. The "EMPTY" placeholder stays, so the disclosed
// command still runs against a router without a key.
func TestFormatBenchyCommandMasksTheAPIKey(t *testing.T) {
	c := BenchyConfig{BaseURL: "http://localhost:8080/v1", ServedModelName: "m", SaveResultPath: "/tmp/r.json"}

	c.APIKey = "sk-secret"
	if got := FormatBenchyCommand(c); strings.Contains(got, "sk-secret") || !strings.Contains(got, "--api-key HIDDEN") {
		t.Errorf("real key not masked: %s", got)
	}
	if args := BuildBenchyArgs(c); !strings.Contains(strings.Join(args, " "), "--api-key sk-secret") {
		t.Errorf("the arguments actually run must keep the real key: %v", args)
	}

	c.APIKey = "EMPTY"
	if got := FormatBenchyCommand(c); !strings.Contains(got, "--api-key EMPTY") {
		t.Errorf("placeholder should be shown as is: %s", got)
	}
}
