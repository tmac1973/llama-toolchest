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

// A failed run's error is stored and shown, so the tokens the process ran
// with are hidden, and a long output keeps its end, where Python prints
// the cause.
func TestBenchyStderrHidesSecretsAndKeepsTheEnd(t *testing.T) {
	out := benchyStderr("GET https://hf.co (Authorization: Bearer hf_secret) failed", "hf_secret", "EMPTY")
	if strings.Contains(out, "hf_secret") || !strings.Contains(out, "Bearer HIDDEN") {
		t.Errorf("token not hidden: %q", out)
	}
	long := strings.Repeat("noise\n", 2000) + "ZeroDivisionError: division by zero"
	out = benchyStderr(long)
	if len(out) > maxBenchyStderr+64 || !strings.HasSuffix(out, "ZeroDivisionError: division by zero") || !strings.HasPrefix(out, "(") {
		t.Errorf("long output not cut to its end: %d bytes, starts %q", len(out), out[:40])
	}
}
