package benchmark

import (
	"slices"
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

// BuildBenchyArgs is the argument list actually run, so each config field
// must map to its flag, and an unset optional field must leave its flag out
// rather than pass an empty or zero value that llama-benchy would reject or
// misread. The fixed parts (tokenizer backends, JSON output to the result
// file) are always there.
func TestBuildBenchyArgs(t *testing.T) {
	full := BenchyConfig{
		BaseURL:         "http://127.0.0.1:8080/v1",
		APIKey:          "EMPTY",
		ServedModelName: "qwen3-8b",
		Tokenizer:       "Qwen/Qwen3-8B",
		PromptSizes:     []int{512, 2048},
		GenSizes:        []int{128},
		Runs:            3,
		Concurrency:     []int{1, 4},
		SaveResultPath:  "/tmp/r.json",
		HFToken:         "hf_secret",
		HFHome:          "/data/hf",
	}
	want := []string{
		"--with", "sentencepiece",
		"--with", "tiktoken",
		"llama-benchy",
		"--base-url", "http://127.0.0.1:8080/v1",
		"--api-key", "EMPTY",
		"--model", "qwen3-8b",
		"--tokenizer", "Qwen/Qwen3-8B",
		"--pp", "512", "--pp", "2048",
		"--tg", "128",
		"--runs", "3",
		"--concurrency", "1", "--concurrency", "4",
		"--format", "json", "--save-result", "/tmp/r.json",
	}
	if got := BuildBenchyArgs(full); !slices.Equal(got, want) {
		t.Errorf("full config:\n got %q\nwant %q", got, want)
	}

	// The token and cache path go through the environment, never the
	// arguments, so they cannot appear in the disclosed command.
	for _, a := range BuildBenchyArgs(full) {
		if strings.Contains(a, "hf_secret") || strings.Contains(a, "/data/hf") {
			t.Errorf("HF token or HF home leaked into the arguments: %q", a)
		}
	}

	minimal := BenchyConfig{BaseURL: "http://h/v1", APIKey: "EMPTY", ServedModelName: "m", SaveResultPath: "/tmp/r.json"}
	want = []string{
		"--with", "sentencepiece",
		"--with", "tiktoken",
		"llama-benchy",
		"--base-url", "http://h/v1",
		"--api-key", "EMPTY",
		"--model", "m",
		"--format", "json", "--save-result", "/tmp/r.json",
	}
	if got := BuildBenchyArgs(minimal); !slices.Equal(got, want) {
		t.Errorf("minimal config:\n got %q\nwant %q", got, want)
	}

	// A zero or negative run count means "use llama-benchy's default".
	for _, runs := range []int{0, -1} {
		c := minimal
		c.Runs = runs
		if slices.Contains(BuildBenchyArgs(c), "--runs") {
			t.Errorf("Runs=%d should leave --runs out", runs)
		}
	}
}

// The disclosed command is meant to be pasted into a shell. Plain values
// are left bare so it stays readable; anything else is single-quoted so
// the shell reads it as one literal argument. It used Go's strconv.Quote,
// which is not shell quoting: an empty value vanished, and backticks, ;,
// | and $ inside double quotes were still acted on by the shell.
func TestFormatBenchyCommandQuoting(t *testing.T) {
	base := BenchyConfig{BaseURL: "http://h/v1", APIKey: "EMPTY", SaveResultPath: "/tmp/r.json"}
	cases := []struct {
		name  string
		model string
		want  string // how the --model value appears in the command
	}{
		{"plain value is bare", "qwen3-8b", "--model qwen3-8b "},
		{"placeholder braces are bare", "{router-served-model-name}", "--model {router-served-model-name} "},
		{"brace expansion is quoted", "{a,b}", "--model '{a,b}' "},
		{"space", "my model", "--model 'my model' "},
		{"double quote", `say "hi"`, `--model 'say "hi"' `},
		{"single quote", "it's", `--model 'it'\''s' `},
		{"backslash", `a\b`, `--model 'a\b' `},
		{"empty stays an argument", "", "--model '' "},
		{"backtick", "a`id`", "--model 'a`id`' "},
		{"semicolon", "a;b", "--model 'a;b' "},
		{"dollar", "a$HOME", "--model 'a$HOME' "},
		{"glob", "a*b", "--model 'a*b' "},
		{"tab", "my\tmodel", "--model 'my\tmodel' "},
	}
	for _, c := range cases {
		cfg := base
		cfg.ServedModelName = c.model
		got := FormatBenchyCommand(cfg)
		if !strings.Contains(got, c.want) {
			t.Errorf("%s: command %q does not contain %q", c.name, got, c.want)
		}
		if !strings.HasPrefix(got, "uvx --with sentencepiece --with tiktoken llama-benchy ") {
			t.Errorf("%s: command %q does not start with the uvx call", c.name, got)
		}
	}
}
