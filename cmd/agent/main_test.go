package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestResolveToolPath(t *testing.T) {
	workDir := t.TempDir()
	tests := []struct {
		name    string
		path    string
		want    string // expected result; empty when an error is expected
		wantErr bool
	}{
		{name: "file inside", path: "a.txt", want: filepath.Join(workDir, "a.txt")},
		{name: "nested inside", path: "sub/b.txt", want: filepath.Join(workDir, "sub", "b.txt")},
		{name: "dot is the work dir", path: ".", want: workDir},
		{name: "cleaned back inside", path: "sub/../a.txt", want: filepath.Join(workDir, "a.txt")},
		{name: "name starting with dots", path: "..a", want: filepath.Join(workDir, "..a")},
		{name: "absolute inside", path: filepath.Join(workDir, "c.txt"), want: filepath.Join(workDir, "c.txt")},
		{name: "parent", path: "..", wantErr: true},
		{name: "dot dot escape", path: "../../etc/passwd", wantErr: true},
		{name: "escape after nested dir", path: "sub/../../x", wantErr: true},
		{name: "absolute outside", path: "/etc/passwd", wantErr: true},
		{name: "sibling with same prefix", path: workDir + "-other/x", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveToolPath(workDir, tt.path)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("resolveToolPath(%q) = %q, want an error", tt.path, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveToolPath(%q) error: %v", tt.path, err)
			}
			if got != tt.want {
				t.Errorf("resolveToolPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestParseContentToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		wantName []string
		wantArgs []string
	}{
		{
			name:     "object arguments",
			content:  `text <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>`,
			wantName: []string{"read_file"},
			wantArgs: []string{`{"path":"a.txt"}`},
		},
		{
			name:     "string-encoded arguments",
			content:  `<tool_call>{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}</tool_call>`,
			wantName: []string{"read_file"},
			wantArgs: []string{`{"path":"a.txt"}`},
		},
		{
			name: "two calls with whitespace",
			content: "<tool_call>\n{\"name\":\"list_directory\",\"arguments\":{\"path\":\".\"}}\n</tool_call>\n" +
				"<tool_call>{\"name\":\"run_command\",\"arguments\":{\"command\":\"ls\"}}</tool_call>",
			wantName: []string{"list_directory", "run_command"},
			wantArgs: []string{`{"path":"."}`, `{"command":"ls"}`},
		},
		{
			name:     "malformed JSON is skipped",
			content:  `<tool_call>{"name":"read_file","arguments":}</tool_call>`,
			wantName: nil,
		},
		{
			name:     "malformed call next to a good one",
			content:  `<tool_call>{bad}</tool_call><tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>`,
			wantName: []string{"read_file"},
			wantArgs: []string{`{"path":"x"}`},
		},
		{
			name:     "no tool calls",
			content:  "just a normal answer",
			wantName: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := parseContentToolCalls(tt.content)
			if len(calls) != len(tt.wantName) {
				t.Fatalf("got %d calls, want %d: %+v", len(calls), len(tt.wantName), calls)
			}
			for i, c := range calls {
				if c.Function.Name != tt.wantName[i] {
					t.Errorf("call %d name = %q, want %q", i, c.Function.Name, tt.wantName[i])
				}
				if c.Type != "function" || c.ID == "" {
					t.Errorf("call %d has type %q and id %q", i, c.Type, c.ID)
				}
				if !json.Valid([]byte(c.Function.Arguments)) {
					t.Errorf("call %d arguments are not valid JSON: %q", i, c.Function.Arguments)
				}
				if c.Function.Arguments != tt.wantArgs[i] {
					t.Errorf("call %d arguments = %q, want %q", i, c.Function.Arguments, tt.wantArgs[i])
				}
			}
		})
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		name string
		s    string
		n    int
		want string
	}{
		{name: "short string unchanged", s: "abc", n: 10, want: "abc"},
		{name: "exact length unchanged", s: "abc", n: 3, want: "abc"},
		{name: "ASCII cut", s: "abcdef", n: 3, want: "abc..."},
		// "é" is 2 bytes, so a cut at byte 2 falls inside it.
		{name: "cut inside 2-byte character", s: "aébc", n: 2, want: "a..."},
		// "€" is 3 bytes (bytes 1 to 3).
		{name: "cut inside 3-byte character", s: "a€b", n: 3, want: "a..."},
		{name: "cut after a full character", s: "a€b", n: 4, want: "a€..."},
		{name: "multi-byte short string unchanged", s: "€€", n: 6, want: "€€"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := truncate(tt.s, tt.n)
			if got != tt.want {
				t.Errorf("truncate(%q, %d) = %q, want %q", tt.s, tt.n, got, tt.want)
			}
			if !utf8.ValidString(got) {
				t.Errorf("truncate(%q, %d) = %q is not valid UTF-8", tt.s, tt.n, got)
			}
		})
	}
}

func TestTruncateResult(t *testing.T) {
	short := "hello"
	if got := truncateResult(short); got != short {
		t.Errorf("short result changed: %q", got)
	}

	ascii := strings.Repeat("a", maxToolResult+10)
	got := truncateResult(ascii)
	if want := strings.Repeat("a", maxToolResult) + "\n... (truncated)"; got != want {
		t.Errorf("ASCII result has length %d, want %d", len(got), len(want))
	}

	// One ASCII byte first, so each 3-byte "€" crosses the byte limit.
	multi := "a" + strings.Repeat("€", maxToolResult)
	got = truncateResult(multi)
	if !utf8.ValidString(got) {
		t.Fatal("truncated result is not valid UTF-8")
	}
	body := strings.TrimSuffix(got, "\n... (truncated)")
	if body == got {
		t.Fatal("truncated result has no truncation note")
	}
	if len(body) > maxToolResult || len(body) < maxToolResult-utf8.UTFMax {
		t.Errorf("truncated body has length %d, want close to %d", len(body), maxToolResult)
	}
}

func TestExecuteToolFiles(t *testing.T) {
	workDir := t.TempDir()
	if err := os.Mkdir(filepath.Join(workDir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workDir, "a.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workDir, "big.txt"), []byte(strings.Repeat("x", maxToolResult+1)), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		tool     string
		args     string
		want     string // exact result, when not empty
		contains string // part of the result, when not empty
	}{
		{name: "read inside", tool: "read_file", args: `{"path":"a.txt"}`, want: "hello"},
		{name: "read absolute inside", tool: "read_file", args: `{"path":"` + filepath.Join(workDir, "a.txt") + `"}`, want: "hello"},
		{name: "read cleaned back inside", tool: "read_file", args: `{"path":"sub/../a.txt"}`, want: "hello"},
		{name: "read large file", tool: "read_file", args: `{"path":"big.txt"}`, contains: "... (truncated)"},
		{name: "read missing file", tool: "read_file", args: `{"path":"none.txt"}`, contains: "error:"},
		{name: "read dot dot escape", tool: "read_file", args: `{"path":"../../etc/passwd"}`, contains: "outside the working directory"},
		{name: "read absolute outside", tool: "read_file", args: `{"path":"/etc/passwd"}`, contains: "outside the working directory"},
		{name: "list default", tool: "list_directory", args: `{}`, want: "a.txt\nbig.txt\nsub/"},
		{name: "list escape", tool: "list_directory", args: `{"path":".."}`, contains: "outside the working directory"},
		{name: "bad arguments", tool: "read_file", args: `not json`, contains: "error parsing arguments"},
		{name: "unknown tool", tool: "write_file", args: `{}`, want: "unknown tool: write_file"},
		{name: "run echo", tool: "run_command", args: `{"command":"echo hi"}`, want: "hi\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := executeTool(tt.tool, tt.args, workDir)
			if tt.want != "" && got != tt.want {
				t.Errorf("result = %q, want %q", got, tt.want)
			}
			if tt.contains != "" && !strings.Contains(got, tt.contains) {
				t.Errorf("result = %q, want it to contain %q", got, tt.contains)
			}
		})
	}
}
