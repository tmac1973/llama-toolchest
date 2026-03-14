package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// --- OpenAI-compatible types ---

type message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type tool struct {
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type toolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type request struct {
	Model    string    `json:"model,omitempty"`
	Messages []message `json:"messages"`
	Tools    []tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
}

type delta struct {
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content"`
	ToolCalls        []toolCall `json:"tool_calls,omitempty"`
}

type choice struct {
	Delta        delta  `json:"delta"`
	FinishReason string `json:"finish_reason"`
}

type streamEvent struct {
	Choices []choice `json:"choices"`
}

// --- Tool definitions ---

var tools = []tool{
	{
		Type: "function",
		Function: toolFunction{
			Name:        "list_directory",
			Description: "List files and directories at the given path",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"Directory path to list (relative to working directory)"}},"required":["path"]}`),
		},
	},
	{
		Type: "function",
		Function: toolFunction{
			Name:        "read_file",
			Description: "Read the contents of a file",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"File path to read (relative to working directory)"}},"required":["path"]}`),
		},
	},
	{
		Type: "function",
		Function: toolFunction{
			Name:        "run_command",
			Description: "Run a shell command and return its output. Use for things like grep, wc, find, etc.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"}},"required":["command"]}`),
		},
	},
}

func executeTool(name, argsJSON, workDir string) string {
	var args map[string]string
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("error parsing arguments: %v", err)
	}

	switch name {
	case "list_directory":
		path := args["path"]
		if path == "" {
			path = "."
		}
		fullPath := filepath.Join(workDir, path)
		entries, err := os.ReadDir(fullPath)
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		var lines []string
		for _, e := range entries {
			suffix := ""
			if e.IsDir() {
				suffix = "/"
			}
			lines = append(lines, e.Name()+suffix)
		}
		return strings.Join(lines, "\n")

	case "read_file":
		path := args["path"]
		fullPath := filepath.Join(workDir, path)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		// Truncate very large files
		content := string(data)
		if len(content) > 8192 {
			content = content[:8192] + "\n... (truncated)"
		}
		return content

	case "run_command":
		command := args["command"]
		cmd := exec.Command("bash", "-c", command)
		cmd.Dir = workDir
		out, err := cmd.CombinedOutput()
		result := string(out)
		if err != nil {
			result += fmt.Sprintf("\nexit status: %v", err)
		}
		if len(result) > 8192 {
			result = result[:8192] + "\n... (truncated)"
		}
		return result

	default:
		return fmt.Sprintf("unknown tool: %s", name)
	}
}

// --- Main ---

func main() {
	apiURL := flag.String("url", "http://localhost:3000/v1/chat/completions", "API endpoint")
	model := flag.String("model", "", "model name (optional)")
	system := flag.String("system", "", "system prompt")
	noTools := flag.Bool("no-tools", false, "disable tool use")
	flag.Parse()

	workDir, _ := os.Getwd()

	var history []message

	defaultSystem := fmt.Sprintf("You are a helpful assistant. You have access to tools that let you explore the filesystem. The current working directory is: %s", workDir)
	if *system != "" {
		defaultSystem = *system
	}
	history = append(history, message{Role: "system", Content: defaultSystem})

	var activeTools []tool
	if !*noTools {
		activeTools = tools
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "/quit" || input == "/exit" {
			break
		}
		if input == "/clear" {
			history = history[:1] // keep system prompt
			fmt.Println("(conversation cleared)")
			continue
		}

		history = append(history, message{Role: "user", Content: input})

		// Loop to handle tool calls (max 10 rounds to prevent infinite loops)
		for round := 0; round < 10; round++ {
			reply, pendingCalls, err := chat(*apiURL, *model, history, activeTools)
			if err != nil {
				fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
				// Only remove the user message on the first round
				if round == 0 {
					history = history[:len(history)-1]
				}
				break
			}

			// Some models (Qwen3) emit tool calls as <tool_call> tags in content
			if len(pendingCalls) == 0 && strings.Contains(reply, "<tool_call>") {
				pendingCalls = parseContentToolCalls(reply)
			}

			// Filter out tool calls with empty/missing function names or arguments
			var validCalls []toolCall
			for _, tc := range pendingCalls {
				if tc.Function.Name == "" {
					continue
				}
				if !json.Valid([]byte(tc.Function.Arguments)) {
					fmt.Fprintf(os.Stderr, "\n[warning: model returned invalid tool arguments for %s: %s]\n",
						tc.Function.Name, truncate(tc.Function.Arguments, 100))
					continue
				}
				validCalls = append(validCalls, tc)
			}

			if len(validCalls) == 0 {
				// Normal text reply (or all tool calls were invalid)
				history = append(history, message{Role: "assistant", Content: reply})
				fmt.Println()
				break
			}

			// Assistant made tool calls — add assistant message then execute each
			history = append(history, message{Role: "assistant", ToolCalls: validCalls})

			for _, tc := range validCalls {
				fmt.Printf("\n[tool: %s(%s)]\n", tc.Function.Name, tc.Function.Arguments)
				result := executeTool(tc.Function.Name, tc.Function.Arguments, workDir)
				fmt.Printf("[result: %s]\n", truncate(result, 200))
				history = append(history, message{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
				})
			}
			// Continue loop — send tool results back for the model to respond
		}
	}
}

// parseContentToolCalls extracts tool calls from models that emit them as
// <tool_call>{"name":"...","arguments":{...}}</tool_call> tags in content.
var toolCallTagRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.+?\})\s*</tool_call>`)

func parseContentToolCalls(content string) []toolCall {
	matches := toolCallTagRe.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}
	var calls []toolCall
	for i, m := range matches {
		var parsed struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &parsed); err != nil {
			continue
		}
		// Arguments may be an object (Qwen3 style) or a JSON string (OpenAI style)
		argsStr := string(parsed.Arguments)
		// If it's already an object, use as-is; otherwise it's a string that needs unwrapping
		if len(argsStr) > 0 && argsStr[0] != '{' {
			var s string
			if json.Unmarshal(parsed.Arguments, &s) == nil {
				argsStr = s
			}
		}
		calls = append(calls, toolCall{
			ID:    fmt.Sprintf("content_call_%d", i),
			Type:  "function",
			Index: i,
			Function: functionCall{
				Name:      parsed.Name,
				Arguments: argsStr,
			},
		})
	}
	return calls
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// --- Chat with streaming ---

func chat(url, model string, messages []message, activeTools []tool) (string, []toolCall, error) {
	reqBody := request{
		Model:    model,
		Messages: messages,
		Tools:    activeTools,
		Stream:   true,
	}
	body, _ := json.Marshal(reqBody)

	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(b))
	}

	var full strings.Builder
	var accumulatedCalls []toolCall
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "data: ") {
			data := line[6:]
			if data == "[DONE]" {
				break
			}
			var event streamEvent
			if json.Unmarshal([]byte(data), &event) == nil && len(event.Choices) > 0 {
				d := event.Choices[0].Delta

				// Accumulate text content
				token := d.Content
				if token == "" {
					token = d.ReasoningContent
				}
				if token != "" {
					fmt.Print(token)
					full.WriteString(token)
				}

				// Accumulate tool calls from streamed deltas.
				// Each delta chunk has an index indicating which tool call it belongs to.
				for _, tc := range d.ToolCalls {
					// Grow slice to fit the index
					for tc.Index >= len(accumulatedCalls) {
						accumulatedCalls = append(accumulatedCalls, toolCall{})
					}
					acc := &accumulatedCalls[tc.Index]
					if tc.ID != "" {
						acc.ID = tc.ID
						acc.Index = tc.Index
						acc.Type = tc.Type
					}
					if tc.Function.Name != "" {
						acc.Function.Name = tc.Function.Name
					}
					acc.Function.Arguments += tc.Function.Arguments
				}
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return full.String(), nil, err
		}
	}

	return full.String(), accumulatedCalls, nil
}
