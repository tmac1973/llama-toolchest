package builder

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"time"
)

// archSourceFile is where llama.cpp lists the model architectures it can
// load, by their GGUF general.architecture names.
const archSourceFile = "src/llama-arch.cpp"

var (
	archNamesMap   = regexp.MustCompile(`LLM_ARCH_NAMES\s*=\s*\{`)
	archNamesEntry = regexp.MustCompile(`\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}`)
)

// ParseArchNames reads the architecture names out of llama-arch.cpp's
// LLM_ARCH_NAMES table:
//
//	static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
//	    { LLM_ARCH_CLIP,  "clip"  }, // dummy, only used by llama-quantize
//	    { LLM_ARCH_LLAMA, "llama" },
//
// "clip" (a placeholder for the quantize tool) and "(unknown)" are left
// out. Nil when the table is not found, so a change in the file's layout
// upstream reads as "not known" rather than "nothing supported".
func ParseArchNames(src []byte) []string {
	loc := archNamesMap.FindIndex(src)
	if loc == nil {
		return nil
	}
	body := src[loc[1]:]
	if end := bytes.Index(body, []byte("};")); end >= 0 {
		body = body[:end]
	}
	var names []string
	for _, m := range archNamesEntry.FindAllSubmatch(body, -1) {
		name := string(m[1])
		if name == "clip" || name == "(unknown)" {
			continue
		}
		names = append(names, name)
	}
	return names
}

// archsFromCheckout reads the names from a checkout's working tree.
func archsFromCheckout(srcDir string) []string {
	src, err := os.ReadFile(filepath.Join(srcDir, archSourceFile))
	if err != nil {
		return nil
	}
	return ParseArchNames(src)
}

// archsAtCommit reads the names as they were at one commit, for builds
// made before the list was recorded.
func archsAtCommit(srcDir, sha string) []string {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "show", sha+":"+archSourceFile).Output()
	if err != nil {
		return nil
	}
	return ParseArchNames(out)
}

// BackfillArchs fills in the architecture list of successful builds made
// before the list was recorded, from the llama.cpp checkout's history.
// A build whose commit the checkout no longer has keeps an empty list,
// which callers treat as "not known".
func (b *Builder) BackfillArchs() {
	b.mu.Lock()
	var todo []BuildResult
	for _, br := range b.builds {
		if br.Status == BuildStatusSuccess && br.GitSHA != "" && len(br.Archs) == 0 {
			todo = append(todo, br)
		}
	}
	b.mu.Unlock()
	if len(todo) == 0 {
		return
	}

	srcDir := filepath.Join(b.dataDir, "llama.cpp")
	found := map[string][]string{}
	for _, br := range todo {
		if names := archsAtCommit(srcDir, br.GitSHA); len(names) > 0 {
			found[br.ID] = names
		}
	}
	if len(found) == 0 {
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	for i := range b.builds {
		if names, ok := found[b.builds[i].ID]; ok && len(b.builds[i].Archs) == 0 {
			b.builds[i].Archs = names
		}
	}
	b.saveBuilds()
	slog.Info("recorded supported architectures for earlier builds", "builds", len(found))
}
