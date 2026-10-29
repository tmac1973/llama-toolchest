package builder

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"time"
)

// archSourceFile is where llama.cpp lists the model architectures it can
// load, by their GGUF general.architecture names.
const archSourceFile = "src/llama-arch.cpp"

var (
	archNamesMap   = regexp.MustCompile(`LLM_ARCH_NAMES\s*=\s*\{`)
	archNamesEntry = regexp.MustCompile(`\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*"([^"]+)"\s*\}`)
	// smTensorFunc opens the function that says which architectures can
	// be loaded with --split-mode tensor, and smTensorStep finds its case
	// labels and the returns they lead to.
	smTensorFunc = regexp.MustCompile(`bool\s+llm_arch_supports_sm_tensor\s*\([^)]*\)\s*\{`)
	smTensorStep = regexp.MustCompile(`case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:|return\s+(true|false)\s*;`)
)

// archNamesTable reads llama-arch.cpp's LLM_ARCH_NAMES table as
// [enum, name] pairs, in the file's order:
//
//	static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
//	    { LLM_ARCH_CLIP,  "clip"  }, // dummy, only used by llama-quantize
//	    { LLM_ARCH_LLAMA, "llama" },
//
// Nil when the table is not found.
func archNamesTable(src []byte) [][2]string {
	loc := archNamesMap.FindIndex(src)
	if loc == nil {
		return nil
	}
	body := src[loc[1]:]
	if end := bytes.Index(body, []byte("};")); end >= 0 {
		body = body[:end]
	}
	var pairs [][2]string
	for _, m := range archNamesEntry.FindAllSubmatch(body, -1) {
		pairs = append(pairs, [2]string{string(m[1]), string(m[2])})
	}
	return pairs
}

// ParseArchNames reads the architecture names out of llama-arch.cpp's
// LLM_ARCH_NAMES table. "clip" (a placeholder for the quantize tool) and
// "(unknown)" are left out. Nil when the table is not found, so a change
// in the file's layout upstream reads as "not known" rather than "nothing
// supported".
func ParseArchNames(src []byte) []string {
	table := archNamesTable(src)
	if table == nil {
		return nil
	}
	var names []string
	for _, p := range table {
		if p[1] == "clip" || p[1] == "(unknown)" {
			continue
		}
		names = append(names, p[1])
	}
	return names
}

// ParseNoTensorSplitArchs reads which architectures llama.cpp refuses to
// load with --split-mode tensor, from llama-arch.cpp:
//
//	bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
//	    switch (arch) {
//	        case LLM_ARCH_GROK:
//	        case LLM_ARCH_MPT:
//	            return false;
//	        default:
//	            return true;
//	    }
//	}
//
// The case labels that lead to "return false" are the refused ones; they
// are reported by their GGUF names from LLM_ARCH_NAMES. A model of one of
// them fails to load with "LLAMA_SPLIT_MODE_TENSOR not implemented for
// architecture". Nil when the function or the table is not found (a
// build from before tensor split, or a change in the file's layout),
// which callers treat as "not known".
func ParseNoTensorSplitArchs(src []byte) []string {
	loc := smTensorFunc.FindIndex(src)
	if loc == nil {
		return nil
	}
	table := archNamesTable(src)
	if table == nil {
		return nil
	}
	names := make(map[string]string, len(table))
	for _, p := range table {
		names[p[0]] = p[1]
	}
	body := src[loc[1]:]
	// The function ends at the first closing brace at the start of a line.
	if end := bytes.Index(body, []byte("\n}")); end >= 0 {
		body = body[:end]
	}
	refused := []string{}
	var pending []string
	for _, m := range smTensorStep.FindAllSubmatch(body, -1) {
		switch {
		case len(m[1]) > 0:
			pending = append(pending, string(m[1]))
		case string(m[2]) == "false":
			for _, enum := range pending {
				if name, ok := names[enum]; ok && !slices.Contains(refused, name) {
					refused = append(refused, name)
				}
			}
			pending = nil
		default:
			pending = nil
		}
	}
	return refused
}

// readArchs reads both lists from one copy of llama-arch.cpp.
func readArchs(src []byte) (archs, noTensorSplit []string) {
	return ParseArchNames(src), ParseNoTensorSplitArchs(src)
}

// archsFromCheckout reads the lists from a checkout's working tree.
func archsFromCheckout(srcDir string) (archs, noTensorSplit []string) {
	src, err := os.ReadFile(filepath.Join(srcDir, archSourceFile))
	if err != nil {
		return nil, nil
	}
	return readArchs(src)
}

// archsAtCommit reads the lists as they were at one commit, for builds
// made before they were recorded.
func archsAtCommit(srcDir, sha string) (archs, noTensorSplit []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "show", sha+":"+archSourceFile).Output()
	if err != nil {
		return nil, nil
	}
	return readArchs(out)
}

// BackfillArchs fills in the architecture lists of successful builds made
// before they were recorded, from the llama.cpp checkout's history. A
// build whose commit the checkout no longer has keeps empty lists, which
// callers treat as "not known". A build from before tensor split existed
// has no tensor-split list to find, and is read again on each start; that
// is one git show per such build.
func (b *Builder) BackfillArchs() {
	b.mu.Lock()
	var todo []BuildResult
	for _, br := range b.builds {
		if br.Status == BuildStatusSuccess && br.GitSHA != "" && (len(br.Archs) == 0 || len(br.NoTensorSplitArchs) == 0) {
			todo = append(todo, br)
		}
	}
	b.mu.Unlock()
	if len(todo) == 0 {
		return
	}

	srcDir := filepath.Join(b.dataDir, "llama.cpp")
	type lists struct{ archs, noTensorSplit []string }
	found := map[string]lists{}
	for _, br := range todo {
		archs, noTS := archsAtCommit(srcDir, br.GitSHA)
		if (len(br.Archs) == 0 && len(archs) > 0) || (len(br.NoTensorSplitArchs) == 0 && len(noTS) > 0) {
			found[br.ID] = lists{archs, noTS}
		}
	}
	if len(found) == 0 {
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	for i := range b.builds {
		l, ok := found[b.builds[i].ID]
		if !ok {
			continue
		}
		if len(b.builds[i].Archs) == 0 {
			b.builds[i].Archs = l.archs
		}
		if len(b.builds[i].NoTensorSplitArchs) == 0 {
			b.builds[i].NoTensorSplitArchs = l.noTensorSplit
		}
	}
	b.saveBuilds()
	slog.Info("recorded supported architectures for earlier builds", "builds", len(found))
}
