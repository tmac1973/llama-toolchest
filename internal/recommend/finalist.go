package recommend

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Unverified is a model that could not be fully checked, and why. The
// feed lists these rather than dropping them, so a missing favourite has
// an explanation.
type Unverified struct {
	Repo   string `json:"repo"`
	Gated  bool   `json:"gated,omitempty"`
	Reason string `json:"reason"`
}

// The reasons, as the feed shows them.
const (
	reasonFiles       = "The file list could not be read from Hugging Face."
	reasonNoFiles     = "The repo has no model files."
	reasonMeta        = "The model's description could not be read from its file."
	reasonMetaMissing = "The model's description is missing values the estimate needs."
	reasonParams      = "The model's size could not be worked out from its files."
	reasonTimeout     = "Hugging Face did not answer in time. Refresh to try again."
)

func reasonUnsupported(arch string) string {
	return "This llama.cpp build does not know the " + arch + " architecture. A newer build may support it."
}

// verdict is the outcome of checking one finalist: verified (Picks set),
// unverified with a reason, or dropped (neither).
type verdict struct {
	unverified string
	dropped    bool
}

// verify checks one finalist in full: its files, its description and a
// pick at every context class. It fills in the group on success.
func (e *Engine) verify(ctx context.Context, hub Hub, g *Group, p Profile) verdict {
	repo := g.Repo()
	if g.unsupported {
		return verdict{unverified: reasonUnsupported(g.Arch)}
	}
	files, err := e.repoFiles(ctx, hub, repo.ID, repo.SHA)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return verdict{unverified: reasonTimeout}
		}
		return verdict{unverified: reasonFiles}
	}
	probe, ok := modelsource.MetaProbeFile(files)
	if !ok {
		return verdict{unverified: reasonNoFiles}
	}
	meta, err := hub.Meta(ctx, repo.ID, probe)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return verdict{unverified: reasonTimeout}
		}
		return verdict{unverified: reasonMeta}
	}
	if meta.NLayers == 0 || meta.NEmbd == 0 {
		return verdict{unverified: reasonMetaMissing}
	}
	// The header is the authority on what the model is: HuggingFace's
	// summary may have described the image reader or a draft model.
	g.Arch = meta.Architecture
	if !p.supports(g.Arch) {
		return verdict{unverified: reasonUnsupported(g.Arch)}
	}
	if encoderArchs[g.Arch] || models.IsMTPHeadArch(g.Arch) {
		return verdict{dropped: true}
	}
	if meta.ContextLength > 0 {
		g.TrainedCtx = meta.ContextLength
	}
	// The same check for the parameter count: if the largest file is not
	// a plausible quant of a model that size, the count belongs to some
	// other file, and the model's size is worked out from its largest
	// file instead.
	if g.Params = modelsource.RepoParams(files, g.Params); g.Params <= 0 {
		return verdict{unverified: reasonParams}
	}
	if g.Params < minParams {
		return verdict{dropped: true}
	}

	var cands []candidateFile
	for _, f := range files {
		if f.IsMMProj {
			g.Vision = true
			if g.MMProjBytes == 0 || f.Size < g.MMProjBytes {
				g.MMProjBytes = f.Size
			}
			continue
		}
		if f.Size <= 0 || !modelsource.PlausibleFile(f, g.Params) {
			continue
		}
		bpw := modelsource.BitsPerWeight(f, g.Params)
		if bpw < smallestBPW {
			continue
		}
		cands = append(cands, candidateFile{file: f, model: modelsource.PlanModel(meta, f, g.Params), bpw: bpw})
	}
	slices.SortStableFunc(cands, func(a, b candidateFile) int { return cmp.Compare(b.file.Size, a.file.Size) })

	g.Picks = map[models.ContextClass]*Pick{}
	for _, class := range Classes {
		if pick := pickFor(cands, class, p.Hardware); pick != nil {
			g.Picks[class] = pick
		}
	}
	if len(g.Picks) == 0 {
		// Nothing runs well enough at any context: too large, or only
		// with layers on the CPU. Not a failure to check, just no fit.
		return verdict{dropped: true}
	}
	return verdict{}
}

// repoFiles lists a repo's GGUF files, from the disk cache when this
// revision was listed before.
func (e *Engine) repoFiles(ctx context.Context, hub Hub, repo, sha string) ([]modelsource.File, error) {
	path := ""
	if e.Dir != "" && sha != "" {
		path = filepath.Join(e.Dir, "repos", safeName(repo)+"@"+sha+".json")
		if data, err := os.ReadFile(path); err == nil {
			var files []modelsource.File
			if json.Unmarshal(data, &files) == nil && len(files) > 0 {
				return files, nil
			}
		}
	}
	files, err := hub.Files(ctx, repo)
	if err != nil {
		return nil, err
	}
	if path != "" && len(files) > 0 {
		if data, err := json.Marshal(files); err == nil && os.MkdirAll(filepath.Dir(path), 0o755) == nil {
			_ = atomicfile.Write(path, data)
		}
	}
	return files, nil
}

func safeName(repo string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		}
		return '_'
	}, strings.ReplaceAll(repo, "/", "--"))
}
