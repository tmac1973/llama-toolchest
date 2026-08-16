package api

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// The Evaluation Data card: the pinned datasets (state + license, from
// phase 02's pinned table) and the KL reference logits cache (list +
// per-row delete). It sits beneath the jobs list on the Benchmarks tab
// and is the only place the user manages what the evaluations keep on
// disk — both sections regenerate automatically when a job needs them,
// which the card says so the user knows the delete is safe.

// evalDataDataset is one dataset row: the pinned table's identity plus
// the on-disk state from evaluate.Verify.
type evalDataDataset struct {
	Name     string
	License  string
	Size     int64
	SizeText string
	State    string // "not downloaded" | "verified" | "present, hash mismatch"
}

// evalDataLogits is one KL reference logits row. The delete form posts
// the row's Filename, which the handler re-validates as a name (it must
// parse as a KL base filename and resolve inside the logits directory)
// rather than trusting it as a path.
type evalDataLogits struct {
	Filename   string
	ModelLabel string // reference repo path, "/" restored from SafeModelID
	Quant      string
	Dataset    string
	ChunksText string // "100 chunks" | "full"
	Size       int64
	SizeText   string
	AgeText    string
}

// evalDataView is the card partial's data.
type evalDataView struct {
	Datasets  []evalDataDataset
	Logits    []evalDataLogits
	TotalSize int64
	TotalText string
}

// handleEvalData returns the card partial (HTMX) or its JSON twin.
//
// GET /api/benchmarks/eval-data
func (s *Server) handleEvalData(w http.ResponseWriter, r *http.Request) {
	root := evaluate.EvalDataRoot(s.cfg.DataDir)

	view := evalDataView{
		Datasets: make([]evalDataDataset, 0, len(evaluate.Datasets())),
	}
	for _, st := range evaluate.Verify(root) {
		row := evalDataDataset{
			Name:    st.Name,
			License: st.License,
			Size:    st.Size,
		}
		switch {
		case !st.Present:
			row.State = "not downloaded"
			row.Size = 0
		case st.Verified:
			row.State = "verified"
		default:
			// Present but not hash-verified: the pinned SHA-256 does not
			// match. Named, not styled as a generic error — the file is
			// likely a manual copy the user made.
			row.State = "present, hash mismatch"
		}
		if row.Size > 0 {
			row.SizeText = evaluate.FormatBytes(row.Size)
		}
		view.Datasets = append(view.Datasets, row)
	}

	for _, info := range evaluate.ListKLBases(root) {
		k := info.Key
		row := evalDataLogits{
			// The name on disk, not a re-render of the key: a legacy
			// pre-fingerprint entry re-renders differently, and the
			// delete has to name the file that is actually there.
			Filename: filepath.Base(info.Path),
			Dataset:  k.Dataset,
			Quant:    k.Quant,
			Size:     info.Size,
			SizeText: evaluate.FormatBytes(info.Size),
			AgeText:  fmtDurationSince(time.Since(info.ModTime)),
			// The filename's ModelID is the SafeModelID form (repo with
			// "/" → "--"); the label restores the slash so the row reads
			// as the repo path it is, with the exact filename in the
			// tooltip.
			ModelLabel: strings.ReplaceAll(k.ModelID, "--", "/"),
			ChunksText: "full",
		}
		if k.Chunks > 0 {
			row.ChunksText = fmt.Sprintf("%d chunks", k.Chunks)
		}
		view.Logits = append(view.Logits, row)
		view.TotalSize += info.Size
	}
	if view.TotalSize > 0 {
		view.TotalText = evaluate.FormatBytes(view.TotalSize)
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "eval_data", view)
		return
	}
	respondJSON(w, view)
}

// handleDeleteKLLogits removes one KL reference logits file, named by
// the row's filename.
//
// POST /api/benchmarks/eval-data/delete-logits
//
// The name is form input, so it is validated as a NAME and never
// trusted as a path: DeleteKLBaseFile rejects anything carrying a
// directory part, anything that does not parse as a KL base filename,
// and anything that would resolve outside the logits directory.
//
// Deletion is refused with 409 while a job is running, using the same
// busy message the restore flow uses for the same conflict: a running
// job may be generating the very file being deleted, or about to read
// it.
func (s *Server) handleDeleteKLLogits(w http.ResponseWriter, r *http.Request) {
	if s.routerBusyWithJob() {
		http.Error(w, "a benchmark job is currently running — cancel it before deleting evaluation data", http.StatusConflict)
		return
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form: "+err.Error(), http.StatusBadRequest)
		return
	}
	filename := r.FormValue("filename")
	if filename == "" {
		http.Error(w, "filename is required", http.StatusBadRequest)
		return
	}

	root := evaluate.EvalDataRoot(s.cfg.DataDir)
	// DeleteKLBaseFile is idempotent (missing file is not an error) —
	// the card's delete button is safe to double-click.
	if err := evaluate.DeleteKLBaseFile(root, filename); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// HTMX: the refreshed card is the response, which re-renders the
	// row without the deleted entry.
	if isHTMX(r) {
		s.handleEvalData(w, r)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
