package api

import (
	"fmt"
	"net/http"
	"strconv"
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

// evalDataLogits is one KL reference logits row. The key fields
// round-trip the delete form: the key is parsed back from the filename,
// and re-rendering it reproduces the filename exactly (the
// ParseKLBaseFilename contract), so no path travels in the form.
type evalDataLogits struct {
	Filename   string
	ModelLabel string // repo path (SafeModelID form) of the reference
	Dataset    string
	ChunksText string // "100 chunks" | "full"
	Size       int64
	SizeText   string
	AgeText    string
	// Delete-form fields (the cache key).
	ModelID string
	Quant   string
	Ctx     int
	Chunks  int
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
			row.SizeText = formatBytes(row.Size)
		}
		view.Datasets = append(view.Datasets, row)
	}

	for _, info := range evaluate.ListKLBases(root) {
		k := info.Key
		row := evalDataLogits{
			Filename:   k.Filename(),
			Dataset:    k.Dataset,
			ModelID:    k.ModelID,
			Quant:      k.Quant,
			Ctx:        k.Ctx,
			Chunks:     k.Chunks,
			Size:       info.Size,
			SizeText:   formatBytes(info.Size),
			AgeText:    fmtDurationSince(time.Since(info.ModTime)),
			ChunksText: "full",
		}
		if k.Chunks > 0 {
			row.ChunksText = fmt.Sprintf("%d chunks", k.Chunks)
		}
		// The filename's ModelID is the SafeModelID form (repo with / →
		// --); the label re-reads it as a repo path so the row says what
		// it is, with the exact filename available in the tooltip.
		row.ModelLabel = k.ModelID
		view.Logits = append(view.Logits, row)
		view.TotalSize += info.Size
	}
	if view.TotalSize > 0 {
		view.TotalText = formatBytes(view.TotalSize)
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "eval_data", view)
		return
	}
	respondJSON(w, view)
}

// handleDeleteKLLogits removes one KL reference logits file. The form
// posts the cache key fields (the row's own cache key — the filename is
// deterministic in the key, so no path travels in the form).
//
// POST /api/benchmarks/eval-data/delete-logits
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
	modelID := r.FormValue("model_id")
	quant := r.FormValue("quant")
	dataset := r.FormValue("dataset")
	chunksStr := r.FormValue("chunks")
	ctxStr := r.FormValue("ctx")
	if modelID == "" || quant == "" || dataset == "" || chunksStr == "" || ctxStr == "" {
		http.Error(w, "model_id, quant, dataset, chunks, and ctx are all required", http.StatusBadRequest)
		return
	}
	chunks, err := strconv.Atoi(chunksStr)
	if err != nil || chunks < 0 {
		http.Error(w, "chunks must be a non-negative integer: "+chunksStr, http.StatusBadRequest)
		return
	}
	ctx, err := strconv.Atoi(ctxStr)
	if err != nil || ctx <= 0 {
		http.Error(w, "ctx must be a positive integer: "+ctxStr, http.StatusBadRequest)
		return
	}
	key := evaluate.KLBaseKey{
		ModelID: modelID,
		Quant:   quant,
		Dataset: dataset,
		Chunks:  chunks,
		Ctx:     ctx,
	}

	root := evaluate.EvalDataRoot(s.cfg.DataDir)
	// DeleteKLBase is idempotent (missing file is not an error) — the
	// card's delete button is safe to double-click.
	if err := evaluate.DeleteKLBase(root, key); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
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
