package api

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The VRAM estimate is fitted to measured loads, kept as a corpus in
// internal/models/vram_corpus_test.go. Every point in it was assembled by
// hand: load the model, watch the GPU counters, read the buffer report,
// transcribe a dozen fields into a Go literal without a typo.
//
// The collector already holds all of that for every load. This turns it
// into the row, so adding evidence costs a copy and paste rather than an
// afternoon — which is the difference between a corpus that grows and one
// that stays at eleven points from one machine.
//
// GET /api/models/{id}/vram-corpus

// handleVRAMCorpus returns a ready-to-paste corpus row for the model's
// last measured load.
func (s *Server) handleVRAMCorpus(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	meas, note, ok := s.measurementFor(m)
	if !ok || len(meas.Report.Entries) == 0 {
		msg := "no measured load for this model — load it and try again"
		if s.logVerbosity() < memoryVerbosityRequired {
			msg = "no measured load for this model: llama.cpp reports what it allocated only at Model loading detail 4 or higher, on the Settings page"
		}
		http.Error(w, msg, http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	fmt.Fprint(w, corpusRow(m, meas, note, s.runningBuild()))
}

// corpusRow renders one measured load as a corpusPoint literal.
//
// Deliberately a string of Go source rather than JSON: the corpus is
// source, the reviewer of a change to it reads Go, and a row that has to
// be converted before it can be pasted is a row that will not be added.
func corpusRow(m *models.Model, meas memreport.Measurement, note loadNote, build string) string {
	cfg := note.cfg
	if cfg == nil {
		cfg = &models.ModelConfig{}
	}
	cards := note.cards
	if cards < 1 {
		cards = 1
	}
	report := meas.Report.ScaleAggregates(cards)

	ctx := cfg.ContextSize
	if ctx == 0 {
		ctx = m.ContextLength
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// %s, %d card(s)", m.PublicName(), cards)
	if build != "" {
		fmt.Fprintf(&b, ", build %s", build)
	}
	fmt.Fprintf(&b, ", measured %s.\n", meas.At.Format("2006-01-02"))

	measured := note.measuredGiB()
	reported := gib(report.GPUMiB(true))
	switch {
	case note.contended:
		b.WriteString("// NOT USABLE AS MEASURED: another model was loading at the same time, so the\n")
		b.WriteString("// card figure below covers both. The buffer report is still this model's own.\n")
		b.WriteString("// Load this model on its own and export again.\n")
	case measured <= 0:
		b.WriteString("// The card counters were not captured for this load, so the measured column\n")
		b.WriteString("// is 0 and has to be filled in by hand. It should be a little above the\n")
		b.WriteString("// buffer report — context and allocator overhead are not itemised.\n")
	default:
		fmt.Fprintf(&b, "// Card counters rose %.2f GiB; llama.cpp itemised %.2f, leaving %.2f unreported.\n",
			measured, reported, measured-reported)
	}
	if m.NLayers == 0 || m.NEmbd == 0 || (m.KVFullPerTok == 0 && m.KVSWAPerTok == 0) {
		b.WriteString("// INCOMPLETE: this registry record has no parsed GGUF metadata, so the estimate\n")
		b.WriteString("// falls back to file size alone and the row cannot test the term model. Rescan\n")
		b.WriteString("// the model, load it again, and export again.\n")
	}
	if host := gib(report.HostMiB(true)); host >= 0.05 {
		fmt.Fprintf(&b, "// A further %.2f GiB went to system memory, which no VRAM figure counts.\n", host)
	}

	fmt.Fprintf(&b, "{%q, %s,\n\t%s, %d, %.2f,\n\t&reportedTerms{weights: %.2f, kv: %.2f, recurrent: %.2f, compute: %.2f}},\n",
		fmt.Sprintf("%s ctx%s ub%d", m.PublicName(), ctxLabel(ctx), cfg.EffectiveUBatchSize()),
		modelLiteral(m),
		configLiteral(cfg),
		cards,
		measured,
		gib(gpuOf(report, memreport.KindModel)),
		gib(gpuOf(report, memreport.KindKV)),
		gib(gpuOf(report, memreport.KindRecurrent)),
		gib(gpuOf(report, memreport.KindCompute)+gpuOf(report, memreport.KindOutput)))
	return b.String()
}

// ctxLabel shortens a context length the way the corpus names it.
func ctxLabel(ctx int) string {
	if ctx >= 1024 && ctx%1024 == 0 {
		return fmt.Sprintf("%dk", ctx/1024)
	}
	return fmt.Sprintf("%d", ctx)
}

// modelLiteral writes the registry fields the estimate reads, and only
// those: a row carrying fields the estimator ignores invites the reader
// to think they were part of the measurement.
func modelLiteral(m *models.Model) string {
	var f []string
	add := func(format string, args ...any) { f = append(f, fmt.Sprintf(format, args...)) }

	add("SizeBytes: gibBytes(%.2f)", models.BytesToGiB(m.SizeBytes))
	// Zero fields are omitted rather than written out: a zero in the
	// registry means "never parsed", and writing it into the corpus would
	// record an absence as a measurement.
	if m.PLEBytes > 0 {
		add("PLEBytes: gibBytes(%.2f)", models.BytesToGiB(m.PLEBytes))
	}
	if m.TokenEmbdBytes > 0 {
		add("TokenEmbdBytes: gibBytes(%.3f)", models.BytesToGiB(m.TokenEmbdBytes))
	}
	if m.NLayers > 0 {
		add("NLayers: %d", m.NLayers)
	}
	if m.AttnLayers > 0 {
		add("AttnLayers: %d", m.AttnLayers)
	}
	if m.NEmbd > 0 {
		add("NEmbd: %d", m.NEmbd)
	}
	if m.NKVHead > 0 {
		add("NKVHead: %d", m.NKVHead)
	}
	if m.KVFullPerTok > 0 {
		add("KVFullPerTok: %d", m.KVFullPerTok)
	}
	if m.KVSWAPerTok > 0 {
		add("KVSWAPerTok: %d", m.KVSWAPerTok)
	}
	if m.SlidingWindow > 0 {
		add("SlidingWindow: %d", m.SlidingWindow)
	}
	if m.IndexerKeyLength > 0 {
		add("IndexerKeyLength: %d", m.IndexerKeyLength)
	}
	if m.ContextLength > 0 {
		add("ContextLength: %d", m.ContextLength)
	}
	return "Model{" + strings.Join(f, ", ") + "}"
}

// configLiteral writes the launch settings that move the estimate.
func configLiteral(cfg *models.ModelConfig) string {
	f := []string{
		fmt.Sprintf("ContextSize: %d", cfg.ContextSize),
		fmt.Sprintf("UBatchSize: %d", cfg.EffectiveUBatchSize()),
	}
	if cfg.KVCacheQuant != "" {
		f = append(f, fmt.Sprintf("KVCacheQuant: %q", cfg.KVCacheQuant))
	}
	if cfg.MmprojPath != "" && !cfg.MmprojDisabled {
		f = append(f, fmt.Sprintf("MmprojPath: %q", cfg.MmprojPath))
	}
	if cfg.MtpPath != "" && !cfg.MtpDisabled {
		f = append(f, fmt.Sprintf("MtpPath: %q", cfg.MtpPath))
	}
	if cfg.SpecType == "draft" && cfg.DraftModelPath != "" {
		f = append(f, fmt.Sprintf("SpecType: %q, DraftModelPath: %q", cfg.SpecType, cfg.DraftModelPath))
	}
	return "ModelConfig{" + strings.Join(f, ", ") + "}"
}
