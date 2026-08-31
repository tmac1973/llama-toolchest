package api

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// What a model really costs, as opposed to what it was predicted to
// cost. The estimate in models.VRAMEstimateForConfigOn is a model of a
// load; this is the load itself, read off the buffer report llama.cpp
// prints while allocating.
//
// The stream it reads is the router's combined log, already broadcast to
// the Server Logs panel. Nothing extra is launched or polled: one
// subscriber, started with the server and running for its lifetime.

// memoryVerbosityRequired is the LLAMA_ARG_LOG_VERBOSITY at which
// llama.cpp starts printing the per-buffer report. Below it there is
// nothing to measure — see internal/config.RuntimeEnvOptions, where the
// setting is exposed, and internal/memreport for the report itself.
const memoryVerbosityRequired = 4

// loadNote is what the configuration was when a load started. Held
// alongside the measurement because a measurement without it says how
// much memory something took without saying what that something was:
// halve the context and the same model measures differently.
//
// Recorded at spawn time rather than read back later, since the user may
// edit the config while the model is still loaded.
type loadNote struct {
	modelID string
	// cards is how many GPUs the load was spread over, needed to turn a
	// tensor-parallel report (which states per-card figures) into totals.
	cards int
	// fingerprint is the memory-relevant part of the launch config.
	fingerprint string
	// baselineGiB and loadedGiB are the card counters summed across GPUs,
	// read when the load started and once it reported ready. Their
	// difference is the quantity the VRAM estimate predicts — which is
	// not the same as the buffer report, since llama.cpp itemises
	// neither context nor allocator overhead.
	baselineGiB float64
	loadedGiB   float64
	// cfg is the config the load ran under, kept whole so a corpus row
	// can be written from it rather than from what is set now.
	cfg *models.ModelConfig
	// contended records that another model was loading during this one,
	// so the card difference covers both. Such a load still measures its
	// own buffers correctly — those are attributed per instance — but its
	// card figure is not evidence of anything.
	contended bool
}

// measuredGiB is the card-counter difference across the load, or zero if
// either end of it is missing.
func (n loadNote) measuredGiB() float64 {
	if n.baselineGiB == 0 || n.loadedGiB == 0 {
		return 0
	}
	return n.loadedGiB - n.baselineGiB
}

// watchRouterMemory feeds every router log line to the collector for the
// life of the server. Started once, from NewServer.
//
// The subscription replays whatever the log broadcaster still holds,
// which is how a measurement survives the router outliving a page load;
// it costs nothing here because the collector keys loads by port and a
// replayed spawn line simply rebuilds the same record.
func (s *Server) watchRouterMemory() {
	ch := s.process.Subscribe()
	for line := range ch {
		switch ev := s.memory.Add(line); ev.Kind {
		case memreport.EventSpawned:
			// Between this line and the buffer report is the whole load,
			// so there is time to resolve the model; doing it here rather
			// than at render time is what pins the note to the
			// configuration the instance was actually launched with, and
			// the card counters to the state before it allocated.
			note := s.loadNoteFor(ev.Model)
			note.baselineGiB = s.gpuUsedGiB()
			note.contended = s.memory.InFlight() > 1
			s.memory.Annotate(ev.Port, note)
		case memreport.EventReady:
			// Reading the counters means waiting for the monitor's next
			// sample. Waiting here would stall the log reader, and a
			// subscriber that falls behind loses lines rather than
			// slowing the producer — so the buffer report of whatever
			// loads next would be the thing lost.
			go s.recordCardCounters(ev.Model, ev.Port)
		}
	}
}

// recordCardCounters completes a load's note with what the GPUs report
// now that it has finished allocating.
func (s *Server) recordCardCounters(model, port string) {
	used, ok := s.gpuUsedAfterFreshSample(monitorSampleWait)
	if !ok {
		return
	}
	meas, found := s.memory.Latest(model)
	if !found || meas.Port != port {
		// The model has been unloaded and loaded again since. The note
		// being completed belongs to a load nobody can ask about now.
		return
	}
	note, _ := meas.Note.(loadNote)
	note.loadedGiB = used
	if s.memory.InFlight() > 0 {
		// Something else started loading while this one finished, so the
		// difference across this load covers more than this model.
		note.contended = true
	}
	s.memory.Annotate(port, note)
}

// monitorSampleWait is how long to wait for the monitor's next reading
// before giving up on the card counters for a load. The monitor polls
// every few seconds; a load that finishes just after a poll waits nearly
// a whole interval.
const monitorSampleWait = 15 * time.Second

// gpuUsedGiB is what the cards report in use right now, summed. Read
// from the monitor's last poll, so it is up to one interval old — which
// is what makes it a baseline rather than a measurement: nothing has
// been allocated yet at the moment it is taken.
func (s *Server) gpuUsedGiB() float64 {
	return usedGiB(s.monitor.Current())
}

// gpuUsedAfterFreshSample waits for a reading taken after this call, so
// the figure includes the allocation that just finished rather than
// whatever the last poll happened to catch mid-load.
func (s *Server) gpuUsedAfterFreshSample(timeout time.Duration) (float64, bool) {
	ch := s.monitor.Subscribe()
	defer s.monitor.Unsubscribe(ch)
	select {
	case m := <-ch:
		return usedGiB(m), true
	case <-time.After(timeout):
		return 0, false
	}
}

func usedGiB(m monitor.Metrics) float64 {
	var total float64
	for _, g := range m.GPU {
		total += float64(g.VRAMUsedMB) / 1024
	}
	return total
}

// loadNoteFor resolves a router model name to the configuration it was
// launched under.
func (s *Server) loadNoteFor(routerName string) loadNote {
	m, cfg := s.findModelByAny(routerName)
	if m == nil {
		return loadNote{}
	}
	// The router reads preset.ini only when it starts, so a child is
	// launched with the config as it stood then — not with an edit made
	// since. Prefer the snapshot for exactly that reason.
	if snap, ok := s.runningConfigFor(m.ID); ok && snap != nil {
		cfg = snap
	}
	// Copied, not referenced: without a snapshot this is the registry's
	// own struct, which the config handler mutates in place.
	launched := *cfg
	return loadNote{
		modelID:     m.ID,
		cards:       models.DeviceCountForConfig(cfg, len(s.monitor.Current().GPU)),
		fingerprint: memoryFingerprint(cfg),
		cfg:         &launched,
	}
}

// memoryFingerprint reduces a config to the fields that change how much
// memory a load takes. Two loads with the same fingerprint are
// comparable; a measurement whose fingerprint no longer matches
// describes a configuration that is no longer set.
//
// Deliberately not a hash: it is compared, never stored, and a readable
// value is worth having in a log line.
func memoryFingerprint(cfg *models.ModelConfig) string {
	if cfg == nil {
		return ""
	}
	return fmt.Sprintf(
		"ctx=%d par=%d b=%d ub=%d kv=%s fa=%t ngl=%d gpus=%s split=%s/%s main=%d ple=%s mmproj=%s/%t spec=%s draft=%s mtp=%s/%t dctx=%d dngl=%d ddev=%s dmoe=%d dkv=%s extra=%s",
		cfg.ContextSize, cfg.Parallel, cfg.EffectiveBatchSize(), cfg.EffectiveUBatchSize(),
		cfg.KVCacheQuant, cfg.FlashAttention, cfg.GPULayers, cfg.GPUAssign,
		cfg.SplitMode, cfg.TensorSplit, cfg.MainGPU, cfg.PLEMode,
		cfg.MmprojPath, cfg.MmprojDisabled,
		cfg.SpecType, cfg.DraftModelPath, cfg.MtpPath, cfg.MtpDisabled,
		cfg.DraftCtxSize, cfg.DraftGPULayers, cfg.DraftDevice, cfg.DraftCPUMoE, cfg.DraftKVCacheQuant,
		cfg.ExtraFlags,
	)
}

// logVerbosity returns the LLAMA_ARG_LOG_VERBOSITY the router will run
// with, which the router passes on to every model instance it starts.
// llama.cpp's own default is 3.
func (s *Server) logVerbosity() int {
	s.cfgMu.Lock()
	pairs := s.cfg.RuntimeEnvPairs()
	s.cfgMu.Unlock()

	for _, kv := range pairs {
		name, value, ok := strings.Cut(kv, "=")
		if !ok || name != "LLAMA_ARG_LOG_VERBOSITY" {
			continue
		}
		if n, err := strconv.Atoi(strings.TrimSpace(value)); err == nil {
			return n
		}
	}
	return 3
}

// measurementFor returns the last load measured for a model. Names are
// tried the way routerStateFor tries them: the registry id is what the
// preset section — and so the router's spawn line — uses, but an
// installation that addresses a model by another name should still find
// its own measurement.
func (s *Server) measurementFor(m *models.Model) (memreport.Measurement, loadNote, bool) {
	if m == nil {
		return memreport.Measurement{}, loadNote{}, false
	}
	for _, name := range []string{s.registry.RouterName(m.ID), m.ID, m.PublicName()} {
		if name == "" {
			continue
		}
		meas, ok := s.memory.Latest(name)
		if !ok {
			continue
		}
		note, _ := meas.Note.(loadNote)
		if note.modelID != "" && note.modelID != m.ID {
			// The name matched a load belonging to another model — an
			// alias two models both answer to, say. Reporting its
			// memory here would be worse than reporting none.
			continue
		}
		return meas, note, true
	}
	return memreport.Measurement{}, loadNote{}, false
}

// memoryTooltip is the text behind one row of the Available Models card:
// what the model took when it was last loaded, next to what it was
// predicted to take, with a plain statement of which one you are reading.
//
// state is the router's word for the model — "loaded", "loading", or
// empty for a model the router knows but has not loaded.
func (s *Server) memoryTooltip(m *models.Model, cfg *models.ModelConfig, state string) string {
	if m == nil {
		return ""
	}
	cards := models.DeviceCountForConfig(cfg, len(s.monitor.Current().GPU))
	estimate := fmt.Sprintf("Estimated for the current settings: %.1f GiB.",
		models.VRAMEstimateForConfigOn(m, cfg, cards))

	meas, note, ok := s.measurementFor(m)
	if !ok || len(meas.Report.Entries) == 0 {
		if state == "loading" {
			return join("Loading. What it really takes is shown once llama.cpp has finished reporting its buffers.", estimate)
		}
		if s.logVerbosity() < memoryVerbosityRequired {
			return join(
				"Not measured. llama.cpp only reports what it allocated when Model loading detail is set to 4 or higher, on the Settings page.",
				estimate)
		}
		return join("Not measured yet — load the model and this shows what llama.cpp allocated for it.", estimate)
	}

	// A tensor-parallel load states its figures per card. Scale before
	// totalling, or a model spread over four GPUs reads as a quarter of
	// its size.
	report := meas.Report.ScaleAggregates(note.cards)
	gpu := gib(report.GPUMiB(true))
	host := gib(report.HostMiB(true))

	lead := "Measured while loading"
	if note.fingerprint != "" && note.fingerprint != memoryFingerprint(cfg) {
		lead = "Measured when it was last loaded, under settings that have since changed — restart the server and load it again to measure the current ones"
	}

	// The three terms are every kind llama.cpp reports, so they add up to
	// the total: recurrent state is part of what the model remembers, and
	// the output buffer is part of what a graph needs to run.
	lines := []string{
		fmt.Sprintf("%s: %.1f GiB of GPU memory — %.1f weights, %.1f KV cache, %.1f working buffers.",
			lead, gpu,
			gib(gpuOf(report, memreport.KindModel)),
			gib(gpuOf(report, memreport.KindKV)+gpuOf(report, memreport.KindRecurrent)),
			gib(gpuOf(report, memreport.KindCompute)+gpuOf(report, memreport.KindOutput))),
	}
	if where := deviceSummary(report, note.cards); where != "" {
		lines = append(lines, where)
	}
	if host >= 0.05 {
		lines = append(lines, fmt.Sprintf("Another %.1f GiB sits in system memory.", host))
	}
	lines = append(lines, estimate)
	lines = append(lines, "The measured figure is llama.cpp's own accounting. A card's reported usage runs a few hundred MiB per GPU above it, for driver overhead llama.cpp does not count.")
	return join(lines...)
}

// gib converts a MiB figure to GiB.
func gib(mib float64) float64 { return mib / 1024 }

// gpuOf totals one kind of allocation on the accelerators only. The host
// side of the same kind is reported separately, since a layer that spilled
// to system memory costs no VRAM.
func gpuOf(r memreport.Report, kind memreport.Kind) float64 {
	return r.SumMiB(func(e memreport.Entry) bool { return e.Kind == kind && e.OnGPU() })
}

// deviceSummary says where the memory went. An aggregate report cannot
// say which card holds what — llama.cpp addresses the group as one
// device — so it reports the spread instead of a breakdown.
func deviceSummary(r memreport.Report, cards int) string {
	var aggregate bool
	parts := map[string]float64{}
	for _, e := range r.Entries {
		if !e.OnGPU() {
			continue
		}
		if e.IsAggregate() {
			aggregate = true
			continue
		}
		parts[e.Device] += e.MiB
	}
	if aggregate {
		if cards > 1 {
			return fmt.Sprintf("Spread evenly across %d GPUs.", cards)
		}
		return ""
	}
	if len(parts) == 0 {
		return ""
	}
	names := make([]string, 0, len(parts))
	for d := range parts {
		names = append(names, d)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("Per GPU: ")
	for i, d := range names {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s %.1f GiB", d, gib(parts[d]))
	}
	b.WriteString(".")
	return b.String()
}

// join builds the tooltip body. Newlines survive in a title attribute in
// every browser this UI targets, and one fact per line is easier to read
// than a paragraph.
func join(lines ...string) string {
	return strings.Join(lines, "\n")
}

// memText is a benchmark run's measured GPU footprint as a table cell.
func memText(m *benchmark.MemorySnapshot) string {
	if m == nil {
		return "—"
	}
	return fmt.Sprintf("%.1f", m.GPUGiB)
}

// memDetail is the breakdown behind that cell, for its tooltip. Every
// figure the run recorded, said in full, because a single number in a
// results table is exactly where a reader needs to know what it counts.
func memDetail(m *benchmark.MemorySnapshot) string {
	if m == nil {
		return "Not measured. llama.cpp reports what it allocated only when Model loading detail is set to 4 or higher, on the Settings page, and runs recorded before that was captured have nothing to show."
	}
	lines := []string{
		fmt.Sprintf("%.1f GiB of GPU memory across %d card(s) — %.1f weights, %.1f KV cache, %.1f working buffers.",
			m.GPUGiB, m.Cards, m.WeightsGiB, m.KVGiB, m.ComputeGiB),
	}
	if m.HostGiB >= 0.05 {
		lines = append(lines, fmt.Sprintf("Another %.1f GiB in system memory.", m.HostGiB))
	}
	switch {
	case m.Contended:
		lines = append(lines, "Another model was loading at the same time, so the card counters could not be attributed to this load. The figures above are this model's own — llama.cpp reports them per instance.")
	case m.CardDeltaGiB > 0:
		lines = append(lines, fmt.Sprintf("The cards themselves reported %.1f GiB, %.1f more than llama.cpp itemises: context and allocator overhead it does not count.",
			m.CardDeltaGiB, m.Unreported()))
	}
	return join(lines...)
}
