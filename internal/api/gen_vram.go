package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// genVRAM is what generation backends hold and reserve on one GPU.
type genVRAM struct {
	// Reserved is what the enabled instances reserve: each one's
	// reservation, or what it actually uses when that is more.
	Reserved int
	// Using is what the running instances actually hold, which the
	// monitor counts as other programs' memory until it is taken out.
	Using int
	// Names lists each reserving instance and its share, for messages.
	Names []string
}

// genVRAMByGPU sums the generation backends' memory per GPU index,
// leaving out exclude (an instance being checked against the others).
// Empty while generation backends are off.
func (s *Server) genVRAMByGPU(exclude string) map[int]*genVRAM {
	return s.genVRAM(exclude, false)
}

// genVRAM is genVRAMByGPU, leaving out the instances that are not
// running when activeOnly is set.
func (s *Server) genVRAM(exclude string, activeOnly bool) map[int]*genVRAM {
	out := map[int]*genVRAM{}
	if s.cfg == nil || !s.cfg.GenerationEnabled || s.gen == nil || s.genMgr == nil {
		return out
	}
	for _, inst := range s.gen.Instances() {
		if inst.ID == exclude {
			continue
		}
		using := 0
		if pid := s.genMgr.PID(inst.ID); pid > 0 {
			if mib, ok := s.genProcVRAM.read(pid); ok {
				using = mib
			} else {
				using = inst.ReservedMiB
			}
		}
		if !inst.Enabled && using == 0 {
			continue
		}
		if activeOnly && using == 0 && !s.genMgr.IsActive(inst.ID) {
			continue
		}
		v := out[inst.GPU]
		if v == nil {
			v = &genVRAM{}
			out[inst.GPU] = v
		}
		v.Using += using
		share := max(inst.ReservedMiB, using)
		if inst.Enabled || using > 0 {
			v.Reserved += share
			v.Names = append(v.Names, fmt.Sprintf("%s %s", inst.Name, mibText(share)))
		}
	}
	return out
}

// applyGenVRAM folds the generation backends into the planner's view of
// the GPUs: what they use leaves "other programs", and what they reserve
// is set apart. With generation backends off this changes nothing.
func (s *Server) applyGenVRAM(hw *models.Hardware) {
	byGPU := s.genVRAMByGPU("")
	for i := range hw.GPUs {
		if v := byGPU[hw.GPUs[i].Index]; v != nil {
			hw.GPUs[i].OtherUsedMiB = max(0, hw.GPUs[i].OtherUsedMiB-v.Using)
			hw.GPUs[i].ReservedMiB = v.Reserved
		}
	}
}

// loadedLLMMiB is the planned GPU memory of the LLMs the router has
// loaded, per GPU index, plus extraID (a model about to load) when set.
// skipLoaded leaves the loaded ones out: with --models-max 1 the loaded
// model is replaced by the new one. gpuMiB is each GPU's room
// (models.GPUMiB), which decides how a model is split over the cards:
// from idleHardware, the room the preset was balanced for.
func (s *Server) loadedLLMMiB(gpuMiB []int, extraID string, skipLoaded bool) map[int]int {
	out := map[int]int{}
	if len(gpuMiB) == 0 {
		return out
	}
	add := func(id string) {
		m, err := s.registry.Get(id)
		if err != nil || m == nil {
			return
		}
		cfg, err := s.registry.GetConfig(m.ID)
		if err != nil {
			return
		}
		addLLMMiB(out, m, cfg, gpuMiB)
	}
	if !skipLoaded && s.process.IsRunning() {
		if list, err := s.process.ListModels(); err == nil {
			for _, rm := range list {
				if rm.Status.Value != "loaded" && rm.Status.Value != "loading" {
					continue
				}
				if reg, _ := s.resolveRouterModel(rm); reg != nil && reg.ID != extraID {
					add(reg.ID)
				}
			}
		}
	}
	if extraID != "" {
		add(extraID)
	}
	return out
}

// addLLMMiB adds a model's planned GPU memory under cfg to out, per GPU,
// split over the cards the way its launch will split it (see
// models.CardLoadsMiB): by each card's room where that differs, so a card
// with a generation reservation is given less.
func addLLMMiB(out map[int]int, m *models.Model, cfg *models.ModelConfig, gpuMiB []int) {
	for g, mib := range models.CardLoadsMiB(m, cfg, gpuMiB) {
		out[g] += mib
	}
}

// genFitCheck checks that an instance reserving reserveMiB on its GPU fits
// next to the loaded LLMs and the other instances' reservations, within
// the planner's budget for the card. The error names what uses the card.
func (s *Server) genFitCheck(inst generation.Instance, reserveMiB int) error {
	if reserveMiB <= 0 {
		return nil
	}
	hw := s.hardware()
	var card *models.GPUSpec
	for i := range hw.GPUs {
		if hw.GPUs[i].Index == inst.GPU {
			card = &hw.GPUs[i]
		}
	}
	if card == nil {
		// No GPU telemetry; nothing to check against.
		return nil
	}
	// hardware() counts this instance among the reservations; check it
	// against the others only.
	others := s.genFitVRAMByGPU(inst.ID)[inst.GPU]
	spec := *card
	spec.ReservedMiB = 0
	var otherNames []string
	if others != nil {
		spec.ReservedMiB = others.Reserved
		otherNames = others.Names
	}
	budget := models.CardBudgetMiB(spec)
	llm := s.loadedLLMMiB(models.GPUMiB(s.idleHardware()), "", false)[inst.GPU]
	if llm+reserveMiB <= budget {
		return nil
	}
	parts := []string{fmt.Sprintf("other programs %s", mibText(spec.OtherUsedMiB))}
	if llm > 0 {
		parts = append(parts, "loaded LLMs "+mibText(llm))
	}
	parts = append(parts, otherNames...)
	return fmt.Errorf("not enough VRAM on GPU %d (%s): %s needs %s, but only %s is left after %s and the safety margin",
		inst.GPU, mibText(spec.VRAMTotalMiB), inst.Name, mibText(reserveMiB), mibText(max(0, budget-llm)), strings.Join(parts, ", "))
}

// genLLMFitError is returned when loading an LLM on demand would crowd out
// the generation backends' reservations.
type genLLMFitError struct{ msg string }

func (e *genLLMFitError) Error() string { return e.msg }

// checkLLMFitsReservations refuses loading an LLM, with its saved config,
// into the router beside what it already has loaded when that would not
// fit next to the generation backends' reservations on a GPU it uses.
// Used for loads on request and the Models page's Load button.
//
// With --models-max 1 the load replaces the loaded model, so the cards
// are taken as they were with nothing of ours loaded: the loaded model's
// own unreported memory is freed with it.
func (s *Server) checkLLMFitsReservations(m *models.Model) error {
	hw, need, replacing := s.llmReservationNeed(m)
	return s.reservationFitError(hw, need, !replacing)
}

// llmReservationNeed is what checkLLMFitsReservations checks: the cards,
// and the planned LLM memory per GPU with m loaded. replacing is set when
// the load replaces the loaded model (--models-max 1). need is nil when
// no reservation can limit the load.
func (s *Server) llmReservationNeed(m *models.Model) (hw models.Hardware, need map[int]int, replacing bool) {
	if !s.genReservationsApply() {
		return models.Hardware{}, nil, false
	}
	replacing = s.cfg.ModelsMax == 1
	idle := s.idleHardware()
	hw = idle
	if !replacing {
		hw = s.hardware()
	}
	return hw, s.loadedLLMMiB(models.GPUMiB(idle), m.ID, replacing), replacing
}

// checkConfigFitsReservations is checkLLMFitsReservations for a benchmark
// cell: m under cfg, alone, because a job restarts the router with only
// the model it measures, on cards as they were with nothing of ours
// loaded.
func (s *Server) checkConfigFitsReservations(m *models.Model, cfg *models.ModelConfig) error {
	if !s.genReservationsApply() {
		return nil
	}
	hw := s.idleHardware()
	need := map[int]int{}
	addLLMMiB(need, m, cfg, models.GPUMiB(hw))
	return s.reservationFitError(hw, need, false)
}

// genReservationsApply reports whether any generation reservation can
// limit LLM loads.
func (s *Server) genReservationsApply() bool {
	return s.cfg != nil && s.cfg.GenerationEnabled && s.gen != nil && len(s.genFitVRAMByGPU("")) > 0
}

// reservationFitError is a *genLLMFitError for the first GPU where need
// (planned LLM memory per GPU) is more than the card leaves beside its
// generation reservations, or nil when it fits everywhere.
func (s *Server) reservationFitError(hw models.Hardware, need map[int]int, suggestUnload bool) error {
	byGPU := s.genFitVRAMByGPU("")
	for _, g := range hw.GPUs {
		v := byGPU[g.Index]
		if v == nil || v.Reserved == 0 {
			continue
		}
		g.ReservedMiB = v.Reserved
		budget := models.CardBudgetMiB(g)
		if n, ok := need[g.Index]; ok && n > budget {
			msg := fmt.Sprintf("Not enough VRAM: %.1f GiB reserved for generation backends on GPU %d (%s); the LLM needs about %s there and %s is left",
				float64(v.Reserved)/1024, g.Index, strings.Join(v.Names, ", "), mibText(n), mibText(max(0, budget)))
			if suggestUnload {
				msg += "; unload a model first"
			}
			return &genLLMFitError{msg: msg}
		}
	}
	return nil
}

// reservationShortGPUs lists the GPUs where need is more than the card
// leaves beside its generation reservations, as reservationFitError
// counts them.
func (s *Server) reservationShortGPUs(hw models.Hardware, need map[int]int) []int {
	byGPU := s.genFitVRAMByGPU("")
	var out []int
	for _, g := range hw.GPUs {
		v := byGPU[g.Index]
		if v == nil || v.Reserved == 0 {
			continue
		}
		g.ReservedMiB = v.Reserved
		if n, ok := need[g.Index]; ok && n > models.CardBudgetMiB(g) {
			out = append(out, g.Index)
		}
	}
	return out
}

// procVRAMCache keeps per-process VRAM readings for a moment: hardware()
// runs for every model row of some pages, and on NVIDIA each reading
// starts nvidia-smi.
type procVRAMCache struct {
	mu    sync.Mutex
	byPID map[int]procVRAMReading
}

type procVRAMReading struct {
	at  time.Time
	mib int
	ok  bool
}

const procVRAMCacheFor = 2 * time.Second

func (c *procVRAMCache) read(pid int) (int, bool) {
	c.mu.Lock()
	if r, ok := c.byPID[pid]; ok && time.Since(r.at) < procVRAMCacheFor {
		c.mu.Unlock()
		return r.mib, r.ok
	}
	c.mu.Unlock()
	mib, ok := monitor.ProcessVRAMMiB(pid)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.byPID == nil || len(c.byPID) > 64 {
		c.byPID = map[int]procVRAMReading{}
	}
	c.byPID[pid] = procVRAMReading{at: time.Now(), mib: mib, ok: ok}
	return mib, ok
}

// genReservationAllocations lists each enabled instance's reservation as
// a GPU map entry on its GPU, so the map and its legend show them. Their
// colors continue the map's palette from firstColor, the number of
// entries already on the map, so every entry has a color of its own; the
// map draws them striped to set them apart from LLMs.
func (s *Server) genReservationAllocations(firstColor int) []models.GPUAllocation {
	if s.cfg == nil || !s.cfg.GenerationEnabled || s.gen == nil {
		return nil
	}
	var out []models.GPUAllocation
	for _, inst := range s.gen.Instances() {
		if !inst.Enabled || inst.ReservedMiB == 0 {
			continue
		}
		gb := float64(inst.ReservedMiB) / 1024
		out = append(out, models.GPUAllocation{
			ModelID: "gen:" + inst.ID, ModelName: inst.Name + " (reserved)",
			GPUs: []int{inst.GPU}, PerGPUGB: gb, TotalGB: gb,
			Color: models.AllocationColor(firstColor + len(out)), Reserved: true,
		})
	}
	return out
}

// genBannerGPU is a GPU where generation and loaded LLMs share the card.
type genBannerGPU struct {
	Index int
	Name  string
}

// genBannerGPUs lists the GPUs where an image or music job is running
// while the router has an LLM loaded there, so chat on them is slowed.
func (s *Server) genBannerGPUs() []genBannerGPU {
	if s.cfg == nil || !s.cfg.GenerationEnabled || s.gen == nil || s.genQueue == nil {
		return nil
	}
	hw := s.idleHardware()
	return bannerGPUs(hw.GPUs, func(gpu int) bool { return s.genQueue.Snapshot(gpu).Running != nil },
		s.loadedLLMMiB(models.GPUMiB(hw), "", false))
}

// bannerGPUs picks the GPUs with a generation job running and LLM memory
// loaded.
func bannerGPUs(gpus []models.GPUSpec, jobRunning func(int) bool, llmMiB map[int]int) []genBannerGPU {
	var out []genBannerGPU
	for _, g := range gpus {
		if jobRunning(g.Index) && llmMiB[g.Index] > 0 {
			out = append(out, genBannerGPU{Index: g.Index, Name: g.Name})
		}
	}
	return out
}

func (s *Server) handleGenBanner(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	if gpus := s.genBannerGPUs(); len(gpus) > 0 {
		s.renderPartial(w, "gen_banner", gpus)
	}
}

func mibText(mib int) string {
	return fmt.Sprintf("%.1f GiB", float64(mib)/1024)
}

// suggestReservation is the reservation a measured peak suggests: 10%
// more, rounded up to the next 512 MiB.
func suggestReservation(peakMiB int) int {
	n := peakMiB * 11 / 10
	return (n + 511) / 512 * 512
}

// genMeasureState remembers how each instance's last measured test run
// ended, so the instance table can say so: what was measured and what
// happened to the reservation, or why it failed. It lasts until the next
// measurement or a restart of llama-toolchest.
type genMeasureState struct {
	mu       sync.Mutex
	outcomes map[string]measureOutcome
}

// measureOutcome is one finished measurement: Note on success, Err on
// failure.
type measureOutcome struct {
	Note string
	Err  string
	At   time.Time
}

func (g *genMeasureState) set(id string, o measureOutcome) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.outcomes == nil {
		g.outcomes = map[string]measureOutcome{}
	}
	g.outcomes[id] = o
}

func (g *genMeasureState) clear(id string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.outcomes, id)
}

func (g *genMeasureState) get(id string) measureOutcome {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.outcomes[id]
}

// measureNote says what a successful measurement found and what it did to
// the reservation. before is the reservation before the run; inst is the
// instance as saved after it. restarted and started say whether it was
// running again afterwards: restarted because it was running before,
// started because Start asked for the measurement.
func measureNote(before int, inst generation.Instance, restarted, started bool) string {
	note := "Peak " + mibText(inst.MeasuredPeakMiB)
	if inst.MeasureEstimated {
		note += " (estimated from the whole GPU's use)"
	}
	note += ". "
	suggest := suggestReservation(inst.MeasuredPeakMiB)
	switch {
	case before == 0:
		note += "Reservation set to " + mibText(inst.ReservedMiB) + "."
	case before == suggest:
		note += "Reservation stays at " + mibText(before) + ", as suggested."
	default:
		note += "Reservation stays at " + mibText(before) + "; this measurement suggests " + mibText(suggest) + "."
	}
	switch {
	case restarted:
		note += " Restarted."
	case started:
		note += " Started."
	}
	return note
}

// measureSongRequest is the fixed request a music instance is measured
// with: the one measured on asmodean on 2026-10-24 (a 60-second song).
var measureSongRequest = map[string]any{
	"caption":         "Upbeat indie folk song with acoustic guitar, warm male vocals, hand claps and a catchy chorus",
	"lyrics":          "[Verse]\nMorning light across the kitchen floor\nCoffee steaming like it did before\n[Chorus]\nWe keep on singing, singing through the day\nLittle songs to chase the clouds away\n[Verse]\nOld guitar is leaning by the door\nEvery string remembers something more\n[Chorus]\nWe keep on singing, singing through the day\nLittle songs to chase the clouds away",
	"duration":        60,
	"vocal_language":  "en",
	"use_cot_caption": false,
	"seed":            42,
}

// handleGenMeasure starts a measured test run in the background; the
// instance table shows its progress and outcome. With start=1 (Start on
// an instance that has no reservation yet) the instance is started once
// the measurement has set one.
func (s *Server) handleGenMeasure(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst, ok := s.genInstance(w, r, store)
	if !ok {
		return
	}
	if err := s.canMeasure(inst); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	s.genMgr.SetMeasuring(inst.ID, true)
	s.genMeasure.clear(inst.ID)
	before, wasRunning := inst.ReservedMiB, s.genMgr.IsActive(inst.ID)
	startAfter := r.FormValue("start") == "1"
	go func() {
		defer s.genMgr.SetMeasuring(inst.ID, false)
		if err := s.measureInstance(inst.ID, startAfter); err != nil {
			slog.Warn("generation instance measurement failed", "instance", inst.ID, "error", err)
			s.genMeasure.set(inst.ID, measureOutcome{Err: err.Error(), At: time.Now()})
			return
		}
		after, err := s.gen.Instance(inst.ID)
		if err != nil {
			return
		}
		s.genMeasure.set(inst.ID, measureOutcome{Note: measureNote(before, after, wasRunning, startAfter && !wasRunning), At: time.Now()})
	}()
	s.genInstancesChanged(w, r)
}

// canMeasure refuses a measured test run while anything else uses the
// instance's GPU: the peak must be this instance's alone.
func (s *Server) canMeasure(inst generation.Instance) error {
	if s.genMgr.IsMeasuring(inst.ID) {
		return errors.New("this instance is already being measured")
	}
	if err := s.genMove.busy(); err != nil {
		return err
	}
	if reason := s.gpuBusyReason(); reason != "" {
		return fmt.Errorf("cannot measure while %s", reason)
	}
	if s.genQueue != nil && s.genQueue.Snapshot(inst.GPU).Busy() {
		return fmt.Errorf("measuring needs GPU %d to itself; wait for its image and music jobs to finish", inst.GPU)
	}
	var busy []string
	for _, other := range s.gen.Instances() {
		if other.ID != inst.ID && other.GPU == inst.GPU && s.genMgr.IsActive(other.ID) {
			busy = append(busy, other.Name)
		}
	}
	if len(busy) > 0 {
		return fmt.Errorf("measuring needs GPU %d to itself; stop %s first", inst.GPU, strings.Join(busy, ", "))
	}
	return nil
}

// measureInstance runs the measured test run: start the instance without
// its VRAM limit, run one fixed generation, record the peak, and set or
// suggest a reservation. The instance is left as it was found: running
// again (with its limit) if it was running, stopped otherwise. With
// startAfter it is started afterwards either way.
func (s *Server) measureInstance(id string, startAfter bool) error {
	inst, err := s.gen.Instance(id)
	if err != nil {
		return err
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return err
	}
	wasRunning := s.genMgr.IsActive(id)
	if wasRunning {
		if err := s.genMgr.Stop(id); err != nil {
			return err
		}
	}
	time.Sleep(2 * time.Second)
	baseline := s.gpuUsedMiB(inst.GPU)

	// The first start has no reservation to check yet; it needs at least
	// 1 GiB of the card's budget free.
	if err := s.genFitCheck(inst, 1024); err != nil {
		return err
	}
	if err := s.launchGenInstance(inst, set, generation.LaunchOptions{OmitMaxVRAM: true}); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, id); err != nil {
		return fmt.Errorf("the instance did not start: %w", err)
	}

	// Per-process readings win whenever any sample had one: a backend
	// such as ace-server only opens the GPU on its first request, so the
	// samples before that have no per-process entry yet. The whole-GPU
	// figure is used only when no sample had one (Vulkan, or PIDs hidden
	// in a container).
	var procPeak, gpuPeak int
	var procOK bool
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		tick := time.NewTicker(300 * time.Millisecond)
		defer tick.Stop()
		for {
			if mib, ok := monitor.ProcessVRAMMiB(s.genMgr.PID(id)); ok {
				procPeak, procOK = max(procPeak, mib), true
			}
			gpuPeak = max(gpuPeak, s.gpuUsedMiB(inst.GPU)-baseline)
			select {
			case <-stop:
				return
			case <-tick.C:
			}
		}
	}()
	runErr := s.runMeasureGeneration(ctx, inst, set)
	close(stop)
	<-sampled
	peak, perProcess := gpuPeak, procOK
	if procOK {
		peak = procPeak
	}
	s.genMgr.Stop(id)
	// The fit check below reads the card's used VRAM. Until the monitor's
	// next reading after the backend has exited, that still includes the
	// backend itself, which then counts as another program and makes its
	// own reservation not fit.
	waitGPUReleased(func() int { return s.gpuUsedMiB(inst.GPU) }, baseline, 20*time.Second, 500*time.Millisecond)
	if runErr != nil {
		return fmt.Errorf("the test generation failed: %w", runErr)
	}
	if peak <= 0 {
		return errors.New("no GPU memory use was seen during the test generation")
	}

	inst, err = s.gen.Instance(id)
	if err != nil {
		return err
	}
	inst.MeasuredPeakMiB, inst.MeasuredAt, inst.MeasureEstimated = peak, time.Now(), !perProcess
	suggest := suggestReservation(peak)
	var fitErr error
	if inst.ReservedMiB == 0 {
		if fitErr = s.genFitCheck(inst, suggest); fitErr == nil {
			inst.ReservedMiB = suggest
		} else {
			inst.Enabled = false
		}
	}
	if err := s.gen.UpdateInstance(inst); err != nil {
		return err
	}
	slog.Info("generation instance measured", "instance", id, "peak_mib", peak, "estimated", !perProcess, "reserved_mib", inst.ReservedMiB)
	if fitErr != nil {
		return fmt.Errorf("measured %s at peak, but the suggested reservation does not fit, so the instance was disabled: %w", mibText(peak), fitErr)
	}
	if wasRunning || startAfter {
		if err := s.startGenInstance(id); err != nil {
			if wasRunning {
				return fmt.Errorf("measured; restarting failed: %w", err)
			}
			return fmt.Errorf("measured; starting failed: %w", err)
		}
	}
	return nil
}

// gpuReleasedSlackMiB is how far above the reading from before a test run
// the card may stay and still count as released: other programs on the
// card can grow a little in the meantime.
const gpuReleasedSlackMiB = 512

// waitGPUReleased waits, for at most limit, until used() falls back to
// within gpuReleasedSlackMiB of baseline, checking every step. It reports
// whether it did.
func waitGPUReleased(used func() int, baseline int, limit, step time.Duration) bool {
	deadline := time.Now().Add(limit)
	for used() > baseline+gpuReleasedSlackMiB {
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(step)
	}
	return true
}

// gpuUsedMiB is the monitor's current used-VRAM reading for a GPU.
func (s *Server) gpuUsedMiB(gpu int) int {
	for _, g := range s.monitor.Current().GPU {
		if g.Index == gpu {
			return g.VRAMUsedMB
		}
	}
	return 0
}

// runMeasureGeneration sends the instance one fixed generation directly:
// an image at the set's test size, or a 60-second song. A set that can
// edit then also edits the image it just made: an edit holds the
// reference image as well, and on Qwen-Image-Edit 2511 it peaked at
// about 21 GiB against 17.4 GiB for a plain image (R9700, 2026-10-25),
// so the reservation must cover the edit.
func (s *Server) runMeasureGeneration(ctx context.Context, inst generation.Instance, set generation.ModelSet) error {
	base := fmt.Sprintf("http://127.0.0.1:%d", inst.Port)
	if set.Kind == generation.KindSD {
		w, h := max(set.DefaultWidth, 64), max(set.DefaultHeight, 64)
		size := fmt.Sprintf("%dx%d", w, h)
		body, _, err := postJSON(ctx, base+"/v1/images/generations", map[string]any{
			"prompt": "a lighthouse on a rocky coast at sunset, photo", "size": size, "n": 1,
			"response_format": "b64_json",
		})
		if err != nil || !slices.Contains(set.Capabilities, "edit") {
			return err
		}
		img, err := firstImage(body)
		if err != nil {
			return fmt.Errorf("reading the test image: %w", err)
		}
		_, _, err = postImageEdit(ctx, base+"/v1/images/edits", img, map[string]string{
			"prompt": "make the sky stormy", "size": size, "n": "1", "response_format": "b64_json",
		})
		if err != nil {
			return fmt.Errorf("the test edit failed: %w", err)
		}
		return nil
	}
	lm, _, err := runAceJob(ctx, base, "/lm", measureSongRequest, nil)
	if err != nil {
		return err
	}
	var enriched any
	if err := json.Unmarshal(lm, &enriched); err != nil {
		return fmt.Errorf("reading the /lm result: %w", err)
	}
	if list, ok := enriched.([]any); ok && len(list) > 0 {
		enriched = list[0]
	}
	req, ok := enriched.(map[string]any)
	if !ok {
		return errors.New("unexpected /lm result")
	}
	req["output_format"] = "mp3"
	_, _, err = runAceJob(ctx, base, "/synth", req, nil)
	return err
}

// firstImage decodes the first image of an OpenAI images response.
func firstImage(body []byte) ([]byte, error) {
	var resp struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 || resp.Data[0].B64JSON == "" {
		return nil, errors.New("the response holds no image")
	}
	return base64.StdEncoding.DecodeString(resp.Data[0].B64JSON)
}

// postImageEdit sends an OpenAI-style image edit: the fields, and img as
// the reference image.
func postImageEdit(ctx context.Context, url string, img []byte, fields map[string]string) ([]byte, string, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			return nil, "", err
		}
	}
	part, err := mw.CreatePart(textproto.MIMEHeader{
		"Content-Disposition": {`form-data; name="image[]"; filename="reference.png"`},
		"Content-Type":        {"image/png"},
	})
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(img); err != nil {
		return nil, "", err
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, &buf)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return doBody(req)
}

// handleGenSetReservation sets an instance's reservation by hand, in GiB.
// It must fit, unless the instance is disabled (its reservation counts
// only once it is enabled, when the fit is checked again).
func (s *Server) handleGenSetReservation(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst, ok := s.genInstance(w, r, store)
	if !ok {
		return
	}
	r.ParseForm()
	gib, err := strconv.ParseFloat(strings.TrimSpace(r.FormValue("reserved_gib")), 64)
	if err != nil || gib < 0 || gib > 1024 {
		http.Error(w, "the reservation must be a number of GiB", http.StatusBadRequest)
		return
	}
	mib := int(gib*1024 + 0.5)
	if inst.Enabled {
		if err := s.genFitCheck(inst, mib); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
	}
	inst.ReservedMiB = mib
	if err := store.UpdateInstance(inst); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genInstancesChanged(w, r)
}
