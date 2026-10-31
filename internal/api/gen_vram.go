package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
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
	"github.com/tmac1973/llama-toolchest/internal/process"
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
	// Evictable is the part of Reserved held by instances a loading LLM
	// may stop: every one not kept running.
	Evictable int
}

// genVRAMByGPU sums the generation backends' memory per GPU index,
// leaving out exclude (an instance being checked against the others).
// Empty while generation backends are off.
func (s *Server) genVRAMByGPU(exclude string) map[int]*genVRAM {
	return s.genVRAM(exclude, false)
}

// genVRAM is genVRAMByGPU, leaving out the instances that are not
// running when activeOnly is set, except those kept running and those
// whose job has the GPU (genJobHasGPU): a job may be starting its backend,
// which is not running yet, and an LLM load must not take the room first.
func (s *Server) genVRAM(exclude string, activeOnly bool) map[int]*genVRAM {
	out := map[int]*genVRAM{}
	if !s.generationOn() || s.gen == nil || s.genMgr == nil {
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
		// A kept-running instance is never stopped to make room, so its
		// reservation counts even while it is stopped.
		if activeOnly && using == 0 && !s.genMgr.IsActive(inst.ID) && !inst.KeepRunning && !s.genJobHasGPU(inst) {
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
			if !inst.KeepRunning {
				v.Evictable += share
			}
		}
	}
	return out
}

// genJobHasGPU reports whether a job for inst has its GPU's turn: it is
// starting the backend or running on it.
func (s *Server) genJobHasGPU(inst generation.Instance) bool {
	if s.genQueue == nil {
		return false
	}
	r := s.genQueue.Snapshot(inst.GPU).Running
	return r != nil && r.InstanceID == inst.ID
}

// applyGenVRAM folds the generation backends into the planner's view of
// the GPUs: what they use leaves "other programs", and what they reserve
// is set apart. With auto-evict on, the part a loading model may take
// back is marked, so a model that does not fit beside it is planned on
// the whole cards (models.PlanFit, models.CardRoom). With generation
// backends off this changes nothing.
func (s *Server) applyGenVRAM(hw *models.Hardware) {
	byGPU := s.genVRAMByGPU("")
	evict := s.genEvictOn()
	for i := range hw.GPUs {
		if v := byGPU[hw.GPUs[i].Index]; v != nil {
			hw.GPUs[i].OtherUsedMiB = max(0, hw.GPUs[i].OtherUsedMiB-v.Using)
			hw.GPUs[i].ReservedMiB = v.Reserved
			if evict {
				hw.GPUs[i].EvictableMiB = v.Evictable
			}
		}
	}
}

// loadedLLMMiB is the planned GPU memory of the LLMs the router has
// loaded, per GPU index, plus extraID (a model about to load) when set.
// skipLoaded leaves the loaded ones out: with --models-max 1 the loaded
// model is replaced by the new one. gpuMiB is each GPU's room
// (models.GPUMiB), which decides how a model is split over the cards:
// from idleHardware, the room the preset was balanced for.
func (s *Server) loadedLLMMiB(room models.CardRoom, extraID string, skipLoaded bool) map[int]int {
	out := map[int]int{}
	if len(room.Shared) == 0 {
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
		addLLMMiB(out, m, cfg, room)
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
func addLLMMiB(out map[int]int, m *models.Model, cfg *models.ModelConfig, room models.CardRoom) {
	for g, mib := range models.CardLoadsMiB(m, cfg, room.For(m, cfg)) {
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
	llm := s.loadedLLMMiB(models.RoomOf(s.idleHardware()), "", false)[inst.GPU]
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
	replacing = s.cfgCopy().ModelsMax == 1
	idle := s.idleHardware()
	hw = idle
	if !replacing {
		hw = s.hardware()
	}
	return hw, s.loadedLLMMiB(models.RoomOf(idle), m.ID, replacing), replacing
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
	addLLMMiB(need, m, cfg, models.RoomOf(hw))
	return s.reservationFitError(hw, need, false)
}

// genReservationsApply reports whether any generation reservation can
// limit LLM loads.
func (s *Server) genReservationsApply() bool {
	return s.generationOn() && s.gen != nil && len(s.genFitVRAMByGPU("")) > 0
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
// a GPU map entry on its GPU, so the map and its legend show them. A
// disabled instance is listed while it runs, as it does while Measure
// tests it; with no reservation yet, its size is what the process uses.
// Their
// colors continue the map's palette from firstColor, the number of
// entries already on the map, so every entry has a color of its own; the
// map draws them striped to set them apart from LLMs.
func (s *Server) genReservationAllocations(firstColor int) []models.GPUAllocation {
	if !s.generationOn() || s.gen == nil {
		return nil
	}
	var out []models.GPUAllocation
	for _, inst := range s.gen.Instances() {
		active := s.genMgr != nil && s.genMgr.IsActive(inst.ID)
		if !inst.Enabled && !active {
			continue
		}
		mib := inst.ReservedMiB
		if mib == 0 && active {
			if pid := s.genMgr.PID(inst.ID); pid > 0 {
				mib, _ = s.genProcVRAM.read(pid)
			}
		}
		if mib == 0 {
			continue
		}
		gb := float64(mib) / 1024
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
	if !s.generationOn() || s.gen == nil || s.genQueue == nil {
		return nil
	}
	hw := s.idleHardware()
	return bannerGPUs(hw.GPUs, func(gpu int) bool { return s.genQueue.Snapshot(gpu).Running != nil },
		s.loadedLLMMiB(models.RoomOf(hw), "", false))
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
	if inst.BudgetMiB > 0 {
		note += "Measured inside a budget of " + mibText(inst.BudgetMiB) + ", which sd-server keeps to from now on: it holds model weights in system RAM and decodes in smaller pieces as needed, so videos take longer. "
	}
	suggest := generation.SuggestReservation(inst.MeasuredPeakMiB)
	capped := inst.MeasuredSuggestedMiB > 0 && inst.MeasuredSuggestedMiB < suggest
	if inst.MeasuredSuggestedMiB > 0 {
		suggest = inst.MeasuredSuggestedMiB
	}
	switch {
	case capped && inst.ReservedMiB == suggest && suggest < inst.MeasuredPeakMiB:
		// No slack was left: capReservation keeps the reservation at the
		// room when the peak is above it.
		over := inst.MeasuredPeakMiB - suggest
		note += "Reservation capped at " + mibText(suggest) + ", all this card can give the instance. The peak is " + mibText(over) + " above it, "
		if over <= measureOverheadMiB {
			note += "within the " + mibText(measureOverheadMiB) + " allowed for memory the backend's own limit does not count, such as the GPU runtime's."
		} else {
			// Image sets are not refused for this (video sets are, in
			// applyMeasurement), so the note must not claim it fits.
			note += "more than the " + mibText(measureOverheadMiB) + " expected for memory the backend's own limit does not count, such as the GPU runtime's."
			if inst.MeasureEstimated {
				note += " The peak is read from the whole GPU, so it also counts other programs' memory that changed during the test; measure again when the card is quieter."
			}
		}
	case capped && inst.ReservedMiB == suggest:
		note += "Reservation capped at " + mibText(suggest) + " to fit this card, leaving up to " + mibText(reserveSlackMiB) +
			" free for other programs' memory to change."
	case before == 0 || (inst.MeasuredFrames > 0 && inst.ReservedMiB == suggest):
		note += "Reservation set to " + mibText(inst.ReservedMiB) + "."
	case before == suggest:
		note += "Reservation stays at " + mibText(before) + ", as suggested."
	default:
		note += "Reservation stays at " + mibText(before) + "; this measurement suggests " + mibText(suggest) + "."
	}
	if inst.MeasuredDecodeFallback {
		note += " The decode did not fit at once, so sd.cpp decoded the video in smaller pieces, which is slower."
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
	set, err := store.ModelSet(inst.ModelSetID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var size generation.VideoSize
	if set.Kind == generation.KindVideo {
		if size, err = videoMeasureSize(r, set); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	budget, err := measureBudget(r, set)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.genMgr.SetMeasuring(inst.ID, true)
	s.genMeasure.clear(inst.ID)
	before, wasRunning := inst.ReservedMiB, s.genMgr.IsActive(inst.ID)
	startAfter := r.FormValue("start") == "1"
	go func() {
		defer s.genMgr.SetMeasuring(inst.ID, false)
		if err := s.measureInstance(inst.ID, startAfter, size, budget); err != nil {
			slog.Warn("failed to measure a generation instance", "instance", inst.ID, "error", err)
			s.genMeasure.set(inst.ID, measureOutcome{Err: err.Error(), At: time.Now()})
			return
		}
		after, err := s.gen.Instance(inst.ID)
		if err != nil {
			return
		}
		note := measureNote(before, after, wasRunning, startAfter && !wasRunning)
		if set.Kind == generation.KindVideo {
			label := generation.PresetLabel(generation.VideoPresets(set), size)
			note = fmt.Sprintf("Measured for %d frames at %dx%d (%s): ", size.Frames, size.Width, size.Height, label) + note
		}
		s.genMeasure.set(inst.ID, measureOutcome{Note: note, At: time.Now()})
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
	// With eviction on, the measurement stops the other backends itself
	// (makeRoomForMeasure), except those kept running.
	evict := s.genEvictOn()
	var busy []string
	for _, other := range s.gen.Instances() {
		if other.ID != inst.ID && other.GPU == inst.GPU && s.genMgr.IsActive(other.ID) && (!evict || other.KeepRunning) {
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
// startAfter it is started afterwards either way. A video instance with a
// budget (MiB, 0 for the whole room) is measured inside it, and runs
// inside it afterwards. With eviction on, room is made on the GPU first
// (makeRoomForMeasure).
func (s *Server) measureInstance(id string, startAfter bool, size generation.VideoSize, budget int) error {
	inst, err := s.gen.Instance(id)
	if err != nil {
		return err
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return err
	}
	// The settling waits below also cover the memory this frees.
	rctx, rcancel := context.WithTimeout(context.Background(), genStartTimeout)
	release, err := s.makeRoomForMeasure(rctx, inst)
	rcancel()
	if err != nil {
		return fmt.Errorf("making room on GPU %d: %w", inst.GPU, err)
	}
	defer release()
	wasRunning := s.genMgr.IsActive(id)
	// The card's reading from before the stop, for the wait to see the
	// backend's memory fall from. A backend that held almost nothing
	// shows no drop, so none is waited for.
	from := s.gpuUsedMiB(inst.GPU)
	if pid := s.genMgr.PID(id); pid > 0 {
		if mib, ok := s.genProcVRAM.read(pid); ok && mib <= gpuReleasedSlackMiB {
			from = -1
		}
	}
	stoppedAt := time.Now()
	if wasRunning {
		if err := s.genMgr.Stop(id); err != nil {
			return err
		}
		// The baseline and the room below read the card's used VRAM.
		// The monitor reads it every few seconds, so a reading from
		// before the backend let go of its memory counts that memory as
		// another program's: Wan2.2 I2V A14B was measured with 16.5 GiB
		// of a 28.5 GiB room that way (R9700, 2026-10-27).
		waitGPUSettled(s.gpuReading(inst.GPU), stoppedAt, from, 20*time.Second, 500*time.Millisecond)
	} else {
		// Another backend on the card may have stopped just before, and
		// still be giving its memory back: a baseline read then counts
		// that memory as other programs', and the peak comes out far too
		// low (HunyuanVideo 1.5 T2V measured 5.9 GiB at 121 frames right
		// after another HunyuanVideo instance stopped; R9700, 2026-10-29).
		waitGPUSettled(s.gpuReading(inst.GPU), time.Now(), -1, 20*time.Second, 500*time.Millisecond)
	}
	baseline := s.gpuUsedMiB(inst.GPU)

	// The first start has no reservation to check yet; it needs at least
	// 1 GiB of the card's budget free.
	if err := s.genFitCheck(inst, 1024); err != nil {
		return err
	}
	// sd-server runs inside the room the card can give the instance, so
	// it does not fill a smaller card and the suggestion always fits.
	// ace-server takes no memory limit and is measured as it is.
	sdKind := set.Kind == generation.KindSD || set.Kind == generation.KindVideo
	room, haveRoom := s.genRoomMiB(inst)
	opts := generation.LaunchOptions{}
	if sdKind && haveRoom {
		opts.MaxVRAMMiB = room
	}
	if budget > 0 {
		if haveRoom && budget > room {
			return fmt.Errorf("a budget of %s is more than the %s this card can give the instance; leave it empty to measure inside the whole room", mibText(budget), mibText(room))
		}
		opts.MaxVRAMMiB = budget
	}
	// The early stop compares the peak with the room, not the budget:
	// sd.cpp goes a little past its budget, and the reservation may take
	// the whole room.
	limit := 0
	if sdKind && haveRoom {
		limit = room
	}
	if err := s.launchGenInstance(inst, set, opts); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), measureTimeout(set.Kind))
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, id); err != nil {
		return fmt.Errorf("the instance did not start: %w", err)
	}

	// Per-process readings win whenever any sample had one: a backend
	// such as ace-server only opens the GPU on its first request, so the
	// samples before that have no per-process entry yet. The whole-GPU
	// figure is used only when no sample had one (Vulkan, or PIDs hidden
	// in a container).
	var procPeak, gpuPeak, ramPeak int
	var procOK bool
	// abortErr is set when the sampler stops the backend itself: a video
	// size that has gone past the room (abortOver, at abortPeak), or a run
	// that is stuck.
	var abortErr error
	var abortPeak int
	var abortOver bool
	mlog := &measureLog{}
	if logs := s.genMgr.Subscribe(id); logs != nil {
		// Unsubscribing does not close the channel, so the reader also
		// ends when Measure returns.
		done := make(chan struct{})
		defer func() {
			s.genMgr.Unsubscribe(id, logs)
			close(done)
		}()
		go func() {
			for {
				select {
				case line := <-logs:
					mlog.add(line, time.Now())
				case <-done:
					return
				}
			}
		}()
	}
	video := set.Kind == generation.KindVideo
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		tick := time.NewTicker(300 * time.Millisecond)
		defer tick.Stop()
		started, busyAt, sawBusy := time.Now(), time.Now(), false
		for {
			pid := s.genMgr.PID(id)
			if mib, ok := monitor.ProcessVRAMMiB(pid); ok {
				procPeak, procOK = max(procPeak, mib), true
			}
			if mib, ok := monitor.ProcessPeakRAMMiB(pid); ok {
				ramPeak = max(ramPeak, mib)
			}
			gpuPeak = max(gpuPeak, s.gpuUsedMiB(inst.GPU)-baseline)
			if busy, ok := s.gpuBusyPercent(inst.GPU); ok && busy >= measureBusyPercent {
				busyAt, sawBusy = time.Now(), true
			}
			if video && abortErr == nil {
				peak := gpuPeak
				if procOK {
					peak = procPeak
				}
				abortErr = measureAbort(measureRunState{
					peak: peak, room: limit, sawBusy: sawBusy,
					quiet: time.Since(latestTime(busyAt, mlog.lastLine(), started)), size: videoSizeText(size, set), stage: mlog.stage(),
				})
				if abortErr != nil {
					abortPeak, abortOver = peak, limit > 0 && peak > limit+measureOverheadMiB
					go s.genMgr.Stop(id)
				}
			}
			select {
			case <-stop:
				return
			case <-tick.C:
			}
		}
	}()
	runErr := s.runMeasureGeneration(ctx, inst, set, size)
	// Read before the Stop below, which leaves every backend stopped: a
	// stopped one now was stopped by the user, not by a crash.
	userStopped := s.genMgr.Status(id).State == process.StateStopped
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
	if abortErr != nil {
		if abortOver {
			// What this size needs is known to be more than the room: a
			// point for predicting the sizes above it.
			point := generation.MeasurePoint{Frames: size.Frames, Width: size.Width, Height: size.Height, PeakMiB: abortPeak, BudgetMiB: budget, Over: true}
			if err := s.gen.EditInstance(id, func(i *generation.Instance) {
				i.MeasurePoints = generation.AddMeasurePoint(i.MeasurePoints, point)
			}); err != nil {
				slog.Warn("could not record a measurement point", "instance", id, "error", err)
			}
		}
		return abortErr
	}
	if runErr != nil {
		if backendGone(runErr) {
			if userStopped {
				return withPeak(errors.New("the instance was stopped during the test generation"), peak)
			}
			return withPeak(backendStoppedError(s.genMgr.RecentLogs(id)), peak)
		}
		return withPeak(fmt.Errorf("the test generation failed: %w", runErr), peak)
	}
	if peak <= 0 {
		return errors.New("no GPU memory use was seen during the test generation")
	}

	inst, err = s.gen.Instance(id)
	if err != nil {
		return err
	}
	inst.MeasuredPeakMiB, inst.MeasuredAt, inst.MeasureEstimated = peak, time.Now(), !perProcess
	inst.MeasuredArgs, inst.MeasuredRAMMiB = runArgs(set, inst), ramPeak
	// The room is worked out again now the backend has stopped: other
	// programs on the card may have grown during a long test run.
	if sdKind {
		room, haveRoom = s.genRoomMiB(inst)
	} else {
		haveRoom = false
	}
	var fitErr error
	inst, fitErr = s.applyMeasurement(inst, set, size, budget, room, haveRoom)
	if video {
		inst.MeasurePoints = generation.AddMeasurePoint(inst.MeasurePoints, generation.MeasurePoint{
			Frames: size.Frames, Width: size.Width, Height: size.Height, PeakMiB: peak, BudgetMiB: budget, Fallback: mlog.decodeFallback()})
		if fitErr == nil {
			inst.MeasuredDecodeFallback, inst.BudgetMiB = mlog.decodeFallback(), budget
		}
	}
	if err := s.gen.UpdateInstance(inst); err != nil {
		return err
	}
	slog.Info("generation instance measured", "instance", id, "peak_mib", peak, "estimated", !perProcess, "reserved_mib", inst.ReservedMiB)
	if fitErr != nil {
		if inst.Enabled {
			return fmt.Errorf("measured %s at peak, but the suggested reservation does not fit, so the reservation stays at %s: %w", mibText(peak), mibText(inst.ReservedMiB), fitErr)
		}
		return fmt.Errorf("measured %s at peak, but the suggested reservation does not fit, so the instance was disabled: %w", mibText(peak), fitErr)
	}
	if wasRunning || startAfter {
		// startGenInstance refuses an instance that is being measured,
		// and the measurement is over.
		s.genMgr.SetMeasuring(id, false)
		if err := s.startGenInstance(id); err != nil {
			if wasRunning {
				return fmt.Errorf("measured; restarting failed: %w", err)
			}
			return fmt.Errorf("measured; starting failed: %w", err)
		}
	}
	return nil
}

// applyMeasurement records what a measurement suggests on inst, whose
// MeasuredPeakMiB is already set. For sd-server instances the suggestion
// is capped at room (the memory the card could give the instance, when
// haveRoom) and kept in MeasuredSuggestedMiB. A video measurement always
// applies, with the size it was made at, since the user chose that size;
// any other applies only to an instance without a reservation. A
// suggestion that does not fit is not applied (a video instance keeps the
// size it was last measured for), and an instance left with no
// reservation is disabled; the error says why.
func (s *Server) applyMeasurement(inst generation.Instance, set generation.ModelSet, size generation.VideoSize, budget, room int, haveRoom bool) (generation.Instance, error) {
	// sd-server can go past --max-vram decoding a video (tiled decoding
	// crashes for the Wan2.2 VAE, so the decode cannot shrink): a video
	// size whose peak is over the room is not granted.
	if set.Kind == generation.KindVideo && haveRoom && inst.MeasuredPeakMiB > room+measureOverheadMiB {
		err := fmt.Errorf("%d frames at %dx%d needs about %s, but this card can give the instance %s; measure a smaller size",
			size.Frames, size.Width, size.Height, mibText(inst.MeasuredPeakMiB), mibText(room))
		if inst.ReservedMiB == 0 {
			inst.Enabled = false
		}
		return inst, err
	}
	// Inside a budget the reservation is at least the budget: sd-server
	// may use all of it on another video, though the test did not (TI2V
	// 5B at 81 frames peaked at 10.6 GiB inside 14 GiB; R9700,
	// 2026-10-29).
	suggest := max(generation.SuggestReservation(inst.MeasuredPeakMiB), budget)
	sdKind := set.Kind == generation.KindSD || set.Kind == generation.KindVideo
	if sdKind && haveRoom {
		suggest = capReservation(suggest, inst.MeasuredPeakMiB, room)
	}
	apply := inst.ReservedMiB == 0 || set.Kind == generation.KindVideo
	if apply {
		if err := s.genFitCheck(inst, suggest); err != nil {
			// A video instance keeps the size it was last measured for:
			// this measurement did not take effect.
			if inst.ReservedMiB == 0 {
				inst.Enabled = false
			}
			return inst, err
		}
		inst.ReservedMiB = suggest
		if set.Kind == generation.KindVideo {
			inst.MeasuredFrames, inst.MeasuredWidth, inst.MeasuredHeight = size.Frames, size.Width, size.Height
		}
	}
	// Kept for "Use suggested" (image re-measurements) and for VideoLimit.
	if sdKind {
		inst.MeasuredSuggestedMiB = suggest
	}
	return inst, nil
}

// backendStoppedError says why the backend stopped during a test
// generation, from the end of its log. Running out of GPU memory is the
// usual reason, and a smaller size helps; a crash such as a CUDA or HIP
// error is quoted instead, since a smaller size may not help (sd.cpp's
// tiled Wan decode crashed this way on an RTX A4000). With neither in the
// log, it is probably memory.
//
// Only the lines after sd.cpp's last "retrying with …" count: the memory
// warnings before it were recovered from. TI2V 5B on the A4000 ran out of
// memory for the whole decode, retried twice, and then crashed ("CONCAT
// failed") decoding in small pieces (2026-10-30).
func backendStoppedError(lines []string) error {
	const memory = "the backend stopped during the test generation, probably because it ran out of GPU memory at this size; its log on this page says more. Try a smaller size"
	lines = lines[max(0, len(lines)-backendLogLines):]
	retried := false
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.Contains(lines[i], "retrying with") {
			lines, retried = lines[i+1:], true
			break
		}
	}
	crash := ""
	for _, line := range lines {
		lower := strings.ToLower(line)
		if slices.ContainsFunc(oomMarkers, func(m string) bool { return strings.Contains(lower, m) }) {
			return errors.New(memory)
		}
		if crash == "" && slices.ContainsFunc(crashMarkers, func(m string) bool { return strings.Contains(lower, m) }) {
			crash = cleanLogLine(line)
		}
	}
	if crash == "" {
		return errors.New(memory)
	}
	if retried {
		return fmt.Errorf("the backend crashed during the test generation (%s), while decoding the video in smaller pieces because the whole decode did not fit. A smaller size, whose decode fits at once, may work; the log on this page says more", crash)
	}
	return fmt.Errorf("the backend crashed during the test generation (%s). Its log shows a crash, not running out of GPU memory, so a smaller size may not help; the log on this page says more", crash)
}

// backendLogLines is how many of a stopped backend's last log lines
// backendStoppedError reads.
const backendLogLines = 80

// oomMarkers and crashMarkers are lowercase text that marks a log line as
// running out of memory, or as a crash.
var (
	oomMarkers   = []string{"out of memory", "outofmemory", "memoryallocation", "cannot make enough memory available", "failed to allocate"}
	crashMarkers = []string{"[e] ", "cuda error", "hip error", "rocm error", "ggml_assert", "terminate called", "segmentation fault", "aborted"}
)

// cleanLogLine shortens a log line for an error message: without sd.cpp's
// "[E] " level and its " --- file.cpp:123" source suffix. A progress bar
// redraws itself with carriage returns and an erase-line code, and an error
// printed while it shows lands on the same line, after the last of them.
func cleanLogLine(line string) string {
	line = strings.TrimRight(line, "\r\n")
	if i := strings.LastIndex(line, "\r"); i >= 0 {
		line = line[i+1:]
	}
	if i := strings.LastIndex(line, "\x1b[K"); i >= 0 {
		line = line[i+len("\x1b[K"):]
	}
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "[E] ")
	if i := strings.LastIndex(line, " --- "); i > 0 {
		line = line[:i]
	}
	return line
}

// reserveSlackMiB is how much of the room a capped reservation leaves
// free, when the measured peak allows it. Other programs on the card grow
// a little between a measurement and the next start: a reservation equal
// to the room was refused at its start on the R9700 when they had grown
// by 0.2 GiB.
const reserveSlackMiB = 512

// capReservation fits a suggested reservation to the room the card can
// give: a suggestion that leaves reserveSlackMiB of the room free is kept;
// otherwise the reservation is the room less that slack, but never below
// the measured peak, nor above the room.
func capReservation(suggest, peak, room int) int {
	if suggest <= room-reserveSlackMiB {
		return suggest
	}
	return max(room-reserveSlackMiB, min(peak, room))
}

// measureOverheadMiB is how far a measured peak may exceed the room it was
// held to and still count as fitting: inside a container the peak is read
// from the whole card and includes the GPU runtime's own memory, which
// --max-vram does not count (about 0.6 GiB on an RTX A4000).
const measureOverheadMiB = 1024

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

// waitGPUSettled waits, for at most limit, until two readings taken after
// since are no more than gpuReleasedSlackMiB apart: the card has stopped
// giving memory back. read returns a reading's time and used MiB. When
// from is not negative, it is the reading from before a stop, and the
// two readings count only after one fell more than gpuReleasedSlackMiB
// below from (or the first reading, when that is higher), or once
// gpuFreedGrace has passed since since: just after a stop the driver may
// not have given the memory back yet (see gpuFreedGrace).
func waitGPUSettled(read func() (time.Time, int), since time.Time, from int, limit, step time.Duration) bool {
	deadline := time.Now().Add(limit)
	var last time.Time
	dropped := from < 0
	prev := -1
	for {
		if at, mib := read(); at.After(since) && at.After(last) {
			if prev < 0 {
				from = max(from, mib)
			}
			if from-mib > gpuReleasedSlackMiB || at.Sub(since) >= gpuFreedGrace {
				dropped = true
			}
			if dropped && prev >= 0 && prev-mib <= gpuReleasedSlackMiB {
				return true
			}
			prev, last = mib, at
		}
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(step)
	}
}

// gpuReading returns a function that reads the monitor's latest sample
// for a GPU: when it was taken and the card's used VRAM.
func (s *Server) gpuReading(gpu int) func() (time.Time, int) {
	return func() (time.Time, int) {
		m := s.monitor.Current()
		for _, g := range m.GPU {
			if g.Index == gpu {
				return m.Timestamp, g.VRAMUsedMB
			}
		}
		return m.Timestamp, 0
	}
}

// measureBusyPercent is the GPU use below which a measured run counts as
// idle, and measureStuckAfter how long it may be idle with a silent log
// before Measure stops it. A stuck tiled decode on ROCm sat at 3% with no
// log line for 45 minutes (R9700, 2026-10-28); real work keeps the GPU far
// busier, even while a long step or decode writes nothing.
const (
	measureBusyPercent = 10
	measureStuckAfter  = 5 * time.Minute
)

// measureRunState is what a video measurement's sampler knows at one
// reading. room is 0 when the run has no room to keep to.
type measureRunState struct {
	peak, room  int
	sawBusy     bool
	quiet       time.Duration
	size, stage string
}

// measureAbort says why a video measurement must be stopped now, or nil:
// its peak has gone past the room (plus what the backend's own limit does
// not count), or the GPU has been idle with a silent log for
// measureStuckAfter, after it was seen busy (so a card without a use
// reading is never taken for stuck).
//
// sd.cpp's fallback to a spatially tiled decode is left to run: on ROCm it
// hung for the Wan2.1 and LTX-2.3 VAEs, which the stuck check catches, but
// it worked for HunyuanVideo 1.5 at 1280x720 (225 s, R9700, 2026-10-29),
// whose untiled decode needs more than a 32 GB card has.
func measureAbort(st measureRunState) error {
	switch {
	case st.room > 0 && st.peak > st.room+measureOverheadMiB:
		return fmt.Errorf("%s needed more than %s %s, more than the %s this card can give the instance; measure a smaller size",
			st.size, mibText(st.peak), st.stage, mibText(st.room))
	case st.sawBusy && st.quiet > measureStuckAfter:
		return fmt.Errorf("the test video stopped making progress %s (the GPU was idle and the backend wrote nothing for %s; it reached %s). On Radeon cards sd.cpp's tiled decode can hang; measure a smaller size, or with more room",
			st.stage, measureStuckAfter, mibText(st.peak))
	}
	return nil
}

// runArgs are the backend flags an instance runs with: its set's, then
// its own extra ones.
func runArgs(set generation.ModelSet, inst generation.Instance) []string {
	return append(slices.Clone(set.RunArgs()), inst.ExtraArgs...)
}

// latestTime is the latest of ts.
func latestTime(ts ...time.Time) time.Time {
	var out time.Time
	for _, t := range ts {
		if t.After(out) {
			out = t
		}
	}
	return out
}

// withPeak adds the peak a failed test run reached to its error.
func withPeak(err error, peak int) error {
	if peak <= 0 {
		return err
	}
	return fmt.Errorf("%w (it reached %s before it stopped)", err, mibText(peak))
}

// videoSizeText names a video size for a message, e.g. "81 frames at
// 1280x704".
func videoSizeText(size generation.VideoSize, set generation.ModelSet) string {
	if size.Frames == 0 {
		size = generation.VideoSize{Frames: set.DefaultFrames, Width: set.DefaultWidth, Height: set.DefaultHeight}
	}
	return fmt.Sprintf("%d frames at %dx%d", size.Frames, size.Width, size.Height)
}

// gpuBusyPercent is the monitor's current use reading for a GPU; ok is
// false without one.
func (s *Server) gpuBusyPercent(gpu int) (int, bool) {
	for _, g := range s.monitor.Current().GPU {
		if g.Index == gpu {
			return g.UtilPercent, true
		}
	}
	return 0, false
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
// an image at the set's test size, a 60-second song, or an upscale of a
// 1024x1024 image. A set that can
// edit then also edits the image it just made: an edit holds the
// reference image as well, and on Qwen-Image-Edit 2511 it peaked at
// about 21 GiB against 17.4 GiB for a plain image (R9700, 2026-10-25),
// so the reservation must cover the edit.
func (s *Server) runMeasureGeneration(ctx context.Context, inst generation.Instance, set generation.ModelSet, size generation.VideoSize) error {
	base := inst.BaseURL()
	if set.Kind == generation.KindVideo {
		return runMeasureVideo(ctx, base, set, size)
	}
	if set.Kind == generation.KindUpscale {
		img, err := measureUpscaleImage()
		if err != nil {
			return err
		}
		_, _, err = postJSON(ctx, base+"/sdcpp/v1/upscale", map[string]any{
			"image": base64.StdEncoding.EncodeToString(img),
		})
		return err
	}
	if set.Kind == generation.KindSD {
		w, h := max(set.DefaultWidth, 64), max(set.DefaultHeight, 64)
		size := fmt.Sprintf("%dx%d", w, h)
		body, _, err := postJSON(ctx, base+"/v1/images/generations", map[string]any{
			"prompt": "a lighthouse on a rocky coast at sunset, photo", "size": size, "n": 1,
			"response_format": "b64_json",
		})
		if err != nil || !slices.Contains(set.CurrentCapabilities(), generation.CapEdit) {
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

// measureUpscaleImage is the 1024x1024 test image an upscaler instance is
// measured with: a colour gradient, as the upscaler has no prompt to make
// one from. sd-server tiles the image, so its content hardly changes the
// peak.
func measureUpscaleImage() ([]byte, error) {
	const n = 1024
	img := image.NewRGBA(image.Rect(0, 0, n, n))
	for y := range n {
		for x := range n {
			img.Set(x, y, color.RGBA{uint8(x / 4), uint8(y / 4), uint8((x + y) / 8), 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("making the test image: %w", err)
	}
	return buf.Bytes(), nil
}

// measureTimeout caps a measured test run, from launching the backend to
// the end of its test generation. A video set may be measured at its
// largest size: that took 21 minutes on an R9700, and a slower card held
// to a smaller budget can take several times as long.
func measureTimeout(kind generation.BackendKind) time.Duration {
	if kind == generation.KindVideo {
		return 2 * time.Hour
	}
	return 10 * time.Minute
}

// measureBudget reads the budget a video instance is to be measured
// inside, from the form's budget_gib, in MiB: 0 (empty) for the whole
// room. Image and music instances take none.
func measureBudget(r *http.Request, set generation.ModelSet) (int, error) {
	text := strings.TrimSpace(r.FormValue("budget_gib"))
	if text == "" || set.Kind != generation.KindVideo {
		return 0, nil
	}
	gib, err := strconv.ParseFloat(text, 64)
	if err != nil || gib < minMeasureBudgetGiB {
		return 0, fmt.Errorf("the budget must be a number of GiB, at least %d, or empty for the whole room", minMeasureBudgetGiB)
	}
	return int(gib * 1024), nil
}

// minMeasureBudgetGiB is the smallest budget Measure takes: below it even
// small video models spend most of their time copying weights in.
const minMeasureBudgetGiB = 4

// videoMeasureSize reads the video size to measure from the form: frames,
// width and height, within the model's limit (either orientation) and in
// multiples of 8.
func videoMeasureSize(r *http.Request, set generation.ModelSet) (generation.VideoSize, error) {
	var v [3]int
	for i, name := range []string{"frames", "width", "height"} {
		n, err := strconv.Atoi(strings.TrimSpace(r.FormValue(name)))
		if err != nil || n <= 0 {
			return generation.VideoSize{}, errors.New("Choose a video size to measure: frames, width and height")
		}
		v[i] = n
	}
	size := generation.VideoSize{Frames: v[0], Width: v[1], Height: v[2]}
	long, short := max(size.Width, size.Height), min(size.Width, size.Height)
	maxLong, maxShort := max(set.MaxWidth, set.MaxHeight), min(set.MaxWidth, set.MaxHeight)
	if (set.MaxFrames > 0 && size.Frames > set.MaxFrames) || (maxShort > 0 && (long > maxLong || short > maxShort)) {
		return generation.VideoSize{}, fmt.Errorf("%s makes at most %d frames and %dx%d; choose a size within that", set.Name, set.MaxFrames, set.MaxWidth, set.MaxHeight)
	}
	if size.Width%8 != 0 || size.Height%8 != 0 {
		return generation.VideoSize{}, errors.New("Width and height must be multiples of 8")
	}
	return size, nil
}

// genRoomMiB is the GPU memory an instance may have on its card: the
// planner's budget for the card, less the loaded LLMs and the other
// instances' reservations, as genFitCheck counts them. ok is false
// without GPU telemetry.
func (s *Server) genRoomMiB(inst generation.Instance) (room int, ok bool) {
	hw := s.hardware()
	for i := range hw.GPUs {
		if hw.GPUs[i].Index != inst.GPU {
			continue
		}
		spec := hw.GPUs[i]
		spec.ReservedMiB = 0
		if others := s.genFitVRAMByGPU(inst.ID)[inst.GPU]; others != nil {
			spec.ReservedMiB = others.Reserved
		}
		llm := s.loadedLLMMiB(models.RoomOf(s.idleHardware()), "", false)[inst.GPU]
		return max(0, models.CardBudgetMiB(spec)-llm), true
	}
	return 0, false
}

// runMeasureVideo makes a video set's test video: its test size, length
// and frame rate, in the heaviest mode the set supports, with drawn
// start and end pictures when it takes them.
func runMeasureVideo(ctx context.Context, base string, set generation.ModelSet, size generation.VideoSize) error {
	caps := set.CurrentCapabilities()
	canStart := slices.Contains(caps, generation.CapStartImage)
	if !canStart && !slices.Contains(caps, generation.CapText) {
		return errors.New("this set has no supported way to make a test video")
	}
	if size.Frames == 0 {
		size = generation.VideoSize{Frames: set.DefaultFrames, Width: set.DefaultWidth, Height: set.DefaultHeight}
	}
	w, h := max(size.Width, 64), max(size.Height, 64)
	// One sampling step: memory peaks in the first step or in the decode,
	// and the steps after it only take time (Wan2.2 TI2V 5B at 81 frames
	// 1280x704 peaked the same with 1 step as with 20, in 166 s against
	// 659 s; R9700, 2026-10-28). A pair runs one step on each expert, so
	// both load; with two low-noise steps and one high-noise step,
	// sd-server master-1015 made no video.
	req := vidGenRequest{
		Prompt: "a sunrise over a calm sea, slow camera pan", Width: w, Height: h, Seed: 42,
		Frames: max(size.Frames, 1), FPS: max(set.DefaultFPS, 1), OutputFormat: "webm", Lora: setLoras(set),
		Steps: 1,
	}
	if set.HasHighNoiseModel() {
		req.HighNoiseSteps = 1
	}
	if canStart {
		req.InitImage = base64.StdEncoding.EncodeToString(measurePicture(w, h, false))
	}
	if slices.Contains(caps, generation.CapEndImage) {
		req.EndImage = base64.StdEncoding.EncodeToString(measurePicture(w, h, true))
	}
	_, err := runVidGen(ctx, base, req, nil)
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
	if !parseForm(w, r) {
		return
	}
	gib, err := strconv.ParseFloat(strings.TrimSpace(r.FormValue("reserved_gib")), 64)
	if err != nil || gib < 0 || gib > 1024 {
		http.Error(w, "the reservation must be a number of GiB", http.StatusBadRequest)
		return
	}
	mib := int(gib*1024 + 0.5)
	// As for Enable: the reservation has to fit only where it counts while
	// the instance is stopped. With eviction on it does not, unless the
	// instance is kept running; starting it makes the room.
	if inst.Enabled && (!s.genEvictOn() || inst.KeepRunning) {
		if err := s.genFitCheck(inst, mib); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
	}
	// A reservation set by hand is sd-server's limit again: a budget
	// chosen at Measure was for the reservation that Measure set.
	inst.ReservedMiB, inst.BudgetMiB = mib, 0
	if err := store.UpdateInstance(inst); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genInstancesChanged(w, r)
}
