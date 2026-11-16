package api

import (
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/internal/serverlog"
)

// genNotesErrorLines is how many of a crashed backend's last log lines
// are searched for error lines. A backend that ran for hours has failed
// requests far back in its log, which say nothing about why it stopped.
// A failed start is searched whole: all of its log is the start.
const genNotesErrorLines = 80

// withoutUsage drops the help text a backend prints after an argument it
// does not know, from its "Usage:" line on. The help is hundreds of lines
// that bury the error above it and has words such as "error" in it. The
// supervisor's own "==> " lines after it are kept.
func withoutUsage(lines []string) []string {
	for i, l := range lines {
		if !strings.HasPrefix(strings.ToLower(l), "usage:") {
			continue
		}
		out := lines[:i:i]
		for _, l := range lines[i+1:] {
			if strings.HasPrefix(l, "==> ") {
				out = append(out, l)
			}
		}
		return out
	}
	return lines
}

// genNotesSource is the engine-notes source for a backend kind.
func genNotesSource(kind generation.BackendKind) string {
	if kind == generation.KindAce {
		return enginenotes.SourceAceServer
	}
	return enginenotes.SourceSDServer
}

// genNotesHooks returns the supervisor hooks that record one start of a
// generation instance in engine notes: a load when it passes its health
// check, and an unload, a crash or a failed load when it ends. A Measure
// run that holds the instance at the start is its job, as a benchmark
// is for the router.
func (s *Server) genNotesHooks(inst generation.Instance, kind generation.BackendKind, build string) (onReady func(), onExit func(process.ExitInfo)) {
	if s.notes == nil {
		return nil, nil
	}
	base := enginenotes.Event{
		Source:    genNotesSource(kind),
		ModelID:   inst.ID,
		ModelName: inst.Name,
		Build:     build,
		GPUs:      models.GPUAssignLabel(strconv.Itoa(inst.GPU)),
	}
	if inst.Port > 0 {
		base.Port = strconv.Itoa(inst.Port)
	}
	if s.genMgr.IsMeasuring(inst.ID) {
		base.JobID, base.JobName = "measure:"+inst.ID, "Measure"
	}
	started := time.Now()
	// The hooks run on different goroutines, and a health check can pass
	// just as the program exits: once it has ended, a late ready counts
	// for nothing.
	var mu sync.Mutex
	var readyAt time.Time
	ended := false

	onReady = func() {
		mu.Lock()
		defer mu.Unlock()
		if ended {
			return
		}
		readyAt = time.Now()
		e := base
		e.Time, e.Kind = readyAt, enginenotes.KindLoad
		e.LoadSeconds = readyAt.Sub(started).Seconds()
		s.notes.Record(e)
		s.setGenNotesOpen(inst.ID, &enginenotes.OpenInstance{
			Source: base.Source, RouterName: inst.ID, Name: inst.Name, Port: base.Port, ReadySince: readyAt,
		})
	}
	onExit = func(info process.ExitInfo) {
		mu.Lock()
		defer mu.Unlock()
		ended = true
		s.setGenNotesOpen(inst.ID, nil)
		s.recordGenExit(base, info, readyAt, time.Now())
	}
	return onReady, onExit
}

// recordGenExit records how a generation instance ended: an unload when
// it was asked to stop or exited with status 0, otherwise a crash, or a
// failed load when it never passed its health check.
func (s *Server) recordGenExit(base enginenotes.Event, info process.ExitInfo, readyAt, now time.Time) {
	e := base
	e.Time, e.ExitStatus = now, info.Status
	if !readyAt.IsZero() {
		e.LoadedSeconds = now.Sub(readyAt).Seconds()
	}
	switch {
	case info.Stopped || info.Err == nil:
		e.Kind = enginenotes.KindUnload
	case readyAt.IsZero():
		e.Kind = enginenotes.KindLoadFailed
	default:
		e.Kind = enginenotes.KindCrash
	}
	if e.IsFailure() {
		var lines []string
		for _, l := range info.Lines {
			if l = generation.CleanLogLine(l); l != "" {
				lines = append(lines, l)
			}
		}
		lines = withoutUsage(lines)
		e.Tail = lines
		searched := lines
		if e.Kind == enginenotes.KindCrash {
			searched = lines[max(0, len(lines)-genNotesErrorLines):]
		}
		e.ErrorLines = serverlog.ErrorLines(searched)
	}
	s.notes.Record(e)
	if !e.IsFailure() {
		return
	}
	msg := e.Source + " instance crashed"
	if e.Kind == enginenotes.KindLoadFailed {
		msg = e.Source + " instance failed to start"
	}
	firstError := ""
	if len(e.ErrorLines) > 0 {
		firstError = e.ErrorLines[0]
	}
	slog.Warn(msg, "instance", e.ModelName, "exit_status", e.ExitStatus,
		"build", e.Build, "gpus", e.GPUs, "job", e.JobName, "error", firstError)
}

// setGenNotesOpen records a generation instance as loaded (in), or as
// ended (nil), for the time-loaded figures.
func (s *Server) setGenNotesOpen(id string, in *enginenotes.OpenInstance) {
	s.genNotesOpenMu.Lock()
	defer s.genNotesOpenMu.Unlock()
	if in == nil {
		delete(s.genNotesOpen, id)
		return
	}
	if s.genNotesOpen == nil {
		s.genNotesOpen = map[string]enginenotes.OpenInstance{}
	}
	s.genNotesOpen[id] = *in
}

// genOpenInstances is the generation instances loaded now.
func (s *Server) genOpenInstances() []enginenotes.OpenInstance {
	s.genNotesOpenMu.Lock()
	defer s.genNotesOpenMu.Unlock()
	out := make([]enginenotes.OpenInstance, 0, len(s.genNotesOpen))
	for _, in := range s.genNotesOpen {
		out = append(out, in)
	}
	return out
}
