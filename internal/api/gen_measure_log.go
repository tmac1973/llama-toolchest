package api

import (
	"strings"
	"sync"
	"time"
)

// measureLog follows a backend's log during a measured test run: which
// stage of a video job is running, what the decode did, and when the last
// line came (for the stuck-run check). The lines are sd.cpp master-1015's.
type measureLog struct {
	mu      sync.Mutex
	sampled bool // the last sampling pass has finished
	decoded bool
	// fallback: the decode ran out of memory and sd.cpp retried it
	// tiled. spatial: that retry tiles spatially as well, which hung on
	// ROCm (Wan2.2 I2V A14B inside 14 GiB, R9700, 2026-10-28).
	fallback bool
	spatial  bool
	last     time.Time
}

// add reads one log line, seen at now.
func (l *measureLog) add(line string, now time.Time) {
	line = cleanLogLine(line)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.last = now
	switch {
	// Not "sampling(high noise) completed": a pair's low-noise pass
	// follows it.
	case strings.Contains(line, "sampling completed"):
		l.sampled = true
	case strings.Contains(line, "decode_first_stage completed"):
		l.decoded = true
	case strings.Contains(line, "retrying with spatial"):
		l.fallback, l.spatial = true, true
	case strings.Contains(line, "decode ran out of memory; retrying with temporal tiling"):
		l.fallback = true
	}
}

// stage names the part of a video job that was running.
func (l *measureLog) stage() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	switch {
	case l.decoded:
		return "after the decode"
	case l.sampled:
		return "during the decode"
	default:
		return "during sampling"
	}
}

// lastLine is when the last log line came; zero before any.
func (l *measureLog) lastLine() time.Time {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.last
}

// spatialFallback says sd.cpp is retrying the decode with spatial tiling.
func (l *measureLog) spatialFallback() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.spatial
}

// decodeFallback says the decode did not fit at once and was retried tiled.
func (l *measureLog) decodeFallback() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.fallback
}
