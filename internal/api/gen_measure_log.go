package api

import (
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// measureLog follows a backend's log during a measured test run: sd.cpp's
// stage times, what the video decode did, and when the last line came
// (for the stuck-run check). The lines are sd.cpp master-1015's.
type measureLog struct {
	mu        sync.Mutex
	encodeSec float64 // get_learned_condition and encode_first_stage
	sampleSec float64 // every sampling pass (two for a Wan2.2 A14B pair)
	decodeSec float64
	sampled   bool
	decoded   bool
	// fallback: the decode ran out of memory and sd.cpp retried it
	// tiled. spatial: that retry tiles spatially as well, which hung on
	// ROCm (Wan2.2 I2V A14B inside 14 GiB, R9700, 2026-10-28).
	fallback bool
	spatial  bool
	last     time.Time
}

// takingRe reads sd.cpp's "completed, taking 26.78s" or "taking 1566 ms".
var takingRe = regexp.MustCompile(`taking ([0-9.]+) ?(ms|s)\b`)

// add reads one log line, seen at now.
func (l *measureLog) add(line string, now time.Time) {
	line = cleanLogLine(line)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.last = now
	switch {
	case strings.Contains(line, "get_learned_condition completed"), strings.Contains(line, "encode_first_stage completed"):
		l.encodeSec += takingSeconds(line)
	case strings.Contains(line, "sampling completed"), strings.Contains(line, "sampling(high noise) completed"):
		l.sampleSec += takingSeconds(line)
		l.sampled = true
	case strings.Contains(line, "decode_first_stage completed"):
		l.decodeSec += takingSeconds(line)
		l.decoded = true
	case strings.Contains(line, "retrying with spatial"):
		l.fallback, l.spatial = true, true
	case strings.Contains(line, "decode ran out of memory; retrying with temporal tiling"):
		l.fallback = true
	}
}

// takingSeconds is the time in a "taking …" line, in seconds; 0 without one.
func takingSeconds(line string) float64 {
	m := takingRe.FindStringSubmatch(line)
	if m == nil {
		return 0
	}
	n, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0
	}
	if m[2] == "ms" {
		n /= 1000
	}
	return n
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

// jobSeconds works out how long a whole video at the measured size takes:
// the encode times, the sampling time per step (measured over
// measuredSteps steps) times the set's steps, and the decode. 0 when the
// log gave no sampling or decode time, or steps is unknown.
func (l *measureLog) jobSeconds(measuredSteps, steps int) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.sampled || !l.decoded || measuredSteps <= 0 || steps <= 0 || l.sampleSec <= 0 {
		return 0
	}
	perStep := l.sampleSec / float64(measuredSteps)
	return int(math.Round(l.encodeSec + perStep*float64(steps) + l.decodeSec))
}

// decodeFallback says the decode did not fit at once and was retried tiled.
func (l *measureLog) decodeFallback() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.fallback
}
