package api

// The client for sd-server's DGL job API (stable-diffusion.cpp master-1116
// on, examples/server/routes_dgl.cpp), which video upscaler sets use:
// POST /sdcpp/v1/dgl/jobs {"source": <script>} answers 202 {id}, and
// GET /sdcpp/v1/dgl/jobs/{id} reports the job. sd-server takes these
// routes only from 127.0.0.1 or ::1 with a local Host and no foreign
// Origin, which toolchest's own calls to http://127.0.0.1:<port> are.
//
// When its context ends, runDGLJob asks sd-server to cancel the job
// (POST /sdcpp/v1/dgl/jobs/{id}/cancel, master-1132 on) and waits a short
// time for it to stop. Builds before master-1132 answer 404 and cannot stop
// a DGL job; then runDGLJob returns errDGLStillRunning as well, and the
// caller stops the instance to free the GPU.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// dglProgress is a DGL job's progress as sd-server reports it. A DGL job
// runs as one step, so Stage is mostly "executing" and Completed/Total
// are often absent.
type dglProgress struct {
	Stage     string `json:"stage"`
	Detail    string `json:"detail"`
	Completed *int   `json:"completed"`
	Total     *int   `json:"total"`
	ElapsedMS int64  `json:"stage_elapsed_ms"`
}

// dglResult is what toolchest's upscale scripts return: the shape of the
// video they saved.
type dglResult struct {
	Width, Height, Frames int
}

type dglJobStatus struct {
	ID       string       `json:"id"`
	Status   string       `json:"status"`
	Progress *dglProgress `json:"progress"`
	Error    dglJobError  `json:"error"`
	Result   *struct {
		Outputs []struct {
			Name  string          `json:"name"`
			Value json.RawMessage `json:"value"`
		} `json:"outputs"`
	} `json:"result"`
}

// errDGLBusy is sd-server's full job queue (HTTP 429).
var errDGLBusy = errors.New("the upscaler is busy; try again later")

// errDGLMemory marks a DGL job that failed for lack of GPU memory.
var errDGLMemory = errors.New("sd.cpp ran out of GPU memory for this clip size or window; send a smaller clip or window, or measure the instance to see what fits. Its log on the Server page says more")

// errDGLStillRunning marks a DGL job that sd-server did not stop when
// runDGLJob's context ended: the build has no DGL cancel route, or the job
// did not stop within dglCancelWait.
var errDGLStillRunning = errors.New("sd-server is still running the upscale job")

// dglCancelWait is how long runDGLJob waits for a cancelled job to stop.
// sd-server stops at its next DGL cancellation check, so the operation
// that is running may finish first.
var dglCancelWait = 30 * time.Second

// dglJobError is a failed DGL job's error: an object with code and message
// (master-1116 on), or a bare string.
type dglJobError struct{ Code, Message string }

func (e *dglJobError) UnmarshalJSON(b []byte) error {
	var text string
	if json.Unmarshal(b, &text) == nil {
		e.Message = text
		return nil
	}
	var obj struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	e.Code, e.Message = obj.Code, obj.Message
	return nil
}

// dglMemoryMarkers are what a DGL job's error says when sd.cpp's engine
// could not make room for a step. From master-1129 the job's error code is
// "out_of_memory" and the message gives the memory details. Before that,
// the engine logged "cannot make enough memory available", but the job's
// own error was only the empty result ("diffusion output shape [] does
// not match input") or "VAE encode failed" (master-1116, RTX A4000:
// RelayVSR ×4 and SeedVR2 window 5 at 832x480, 2026-11-24).
var dglMemoryMarkers = []string{"output shape [] does not match", "vae encode failed", "vae decode failed", "out of memory", "failed to allocate", "cannot make enough memory"}

// dglMemoryError is errDGLMemory with sd.cpp's own words, kept apart so
// Measure can give them with its own advice.
type dglMemoryError struct{ sdcpp string }

func (e dglMemoryError) Error() string {
	return fmt.Sprintf("%s (sd.cpp: %s)", errDGLMemory, e.sdcpp)
}

func (e dglMemoryError) Is(target error) bool { return target == errDGLMemory }

// runDGLJob runs one DGL script on the instance at base
// (http://127.0.0.1:port) and returns what it returned. onProgress, when
// set, hears each change of progress. When ctx ends first, runDGLJob
// returns ctx's error at once and sends nothing more to sd-server.
func runDGLJob(ctx context.Context, base, source string, onProgress func(dglProgress)) (dglResult, error) {
	st, err := runDGLScript(ctx, base, source, onProgress)
	if err != nil {
		return dglResult{}, err
	}
	return st.result()
}

// runDGLScript is runDGLJob for a script that returns something else: it
// returns the completed job's status as sd-server reported it.
func runDGLScript(ctx context.Context, base, source string, onProgress func(dglProgress)) (dglJobStatus, error) {
	id, err := submitDGL(ctx, base, source)
	if err != nil {
		return dglJobStatus{}, err
	}
	jobURL := base + "/sdcpp/v1/dgl/jobs/" + id
	var last dglProgress
	for {
		select {
		case <-ctx.Done():
			return dglJobStatus{}, cancelDGL(base, id, ctx.Err())
		case <-time.After(vidGenPoll):
		}
		out, _, err := getBody(ctx, jobURL)
		if err != nil {
			if ctx.Err() != nil {
				return dglJobStatus{}, cancelDGL(base, id, ctx.Err())
			}
			return dglJobStatus{}, err
		}
		var st dglJobStatus
		if err := json.Unmarshal(out, &st); err != nil {
			return dglJobStatus{}, fmt.Errorf("reading the DGL job status: %w", err)
		}
		switch st.Status {
		case "completed":
			return st, nil
		case "failed":
			return dglJobStatus{}, dglError(st.Error)
		case "cancelled":
			return dglJobStatus{}, errors.New("sd-server cancelled the upscale job")
		}
		if st.Progress != nil && onProgress != nil && !sameProgress(*st.Progress, last) {
			last = *st.Progress
			onProgress(last)
		}
	}
}

func sameProgress(a, b dglProgress) bool {
	eq := func(x, y *int) bool { return (x == nil && y == nil) || (x != nil && y != nil && *x == *y) }
	return a.Stage == b.Stage && a.Detail == b.Detail && eq(a.Completed, b.Completed) && eq(a.Total, b.Total)
}

// cancelDGL asks sd-server to cancel job id after runDGLJob's context
// ended with cause, and waits up to dglCancelWait for the job to stop. It
// returns cause, joined with errDGLStillRunning when the job may still run.
func cancelDGL(base, id string, cause error) error {
	ctx, done := context.WithTimeout(context.Background(), dglCancelWait)
	defer done()
	jobURL := base + "/sdcpp/v1/dgl/jobs/" + id
	still := fmt.Errorf("%w (%w)", cause, errDGLStillRunning)
	req, err := http.NewRequestWithContext(ctx, "POST", jobURL+"/cancel", strings.NewReader("{}"))
	if err != nil {
		return still
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := genHTTPClient.Do(req)
	if err != nil {
		slog.Warn("could not ask sd-server to cancel a DGL job", "job", id, "error", err)
		return still
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		// 404 before master-1132: no DGL cancel route.
		slog.Info("sd-server cannot cancel DGL jobs (stable-diffusion.cpp before master-1132)", "job", id, "status", resp.StatusCode)
		return still
	}
	for {
		out, _, err := getBody(ctx, jobURL)
		if err == nil {
			var st dglJobStatus
			if json.Unmarshal(out, &st) == nil {
				switch st.Status {
				case "cancelled", "failed", "completed":
					return cause
				}
			}
		}
		select {
		case <-ctx.Done():
			slog.Warn("sd-server did not stop a cancelled DGL job in time", "job", id, "wait", dglCancelWait)
			return still
		case <-time.After(vidGenPoll):
		}
	}
}

func submitDGL(ctx context.Context, base, source string) (string, error) {
	body, err := json.Marshal(map[string]string{"source": source})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", base+"/sdcpp/v1/dgl/jobs", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := genHTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		return "", errDGLBusy
	case resp.StatusCode == http.StatusForbidden:
		slog.Error("sd-server refused a DGL request as not local; toolchest sends it to 127.0.0.1", "url", req.URL.String(), "host", req.Host, "body", string(out))
		return "", errors.New("sd-server refused the DGL request (local-only check); this is a toolchest bug")
	case resp.StatusCode >= 300:
		return "", fmt.Errorf("sd-server answered HTTP %d to the upscale job: %.200s", resp.StatusCode, out)
	}
	var job struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(out, &job); err != nil || job.ID == "" {
		return "", errors.New("sd-server returned no upscale job ID")
	}
	return job.ID, nil
}

// result reads the width, height and frames the script returned. sd-server
// sends 64-bit integers as strings and smaller ones as numbers.
func (st dglJobStatus) result() (dglResult, error) {
	if st.Result == nil {
		return dglResult{}, errors.New("sd-server finished the upscale job without a result")
	}
	var r dglResult
	found := 0
	for _, o := range st.Result.Outputs {
		var dst *int
		switch o.Name {
		case "width":
			dst = &r.Width
		case "height":
			dst = &r.Height
		case "frames":
			dst = &r.Frames
		default:
			continue
		}
		v, err := jsonInt(o.Value)
		if err != nil {
			return dglResult{}, fmt.Errorf("reading the upscale result's %s: %w", o.Name, err)
		}
		*dst = v
		found++
	}
	if found != 3 {
		return dglResult{}, errors.New("sd-server's upscale result lacks the width, height or frame count")
	}
	return r, nil
}

func jsonInt(raw json.RawMessage) (int, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return strconv.Atoi(s)
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0, err
	}
	return int(f), nil
}

// dglError turns a failed DGL job's message into the job's error, with
// clearer words for the errors a user can act on.
func dglError(e dglJobError) error {
	msg := firstNonEmpty(e.Message, e.Code)
	if msg == "" {
		msg = "no reason given"
	}
	if e.Code == "out_of_memory" {
		return dglMemoryError{sdcpp: msg}
	}
	lower := strings.ToLower(msg)
	for _, m := range dglMemoryMarkers {
		if strings.Contains(lower, m) {
			return dglMemoryError{sdcpp: msg}
		}
	}
	switch {
	case strings.Contains(msg, "exceeds") && strings.Contains(msg, "MiB"),
		strings.Contains(msg, "exceeds") && strings.Contains(msg, "GiB"):
		return fmt.Errorf("the clip is too large for sd.cpp's DGL limits: %s", msg)
	case strings.Contains(msg, "unknown function") && strings.Contains(msg, "relayvsr::"):
		return fmt.Errorf("this stable-diffusion.cpp build has no RelayVSR support; use master-1116 or newer: %s", msg)
	case strings.Contains(msg, "unknown function") && strings.Contains(msg, "seedvr2::"):
		return fmt.Errorf("this stable-diffusion.cpp build has no SeedVR2 support; use master-1116 or newer: %s", msg)
	}
	return fmt.Errorf("sd.cpp: %s", msg)
}
