package api

// The client for sd-server's DGL job API (stable-diffusion.cpp master-1116
// on, examples/server/routes_dgl.cpp), which video upscaler sets use:
// POST /sdcpp/v1/dgl/jobs {"source": <script>} answers 202 {id}, and
// GET /sdcpp/v1/dgl/jobs/{id} reports the job. sd-server takes these
// routes only from 127.0.0.1 or ::1 with a local Host and no foreign
// Origin, which toolchest's own calls to http://127.0.0.1:<port> are.
//
// A DGL job cannot be cancelled: sd-server's /sdcpp/v1/jobs/{id}/cancel
// answers 404 for DGL jobs. runDGLJob only stops waiting when its context
// ends; to free the GPU, the caller stops the instance.

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
	Error    vidJobError  `json:"error"`
	Result   *struct {
		Outputs []struct {
			Name  string          `json:"name"`
			Value json.RawMessage `json:"value"`
		} `json:"outputs"`
	} `json:"result"`
}

// errDGLBusy is sd-server's full job queue (HTTP 429).
var errDGLBusy = errors.New("the upscaler is busy; try again later")

// runDGLJob runs one DGL script on the instance at base
// (http://127.0.0.1:port) and returns what it returned. onProgress, when
// set, hears each change of progress. When ctx ends first, runDGLJob
// returns ctx's error at once and sends nothing more to sd-server.
func runDGLJob(ctx context.Context, base, source string, onProgress func(dglProgress)) (dglResult, error) {
	id, err := submitDGL(ctx, base, source)
	if err != nil {
		return dglResult{}, err
	}
	jobURL := base + "/sdcpp/v1/dgl/jobs/" + id
	var last dglProgress
	for {
		select {
		case <-ctx.Done():
			return dglResult{}, ctx.Err()
		case <-time.After(vidGenPoll):
		}
		out, _, err := getBody(ctx, jobURL)
		if err != nil {
			if ctx.Err() != nil {
				return dglResult{}, ctx.Err()
			}
			return dglResult{}, err
		}
		var st dglJobStatus
		if err := json.Unmarshal(out, &st); err != nil {
			return dglResult{}, fmt.Errorf("reading the DGL job status: %w", err)
		}
		switch st.Status {
		case "completed":
			return st.result()
		case "failed":
			return dglResult{}, dglError(string(st.Error))
		case "cancelled":
			return dglResult{}, errors.New("sd-server cancelled the upscale job")
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
func dglError(msg string) error {
	if msg == "" {
		msg = "no reason given"
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
