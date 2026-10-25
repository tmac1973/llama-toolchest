package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

// aceJobPoll is how often a running ace-server job is checked.
const aceJobPoll = 250 * time.Millisecond

// runAceJob submits an ace-server job (POST path), waits for it, and returns
// its result body and Content-Type. onJobID, when set, is told the backend's
// job ID as soon as it is known, so the job can be cancelled. When ctx ends
// first, the backend job is cancelled with ace-server's
// POST /job?id=N&cancel=1 and ctx's error is returned. With files, the
// request is multipart: body in a "request" part and each file in a part of
// its own, as ace-server's /synth takes source and reference audio.
func runAceJob(ctx context.Context, base, path string, body any, onJobID func(string), files ...aceFile) ([]byte, string, error) {
	var out []byte
	var err error
	if len(files) > 0 {
		out, _, err = postAceMultipart(ctx, base+path, body, files)
	} else {
		out, _, err = postJSON(ctx, base+path, body)
	}
	if err != nil {
		return nil, "", err
	}
	var job struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(out, &job); err != nil || job.ID == "" {
		return nil, "", fmt.Errorf("%s returned no job ID", path)
	}
	if onJobID != nil {
		onJobID(job.ID)
	}
	for {
		select {
		case <-ctx.Done():
			// Not tied to ctx, which has ended.
			cctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			postJSON(cctx, base+"/job?id="+job.ID+"&cancel=1", nil)
			cancel()
			return nil, "", ctx.Err()
		case <-time.After(aceJobPoll):
		}
		st, _, err := getBody(ctx, base+"/job?id="+job.ID)
		if err != nil {
			if ctx.Err() != nil {
				continue // cancelled while polling; handled above
			}
			return nil, "", err
		}
		var status struct {
			Status string `json:"status"`
		}
		json.Unmarshal(st, &status)
		switch status.Status {
		case "done":
			return getBody(ctx, base+"/job?id="+job.ID+"&result=1")
		case "failed", "cancelled", "error":
			return nil, "", fmt.Errorf("ace-server %s job %s", strings.TrimPrefix(path, "/"), status.Status)
		}
	}
}

// audioPart returns the first audio part of a multipart/mixed body: what
// ace-server's /synth result holds, next to the song's latent.
func audioPart(contentType string, body []byte) (data []byte, partType string, err error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return nil, "", fmt.Errorf("unexpected synth result type %q", contentType)
	}
	mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			return nil, "", errors.New("the synth result holds no audio")
		}
		if err != nil {
			return nil, "", fmt.Errorf("reading the synth result: %w", err)
		}
		if ct := part.Header.Get("Content-Type"); strings.HasPrefix(ct, "audio/") {
			data, err := io.ReadAll(part)
			return data, ct, err
		}
	}
}

func postJSON(ctx context.Context, url string, body any) ([]byte, string, error) {
	var rd io.Reader = http.NoBody
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, "", err
		}
		rd = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, rd)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	return doBody(req)
}

// aceFile is one file part of a multipart ace-server request.
type aceFile struct {
	field, name string
	data        []byte
}

func postAceMultipart(ctx context.Context, url string, body any, files []aceFile) ([]byte, string, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.WriteField("request", string(raw)); err != nil {
		return nil, "", err
	}
	for _, f := range files {
		part, err := mw.CreateFormFile(f.field, f.name)
		if err != nil {
			return nil, "", err
		}
		part.Write(f.data)
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

func getBody(ctx context.Context, url string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, "", err
	}
	return doBody(req)
}

func doBody(req *http.Request) ([]byte, string, error) {
	resp, err := genHTTPClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("%s answered HTTP %d: %.200s", req.URL.Path, resp.StatusCode, out)
	}
	return out, resp.Header.Get("Content-Type"), nil
}
