package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// sseKeepAliveInterval is how often a quiet stream sends a comment line.
// The write fails once the browser has gone without closing the
// connection (a laptop put to sleep, a network that dropped), so the
// stream ends instead of waiting for a log line that may never come.
var sseKeepAliveInterval = 30 * time.Second

// SSEWriter wraps an http.ResponseWriter for Server-Sent Events.
type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func NewSSEWriter(w http.ResponseWriter) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Send the headers now: the browser reports the stream as open (and
	// the page clears a log panel before the replay) without waiting for
	// the first line.
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	return &SSEWriter{w: w, flusher: flusher}, nil
}

func (s *SSEWriter) SendEvent(event, data string) error {
	if event != "" {
		fmt.Fprintf(s.w, "event: %s\n", event)
	}
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()
	return nil
}

func (s *SSEWriter) SendData(data string) error {
	return s.SendEvent("", data)
}

// SendLine sends data with a trailing newline embedded using multi-line SSE format.
// SSE spec: each "data:" line appends value+LF to the buffer; trailing LF is stripped.
// So "data: X\ndata:\n\n" produces event data = "X\n".
func (s *SSEWriter) SendLine(data string) error {
	fmt.Fprintf(s.w, "data: %s\ndata:\n\n", data)
	s.flusher.Flush()
	return nil
}

// SendKeepAlive sends a comment line, which browsers ignore. It returns
// the write error, which means the client is gone.
func (s *SSEWriter) SendKeepAlive() error {
	if _, err := fmt.Fprint(s.w, ": keep-alive\n\n"); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// StreamLines streams log lines from a channel, sending each as an SSE line
// event. When the channel closes, a "done" event is sent with the given message.
// Exits when the channel closes, the context is cancelled, or a keep-alive
// cannot be written. Pages close the stream when the done event arrives;
// see web/static/live-stream.js.
func StreamLines(w http.ResponseWriter, ctx context.Context, ch <-chan string, doneMsg string) {
	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	keepAlive := time.NewTicker(sseKeepAliveInterval)
	defer keepAlive.Stop()
	for {
		select {
		case <-keepAlive.C:
			if sse.SendKeepAlive() != nil {
				return
			}
		case line, ok := <-ch:
			if !ok {
				sse.SendEvent("done", doneMsg)
				return
			}
			sse.SendLine(line)
		case <-ctx.Done():
			return
		}
	}
}
