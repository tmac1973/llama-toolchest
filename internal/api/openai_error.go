package api

import (
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// maxImageRequestBytes bounds an image or video request body read into
// memory: a prompt, or an edit's source image and mask.
const maxImageRequestBytes = 128 << 20

// openAIError writes an error in the OpenAI API's format.
func openAIError(w http.ResponseWriter, status int, errType, code, message string) {
	respondJSONStatus(w, status, map[string]any{
		"error": map[string]any{"message": message, "type": errType, "code": code},
	})
}

// writeOpenAIError writes an error in the OpenAI API's format, typed by its
// status: server_error from 500 up, invalid_request_error below.
func writeOpenAIError(w http.ResponseWriter, status int, code, message string) {
	errType := "invalid_request_error"
	if status >= 500 {
		errType = "server_error"
	}
	openAIError(w, status, errType, code, message)
}

// requestError is a refused request: its HTTP status, the OpenAI error
// code and the message.
type requestError struct {
	status int
	code   string
	msg    string
}

func (e *requestError) Error() string { return e.msg }

func badRequest(code, format string, args ...any) *requestError {
	return &requestError{http.StatusBadRequest, code, fmt.Sprintf(format, args...)}
}

// tooLargeError is the answer to a body over maxImageRequestBytes.
func tooLargeError() *requestError {
	return &requestError{http.StatusRequestEntityTooLarge, "too_large",
		fmt.Sprintf("the request is larger than %d MiB", maxImageRequestBytes>>20)}
}

func writeRequestError(w http.ResponseWriter, e *requestError) {
	writeOpenAIError(w, e.status, e.code, e.msg)
}

// notEnabledMessage is the 404 text for a generation kind ("image",
// "video", "music") on a server without generation.
func notEnabledMessage(kind string) string {
	return kind + " generation is not enabled on this server"
}

func writeNotEnabled(w http.ResponseWriter, kind string) {
	writeOpenAIError(w, http.StatusNotFound, "not_found", notEnabledMessage(kind))
}

// writeQueueError answers a failed queue submit: 429 with Retry-After when
// the GPU's queue is full, 500 otherwise.
func writeQueueError(w http.ResponseWriter, err error) {
	if errors.Is(err, generation.ErrQueueFull) {
		w.Header().Set("Retry-After", "30")
		openAIError(w, http.StatusTooManyRequests, "server_error", "queue_full", err.Error())
		return
	}
	openAIError(w, http.StatusInternalServerError, "server_error", "queue_error", err.Error())
}

// writeJobAccepted answers 202 for a queued job.
func writeJobAccepted(w http.ResponseWriter, id, status, phase string, position int) {
	respondJSONStatus(w, http.StatusAccepted, map[string]any{"id": id, "status": status, "phase": phase, "queue_position": position})
}

// readImageBody reads a request body of at most maxImageRequestBytes.
func readImageBody(r *http.Request) ([]byte, *requestError) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxImageRequestBytes+1))
	if err != nil {
		return nil, badRequest("bad_request", "reading the request: %v", err)
	}
	if len(body) > maxImageRequestBytes {
		return nil, tooLargeError()
	}
	return body, nil
}
