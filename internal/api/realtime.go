package api

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
)

// realtimePath is llama-server's realtime voice route: a WebSocket session
// in the OpenAI realtime protocol, for live transcription and voice chat.
const realtimePath = "/v1/realtime"

// realtimeKeyProtocol starts the WebSocket subprotocol in which OpenAI
// clients running in a browser send the API key, since a browser cannot
// set an Authorization header on a WebSocket:
// "openai-insecure-api-key.<key>".
const realtimeKeyProtocol = "openai-insecure-api-key."

// realtimeProtocol is the subprotocol OpenAI realtime clients offer
// beside the key.
const realtimeProtocol = "realtime"

// websocketProtocols lists the subprotocols a WebSocket request offers,
// from all its Sec-WebSocket-Protocol headers.
func websocketProtocols(h http.Header) []string {
	var out []string
	for _, v := range h.Values("Sec-WebSocket-Protocol") {
		for p := range strings.SplitSeq(v, ",") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

// realtimeSubprotocolKey returns the API key a request to the realtime
// route sends as a subprotocol, if any.
func realtimeSubprotocolKey(r *http.Request) (string, bool) {
	if r.URL.Path != realtimePath {
		return "", false
	}
	for _, p := range websocketProtocols(r.Header) {
		if key, ok := strings.CutPrefix(p, realtimeKeyProtocol); ok {
			return key, true
		}
	}
	return "", false
}

// realtimeHandler serves GET /v1/realtime. The session names its model in
// the query string; the model is loaded as for a JSON request, with the
// same room checks, and the session counts as one request for the model
// until either side closes it, so a generation job does not unload the
// model during a call. The API key subprotocol is removed before the
// request goes on, so the key does not reach llama-server.
func (s *Server) realtimeHandler(proxy http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if protos := websocketProtocols(r.Header); len(protos) > 0 {
			kept := slices.DeleteFunc(protos, func(p string) bool {
				return strings.HasPrefix(p, realtimeKeyProtocol)
			})
			r.Header.Del("Sec-WebSocket-Protocol")
			if len(kept) > 0 {
				r.Header.Set("Sec-WebSocket-Protocol", strings.Join(kept, ", "))
			}
		}
		if model := r.URL.Query().Get("model"); model != "" {
			body, _ := json.Marshal(map[string]string{"model": model})
			end, ok := s.loadForProxy(w, r, body)
			if !ok {
				return
			}
			// ServeHTTP returns once both sides of the session have closed.
			defer end()
		}
		proxy.ServeHTTP(w, r)
	}
}

// acceptRealtimeProtocol answers a WebSocket upgrade with the "realtime"
// subprotocol when the client offered it and llama-server chose none. A
// browser fails a handshake whose answer names no subprotocol after it
// offered some, and llama-server never names one.
func acceptRealtimeProtocol(resp *http.Response) {
	if resp.Header.Get("Sec-WebSocket-Protocol") != "" || resp.Request == nil {
		return
	}
	if slices.Contains(websocketProtocols(resp.Request.Header), realtimeProtocol) {
		resp.Header.Set("Sec-WebSocket-Protocol", realtimeProtocol)
	}
}
