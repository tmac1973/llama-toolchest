package api

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// apiKeyAuth returns middleware that checks for a valid Bearer token.
// If no API key is configured, all requests are allowed.
func (s *Server) apiKeyAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.APIKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			writeAuthError(w, "missing API key")
			return
		}

		// Constant-time, so the response time says nothing about how much
		// of a guessed key was right.
		token := strings.TrimPrefix(auth, "Bearer ")
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.cfg.APIKey)) != 1 {
			writeAuthError(w, "invalid API key")
			return
		}

		next.ServeHTTP(w, r)
	})
}

// writeAuthError answers 401 with an OpenAI-shaped error body, sent as
// JSON so OpenAI clients can read the message.
func writeAuthError(w http.ResponseWriter, message string) {
	respondJSONStatus(w, http.StatusUnauthorized, map[string]any{
		"error": map[string]any{"message": message, "type": "auth_error"},
	})
}
