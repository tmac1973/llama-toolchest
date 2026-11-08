package api

import (
	"strings"
	"time"
)

// chatPathSuffixes are the request paths the virtual model name applies
// to: the chat and text completion APIs, with or without /v1.
var chatPathSuffixes = []string{"/chat/completions", "/completions", "/messages", "/responses"}

// isChatPath reports whether path ends in one of chatPathSuffixes.
func isChatPath(path string) bool {
	for _, suffix := range chatPathSuffixes {
		if strings.HasSuffix(path, suffix) {
			return true
		}
	}
	return false
}

// noteChatUse records router model id as the chat model used last, for
// the virtual model name after a restart. Forks are not recorded: one
// does not start on request, so after a restart it could only answer 503.
func (s *Server) noteChatUse(id string) {
	if s.defaultState != nil {
		s.defaultState.note(id, false, time.Now())
	}
}
