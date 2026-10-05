package api

import (
	"os"
	"strings"
)

// envLine is one row of the effective-environment preview: the KEY=VALUE
// pair as configured, and — when the service environment already defines
// the same name — that name, because the inherited value wins instead.
// Only the name is shown: the inherited value can be a token set in the
// container or systemd environment, and this page has no login. Mirrors
// applyExtraEnv in internal/process/manager.go, which gives the
// inherited environment precedence over UI-set values so a systemd
// drop-in or container env stays authoritative.
type envLine struct {
	Text       string
	Overridden string
}

// effectiveEnvLines renders the configured runtime environment as the
// launch will apply it, annotating entries the inherited service
// environment overrides.
func (s *Server) effectiveEnvLines() []envLine {
	var out []envLine
	for _, kv := range s.cfg.EnvSet().Pairs() {
		line := envLine{Text: kv}
		name := kv
		if i := strings.IndexByte(kv, '='); i > 0 {
			name = kv[:i]
		}
		if _, ok := os.LookupEnv(name); ok {
			line.Overridden = name
		}
		out = append(out, line)
	}
	return out
}
