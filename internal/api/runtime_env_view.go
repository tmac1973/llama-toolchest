package api

import (
	"os"
	"strings"
)

// envLine is one row of the effective-environment preview: the KEY=VALUE
// pair as configured, and — when the service environment already defines
// the same name — the inherited value that wins instead. Mirrors
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
		if inherited, ok := os.LookupEnv(name); ok {
			line.Overridden = name + "=" + inherited
		}
		out = append(out, line)
	}
	return out
}
