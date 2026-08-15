package api

import (
	"fmt"
	"net/http"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/backup"
)

// handleBackupExport serves the configuration backup as a JSON download.
// Same-origin UI route like the rest of /api (apiKeyAuth guards /v1
// only) — a secrets-bearing export has the same exposure as the existing
// settings API, which the help page states; secrets are included only on
// the explicit ?secrets=1 opt-in.
func (s *Server) handleBackupExport(w http.ResponseWriter, r *http.Request) {
	includeSecrets := r.URL.Query().Get("secrets") == "1"

	s.cfgMu.Lock()
	f := backup.Assemble(s.cfg, s.builder, s.registry, s.monitor.Current().GPU, includeSecrets)
	s.cfgMu.Unlock()

	data, err := f.Marshal()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=llama-toolchest-backup-%s.json", time.Now().Format("2006-01-02")))
	w.Write(data)
}
