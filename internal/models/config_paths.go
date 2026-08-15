package models

import (
	"fmt"
	"os"
	"path/filepath"
)

// ResolveConfigPaths rewrites a config's machine-local GGUF path fields
// (MmprojPath, MtpPath, DraftModelPath) for this machine, returning a
// warning per field it had to blank. Used by the backup restore engine
// at apply time and by the registry's pending-config claim — it lives in
// this package so the registry can call it without an import cycle
// (internal/backup already imports internal/models).
//
// The backup export emits exactly two path shapes: relatives (the file
// was under the source's models dir) and absolutes (it wasn't). A
// relative path joins onto modelsDir; an absolute path is used as-is;
// either way, a path whose file doesn't exist here is blanked with a
// warning rather than shipped to llama-server as a broken flag. The
// disabled flags (MmprojDisabled, MtpDisabled) are left untouched.
func ResolveConfigPaths(cfg *ModelConfig, modelsDir string) []string {
	var warnings []string
	resolve := func(field string, p *string) {
		if *p == "" {
			return
		}
		full := *p
		if !filepath.IsAbs(full) {
			full = filepath.Join(modelsDir, full)
		}
		if st, err := os.Stat(full); err != nil || st.IsDir() {
			warnings = append(warnings, fmt.Sprintf("%s %q not found on this machine — cleared", field, *p))
			*p = ""
			return
		}
		*p = full
	}
	resolve("mmproj", &cfg.MmprojPath)
	resolve("mtp model", &cfg.MtpPath)
	resolve("draft model", &cfg.DraftModelPath)
	return warnings
}
