package config

import (
	"os"
	"path/filepath"
	"runtime"
)

const appName = "llama-toolchest"

// platformAppDirs returns the candidate app-scoped directories for the
// current OS in priority order: ~/Library/Application Support on darwin,
// %LOCALAPPDATA% on windows, and on other systems the given XDG env var
// followed by the home-relative fallback (e.g. ".local/share"). Empty when
// neither the env var nor a home directory is resolvable. Shared by
// DefaultDataDir and configSearchPaths so the per-OS resolution lives in
// one place.
func platformAppDirs(xdgVar, homeRel string) []string {
	switch runtime.GOOS {
	case "darwin":
		if home, err := os.UserHomeDir(); err == nil {
			return []string{filepath.Join(home, "Library", "Application Support", appName)}
		}
	case "windows":
		if dir := os.Getenv("LOCALAPPDATA"); dir != "" {
			return []string{filepath.Join(dir, appName)}
		}
	default:
		var dirs []string
		if dir := os.Getenv(xdgVar); dir != "" {
			dirs = append(dirs, filepath.Join(dir, appName))
		}
		if home, err := os.UserHomeDir(); err == nil {
			dirs = append(dirs, filepath.Join(home, homeRel, appName))
		}
		return dirs
	}
	return nil
}

// DefaultDataDir returns the platform-appropriate data directory for a host
// install. Containers override this via the YAML's data_dir field.
func DefaultDataDir() string {
	if dirs := platformAppDirs("XDG_DATA_HOME", filepath.Join(".local", "share")); len(dirs) > 0 {
		return dirs[0]
	}
	return "/data"
}

// DefaultConfigPath returns the config file path to load. It honors
// LLAMA_TOOLCHEST_CONFIG if set; otherwise it returns the first candidate
// location that exists, falling back to the preferred user-scope path for
// writing when none exist.
//
// The existence check is what lets a system-scope install find
// /etc/llama-toolchest/llama-toolchest.yaml without the systemd unit having to
// pass --config or set an env var. Before this, the default resolved only to
// ~/.config (or /data), so a root service silently ignored the documented
// /etc config and used built-in defaults instead (issue #61).
func DefaultConfigPath() string {
	if path := os.Getenv("LLAMA_TOOLCHEST_CONFIG"); path != "" {
		return path
	}
	candidates := configSearchPaths()
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	// Nothing exists yet — return the preferred writable default (the first
	// candidate), so a later save lands in the conventional per-user location.
	if len(candidates) > 0 {
		return candidates[0]
	}
	return "/data/config/llama-toolchest.yaml"
}

// configSearchPaths returns the candidate config locations for the current OS,
// in priority order. The first existing one wins in DefaultConfigPath.
func configSearchPaths() []string {
	const file = "llama-toolchest.yaml"
	var paths []string
	for _, dir := range platformAppDirs("XDG_CONFIG_HOME", ".config") {
		paths = append(paths, filepath.Join(dir, file))
	}
	switch runtime.GOOS {
	case "darwin", "windows":
		if len(paths) == 0 {
			return []string{"/data/config/llama-toolchest.yaml"}
		}
	default:
		// System-scope install location, checked after the per-user paths so a
		// user config still takes precedence when both are present. Appended
		// even when no per-user dir resolves (e.g. a root service without
		// HOME), so /etc configs are still found — see issue #61.
		paths = append(paths, filepath.Join("/etc", appName, file))
	}
	return paths
}
