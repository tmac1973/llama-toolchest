package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/api"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// Build info, populated via -ldflags by goreleaser.
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	configPath := flag.String("config", config.DefaultConfigPath(), "config file path")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("llama-toolchest %s (commit %s, built %s)\n", version, commit, date)
		return
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	// log_level was accepted in config and documented but never wired to
	// slog, so it silently did nothing and debug logging was
	// unreachable.
	configureLogging(cfg.LogLevel)

	if err := initDataDir(cfg); err != nil {
		slog.Warn("could not init data dir (expected in local dev)", "error", err)
	}

	srv := api.NewServer(cfg, *configPath)
	srv.SetVersion(version)

	httpSrv := &http.Server{
		Addr:    cfg.ListenAddr,
		Handler: srv.Router(),
	}

	// Graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("listening", "addr", cfg.ListenAddr)
		if err := httpSrv.ListenAndServe(); err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")

	// Stop the child llama-server in parallel with draining HTTP connections.
	// The HTTP deadline keeps us well under systemd's TimeoutStopSec (90s) even
	// when slow handlers (benchmark warmup, /api/monitor polls) are in flight.
	done := make(chan struct{})
	go func() {
		srv.Shutdown()
		close(done)
	}()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		slog.Warn("http shutdown", "error", err)
	}
	<-done
}

func initDataDir(cfg *config.Config) error {
	dirs := []string{
		filepath.Join(cfg.DataDir, "config"),
		filepath.Join(cfg.DataDir, "builds"),
		cfg.ModelsPath(),
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("creating %s: %w", dir, err)
		}
	}
	// A crash mid-KL-base-generation leaves a <name>.kld.partial behind
	// (clean exits rename it, failures delete it); remove stale corpses
	// once at startup, next to the directory creation.
	if n, err := evaluate.CleanStalePartials(evaluate.EvalDataRoot(cfg.DataDir)); err != nil {
		slog.Warn("cleaning stale KL cache partials", "error", err)
	} else if n > 0 {
		slog.Info("removed stale KL cache partials", "count", n)
	}
	return nil
}

// configureLogging installs the default slog handler at the configured
// level. Unknown values fall back to info rather than failing startup.
func configureLogging(level string) {
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})))
}
