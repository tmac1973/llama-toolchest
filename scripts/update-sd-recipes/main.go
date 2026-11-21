// Command update-sd-recipes refreshes the recipe snapshot that
// llama-toolchest embeds (internal/generation/sdrecipes) from a
// stable-diffusion.cpp checkout's examples/server/recipes/*.json.
//
//	go run ./scripts/update-sd-recipes <stable-diffusion.cpp checkout>
//
// It deletes the old *.json files, copies the new ones and writes SOURCE
// with the checkout's tag, commit and date. Run it from the repository
// root.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const dest = "internal/generation/sdrecipes"

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: go run ./scripts/update-sd-recipes <stable-diffusion.cpp checkout>")
		os.Exit(2)
	}
	if err := run(os.Args[1]); err != nil {
		fmt.Fprintln(os.Stderr, "update-sd-recipes:", err)
		os.Exit(1)
	}
}

func run(checkout string) error {
	src := filepath.Join(checkout, "examples", "server", "recipes")
	files, err := filepath.Glob(filepath.Join(src, "*.json"))
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fmt.Errorf("no recipe files in %s", src)
	}
	if _, err := os.Stat(dest); err != nil {
		return fmt.Errorf("%s not found; run this from the repository root", dest)
	}
	old, err := filepath.Glob(filepath.Join(dest, "*.json"))
	if err != nil {
		return err
	}
	for _, f := range old {
		if err := os.Remove(f); err != nil {
			return err
		}
	}
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dest, filepath.Base(f)), data, 0o644); err != nil {
			return err
		}
	}
	tag, err := git(checkout, "describe", "--tags", "--always")
	if err != nil {
		return err
	}
	sha, err := git(checkout, "rev-parse", "HEAD")
	if err != nil {
		return err
	}
	date, err := git(checkout, "log", "-1", "--format=%cs")
	if err != nil {
		return err
	}
	source := fmt.Sprintf("stable-diffusion.cpp %s %s %s\n", tag, sha, date)
	if err := os.WriteFile(filepath.Join(dest, "SOURCE"), []byte(source), 0o644); err != nil {
		return err
	}
	fmt.Printf("copied %d recipe files; %s", len(files), source)
	return nil
}

func git(dir string, args ...string) (string, error) {
	out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return strings.TrimSpace(string(out)), nil
}
