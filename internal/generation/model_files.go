package generation

import (
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
)

// modelFileExts are the files the custom set form suggests for a model
// slot.
var modelFileExts = []string{".gguf", ".safetensors", ".pth", ".ckpt"}

// ListModelFiles returns the files under dir that the custom set form
// suggests: model files, and tokenizer JSON files for the tokenizer slot.
// Both are full paths, sorted, and together at most max; truncated says
// the walk stopped there. It reads no file, skips hidden folders and
// unfinished downloads (.part), and does not follow linked folders.
func ListModelFiles(dir string, max int) (models, tokenizers []string, truncated bool) {
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// An unreadable folder is skipped, not the whole walk.
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if path != dir && strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(name))
		isModel := slices.Contains(modelFileExts, ext)
		isTokenizer := ext == ".json" && strings.Contains(strings.ToLower(name), "tokenizer")
		if !isModel && !isTokenizer {
			return nil
		}
		if len(models)+len(tokenizers) == max {
			truncated = true
			return filepath.SkipAll
		}
		if isModel {
			models = append(models, path)
		} else {
			tokenizers = append(tokenizers, path)
		}
		return nil
	})
	slices.Sort(models)
	slices.Sort(tokenizers)
	return models, tokenizers, truncated
}
