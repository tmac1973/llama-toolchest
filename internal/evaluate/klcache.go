// KL reference logits cache: key → filename, listing, deletion, the
// .partial interruption discipline, and the disk-space guard.
//
// Files live under <root>/logits/ (root is the eval-data root, passed in
// by callers) as
//
//	<SafeModelID>~<quant>~<dataset>~c<chunks>~ctx<n>~f<fingerprint>.kld
//
// The field separator is ~, NOT --: SafeModelID maps "/" to "--", so a --
// separator would make reverse-parsing the filename ambiguous. ~ is
// filesystem-safe and appears in neither SafeModelID output nor quant
// names, keeping the key → filename → key round trip lossless.

package evaluate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v4/disk"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

const (
	klBaseExt       = ".kld"
	klPartialSuffix = ".partial"
	klKeySeparator  = "~"
)

// KLBaseKey identifies one KL reference logits file.
//
// Chunks is in the key because a base generated at 100 chunks cannot serve
// a full-run comparison — quick and full bases cache separately. Ctx is in
// the key because the base file embeds its n_ctx and the embedded value
// silently wins in the comparison: llama-perplexity only logs on a
// mismatch, it does not reject (perplexity.cpp:1717-1722), so this key is
// the ONLY consistency guard. Today ctx is always EvalContextSize (512);
// keying it means a future constant change invalidates cleanly instead of
// mis-serving.
//
// Fingerprint covers the reference's numerics-affecting evaluation flags
// (KLFlagFingerprint). Without it, editing the reference model's config —
// its KV cache type, say — leaves the old logits in place under an
// unchanged name, and every later comparison silently scores against a
// reference that no longer matches its own stated settings.
type KLBaseKey struct {
	ModelID     string
	Quant       string
	Dataset     string
	Chunks      int // 0 = full run (no --chunks cap)
	Ctx         int
	Fingerprint string // KLFlagFingerprint of the generation flags; "" only for legacy files
}

// Filename renders the deterministic cache filename for the key.
func (k KLBaseKey) Filename() string {
	return strings.Join([]string{
		huggingface.SafeModelID(k.ModelID),
		k.Quant,
		k.Dataset,
		"c" + strconv.Itoa(k.Chunks),
		"ctx" + strconv.Itoa(k.Ctx),
		"f" + k.fingerprintOrNone(),
	}, klKeySeparator) + klBaseExt
}

// fingerprintOrNone renders the fingerprint field, substituting "none"
// for an empty one so the filename always has six fields.
func (k KLBaseKey) fingerprintOrNone() string {
	if k.Fingerprint == "" {
		return "none"
	}
	return k.Fingerprint
}

// klNumericsFlags are the evaluation flags that change the LOGITS a
// generation run produces, as opposed to how fast it produces them.
// Only these enter the cache fingerprint.
//
// Deliberately excluded: --n-gpu-layers, --threads, --direct-io, and the
// placement flags (--device / --tensor-split / --split-mode / --main-gpu).
// They pick kernels and hardware, not arithmetic, and a base file is tens
// of GiB for a large-vocabulary model — keying on them would throw the
// cache away every time a user retunes an unrelated performance setting.
// The caveat that different backends are not bit-identical is real but
// far below the quantization differences these comparisons measure.
var klNumericsFlags = map[string]bool{
	"--flash-attn":   true,
	"--cache-type-k": true,
	"--cache-type-v": true,
	"--batch-size":   true,
	"--ubatch-size":  true,
}

// KLFlagFingerprint reduces a generation flag list to a short stable hex
// digest of its numerics-affecting entries. Flag order does not change
// the result: the pairs are sorted first, so two callers assembling the
// same settings in a different order share one cache entry.
func KLFlagFingerprint(flags []string) string {
	var pairs []string
	for i := 0; i < len(flags); i++ {
		if !klNumericsFlags[flags[i]] {
			continue
		}
		if i+1 < len(flags) && !strings.HasPrefix(flags[i+1], "--") {
			pairs = append(pairs, flags[i]+"="+flags[i+1])
			i++
			continue
		}
		pairs = append(pairs, flags[i])
	}
	sort.Strings(pairs)
	sum := sha256.Sum256([]byte(strings.Join(pairs, " ")))
	return hex.EncodeToString(sum[:])[:12]
}

// KLBasePath returns the final on-disk path of a base file under root.
func KLBasePath(root string, key KLBaseKey) string {
	return filepath.Join(root, "logits", key.Filename())
}

// KLBasePartialPath returns the in-progress generation destination: the
// final name plus .partial. Generation writes here, a clean exit renames
// it into place, and a failure or cancel deletes it (see the package doc
// and CleanStalePartials for why the stakes are higher, not lower, than
// for datasets).
func KLBasePartialPath(root string, key KLBaseKey) string {
	return KLBasePath(root, key) + klPartialSuffix
}

// ParseKLBaseFilename inverts Filename. The returned key's ModelID is the
// SafeModelID (filesystem) form — reverse-mapping "--" back to "/" is not
// unique — so re-rendering it reproduces the filename exactly; callers
// holding the original model ID compare by KLBasePath, not struct
// equality.
func ParseKLBaseFilename(name string) (KLBaseKey, error) {
	var k KLBaseKey
	base, ok := strings.CutSuffix(name, klBaseExt)
	if !ok {
		return k, fmt.Errorf("not a KL base filename (no %s suffix): %s", klBaseExt, name)
	}
	parts := strings.Split(base, klKeySeparator)
	// Six fields today. Five is the pre-fingerprint layout: accepted so
	// such a file still LISTS and can be deleted from the card, but its
	// empty fingerprint re-renders as "~fnone", so a current key never
	// resolves to it and it can never be served.
	if len(parts) != 5 && len(parts) != 6 {
		return k, fmt.Errorf("malformed KL base filename (want 5 or 6 ~-separated fields, got %d): %s", len(parts), name)
	}
	k.ModelID, k.Quant, k.Dataset = parts[0], parts[1], parts[2]
	if k.ModelID == "" || k.Quant == "" || k.Dataset == "" {
		return KLBaseKey{}, fmt.Errorf("malformed KL base filename (empty field): %s", name)
	}
	if !strings.HasPrefix(parts[3], "c") {
		return KLBaseKey{}, fmt.Errorf("malformed KL base filename (chunks field %q): %s", parts[3], name)
	}
	chunks, err := strconv.Atoi(parts[3][1:])
	if err != nil || chunks < 0 {
		return KLBaseKey{}, fmt.Errorf("malformed KL base filename (chunks field %q): %s", parts[3], name)
	}
	k.Chunks = chunks
	if !strings.HasPrefix(parts[4], "ctx") {
		return KLBaseKey{}, fmt.Errorf("malformed KL base filename (ctx field %q): %s", parts[4], name)
	}
	ctx, err := strconv.Atoi(parts[4][3:])
	if err != nil || ctx <= 0 {
		return KLBaseKey{}, fmt.Errorf("malformed KL base filename (ctx field %q): %s", parts[4], name)
	}
	k.Ctx = ctx
	if len(parts) == 6 {
		if !strings.HasPrefix(parts[5], "f") || len(parts[5]) < 2 {
			return KLBaseKey{}, fmt.Errorf("malformed KL base filename (fingerprint field %q): %s", parts[5], name)
		}
		if fp := parts[5][1:]; fp != "none" {
			k.Fingerprint = fp
		}
	}
	return k, nil
}

// KLBaseInfo is one listed cache entry: the key parsed back from the
// filename, plus size and mtime.
type KLBaseInfo struct {
	Key     KLBaseKey
	Path    string
	Size    int64
	ModTime time.Time
}

// ListKLBases lists the cache entries under root, sorted by filename.
// .partial files (in-progress or orphaned generations) and unparseable
// names are ignored — a presence-keyed cache must never serve a corpse.
func ListKLBases(root string) []KLBaseInfo {
	dir := LogitsDir(root)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []KLBaseInfo
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, klPartialSuffix) || !strings.HasSuffix(name, klBaseExt) {
			continue
		}
		key, err := ParseKLBaseFilename(name)
		if err != nil {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out = append(out, KLBaseInfo{
			Key:     key,
			Path:    filepath.Join(dir, name),
			Size:    info.Size(),
			ModTime: info.ModTime(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// HasKLBase reports whether the final base file exists under root. It
// ignores .partial files by construction (it stats the final name).
func HasKLBase(root string, key KLBaseKey) bool {
	_, err := os.Stat(KLBasePath(root, key))
	return err == nil
}

// DeleteKLBase removes a base file under root. Deleting a missing entry is
// not an error (the UI's delete button is idempotent).
func DeleteKLBase(root string, key KLBaseKey) error {
	if err := os.Remove(KLBasePath(root, key)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// DeleteKLBaseFile removes the named base file from root's logits dir.
//
// The name comes from an HTTP form, so it is validated as a name and not
// trusted as a path: it must be a bare filename (no directory part) that
// parses as a KL base filename, and the resolved path must still sit
// directly inside the logits directory. Deleting a missing entry is not
// an error, matching DeleteKLBase.
func DeleteKLBaseFile(root, name string) error {
	if name == "" || name != filepath.Base(name) {
		return fmt.Errorf("not a KL base filename: %q", name)
	}
	if _, err := ParseKLBaseFilename(name); err != nil {
		return err
	}
	dir := LogitsDir(root)
	path := filepath.Join(dir, name)
	if filepath.Dir(path) != filepath.Clean(dir) {
		return fmt.Errorf("not a KL base filename: %q", name)
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// CleanStalePartials removes interrupted-generation corpses
// (<name>.kld.partial) from the logits dir and returns how many it
// removed. A clean exit renames the partial into place and a failure or
// cancel deletes it; a crash in between leaves the file behind, and since
// HasKLBase and ListKLBases ignore partials a stale one would otherwise
// just sit on disk forever. Called once at process startup, next to the
// startup directory creation. A missing logits dir is not an error.
func CleanStalePartials(root string) (int, error) {
	dir := LogitsDir(root)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	removed := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), klPartialSuffix) {
			continue
		}
		if err := os.Remove(filepath.Join(dir, e.Name())); err == nil {
			removed++
		}
	}
	return removed, nil
}

// KLDefaultVocabSize is the worst-case vocab size used when a model's real
// size is unknown (e.g. a registry record parsed before GGUFMeta.VocabSize
// existed). It covers the current large vocabularies (Qwen's ~248K) and is
// a conservative overestimate that can only refuse early, never
// under-reserve.
const KLDefaultVocabSize = 262144

// KLSizeSafetyFactor pads the byte estimate for the file header (8-byte
// magic, n_ctx, n_vocab, n_chunk) and the n_chunk×n_ctx evaluation-token
// array, which the per-token term below does not count.
const KLSizeSafetyFactor = 1.15

// KLBaseSizeEstimate estimates the bytes a base file written at the given
// shape will occupy. From the tool's own math (writer at
// perplexity.cpp:461-470, :523-525, :619; per-token layout confirmed on
// the reader side at :1752): per chunk the tool stores
// n_ctx−1−n_ctx/2 evaluated tokens, each carrying 2×((n_vocab+1)/2)+4
// uint16 values, so
//
//	estimate = chunks × (n_ctx − 1 − n_ctx/2) × (2×((n_vocab+1)/2) + 4) × 2 bytes
//
// with the KLSizeSafetyFactor pad. Cross-checked against the perplexity
// README's own scale figures (≈11 GiB for a full LLaMA-2 wikitext base,
// ≈37 GiB for LLaMA-3 — vocab-size driven). An unknown vocab (<= 0) uses
// KLDefaultVocabSize.
func KLBaseSizeEstimate(chunks, ctx, vocab int) int64 {
	if chunks <= 0 || ctx <= 0 {
		return 0
	}
	if vocab <= 0 {
		vocab = KLDefaultVocabSize
	}
	tokensPerChunk := ctx - 1 - ctx/2
	if tokensPerChunk < 0 {
		tokensPerChunk = 0
	}
	u16PerToken := 2*((vocab+1)/2) + 4
	return int64(float64(chunks) * float64(tokensPerChunk) * float64(u16PerToken) * 2 * KLSizeSafetyFactor)
}

// freeSpaceAt reports the free bytes on the filesystem hosting path, or
// -1 when unknown. -1 means "unknown", never "zero": a failed statfs must
// not block generation (same convention as the model downloader's
// freeBytesAt in internal/huggingface). It is a variable so tests can
// fake free space without a full filesystem.
var freeSpaceAt = func(path string) int64 {
	usage, err := disk.Usage(path)
	if err != nil || usage == nil {
		return -1
	}
	return int64(usage.Free)
}

// CheckKLBaseSpace refuses a KL base generation when the filesystem
// hosting root cannot absorb the estimated file size plus the 2 GiB
// safety margin the model downloader also enforces
// (huggingface.DiskSafetyMarginBytes). Unknown free space is allowed
// through; the refusal error names the estimate.
func CheckKLBaseSpace(root string, estimate int64) error {
	free := freeSpaceAt(root)
	if free < 0 {
		return nil
	}
	needed := estimate + huggingface.DiskSafetyMarginBytes
	if free < needed {
		return fmt.Errorf("insufficient disk space for KL reference logits: estimated %s plus the %s download safety margin, but only %s free at %s",
			FormatBytes(estimate), FormatBytes(huggingface.DiskSafetyMarginBytes), FormatBytes(free), root)
	}
	return nil
}

// FormatBytes renders a byte count in binary units (B, KiB, MiB, GiB, …).
// Exported because the same figures appear in this package's disk-guard
// refusals, the job runner's progress lines, and the evaluation-data
// card: one formatter, so a size never reads two ways.
func FormatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return strconv.FormatInt(b, 10) + " B"
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}
