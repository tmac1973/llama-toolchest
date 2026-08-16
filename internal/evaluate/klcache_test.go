package evaluate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

// ---------- key → filename → key ----------

func TestKLBaseFilenameExact(t *testing.T) {
	k := KLBaseKey{ModelID: "unsloth/Qwen3.5-4B-GGUF", Quant: "Q4_K_M", Dataset: "wikitext-2", Chunks: 100, Ctx: 512, Fingerprint: "abc123def456"}
	want := "unsloth--Qwen3.5-4B-GGUF~Q4_K_M~wikitext-2~c100~ctx512~fabc123def456.kld"
	if got := k.Filename(); got != want {
		t.Errorf("Filename() = %q, want %q", got, want)
	}
	if got := KLBasePath("/data/eval-data", k); got != filepath.Join("/data/eval-data", "logits", want) {
		t.Errorf("KLBasePath = %q", got)
	}
	if got := KLBasePartialPath("/data/eval-data", k); got != filepath.Join("/data/eval-data", "logits", want+".partial") {
		t.Errorf("KLBasePartialPath = %q", got)
	}
}

func TestKLBaseFilenameRoundTrip(t *testing.T) {
	keys := []KLBaseKey{
		{ModelID: "unsloth/Qwen3.5-4B-GGUF", Quant: "Q4_K_M", Dataset: "wikitext-2", Chunks: 100, Ctx: 512, Fingerprint: "abc123def456"},
		{ModelID: "org/repo", Quant: "Q8_0", Dataset: "hellaswag", Chunks: 0, Ctx: 512, Fingerprint: "0011223344ff"}, // full run
		{ModelID: "a/b/c/d", Quant: "Q4_K_XL", Dataset: "winogrande", Chunks: 4096, Ctx: 1024, Fingerprint: "deadbeefcafe"},
		{ModelID: "org/repo", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 100, Ctx: 512}, // no fingerprint -> "fnone"
	}
	for _, k := range keys {
		fn := k.Filename()
		back, err := ParseKLBaseFilename(fn)
		if err != nil {
			t.Fatalf("parse %q: %v", fn, err)
		}
		if back.Quant != k.Quant || back.Dataset != k.Dataset || back.Chunks != k.Chunks || back.Ctx != k.Ctx || back.Fingerprint != k.Fingerprint {
			t.Errorf("round trip mismatch for %q: got %+v, want quant=%s dataset=%s chunks=%d ctx=%d",
				fn, back, k.Quant, k.Dataset, k.Chunks, k.Ctx)
		}
		if back.ModelID != huggingface.SafeModelID(k.ModelID) {
			t.Errorf("round trip ModelID = %q, want %q (SafeModelID form)", back.ModelID, huggingface.SafeModelID(k.ModelID))
		}
		if back.Filename() != fn {
			t.Errorf("re-render %q != %q", back.Filename(), fn)
		}
	}
}

func TestParseKLBaseFilenameErrors(t *testing.T) {
	bad := []string{
		"no-suffix-at-all",
		"a--b~Q4_K_M~wikitext-2~c100.kld",           // 4 fields
		"a~b~c~d~e~f~g.kld",                         // 7 fields
		"~Q4_K_M~wikitext-2~c100~ctx512.kld",        // empty model
		"a--b~~wikitext-2~c100~ctx512.kld",          // empty quant
		"a--b~Q4_K_M~wikitext-2~x100~ctx512.kld",    // chunks prefix
		"a--b~Q4_K_M~wikitext-2~c100~x512.kld",      // ctx prefix
		"a--b~Q4_K_M~wikitext-2~c~ctx512.kld",       // non-numeric chunks
		"a--b~Q4_K_M~wikitext-2~c100~ctx.kld",       // non-numeric ctx
		"a--b~Q4_K_M~wikitext-2~c-5~ctx512.kld",     // negative chunks
		"a--b~Q4_K_M~wikitext-2~c100~ctx0.kld",      // zero ctx
		"a--b~Q4_K_M~wikitext-2~c100~ctx512~x1.kld", // fingerprint prefix
		"a--b~Q4_K_M~wikitext-2~c100~ctx512~f.kld",  // empty fingerprint
	}
	for _, name := range bad {
		if _, err := ParseKLBaseFilename(name); err == nil {
			t.Errorf("ParseKLBaseFilename(%q) = nil error, want failure", name)
		}
	}
}

// ---------- list / has / delete ----------

func writeBaseFile(t *testing.T, root string, key KLBaseKey, size int) string {
	t.Helper()
	path := KLBasePath(root, key)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestKLBaseListHasDelete(t *testing.T) {
	root := t.TempDir()
	k1 := KLBaseKey{ModelID: "org/a", Quant: "Q4_K_M", Dataset: "wikitext-2", Chunks: 100, Ctx: 512}
	k2 := KLBaseKey{ModelID: "org/b", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 0, Ctx: 512}
	p1 := writeBaseFile(t, root, k1, 1000)
	p2 := writeBaseFile(t, root, k2, 2000)

	// A .partial corpse and an unrelated file must be invisible to the
	// cache — presence-keyed listing would otherwise serve the corpse.
	logits := LogitsDir(root)
	if err := os.WriteFile(KLBasePartialPath(root, k1), []byte("truncated header"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(logits, "notes.txt"), []byte("junk"), 0o644); err != nil {
		t.Fatal(err)
	}

	if !HasKLBase(root, k1) || !HasKLBase(root, k2) {
		t.Error("HasKLBase = false for existing bases")
	}
	kMissing := KLBaseKey{ModelID: "org/none", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 0, Ctx: 512}
	if HasKLBase(root, kMissing) {
		t.Error("HasKLBase = true for a missing base")
	}

	list := ListKLBases(root)
	if len(list) != 2 {
		names := make([]string, 0, len(list))
		for _, info := range list {
			names = append(names, filepath.Base(info.Path))
		}
		t.Fatalf("ListKLBases = %d entries %v, want 2 (partial and junk ignored)", len(list), names)
	}
	byModel := map[string]KLBaseInfo{}
	for _, info := range list {
		byModel[info.Key.ModelID] = info
		if info.Size <= 0 || info.ModTime.IsZero() {
			t.Errorf("entry %s: size/mtime not populated: %+v", info.Path, info)
		}
	}
	if got := byModel["org--a"]; got.Size != 1000 || got.Key.Chunks != 100 || got.Key.Quant != "Q4_K_M" || got.Key.Ctx != 512 {
		t.Errorf("parsed entry for org/a wrong: %+v", got)
	}
	if got := byModel["org--b"]; got.Size != 2000 || got.Key.Chunks != 0 || got.Key.Quant != "Q8_0" {
		t.Errorf("parsed entry for org/b wrong: %+v", got)
	}
	// Sorted by filename.
	if filepath.Base(list[0].Path) > filepath.Base(list[1].Path) {
		t.Errorf("ListKLBases not sorted: %q before %q", list[0].Path, list[1].Path)
	}

	// Delete: one goes, the other stays; deleting a missing entry is
	// not an error (idempotent for the UI's delete button).
	if err := DeleteKLBase(root, k1); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p1); !os.IsNotExist(err) {
		t.Errorf("k1 still present after delete")
	}
	if err := DeleteKLBase(root, k1); err != nil {
		t.Errorf("second delete = %v, want nil", err)
	}
	if !HasKLBase(root, k2) {
		t.Error("k2 lost by deleting k1")
	}
	if _, err := os.Stat(p2); err != nil {
		t.Errorf("k2 file stat after deleting k1: %v", err)
	}
	if err := DeleteKLBase(root, kMissing); err != nil {
		t.Errorf("delete missing = %v, want nil", err)
	}
	if list := ListKLBases(root); len(list) != 1 || list[0].Key.ModelID != "org--b" {
		t.Errorf("list after delete = %+v, want only org/b", list)
	}
}

func TestListKLBasesMissingDir(t *testing.T) {
	if list := ListKLBases(t.TempDir()); list != nil {
		t.Errorf("ListKLBases on missing dir = %v, want nil", list)
	}
}

// ---------- .partial discipline ----------

func TestCleanStalePartials(t *testing.T) {
	root := t.TempDir()
	k1 := KLBaseKey{ModelID: "org/a", Quant: "Q4_K_M", Dataset: "wikitext-2", Chunks: 100, Ctx: 512}
	k2 := KLBaseKey{ModelID: "org/b", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 0, Ctx: 512}

	// A finished base plus two stale partials (a crash skipped the
	// delete) plus an unrelated .partial-named non-kld file.
	writeBaseFile(t, root, k1, 10)
	logits := LogitsDir(root)
	for _, p := range []string{
		KLBasePartialPath(root, k1),
		KLBasePartialPath(root, k2),
	} {
		if err := os.WriteFile(p, []byte("corpse"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(logits, "other.partial"), []byte("not ours"), 0o644); err != nil {
		t.Fatal(err)
	}

	removed, err := CleanStalePartials(root)
	if err != nil {
		t.Fatal(err)
	}
	if removed != 3 {
		t.Errorf("removed = %d, want 3", removed)
	}
	for _, p := range []string{
		KLBasePartialPath(root, k1),
		KLBasePartialPath(root, k2),
		filepath.Join(logits, "other.partial"),
	} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s still present after CleanStalePartials", p)
		}
	}
	// The finished base is untouched.
	if !HasKLBase(root, k1) {
		t.Error("finished base removed by CleanStalePartials")
	}

	// Idempotent: nothing left to clean.
	removed, err = CleanStalePartials(root)
	if err != nil || removed != 0 {
		t.Errorf("second clean = (%d, %v), want (0, nil)", removed, err)
	}
}

func TestCleanStalePartialsMissingDir(t *testing.T) {
	removed, err := CleanStalePartials(t.TempDir())
	if err != nil || removed != 0 {
		t.Errorf("clean on missing dir = (%d, %v), want (0, nil)", removed, err)
	}
}

// ---------- disk guard ----------

func fakeFree(t *testing.T, free int64) {
	t.Helper()
	orig := freeSpaceAt
	freeSpaceAt = func(string) int64 { return free }
	t.Cleanup(func() { freeSpaceAt = orig })
}

func TestCheckKLBaseSpaceRefusal(t *testing.T) {
	root := t.TempDir()
	const estimate = 5 << 30 // 5 GiB

	// 1 GiB free < 5 GiB + 2 GiB margin → refuse, naming the estimate.
	fakeFree(t, 1<<30)
	err := CheckKLBaseSpace(root, estimate)
	if err == nil {
		t.Fatal("want refusal with insufficient space")
	}
	if !strings.Contains(err.Error(), "5.0 GiB") {
		t.Errorf("refusal must name the estimate: %v", err)
	}
	if !strings.Contains(err.Error(), "2.0 GiB") {
		t.Errorf("refusal must name the safety margin: %v", err)
	}

	// Just enough: estimate + margin → allow.
	fakeFree(t, estimate+huggingface.DiskSafetyMarginBytes)
	if err := CheckKLBaseSpace(root, estimate); err != nil {
		t.Errorf("estimate+margin must be allowed: %v", err)
	}

	// One byte short → refuse.
	fakeFree(t, estimate+huggingface.DiskSafetyMarginBytes-1)
	if err := CheckKLBaseSpace(root, estimate); err == nil {
		t.Error("one byte short of estimate+margin must refuse")
	}

	// Plenty of space → allow.
	fakeFree(t, 100<<30)
	if err := CheckKLBaseSpace(root, estimate); err != nil {
		t.Errorf("plenty of space must be allowed: %v", err)
	}

	// Unknown free space (-1) → allow through (a failed statfs must not
	// block generation).
	fakeFree(t, -1)
	if err := CheckKLBaseSpace(root, estimate); err != nil {
		t.Errorf("unknown free space must be allowed: %v", err)
	}
}

// ---------- size estimate ----------

func TestKLBaseSizeEstimateFormula(t *testing.T) {
	// ctx=512: tokensPerChunk = 512-1-256 = 255.
	// vocab=32000 (LLaMA-2 class): u16PerToken = 2*((32000+1)/2)+4 = 32004.
	// 10 chunks: 10 × 255 × 32004 × 2 = 163,220,400; × 1.15 = 187,703,460.
	if got := KLBaseSizeEstimate(10, 512, 32000); got != 187_703_460 {
		t.Errorf("estimate(10, 512, 32000) = %d, want 187703460", got)
	}
	// Odd vocab: (32001+1)/2 = 16001 → u16 = 32006.
	wantOdd := int64(float64(255*32006*2) * 1.15)
	if got := KLBaseSizeEstimate(1, 512, 32001); got != wantOdd {
		t.Errorf("estimate(1, 512, 32001) = %d, want %d", got, wantOdd)
	}
	// ctx=4096: tokensPerChunk = 4096-1-2048 = 2047. (Via a variable:
	// the product ×1.15 is not a whole number, so a constant conversion
	// would not compile — the implementation truncates at runtime.)
	f4k := float64(2047) * float64(32004) * 2 * 1.15
	if got := KLBaseSizeEstimate(1, 4096, 32000); got != int64(f4k) {
		t.Errorf("estimate(1, 4096, 32000) = %d, want %d", got, int64(f4k))
	}
}

func TestKLBaseSizeEstimateFallbackVocab(t *testing.T) {
	// Unknown vocab (0) falls back to the documented worst case 262144:
	// u16PerToken = 2*((262144+1)/2)+4 = 262148; 1 chunk at 512:
	// 255 × 262148 × 2 = 133,695,480; × 1.15 = 153,749,802.
	if got := KLBaseSizeEstimate(1, 512, 0); got != 153_749_802 {
		t.Errorf("estimate(1, 512, 0) = %d, want 153749802", got)
	}
	// Negative vocab behaves like unknown.
	if got := KLBaseSizeEstimate(1, 512, -7); got != 153_749_802 {
		t.Errorf("estimate(1, 512, -7) = %d, want 153749802", got)
	}
	// The fallback is an OVERestimate for every realistic real vocab, so
	// it can only refuse early, never under-reserve.
	if KLBaseSizeEstimate(1, 512, 0) < KLBaseSizeEstimate(1, 512, 248000) {
		t.Error("fallback estimate must dominate real-vocab estimates")
	}
}

func TestKLBaseSizeEstimateScaleAndEdges(t *testing.T) {
	// Vocab-driven: LLaMA-3-class (128256) bases are ~4× LLaMA-2-class
	// (32000) at the same shape — the README's 11 GiB → 37 GiB direction.
	small := KLBaseSizeEstimate(100, 512, 32000)
	big := KLBaseSizeEstimate(100, 512, 128256)
	if big <= small*3 {
		t.Errorf("vocab scaling too weak: %d vs %d", big, small)
	}
	// Monotonic in chunks.
	if KLBaseSizeEstimate(200, 512, 32000) <= small {
		t.Error("estimate must grow with chunks")
	}
	// Degenerate inputs estimate zero (nothing to refuse against).
	if KLBaseSizeEstimate(0, 512, 32000) != 0 || KLBaseSizeEstimate(10, 0, 32000) != 0 {
		t.Error("zero chunks/ctx must estimate 0")
	}
}

// ---------- formatBytes ----------

func TestFormatBytes(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KiB"},
		{1 << 30, "1.0 GiB"},
		{2 * (1 << 30), "2.0 GiB"},
		{1 << 40, "1.0 TiB"},
	}
	for _, c := range cases {
		if got := FormatBytes(c.in); got != c.want {
			t.Errorf("FormatBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ---------- fingerprint ----------

// The fingerprint covers the flags that change the LOGITS, so two
// generations that would produce different reference probabilities
// cannot share one cache entry.
func TestKLFlagFingerprintDistinguishesNumerics(t *testing.T) {
	f16 := []string{"--n-gpu-layers", "999", "--threads", "8", "--flash-attn", "on"}
	q8 := []string{"--n-gpu-layers", "999", "--threads", "8", "--flash-attn", "on",
		"--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	if KLFlagFingerprint(f16) == KLFlagFingerprint(q8) {
		t.Error("an f16 and a q8_0 KV cache share a fingerprint — a config change would silently reuse stale logits")
	}
	noFA := []string{"--flash-attn", "off"}
	withFA := []string{"--flash-attn", "on"}
	if KLFlagFingerprint(noFA) == KLFlagFingerprint(withFA) {
		t.Error("flash-attn on and off share a fingerprint")
	}
}

// Performance-only settings do not change the reference, and a base
// file is tens of GiB — retuning them must not discard the cache.
func TestKLFlagFingerprintIgnoresPerformanceFlags(t *testing.T) {
	a := []string{"--n-gpu-layers", "999", "--threads", "8", "--flash-attn", "on", "--device", "ROCm0"}
	b := []string{"--n-gpu-layers", "40", "--threads", "16", "--flash-attn", "on", "--direct-io"}
	if KLFlagFingerprint(a) != KLFlagFingerprint(b) {
		t.Error("gpu layers / threads / placement changed the fingerprint; they select kernels, not arithmetic")
	}
}

// Flag ORDER is an assembly detail, not a setting: two callers building
// the same evaluation differently must hit the same cache entry.
func TestKLFlagFingerprintOrderIndependent(t *testing.T) {
	a := []string{"--cache-type-k", "q8_0", "--flash-attn", "on", "--batch-size", "512"}
	b := []string{"--batch-size", "512", "--flash-attn", "on", "--cache-type-k", "q8_0"}
	if KLFlagFingerprint(a) != KLFlagFingerprint(b) {
		t.Error("fingerprint depends on flag order")
	}
}

// A pre-fingerprint filename still lists and deletes (so it is not an
// invisible tens-of-GiB orphan) but can never be SERVED: the current
// key renders "~fnone", a different name from the legacy five-field
// one.
func TestLegacyFilenameListsButNeverServes(t *testing.T) {
	root := t.TempDir()
	dir := LogitsDir(root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	legacy := "u--M~Q8_0~wikitext-2~c100~ctx512.kld"
	if err := os.WriteFile(filepath.Join(dir, legacy), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	list := ListKLBases(root)
	if len(list) != 1 || filepath.Base(list[0].Path) != legacy {
		t.Fatalf("legacy entry not listed: %+v", list)
	}
	key := KLBaseKey{ModelID: "u/M", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 100, Ctx: 512,
		Fingerprint: KLFlagFingerprint([]string{"--flash-attn", "on"})}
	if HasKLBase(root, key) {
		t.Error("a legacy file was served for a fingerprinted key")
	}
	if err := DeleteKLBaseFile(root, legacy); err != nil {
		t.Errorf("legacy entry cannot be deleted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, legacy)); !os.IsNotExist(err) {
		t.Error("legacy entry still on disk after delete")
	}
}

// ---------- delete by name ----------

// The delete name arrives from an HTTP form, so it is validated as a
// name: no directory part, must parse as a KL base filename, must
// resolve inside the logits directory.
func TestDeleteKLBaseFileRejectsPaths(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(LogitsDir(root), 0o755); err != nil {
		t.Fatal(err)
	}
	// A file one level up that a traversal would reach.
	outside := filepath.Join(root, "victim~Q~wikitext-2~c1~ctx1~fnone.kld")
	if err := os.WriteFile(outside, []byte("keep me"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{
		"../victim~Q~wikitext-2~c1~ctx1~fnone.kld",
		"/etc/passwd",
		"sub/dir~Q~wikitext-2~c1~ctx1~fnone.kld",
		"not-a-kl-base.txt",
		"",
	} {
		if err := DeleteKLBaseFile(root, name); err == nil {
			t.Errorf("DeleteKLBaseFile(%q) = nil error, want rejection", name)
		}
	}
	if _, err := os.Stat(outside); err != nil {
		t.Errorf("a file outside the logits dir was removed: %v", err)
	}
}

// Deleting an entry that is not there is not an error: the card's
// button must survive a double click.
func TestDeleteKLBaseFileIdempotent(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(LogitsDir(root), 0o755); err != nil {
		t.Fatal(err)
	}
	name := "u--M~Q8_0~wikitext-2~c100~ctx512~fabc123def456.kld"
	if err := os.WriteFile(filepath.Join(LogitsDir(root), name), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := DeleteKLBaseFile(root, name); err != nil {
			t.Fatalf("delete %d: %v", i, err)
		}
	}
}
