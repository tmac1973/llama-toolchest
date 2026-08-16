// Pinned evaluation datasets and the eval-data directory layout.
//
// Layout under the data dir (the root is <dataDir>/eval-data/ and is passed
// in by callers as a plain string — this package never derives it from a
// config):
//
//	eval-data/
//	├── datasets/wikitext-2-raw-test.txt
//	├── datasets/hellaswag_val_full.txt
//	├── datasets/winogrande-debiased-eval.csv
//	└── logits/<SafeModelID>~<quant>~<dataset>~c<chunks>~ctx<n>~f<fingerprint>.kld
//
// (The logits filename scheme is owned by klcache.go; it lives here only so
// the whole tree is visible in one place.)

package evaluate

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// EvalDataDirName is the top-level eval-data directory under the data dir.
const EvalDataDirName = "eval-data"

// EvalDataRoot returns the eval data root under dataDir.
func EvalDataRoot(dataDir string) string {
	return filepath.Join(dataDir, EvalDataDirName)
}

// DatasetsDir returns the directory holding the pinned dataset files.
func DatasetsDir(root string) string {
	return filepath.Join(root, "datasets")
}

// LogitsDir returns the directory holding KL reference logits files.
func LogitsDir(root string) string {
	return filepath.Join(root, "logits")
}

// Dataset is one pinned evaluation dataset: where it comes from, how it is
// verified, and under which license it ships. The pinned table is the
// SINGLE source for these facts — the UI's evaluation-data card and the
// help text render the license strings from it, nowhere else.
type Dataset struct {
	// Name is the canonical dataset name, identical to Mode.DatasetName()
	// (the engine and the table share the names).
	Name string
	// File is the file name under datasets/.
	File string
	// URL is the download URL of the artifact. Plain net/http: these are
	// static files, not HF-API objects — the HF client's auth/resume
	// machinery is not needed.
	URL string
	// SHA256 is the pinned hex SHA-256 of the download ARTIFACT, verified
	// at download time — the zip for an archive entry, the file itself
	// otherwise.
	SHA256 string
	// StoredSHA256 is the pinned hex SHA-256 of the file that ends up on
	// disk. It differs from SHA256 only for archive entries, where the
	// artifact is a zip and the stored file is the extracted member, and
	// is what Verify re-hashes. Leaving it empty for a plain file means
	// the two hashes are the same fact, so only archives carry both.
	StoredSHA256 string
	// License is the dataset's license, shown to the user.
	License string
	// ApproxSize is the approximate size in bytes of the stored file.
	ApproxSize int64
	// ExtractFile is the member to extract when the artifact is an
	// archive ("wikitext-2-raw/wiki.test.raw" for the wikitext-2 zip);
	// empty for plain files. The stored file is the extraction, not the
	// archive.
	ExtractFile string
}

// pinnedDatasets is the pinned dataset table, one entry per dataset, in a
// fixed order (the order the UI lists them). The SHA-256s were pinned at
// implementation time by downloading each artifact once and hashing it:
//
//   - wikitext-2 (CC BY-SA 3.0): the raw test file llama.cpp's own
//     perplexity documentation uses — the wiki.test.raw member of
//     ggml-org/ci's wikitext-2-raw-v1.zip, the source of llama.cpp's
//     scripts/get-wikitext-2.sh.
//   - hellaswag (MIT): the preprocessed validation file the --hellaswag
//     mode consumes (klosax/hellaswag_text_data, the source of llama.cpp's
//     scripts/get-hellaswag.sh; raw.githubusercontent can 404 transiently,
//     hence the download retries below).
//   - winogrande (Apache-2.0): the debiased eval CSV the format
//     load_winogrande_from_csv parses (ikawrakow/winogrande-eval-for-
//     llama.cpp, the dataset named in perplexity.cpp's comment).
var pinnedDatasets = []Dataset{
	{
		Name: "wikitext-2",
		File: "wikitext-2-raw-test.txt",
		URL:  "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
		// The zip, checked on download...
		SHA256: "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
		// ...and the extracted wiki.test.raw, which is what Verify sees.
		StoredSHA256: "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
		License:      "CC BY-SA 3.0",
		ApproxSize:   1_290_590, // wiki.test.raw after extraction
		ExtractFile:  "wikitext-2-raw/wiki.test.raw",
	},
	{
		Name:       "hellaswag",
		File:       "hellaswag_val_full.txt",
		URL:        "https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt",
		SHA256:     "d572539320eb2050e858ca34b495bbe2103e3b3f1391a9c3bcdf215b0bb93bd1",
		License:    "MIT",
		ApproxSize: 7_770_677,
	},
	{
		Name:       "winogrande",
		File:       "winogrande-debiased-eval.csv",
		URL:        "https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/main/winogrande-debiased-eval.csv",
		SHA256:     "6726173ef65ffdc4abb0d28b755375d288ffb3eb41633bd833288c411c8ebf7c",
		License:    "Apache-2.0",
		ApproxSize: 155_325,
	},
}

// Datasets returns the pinned table in display order.
func Datasets() []Dataset {
	out := make([]Dataset, len(pinnedDatasets))
	copy(out, pinnedDatasets)
	return out
}

// LookupDataset finds a pinned dataset by canonical name.
func LookupDataset(name string) (Dataset, bool) {
	for _, ds := range pinnedDatasets {
		if ds.Name == name {
			return ds, true
		}
	}
	return Dataset{}, false
}

// DatasetPath returns the on-disk path of a pinned dataset under root.
func DatasetPath(root, name string) string {
	ds, ok := LookupDataset(name)
	if !ok {
		return ""
	}
	return filepath.Join(root, "datasets", ds.File)
}

// EnsureDataset returns the path of the named dataset under root,
// downloading it on first use. A present file is returned as-is — it was
// hash-verified when it was downloaded, and re-verifying on every call
// would re-read multi-megabyte files for nothing (Verify re-hashes for the
// UI's state display instead). Absent files are downloaded to a temp file
// in the same directory, SHA-256 verified, then renamed into place, so a
// crash or cancel never leaves a partial dataset behind.
func EnsureDataset(ctx context.Context, root, name string) (string, error) {
	ds, ok := LookupDataset(name)
	if !ok {
		return "", fmt.Errorf("unknown evaluation dataset %q", name)
	}
	final := DatasetPath(root, name)
	if _, err := os.Stat(final); err == nil {
		return final, nil
	}
	if err := os.MkdirAll(DatasetsDir(root), 0o755); err != nil {
		return "", err
	}

	artifact, err := downloadArtifact(ctx, DatasetsDir(root), ds)
	if err != nil {
		return "", err
	}
	defer os.Remove(artifact) // no-op after a successful rename

	if ds.ExtractFile == "" {
		if err := os.Rename(artifact, final); err != nil {
			return "", fmt.Errorf("installing dataset %s: %w", ds.Name, err)
		}
		return final, nil
	}

	// Archive artifact: extract the single pinned member, then rename it
	// into place (the zip itself never stays on disk).
	extracted, err := extractMember(artifact, ds.ExtractFile, DatasetsDir(root), ds.Name)
	if err != nil {
		return "", err
	}
	defer os.Remove(extracted)
	if err := os.Rename(extracted, final); err != nil {
		return "", fmt.Errorf("installing dataset %s: %w", ds.Name, err)
	}
	return final, nil
}

// downloadAttempts and downloadRetryBackoff bound the retry loop for
// transient transfer failures (network errors, non-200 statuses — raw.
// githubusercontent.com is known to 404 intermittently). A hash mismatch
// is definitive and is NOT retried.
const (
	downloadAttempts     = 3
	downloadRetryBackoff = 500 * time.Millisecond
)

// maxDatasetBytes caps a single download: the largest pinned artifact is
// under 5 MiB, so a gigabyte of bytes is already an absurd upper bound
// that only guards against a broken Content-Length.
const maxDatasetBytes = 1 << 30

// transientDownloadError marks transfer failures that are worth retrying.
type transientDownloadError struct{ err error }

func (e *transientDownloadError) Error() string { return e.err.Error() }
func (e *transientDownloadError) Unwrap() error { return e.err }

func downloadArtifact(ctx context.Context, dir string, ds Dataset) (string, error) {
	var lastErr error
	for attempt := 1; attempt <= downloadAttempts; attempt++ {
		if attempt > 1 {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(downloadRetryBackoff * time.Duration(attempt-1)):
			}
		}
		path, err := fetchOnce(ctx, dir, ds)
		if err == nil {
			return path, nil
		}
		var transient *transientDownloadError
		if !errors.As(err, &transient) {
			return "", err
		}
		lastErr = err
	}
	return "", lastErr
}

// fetchOnce downloads the artifact to a fresh temp file in dir and
// verifies its SHA-256. On any failure the temp file is removed and the
// error returned: a transfer failure is wrapped as transient (retry), a
// hash mismatch is returned verbatim with expected vs got named.
func fetchOnce(ctx context.Context, dir string, ds Dataset) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ds.URL, nil)
	if err != nil {
		return "", err
	}
	// No client timeout: these are static files and the context governs
	// the lifetime (a slow link is not a failure).
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", &transientDownloadError{err: fmt.Errorf("downloading %s: %w", ds.Name, err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", &transientDownloadError{err: fmt.Errorf("downloading %s: HTTP %d", ds.Name, resp.StatusCode)}
	}

	tmp, err := os.CreateTemp(dir, ds.File+".*.tmp")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()

	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, maxDatasetBytes))
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmpName)
		return "", &transientDownloadError{err: fmt.Errorf("downloading %s: %w", ds.Name, err)}
	}

	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, ds.SHA256) {
		os.Remove(tmpName)
		return "", fmt.Errorf("dataset %s: SHA-256 mismatch: expected %s, got %s", ds.Name, ds.SHA256, got)
	}
	return tmpName, nil
}

// extractMember pulls member out of the zip at zipPath into a fresh temp
// file in dir and returns its path.
func extractMember(zipPath, member, dir, name string) (string, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return "", fmt.Errorf("extracting dataset %s: %w", name, err)
	}
	defer zr.Close()

	var entry *zip.File
	for _, f := range zr.File {
		if f.Name == member {
			entry = f
			break
		}
	}
	if entry == nil {
		return "", fmt.Errorf("dataset %s: artifact is missing member %q", name, member)
	}

	rc, err := entry.Open()
	if err != nil {
		return "", fmt.Errorf("extracting dataset %s: %w", name, err)
	}
	defer rc.Close()

	tmp, err := os.CreateTemp(dir, name+".*.tmp")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	_, err = io.Copy(tmp, rc)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("extracting dataset %s: %w", name, err)
	}
	return tmpName, nil
}

// DatasetStatus is one dataset's on-disk state, as displayed by the UI.
type DatasetStatus struct {
	Dataset
	Path     string `json:"path"`
	Present  bool   `json:"present"`
	Size     int64  `json:"size,omitempty"`
	Verified bool   `json:"verified"`
}

// storedHash returns the pinned hash of the file that lands on disk:
// the extracted member's for an archive entry, the artifact's otherwise.
// Verifying an extracted file against the ZIP's hash can only ever
// fail, which is how a correctly downloaded wikitext-2 came to be
// reported as corrupt.
func (d Dataset) storedHash() string {
	if d.StoredSHA256 != "" {
		return d.StoredSHA256
	}
	return d.SHA256
}

// Verify re-hashes every pinned dataset present under root against its
// stored-file SHA-256 and returns one status per dataset in table order.
// It is the UI's state display — the deliberate place where hashing
// happens outside the download path (EnsureDataset itself does not
// re-verify).
//
// The result is cached per (path, size, mtime): the card re-renders on
// every job action, and re-reading ~9 MB of dataset to answer a question
// whose inputs have not changed is waste. A file rewritten in place
// changes its mtime, so the cache cannot serve a stale verdict.
func Verify(root string) []DatasetStatus {
	out := make([]DatasetStatus, 0, len(pinnedDatasets))
	for _, ds := range pinnedDatasets {
		st := DatasetStatus{Dataset: ds, Path: DatasetPath(root, ds.Name)}
		info, err := os.Stat(st.Path)
		if err != nil {
			out = append(out, st)
			continue
		}
		st.Present = true
		st.Size = info.Size()
		st.Verified = verifiedCached(st.Path, info, ds.storedHash())
		out = append(out, st)
	}
	return out
}

// verifyCacheEntry is one memoized hash verdict, keyed by the file
// identity Verify observed when it computed it.
type verifyCacheEntry struct {
	size     int64
	modTime  time.Time
	hash     string // the pinned hash the verdict was computed against
	verified bool
}

var (
	verifyCacheMu sync.Mutex
	verifyCache   = map[string]verifyCacheEntry{}
)

// verifiedCached reports whether the file at path matches want, reusing
// a previous verdict when size, mtime, and the pinned hash are all
// unchanged.
func verifiedCached(path string, info os.FileInfo, want string) bool {
	key := path
	verifyCacheMu.Lock()
	e, ok := verifyCache[key]
	verifyCacheMu.Unlock()
	if ok && e.size == info.Size() && e.modTime.Equal(info.ModTime()) && e.hash == want {
		return e.verified
	}

	sum, err := fileSHA256(path)
	if err != nil {
		// Unreadable right now: report unverified without caching, so a
		// transient error does not stick.
		return false
	}
	verified := strings.EqualFold(sum, want)

	verifyCacheMu.Lock()
	verifyCache[key] = verifyCacheEntry{size: info.Size(), modTime: info.ModTime(), hash: want, verified: verified}
	verifyCacheMu.Unlock()
	return verified
}

// fileSHA256 hashes the whole file at path.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
