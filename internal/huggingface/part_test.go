package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
	"github.com/tmac1973/llama-toolchest/internal/filehash"
)

// partFixture is a downloader whose files come from handler, with a
// .part holding part and no final file.
func partFixture(t *testing.T, part string, handler http.HandlerFunc) (*Downloader, string, *filehash.Store) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	d := NewDownloader(dir, dir, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	store := filehash.Open(filepath.Join(dir, "hashes.json"))
	d.SetHashStore(store)
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), []byte(part), 0o644); err != nil {
		t.Fatal(err)
	}
	return d, dir, store
}

func fetchPart(d *Downloader, dir string, wantSize int64) (int64, error) {
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}
	return d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, wantSize, dl)
}

func checkInPlace(t *testing.T, dir, want string, store *filehash.Store) {
	t.Helper()
	final := filepath.Join(dir, "m.gguf")
	if got, err := os.ReadFile(final); err != nil || string(got) != want {
		t.Errorf("file = %q, %v; want %q", got, err, want)
	}
	if _, err := os.Stat(final + ".part"); !os.IsNotExist(err) {
		t.Error(".part file left behind")
	}
	if got, ok := store.Lookup(final); !ok || got != sha(want) {
		t.Errorf("recorded hash = %q, %v; want the file's hash", got, ok)
	}
}

// A .part of the expected size is put in place without a request.
func TestWholePartOfExpectedSizeIsPutInPlace(t *testing.T) {
	d, dir, store := partFixture(t, "whole file", func(w http.ResponseWriter, r *http.Request) {
		t.Error("a complete .part was requested again")
	})
	n, err := fetchPart(d, dir, int64(len("whole file")))
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(len("whole file")) {
		t.Errorf("size = %d, want %d", n, len("whole file"))
	}
	checkInPlace(t, dir, "whole file", store)
}

// A complete .part is still checked against the expected hash.
func TestWholePartWithWrongHashIsRemoved(t *testing.T) {
	d, dir, _ := partFixture(t, "whole file", func(w http.ResponseWriter, r *http.Request) {
		t.Error("a complete .part was requested again")
	})
	d.ExpectSHA256(filepath.Join(dir, "m.gguf"), sha("other file"))
	if _, err := fetchPart(d, dir, int64(len("whole file"))); err == nil {
		t.Fatal("a .part with the wrong hash was accepted")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); !os.IsNotExist(err) {
		t.Error("a .part with the wrong hash was put in place")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf.part")); !os.IsNotExist(err) {
		t.Error("the .part with the wrong hash was kept")
	}
}

// A .part larger than the expected size is downloaded again from the
// start.
func TestPartLargerThanExpectedStartsAgain(t *testing.T) {
	d, dir, store := partFixture(t, "far too many bytes", func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Range"); got != "" {
			t.Errorf("Range = %q, want none", got)
		}
		io.WriteString(w, "new")
	})
	if _, err := fetchPart(d, dir, int64(len("new"))); err != nil {
		t.Fatal(err)
	}
	checkInPlace(t, dir, "new", store)
}

// With no expected size, a 416 whose total is the size of the .part
// means the .part is the whole file.
func TestRangeNotSatisfiableAtPartSizeFinishesPart(t *testing.T) {
	const whole = "whole file"
	d, dir, store := partFixture(t, whole, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", len(whole)))
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	})
	if _, err := fetchPart(d, dir, 0); err != nil {
		t.Fatal(err)
	}
	checkInPlace(t, dir, whole, store)
}

// When the 416 gives no size, as Hugging Face's CDN does for large
// files, a HEAD request gives it.
func TestRangeNotSatisfiableAskedByHead(t *testing.T) {
	const whole = "whole file"
	d, dir, store := partFixture(t, whole, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprint(len(whole)))
			return
		}
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	})
	if _, err := fetchPart(d, dir, 0); err != nil {
		t.Fatal(err)
	}
	checkInPlace(t, dir, whole, store)
}

// Any other 416 is still an error, and the .part is not put in place.
func TestRangeNotSatisfiableOtherwiseFails(t *testing.T) {
	for name, c := range map[string]struct {
		header     string
		headStatus int
	}{
		"other total":      {"bytes */99", http.StatusOK},
		"HEAD other size":  {"", http.StatusOK},
		"HEAD not allowed": {"", http.StatusUnauthorized},
	} {
		t.Run(name, func(t *testing.T) {
			d, dir, _ := partFixture(t, "whole file", func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodHead {
					// A 200 gives a size other than the .part's. An error
					// answer gives the .part's size, which must not count.
					if c.headStatus == http.StatusOK {
						w.Header().Set("Content-Length", "99")
					} else {
						w.Header().Set("Content-Length", fmt.Sprint(len("whole file")))
					}
					w.WriteHeader(c.headStatus)
					return
				}
				if c.header != "" {
					w.Header().Set("Content-Range", c.header)
				}
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			})
			if _, err := fetchPart(d, dir, 0); err == nil {
				t.Fatal("a 416 that does not match the .part was accepted")
			}
			if _, err := os.Stat(filepath.Join(dir, "m.gguf")); !os.IsNotExist(err) {
				t.Error("the .part was put in place")
			}
		})
	}
}
