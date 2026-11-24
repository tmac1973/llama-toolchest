//go:build fieldtest

package videofile

import (
	"os"
	"strings"
	"testing"
)

// TestProbeField probes real clips named in VIDEOFILE_PROBE
// (comma-separated absolute paths; go test runs in the package folder) and logs what it finds. It is for field tests
// only:
//
//	VIDEOFILE_PROBE=$PWD/a.webm,$PWD/b.avi go test -tags fieldtest -run TestProbeField -v ./internal/videofile/
func TestProbeField(t *testing.T) {
	list := os.Getenv("VIDEOFILE_PROBE")
	if list == "" {
		t.Skip("set VIDEOFILE_PROBE to a comma-separated list of clips")
	}
	for _, path := range strings.Split(list, ",") {
		f, err := os.Open(path)
		if err != nil {
			t.Errorf("%s: %v", path, err)
			continue
		}
		st, err := f.Stat()
		if err != nil {
			f.Close()
			t.Errorf("%s: %v", path, err)
			continue
		}
		info, err := Probe(f, st.Size())
		f.Close()
		if err != nil {
			t.Errorf("%s: %v", path, err)
			continue
		}
		t.Logf("%s: %+v (%.3f fps, %.1f MiB decoded)", path, info, info.FPS(), float64(info.DecodedBytes())/(1<<20))
	}
}
