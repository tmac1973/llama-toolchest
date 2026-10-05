package modelsource

import (
	"net/http"
	"testing"
)

// A resumed download is told apart from a restarted one by this helper,
// because ModelScope labels a partial body 200. Getting it wrong appends
// a tail to a partial file and silently corrupts the result.
func TestResponseIsPartial(t *testing.T) {
	mk := func(status int, contentRange string) *http.Response {
		h := http.Header{}
		if contentRange != "" {
			h.Set("Content-Range", contentRange)
		}
		return &http.Response{StatusCode: status, Header: h}
	}
	tests := []struct {
		name string
		resp *http.Response
		want bool
	}{
		{"conforming 206", mk(206, "bytes 100-1123/2275379008"), true},
		{"ModelScope's 200 with Content-Range", mk(200, "bytes 100-1123/2275379008"), true},
		{"plain 200, whole file", mk(200, ""), false},
		{"206 without the header", mk(206, ""), true},
	}
	for _, tt := range tests {
		if got := ResponseIsPartial(tt.resp); got != tt.want {
			t.Errorf("%s: ResponseIsPartial = %v, want %v", tt.name, got, tt.want)
		}
	}
}
