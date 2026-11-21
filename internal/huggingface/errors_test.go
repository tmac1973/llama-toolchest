package huggingface

import (
	"net/http"
	"strings"
	"testing"
)

func TestStatusError(t *testing.T) {
	resp := func(status int, code, msg string) *http.Response {
		h := http.Header{}
		if code != "" {
			h.Set("X-Error-Code", code)
		}
		if msg != "" {
			h.Set("X-Error-Message", msg)
		}
		return &http.Response{StatusCode: status, Header: h}
	}
	cases := []struct {
		name     string
		resp     *http.Response
		hasToken bool
		want     []string
		notWant  []string
	}{
		{"gated, no token", resp(401, "GatedRepo", "Access is restricted. Please log in."), false,
			[]string{"gated model", "agree to its access terms", "save a HuggingFace token", "(HTTP 401)"}, []string{"refused the saved token"}},
		{"gated, token refused", resp(403, "GatedRepo", "Please enable access to public gated repositories in your fine-grained token settings."), true,
			[]string{"refused the saved token", "Read access to contents of all public gated repos", "Hugging Face said: Please enable", "(HTTP 403)"}, nil},
		{"403 without a code", resp(403, "", ""), true,
			[]string{"refused the saved token"}, []string{"Hugging Face said"}},
		{"bad token", resp(401, "", "Invalid username or password."), true,
			[]string{"did not accept the saved token"}, nil},
		{"missing repo, no token", resp(401, "RepoNotFound", ""), false,
			[]string{"cannot find org/model", "private repository"}, nil},
		{"missing file", resp(404, "EntryNotFound", "Entry not found"), false,
			[]string{"not in org/model"}, []string{"Entry not found"}},
		{"server error", resp(503, "", ""), false,
			[]string{"server problem", "(HTTP 503)"}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := statusError(c.resp, "org/model", c.hasToken).Error()
			for _, w := range c.want {
				if !strings.Contains(got, w) {
					t.Errorf("message %q lacks %q", got, w)
				}
			}
			for _, w := range c.notWant {
				if strings.Contains(got, w) {
					t.Errorf("message %q should not contain %q", got, w)
				}
			}
		})
	}
}
