package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// bannerServer is a Server with no builds and a fixed idea of the running
// toolchain, which is the state a freshly installed container is in.
func bannerServer(t *testing.T, current string) *Server {
	t.Helper()
	s := &Server{
		registry:        models.NewRegistry(t.TempDir(), t.TempDir()),
		builder:         builder.NewBuilder(t.TempDir()),
		currentBuildEnv: func(string) string { return current },
	}
	s.pages = s.parseTemplates()
	return s
}

func buildsPage(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleBuildsPage(rec, httptest.NewRequest("GET", "/builds", nil))
	if rec.Code != 200 {
		t.Fatalf("builds page: HTTP %d — %s", rec.Code, rec.Body.String())
	}
	return rec.Body.String()
}

// The page has to name the toolchain even with no builds at all. That is
// exactly the state after switching ROCm variants, and it is when the question
// "what is this actually running?" is most worth answering — before the
// per-build column has anything in it.
func TestBuildsPageNamesTheRunningToolchainWithNoBuilds(t *testing.T) {
	s := bannerServer(t, "rocm 10.0.0")
	out := buildsPage(t, s)

	if !strings.Contains(out, "rocm 10.0.0") {
		t.Errorf("the builds page does not say what it is running\n%s", out)
	}
	if !strings.Contains(out, "needs rebuilding before it will load") {
		t.Errorf("the banner does not explain why the version matters\n%s", out)
	}
}

// The base image can only be told to us by the image itself, so a container
// built without it — the stable Fedora one — must simply say less rather than
// render an empty "built on" fragment.
func TestBuildsPageOmitsTheBaseImageWhenUnknown(t *testing.T) {
	t.Setenv("LLAMA_TOOLCHEST_ROCM_BASE_IMAGE", "")
	out := buildsPage(t, bannerServer(t, "rocm 7.2.4"))
	if strings.Contains(out, "built on") {
		t.Errorf("the page claims a base image it was not given\n%s", out)
	}
	if !strings.Contains(out, "rocm 7.2.4") {
		t.Errorf("the toolchain is missing\n%s", out)
	}

	t.Setenv("LLAMA_TOOLCHEST_ROCM_BASE_IMAGE", "docker.io/rocm/dev-ubuntu-24.04:10.0.0-full")
	out = buildsPage(t, bannerServer(t, "rocm 10.0.0"))
	if !strings.Contains(out, "built on") || !strings.Contains(out, "dev-ubuntu-24.04:10.0.0-full") {
		t.Errorf("the page does not name the base image it was given\n%s", out)
	}
}

// Nothing is stated when the toolchain cannot be read — a host install with no
// ROCm, or a CPU-only machine. Saying nothing is correct; saying "running "
// with a blank would not be.
func TestBuildsPageSaysNothingWhenTheToolchainIsUnknown(t *testing.T) {
	out := buildsPage(t, bannerServer(t, ""))
	if strings.Contains(out, "Running <strong>") {
		t.Errorf("the page rendered an empty toolchain banner\n%s", out)
	}
	// The rest of the page must still be there.
	if !strings.Contains(out, "New Build") {
		t.Errorf("the builds page did not render\n%s", out)
	}
}
