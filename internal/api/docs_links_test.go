package api

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestDocsURL(t *testing.T) {
	cases := []struct{ ref, want string }{
		{"", docsBaseURL},
		{"gpu/", docsBaseURL + "gpu/"},
		{"gpu/#parts", docsBaseURL + "gpu/#parts"},
		{"api", docsBaseURL + "api/"},
		{"gpu/eviction#modes", docsBaseURL + "gpu/eviction/#modes"},
	}
	for _, c := range cases {
		if got := docsURL(c.ref); got != c.want {
			t.Errorf("docsURL(%q) = %q, want %q", c.ref, got, c.want)
		}
	}
}

// TestDocsLinksExist fails when a template links to a docs page or heading
// that docs/ does not have, for example after a page moves or a heading id
// changes. docs.yml runs it too, because a docs-only pull request does not
// run ci.yml.
func TestDocsLinksExist(t *testing.T) {
	if _, err := os.Stat("../../docs"); err != nil {
		t.Skip("no docs/ folder next to the code; the docs links are checked in the repo only")
	}
	problems, err := checkDocsLinks("../../web/templates", "../../docs")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range problems {
		t.Error(p)
	}
}

func TestCheckDocsLinksFixture(t *testing.T) {
	problems, err := checkDocsLinks("testdata/docs_links/templates", "testdata/docs_links/docs")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		`sample.html: docsURL "gpu/eviction#nope": no heading with id "nope" in docs/gpu/eviction.md`,
		`sample.html: docsURL "gpu/missing": no page docs/gpu/missing.md`,
	}
	if strings.Join(problems, "\n") != strings.Join(want, "\n") {
		t.Errorf("problems:\n%s\nwant:\n%s", strings.Join(problems, "\n"), strings.Join(want, "\n"))
	}
}

// docsRefInTemplate finds the ref of each docsURL call in a template, such
// as {{docsURL "gpu/eviction#modes"}}.
var docsRefInTemplate = regexp.MustCompile(`docsURL\s+"([^"]*)"`)

// checkDocsLinks reads every .html file under templatesDir and checks each
// docsURL ref against the Markdown pages in docsDir. It returns one message
// for each ref whose page does not exist, or whose page has no heading with
// that explicit id ("## Modes { #modes }"). Only explicit ids count, so a
// heading the app links to cannot change its id by a change of its text.
func checkDocsLinks(templatesDir, docsDir string) ([]string, error) {
	var problems []string
	err := filepath.WalkDir(templatesDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(path) != ".html" {
			return err
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(templatesDir, path)
		for _, m := range docsRefInTemplate.FindAllStringSubmatch(string(src), -1) {
			if msg := checkDocsRef(m[1], docsDir); msg != "" {
				problems = append(problems, fmt.Sprintf("%s: docsURL %q: %s", rel, m[1], msg))
			}
		}
		return nil
	})
	return problems, err
}

// checkDocsRef checks one docsURL ref and returns what is wrong with it, or
// "" when its page and anchor exist.
func checkDocsRef(ref, docsDir string) string {
	if ref == "" {
		return ""
	}
	page, anchor, hasAnchor := strings.Cut(ref, "#")
	file := page + ".md"
	if page == "" || strings.HasSuffix(page, "/") {
		file = page + "index.md"
	}
	src, err := os.ReadFile(filepath.Join(docsDir, filepath.FromSlash(file)))
	if err != nil {
		return "no page docs/" + file
	}
	if !hasAnchor {
		return ""
	}
	id := regexp.MustCompile(`(?m)^#{1,6} .*\{[^}]*#` + regexp.QuoteMeta(anchor) + `[\s}]`)
	if !id.Match(src) {
		return fmt.Sprintf("no heading with id %q in docs/%s", anchor, file)
	}
	return ""
}
