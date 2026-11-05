package api

import "strings"

// docsBaseURL is the address of the documentation site, built from docs/ and
// published by .github/workflows/docs.yml.
const docsBaseURL = "https://tmac1973.github.io/llama-toolchest/"

// docsURL turns a docs reference into the full URL of that page on the
// documentation site. Templates use it for their "More in the docs" links.
//
// ref is a page, optionally followed by "#anchor". The page is its path
// under docs/ without ".md", such as "gpu/eviction"; an index.md page is its
// folder with a trailing slash, such as "gpu/". The empty ref is the home
// page. So "gpu/eviction#modes" gives docsBaseURL+"gpu/eviction/#modes",
// because the site uses a folder for each page.
//
// TestDocsLinksExist checks that every ref in the templates names a page and
// an explicit heading id that exist in docs/.
func docsURL(ref string) string {
	page, anchor, hasAnchor := strings.Cut(ref, "#")
	u := docsBaseURL + page
	if page != "" && !strings.HasSuffix(page, "/") {
		u += "/"
	}
	if hasAnchor {
		u += "#" + anchor
	}
	return u
}
