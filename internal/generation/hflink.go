package generation

import (
	"net/url"
	"strings"
)

// hfHosts are the host names a Hugging Face link may use.
var hfHosts = []string{"huggingface.co", "www.huggingface.co"}

// hfPath returns the path segments of a Hugging Face URL, decoded, with
// the query and fragment left out. ok is false for any other URL.
func hfPath(s string) (segments []string, ok bool) {
	u, err := url.Parse(strings.TrimSpace(s))
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") {
		return nil, false
	}
	host := strings.ToLower(u.Host)
	found := false
	for _, h := range hfHosts {
		if host == h {
			found = true
			break
		}
	}
	if !found {
		return nil, false
	}
	for _, seg := range strings.Split(strings.Trim(u.Path, "/"), "/") {
		if seg != "" {
			segments = append(segments, seg)
		}
	}
	return segments, true
}

// ParseHFLink reads a link to one file on Hugging Face, such as
// https://huggingface.co/<org>/<repo>/blob/<rev>/<path> (or /resolve/
// in place of /blob/). It returns the repository ID, the revision and
// the file's path in the repository. ok is false for anything else,
// including a link to a repository page or a local path.
func ParseHFLink(s string) (repo, rev, file string, ok bool) {
	segs, ok := hfPath(s)
	if !ok {
		return "", "", "", false
	}
	for i, seg := range segs {
		if seg != "blob" && seg != "resolve" {
			continue
		}
		// The repository ID has one or two segments, and the revision and
		// at least one file segment follow the keyword.
		if i < 1 || i > 2 || len(segs) < i+3 {
			return "", "", "", false
		}
		return strings.Join(segs[:i], "/"), segs[i+1], strings.Join(segs[i+2:], "/"), true
	}
	return "", "", "", false
}

// HFLink writes the link to one file on Hugging Face that ParseHFLink
// reads back. An empty revision writes main.
func HFLink(repo, rev, file string) string {
	if rev == "" {
		rev = "main"
	}
	var parts []string
	for _, p := range [][]string{strings.Split(repo, "/"), {"blob", rev}, strings.Split(file, "/")} {
		for _, seg := range p {
			parts = append(parts, url.PathEscape(seg))
		}
	}
	return "https://huggingface.co/" + strings.Join(parts, "/")
}

// ModelRepo is the Hugging Face repository ID that the recipe's
// ModelURL names (https://huggingface.co/<org>/<name>, optionally with
// /tree/<rev>), or "" when it names none.
func (r *Recipe) ModelRepo() string {
	segs, ok := hfPath(r.ModelURL)
	if !ok || len(segs) < 2 {
		return ""
	}
	if len(segs) != 2 && !(len(segs) == 4 && segs[2] == "tree") {
		return ""
	}
	return segs[0] + "/" + segs[1]
}
