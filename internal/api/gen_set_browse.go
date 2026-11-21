package api

import (
	"context"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// hfBrowseTimeout bounds the file listing of the Browse dialog.
const hfBrowseTimeout = 15 * time.Second

// loadableExts are the file types sd-server, ace-server and the upscaler
// load. The Browse dialog shows only these unless asked for all files.
var loadableExts = []string{".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}

// genBrowseFile is one file the Browse dialog offers.
type genBrowseFile struct {
	Path, Size, Link string
	// From names the catalog set that uses the file, for a suggestion.
	From string
}

// genBrowseRepo is one repository the dialog offers to list.
type genBrowseRepo struct {
	Repo, URL string
	Current   bool
}

// genBrowseView is the Browse dialog's body (gen_set_browse.html).
type genBrowseView struct {
	Slot, Kind, Recipe, Repo string
	All                      bool
	Repos                    []genBrowseRepo
	Suggestions              []genBrowseFile
	Files                    []genBrowseFile
	// Hidden counts the files the filter left out; ToggleURL shows them
	// (or hides them again when All).
	Hidden    int
	ToggleURL string
	// Diffusers is true when the repository is in the Diffusers layout.
	Diffusers bool
	Error     string
}

// browseURL is the dialog's own address for a repository.
func browseURL(slot, kind, recipe, repo string, all bool) string {
	q := url.Values{"slot": {slot}, "kind": {kind}, "recipe": {recipe}, "repo": {repo}}
	if all {
		q.Set("all", "1")
	}
	return "/api/gen/models/sets/form/browse?" + q.Encode()
}

// repoFromInput reads a repository ID typed or pasted into the dialog:
// "org/name", or a link to its page or to one of its files.
func repoFromInput(s string) string {
	s = strings.TrimSpace(s)
	if repo, _, _, ok := generation.ParseHFLink(s); ok {
		return repo
	}
	if u, err := url.Parse(s); err == nil && u.Host != "" {
		segs := strings.Split(strings.Trim(u.Path, "/"), "/")
		if len(segs) >= 2 {
			return segs[0] + "/" + segs[1]
		}
		return ""
	}
	return strings.Trim(s, "/")
}

// gbText writes a file size for people in GB, as the generation set
// rows and the space check do: one decimal, or MB below 1 GB.
func gbText(n int64) string {
	if n >= 1e9 {
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	}
	return fmt.Sprintf("%.0f MB", float64(n)/1e6)
}

// loadable reports whether the slot can take the file: a JSON file for
// the tokenizer, a model file for the rest.
func loadable(slot, path string) bool {
	lower := strings.ToLower(path)
	if slot == "tokenizer" {
		return strings.HasSuffix(lower, ".json")
	}
	for _, ext := range loadableExts {
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// handleGenSetBrowse serves the Browse dialog of one file field of the
// custom set form: the recipe's repository and the files catalog sets
// use for the slot, and the files of one repository. A failed listing is
// shown in the dialog, so the form stays usable.
func (s *Server) handleGenSetBrowse(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	v := genBrowseView{Slot: r.FormValue("slot"), Kind: r.FormValue("kind"), Recipe: r.FormValue("recipe"),
		All: r.FormValue("all") == "1"}
	if v.Slot == "" {
		http.Error(w, "no slot", http.StatusBadRequest)
		return
	}
	kind := generation.BackendKind(v.Kind)

	var repos []string
	if kind == generation.KindSD || kind == generation.KindVideo {
		if rec, ok := generation.ActiveRecipes().Get(v.Recipe); ok && rec.ModelRepo() != "" {
			repos = append(repos, rec.ModelRepo())
		}
	}
	for _, sg := range generation.SlotSuggestions(kind, v.Recipe, v.Slot) {
		if !slices.Contains(repos, sg.Repo) {
			repos = append(repos, sg.Repo)
		}
		v.Suggestions = append(v.Suggestions, genBrowseFile{Path: sg.Repo + "/" + sg.File, Size: gbText(sg.Bytes),
			Link: generation.HFLink(sg.Repo, "main", sg.File), From: sg.SetName})
	}
	v.Repo = repoFromInput(r.FormValue("repo"))
	if v.Repo == "" && len(repos) > 0 {
		v.Repo = repos[0]
	}
	for _, repo := range repos {
		v.Repos = append(v.Repos, genBrowseRepo{Repo: repo, URL: browseURL(v.Slot, v.Kind, v.Recipe, repo, false), Current: repo == v.Repo})
	}

	if v.Repo != "" {
		ctx, cancel := context.WithTimeout(r.Context(), hfBrowseTimeout)
		files, err := s.repoFiles(ctx, v.Repo, "main")
		cancel()
		if err != nil {
			v.Error = hfResolveError(ctx, "Listing "+v.Repo, err).Error()
		} else {
			v.fill(files)
		}
	}
	respondHTML(w)
	s.renderPartial(w, "gen_set_browse", v)
}

// fill lists the repository's files, filtered unless All.
func (v *genBrowseView) fill(files []modelsource.RepoFile) {
	slices.SortFunc(files, func(a, b modelsource.RepoFile) int { return strings.Compare(a.Path, b.Path) })
	for _, f := range files {
		if f.Path == "model_index.json" {
			v.Diffusers = true
		}
		if !v.All && !loadable(v.Slot, f.Path) {
			v.Hidden++
			continue
		}
		v.Files = append(v.Files, genBrowseFile{Path: f.Path, Size: gbText(f.Size), Link: generation.HFLink(v.Repo, "main", f.Path)})
	}
	if v.All || v.Hidden > 0 {
		v.ToggleURL = browseURL(v.Slot, v.Kind, v.Recipe, v.Repo, !v.All)
	}
}

// handleGenSetFormSize serves the custom set form's download-size line:
// how much its Hugging Face links download, and whether that fits. It is
// a guide; Add set checks again.
func (s *Server) handleGenSetFormSize(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	respondHTML(w)
	line := func(text string, bad bool) {
		style := ""
		if bad {
			style = ` style="color:var(--pico-del-color);"`
		}
		fmt.Fprintf(w, `<small%s>%s</small>`, style, html.EscapeString(text))
	}
	r.ParseForm()
	comps, err := formComponents(r)
	if err != nil {
		line(err.Error(), true)
		return
	}
	if !slices.ContainsFunc(comps, func(c generation.Component) bool { return c.Repo != "" }) {
		return
	}
	comps, err = s.resolveHFComponents(r.Context(), comps, nil)
	if err != nil {
		line(err.Error(), true)
		return
	}
	need := downloadBytes(comps)
	if need == 0 {
		line("Download: nothing. The linked files are already in the models folder.", false)
		return
	}
	if err := s.checkDownloadSpace(comps); err != nil {
		line(err.Error(), true)
		return
	}
	text := "Download: " + gbText(need)
	if avail := s.downloader.AvailableForDownload(); avail >= 0 {
		text += " · " + gbText(avail) + " free"
	}
	line(text, false)
}
