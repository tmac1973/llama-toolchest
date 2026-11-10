package api

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"sort"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/patchsets"
)

func (s *Server) handleListBackends(w http.ResponseWriter, r *http.Request) {
	backends := builder.DetectBackends()
	respondJSON(w, backends)
}

func (s *Server) handleProfileOptions(w http.ResponseWriter, r *http.Request) {
	profile := r.URL.Query().Get("profile")
	project := r.URL.Query().Get("project")
	if _, ok := patchSetChoice(project); ok {
		project = builder.ProjectLlamaCpp
	}
	proj, ok := builder.FindProject(project)
	if !ok {
		http.Error(w, "unknown project: "+project, http.StatusBadRequest)
		return
	}
	options := builder.ProjectOptions(proj.ID, profile)

	// Read current toggle states from query params (sent on re-fetch)
	overrides := make(map[string]bool)
	hasOverrides := false
	for _, opt := range options {
		key := "opt_" + opt.Flag
		if r.URL.Query().Has(key) {
			hasOverrides = true
			overrides[opt.Flag] = r.URL.Query().Get(key) == "on"
		}
	}
	extraCMake := r.URL.Query().Get("extra_cmake")

	// Applying a saved flag set: its stored states replace the live ones,
	// and the Build Tag is filled with the preset name via an OOB swap
	// (below). Later toggle edits re-fetch without preset=, so tweaks on
	// top of a loaded preset stick until saved again.
	var loadedPreset *builder.FlagPreset
	if name := r.URL.Query().Get("preset"); name != "" {
		if p, ok := s.builder.FindFlagPreset(name); ok && p.Profile == profile {
			hasOverrides = true
			overrides = p.Options
			extraCMake = p.ExtraCMake
			loadedPreset = &p
		}
	}

	if isHTMX(r) {
		respondHTML(w)
		if len(options) == 0 {
			return
		}

		data := buildOptionsData{ExtraCMake: extraCMake}
		effectiveOverrides := make(map[string]bool, len(options))
		for _, opt := range options {
			on := opt.Default
			if hasOverrides {
				on = overrides[opt.Flag]
			}
			effectiveOverrides[opt.Flag] = on
			data.Options = append(data.Options, buildOptionRow{BuildOption: opt, Checked: on})
		}

		// The ROCm install picker, only when there is a choice to make.
		var pin *builder.ROCmInstall
		prof, profOK := builder.FindProfile(profile)
		if profOK && prof.Backend == "rocm" {
			installs := s.rocmInstalls()
			chosenRoot := ""
			if len(installs) > 1 {
				chosen := installs[0]
				if i, ok := builder.FindROCmInstall(installs, r.URL.Query().Get("rocm_root")); ok {
					chosen = i
				}
				for _, i := range installs {
					data.ROCmInstalls = append(data.ROCmInstalls, rocmInstallChoice{
						Root: i.Root, Label: i.Label(), Selected: i.Root == chosen.Root,
					})
				}
				data.ROCmWarning = chosen.Damaged
				chosenRoot = chosen.Root
			}
			// A build can be pinned with no picker too (a lone 10.x install,
			// or a stray hipconfig), so ask the build's own rule.
			if inst, pinned, err := builder.ChooseROCm(installs, chosenRoot, s.hipconfigUsable()); err == nil && pinned {
				pin = inst
			}
		}

		// Show effective cmake flags with current toggle states
		if profOK && !proj.SupportsBackend(prof.Backend) {
			data.ProfileWarning = fmt.Sprintf("%s can only be built with the %s profiles.", proj.ID, strings.Join(proj.Backends, ", "))
		} else if profOK {
			flags := effectiveCMakeFlags(proj, prof, options, effectiveOverrides, pin)
			if extraCMake != "" {
				flags += " " + extraCMake
			}
			data.ShowEffective = true
			data.EffectiveFlags = flags
		}
		if loadedPreset != nil {
			data.PresetTag = loadedPreset.Name
		}
		s.renderPartial(w, "build_options", data)
		return
	}

	respondJSON(w, options)
}

// buildOptionsData feeds the build_options partial.
type buildOptionsData struct {
	Options    []buildOptionRow
	ExtraCMake string
	// ShowEffective is false for an unknown profile, which has no flags
	// to preview.
	ShowEffective  bool
	EffectiveFlags string
	// PresetTag is the name of the saved flag set just applied, which
	// fills the Build Tag; empty otherwise.
	PresetTag string
	// ROCmInstalls fills the ROCm install picker; empty, and the picker
	// hidden, unless the profile is rocm and the machine has more than one.
	ROCmInstalls []rocmInstallChoice
	// ROCmWarning explains what is wrong with the chosen install, if
	// anything (builder.ROCmInstall.Damaged).
	ROCmWarning string
	// ProfileWarning says the chosen project cannot be built with the
	// chosen profile; the flag preview is left out then.
	ProfileWarning string
}

// rocmInstallChoice is one option in the ROCm install picker.
type rocmInstallChoice struct {
	Root, Label string
	Selected    bool
}

// buildOptionRow is one toggle: the option and whether it is on.
type buildOptionRow struct {
	builder.BuildOption
	Checked bool
}

// effectiveCMakeFlags previews the flags a build will get. pin is the ROCm
// install it will be pinned to, nil when it will not be (one install, or not
// rocm), and adds the flags the build adds for it.
func effectiveCMakeFlags(proj builder.Project, prof builder.BuildProfile, options []builder.BuildOption, overrides map[string]bool, pin *builder.ROCmInstall) string {
	flags := make(map[string]string)
	for k, v := range prof.CMakeFlags {
		flags[k] = v
	}
	builder.ApplyOptionOverrides(flags, options, overrides)
	proj.MapFlags(flags)
	if pin != nil {
		builder.PinROCmFlags(flags, *pin)
	}
	var parts []string
	for k, v := range flags {
		parts = append(parts, fmt.Sprintf("-D%s=%s", k, v))
	}
	// Sort for stable display
	sort.Strings(parts)
	return strings.Join(parts, " ")
}

func (s *Server) handleListRefs(w http.ResponseWriter, r *http.Request) {
	refresh := r.URL.Query().Get("refresh") == "1"
	// A patch set offers its tested release, its newest one (read from
	// the set's repo when it builds) and its recent releases.
	if set, ok := patchSetChoice(r.URL.Query().Get("project")); ok {
		rels, err := s.builder.PatchSetReleases(set.ID, refresh)
		data := patchRefOptionsFor(set, rels, err)
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "git_ref_options", data)
			return
		}
		refs := []string{set.Pinned.Tag, "latest"}
		for _, o := range data.PatchRecent {
			refs = append(refs, o.Ref)
		}
		respondJSON(w, refs)
		return
	}
	proj, ok := builder.FindProject(r.URL.Query().Get("project"))
	if !ok {
		http.Error(w, "unknown project", http.StatusBadRequest)
		return
	}

	var refs []string
	var err error
	if refresh {
		refs, err = s.builder.FetchRefsFor(proj.ID)
		if err != nil {
			// Return cached if fetch fails
			refs = s.builder.CachedRefsFor(proj.ID)
		}
	} else {
		refs = s.builder.CachedRefsFor(proj.ID)
	}

	if isHTMX(r) {
		respondHTML(w)
		if proj.ID != builder.ProjectLlamaCpp {
			s.renderPartial(w, "git_ref_options", projectRefOptionsFor(refs, err))
			return
		}
		s.renderPartial(w, "git_ref_options", gitRefOptionsFor(refs, s.builder.ReleaseAnchors(), err))
		return
	}

	respondJSON(w, refs)
}

// gitRefOptionsData feeds the git_ref_options partial.
type gitRefOptionsData struct {
	Groups []gitRefGroup // only groups with at least one tag
	// Error is why a refresh failed when nothing was cached either.
	Error string
	// PatchRelease is a patch set's tested release tag and PatchLabel its
	// short name ("r206"). When set, the picker offers it (first,
	// chosen), the set's newest release and PatchRecent.
	PatchRelease, PatchLabel string
	// PatchRecent are the set's newest releases other than the tested
	// one, newest first; Label is "r220 · b11887".
	PatchRecent []gitRefOption
}

// patchRecentMax is how many recent releases the patch set picker lists.
// The set's repo makes several releases a day, so the list stops here.
const patchRecentMax = 10

// patchRefOptionsFor fills the release picker of a patch set from its
// releases, newest first. A listing error (no clone yet) leaves only the
// tested and the newest release, with the error as a note.
func patchRefOptionsFor(set patchsets.PatchSet, rels []builder.PatchSetRelease, err error) gitRefOptionsData {
	data := gitRefOptionsData{PatchRelease: set.Pinned.Tag, PatchLabel: releaseLabel(set.Pinned.Tag)}
	if err != nil {
		data.Error = err.Error()
		return data
	}
	for _, r := range rels {
		if len(data.PatchRecent) == patchRecentMax {
			break
		}
		if r.Tag == set.Pinned.Tag {
			continue
		}
		label := fmt.Sprintf("r%d · b%d", r.N, r.BaseCount)
		if r.BaseCount < 0 {
			base, _, _ := patchsets.ParseTag(r.Tag)
			label = fmt.Sprintf("r%d · base %s", r.N, base)
		}
		data.PatchRecent = append(data.PatchRecent, gitRefOption{Ref: r.Tag, Label: label})
	}
	return data
}

type gitRefGroup struct {
	Label string
	Refs  []gitRefOption
}

// gitRefOption is one tag. A release carries the nightly it was cut
// from (Anchor), when known.
type gitRefOption struct {
	Ref       string
	Anchor    int
	HasAnchor bool
	// Label is the text shown for a patch set release; empty for a tag.
	Label string
}

// gitRefOptionsFor groups the upstream tags for the git ref picker. There
// are two tag families since upstream added semver releases (Aug 2026):
// v* release tags and b* nightlies. Grouping them lets the picker say
// which is which; "latest" still means the newest nightly. Releases are
// labeled with the nightly they were cut from ("v0.2.0 (b10500)") so
// their position on the b-scale is readable without leaving the picker.
// Values stay bare tags.
func gitRefOptionsFor(refs []string, anchors map[string]int, err error) gitRefOptionsData {
	var data gitRefOptionsData
	isRelease := func(ref string) bool { return strings.HasPrefix(ref, "v") }
	for _, g := range []struct {
		label   string
		release bool
	}{{"Releases", true}, {"Nightly builds", false}} {
		group := gitRefGroup{Label: g.label}
		for _, ref := range refs {
			if isRelease(ref) == g.release {
				n, found := anchors[ref]
				group.Refs = append(group.Refs, gitRefOption{Ref: ref, Anchor: n, HasAnchor: found})
			}
		}
		if len(group.Refs) > 0 {
			data.Groups = append(data.Groups, group)
		}
	}
	// A refresh that fails with nothing cached leaves the picker holding
	// only "latest", which reads as "upstream has no tags" rather than
	// "the refresh didn't work". Say which it was; the template shows it
	// as a disabled option so it can never be submitted as a ref.
	if err != nil && len(refs) == 0 {
		data.Error = err.Error()
	}
	return data
}

// projectRefOptionsFor lists a generation backend's tags for the git ref
// picker. They have one tag family and no release anchors.
func projectRefOptionsFor(refs []string, err error) gitRefOptionsData {
	var data gitRefOptionsData
	if len(refs) > 0 {
		group := gitRefGroup{Label: "Tags"}
		for _, ref := range refs {
			group.Refs = append(group.Refs, gitRefOption{Ref: ref})
		}
		data.Groups = append(data.Groups, group)
	}
	if err != nil && len(refs) == 0 {
		data.Error = err.Error()
	}
	return data
}

func (s *Server) handleListBuilds(w http.ResponseWriter, r *http.Request) {
	builds := s.builder.ListAll()

	// If request is from htmx, return HTML partial
	if isHTMX(r) {
		if len(builds) == 0 {
			w.Write([]byte("<p>No builds yet.</p>"))
			return
		}
		rows := make([]buildRow, len(builds))
		for i := range builds {
			rows[i] = s.buildRowFor(&builds[i])
		}
		respondHTML(w)
		s.renderPartial(w, "build_table", rows)
		return
	}

	respondJSON(w, builds)
}

// buildDuplicateData feeds the build_duplicate partial: the build that
// already exists and the request to resubmit with force=1.
type buildDuplicateData struct {
	ID, Project, Profile, GitRef, Tag string
}

func (s *Server) handleTriggerBuild(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Project string `json:"project"`
		Profile string `json:"profile"`
		GitRef  string `json:"git_ref"`
		Tag     string `json:"tag"`
		Force   bool   `json:"force"`
		// ROCmRoot picks the ROCm install when the machine has several;
		// empty for the newest.
		ROCmRoot string `json:"rocm_root"`
		// StartToken names this start for /api/builds/start/{token}/logs,
		// which streams the fetch and checkout lines before the build has
		// an ID. Optional.
		StartToken string `json:"start_token"`
	}

	// Support both JSON and form-encoded
	if isJSONRequest(r) {
		if !decodeJSONBody(w, r, &req) {
			return
		}
	} else {
		if !parseForm(w, r) {
			return
		}
		req.Project = r.FormValue("project")
		req.Profile = r.FormValue("profile")
		req.GitRef = r.FormValue("git_ref")
		req.Tag = r.FormValue("tag")
		req.Force = r.FormValue("force") == "1"
		req.ROCmRoot = r.FormValue("rocm_root")
		req.StartToken = r.FormValue("start_token")
	}

	// A patch set choice builds llama.cpp, with llama.cpp's options.
	set, patched := patchSetChoice(req.Project)
	if !patched && strings.HasPrefix(req.Project, patchSetChoicePrefix) {
		http.NotFound(w, r)
		return
	}
	optionsProject := req.Project
	if patched {
		optionsProject = builder.ProjectLlamaCpp
	}

	// Collect option overrides and extra cmake flags from form
	var optionOverrides map[string]bool
	var extraCMake string
	if !isJSONRequest(r) {
		options := builder.ProjectOptions(optionsProject, req.Profile)
		optionOverrides = make(map[string]bool)
		for _, opt := range options {
			// Checkboxes only send a value when checked
			optionOverrides[opt.Flag] = r.FormValue("opt_"+opt.Flag) == "on"
		}
		extraCMake = r.FormValue("extra_cmake")
	}

	// Use background context — the build must outlive the HTTP request.
	if req.Project == "" {
		req.Project = builder.ProjectLlamaCpp
	}
	if f, ok := forks.Find(req.Project); ok {
		if !s.forksOn() {
			featureOff(w, "Forks")
			return
		}
		// A fork builds only its latest commit: it moves too fast to pin.
		if req.GitRef != "" && req.GitRef != "latest" {
			http.Error(w, fmt.Sprintf("%s builds only its latest commit", f.Name), http.StatusBadRequest)
			return
		}
		req.GitRef = "latest"
	}
	if patched {
		// A patch set builds one of its releases, by tag, or its newest
		// one; the builder checks that the release exists.
		if _, _, ok := patchsets.ParseTag(req.GitRef); !ok && req.GitRef != "" && req.GitRef != "latest" {
			http.Error(w, fmt.Sprintf("%s builds only its releases (for example %s) or its newest release", set.Name, set.Pinned.Tag), http.StatusBadRequest)
			return
		}
		if req.GitRef == "" {
			req.GitRef = set.Pinned.Tag
		}
	}
	ctx := builder.WithStartToken(context.Background(), req.StartToken)
	var result *builder.BuildResult
	var err error
	if patched {
		result, err = s.builder.BuildPatched(ctx, builder.PatchSpec{SetID: set.ID, Release: req.GitRef}, req.Profile, req.Tag, req.Force, optionOverrides, extraCMake, req.ROCmRoot)
	} else {
		result, err = s.builder.BuildProject(ctx, req.Project, req.Profile, req.GitRef, req.Tag, req.Force, optionOverrides, extraCMake, req.ROCmRoot)
	}
	if err != nil {
		var dup *builder.DuplicateBuildError
		if errors.As(err, &dup) {
			if isHTMX(r) {
				respondHTML(w)
				s.renderPartial(w, "build_duplicate", buildDuplicateData{
					ID: dup.ID, Project: req.Project, Profile: req.Profile, GitRef: req.GitRef, Tag: req.Tag,
				})
				// The prompt replaces the log area; the running builds'
				// logs stay under it.
				s.renderPartial(w, "build_logs", s.buildLogPanels())
				return
			}
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		slog.Error("build failed to start", "profile", req.Profile, "git_ref", req.GitRef, "error", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Return the log panels for htmx to swap in: the new build's and those
	// of the builds still running, so a build started beside another does
	// not hide its log.
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "build_logs", s.buildLogPanels())
		return
	}

	w.WriteHeader(http.StatusAccepted)
	respondJSON(w, result)
}

func (s *Server) handleBuildLogs(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	// Always use SubscribeLogs which replays history and streams new lines.
	// This handles both in-progress builds and reconnections after tab switches.
	sub := s.builder.SubscribeLogs(id)
	if sub == nil {
		// No history — try the raw channel as last resort (shouldn't happen normally)
		ch, ok := s.builder.LogChannel(id)
		if !ok {
			http.NotFound(w, r)
			return
		}
		StreamLines(w, r.Context(), ch, "Build complete")
		return
	}
	defer s.builder.UnsubscribeLogs(id, sub)
	StreamLines(w, r.Context(), sub, "Build complete")
}

// handleActiveBuildLog returns the build_log partial for the most recent build,
// allowing the builds page to reconnect after tab switches.
func (s *Server) handleActiveBuildLog(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	s.renderPartial(w, "build_logs", s.buildLogPanels())
}

// buildLogPanels is the builds whose log the Builds page shows: every
// running build, newest first, or else the most recent build, so its
// finished log stays in view.
func (s *Server) buildLogPanels() []builder.BuildResult {
	if running := s.builder.RunningBuilds(); len(running) > 0 {
		return running
	}
	lastID := s.builder.LastBuildID()
	if lastID == "" {
		return nil
	}
	status := s.builder.BuildStatus(lastID)
	if status == "" {
		return nil
	}
	return []builder.BuildResult{{ID: lastID, Status: status}}
}

// handleBuildStartLogs streams the fetch and checkout lines of a build
// start named by its token, before the build has an ID. The page shows
// them in a "Starting" panel until the start request returns.
func (s *Server) handleBuildStartLogs(w http.ResponseWriter, r *http.Request) {
	ch, done, ok := s.builder.SubscribeStart(chi.URLParam(r, "token"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	defer done()
	StreamLines(w, r.Context(), ch, "Started")
}

// resolveActiveBuild returns the build the router should run: the explicitly
// selected cfg.ActiveBuild when it exists and built successfully, otherwise
// the successful build with the newest GitRef. Returns nil when no runnable
// build exists.
func (s *Server) resolveActiveBuild() *builder.BuildResult {
	return s.resolveBuild(s.activeBuild())
}

// buildRow is a build plus what the Builds page needs to say about it. The
// BuildResult is embedded so the template's existing field references keep
// working unchanged.
//
// One text field and one title field cover all three states, so the template
// carries no wording of its own and the table and the info modal cannot drift
// apart.
type buildRow struct {
	*builder.BuildResult
	// BuiltAgainstText is the stamp, or an em-dash when the build has none.
	BuiltAgainstText string
	// Mismatch is true only when both stamps are known, name the same
	// backend, and differ.
	Mismatch bool
	// BuiltAgainstTitle explains the cell: why it is flagged, or why it is
	// blank. Empty only when the build is stamped and matches.
	BuiltAgainstTitle string
	// Untested marks a patched build of a release other than the tested
	// one.
	Untested bool
	// PatchLabel names a patched build's patch set and release ("RDNA
	// boosts r206"), and PatchTitle is its tooltip. Empty on an
	// unpatched build.
	PatchLabel, PatchTitle string
}

// notRecordedTitle is the tooltip for a build with no stamp. Worded as a
// near-twin of the cmake-flags fallback in build_info.html ("cmake flags
// not recorded — this build predates flag tracking") so the two read as one
// convention. Used verbatim by both the table and the info modal.
const notRecordedTitle = "Not recorded — this build predates build-environment tracking."

// buildRowFor decides what the page says about one build.
//
// The mismatch wording is deliberately "may fail to load" rather than "will
// not run", and says "linked against the libraries" rather than naming ROCm as
// the cause. Both were measured on this project's own two images.
//
// A build made under ROCm 10.0.0 loads and enumerates the GPU under ROCm 7.2.4
// once the library path is available, because both ship the same hipBLAS
// soname — so "will not run" would be false.
//
// And the failures that do occur are not always about ROCm. Going one way, the
// ROCm 10 image bakes no usable ROCm path into libggml-hip.so, so nothing finds
// hipBLAS. Going the other, a Fedora-built binary fails on Ubuntu with
// "libcrypto.so.3: version OPENSSL_3.3.0 not found" — an OpenSSL symbol
// version, nothing to do with ROCm at all. The stamp is a useful proxy for
// "this was built somewhere else"; it is not a diagnosis.
func (s *Server) buildRowFor(b *builder.BuildResult) buildRow {
	row := buildRow{BuildResult: b, BuiltAgainstText: "—"}
	if b == nil {
		return row
	}
	if b.PatchSet != "" {
		var tested bool
		row.PatchLabel, tested = patchLabel(b)
		row.Untested = !tested
		row.PatchTitle = fmt.Sprintf("%s · Base: %s (b%d)", b.PatchRelease, safeShortSHA(b.PatchBase), b.CommitCount)
	}
	current := s.currentBuildEnvFor(buildBackend(b))
	gone, perInstall := builder.InstallGone(b.BuiltAgainst, s.rocmInstalls())
	switch {
	case b.BuiltAgainst == "":
		row.BuiltAgainstTitle = notRecordedTitle
	case perInstall && gone:
		_, root, _ := builder.ROCmStampInstall(b.BuiltAgainst)
		row.BuiltAgainstText = b.BuiltAgainst
		row.Mismatch = true
		row.BuiltAgainstTitle = fmt.Sprintf(
			"Built against %s. That ROCm install is no longer on this machine, so the build may fail to load; rebuild it if the server does not start. It loads again if %s comes back.",
			b.BuiltAgainst, root)
	case perInstall:
		row.BuiltAgainstText = b.BuiltAgainst
	case builder.StampMismatch(b.BuiltAgainst, current):
		row.BuiltAgainstText = b.BuiltAgainst
		row.Mismatch = true
		row.BuiltAgainstTitle = fmt.Sprintf(
			"Built against %s; this container runs %s. The build is linked against the libraries of its own image, so it may fail to load here; rebuild it if the server does not start.",
			b.BuiltAgainst, current)
	default:
		row.BuiltAgainstText = b.BuiltAgainst
	}
	return row
}

// serverBuildChoice is one entry in the Server page's build picker: a build,
// whether it can run here, and whether the picker refuses it.
type serverBuildChoice struct {
	buildRow
	// Disabled makes the option unselectable. Three deliberate exceptions,
	// each of which would otherwise leave someone stuck:
	//   - a build with no stamp is never disabled, because it cannot be judged;
	//   - when every stamped build mismatches, none is disabled, since
	//     refusing all of them would offer nothing that can be started;
	//   - the build already selected is never disabled, because a disabled
	//     selected <option> cannot be submitted and the form would silently
	//     post a different value on the next change.
	Disabled bool
}

// serverBuildChoices builds the Server page's picker. allMismatched is true
// when nothing here could run: every successful build was built somewhere else,
// so refusing them all would leave nothing to start.
//
// The test is "is there any candidate", not "does every stamped build
// mismatch". An unstamped build is a candidate — it cannot be judged, so it
// might well work — and while one exists there is somewhere to fall back to and
// the mismatches can safely be refused.
func (s *Server) serverBuildChoices(activeBuild string) (choices []serverBuildChoice, allMismatched bool) {
	builds := s.builder.List()
	successful, candidates := 0, 0
	rows := make([]buildRow, 0, len(builds))
	for i := range builds {
		row := s.buildRowFor(&builds[i])
		rows = append(rows, row)
		if builds[i].Status == builder.BuildStatusSuccess {
			successful++
			if !row.Mismatch {
				candidates++
			}
		}
	}
	allMismatched = successful > 0 && candidates == 0

	choices = make([]serverBuildChoice, 0, len(rows))
	for _, row := range rows {
		c := serverBuildChoice{buildRow: row}
		c.Disabled = row.Mismatch && !allMismatched && row.ID != activeBuild
		choices = append(choices, c)
	}
	return choices, allMismatched
}

// currentBuildEnvFor reports the toolchain the running container has, through
// an overridable field so tests can decide what "current" is. Without that seam
// a render test would pass or fail according to whichever ROCm happens to be
// installed on the machine running `go test`.
func (s *Server) currentBuildEnvFor(backend string) string {
	if s.currentBuildEnv != nil {
		return s.currentBuildEnv(backend)
	}
	return builder.CurrentBuildEnv(backend)
}

// rocmInstalls lists the ROCm installs here, through an overridable field for
// the same reason as currentBuildEnvFor.
func (s *Server) rocmInstalls() []builder.ROCmInstall {
	if s.rocmInstallsFn != nil {
		return s.rocmInstallsFn()
	}
	return builder.CachedROCmInstalls()
}

// hipconfigUsable is builder.HipconfigUsable, through an overridable field
// for the same reason as currentBuildEnvFor.
func (s *Server) hipconfigUsable() bool {
	if s.hipconfigUsableFn != nil {
		return s.hipconfigUsableFn()
	}
	return builder.HipconfigUsable()
}

// inContainer reports whether toolchest runs inside a container, through
// an overridable field for the same reason as currentBuildEnvFor.
func (s *Server) inContainer() bool {
	if s.inContainerFn != nil {
		return s.inContainerFn()
	}
	return builder.RunningInContainer()
}

// buildBackend returns a build's backend ("rocm", "cuda", ...), resolved
// through its profile. It selects the llama.cpp device-name prefix
// (ROCm0, CUDA0, ...) when the preset emits per-model device lists.
func buildBackend(b *builder.BuildResult) string {
	if b == nil {
		return ""
	}
	if p, ok := builder.FindProfile(b.Profile); ok {
		return p.Backend
	}
	return b.Profile
}

// activeBackend is buildBackend for the build the router would launch
// with right now (saved selection, latest-successful fallback).
func (s *Server) activeBackend() string {
	return buildBackend(s.resolveActiveBuild())
}

// buildTarget describes a build for writing launch options: its backend,
// and its llama.cpp version, which decides the spelling of options that
// have been renamed or removed upstream. A build whose version cannot be
// told is written for as current llama.cpp.
func buildTarget(b *builder.BuildResult) models.Target {
	t := models.Target{Backend: buildBackend(b)}
	if b != nil {
		if n, ok := b.Rank(); ok {
			t.Version = n
		}
	}
	return t
}

// activeTarget is buildTarget for the build the router would launch with
// right now.
func (s *Server) activeTarget() models.Target {
	return s.targetFor(s.resolveActiveBuild())
}

// targetFor is buildTarget with this machine's GPUs, which the preset
// needs to balance expert offload across cards. The preset is read when
// the router starts, with nothing of ours loaded, so it is balanced for
// the room the cards have then.
func (s *Server) targetFor(b *builder.BuildResult) models.Target {
	t := buildTarget(b)
	room := models.RoomOf(s.idleHardware())
	t.GPUMiB, t.WholeGPUMiB = room.Shared, room.Whole
	return t
}

// resolveBuild returns the build to launch for an explicit id, falling
// back to the newest successful build when the id is empty or unusable.
// Takes the id as a parameter so callers that already read cfg under the
// lock don't read it again unguarded.
func (s *Server) resolveBuild(id string) *builder.BuildResult {
	if id != "" {
		// An explicit choice is honoured even when it cannot run here. The
		// picker marks and refuses those, and an API caller that names one
		// anyway has decided; the loader error is then the answer.
		if b, ok := s.builder.Find(id); ok && b.Status == builder.BuildStatusSuccess && b.ProjectID() == builder.ProjectLlamaCpp {
			return b
		}
	}

	// No choice saved: the newest build that can actually run in this
	// container image, rather than simply the newest.
	//
	// Without the filter, "Auto" picks a build compiled in the other image and
	// the router fails to start, which is the most confusing possible default
	// — the one option that looks safest silently choosing a broken build.
	// Builds with no stamp are not skipped: they cannot be judged, so they
	// stay candidates.
	//
	// Patched builds are passed over while an upstream build can run: a
	// patched build ranks at its upstream base, so without this, building
	// a patch set could change the router's build without the user
	// choosing it (plan/rdna-boosts).
	ranked := s.builder.SuccessfulBuildsRanked()
	for _, patched := range []bool{false, true} {
		for i := range ranked {
			if (ranked[i].PatchSet != "") == patched && !s.buildRowFor(&ranked[i]).Mismatch {
				res := ranked[i]
				return &res
			}
		}
	}
	// Every build was made somewhere else. Return the newest anyway: the
	// alternative is refusing to start at all, and the same reasoning applies
	// here as in the picker's last-resort case.
	if len(ranked) > 0 {
		res := ranked[0]
		return &res
	}
	return nil
}

// resolveBackendBuild returns the build a generation backend instance runs:
// the named build when it is a successful build of the backend's project,
// otherwise the newest successful one. Nil when there is none.
func (s *Server) resolveBackendBuild(kind, id string) *builder.BuildResult {
	p, ok := builder.ProjectForKind(kind)
	if !ok {
		return nil
	}
	project := p.ID
	if id != "" {
		if b, ok := s.builder.Find(id); ok && b.Status == builder.BuildStatusSuccess && b.ProjectID() == project {
			return b
		}
	}
	if ranked := s.builder.SuccessfulBuildsRankedFor(project); len(ranked) > 0 {
		return &ranked[0]
	}
	return nil
}

// resolveForkBuild returns the build a fork instance runs: the named
// build when it is a successful build of the fork, otherwise the newest
// successful one. Nil when there is none.
func (s *Server) resolveForkBuild(forkID, id string) *builder.BuildResult {
	if p, ok := builder.FindProject(forkID); !ok || !p.IsFork() {
		return nil
	}
	if id != "" {
		if b, ok := s.builder.Find(id); ok && b.Status == builder.BuildStatusSuccess && b.ProjectID() == forkID {
			return b
		}
	}
	if ranked := s.builder.SuccessfulBuildsRankedFor(forkID); len(ranked) > 0 {
		return &ranked[0]
	}
	return nil
}

// buildInfoData feeds the build_info partial.
type buildInfoData struct {
	buildRow
	ShortSHA string
	// FlagsText is the cmake flags one per line, sorted; empty for a
	// build from before flags were recorded.
	FlagsText string
	// PatchReleaseURL is a patched build's release page, ShortBase its
	// upstream base; both empty on an unpatched build.
	PatchReleaseURL, ShortBase string
}

func (s *Server) handleBuildInfo(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	found, ok := s.builder.Find(id)
	if !ok {
		http.Error(w, "build not found", http.StatusNotFound)
		return
	}

	if !isHTMX(r) {
		respondJSON(w, found)
		return
	}

	var flags []string
	for _, k := range slices.Sorted(maps.Keys(found.CMakeFlags)) {
		flags = append(flags, "-D"+k+"="+found.CMakeFlags[k])
	}
	respondHTML(w)
	data := buildInfoData{
		buildRow:  s.buildRowFor(found),
		ShortSHA:  safeShortSHA(found.GitSHA),
		FlagsText: strings.Join(flags, " \\\n  "),
		ShortBase: safeShortSHA(found.PatchBase),
	}
	if set, ok := patchsets.Find(found.PatchSet); ok {
		data.PatchReleaseURL = set.Repo + "/releases/tag/" + found.PatchRelease
	}
	s.renderPartial(w, "build_info", data)
}

func safeShortSHA(sha string) string {
	if len(sha) < 7 {
		return sha
	}
	return sha[:7]
}

func (s *Server) handleDeleteBuild(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.builder.Delete(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// htmx: return empty to remove the row
	if isHTMX(r) {
		w.WriteHeader(http.StatusOK)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// renderPartial executes a partial template, writing directly to w.
func (s *Server) renderPartial(w http.ResponseWriter, name string, data any) {
	// Partials are shared across all page clones. Try each until one
	// succeeds. Buffer output to avoid writing partial results on error.
	var lastErr error
	for _, tmpl := range s.pages {
		var buf bytes.Buffer
		if err := tmpl.ExecuteTemplate(&buf, name, data); err == nil {
			buf.WriteTo(w)
			return
		} else {
			lastErr = err
		}
	}
	if lastErr != nil {
		slog.Error("renderPartial failed on all page templates", "name", name, "error", lastErr)
	}
}

// tensorSplitRefused reports whether the build the router runs refuses to
// load a model of this architecture with --split-mode tensor. False when
// that is not known: no build, a build without a recorded list, or a
// model whose architecture was not read.
func (s *Server) tensorSplitRefused(arch string) bool {
	b := s.resolveActiveBuild()
	return b != nil && arch != "" && slices.Contains(b.NoTensorSplitArchs, arch)
}

// supportedArchs is the set of model architectures the build the router
// runs can load. known is false when that is not known (no build, or one
// without a recorded list), and then nothing should be judged by it: a
// missing list must never count against a model.
func (s *Server) supportedArchs() (archs map[string]bool, known bool) {
	b := s.resolveActiveBuild()
	if b == nil || len(b.Archs) == 0 {
		return nil, false
	}
	archs = make(map[string]bool, len(b.Archs))
	for _, a := range b.Archs {
		archs[a] = true
	}
	return archs, true
}

// patchSetChoicePrefix starts a Builds page project value that names a
// curated patch set ("patchset:rdna-boosts"). The build is a llama.cpp
// build with the set applied.
const patchSetChoicePrefix = "patchset:"

// patchSetChoice returns the curated patch set a project value names.
func patchSetChoice(value string) (patchsets.PatchSet, bool) {
	id, ok := strings.CutPrefix(value, patchSetChoicePrefix)
	if !ok {
		return patchsets.PatchSet{}, false
	}
	return patchsets.Find(id)
}

// releaseLabel is a patch set release's short name: "r206" for
// v16-6050b467c-r206, or the tag itself when it has no number.
func releaseLabel(tag string) string {
	if n, ok := patchsets.ReleaseNumber(tag); ok {
		return fmt.Sprintf("r%d", n)
	}
	return tag
}

// patchLabel names a patched build's patch set and release ("RDNA boosts
// r206") and reports whether that release is the tested one. Every page
// that lists builds uses it, so they name a patched build the same way.
// Empty on an unpatched build.
func patchLabel(b *builder.BuildResult) (label string, tested bool) {
	if b == nil || b.PatchSet == "" {
		return "", false
	}
	set, ok := patchsets.Find(b.PatchSet)
	if !ok {
		return b.PatchSet + " " + releaseLabel(b.PatchRelease), false
	}
	return set.Name + " " + releaseLabel(b.PatchRelease), b.PatchRelease == set.Pinned.Tag
}
