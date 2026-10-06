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
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func (s *Server) handleListBackends(w http.ResponseWriter, r *http.Request) {
	backends := builder.DetectBackends()
	respondJSON(w, backends)
}

func (s *Server) handleProfileOptions(w http.ResponseWriter, r *http.Request) {
	profile := r.URL.Query().Get("profile")
	options := builder.ProfileOptions(profile)

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

		// Show effective cmake flags with current toggle states
		if prof, ok := builder.FindProfile(profile); ok {
			flags := effectiveCMakeFlags(prof, options, effectiveOverrides)
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
}

// buildOptionRow is one toggle: the option and whether it is on.
type buildOptionRow struct {
	builder.BuildOption
	Checked bool
}

func effectiveCMakeFlags(prof builder.BuildProfile, options []builder.BuildOption, overrides map[string]bool) string {
	flags := make(map[string]string)
	for k, v := range prof.CMakeFlags {
		flags[k] = v
	}
	builder.ApplyOptionOverrides(flags, options, overrides)
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

	var refs []string
	var err error
	if refresh {
		refs, err = s.builder.FetchRefs()
		if err != nil {
			// Return cached if fetch fails
			refs = s.builder.CachedRefs()
		}
	} else {
		refs = s.builder.CachedRefs()
	}

	if isHTMX(r) {
		respondHTML(w)
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

func (s *Server) handleListBuilds(w http.ResponseWriter, r *http.Request) {
	builds := s.builder.List()

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
	ID, Profile, GitRef, Tag string
}

func (s *Server) handleTriggerBuild(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Profile string `json:"profile"`
		GitRef  string `json:"git_ref"`
		Tag     string `json:"tag"`
		Force   bool   `json:"force"`
	}

	// Support both JSON and form-encoded
	if isJSONRequest(r) {
		if !decodeJSONBody(w, r, &req) {
			return
		}
	} else {
		r.ParseForm()
		req.Profile = r.FormValue("profile")
		req.GitRef = r.FormValue("git_ref")
		req.Tag = r.FormValue("tag")
		req.Force = r.FormValue("force") == "1"
	}

	// Collect option overrides and extra cmake flags from form
	var optionOverrides map[string]bool
	var extraCMake string
	if !isJSONRequest(r) {
		options := builder.ProfileOptions(req.Profile)
		optionOverrides = make(map[string]bool)
		for _, opt := range options {
			// Checkboxes only send a value when checked
			optionOverrides[opt.Flag] = r.FormValue("opt_"+opt.Flag) == "on"
		}
		extraCMake = r.FormValue("extra_cmake")
	}

	// Use background context — the build must outlive the HTTP request.
	result, err := s.builder.Build(context.Background(), req.Profile, req.GitRef, req.Tag, req.Force, optionOverrides, extraCMake)
	if err != nil {
		var dup *builder.DuplicateBuildError
		if errors.As(err, &dup) {
			if isHTMX(r) {
				respondHTML(w)
				s.renderPartial(w, "build_duplicate", buildDuplicateData{
					ID: dup.ID, Profile: req.Profile, GitRef: req.GitRef, Tag: req.Tag,
				})
				return
			}
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		slog.Error("build failed to start", "profile", req.Profile, "git_ref", req.GitRef, "error", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Return the log streaming partial for htmx to swap in
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "build_log", result)
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
	lastID := s.builder.LastBuildID()
	if lastID == "" {
		return
	}

	status := s.builder.BuildStatus(lastID)
	if status == "" {
		return
	}

	// Show the log panel for running or recently completed builds
	s.renderPartial(w, "build_log", &builder.BuildResult{ID: lastID, Status: status})
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
	current := s.currentBuildEnvFor(buildBackend(b))
	switch {
	case b.BuiltAgainst == "":
		row.BuiltAgainstTitle = notRecordedTitle
	case builder.StampMismatch(b.BuiltAgainst, current):
		row.BuiltAgainstText = b.BuiltAgainst
		row.Mismatch = true
		row.BuiltAgainstTitle = fmt.Sprintf(
			"Built against %s; this container runs %s. A build is linked against the libraries of the image it was made in, so it may fail to load here — rebuild it if the server does not start. Nothing has been deleted; it is still here if you switch back to %s.",
			b.BuiltAgainst, current, b.BuiltAgainst)
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
// needs to balance expert offload across cards.
func (s *Server) targetFor(b *builder.BuildResult) models.Target {
	t := buildTarget(b)
	t.GPUMiB = models.GPUMiB(s.hardware())
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
		if b, ok := s.builder.Find(id); ok && b.Status == builder.BuildStatusSuccess {
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
	ranked := s.builder.SuccessfulBuildsRanked()
	for i := range ranked {
		if !builder.StampMismatch(ranked[i].BuiltAgainst, s.currentBuildEnvFor(buildBackend(&ranked[i]))) {
			res := ranked[i]
			return &res
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

// buildInfoData feeds the build_info partial.
type buildInfoData struct {
	buildRow
	ShortSHA string
	// FlagsText is the cmake flags one per line, sorted; empty for a
	// build from before flags were recorded.
	FlagsText string
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
	s.renderPartial(w, "build_info", buildInfoData{
		buildRow:  s.buildRowFor(found),
		ShortSHA:  safeShortSHA(found.GitSHA),
		FlagsText: strings.Join(flags, " \\\n  "),
	})
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
