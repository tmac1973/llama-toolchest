package patchsets

import (
	"regexp"
	"testing"
)

func TestCuratedPatchSetsAreComplete(t *testing.T) {
	idOK := regexp.MustCompile(`^[a-z0-9-]+$`)
	shaOK := regexp.MustCompile(`^[0-9a-f]{40}$`)
	seen := map[string]bool{}
	short := map[string]bool{}
	for _, p := range Curated() {
		if !idOK.MatchString(p.ID) || seen[p.ID] {
			t.Errorf("patch set ID %q is not a unique lowercase slug", p.ID)
		}
		seen[p.ID] = true
		if !idOK.MatchString(p.ShortName) || short[p.ShortName] {
			t.Errorf("%s: short name %q is not a unique lowercase slug", p.ID, p.ShortName)
		}
		short[p.ShortName] = true
		if !shaOK.MatchString(p.Pinned.Base) || !shaOK.MatchString(p.Pinned.Tree) {
			t.Errorf("%s: base %q and tree %q must be full SHAs", p.ID, p.Pinned.Base, p.Pinned.Tree)
		}
		if p.Pinned.NBlocks <= 0 {
			t.Errorf("%s: pinned release has no patches", p.ID)
		}
		if _, ok := ReleaseNumber(p.Pinned.Tag); !ok {
			t.Errorf("%s: pinned tag %q is not a release tag", p.ID, p.Pinned.Tag)
		}
		if len(p.Backends) == 0 || p.Repo == "" || p.Name == "" {
			t.Errorf("%s: needs a name, a repo and backends", p.ID)
		}
	}
}

func TestFind(t *testing.T) {
	p, ok := Find("rdna-boosts")
	if !ok || p.ShortName != "rdna" {
		t.Fatalf("Find(rdna-boosts) = %+v, %v", p, ok)
	}
	if _, ok := Find("nope"); ok {
		t.Error("Find(nope) found a patch set")
	}
}

func TestReleaseNumber(t *testing.T) {
	for tag, want := range map[string]int{
		"v16-6050b467c-r206": 206,
		"v1-abc-r1":          1,
	} {
		if n, ok := ReleaseNumber(tag); !ok || n != want {
			t.Errorf("ReleaseNumber(%q) = %d, %v; want %d", tag, n, ok, want)
		}
	}
	for _, tag := range []string{"", "b11887", "v16-6050b467c", "v16-XYZ-r2", "v16-6050b467c-r206-beta"} {
		if _, ok := ReleaseNumber(tag); ok {
			t.Errorf("ReleaseNumber(%q) parsed", tag)
		}
	}
}
