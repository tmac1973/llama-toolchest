package models

import (
	"cmp"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// sizeToken matches a parameter count in a repository name: "-9B",
// "-0.8B", "-500M". Everything from it on is dropped to get the family
// name: "Qwen3.5-9B-Instruct" → "Qwen3.5".
var sizeToken = regexp.MustCompile(`-(\d+(?:\.\d+)?)([BbMm])\b`)

// FamilyAndSize splits a repository name (the part after the owner) into
// its family and parameter count. ok is false when the name has no size.
func FamilyAndSize(name string) (family string, params float64, ok bool) {
	loc := sizeToken.FindStringSubmatchIndex(name)
	if loc == nil {
		return "", 0, false
	}
	n, err := strconv.ParseFloat(name[loc[2]:loc[3]], 64)
	if err != nil {
		return "", 0, false
	}
	switch strings.ToUpper(name[loc[4]:loc[5]]) {
	case "B":
		n *= 1e9
	case "M":
		n *= 1e6
	}
	return name[:loc[0]], n, true
}

// lineage is what a draft model is compared on: the publisher of the
// model it was made from, when the file records one, and its family.
type lineage struct {
	publisher string
	family    string
}

func lineageOf(modelID, baseRepo string) lineage {
	var l lineage
	name := baseRepo
	if baseRepo != "" {
		if i := strings.IndexByte(baseRepo, '/'); i >= 0 {
			l.publisher = strings.ToLower(baseRepo[:i])
			name = baseRepo[i+1:]
		}
	} else {
		// No base recorded: the model's own repository, whose owner is as
		// likely a quantizer as the publisher, so it names only a family.
		name = modelID
		if i := strings.IndexByte(name, '/'); i >= 0 {
			name = name[i+1:]
		}
		name = strings.TrimSuffix(strings.TrimSuffix(name, "-GGUF"), "-gguf")
	}
	if fam, _, ok := FamilyAndSize(name); ok {
		l.family = strings.ToLower(fam)
	}
	return l
}

// RelatedDrafts narrows draft-model candidates to the ones closest to m,
// smallest first. The registry matches a draft model on architecture and
// size alone, which a finetune for some other purpose passes as easily as
// the publisher's own small model: a 3D-code finetune of Qwen3.5-4B was
// measured as the drafter for a reasoning finetune of Qwen3.8-27B, where
// it has little chance of guessing the larger model's tokens. A candidate
// from the same publisher as m's base model counts, and so does one of the
// same family; the ones that share both come first. Only the closest
// match is kept, and when nothing matches the list is kept whole, since a
// draft model of the same architecture may still pay off.
func RelatedDrafts(m *Model, cands []DraftCandidate) []DraftCandidate {
	cands = slices.Clone(cands)
	slices.SortStableFunc(cands, func(a, b DraftCandidate) int {
		if c := cmp.Compare(a.SizeGB, b.SizeGB); c != 0 {
			return c
		}
		return strings.Compare(a.ID, b.ID)
	})
	if m == nil {
		return cands
	}
	target := lineageOf(m.ModelID, m.BaseModelRepo)
	score := func(c DraftCandidate) int {
		l := lineageOf(c.ModelID, c.BaseModelRepo)
		s := 0
		if target.publisher != "" && l.publisher == target.publisher {
			s += 2
		}
		if target.family != "" && l.family == target.family {
			s++
		}
		return s
	}
	best := 0
	for _, c := range cands {
		best = max(best, score(c))
	}
	if best == 0 {
		return cands
	}
	return slices.DeleteFunc(cands, func(c DraftCandidate) bool { return score(c) < best })
}
