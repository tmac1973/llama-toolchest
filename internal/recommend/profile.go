// Package recommend builds the list of GGUF models from HuggingFace that
// should run well on this machine, for the Download Models tab's
// "Find recommended models". Every fit figure comes from models.PlanFit,
// the planner Autoconfigure uses, so the two never disagree.
//
// See plan/recommend-models for the design.
package recommend

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Profile is the machine a list is worked out for.
type Profile struct {
	Hardware models.Hardware
	// BuildID is the llama.cpp build the router runs, and Archs the model
	// architectures it can load. ArchsKnown is false when the list is not
	// known; architectures are then not checked at all.
	BuildID    string
	Archs      map[string]bool
	ArchsKnown bool
}

// Key identifies the profile: a pool built for one key is reused for as
// long as the key stays the same.
func (p Profile) Key() string {
	h := sha256.New()
	fmt.Fprintf(h, "%d|%d|", p.Hardware.LogicalCores, p.Hardware.RAMTotalMiB)
	for _, g := range p.Hardware.GPUs {
		fmt.Fprintf(h, "%s|%d|%v;", g.Name, g.VRAMTotalMiB, g.IsIGPU)
	}
	fmt.Fprintf(h, "|%s|%v|", p.BuildID, p.ArchsKnown)
	archs := make([]string, 0, len(p.Archs))
	for a := range p.Archs {
		archs = append(archs, a)
	}
	slices.Sort(archs)
	fmt.Fprint(h, strings.Join(archs, ","))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// supports reports whether the active build can load arch, or true when
// that is not known.
func (p Profile) supports(arch string) bool {
	return !p.ArchsKnown || p.Archs[arch]
}

//go:embed publishers.conf
var publishersConf string

// trustedPublishers is publishers.conf, in order of preference.
var trustedPublishers = func() []string {
	var out []string
	for _, line := range strings.Split(publishersConf, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			out = append(out, line)
		}
	}
	return out
}()

// publisherRank is a trusted publisher's place in the list, or -1.
func publisherRank(author string) int {
	for i, p := range trustedPublishers {
		if strings.EqualFold(p, author) {
			return i
		}
	}
	return -1
}
