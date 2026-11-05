package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func threeCards() models.Hardware {
	return models.Hardware{LogicalCores: 16, GPUs: []models.GPUSpec{
		{Index: 0, Name: "RTX A4000", VRAMTotalMiB: 16376, ReservedMiB: 6144},
		{Index: 1, Name: "RTX A4000", VRAMTotalMiB: 16376},
		{Index: 2, Name: "RTX A4000", VRAMTotalMiB: 16376},
		{Index: 3, Name: "iGPU", VRAMTotalMiB: 2048, IsIGPU: true},
	}}
}

// The dialog offers each dedicated card, all checked, and says what is
// reserved on each for generation. One card is no choice, and no list.
func TestAutoconfigDialogOffersTheGPUs(t *testing.T) {
	s := newHelperServer(t)
	hw := threeCards()
	s.testHardware = &hw
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil)
	for _, want := range []string{"Which GPUs can the model use?", `name="gpu_choice" value="1"`,
		`name="gpu" value="0" checked`, `name="gpu" value="2" checked`, "6.0 GiB reserved"} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog missing %q", want)
		}
	}
	if strings.Contains(out, `name="gpu" value="3"`) {
		t.Error("the integrated GPU is offered")
	}

	one := models.Hardware{GPUs: []models.GPUSpec{{Index: 0, VRAMTotalMiB: 16376}}}
	s.testHardware = &one
	if out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil); strings.Contains(out, "Which GPUs") {
		t.Error("the GPU list is shown on a machine with one GPU")
	}
}

// Autoconfigure plans on the cards as they are with nothing of ours
// loaded, where its test load runs: what a loaded model holds is not
// counted as other programs' memory.
func TestAutoconfigPlansOnTheIdleCards(t *testing.T) {
	s := newHelperServer(t)
	hw := threeCards()
	hw.GPUs[2].OtherUsedMiB = 15800 // a loaded model, as the live reading counts it
	s.testHardware = &hw
	s.otherVRAM.idleOther = []int{0, 0, 0, 0}
	if got := s.autoconfigHardware().GPUs[2].OtherUsedMiB; got != 0 {
		t.Errorf("planned with %d MiB of other programs on GPU 2, want the idle reading's 0", got)
	}
}

func TestChosenGPUs(t *testing.T) {
	hw := threeCards()
	read := func(form url.Values) ([]int, bool) {
		req := httptest.NewRequest("POST", "/", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		return chosenGPUs(req, hw)
	}
	cases := []struct {
		name string
		form url.Values
		want []int
		ok   bool
	}{
		{"no choice offered", url.Values{}, nil, true},
		{"some cards", url.Values{"gpu_choice": {"1"}, "gpu": {"2", "1"}}, []int{1, 2}, true},
		{"every card", url.Values{"gpu_choice": {"1"}, "gpu": {"0", "1", "2"}}, nil, true},
		{"none checked", url.Values{"gpu_choice": {"1"}}, nil, false},
		{"only cards not offered", url.Values{"gpu_choice": {"1"}, "gpu": {"3", "9", "x"}}, nil, false},
	}
	for _, c := range cases {
		got, ok := read(c.form)
		if ok != c.ok || len(got) != len(c.want) || (len(got) > 0 && (got[0] != c.want[0] || got[len(got)-1] != c.want[len(c.want)-1])) {
			t.Errorf("%s: got %v %v, want %v %v", c.name, got, ok, c.want, c.ok)
		}
	}
}

func TestAutoconfigStartNeedsAGPU(t *testing.T) {
	s := newHelperServer(t)
	hw := threeCards()
	s.testHardware = &hw
	out := s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig",
		url.Values{"context_class": {"short"}, "gpu_choice": {"1"}})
	if !strings.Contains(out, "Choose at least one GPU") {
		t.Errorf("a run with no GPU checked was not refused:\n%s", out)
	}
	if s.autoconf.run != nil {
		t.Error("a run started with no GPU checked")
	}
}

// Autotune counts the cards the profile uses, not every card in the
// machine: a profile fitted on one GPU has no split to measure.
func TestAutotuneHardwareCountsTheProfileGPUs(t *testing.T) {
	s := newHelperServer(t)
	hw := threeCards()
	s.testHardware = &hw
	cases := []struct {
		cfg  models.ModelConfig
		want int
	}{
		{models.ModelConfig{}, 3},
		{models.ModelConfig{GPUAssign: "all"}, 3},
		{models.ModelConfig{GPUAssign: "1-2"}, 2},
		{models.ModelConfig{GPUAssign: "2"}, 1},
		{models.ModelConfig{GPUAssign: "custom", TensorSplit: "0,3,2,0"}, 2},
		{models.ModelConfig{GPUAssign: "2-3"}, 1}, // the iGPU does not count
	}
	for _, c := range cases {
		if got, _ := s.autotuneHardware(c.cfg); got != c.want {
			t.Errorf("%q split %q: %d cards, want %d", c.cfg.GPUAssign, c.cfg.TensorSplit, got, c.want)
		}
	}
}

// The dialog offers to plan on the whole GPUs only when auto-evict may
// take some reservation back (models.GPUSpec.EvictableMiB).
func TestAutoconfigDialogOffersTheWholeGPUs(t *testing.T) {
	s := newHelperServer(t)
	hw := threeCards()
	s.testHardware = &hw
	if out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil); strings.Contains(out, `name="whole_gpus"`) {
		t.Error("offered with nothing evictable")
	}
	hw.GPUs[0].EvictableMiB = 6144
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil)
	for _, want := range []string{`name="whole_gpus" value="1"`, "Use the whole GPUs, split evenly", "(6.0 GiB reserved)"} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog missing %q", want)
		}
	}
}
