package api

import (
	"fmt"
	"maps"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// maxVRAMMarginGiB is the largest margin the settings take: more than
// any card has, so a typo is refused rather than saved.
const maxVRAMMarginGiB = 1024

// vramMarginView is what the Settings page's GPU memory margin card
// (vram_margin_card partial) shows.
type vramMarginView struct {
	// GiB is the margin for every GPU, as the input shows it.
	GiB  string
	GPUs []vramMarginGPU
}

// vramMarginGPU is one detected GPU's row. GiB is its own margin, or ""
// when it uses the margin for every GPU.
type vramMarginGPU struct {
	Index int
	Name  string
	Total string
	GiB   string
	IGPU  bool
}

// gibText is mib in GiB for an input field: at most two decimals, no
// trailing zeros.
func gibText(mib int) string {
	return strconv.FormatFloat(math.Round(float64(mib)/1024*100)/100, 'f', -1, 64)
}

// vramMarginInfo builds the card's view from the settings and the GPUs
// the system monitor reports.
func (s *Server) vramMarginInfo() vramMarginView {
	cfg := s.cfgCopy()
	v := vramMarginView{GiB: gibText(max(cfg.VRAMMarginMiB, models.WorkingMemoryMiB))}
	for _, g := range s.hardware().GPUs {
		row := vramMarginGPU{Index: g.Index, Name: g.Name, Total: mibText(g.VRAMTotalMiB), IGPU: g.IsIGPU}
		if mib, ok := cfg.VRAMMarginByGPU[g.Index]; ok {
			row.GiB = gibText(max(mib, models.WorkingMemoryMiB))
		}
		v.GPUs = append(v.GPUs, row)
	}
	return v
}

// handleVRAMMarginCard renders the Settings page's GPU memory margin card.
func (s *Server) handleVRAMMarginCard(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	s.renderPartial(w, "vram_margin_card", s.vramMarginInfo())
}

// parseVRAMMarginGiB reads a margin in GiB as MiB. Blank is 0, the
// default, and ok is false for a field left blank. Less than 1 GiB is
// refused: llama.cpp needs that much for its working memory
// (models.WorkingMemoryMiB).
func parseVRAMMarginGiB(field, text string) (mib int, ok bool, err error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0, false, nil
	}
	v, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsNaN(v) || v > maxVRAMMarginGiB {
		return 0, false, fmt.Errorf("%s must be a number of GiB from 1 to %d, not %q", field, maxVRAMMarginGiB, text)
	}
	if mib = int(math.Round(v * 1024)); mib < models.WorkingMemoryMiB {
		return 0, false, fmt.Errorf("%s must be at least 1 GiB, not %q: llama.cpp needs that much free in real chats, and so does the driver", field, text)
	}
	return mib, true, nil
}

// formVRAMMargin reads the card's form: vram_margin_gib for every GPU
// and vram_margin_gpu_<index> for each GPU row, where blank means the
// GPU uses the margin for every GPU. Rows of GPUs the form does not show
// keep their saved value.
func formVRAMMargin(r *http.Request, saved map[int]int) (all int, byGPU map[int]int, err error) {
	all, _, err = parseVRAMMarginGiB("The margin for every GPU", r.FormValue("vram_margin_gib"))
	if err != nil {
		return 0, nil, err
	}
	byGPU = maps.Clone(saved)
	if byGPU == nil {
		byGPU = map[int]int{}
	}
	for key := range r.Form {
		idx, found := strings.CutPrefix(key, "vram_margin_gpu_")
		if !found {
			continue
		}
		g, err := strconv.Atoi(idx)
		if err != nil || g < 0 {
			return 0, nil, fmt.Errorf("unknown GPU field %q", key)
		}
		mib, ok, err := parseVRAMMarginGiB(fmt.Sprintf("The margin for GPU %d", g), r.FormValue(key))
		if err != nil {
			return 0, nil, err
		}
		if ok {
			byGPU[g] = mib
		} else {
			delete(byGPU, g)
		}
	}
	if len(byGPU) == 0 {
		byGPU = nil
	}
	return all, byGPU, nil
}

// validVRAMMarginMiB checks a margin in MiB from the JSON API or a
// backup: 0 is the default (1 GiB), and anything else must be from
// models.WorkingMemoryMiB up.
func validVRAMMarginMiB(field string, mib int) error {
	if mib != 0 && (mib < models.WorkingMemoryMiB || mib > maxVRAMMarginGiB*1024) {
		return fmt.Errorf("%s must be 0 (the default, 1 GiB) or from %d to %d MiB, not %d",
			field, models.WorkingMemoryMiB, maxVRAMMarginGiB*1024, mib)
	}
	return nil
}
