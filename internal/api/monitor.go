package api

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/tmlabonte/llamactl/internal/monitor"
)

func (s *Server) handleMonitorStream(w http.ResponseWriter, r *http.Request) {
	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ch := s.monitor.Subscribe()
	defer s.monitor.Unsubscribe(ch)

	// Send current state immediately
	data, _ := json.Marshal(s.monitor.Current())
	sse.SendEvent("metrics", string(data))

	for {
		select {
		case m, ok := <-ch:
			if !ok {
				return
			}
			data, _ := json.Marshal(m)
			sse.SendEvent("metrics", string(data))
		case <-r.Context().Done():
			return
		}
	}
}

func (s *Server) handleMonitorStatus(w http.ResponseWriter, r *http.Request) {
	m := s.monitor.Current()

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "monitor_bar", monitorBarData(m))
		return
	}

	respondJSON(w, m)
}

// monitorBarGPU holds precomputed GPU display values for the template.
type monitorBarGPU struct {
	Index       int
	Name        string
	UtilPercent int
	VRAMPercent int
	VRAMGB      string
	Details     string
}

// monitorBarData prepares monitor metrics for the monitor_bar template.
func monitorBarData(m monitor.Metrics) struct {
	GPUs       []monitorBarGPU
	CPUPercent float64
	RAMPercent int
	RAMGB      string
} {
	gpus := make([]monitorBarGPU, len(m.GPU))
	for i, gpu := range m.GPU {
		vramPct := 0
		if gpu.VRAMTotalMB > 0 {
			vramPct = gpu.VRAMUsedMB * 100 / gpu.VRAMTotalMB
		}
		details := fmt.Sprintf("%d°C", gpu.TempC)
		if gpu.PowerW > 0 {
			details += fmt.Sprintf(" · %.0fW", gpu.PowerW)
		}
		gpus[i] = monitorBarGPU{
			Index:       gpu.Index,
			Name:        gpu.Name,
			UtilPercent: gpu.UtilPercent,
			VRAMPercent: vramPct,
			VRAMGB:      fmt.Sprintf("%.1f/%.1fGB", float64(gpu.VRAMUsedMB)/1024, float64(gpu.VRAMTotalMB)/1024),
			Details:     details,
		}
	}
	ramPct := 0
	if m.Memory.TotalMB > 0 {
		ramPct = m.Memory.UsedMB * 100 / m.Memory.TotalMB
	}
	return struct {
		GPUs       []monitorBarGPU
		CPUPercent float64
		RAMPercent int
		RAMGB      string
	}{
		GPUs:       gpus,
		CPUPercent: m.CPU.UsagePercent,
		RAMPercent: ramPct,
		RAMGB:      fmt.Sprintf("%.1f/%.1fGB", float64(m.Memory.UsedMB)/1024, float64(m.Memory.TotalMB)/1024),
	}
}
