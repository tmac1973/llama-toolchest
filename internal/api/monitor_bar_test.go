package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// The sidebar's line under each GPU: fan duty when the card reports it,
// and the RPM when it reports that too. NVIDIA reports no RPM. A card
// whose fans have stopped says so; a card without a fan says nothing.
func TestMonitorBarShowsTheFan(t *testing.T) {
	m := monitor.Metrics{GPU: []monitor.GPUInfo{
		{Index: 0, TempC: 38, PowerW: 21, HasFan: true, FanPercent: 20, FanRPM: 1083},
		{Index: 1, TempC: 47, PowerW: 40, HasFan: true, FanPercent: 45},
		{Index: 2, TempC: 47, PowerW: 37, HasFan: true},
		{Index: 3, TempC: 41},
	}}
	want := []string{"38°C · 21W · Fan\u00a020% · 1083\u00a0RPM", "47°C · 40W · Fan\u00a045%", "47°C · 37W · Fan\u00a0stopped", "41°C"}
	for i, g := range monitorBarData(m).GPUs {
		if g.Details != want[i] {
			t.Errorf("GPU %d details = %q, want %q", i, g.Details, want[i])
		}
	}
}
