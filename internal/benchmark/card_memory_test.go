package benchmark

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

func threeCards(used ...int) monitor.Metrics {
	var m monitor.Metrics
	for i, u := range used {
		m.GPU = append(m.GPU, monitor.GPUInfo{Index: i, VRAMUsedMB: u, VRAMTotalMB: 16376})
	}
	return m
}

func TestCardsInUse(t *testing.T) {
	m := threeCards(15879, 14649, 14949)
	m.GPU = append(m.GPU, monitor.GPUInfo{Index: 3, VRAMUsedMB: 1000, VRAMTotalMB: 2048, IsIGPU: true})

	all := CardsInUse(m, ConfigSnapshot{GPULayers: 999, GPUAssign: "all"})
	if len(all) != 3 {
		t.Errorf("all cards = %+v, want the three discrete GPUs", all)
	}
	one := CardsInUse(m, ConfigSnapshot{GPULayers: 999, GPUAssign: "1"})
	if len(one) != 1 || one[0].Index != 1 {
		t.Errorf("GPU 1 only = %+v", one)
	}
	split := CardsInUse(m, ConfigSnapshot{GPULayers: 999, TensorSplit: "0,1,1"})
	if len(split) != 2 || split[0].Index != 1 || split[1].Index != 2 {
		t.Errorf("tensor split 0,1,1 = %+v, want GPUs 1 and 2", split)
	}
	if cpu := CardsInUse(m, ConfigSnapshot{GPULayers: 0}); len(cpu) != 0 {
		t.Errorf("a model kept on the CPU uses no GPU, got %+v", cpu)
	}
}

// The case this was written for: autotune's winner left 497 MiB on GPU 0
// and crashed on the first real chat.
func TestMemoryShortfall(t *testing.T) {
	run := BenchmarkRun{Cards: CardsInUse(threeCards(15879, 14649, 14949), ConfigSnapshot{GPULayers: 999})}
	c, ok := run.TightestCard()
	if !ok || c.Index != 0 || c.FreeMiB() != 497 {
		t.Fatalf("tightest card = %+v (%v), want GPU 0 with 497 MiB free", c, ok)
	}
	msg := run.MemoryShortfall()
	if !strings.Contains(msg, "497 MiB free on GPU 0") || !strings.Contains(msg, "1024 MiB") {
		t.Errorf("shortfall = %q", msg)
	}

	// After lowering the prompt batch it had 1,349 MiB left, and ran.
	run.Cards = CardsInUse(threeCards(15027, 13797, 14099), ConfigSnapshot{GPULayers: 999})
	if msg := run.MemoryShortfall(); msg != "" {
		t.Errorf("shortfall = %q, want none with 1,349 MiB free", msg)
	}
	// A run without card readings is not judged.
	if msg := (BenchmarkRun{}).MemoryShortfall(); msg != "" {
		t.Errorf("shortfall without cards = %q", msg)
	}
}

// A completed run records what was left on each card it used.
func TestRunRecordsCardMemory(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999, ContextSize: 8192},
		metrics: threeCards(12000, 13000)}
	done, store := runJob(t, oneCellJob(nil), env)
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("runs = %d", len(runs))
	}
	if len(runs[0].Cards) != 2 || runs[0].Cards[1].UsedMiB != 13000 {
		t.Errorf("cards = %+v", runs[0].Cards)
	}
}
