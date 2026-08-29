package memreport

import (
	"math"
	"strings"
	"testing"
)

// Verbatim lines from a Qwen3.8-27B load on the ROCm box, captured at
// LLAMA_ARG_LOG_VERBOSITY=4 through /api/service/logs. Kept exactly as
// they arrived — SSE "data: " prefix, router pid prefix, and llama.cpp's
// ragged column alignment — because every one of those is something the
// parser has to survive in production.
const realLog = `data: [33899] 0.14.275.751 I load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
data: [33899] 0.14.275.753 I load_tensors:       Meta() model buffer size =  7252.51 MiB
data: [33899] 0.16.135.888 I llama_kv_cache: size = 16384.00 MiB (262144 cells,  16 layers,  4/1 seqs), K (f16): 8192.00 MiB, V (f16): 8192.00 MiB
data: [33899] 0.16.162.940 I llama_memory_recurrent:     Meta() RS buffer size =   448.88 MiB
data: [33899] 0.16.260.662 I sched_reserve:     Meta() compute buffer size =   384.95 MiB
data: [33899] 0.16.260.669 I sched_reserve:  ROCm_Host compute buffer size =   276.02 MiB
data: [33899] 0.17.752.569 I llama_context:  ROCm_Host  output buffer size =     3.79 MiB
data: [33899] 0.17.783.489 I llama_kv_cache:     Meta() KV buffer size =   256.00 MiB
data: [33899] 0.17.873.961 I sched_reserve:     Meta() compute buffer size =   324.02 MiB
data: [33899] 0.17.873.969 I sched_reserve:  ROCm_Host compute buffer size =   276.02 MiB
data: [33899] 0.02.652.943 I srv  llama_server: model loaded
data: [33899] 0.00.688.812 D load_tensors: layer   1 assigned to device Meta(), is_swa = 0`

func parseAll(t *testing.T, log string) Report {
	t.Helper()
	var r Report
	for _, l := range strings.Split(log, "\n") {
		r.Add(l)
	}
	return r
}

func close(a, b float64) bool { return math.Abs(a-b) < 0.005 }

func TestParsesRealLoad(t *testing.T) {
	r := parseAll(t, realLog)
	if len(r.Entries) != 9 {
		t.Fatalf("got %d entries, want 9:\n%+v", len(r.Entries), r.Entries)
	}
	byKind := r.ByKind()
	for _, tt := range []struct {
		kind Kind
		want float64
	}{
		{KindModel, 1288.28 + 7252.51},
		{KindKV, 256.00}, // the "size =" summary is not a buffer line
		{KindRecurrent, 448.88},
		{KindCompute, 384.95 + 276.02 + 324.02 + 276.02},
		{KindOutput, 3.79},
	} {
		if !close(byKind[tt.kind], tt.want) {
			t.Errorf("%s = %.2f, want %.2f", tt.kind, byKind[tt.kind], tt.want)
		}
	}
}

// The "size = N MiB (cells, layers...)" summary restates a total that the
// per-device buffer lines already carry. Counting it would double the KV
// figure — 16384 MiB against a real 256 in this load.
func TestSizeSummaryIsNotABuffer(t *testing.T) {
	line := `[33899] 0.16.135.888 I llama_kv_cache: size = 16384.00 MiB (262144 cells,  16 layers,  4/1 seqs), K (f16): 8192.00 MiB, V (f16): 8192.00 MiB`
	if e, ok := ParseLine(line); ok {
		t.Errorf("parsed a size summary as a buffer: %+v", e)
	}
}

func TestHostAndDeviceClassification(t *testing.T) {
	for _, tt := range []struct {
		device string
		onGPU  bool
	}{
		{"Meta()", true}, // llama.cpp's aggregate when one device spans cards
		{"ROCm0", true},
		{"CUDA1", true},
		{"Vulkan0", true},
		{"CPU", false},
		{"CPU_Mapped", false}, // weights left in the mmap
		{"ROCm_Host", false},  // pinned staging
		{"CUDA_Host", false},
	} {
		if got := (Entry{Device: tt.device}).OnGPU(); got != tt.onGPU {
			t.Errorf("%s: OnGPU = %v, want %v", tt.device, got, tt.onGPU)
		}
	}
}

// CPU_Mapped weights appeared on a model with no per-layer embedding
// table, so "weights are in VRAM" is wrong for reasons beyond the PLE
// table. Splitting the model kind by device is what shows that.
func TestModelWeightsSplitAcrossHostAndDevice(t *testing.T) {
	r := parseAll(t, realLog)
	host := r.SumMiB(func(e Entry) bool { return e.Kind == KindModel && !e.OnGPU() })
	gpu := r.SumMiB(func(e Entry) bool { return e.Kind == KindModel && e.OnGPU() })
	if !close(host, 1288.28) {
		t.Errorf("host-side weights = %.2f, want 1288.28", host)
	}
	if !close(gpu, 7252.51) {
		t.Errorf("device-side weights = %.2f, want 7252.51", gpu)
	}
}

// Compute buffers are reserved more than once per load. Whether those
// coalesce is unresolved, so both readings must be available and must
// actually differ.
func TestComputeAggregationOffersBothReadings(t *testing.T) {
	r := parseAll(t, realLog)
	sum := r.SumMiB(func(e Entry) bool { return e.Kind == KindCompute && e.OnGPU() })
	max := r.MaxComputeMiB(true)
	if !close(sum, 384.95+324.02) {
		t.Errorf("summed GPU compute = %.2f, want 708.97", sum)
	}
	if !close(max, 384.95) {
		t.Errorf("peak GPU compute = %.2f, want 384.95", max)
	}
	if close(sum, max) {
		t.Error("the two readings agree; the test model no longer exercises repeated reserves")
	}
}

func TestGPUTotalTracksTheComputeReading(t *testing.T) {
	r := parseAll(t, realLog)
	weights, kv, rs := 7252.51, 256.00, 448.88
	if got := r.GPUMiB(true); !close(got, weights+kv+rs+384.95+324.02) {
		t.Errorf("GPUMiB(sum) = %.2f", got)
	}
	if got := r.GPUMiB(false); !close(got, weights+kv+rs+384.95) {
		t.Errorf("GPUMiB(peak) = %.2f", got)
	}
	if got := r.HostMiB(false); !close(got, 1288.28+3.79+276.02) {
		t.Errorf("HostMiB(peak) = %.2f", got)
	}
}

// Concurrently loading models interleave in one stream.
func TestInstancesAreSeparable(t *testing.T) {
	log := `[111] 0.1 I load_tensors:  ROCm0 model buffer size =  100.00 MiB
[222] 0.1 I load_tensors:  ROCm0 model buffer size =  900.00 MiB
[111] 0.2 I sched_reserve: ROCm0 compute buffer size =   10.00 MiB`
	r := parseAll(t, log)
	if got := r.Instances(); len(got) != 2 {
		t.Fatalf("instances = %v, want two", got)
	}
	if got := r.ForInstance("111").SumMiB(nil); !close(got, 110.00) {
		t.Errorf("instance 111 total = %.2f, want 110.00", got)
	}
	if got := r.ForInstance("222").SumMiB(nil); !close(got, 900.00) {
		t.Errorf("instance 222 total = %.2f, want 900.00", got)
	}
}

// Lines without the router prefix (the router's own output) still parse.
func TestUnprefixedLineParses(t *testing.T) {
	e, ok := ParseLine(`0.14.275.753 I load_tensors:       ROCm0 model buffer size =  7252.51 MiB`)
	if !ok {
		t.Fatal("did not parse")
	}
	if e.Instance != "" || e.Device != "ROCm0" || !close(e.MiB, 7252.51) {
		t.Errorf("got %+v", e)
	}
}

func TestNonBufferLinesIgnored(t *testing.T) {
	for _, l := range []string{
		"",
		"data: ",
		`[1] 0.1 I srv  llama_server: model loaded`,
		`[1] 0.1 D load_tensors: layer 1 assigned to device Meta(), is_swa = 0`,
		`[1] 0.1 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)`,
		`[1] 0.1 I load_model: [mtmd] estimated worst-case memory usage of mmproj is 1161.02 MiB`,
	} {
		if e, ok := ParseLine(l); ok {
			t.Errorf("parsed %q as %+v", l, e)
		}
	}
}

// A complete report from Qwen3.8-Flash-Next on four ROCm cards, captured
// at verbosity 4. Kept whole because it is the one load where every term
// is known: the file is 103.69 GiB, the per-layer embedding table is
// 26.82 GiB, and total VRAM measured 107.35 GiB from the GPU counters.
// Those three independent figures are what the parser is checked against.
const flashNextLog = `[44479] 0.12.132.471 I load_tensors:   CPU_Mapped model buffer size = 28110.09 MiB
[44479] 0.12.132.473 I load_tensors:        ROCm0 model buffer size = 21297.21 MiB
[44479] 0.12.132.473 I load_tensors:        ROCm1 model buffer size = 18980.54 MiB
[44479] 0.12.132.474 I load_tensors:        ROCm2 model buffer size = 19230.54 MiB
[44479] 0.12.132.475 I load_tensors:        ROCm3 model buffer size = 18548.17 MiB
[44479] 0.22.653.808 I llama_context:  ROCm_Host  output buffer size =     3.79 MiB
[44479] 0.22.655.512 I llama_kv_cache:      ROCm0 KV buffer size =   816.00 MiB
[44479] 0.22.660.169 I llama_kv_cache:      ROCm1 KV buffer size =   816.00 MiB
[44479] 0.22.664.791 I llama_kv_cache:      ROCm2 KV buffer size =   816.00 MiB
[44479] 0.22.669.395 I llama_kv_cache:      ROCm3 KV buffer size =   816.00 MiB
[44479] 0.22.674.696 I llama_memory_recurrent:      ROCm0 RS buffer size =   126.09 MiB
[44479] 0.22.675.188 I llama_memory_recurrent:      ROCm1 RS buffer size =   112.22 MiB
[44479] 0.22.675.666 I llama_memory_recurrent:      ROCm2 RS buffer size =   112.22 MiB
[44479] 0.22.675.946 I llama_memory_recurrent:      ROCm3 RS buffer size =    99.75 MiB
[44479] 0.22.683.127 I llama_kv_cache:      ROCm0 KV buffer size =   306.00 MiB
[44479] 0.22.684.183 I llama_kv_cache:      ROCm1 KV buffer size =   306.00 MiB
[44479] 0.22.685.196 I llama_kv_cache:      ROCm2 KV buffer size =   306.00 MiB
[44479] 0.22.686.191 I llama_kv_cache:      ROCm3 KV buffer size =   306.00 MiB
[44479] 1.01.762.088 I sched_reserve:      ROCm0 compute buffer size =  6047.40 MiB
[44479] 1.01.762.099 I sched_reserve:      ROCm1 compute buffer size =  6311.40 MiB
[44479] 1.01.762.099 I sched_reserve:      ROCm2 compute buffer size =  6311.40 MiB
[44479] 1.01.762.100 I sched_reserve:      ROCm3 compute buffer size =  6311.40 MiB
[44479] 1.01.762.101 I sched_reserve:  ROCm_Host compute buffer size = 52267.48 MiB`

const gib = 1024.0

// Weights reported across every device must add back up to the file on
// disk. This is the check that the parser is reading whole numbers and
// not, say, dropping a device whose name it failed to match.
func TestFlashNextWeightsReconcileWithFileSize(t *testing.T) {
	r := parseAll(t, flashNextLog)
	weights := r.SumMiB(func(e Entry) bool { return e.Kind == KindModel }) / gib
	const fileGiB = 103.69
	if math.Abs(weights-fileGiB) > 0.05 {
		t.Errorf("weights total %.2f GiB, want the file's %.2f", weights, fileGiB)
	}
}

// The finding this whole exercise produced: the per-layer embedding table
// is held CPU_Mapped, so it is host memory and never counts toward VRAM.
// The estimate treats it as resident weights whenever the mode is off.
func TestFlashNextPLETableIsHostMapped(t *testing.T) {
	r := parseAll(t, flashNextLog)
	host := r.SumMiB(func(e Entry) bool { return e.Kind == KindModel && !e.OnGPU() }) / gib
	const pleGiB = 26.82
	if host < pleGiB {
		t.Errorf("host-mapped weights %.2f GiB are smaller than the %.2f GiB table; it cannot be held there", host, pleGiB)
	}
	// Everything above the table is other weights that also stayed
	// mapped — evidence that excluding only the table would still be wrong.
	if excess := host - pleGiB; excess < 0.1 {
		t.Errorf("host-mapped weights exceed the table by only %.2f GiB; the fixture no longer shows the effect", excess)
	}
}

// The parser's GPU total must land close to what the GPU counters read,
// or it is not measuring the thing the estimate is trying to predict.
func TestFlashNextGPUTotalMatchesMeasuredVRAM(t *testing.T) {
	r := parseAll(t, flashNextLog)
	got := r.GPUMiB(true) / gib
	const measuredGiB = 107.35 // summed across the four cards while loaded
	gap := measuredGiB - got
	if gap < 0 || gap > 3 {
		t.Errorf("reported GPU total %.2f GiB against %.2f measured (gap %.2f); "+
			"expected the report to account for all but a couple of GiB of context overhead",
			got, measuredGiB, gap)
	}
}

// Compute buffers are the term the estimate omits entirely, and on this
// load they are larger than the KV cache by more than five times. Named
// so a future change that quietly drops them fails loudly.
func TestFlashNextComputeBuffersDominateTheShortfall(t *testing.T) {
	r := parseAll(t, flashNextLog)
	k := r.ByKind()
	compute, kv := k[KindCompute], k[KindKV]
	gpuCompute := r.SumMiB(func(e Entry) bool { return e.Kind == KindCompute && e.OnGPU() }) / gib
	if gpuCompute < 20 {
		t.Errorf("GPU compute buffers %.2f GiB, expected ~24 — the dominant unmodelled term", gpuCompute)
	}
	if compute <= kv {
		t.Errorf("compute %.2f MiB <= KV %.2f MiB; the fixture no longer shows compute dominating", compute, kv)
	}
	// Host-side compute is larger still, which is what the +53 GiB of
	// host memory seen during a load actually was.
	hostCompute := r.SumMiB(func(e Entry) bool { return e.Kind == KindCompute && !e.OnGPU() }) / gib
	if hostCompute < 40 {
		t.Errorf("host compute buffers %.2f GiB, expected ~51", hostCompute)
	}
}

// One reserve round per device here, so the two aggregations agree. If a
// future capture makes them differ, the ambiguity documented on SumMiB
// is live again and step 3 has data to settle it.
func TestFlashNextHasASingleReserveRound(t *testing.T) {
	r := parseAll(t, flashNextLog)
	sum := r.SumMiB(func(e Entry) bool { return e.Kind == KindCompute && e.OnGPU() })
	if max := r.MaxComputeMiB(true); math.Abs(sum-max) > 0.005 {
		t.Errorf("sum %.2f and peak %.2f disagree; this load reserves once per device", sum, max)
	}
}
