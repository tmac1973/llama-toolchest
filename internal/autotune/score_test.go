package autotune

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// run builds a benchmark run with one per-size row per (size, pp, tg).
func run(rows ...[4]float64) *benchmark.BenchmarkRun {
	var sizes []benchmark.SizeSummary
	for _, r := range rows {
		sizes = append(sizes, benchmark.SizeSummary{
			PromptTokens: int(r[0]), PPMean: r[1], TGMean: r[2],
			PPStd: r[3], TGStd: r[3], Count: 3,
		})
	}
	return &benchmark.BenchmarkRun{Summary: &benchmark.BenchmarkSummary{PerSize: sizes}}
}

func TestScoreRuns(t *testing.T) {
	// The measured sizes never land exactly on the asked-for ones.
	chat := run([4]float64{498, 900, 40, 0}, [4]float64{4002, 1500, 35, 0})
	runs := map[string]*benchmark.BenchmarkRun{"autotune-chat": chat}

	gen, err := ScoreRuns(UseChat, GoalGeneration, runs)
	if err != nil || gen.Value != 40 {
		t.Errorf("generation = %+v, %v; want the 512-token row's 40", gen, err)
	}
	prompt, err := ScoreRuns(UseChat, GoalPrompt, runs)
	if err != nil || prompt.Value != 1500 {
		t.Errorf("prompt = %+v, %v; want the 4096-token row's 1500", prompt, err)
	}
	resp, err := ScoreRuns(UseChat, GoalResponse, runs)
	if err != nil {
		t.Fatal(err)
	}
	want := -(512.0/900 + 256.0/40) // read the prompt, then write the answer
	if math.Abs(resp.Value-want) > 1e-9 {
		t.Errorf("response = %v, want %v", resp.Value, want)
	}
	if resp.Value >= 0 {
		t.Error("response time must be carried as negative seconds, so higher is better everywhere")
	}
}

// A mixed use case measures two presets: speeds average, response times
// add up, because the workload is one of each.
func TestScoreRunsMixed(t *testing.T) {
	runs := map[string]*benchmark.BenchmarkRun{
		"autotune-chat": run([4]float64{512, 1000, 40, 0}, [4]float64{4096, 1000, 35, 0}),
		"autotune-code": run([4]float64{1536, 500, 20, 0}),
	}
	gen, err := ScoreRuns(UseMixed, GoalGeneration, runs)
	if err != nil || gen.Value != 30 {
		t.Errorf("generation = %+v, %v; want the average of 40 and 20", gen, err)
	}
	resp, _ := ScoreRuns(UseMixed, GoalResponse, runs)
	want := -(512.0/1000 + 256.0/40) - (1536.0/500 + 512.0/20)
	if math.Abs(resp.Value-want) > 1e-9 {
		t.Errorf("response = %v, want %v", resp.Value, want)
	}
}

func TestScoreRunsMissingMeasurement(t *testing.T) {
	if _, err := ScoreRuns(UseChat, GoalGeneration, nil); !errors.Is(err, ErrNoMeasurement) {
		t.Errorf("err = %v, want ErrNoMeasurement", err)
	}
	empty := map[string]*benchmark.BenchmarkRun{"autotune-chat": run()}
	if _, err := ScoreRuns(UseChat, GoalPrompt, empty); !errors.Is(err, ErrNoMeasurement) {
		t.Errorf("a run with no rows: err = %v, want ErrNoMeasurement", err)
	}
}

// The spread propagates through the response time: a speed known to
// within 10% makes a response time known to about 10%.
func TestScoreRunsCarriesTheSpread(t *testing.T) {
	runs := map[string]*benchmark.BenchmarkRun{"autotune-chat": run([4]float64{512, 1000, 40, 4})}
	resp, err := ScoreRuns(UseChat, GoalResponse, runs)
	if err != nil {
		t.Fatal(err)
	}
	// Generation dominates: 256/40 = 6.4s, and 10% of that is 0.64s.
	if resp.Std < 0.5 || resp.Std > 0.8 {
		t.Errorf("response std = %v, want about 0.64", resp.Std)
	}
}

func TestBeatsIsTheNoiseRule(t *testing.T) {
	a := Score{Value: 100, Std: 3}
	if Beats(Score{Value: 104, Std: 3}, a) {
		t.Error("a 4% gain inside the spread of two measurements counts as a win")
	}
	if !Beats(Score{Value: 120, Std: 3}, a) {
		t.Error("a 20% gain outside the spread does not count as a win")
	}
}

func cand(label string, cfg benchmark.ConfigSnapshot, values map[string]string, gen float64, std float64) Candidate {
	return Candidate{Values: values, Label: label, Config: cfg,
		Scores: map[Goal]Score{GoalGeneration: {Value: gen, Std: std}}}
}

// Among candidates that measure the same, the plainest wins: MTP alone
// over MTP with an assist, when the pair's gain is inside the noise.
func TestPickKeepsTheSimplerSettingInsideTheNoise(t *testing.T) {
	base := benchmark.ConfigSnapshot{}
	mtp := benchmark.ConfigSnapshot{SpecType: "draft-mtp"}
	pair := benchmark.ConfigSnapshot{SpecType: "draft-mtp", SpecAssist: "ngram-mod"}

	cands := []Candidate{
		cand("baseline", base, nil, 40, 1),
		cand("MTP", mtp, map[string]string{"spec_type": "draft-mtp"}, 60, 2),
		cand("MTP + assist", pair, map[string]string{"spec_type": "draft-mtp+ngram-mod"}, 61, 2),
	}
	winner, ranked, ok := Pick(cands, GoalGeneration, base)
	if !ok || winner.Label != "MTP" {
		t.Errorf("winner = %q, want MTP (the pair's extra 1 is inside the noise)", winner.Label)
	}
	if len(ranked) != 3 || ranked[0].Label != "MTP + assist" {
		t.Errorf("ranked = %v, want fastest first", labels(ranked))
	}

	// A real gain wins, complexity or not.
	cands[2] = cand("MTP + assist", pair, map[string]string{"spec_type": "draft-mtp+ngram-mod"}, 80, 2)
	winner, _, _ = Pick(cands, GoalGeneration, base)
	if winner.Label != "MTP + assist" {
		t.Errorf("winner = %q, want the measurably faster pair", winner.Label)
	}
}

func TestPickWithoutMeasurements(t *testing.T) {
	if _, _, ok := Pick([]Candidate{{Label: "failed"}}, GoalGeneration, benchmark.ConfigSnapshot{}); ok {
		t.Error("a candidate with no score for the goal was picked")
	}
}

func labels(cs []Candidate) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c.Label)
	}
	return out
}

func TestComplexityCountsWhatWasAdded(t *testing.T) {
	base := benchmark.ConfigSnapshot{UBatchSize: 512, Threads: 8}
	if n := Complexity(base, base); n != 0 {
		t.Errorf("the starting profile has complexity %d, want 0", n)
	}
	batch := base
	batch.UBatchSize = 1024
	spec := base
	spec.SpecType, spec.DraftMax = "draft-mtp", 6
	if Complexity(spec, base) <= Complexity(batch, base) {
		t.Error("speculative decoding should count for more than a batch size")
	}
}

func TestDescribe(t *testing.T) {
	cases := map[string]string{
		"":                                      "Starting profile, unchanged",
		"ubatch_size=1024":                      "prompt batch 1024",
		"spec_type=draft-mtp":                   "MTP draft layers",
		"spec_type=draft-mtp+ngram-mod":         "MTP draft layers with an ngram-mod assist",
		"spec_type=draft-mtp:draft_max=3":       "MTP draft layers, 3 draft tokens",
		"spec_type=none":                        "no speculative decoding",
		"flash_attention=false":                 "flash attention off",
		"split_mode=tensor":                     "tensor split",
		"cpu_moe=12":                            "12 expert layers on the CPU",
		"spec_type=draft:draft_model=a--b.gguf": "a draft model, drafting with b.gguf",
	}
	for in, want := range cases {
		values := map[string]string{}
		if in != "" {
			k, v, _ := strings.Cut(in, "=")
			values[k] = v
		}
		if got := Describe(values); got != want {
			t.Errorf("Describe(%q) = %q, want %q", in, got, want)
		}
	}
	if got := Describe(map[string]string{"ubatch_size": "1024", "flash_attention": "true"}); got != "flash attention on, prompt batch 1024" {
		t.Errorf("two values = %q", got)
	}
}

func TestCommonFailure(t *testing.T) {
	load404 := "failed to load model: HTTP 404: File Not Found"
	cases := []struct {
		name   string
		failed []FailedCell
		want   string
	}{
		{"none", nil, ""},
		{"all the same", []FailedCell{{Label: "a", Error: load404}, {Label: "b", Error: load404}}, load404},
		{"different", []FailedCell{{Label: "a", Error: load404}, {Label: "b", Error: "out of memory"}}, ""},
		{"empty reason", []FailedCell{{Label: "a"}}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commonFailure(tc.failed); got != tc.want {
				t.Errorf("commonFailure = %q, want %q", got, tc.want)
			}
		})
	}
}
