package evaluate

import (
	"errors"
	"strings"
	"testing"
)

// Canned combined-output blocks, shaped on real llama-perplexity runs:
// stdout and stderr interleaved exactly as step 4's combined buffer
// captures them (LOG lines on stdout, LOG_INF lines on stderr). The
// per-mode score lines are the source's exact formats.

const perplexityOutput = `llama_model_loader: loaded GGUF v3 (arch: llama)
common_init_from_params: model loaded
perplexity: tokenizing the input ..
perplexity: calculating perplexity over 100 chunks, n_ctx=512, batch_size=2048, n_seq=1
[1]9.8765,[2]9.4321,[3]9.2109,[4]9.1010,[5]9.0512,[6]9.0321,[7]9.0234,[8]9.0189,[9]9.0156,[10]9.0123,
Final estimate: PPL = 9.0123 +/- 0.00456
`

// HellaSwag prints one line per task; the LAST line (400 tasks) is the
// score. The earlier lines exist to prove the parser takes the last
// match, not the first (the first line says 80%).
const hellaswagOutput = `llama_model_loader: loaded GGUF v3 (arch: llama)
common_init_from_params: model loaded
hellaswag_score : calculating hellaswag score over selected tasks.
1	80.00000000%	[13.3603%, 99.0340%]
2	75.00000000%	[26.9181%, 92.1493%]
3	76.66666667%	[33.5210%, 92.5367%]
4	75.00000000%	[38.4925%, 91.2206%]
400	74.25000000%	[70.5511%, 77.8154%]
`

const winograndeOutput = `llama_model_loader: loaded GGUF v3 (arch: llama)
common_init_from_params: model loaded
winogrande_score : calculating winogrande score over selected tasks.
1	50.0000	 0.123456  0.123456  0  1
2	50.0000	 0.123456  0.123456  1  1
3	66.6667	 0.123456  0.123456  1  1
400	74.2500	 0.123456  0.123456  1  0
Final Winogrande score(400 tasks): 74.2500 +/- 2.1234
`

const klOutput = `llama_model_loader: loaded GGUF v3 (arch: llama)
common_init_from_params: model loaded
common_init_from_params: loading logits from 'ref.kld'
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1     9.0123 ±    0.0046      0.000012 ±    0.000012      0.004123 ±    0.000123      0.012 ±    0.003 %     95.123 ±    0.543 %
   2     9.0111 ±    0.0046      0.000011 ±    0.000011      0.004100 ±    0.000120      0.012 ±    0.003 %     95.120 ±    0.544 %
====== Perplexity statistics ======
Mean PPL(Q)                   :     9.0123 ±     0.0046
Mean PPL(base)                :     9.0111 ±     0.0046
Cor(ln(PPL(Q)), ln(PPL(base))):   99.98%
Mean ln(PPL(Q)/PPL(base))     :     0.0001 ±     0.0001
Mean PPL(Q)/PPL(base)         :     1.0001 ±     0.0001
Mean PPL(Q)-PPL(base)         :     0.0012 ±     0.0001

====== KL divergence statistics ======
Mean    KLD:   0.004123 ±   0.000123
Maximum KLD:   0.123456
99.9%   KLD:   0.098765
99.0%   KLD:   0.090000
95.0%   KLD:   0.085000
90.0%   KLD:   0.080000
Median  KLD:   0.004000
10.0%   KLD:   0.000100
 5.0%   KLD:   0.000050
 1.0%   KLD:   0.000010
 0.1%   KLD:   0.000001
Minimum KLD:   0.000000

====== Token probability statistics ======
Mean    Δp:  0.012 ±  0.003 %
Maximum Δp:  0.500%
99.9%   Δp:  0.300%
Median  Δp:  0.010%
Minimum Δp:  0.000%
RMS Δp    :  0.050 ±  0.002 %
Same top p:  95.123 ±  0.543 %
`

func TestParsePerplexity(t *testing.T) {
	res, err := parse(Spec{Mode: ModePerplexity, Chunks: 100}, perplexityOutput)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if res.Mode != "perplexity" {
		t.Errorf("Mode = %q, want perplexity", res.Mode)
	}
	if res.Dataset != "wikitext-2" {
		t.Errorf("Dataset = %q, want wikitext-2", res.Dataset)
	}
	if res.ContextSize != EvalContextSize {
		t.Errorf("ContextSize = %d, want %d", res.ContextSize, EvalContextSize)
	}
	if res.Chunks != 100 {
		t.Errorf("Chunks = %d, want 100 (copied from Spec.Chunks)", res.Chunks)
	}
	if res.Perplexity != 9.0123 {
		t.Errorf("Perplexity = %v, want 9.0123", res.Perplexity)
	}
	if res.PerplexityErr != 0.00456 {
		t.Errorf("PerplexityErr = %v, want 0.00456", res.PerplexityErr)
	}
}

func TestParseHellaSwag(t *testing.T) {
	res, err := parse(Spec{Mode: ModeHellaSwag}, hellaswagOutput)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if res.Mode != "hellaswag" {
		t.Errorf("Mode = %q, want hellaswag", res.Mode)
	}
	if res.Dataset != "hellaswag" {
		t.Errorf("Dataset = %q, want hellaswag", res.Dataset)
	}
	if res.ContextSize != EvalContextSize {
		t.Errorf("ContextSize = %d, want %d", res.ContextSize, EvalContextSize)
	}
	if res.Tasks != 400 {
		t.Errorf("Tasks = %d, want 400 (the LAST per-task line)", res.Tasks)
	}
	if res.Accuracy != 74.25 {
		t.Errorf("Accuracy = %v, want 74.25 (last line, not the first)", res.Accuracy)
	}
	// HellaSwag's CI is asymmetric — the bounds come from the brackets.
	if res.AccuracyCILow != 70.5511 {
		t.Errorf("AccuracyCILow = %v, want 70.5511", res.AccuracyCILow)
	}
	if res.AccuracyCIHigh != 77.8154 {
		t.Errorf("AccuracyCIHigh = %v, want 77.8154", res.AccuracyCIHigh)
	}
	if res.AccuracyCILow >= res.Accuracy || res.Accuracy >= res.AccuracyCIHigh {
		t.Errorf("CI bounds %v..%v do not bracket accuracy %v", res.AccuracyCILow, res.AccuracyCIHigh, res.Accuracy)
	}
}

func TestParseWinogrande(t *testing.T) {
	res, err := parse(Spec{Mode: ModeWinogrande}, winograndeOutput)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if res.Mode != "winogrande" {
		t.Errorf("Mode = %q, want winogrande", res.Mode)
	}
	if res.Dataset != "winogrande" {
		t.Errorf("Dataset = %q, want winogrande", res.Dataset)
	}
	if res.Tasks != 400 {
		t.Errorf("Tasks = %d, want 400", res.Tasks)
	}
	if res.Accuracy != 74.25 {
		t.Errorf("Accuracy = %v, want 74.25", res.Accuracy)
	}
	// The symmetric +/- error maps onto acc−err / acc+err.
	if res.AccuracyCILow != 74.25-2.1234 {
		t.Errorf("AccuracyCILow = %v, want %v", res.AccuracyCILow, 74.25-2.1234)
	}
	if res.AccuracyCIHigh != 74.25+2.1234 {
		t.Errorf("AccuracyCIHigh = %v, want %v", res.AccuracyCIHigh, 74.25+2.1234)
	}
}

func TestParseKLDivergence(t *testing.T) {
	res, err := parse(Spec{Mode: ModeKLDiv, Chunks: 100, KLBasePath: "ref.kld"}, klOutput)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if res.Mode != "kl-divergence" {
		t.Errorf("Mode = %q, want kl-divergence", res.Mode)
	}
	if res.Dataset != "wikitext-2" {
		t.Errorf("Dataset = %q, want wikitext-2", res.Dataset)
	}
	if res.Chunks != 100 {
		t.Errorf("Chunks = %d, want 100", res.Chunks)
	}
	if res.KLMean != 0.004123 {
		t.Errorf("KLMean = %v, want 0.004123", res.KLMean)
	}
	if res.KLMeanErr != 0.000123 {
		t.Errorf("KLMeanErr = %v, want 0.000123", res.KLMeanErr)
	}
	if res.KLMax != 0.123456 {
		t.Errorf("KLMax = %v, want 0.123456", res.KLMax)
	}
	if res.KLP999 != 0.098765 {
		t.Errorf("KLP999 = %v, want 0.098765", res.KLP999)
	}
	if res.SameTopPct != 95.123 {
		t.Errorf("SameTopPct = %v, want 95.123", res.SameTopPct)
	}
	if res.SameTopPctErr != 0.543 {
		t.Errorf("SameTopPctErr = %v, want 0.543", res.SameTopPctErr)
	}
}

// Mismatch cases: output that lacks a score line must fail with a
// ParseError that NAMES the missing line, so a llama.cpp format change
// is loud and identifiable rather than a silent zero score.

func assertParseErrorNames(t *testing.T, spec Spec, output, wantSubstr, what string) {
	t.Helper()
	_, err := parse(spec, output)
	if err == nil {
		t.Fatalf("%s: want missing-line error, got success", what)
	}
	var pe *ParseError
	if !errors.As(err, &pe) {
		t.Fatalf("%s: want *ParseError, got %T (%v)", what, err, err)
	}
	if !strings.Contains(pe.Line, wantSubstr) {
		t.Errorf("%s: ParseError.Line = %q, want it to name %q", what, pe.Line, wantSubstr)
	}
}

func TestParseMissingLines(t *testing.T) {
	assertParseErrorNames(t, Spec{Mode: ModePerplexity},
		"perplexity: calculating perplexity over 100 chunks, n_ctx=512\n[1]9.0123,\n",
		"Final estimate: PPL", "perplexity without final estimate")

	assertParseErrorNames(t, Spec{Mode: ModeWinogrande},
		"winogrande_score : calculating winogrande score over selected tasks.\n1\t50.0000\t 0.1 0.1 0 1\n",
		"Final Winogrande score", "winogrande without final score")

	assertParseErrorNames(t, Spec{Mode: ModeHellaSwag},
		"hellaswag_score : calculating hellaswag score over selected tasks.\n",
		"<acc>%", "hellaswag without per-task lines")

	assertParseErrorNames(t, Spec{Mode: ModeKLDiv},
		"====== KL divergence statistics ======\nMaximum KLD:   0.123456\n99.9%   KLD:   0.098765\nSame top p:  95.123 ±  0.543 %\n",
		"Mean    KLD", "KL block without the mean line")
}

// A KL block missing exactly one of its lines must name THAT line, not
// just "something KL-ish".
func TestParseKLMissingSpecificLine(t *testing.T) {
	cases := []struct {
		dropLine string // the label to strip from klOutput
		wantSub  string // what the error must name
	}{
		{"Maximum KLD:", "Maximum KLD"},
		{"99.9%   KLD:", "99.9%   KLD"},
		{"Same top p:", "Same top p"},
	}
	for _, tc := range cases {
		var b strings.Builder
		for _, line := range strings.Split(klOutput, "\n") {
			if strings.HasPrefix(line, tc.dropLine) {
				continue
			}
			b.WriteString(line + "\n")
		}
		assertParseErrorNames(t, Spec{Mode: ModeKLDiv}, b.String(), tc.wantSub, "KL missing "+tc.dropLine)
	}
}

// A format change the parsers must catch: the same numbers printed at
// different precision (e.g. %.2lf instead of %.4lf) is a mismatch.
func TestParseRejectsChangedPrecision(t *testing.T) {
	assertParseErrorNames(t, Spec{Mode: ModePerplexity},
		"Final estimate: PPL = 9.01 +/- 0.005\n",
		"Final estimate: PPL", "PPL line at 2-decimal precision")

	assertParseErrorNames(t, Spec{Mode: ModeWinogrande},
		"Final Winogrande score(400 tasks): 74.25 +/- 2.12\n",
		"Final Winogrande score", "Winogrande line at 2-decimal precision")
}
