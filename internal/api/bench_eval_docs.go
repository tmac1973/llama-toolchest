package api

import (
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// How to read a capability score.
//
// A capability result is a bare number with no units and no natural
// sense of scale — "PPL 8.043 ±0.13" tells someone who did not build
// the evaluation nothing at all. Every surface that shows a score
// therefore also has to answer three questions: what the number
// measures, what a good and a bad value look like, and where the fuller
// description lives.
//
// This table is the single source for those answers. The run detail
// view, the About modal and the help page all render it, so the
// explanation cannot say one thing in one place and something else in
// another.
//
// The writing is aimed at someone choosing between model files, not at
// someone who already knows what a logit is. That means: short
// sentences, no term used before it is explained, "compressed copy"
// rather than "quantization" on first use, "margin of error" rather
// than "confidence interval", and worked examples instead of an
// abstract direction of better. Where a threshold is quoted it comes
// from one of the linked sources rather than from taste.

// evalExample is one worked reading of a score: a value, a one-word
// verdict, and what it means in practice. Rendered as a small table —
// "is 8.043 good?" is the actual question, and a scale of examples
// answers it in a way that a sentence about direction does not.
type evalExample struct {
	Value   string
	Verdict string
	Meaning string
}

// evalLink is one external source, with a note saying why a reader
// would open it. The note matters because the same llama.cpp page
// documents both perplexity and KL divergence, and a link labelled only
// "perplexity documentation" looks wrong under a KL score.
type evalLink struct {
	URL   string
	Label string
	Note  string
}

// evalDoc is the interpretation guidance for one evaluation mode.
type evalDoc struct {
	// Headline states, in one sentence, what the number measures.
	Headline string
	// Reading are the practical rules: which comparisons are valid,
	// what the error figure means, when the result is trustworthy.
	Reading []string
	// Examples are the good-versus-bad scale.
	Examples []evalExample
	// Links are the external sources, most relevant first.
	Links []evalLink
}

// The llama.cpp source for both perplexity and KL divergence.
//
// KL divergence in llama.cpp is a MODE OF THE PERPLEXITY TOOL — it is
// reached with --kl-divergence and --kl-divergence-base on the same
// llama-perplexity binary — so this one README is the correct
// documentation for both, and the label says so rather than leaving a
// KL score pointing at something that reads like the wrong page.
var llamaCppMeasurementDoc = evalLink{
	URL:   "https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md",
	Label: "llama.cpp measurement guide",
	Note:  "The documentation for the tool that produced this number. It covers perplexity and KL divergence together, because llama.cpp measures both with the same program. It also publishes a table of results for every compression level of Llama 3 8B.",
}

var evalDocs = map[evaluate.Mode]evalDoc{
	evaluate.ModePerplexity: {
		Headline: "Perplexity measures how often the model was surprised by the next word in a sample of ordinary writing. A lower number means it predicted better.",
		Reading: []string{
			"A perplexity number on its own cannot tell you whether a model is good. It only means something next to another number from the same model.",
			"Never compare perplexity between two different models. Models cut text into pieces in different ways, so their scores sit on different scales — a Qwen score and a Llama score cannot be compared, even on the same text.",
			"The number after ± is the margin of error. If two results' ranges overlap, this test has not found a real difference between them.",
			"A short run has a wide margin of error — often wider than the difference between two compressed copies of a model. To compare compressed copies, use KL divergence instead: it looks far more closely.",
		},
		Examples: []evalExample{
			{
				Value:   "5 to 15",
				Verdict: "Normal",
				Meaning: "The usual range for a modern model on this text. Its exact position within that range depends on the model, so this on its own is not a grade.",
			},
			{
				Value:   "0.05 above the largest version of the same model",
				Verdict: "Excellent",
				Meaning: "The compressed copy behaves practically the same as the original.",
			},
			{
				Value:   "0.2 above",
				Verdict: "Fine",
				Meaning: "About the gap llama.cpp measures between an uncompressed Llama 3 8B and a 4-bit copy of it. A small, real cost.",
			},
			{
				Value:   "1.0 or more above",
				Verdict: "Poor",
				Meaning: "Clear quality loss. Try a larger version of the model file.",
			},
			{
				Value:   "100 or more",
				Verdict: "Broken",
				Meaning: "Not a quality problem. The model file or the settings are wrong — this is not a score to compare.",
			},
		},
		Links: []evalLink{llamaCppMeasurementDoc},
	},
	evaluate.ModeKLDiv: {
		Headline: "KL divergence measures how differently a compressed copy of a model behaves compared with the original. 0 means no difference at all.",
		Reading: []string{
			"This is the right test for the question \"how much did compressing this model change it?\". It checks the model's choices at every position in the text, so it notices differences that perplexity is not precise enough to see.",
			"\"Same top token\" is the same finding in everyday terms: how often both versions would have picked the same next word.",
			"Read the average. The maximum is only the single worst position in the whole text, and one strange sentence can make it look alarming.",
			"The score only means something against the reference model named beside it. A different reference is a different measurement.",
		},
		Examples: []evalExample{
			{
				Value:   "0.000 to 0.002",
				Verdict: "Excellent",
				Meaning: "Practically the same model. You would not notice the difference in use. This is typical of llama.cpp's 8-bit copies.",
			},
			{
				Value:   "around 0.01",
				Verdict: "Good",
				Meaning: "A small but real change. Fine for everyday use. Typical of llama.cpp's 5-bit copies.",
			},
			{
				Value:   "around 0.03",
				Verdict: "Noticeable",
				Meaning: "Measurably different. Expect the occasional different answer. Typical of llama.cpp's 4-bit copies.",
			},
			{
				Value:   "0.1 or more",
				Verdict: "Poor",
				Meaning: "Heavily changed. Use a larger version of the model file if quality matters.",
			},
			{
				Value:   "Same top token above 99%",
				Verdict: "Excellent",
				Meaning: "The two versions almost always want to say the same next word.",
			},
			{
				Value:   "Same top token below 90%",
				Verdict: "Poor",
				Meaning: "The two versions disagree often enough to change how the model writes.",
			},
		},
		Links: []evalLink{
			llamaCppMeasurementDoc,
			{
				URL:   "https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/",
				Label: "Measuring model quantisation quality with KL divergence",
				Note:  "A plain-language explanation of what the number means and where the thresholds come from, worked through on Qwen models.",
			},
			{
				URL:   "https://github.com/ggml-org/llama.cpp/discussions/4110",
				Label: "Why KL divergence is better than perplexity for this",
				Note:  "The llama.cpp discussion that explains why perplexity is a poor way to judge compression loss and this is a better one.",
			},
		},
	},
	evaluate.ModeHellaSwag: {
		Headline: "HellaSwag is a common-sense test. The model reads the start of a sentence and picks which of four endings makes sense. The score is the percentage it got right.",
		Reading: []string{
			"There are four endings to choose from, so a model that is only guessing scores about 25%. A score near 25% means something is wrong, not that the model is weak.",
			"The two numbers in brackets are the margin of error. If two models' ranges overlap, this test has not found a real difference between them.",
			"Fewer questions means a wider margin. Use the quick run for a rough check and the full run for a number you would quote to someone else.",
			"Unlike perplexity, this score CAN be compared between different models. It simply counts correct answers, so nothing about the model changes the scale.",
		},
		Examples: []evalExample{
			{Value: "about 25%", Verdict: "Broken", Meaning: "The same as guessing. The model, the file or the settings are wrong."},
			{Value: "40 to 55%", Verdict: "Weak", Meaning: "Better than guessing, but the model is struggling with the task."},
			{Value: "70 to 80%", Verdict: "Good", Meaning: "The usual range for a capable small model."},
			{Value: "85% and above", Verdict: "Strong", Meaning: "Among the better results for a model you can run at home."},
			{Value: "95.6%", Verdict: "Human level", Meaning: "What people score on this test. The best reported models are close to it."},
		},
		Links: []evalLink{
			{
				URL:   "https://rowanzellers.com/hellaswag/",
				Label: "HellaSwag dataset page",
				Note:  "The people who built the test explain what it asks, show example questions, and publish the leaderboard.",
			},
		},
	},
	evaluate.ModeWinogrande: {
		Headline: "Winogrande tests whether the model works out who or what a word like \"it\" or \"they\" refers to. The score is the percentage of sentences it got right.",
		Reading: []string{
			"There are two choices, so a model that is only guessing scores about 50%. A score near 50% means something is wrong, not that the model is weak.",
			"Because guessing already scores 50%, the useful range runs from 50% to 100%, not from 0%. A jump from 60% to 70% is a large improvement here.",
			"The number after ± is the margin of error, and fewer questions makes it wider. Use the quick run for a rough check and the full run for a number you would quote.",
			"Like HellaSwag and unlike perplexity, this score CAN be compared between different models.",
		},
		Examples: []evalExample{
			{Value: "about 50%", Verdict: "Broken", Meaning: "The same as guessing. The model, the file or the settings are wrong."},
			{Value: "55 to 65%", Verdict: "Weak", Meaning: "Only a little better than guessing."},
			{Value: "70 to 78%", Verdict: "Good", Meaning: "The range the strongest models reached when this test was published."},
			{Value: "80% and above", Verdict: "Strong", Meaning: "Better than the models the test was designed to defeat."},
			{Value: "94%", Verdict: "Human level", Meaning: "What people score on this test."},
		},
		Links: []evalLink{
			{
				URL:   "https://arxiv.org/abs/1907.10641",
				Label: "WinoGrande: the paper behind the test",
				Note:  "Explains what the test asks and why it is hard, and reports the human score.",
			},
		},
	},
}

// evalDocFor returns the guidance for a mode name as stored on a run's
// EvalScores, and whether one exists. A run whose mode this build does
// not recognise renders without guidance rather than with wrong
// guidance.
func evalDocFor(mode string) (evalDoc, bool) {
	d, ok := evalDocs[evaluate.Mode(mode)]
	return d, ok
}

// evalDocForPreset returns the guidance for a preset's mode. Used by
// the About modal's capability table, which lists presets rather than
// stored runs.
func evalDocForPreset(p benchmark.Preset) (evalDoc, bool) {
	d, ok := evalDocs[p.EvalMode]
	return d, ok
}

// PrimaryLink is the source to show where there is room for only one.
func (d evalDoc) PrimaryLink() evalLink {
	if len(d.Links) == 0 {
		return evalLink{}
	}
	return d.Links[0]
}
