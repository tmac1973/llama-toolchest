package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// A reranker whose file has no classification head is flagged on its
// config page and on its card, with the same text; one with a head, and
// a record whose tensor table was never read, are not.
func TestRerankHeadWarning(t *testing.T) {
	s := newTestServer(t)
	headless := &models.Model{ID: "headless", ModelID: "org/Qwen3-Reranker-0.6B-GGUF", Filename: "h.gguf",
		Arch: "qwen3", ClassifierChecked: true}
	good := &models.Model{ID: "good", ModelID: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", Filename: "g.gguf",
		Arch: "qwen3", PoolingType: models.PoolingRank, ClassifierChecked: true, HasClassifierHead: true,
		HasChatTemplate: true, SupportsTools: true} // Qwen3's template handles tools
	unread := &models.Model{ID: "unread", ModelID: "gpustack/bge-reranker-v2-m3-GGUF", Filename: "u.gguf", Arch: "bert"}
	for _, m := range []*models.Model{headless, good, unread} {
		if err := s.registry.Add(m); err != nil {
			t.Fatal(err)
		}
	}

	for _, tt := range []struct {
		id   string
		want bool
	}{{"headless", true}, {"good", false}, {"unread", false}} {
		data, err := s.configPanelData(tt.id)
		if err != nil {
			t.Fatal(err)
		}
		rec := httptest.NewRecorder()
		s.renderPartial(rec, "model_config", data)
		if got := strings.Contains(rec.Body.String(), rerankHeadWarning); got != tt.want {
			t.Errorf("%s: config page shows the warning = %v, want %v", tt.id, got, tt.want)
		}

		m, _ := s.registry.Get(tt.id)
		rec = httptest.NewRecorder()
		s.renderModelCard(rec, m, nil, false, false)
		card := rec.Body.String()
		if got := strings.Contains(card, "check file"); got != tt.want {
			t.Errorf("%s: card shows the badge = %v, want %v", tt.id, got, tt.want)
		}
		if strings.Contains(card, ">tools</mark>") {
			t.Errorf("%s: a reranker's card shows the tools badge", tt.id)
		}
		if tt.want && !strings.Contains(card, `title="`+rerankHeadWarning+`"`) {
			t.Errorf("%s: the badge does not carry the warning text:\n%s", tt.id, card)
		}
	}
}

// The capability object says what a reranker is, so a client can pick
// one for /v1/rerank.
func TestBuildCapabilitiesRerank(t *testing.T) {
	s := newTestServer(t)
	// Qwen3-Reranker keeps Qwen3's chat template, which handles tools.
	m := &models.Model{ID: "r", ModelID: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", Arch: "qwen3", ContextLength: 40960,
		PoolingType: models.PoolingRank, HasChatTemplate: true, SupportsTools: true}
	caps := s.buildCapabilities(m, nil)
	if caps["rerank"] != true || caps["embedding"] != false || caps["chat"] != false || caps["tools"] != false {
		t.Errorf("rerank %v, embedding %v, chat %v, tools %v; want true, false, false, false",
			caps["rerank"], caps["embedding"], caps["chat"], caps["tools"])
	}
	chat := s.buildCapabilities(&models.Model{ID: "c", ModelID: "org/Chat-GGUF", HasChatTemplate: true, SupportsTools: true}, nil)
	if chat["rerank"] != false || chat["tools"] != true {
		t.Errorf("a chat model reports rerank %v, tools %v; want false, true", chat["rerank"], chat["tools"])
	}
}
