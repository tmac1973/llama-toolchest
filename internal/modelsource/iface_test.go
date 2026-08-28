package modelsource_test

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelscope"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Both clients must stay interchangeable: the whole design rests on the
// API layer being able to pick one by source id without any downstream
// handler, template or struct knowing which it got.
func TestBothClientsImplementSource(t *testing.T) {
	var _ modelsource.Client = huggingface.NewClient("")
	var _ modelsource.Client = modelscope.NewClient("")
}
