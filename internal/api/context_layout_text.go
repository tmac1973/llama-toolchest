package api

import (
	"fmt"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// contextLayoutText says in words how the config divides the context
// between conversations, for the line under Parallel Conversations.
// warning is set when the setting can fail at run time or has no
// effect.
func contextLayoutText(cfg *models.ModelConfig, trained int, t models.Target) (text, warning string) {
	l := cfg.ContextLayoutFor(trained, t)
	tok := groupThousands
	switch {
	case l.Pool <= 0:
		return "", ""
	case cfg.Parallel <= 0:
		text = fmt.Sprintf("%d conversations at once (llama.cpp's default), sharing %s tokens of context. One conversation can use all of it.",
			l.Slots, tok(l.Pool))
	case l.Slots == 1:
		text = fmt.Sprintf("One conversation at a time, with all %s tokens of context. Other requests wait until it finishes.", tok(l.Pool))
	case !l.Shared:
		text = fmt.Sprintf("%d conversations at once, each with its own %s tokens (%s ÷ %d).",
			l.Slots, tok(l.PerConversation), tok(l.Pool), l.Slots)
	default:
		text = fmt.Sprintf("%d conversations at once, sharing %s tokens. Each can use up to %s.",
			l.Slots, tok(l.Pool), tok(l.PerConversation))
		if cfg.ContextPerSlot > 0 && !t.ContextPerSlotOption() {
			warning = "The active llama.cpp build is older than b10662 and has no per-conversation limit, so each conversation can use the whole pool."
		} else if cfg.ContextPerSlot > l.Pool {
			warning = "The limit is larger than the context, so it has no effect."
		}
		if need := l.Slots * l.PerConversation; need > l.Pool && warning == "" {
			warning = fmt.Sprintf("If all %d reach that at once they need %s tokens, %.1f× the context. llama.cpp then clears waiting conversations, which must re-read their prompt, and if that is not enough a request fails with an error.",
				l.Slots, tok(need), float64(need)/float64(l.Pool))
		}
	}
	return text, warning
}
