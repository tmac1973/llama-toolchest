package models

import "strconv"

// ContextLayout is how llama-server divides a model's context between
// the conversations it serves at once, as launched from a config.
type ContextLayout struct {
	// Slots is how many conversations run at the same time.
	Slots int
	// Shared is true when the slots draw from one pool (--kv-unified)
	// instead of each owning an equal share.
	Shared bool
	// Pool is the whole context, in tokens.
	Pool int
	// PerConversation is the most one conversation can use: a client
	// compacts on this.
	PerConversation int
}

// ContextLayoutFor works out the layout the config launches with. trained
// is the model's trained context, used when ContextSize is 0. t decides
// whether the build has --kv-unified-per-slot.
//
// Parallel 0 writes nothing, and llama-server then runs four slots that
// share the whole context. Parallel 1 is one conversation at a time. From
// 2, each slot owns an equal share unless SharedContext is on.
func (c *ModelConfig) ContextLayoutFor(trained int, t Target) ContextLayout {
	ctx := c.ContextSize
	if ctx <= 0 {
		ctx = trained
	}
	switch {
	case c.Parallel <= 0:
		return ContextLayout{Slots: defaultServerSlots, Shared: true, Pool: ctx, PerConversation: ctx}
	case c.Parallel == 1:
		return ContextLayout{Slots: 1, Pool: ctx, PerConversation: ctx}
	case !c.SharedContext:
		return ContextLayout{Slots: c.Parallel, Pool: ctx, PerConversation: ctx / c.Parallel}
	}
	limit := 0
	if t.contextPerSlotOption() {
		limit = c.ContextPerSlot
	}
	// Without a context size, llama-server sizes the pool to hold every
	// conversation at its limit.
	if c.ContextSize <= 0 && limit > 0 {
		ctx = c.Parallel * limit
	}
	per := ctx
	if limit > 0 && limit < per {
		per = limit
	}
	if trained > 0 && per > trained {
		per = trained
	}
	return ContextLayout{Slots: c.Parallel, Shared: true, Pool: ctx, PerConversation: per}
}

// sharedContextParams returns the options for the config's slots:
// parallel from 1 (0 leaves llama-server's own four shared slots), and
// kv-unified with the per-conversation limit when the context is shared.
func sharedContextParams(c *ModelConfig, t Target) []specParam {
	if c.Parallel < 1 {
		return nil
	}
	out := []specParam{{"parallel", strconv.Itoa(c.Parallel)}}
	if c.Parallel > 1 && c.SharedContext {
		out = append(out, specParam{"kv-unified", "true"})
		if c.ContextPerSlot > 0 && t.contextPerSlotOption() {
			out = append(out, specParam{"kv-unified-per-slot", strconv.Itoa(c.ContextPerSlot)})
		}
	}
	return out
}
