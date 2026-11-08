package api

import (
	"sync"
	"time"
)

// forkUse keeps when each fork instance last took a chat request, so the
// virtual model name can go to the running model used last. Router
// models have the same in llmUse.
type forkUse struct {
	mu   sync.Mutex
	last map[string]time.Time
	// now is the clock; nil means time.Now. Tests set it so two uses get
	// different times without a sleep.
	now func() time.Time
}

// note counts fork instance id as used now.
func (u *forkUse) note(id string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.last == nil {
		u.last = map[string]time.Time{}
	}
	at := time.Now()
	if u.now != nil {
		at = u.now()
	}
	u.last[id] = at
}

// lastUse is when fork instance id last took a chat request; zero when
// never.
func (u *forkUse) lastUse(id string) time.Time {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.last[id]
}
