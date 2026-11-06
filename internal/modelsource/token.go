package modelsource

import (
	"net/http"
	"sync"
)

// BearerToken is an access token that Settings can replace while requests
// run, so a token saved there applies to the next request rather than the
// next restart. The zero value has no token. The model source clients and
// the preset fetcher embed it.
type BearerToken struct {
	mu    sync.RWMutex
	token string
}

// SetToken replaces the token; "" removes it.
func (t *BearerToken) SetToken(token string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.token = token
}

// Token returns the current token, or "".
func (t *BearerToken) Token() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.token
}

// SetAuth adds "Authorization: Bearer <token>" to req when there is a
// token.
func (t *BearerToken) SetAuth(req *http.Request) {
	if tok := t.Token(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
}
