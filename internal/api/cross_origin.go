package api

import "net/http"

// crossOriginProtection rejects state-changing browser requests that come
// from another site, so a page elsewhere cannot use a visitor's browser to
// change settings, restore a backup or start jobs here (CSRF). It uses the
// browser's Sec-Fetch-Site header, or the Origin header compared with Host;
// requests with neither, such as curl or an SDK, are not browser requests
// and pass. GET, HEAD and OPTIONS always pass.
//
// The OpenAI-compatible /v1 routes and the ACE-Step job routes are left
// out: they take the API key instead, and are called by API clients rather
// than by this server's pages.
func crossOriginProtection(next http.Handler) http.Handler {
	p := http.NewCrossOriginProtection()
	p.AddInsecureBypassPattern("/v1/")
	p.AddInsecureBypassPattern("POST /release_task")
	p.AddInsecureBypassPattern("POST /query_result")
	p.SetDenyHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "cross-site request refused: open this page from the toolchest itself", http.StatusForbidden)
	}))
	return p.Handler(next)
}
