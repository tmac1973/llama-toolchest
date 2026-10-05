package modelsource

import "net/http"

// ResponseIsPartial reports whether a response to a ranged request
// carries only part of the file. The status code alone is not enough:
// ModelScope answers a ranged request with 200 plus a Content-Range header
// rather than the 206 the RFC requires (see modelscope.Client.DownloadURL).
// Reading that 200 as "the server ignored my Range" would make a resumed
// download write only the tail into the file, a corrupt model that looks
// complete. A Content-Range header means partial, whatever the status.
func ResponseIsPartial(resp *http.Response) bool {
	return resp.StatusCode == http.StatusPartialContent || resp.Header.Get("Content-Range") != ""
}
