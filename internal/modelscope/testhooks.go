package modelscope

import (
	"net/http"
	"strings"
)

// This file is compiled into the package rather than kept in a _test.go
// file because the tests that need it live in other packages: an
// end-to-end check that a token saved in Settings reaches the wire has to
// observe what this client sends, and the fields that decide that are
// unexported. Nothing in the shipping code paths calls it.

// SetHTTPClientForTest redirects this client's requests at a stub server,
// keeping the real URL-building and auth code under test.
func (c *Client) SetHTTPClientForTest(hc *http.Client, baseURL string) {
	c.httpClient = hc
	c.httpClient.Transport = testRewriteHost{
		host: strings.TrimPrefix(strings.TrimPrefix(baseURL, "http://"), "https://"),
		rt:   http.DefaultTransport,
	}
}

// testRewriteHost sends every request to host instead of modelscope.cn,
// leaving the path, query and headers the client built untouched.
type testRewriteHost struct {
	host string
	rt   http.RoundTripper
}

func (r testRewriteHost) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = r.host
	return r.rt.RoundTrip(req)
}
