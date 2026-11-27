package huggingface

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// The file client must not use HTTP/2, even with a server that offers
// it: Go puts all HTTP/2 downloads from one host on one connection, and
// Hugging Face's CDN then starves all but one of them.
func TestFileClientUsesHTTP1(t *testing.T) {
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.Proto))
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	// Trust the test server's certificate, but keep the rest of the
	// file client's TLS config: what it offers the server in ALPN is
	// what this test checks.
	tr := fileClient.Transport.(*http.Transport).Clone()
	tr.TLSClientConfig.RootCAs = srv.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs
	client := &http.Client{Transport: tr}

	// The test server's own client does use HTTP/2, so the test can see
	// the difference.
	resp, err := srv.Client().Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.ProtoMajor != 2 {
		t.Skipf("test server did not offer HTTP/2 (got %s)", resp.Proto)
	}

	resp, err = client.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.ProtoMajor != 1 {
		t.Fatalf("file client used %s, want HTTP/1.1", resp.Proto)
	}
}
