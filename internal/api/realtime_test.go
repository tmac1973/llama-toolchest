package api

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// serveRealtime answers a WebSocket upgrade as llama-server does (no
// subprotocol in the answer), then echoes what the client sends.
func (f *fakeRouter) serveRealtime(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.realtimeProtocols = append(f.realtimeProtocols, r.Header.Get("Sec-WebSocket-Protocol"))
	f.mu.Unlock()
	conn, brw, err := http.NewResponseController(w).Hijack()
	if err != nil {
		return
	}
	defer conn.Close()
	brw.WriteString("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: x\r\n\r\n")
	brw.Flush()
	io.Copy(conn, brw)
}

func (f *fakeRouter) realtimeCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.realtimeProtocols...)
}

// dialRealtime opens a WebSocket handshake to path on srv with the given
// subprotocols and Authorization header, and returns the connection and
// the answer's status line and headers.
func dialRealtime(t *testing.T, srv *httptest.Server, path, protocols, auth string) (net.Conn, *bufio.Reader, *http.Response) {
	t.Helper()
	u, _ := url.Parse(srv.URL)
	conn, err := net.Dial("tcp", u.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	req := "GET " + path + " HTTP/1.1\r\nHost: " + u.Host +
		"\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nAccept-Encoding: gzip\r\n"
	if protocols != "" {
		req += "Sec-WebSocket-Protocol: " + protocols + "\r\n"
	}
	if auth != "" {
		req += "Authorization: " + auth + "\r\n"
	}
	if _, err := io.WriteString(conn, req+"\r\n"); err != nil {
		t.Fatal(err)
	}
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, nil)
	if err != nil {
		t.Fatal(err)
	}
	return conn, br, resp
}

// A realtime session loads its model from the query, holds the model's
// use count until it closes, and may send the API key as a subprotocol,
// which does not reach llama-server.
func TestRealtimeSession(t *testing.T) {
	s, f, m := startFakeRouter(t)
	s.cfg.APIKey = "secret"
	srv := httptest.NewServer(s.buildRouter())
	t.Cleanup(srv.Close)
	f.reset(map[string]string{m.ID: "unloaded"}, func(f *fakeRouter, n int) {
		if f.status[m.ID] == "loading" {
			f.status[m.ID] = "loaded"
		}
	})
	path := realtimePath + "?model=" + url.QueryEscape(m.PublicName())

	for _, c := range []struct{ name, protocols, auth string }{
		{"no key", "realtime", ""},
		{"wrong key", "realtime, " + realtimeKeyProtocol + "nope", ""},
	} {
		_, _, resp := dialRealtime(t, srv, path, c.protocols, c.auth)
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s: status %d, want 401", c.name, resp.StatusCode)
		}
	}
	// The subprotocol is taken only on the realtime route.
	if _, _, resp := dialRealtime(t, srv, "/v1/models", realtimeKeyProtocol+"secret", ""); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("/v1/models with the key as a subprotocol: status %d, want 401", resp.StatusCode)
	}
	if len(f.realtimeCalls()) != 0 || len(f.loadCalls()) != 0 {
		t.Fatal("a session without the key reached the router")
	}

	conn, br, resp := dialRealtime(t, srv, path, "realtime, "+realtimeKeyProtocol+"secret", "")
	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("status %d, want 101", resp.StatusCode)
	}
	if got := resp.Header.Get("Sec-WebSocket-Protocol"); got != realtimeProtocol {
		t.Errorf("answer subprotocol %q, want %q", got, realtimeProtocol)
	}
	if got := f.realtimeCalls(); len(got) != 1 || got[0] != realtimeProtocol {
		t.Errorf("llama-server got subprotocols %q, want only %q", got, realtimeProtocol)
	}
	if loads := f.loadCalls(); len(loads) != 1 || loads[0] != s.registry.RouterName(m.ID) {
		t.Errorf("load calls = %v, want one for %s", loads, m.ID)
	}

	// The session carries data both ways.
	io.WriteString(conn, "ping")
	buf := make([]byte, 4)
	if _, err := io.ReadFull(br, buf); err != nil || string(buf) != "ping" {
		t.Fatalf("echo = %q, %v", buf, err)
	}

	// While it is open, an unload for a generation job waits.
	short, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := s.llmUse.evict(short, m.ID); err == nil {
		t.Fatal("an unload did not wait for an open realtime session")
	}

	// Once it is closed, the unload goes ahead.
	conn.Close()
	testutil.Eventually(t, func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		done, err := s.llmUse.evict(ctx, m.ID)
		if err == nil {
			done()
		}
		return err == nil
	}, "the session's count was not released when it closed")
}

// With the Authorization header, as the OpenAI Python SDK sends it, and
// no subprotocols offered, none is added to the answer.
func TestRealtimeSessionWithBearerKey(t *testing.T) {
	s, f, m := startFakeRouter(t)
	s.cfg.APIKey = "secret"
	srv := httptest.NewServer(s.buildRouter())
	t.Cleanup(srv.Close)
	f.reset(map[string]string{m.ID: "loaded"}, nil)

	_, _, resp := dialRealtime(t, srv, realtimePath+"?model="+url.QueryEscape(m.ID), "", "Bearer secret")
	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("status %d, want 101", resp.StatusCode)
	}
	if got := resp.Header.Get("Sec-WebSocket-Protocol"); got != "" {
		t.Errorf("answer subprotocol %q, want none", got)
	}
	if got := f.realtimeCalls(); len(got) != 1 || got[0] != "" {
		t.Errorf("llama-server got subprotocols %q", got)
	}
}

func TestWebsocketProtocols(t *testing.T) {
	h := http.Header{}
	h.Add("Sec-WebSocket-Protocol", "realtime, openai-insecure-api-key.k")
	h.Add("Sec-WebSocket-Protocol", " openai-beta.realtime-v1 ,")
	if got := strings.Join(websocketProtocols(h), "|"); got != "realtime|openai-insecure-api-key.k|openai-beta.realtime-v1" {
		t.Errorf("protocols = %q", got)
	}
}
