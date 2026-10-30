// Package enginenotes keeps a record of what llama-server's router and
// its model instances did: which models loaded, which failed to load,
// which crashed and which were unloaded, and when the router itself
// stopped or died.
//
// The router's output is held only in a short buffer in memory, emptied
// on every router start, and benchmark jobs restart the router for every
// cell. A crash is usually gone from that buffer within minutes, and a
// crash of one model instance does not stop the router, which just loads
// the model again on the next request. This package reads the log as it
// arrives and keeps what matters, so the count and the reason survive.
//
// It imports nothing from this project but internal/serverlog, which
// knows the shape of the router's lines, and internal/atomicfile, which
// writes the saved copy (engine-notes.json) safely.
package enginenotes
