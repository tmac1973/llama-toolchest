// Drives web/static/live-stream.js with a fake EventSource and document:
// tab visibility and server events in, connections out. Run by
// internal/api/js_live_stream_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

// FakeSource records every connection the page opens.
const opened = [];
class FakeSource {
  constructor(url, options) {
    this.url = url; this.options = options; this.readyState = 0; this.closed = false;
    this.listeners = {}; this.onopen = this.onmessage = this.onerror = null;
    opened.push(this);
  }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  close() { this.closed = true; this.readyState = 2; }
  // Server side of the fake.
  open() { this.readyState = 1; this.onopen && this.onopen({ type: 'open' }); }
  send(type, data) {
    const e = { type, data };
    if (type === 'message') { this.onmessage && this.onmessage(e); return; }
    (this.listeners[type] || []).forEach(f => f(e));
  }
  fail(readyState) { this.readyState = readyState; this.onerror && this.onerror({ type: 'error' }); }
}
const live = () => opened.filter(s => !s.closed);

const docListeners = {};
global.document = {
  hidden: false,
  addEventListener(t, f) { (docListeners[t] = docListeners[t] || []).push(f); },
  removeEventListener(t, f) { docListeners[t] = (docListeners[t] || []).filter(g => g !== f); },
};
const setHidden = hidden => {
  document.hidden = hidden;
  (docListeners.visibilitychange || []).slice().forEach(f => f());
};
global.htmx = {};
global.window = global;
window.EventSource = FakeSource;

eval(fs.readFileSync(path.join(__dirname, 'live-stream.js'), 'utf8'));

// A stream connects at once in a shown tab and passes events on.
const got = [];
let opens = 0;
const s = new LiveStream('/api/service/logs');
s.onopen = () => opens++;
s.onmessage = e => got.push(e.data);
s.addEventListener('progress', e => got.push('progress:' + e.data));
assert.strictEqual(live().length, 1);
const first = live()[0];
first.open();
first.send('message', 'line 1');
first.send('progress', '50%');
assert.deepStrictEqual(got, ['line 1', 'progress:50%']);
assert.strictEqual(opens, 1);
assert.strictEqual(s.readyState, LiveStream.OPEN);

// Hiding the tab closes the connection; showing it opens a new one, with
// the same handlers and named-event listeners.
setHidden(true);
assert.strictEqual(first.closed, true, 'the connection stayed open in a hidden tab');
assert.strictEqual(live().length, 0);
assert.strictEqual(s.readyState, LiveStream.CONNECTING);
setHidden(false);
assert.strictEqual(live().length, 1, 'the stream did not open again when the tab was shown');
const second = live()[0];
second.open();
second.send('progress', '60%');
assert.strictEqual(opens, 2);
assert.deepStrictEqual(got.slice(-1), ['progress:60%']);

// A late event from the closed connection is ignored.
first.send('message', 'stale');
assert.ok(!got.includes('stale'));

// The done event is passed on, then the stream closes for good and stays
// closed when the tab is hidden and shown again.
let doneMsg = null;
s.addEventListener('done', e => { doneMsg = e.data; });
second.send('done', 'Build complete');
assert.strictEqual(doneMsg, 'Build complete');
assert.strictEqual(second.closed, true);
assert.strictEqual(s.readyState, LiveStream.CLOSED);
setHidden(true); setHidden(false);
assert.strictEqual(live().length, 0, 'a finished stream connected again');
assert.strictEqual((docListeners.visibilitychange || []).length, 0, 'a finished stream still watches the tab');

// A stream made in a hidden tab waits until the tab is shown.
setHidden(true);
const waiting = new LiveStream('/api/builds/x/logs');
assert.strictEqual(live().length, 0);
setHidden(false);
assert.strictEqual(live().length, 1);

// A refused stream (not found) reports the error as closed, so htmx can
// decide to try again; a passing network error leaves it connecting.
let errors = 0;
waiting.onerror = () => errors++;
live()[0].fail(0);
assert.strictEqual(waiting.readyState, LiveStream.CONNECTING);
live()[0].fail(2);
assert.strictEqual(errors, 2);
assert.strictEqual(waiting.readyState, LiveStream.CLOSED);
setHidden(true); setHidden(false);
assert.strictEqual(live().length, 0, 'a refused stream connected again by itself');

// htmx's SSE extension gets LiveStreams.
const h = htmx.createEventSource('/api/builds/y/logs');
assert.ok(h instanceof LiveStream);
assert.strictEqual(live()[0].options.withCredentials, true);
h.close();
assert.strictEqual(live().length, 0);

// An htmx panel marked data-sse-replay is emptied when its stream opens;
// other elements are left alone.
const panel = { textContent: 'old lines', hasAttribute: a => a === 'data-sse-replay' };
const other = { textContent: 'keep', hasAttribute: () => false };
docListeners['htmx:sseOpen'].forEach(f => { f({ target: panel }); f({ target: other }); });
assert.strictEqual(panel.textContent, '');
assert.strictEqual(other.textContent, 'keep');

console.log('ALL PASS');
