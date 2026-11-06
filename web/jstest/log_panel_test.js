// Drives web/static/log-panel.js with the stub document in dom.js: live
// tail scrolls on new text and stops when switched off, Copy uses the
// clipboard API or the textarea fallback and shows a failure, Clear
// empties the panel (after the server, with data-clear-url), and a panel
// added by an htmx swap is set up once. Run by
// internal/api/js_log_panel_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { El, makeDocument } = require('./dom.js');

const doc = makeDocument();
global.window = global;

// MutationObserver: the test calls changed(pre) for new text in pre.
const observers = [];
global.MutationObserver = class {
  constructor(f){ this.f = f; }
  observe(node){ observers.push({node, f: this.f}); }
};
const changed = pre => observers.filter(o => o.node === pre).forEach(o => o.f([]));

// Timers run only when the test says so.
let timers = [];
global.setTimeout = f => { timers.push(f); };
const runTimers = () => { const t = timers; timers = []; t.forEach(f => f()); };
const settle = () => new Promise(r => setImmediate(r));

// Clipboard: the async API and execCommand('copy') record what they got.
let written = [];
let clipboardFails = false;
// Newer node has its own read-only global navigator, so it is redefined.
Object.defineProperty(global, 'navigator', { configurable: true, value: { clipboard: { writeText: async t => {
  if (clipboardFails) throw new Error('not allowed');
  written.push(t);
} } } });
let execCopied = [];
let execOK = true;
doc.execCommand = cmd => {
  assert.strictEqual(cmd, 'copy');
  const ta = doc.body.querySelector('textarea');
  assert.ok(ta, 'execCommand ran without the textarea in the page');
  if (execOK) execCopied.push(ta.value);
  return execOK;
};

let errors = [];
global.showError = (msg, from) => { errors.push({msg, from}); };

let apiCalls = [];
let apiResult = Promise.resolve();
global.apiCall = (url, opts, what) => { apiCalls.push({url, opts, what}); return apiResult; };

function mkPanel(clearUrl){
  const panel = doc.body.appendChild(new El('article')); panel.className = 'log-panel';
  if (clearUrl) panel.dataset.clearUrl = clearUrl;
  const tail = panel.appendChild(new El('input')); tail.className = 'log-tail-toggle'; tail.checked = true;
  const copy = panel.appendChild(new El('button')); copy.className = 'log-copy-btn'; copy.textContent = 'Copy';
  const clear = panel.appendChild(new El('button')); clear.className = 'log-clear-btn';
  const pre = panel.appendChild(new El('pre')); pre.textContent = 'line 1\nline 2\n';
  return {panel, tail, copy, clear, pre};
}

const a = mkPanel('/api/service/logs');
eval(fs.readFileSync(path.join(__dirname, 'log-panel.js'), 'utf8'));
doc.fire('DOMContentLoaded');

// Only copyToClipboard is global; the rest stays inside the script.
assert.strictEqual(typeof window.copyToClipboard, 'function', 'copyToClipboard is not exported');
assert.strictEqual(typeof global.initLogPanels, 'undefined', 'initLogPanels leaked into the global scope');
assert.strictEqual(typeof global.copyWithTextarea, 'undefined', 'copyWithTextarea leaked into the global scope');

// Live tail on: new text scrolls to the bottom.
a.pre.scrollHeight = 500;
changed(a.pre);
assert.strictEqual(a.pre.scrollTop, 500, 'new text did not scroll with live tail on');

// Live tail off: new text leaves the scroll position. On again: back to
// the bottom at once.
a.tail.checked = false; a.tail.dispatch('change');
a.pre.scrollHeight = 900;
changed(a.pre);
assert.strictEqual(a.pre.scrollTop, 500, 'new text scrolled with live tail off');
a.tail.checked = true; a.tail.dispatch('change');
assert.strictEqual(a.pre.scrollTop, 900, 'turning live tail on did not go to the bottom');

(async () => {
  // Copy in a secure context: the clipboard API gets the log text.
  window.isSecureContext = true;
  a.copy.click(); await settle();
  assert.deepStrictEqual(written, ['line 1\nline 2\n']);
  assert.strictEqual(a.copy.textContent, 'Copied!');
  runTimers();
  assert.strictEqual(a.copy.textContent, 'Copy', 'the button label did not come back');

  // The clipboard API refuses: the textarea fallback copies, and the
  // textarea is taken out of the page again.
  clipboardFails = true;
  a.copy.click(); await settle();
  assert.deepStrictEqual(execCopied, ['line 1\nline 2\n'], 'the textarea fallback did not copy');
  assert.strictEqual(doc.body.querySelector('textarea'), null, 'the textarea was left in the page');
  assert.strictEqual(a.copy.textContent, 'Copied!');
  runTimers();

  // Not a secure context (HTTP on a LAN address): the textarea at once.
  clipboardFails = false; written = []; execCopied = [];
  window.isSecureContext = false;
  a.copy.click(); await settle();
  assert.deepStrictEqual(written, [], 'the clipboard API was used outside a secure context');
  assert.deepStrictEqual(execCopied, ['line 1\nline 2\n']);
  runTimers();

  // Both ways fail: the button says so and the error notice tells the user.
  execOK = false;
  a.copy.click(); await settle();
  assert.strictEqual(a.copy.textContent, 'Failed');
  assert.strictEqual(errors.length, 1, 'a failed copy showed no error');
  assert.ok(errors[0].msg.startsWith('Copying the log failed: copy was refused'), errors[0].msg);
  assert.strictEqual(errors[0].from, a.copy);
  runTimers();
  assert.strictEqual(a.copy.textContent, 'Copy', 'the button label did not come back after a failure');
  execOK = true;

  // Clear with a server buffer: the panel empties only after the server
  // has cleared it.
  apiResult = Promise.reject(new Error('500'));
  a.clear.click(); await settle();
  assert.deepStrictEqual(apiCalls.map(c => [c.url, c.opts.method]), [['/api/service/logs', 'DELETE']]);
  assert.strictEqual(a.pre.textContent, 'line 1\nline 2\n', 'a failed clear emptied the panel');
  apiResult = Promise.resolve();
  a.clear.click(); await settle();
  assert.strictEqual(a.pre.textContent, '', 'a clear that worked left the text');

  // A panel added by an htmx swap is set up after htmx:afterSettle; the
  // first panel is not set up a second time.
  const b = mkPanel();
  doc.fire('htmx:afterSettle');
  assert.strictEqual(observers.filter(o => o.node === a.pre).length, 1, 'the first panel was set up twice');
  assert.strictEqual(observers.filter(o => o.node === b.pre).length, 1, 'the new panel was not set up');
  written = []; window.isSecureContext = true;
  a.copy.click(); await settle();
  assert.strictEqual(written.length, 1, 'one click copied more than once');
  runTimers();

  // Clear without a server buffer: the panel empties at once.
  apiCalls = [];
  b.clear.click();
  assert.strictEqual(b.pre.textContent, '');
  assert.deepStrictEqual(apiCalls, [], 'a panel without data-clear-url called the server');

  console.log('ALL PASS');
})().catch(e => { console.error(e); process.exit(1); });
