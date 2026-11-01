// Drives web/static/errors.js with a small stub document: htmx events
// in, the error notice out. Run by internal/api/js_errors_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

class El {
  constructor(tag){ this.tag=tag; this.children=[]; this.attrs={}; this.parent=null;
    this.className=''; this.textContent=''; this.hidden=false; this.listeners={}; }
  appendChild(c){ c.parent=this; this.children.push(c); return c; }
  insertBefore(n, ref){ if(n.parent) n.parent.children.splice(n.parent.children.indexOf(n),1);
    n.parent=this; const i=ref?this.children.indexOf(ref):-1;
    if(i<0) this.children.push(n); else this.children.splice(i,0,n); return n; }
  get firstChild(){ return this.children[0] || null; }
  get parentNode(){ return this.parent; }
  get nextSibling(){ if(!this.parent) return null; const s=this.parent.children; return s[s.indexOf(this)+1] || null; }
  setAttribute(k,v){ this.attrs[k]=String(v); }
  getAttribute(k){ return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(t,f){ (this.listeners[t]=this.listeners[t]||[]).push(f); }
  click(){ (this.listeners.click||[]).forEach(f=>f()); }
  // closest and querySelector know "[attr]", "tag[attr]", ".class" and
  // "tag", which is all errors.js asks for.
  closest(sel){ const m=sel.match(/^(\w*)\[([\w-]+)\]$/); let n=this;
    while(n){ if(m && (!m[1] || n.tag===m[1]) && n.getAttribute(m[2])!==null) return n; n=n.parent; } return null; }
  querySelector(sel){ const all=[];
    (function walk(e){ for(const c of e.children){ all.push(c); walk(c); } })(this);
    if(sel.startsWith('.')) return all.find(e=>e.className===sel.slice(1)) || null;
    return all.find(e=>e.tag===sel) || null; }
}

const main = new El('main');
const listeners = {};
global.document = {
  addEventListener(t,f){ (listeners[t]=listeners[t]||[]).push(f); },
  createElement(tag){ return new El(tag); },
  querySelector(sel){ return sel==='main' ? main : null; },
  body: main,
};
global.window = global;
const fire = (type, detail) => (listeners[type]||[]).forEach(f=>f({detail}));
const xhr = (status, text, ct) => ({status, responseText: text, getResponseHeader: () => ct || 'text/plain; charset=utf-8'});

eval(fs.readFileSync(path.join(__dirname, 'errors.js'), 'utf8'));

const notice = () => main.children.find(c=>c.attrs.id==='error-notice' || c.id==='error-notice');
const text = () => notice().querySelector('.error-notice-text').textContent;

// A refused htmx action shows the server's plain-text reason.
const download = new El('button'); download.setAttribute('data-error-label', 'Download refused');
fire('htmx:responseError', {elt: download, xhr: xhr(507, 'not enough disk space: the model needs 41.0 GiB\n')});
assert.ok(notice(), 'no notice was added');
assert.strictEqual(notice().hidden, false);
assert.strictEqual(text(), 'Download refused: not enough disk space: the model needs 41.0 GiB');

// The same element succeeding clears it; another element succeeding does not.
fire('htmx:afterRequest', {elt: new El('button'), successful: true});
assert.strictEqual(notice().hidden, false, 'an unrelated success cleared the notice');
fire('htmx:afterRequest', {elt: download, successful: true});
assert.strictEqual(notice().hidden, true, 'a retry that worked did not clear the notice');

// JSON errors show their message; an empty body names the status.
const btn = new El('button');
fire('htmx:responseError', {elt: btn, xhr: xhr(401, '{"error":{"message":"invalid API key"}}', 'application/json')});
assert.strictEqual(text(), 'Request failed: invalid API key');
fire('htmx:responseError', {elt: btn, xhr: xhr(500, '')});
assert.strictEqual(text(), 'Request failed: The server answered with status 500.');

// The close button hides it.
notice().children.find(c=>c.className==='error-notice-close').click();
assert.strictEqual(notice().hidden, true);

// Polling and elements that show their own errors are left alone.
const poll = new El('div'); poll.setAttribute('hx-trigger', 'load, every 3s');
fire('htmx:responseError', {elt: poll, xhr: xhr(503, 'down')});
fire('htmx:sendError', {elt: poll});
assert.strictEqual(notice().hidden, true, 'a poll showed an error');
const form = new El('form'); form.setAttribute('data-own-errors', '');
const field = form.appendChild(new El('input'));
fire('htmx:responseError', {elt: field, xhr: xhr(400, 'bad value')});
assert.strictEqual(notice().hidden, true, 'a self-handling form showed an error');

// An unreachable server says so.
fire('htmx:sendError', {elt: btn});
assert.strictEqual(text(), 'Request failed: the server could not be reached.');

// A request from inside an open dialog shows the notice in the dialog,
// under its header: a modal dialog covers the page. A later error from
// outside puts it back at the top of the page.
const theNotice = notice();
const dialog = main.appendChild(new El('dialog')); dialog.setAttribute('open', '');
const article = dialog.appendChild(new El('article'));
const header = article.appendChild(new El('header'));
const dialogForm = article.appendChild(new El('form')); dialogForm.setAttribute('data-error-label', 'Set not added');
fire('htmx:responseError', {elt: dialogForm, xhr: xhr(400, 'upscaler: /home/x.pth does not exist')});
assert.strictEqual(theNotice.parent, article, 'the notice is not in the dialog');
assert.strictEqual(article.children.indexOf(theNotice), 1, 'the notice is not under the dialog header');
assert.ok(!main.children.includes(theNotice), 'the notice is still at the top of the page too');
assert.strictEqual(theNotice.hidden, false);
fire('htmx:responseError', {elt: dialogForm, xhr: xhr(400, 'again')});
assert.strictEqual(article.children.indexOf(theNotice), 1, 'a second error moved the notice');
fire('htmx:responseError', {elt: btn, xhr: xhr(500, 'outside')});
assert.strictEqual(theNotice.parent, main, 'an error outside the dialog left the notice in it');
assert.strictEqual(main.firstChild, theNotice);

// apiCall: OK resolves; an error shows the notice and rejects.
(async () => {
  global.fetch = async () => ({ok: true, status: 200});
  const r = await apiCall('/x', {}, 'Cancel failed');
  assert.strictEqual(r.status, 200);

  global.fetch = async () => ({ok: false, status: 409, headers: {get: () => 'text/plain'},
    text: async () => 'this job is running — cancel it before deleting'});
  let rejected = false;
  await apiCall('/x', {}, 'Delete failed').catch(() => { rejected = true; });
  assert.ok(rejected, 'apiCall resolved on an error');
  assert.strictEqual(text(), 'Delete failed: this job is running — cancel it before deleting');
  console.log('ALL PASS');
})().catch(e => { console.error(e); process.exit(1); });
