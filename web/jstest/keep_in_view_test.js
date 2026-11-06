// Drives web/static/keep-in-view.js with a small stub document: a click
// that swaps a panel for something shorter brings the panel back into
// view; polling and swaps that leave it on screen do not move the page.
// Run by internal/api/js_keep_in_view_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

class El {
  constructor(id){ this.id=id||''; this.parent=null; this.isConnected=true;
    this.rect={top:0, bottom:100}; this.scrolled=null; }
  contains(n){ while(n){ if(n===this) return true; n=n.parent; } return false; }
  getBoundingClientRect(){ return this.rect; }
  scrollIntoView(o){ this.scrolled=o; }
}

const listeners = {};
const byID = {};
global.document = {
  addEventListener(t,f){ (listeners[t]=listeners[t]||[]).push(f); },
  getElementById(id){ return byID[id] || null; },
};
const fire = (type, detail) => (listeners[type]||[]).forEach(f=>f({detail}));
eval(fs.readFileSync(path.join(__dirname, 'keep-in-view.js'), 'utf8'));

const click = {type:'click', isTrusted:true};
function swap(target, elt, ev, after){
  fire('htmx:beforeSwap', {target, elt, requestConfig:{elt, triggeringEvent:ev}});
  after();
  fire('htmx:afterSwap', {target});
}

// Save and apply: the review becomes a short message above the screen.
const panel = new El('autoconfig-m'); const btn = new El(); btn.parent = panel;
swap(panel, btn, click, () => { panel.rect = {top:-527, bottom:-407}; });
assert.deepStrictEqual(panel.scrolled, {block:'start'}, 'the panel above the screen was not brought back');

// The panel still on screen after the swap: no scroll.
const p2 = new El('p2'); const b2 = new El(); b2.parent = p2;
swap(p2, b2, click, () => { p2.rect = {top:-300, bottom:200}; });
assert.strictEqual(p2.scrolled, null, 'a panel that is still on screen was scrolled');

// Polling: no trusted event, never moves the page.
const p3 = new El('p3');
swap(p3, p3, undefined, () => { p3.rect = {top:-500, bottom:-400}; });
assert.strictEqual(p3.scrolled, null, 'a polled swap moved the page');

// A click outside the target (a button elsewhere filling a panel): no scroll.
const p4 = new El('p4'); const b4 = new El();
swap(p4, b4, click, () => { p4.rect = {top:-500, bottom:-400}; });
assert.strictEqual(p4.scrolled, null, 'a swap of a panel the user did not act in moved the page');

// outerHTML: the old target is gone, the new one is found by id.
const old = new El('card-x'); const b5 = new El(); b5.parent = old;
const fresh = new El('card-x'); fresh.rect = {top:-800, bottom:-760};
swap(old, b5, click, () => { old.isConnected = false; byID['card-x'] = fresh; });
assert.deepStrictEqual(fresh.scrolled, {block:'start'}, 'the new element of an outerHTML swap was not brought back');

// An element with its own show:none (the Configure form): left alone.
const p6 = new El('p6'); const b6 = new El(); b6.parent = p6;
b6.closest = () => ({getAttribute: () => 'outerHTML focus-scroll:false show:none'});
swap(p6, b6, click, () => { p6.rect = {top:-500, bottom:-400}; });
assert.strictEqual(p6.scrolled, null, 'an element with show:none was scrolled');

console.log('ALL PASS');
