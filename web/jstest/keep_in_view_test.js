// Drives web/static/keep-in-view.js with a small stub document: a swap
// on screen pauses scroll anchoring for two frames and keeps the clicked
// or top element where it was; a click that swaps a
// panel for something shorter brings the panel back into view; polling
// and swaps that leave it on screen do not move the page.
// Run by internal/api/js_keep_in_view_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

class El {
  constructor(id){ this.id=id||''; this.parent=null; this.isConnected=true;
    this.rect={top:0, bottom:100}; this.scrolled=null; this.style={};
    this.overflowY='visible'; this.scrollHeight=0; this.clientHeight=0; }
  get parentElement(){ return this.parent; }
  get children(){ return this.kids || []; }
  get tagName(){ return this.tag || 'DIV'; }
  add(c){ (this.kids=this.kids||[]).push(c); c.parent=this; return c; }
  contains(n){ while(n){ if(n===this) return true; n=n.parent; } return false; }
  getBoundingClientRect(){ return this.rect; }
  scrollIntoView(o){ this.scrolled=o; }
}

const listeners = {};
const byID = {};
const page = new El('page');
global.document = {
  addEventListener(t,f){ (listeners[t]=listeners[t]||[]).push(f); },
  getElementById(id){ return byID[id] || null; },
  body: new El('body'),
  scrollingElement: page,
};
global.window = {innerHeight: 600};
global.getComputedStyle = el => ({overflowY: el.overflowY});
let frames = [];
global.requestAnimationFrame = f => frames.push(f);
function runFrame(){ const f = frames; frames = []; f.forEach(x => x()); }
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

// Scroll anchoring: a swap on screen pauses it on its scroll area for
// two frames, then turns it back on.
const scroller = new El('scroller'); scroller.overflowY = 'auto';
scroller.scrollHeight = 5000; scroller.clientHeight = 600;
const table = new El('gen-instances'); table.parent = scroller; table.rect = {top:-3000, bottom:300};
fire('htmx:beforeSwap', {target: table, elt: table, requestConfig: {elt: table}});
assert.strictEqual(scroller.style.overflowAnchor, 'none', 'anchoring was not paused for a swap on screen');
runFrame();
assert.strictEqual(scroller.style.overflowAnchor, 'none', 'anchoring came back after one frame');
runFrame();
assert.strictEqual(scroller.style.overflowAnchor, '', 'anchoring did not come back after two frames');
fire('htmx:afterSwap', {target: table});

// A swap above the screen keeps anchoring, so its change in height does
// not move what is on screen.
const above = new El('above'); above.parent = scroller; above.rect = {top:-900, bottom:-200};
fire('htmx:beforeSwap', {target: above, elt: above, requestConfig: {elt: above}});
assert.strictEqual(scroller.style.overflowAnchor, '', 'anchoring was paused for a swap above the screen');
assert.strictEqual(frames.length, 0);
fire('htmx:afterSwap', {target: above});

// No scrolling ancestor: the page itself.
const free = new El('free'); free.rect = {top:100, bottom:200};
fire('htmx:beforeSwap', {target: free, elt: free, requestConfig: {elt: free}});
assert.strictEqual(page.style.overflowAnchor, 'none', 'anchoring was not paused on the page');
runFrame(); runFrame();
assert.strictEqual(page.style.overflowAnchor, '');

// The clicked button's row: rows above it get shorter in the new table,
// so it moves up 93 px; the script scrolls up by the same, and it stays.
const sc2 = new El('sc2'); sc2.overflowY = 'auto'; sc2.scrollHeight = 5000; sc2.clientHeight = 600;
sc2.scrollTop = 4361;
const list = new El('list'); list.parent = sc2; list.rect = {top:-3000, bottom:400};
function row(top){ const r = new El(); r.tag = 'TR'; r.rect = {top, bottom: top+280};
  const b = new El(); b.tag = 'BUTTON'; b.rect = {top: top+20, bottom: top+40}; r.add(b); return r; }
list.add(row(-200)); const target = list.add(row(264));
const measure = target.children[0];
fire('htmx:beforeSwap', {target: list, elt: measure, requestConfig: {elt: measure, triggeringEvent: click}});
list.kids = []; list.add(row(-293)); list.add(row(171));
fire('htmx:afterSwap', {target: list});
assert.strictEqual(sc2.scrollTop, 4361 - 93, 'the clicked row was not kept in place');
runFrame(); runFrame();

// A poll: the anchor is the element at the top of what shows.
sc2.scrollTop = 1000; sc2.rect = {top: 0, bottom: 600};
list.kids = []; const first = list.add(row(-100)); list.add(row(300));
global.document.elementFromPoint = () => first.children[0];
fire('htmx:beforeSwap', {target: list, elt: list, requestConfig: {elt: list}});
list.kids = []; list.add(row(-60)); list.add(row(300));
fire('htmx:afterSwap', {target: list});
assert.strictEqual(sc2.scrollTop, 1040, 'the top element of a polled swap was not kept in place');
runFrame(); runFrame();

// The row stays but its button sits lower (a row in another state has
// other buttons): the row is the anchor, so nothing moves.
sc2.scrollTop = 700;
list.kids = []; const r0 = list.add(row(100));
fire('htmx:beforeSwap', {target: list, elt: r0.children[0], requestConfig: {elt: r0.children[0], triggeringEvent: click}});
list.kids = []; list.add(row(100)).children[0].rect = {top: 158, bottom: 178};
fire('htmx:afterSwap', {target: list});
assert.strictEqual(sc2.scrollTop, 700, 'a button that moved inside its row moved the page');
runFrame(); runFrame();

// New content of another shape: tags differ, nothing moves.
sc2.scrollTop = 500;
list.kids = []; const r1 = list.add(row(100));
fire('htmx:beforeSwap', {target: list, elt: r1.children[0], requestConfig: {elt: r1.children[0], triggeringEvent: click}});
list.kids = []; const p = new El(); p.tag = 'P'; p.rect = {top: 100, bottom: 130}; list.add(p);
fire('htmx:afterSwap', {target: list});
assert.strictEqual(sc2.scrollTop, 500, 'a swap to other content moved the page');

console.log('ALL PASS');
