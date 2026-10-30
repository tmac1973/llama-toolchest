// Drives web/static/log-resize.js with a fake section, grip, window and
// storage: drags and double-clicks in, height and saved value out. Run by
// internal/api/js_log_resize_test.go.
const assert = require('assert');
const fs = require('fs');
const path = require('path');

class Target {
  constructor() { this.listeners = {}; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, e) { (this.listeners[t] || []).forEach(f => f(Object.assign({ preventDefault() {} }, e))); }
}

function fakeSection(offsetHeight) {
  const classes = new Set();
  return {
    style: { height: '' },
    offsetHeight,
    classList: { add: c => classes.add(c), remove: c => classes.delete(c), has: c => classes.has(c) },
  };
}

function fakeStorage(initial) {
  const data = Object.assign({}, initial);
  return {
    data,
    getItem: k => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = v; },
    removeItem: k => { delete data[k]; },
  };
}

const throwingStorage = {
  getItem() { throw new Error('denied'); },
  setItem() { throw new Error('denied'); },
  removeItem() { throw new Error('denied'); },
};

global.window = global;
global.document = new Target();
eval(fs.readFileSync(path.join(__dirname, 'log-resize.js'), 'utf8'));
const { clampHeight, init } = window.LogResize;
const KEY = 'llama-toolchest.serverLogHeight';

// clampHeight: 300 is the floor, the viewport less 120 the ceiling, and
// the result is a whole pixel.
assert.strictEqual(clampHeight(100, 1000), 300);
assert.strictEqual(clampHeight(2000, 1000), 880);
assert.strictEqual(clampHeight(512.6, 1000), 513);
// A viewport too small for both limits still gets the floor.
assert.strictEqual(clampHeight(500, 350), 300);

// Nothing saved: the section keeps its default size.
{
  const section = fakeSection(400);
  init(section, new Target(), fakeStorage({}), Object.assign(new Target(), { innerHeight: 1000 }));
  assert.strictEqual(section.style.height, '');
  assert.ok(!section.classList.has('sized'));
}

// Storage that refuses access: no error, default size.
{
  const section = fakeSection(400);
  init(section, new Target(), throwingStorage, Object.assign(new Target(), { innerHeight: 1000 }));
  assert.strictEqual(section.style.height, '');
}

// A saved height is restored, clamped to the window.
{
  const section = fakeSection(400);
  init(section, new Target(), fakeStorage({ [KEY]: '520' }), Object.assign(new Target(), { innerHeight: 1000 }));
  assert.strictEqual(section.style.height, '520px');
  assert.ok(section.classList.has('sized'));

  const small = fakeSection(400);
  init(small, new Target(), fakeStorage({ [KEY]: '900' }), Object.assign(new Target(), { innerHeight: 800 }));
  assert.strictEqual(small.style.height, '680px');
}

// Dragging sets and saves the height; a window that shrinks and grows
// again gives the chosen height back; a double-click forgets it.
{
  const section = fakeSection(400);
  const grip = new Target();
  const storage = fakeStorage({});
  const win = Object.assign(new Target(), { innerHeight: 1000 });
  init(section, grip, storage, win);

  grip.fire('pointerdown', { clientY: 600, pointerId: 1 });
  grip.fire('pointermove', { clientY: 900 });
  assert.strictEqual(section.style.height, '700px');
  assert.ok(section.classList.has('sized'));
  assert.strictEqual(storage.data[KEY], undefined, 'saved only when the drag ends');
  grip.fire('pointerup', {});
  assert.strictEqual(storage.data[KEY], '700');

  // Moves after the drag ended change nothing.
  grip.fire('pointermove', { clientY: 100 });
  assert.strictEqual(section.style.height, '700px');

  win.innerHeight = 600;
  win.fire('resize', {});
  assert.strictEqual(section.style.height, '480px');
  win.innerHeight = 1000;
  win.fire('resize', {});
  assert.strictEqual(section.style.height, '700px');

  grip.fire('dblclick', {});
  assert.strictEqual(section.style.height, '');
  assert.ok(!section.classList.has('sized'));
  assert.strictEqual(storage.data[KEY], undefined);

  // After a reset, a resize leaves the default size alone.
  win.fire('resize', {});
  assert.strictEqual(section.style.height, '');
}

// Dragging up stops at the 300 px floor.
{
  const section = fakeSection(400);
  const grip = new Target();
  init(section, grip, fakeStorage({}), Object.assign(new Target(), { innerHeight: 1000 }));
  grip.fire('pointerdown', { clientY: 600, pointerId: 1 });
  grip.fire('pointermove', { clientY: 100 });
  assert.strictEqual(section.style.height, '300px');
}

console.log('ALL PASS');
