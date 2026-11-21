// Drives genSetUseFile of web/static/gen-set-browse.js with small stubs:
// Use writes the link into the slot's field of the form next to the
// dialog, fires a bubbling change event, and closes the dialog. A
// button for a slot the form does not have changes nothing. Run by
// internal/api/js_gen_set_browse_test.go.
const assert = require('assert');
const path = require('path');

global.Event = class { constructor(type, opts) { this.type = type; this.bubbles = !!(opts && opts.bubbles); } };
const { genSetUseFile } = require(path.join(__dirname, 'gen-set-browse.js'));

function setup() {
  const events = [];
  const fields = {
    'input[name="slot_vae"]': { value: '', dispatchEvent: e => events.push(e) },
  };
  const form = { querySelector: sel => fields[sel] || null };
  const parent = { querySelector: sel => (sel === 'form' ? form : null) };
  const dialog = { parentElement: parent, closed: false, close() { this.closed = true; } };
  const button = slot => ({
    dataset: { slot, link: 'https://huggingface.co/org/repo/blob/main/ae.safetensors' },
    closest: sel => (sel === 'dialog' ? dialog : null),
  });
  return { events, fields, dialog, button };
}

{
  const t = setup();
  genSetUseFile(t.button('vae'));
  assert.strictEqual(t.fields['input[name="slot_vae"]'].value, 'https://huggingface.co/org/repo/blob/main/ae.safetensors');
  assert.strictEqual(t.events.length, 1);
  assert.strictEqual(t.events[0].type, 'change');
  assert.ok(t.events[0].bubbles, 'the change event must bubble to the form');
  assert.ok(t.dialog.closed, 'Use closes the dialog');
}
{
  const t = setup();
  genSetUseFile(t.button('llm'));
  assert.strictEqual(t.fields['input[name="slot_vae"]'].value, '');
  assert.strictEqual(t.events.length, 0);
  assert.ok(!t.dialog.closed, 'a slot the form lacks leaves the dialog open');
}
console.log('ALL PASS');
