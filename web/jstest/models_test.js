// Drives the models page's own script against a stub, to check the one
// piece of it with a decision in it: reloading the Configure panel after
// Autoconfigure or Autotune has saved something into it.
const assert = require('assert');
const fs = require('fs');

const handlers = {};
const elements = {};
const ajaxCalls = [];

global.document = {
  body: { addEventListener(name, fn) { handlers[name] = fn; } },
  getElementById(id) { return elements[id] || null; },
};
global.htmx = {
  ajax(method, url, opts) { ajaxCalls.push({ method, url, opts }); },
};

eval(fs.readFileSync('models.js', 'utf8'));

assert.strictEqual(typeof handlers.modelConfigStale, 'function', 'the page listens for a stale config panel');
assert.strictEqual(typeof handlers.vramUpdated, 'function', 'the page listens for a VRAM update');

const id = 'unsloth--Qwen3.5-4B-GGUF--Qwen3.5-4B-Q4_K_M';
const dom = id.replace(/[^A-Za-z0-9_-]/g, '_');

// An open panel is reloaded in place.
elements['model-config-' + dom] = { innerHTML: '<div>the settings from before the save</div>' };
handlers.modelConfigStale({ detail: { id: id, dom: dom } });
assert.strictEqual(ajaxCalls.length, 1, 'an open Configure panel is reloaded');
const c = ajaxCalls[0];
assert.strictEqual(c.method, 'GET', 'it is a GET of that model\'s config');
assert.strictEqual(c.url, '/api/models/' + id + '/config', 'it is a GET of that model\'s config');
assert.strictEqual(c.opts.target, elements['model-config-' + dom], 'it swaps into that panel');
assert.strictEqual(c.opts.swap, 'innerHTML', 'it swaps into that panel');

// A closed panel has nothing to reload: opening it fetches the settings
// anyway, and a fetch here would open a panel the user did not open.
ajaxCalls.length = 0;
elements['model-config-' + dom].innerHTML = '';
handlers.modelConfigStale({ detail: { id: id, dom: dom } });
assert.strictEqual(ajaxCalls.length, 0, 'a closed Configure panel is left closed');

// A card that is not on this page, and an event with nothing in it.
ajaxCalls.length = 0;
handlers.modelConfigStale({ detail: { id: 'gone', dom: 'gone' } });
handlers.modelConfigStale({ detail: {} });
handlers.modelConfigStale({});
assert.strictEqual(ajaxCalls.length, 0, 'an event for a card that is not here does nothing');

// The VRAM cell still updates from the same script.
elements['vram-' + id] = { textContent: '' };
handlers.vramUpdated({ detail: { id: id, vram: '12.3 GiB' } });
assert.strictEqual(elements['vram-' + id].textContent, '12.3 GiB', 'the VRAM cell is updated');

console.log('ALL PASS');
