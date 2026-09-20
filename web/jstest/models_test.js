// Drives the models page's own script against a stub, to check the one
// piece of it with a decision in it: reloading the Configure panel after
// Autoconfigure or Autotune has saved something into it.
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

let failures = 0;
function check(name, cond) {
  if (!cond) { failures++; console.log('FAIL: ' + name); } else { console.log('pass: ' + name); }
}

check('the page listens for a stale config panel', typeof handlers.modelConfigStale === 'function');
check('the page listens for a VRAM update', typeof handlers.vramUpdated === 'function');

const id = 'unsloth--Qwen3.5-4B-GGUF--Qwen3.5-4B-Q4_K_M';
const dom = id.replace(/[^A-Za-z0-9_-]/g, '_');

// An open panel is reloaded in place.
elements['model-config-' + dom] = { innerHTML: '<div>the settings from before the save</div>' };
handlers.modelConfigStale({ detail: { id: id, dom: dom } });
check('an open Configure panel is reloaded', ajaxCalls.length === 1);
if (ajaxCalls.length === 1) {
  const c = ajaxCalls[0];
  check('it is a GET of that model\'s config', c.method === 'GET' && c.url === '/api/models/' + id + '/config');
  check('it swaps into that panel', c.opts.target === elements['model-config-' + dom] && c.opts.swap === 'innerHTML');
}

// A closed panel has nothing to reload: opening it fetches the settings
// anyway, and a fetch here would open a panel the user did not open.
ajaxCalls.length = 0;
elements['model-config-' + dom].innerHTML = '';
handlers.modelConfigStale({ detail: { id: id, dom: dom } });
check('a closed Configure panel is left closed', ajaxCalls.length === 0);

// A card that is not on this page, and an event with nothing in it.
ajaxCalls.length = 0;
handlers.modelConfigStale({ detail: { id: 'gone', dom: 'gone' } });
handlers.modelConfigStale({ detail: {} });
handlers.modelConfigStale({});
check('an event for a card that is not here does nothing', ajaxCalls.length === 0);

// The VRAM cell still updates from the same script.
elements['vram-' + id] = { textContent: '' };
handlers.vramUpdated({ detail: { id: id, vram: '12.3 GiB' } });
check('the VRAM cell is updated', elements['vram-' + id].textContent === '12.3 GiB');

if (failures) { console.log(failures + ' FAILED'); process.exit(1); }
console.log('ALL PASS');
