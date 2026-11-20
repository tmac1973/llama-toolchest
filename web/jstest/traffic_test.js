// Drives the store of web/static/traffic.js: events of parallel requests
// stay with their own request, the filters pick by model and engine, the
// oldest requests go past the limit, and a streamed answer is put back
// together. Run by internal/api/js_traffic_test.go.
const assert = require('assert');
const path = require('path');
const { TrafficStore, targetLabel, pretty, assemble } = require(path.join(__dirname, 'traffic.js'));

const sse = obj => 'data: ' + JSON.stringify(obj) + '\n\n';
const delta = d => sse({ choices: [{ delta: d }] });

const store = new TrafficStore(3);
// Two streams that run at the same time, their events mixed.
store.apply({ type: 'start', id: 1, method: 'POST', path: '/v1/chat/completions', time: '2026-11-19T10:00:00Z' });
store.apply({ type: 'start', id: 2, method: 'POST', path: '/v1/chat/completions', time: '2026-11-19T10:00:01Z' });
store.apply({ type: 'request', id: 1, target: 'router', model: 'qwen', body: '{"model":"qwen"}' });
store.apply({ type: 'request', id: 2, target: 'fork:avalon', model: 'avalon', body: '{"model":"avalon"}' });
store.apply({ type: 'response', id: 2, status: 200, content_type: 'text/event-stream' });
store.apply({ type: 'response', id: 1, status: 200, content_type: 'text/event-stream' });
store.apply({ type: 'chunk', id: 1, data: delta({ reasoning_content: 'think ' }) });
store.apply({ type: 'chunk', id: 2, data: delta({ content: 'Hel' }) });
store.apply({ type: 'chunk', id: 1, data: delta({ content: 'one' }) });
// A chunk can end in the middle of an SSE line.
const tail = delta({ content: 'lo' }) + 'data: [DONE]\n\n';
store.apply({ type: 'chunk', id: 2, data: tail.slice(0, 10) });
store.apply({ type: 'chunk', id: 2, data: tail.slice(10) });
store.apply({ type: 'end', id: 2, status: 200, ms: 1500 });

const a1 = assemble(store.byId[1]);
const a2 = assemble(store.byId[2]);
assert.strictEqual(a1.text, 'one', 'request 1 got another stream\'s text');
assert.strictEqual(a1.reasoning, 'think ');
assert.strictEqual(a2.text, 'Hello', 'request 2 was not put back together');
assert.strictEqual(a1.finish, '', 'request 1 has not stopped yet');
assert.strictEqual(store.byId[1].done, false);
assert.strictEqual(store.byId[2].done, true);
assert.strictEqual(store.byId[2].ms, 1500);

// Filters, newest first.
assert.deepStrictEqual(store.list('', '').map(x => x.id), [2, 1]);
assert.deepStrictEqual(store.list('qwen', '').map(x => x.id), [1]);
assert.deepStrictEqual(store.list('', 'fork:avalon').map(x => x.id), [2]);
assert.deepStrictEqual(store.list('qwen', 'fork:avalon').map(x => x.id), []);
assert.deepStrictEqual(store.models(), ['avalon', 'qwen']);
assert.deepStrictEqual(store.targets(), ['fork:avalon', 'router']);

// Past the limit the oldest goes.
store.apply({ type: 'start', id: 3, path: '/v1/embeddings' });
store.apply({ type: 'start', id: 4, path: '/v1/embeddings' });
assert.deepStrictEqual(store.order.map(x => x.id), [2, 3, 4]);
assert.strictEqual(store.byId[1], undefined);
// Events of a request that is gone, or that started before the window
// opened, are left out.
assert.strictEqual(store.apply({ type: 'chunk', id: 1, data: 'late' }), null);
assert.strictEqual(store.apply({ type: 'end', id: 9, status: 503 }), null);
assert.deepStrictEqual(store.order.map(x => x.id), [2, 3, 4]);

// Tool calls come in pieces by index; a whole JSON answer works too.
const t = new TrafficStore();
[1, 2, 3].forEach(id => t.apply({ type: 'start', id }));
t.apply({ type: 'response', id: 1, status: 200, content_type: 'text/event-stream' });
t.apply({ type: 'chunk', id: 1, data:
    delta({ tool_calls: [{ index: 0, function: { name: 'get_weather', arguments: '{"ci' } }] }) +
    delta({ tool_calls: [{ index: 0, function: { arguments: 'ty":"Oslo"}' } }] }) });
assert.deepStrictEqual(assemble(t.byId[1]).tools, [{ name: 'get_weather', args: '{"city":"Oslo"}' }]);
t.apply({ type: 'response', id: 2, status: 200, content_type: 'application/json' });
t.apply({ type: 'chunk', id: 2, data: JSON.stringify({ choices: [{ message: { content: 'whole' }, finish_reason: 'stop' }] }) });
assert.strictEqual(assemble(t.byId[2]).text, 'whole');
assert.strictEqual(assemble(t.byId[2]).finish, 'stop');
// All tokens spent on reasoning: no answer, but the reason shows.
t.apply({ type: 'start', id: 4 });
t.apply({ type: 'response', id: 4, status: 200, content_type: 'text/event-stream' });
t.apply({ type: 'chunk', id: 4, data: delta({ reasoning_content: 'hmm' }) +
    sse({ choices: [{ delta: {}, finish_reason: 'length' }] }) });
assert.deepStrictEqual([assemble(t.byId[4]).text, assemble(t.byId[4]).finish], ['', 'length']);
// Embeddings have nothing to put together.
t.apply({ type: 'response', id: 3, status: 200, content_type: 'application/json' });
t.apply({ type: 'chunk', id: 3, data: '{"data":[{"embedding":[0.1]}]}' });
assert.strictEqual(assemble(t.byId[3]), null);

assert.strictEqual(targetLabel('router'), 'Router');
assert.strictEqual(targetLabel('fork:avalon'), 'Fork avalon');
assert.strictEqual(pretty('{"a":1}'), '{\n  "a": 1\n}');
assert.strictEqual(pretty('{"a":1, cut'), '{"a":1, cut');

console.log('ALL PASS');
