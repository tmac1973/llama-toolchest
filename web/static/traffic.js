// Live traffic viewer (Service page): the requests that go through the
// proxy, and their responses as they arrive.
//
// The server records only while this viewer's stream is open, and sends
// each event with the ID of its exchange (one request and its response).
// TrafficStore puts the events back together per exchange, so requests
// that run at the same time stay apart; the dialog lists the exchanges,
// filtered by model and engine, and shows one at a time.
(function () {
    // TrafficStore keeps the newest max exchanges.
    function TrafficStore(max) {
        this.max = max || 50;
        this.byId = {};
        this.order = []; // oldest first
    }

    TrafficStore.prototype._get = function (id) {
        var x = this.byId[id];
        if (x) return x;
        x = {
            id: id, time: '', method: '', path: '', target: '', model: '',
            body: '', note: '', status: 0, contentType: '', response: '',
            cut: false, done: false, ms: 0
        };
        this.byId[id] = x;
        this.order.push(x);
        while (this.order.length > this.max) {
            delete this.byId[this.order.shift().id];
        }
        return x;
    };

    // apply adds one server event and returns the exchange it changed.
    // Events of a request this window did not see start (another window
    // had the recording on, or the request was dropped) are left out.
    TrafficStore.prototype.apply = function (ev) {
        if (!ev || !ev.id) return null;
        if (ev.type !== 'start' && !this.byId[ev.id]) return null;
        var x = this._get(ev.id);
        switch (ev.type) {
        case 'start':
            x.time = ev.time || '';
            x.method = ev.method || '';
            x.path = ev.path || '';
            break;
        case 'request':
            x.target = ev.target || '';
            x.model = ev.model || '';
            x.body = ev.body || '';
            x.note = ev.note || '';
            break;
        case 'response':
            x.status = ev.status || 0;
            x.contentType = ev.content_type || '';
            break;
        case 'chunk':
            x.response += ev.data || '';
            break;
        case 'cut':
            x.cut = true;
            break;
        case 'end':
            x.done = true;
            x.status = ev.status || x.status;
            x.ms = ev.ms || 0;
            break;
        }
        return x;
    };

    TrafficStore.prototype.clear = function () {
        this.byId = {};
        this.order = [];
    };

    function uniqueSorted(list, key) {
        var seen = {};
        list.forEach(function (x) { if (x[key]) seen[x[key]] = true; });
        return Object.keys(seen).sort();
    }

    TrafficStore.prototype.models = function () { return uniqueSorted(this.order, 'model'); };
    TrafficStore.prototype.targets = function () { return uniqueSorted(this.order, 'target'); };

    // list is the exchanges that match the filters ('' = all), newest first.
    TrafficStore.prototype.list = function (model, target) {
        return this.order.filter(function (x) {
            return (!model || x.model === model) && (!target || x.target === target);
        }).reverse();
    };

    // targetLabel names an engine for people: the router, or a fork
    // instance by its name.
    function targetLabel(t) {
        if (t === 'router') return 'Router';
        if (t && t.indexOf('fork:') === 0) return 'Fork ' + t.slice(5);
        return t || '—';
    }

    // pretty indents JSON; anything else comes back as it is.
    function pretty(text) {
        try {
            return JSON.stringify(JSON.parse(text), null, 2);
        } catch (e) {
            return text;
        }
    }

    // assemble reads a response for people: the answer text, the
    // reasoning, the tool calls and why the engine stopped, from a streamed (SSE) or a whole JSON
    // response. null when there is nothing to show that way.
    function assemble(x) {
        var out = { text: '', reasoning: '', tools: [], finish: '' };
        var tools = {};
        function addChoice(c) {
            if (!c) return;
            if (typeof c.finish_reason === 'string') out.finish = c.finish_reason;
            var d = c.delta || c.message || {};
            if (typeof c.text === 'string') out.text += c.text;
            if (typeof d.content === 'string') out.text += d.content;
            var r = d.reasoning_content || d.reasoning;
            if (typeof r === 'string') out.reasoning += r;
            (d.tool_calls || []).forEach(function (tc, i) {
                var k = tc.index !== undefined ? tc.index : i;
                var t = tools[k] || (tools[k] = { name: '', args: '' });
                if (tc.function) {
                    if (tc.function.name) t.name += tc.function.name;
                    if (tc.function.arguments) t.args += tc.function.arguments;
                }
            });
        }
        if (x.contentType.indexOf('text/event-stream') >= 0) {
            x.response.split('\n').forEach(function (line) {
                if (line.indexOf('data:') !== 0) return;
                var data = line.slice(5).trim();
                if (!data || data === '[DONE]') return;
                try {
                    var j = JSON.parse(data);
                    (j.choices || []).forEach(addChoice);
                } catch (e) { /* a line cut in two by the size limit */ }
            });
        } else {
            try {
                var j = JSON.parse(x.response);
                (j.choices || []).forEach(addChoice);
            } catch (e) {
                return null;
            }
        }
        Object.keys(tools).sort().forEach(function (k) { out.tools.push(tools[k]); });
        if (!out.text && !out.reasoning && !out.tools.length && !out.finish) return null;
        return out;
    }

    var api = { TrafficStore: TrafficStore, targetLabel: targetLabel, pretty: pretty, assemble: assemble };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
        return;
    }
    window.Traffic = api;

    // ---- The dialog ----

    function el(tag, text, cls) {
        var e = document.createElement(tag);
        if (text !== undefined && text !== null) e.textContent = text;
        if (cls) e.className = cls;
        return e;
    }

    function shortTime(iso) {
        var d = new Date(iso);
        if (isNaN(d)) return '';
        return d.toLocaleTimeString();
    }

    function setup() {
        var dialog = document.getElementById('traffic-dialog');
        if (!dialog) return;
        var store = new TrafficStore(50);
        var listEl = document.getElementById('traffic-list');
        var detailEl = document.getElementById('traffic-detail');
        var modelSel = document.getElementById('traffic-model');
        var targetSel = document.getElementById('traffic-target');
        var pauseBox = document.getElementById('traffic-pause');
        var stateEl = document.getElementById('traffic-state');
        var source = null;
        var selected = null;
        var rendered = {}; // exchange id → rendered prompt or error
        var pending = false;
        var frozen = null; // while paused: the exchanges the list shows

        // fillSelect rebuilds a filter's options only when they change,
        // so a list that is open stays open.
        function fillSelect(sel, values, label) {
            var key = values.join('\n');
            if (sel.getAttribute('data-values') === key) return;
            sel.setAttribute('data-values', key);
            var cur = sel.value;
            sel.textContent = '';
            var all = el('option', label);
            all.value = '';
            sel.appendChild(all);
            values.forEach(function (v) {
                var o = el('option', sel === targetSel ? targetLabel(v) : v);
                o.value = v;
                sel.appendChild(o);
            });
            sel.value = values.indexOf(cur) >= 0 ? cur : '';
        }

        function statusText(x) {
            if (!x.done) return x.status ? 'streaming…' : 'waiting…';
            return x.status + ' · ' + (x.ms / 1000).toFixed(1) + ' s';
        }

        function renderList() {
            fillSelect(modelSel, store.models(), 'All models');
            fillSelect(targetSel, store.targets(), 'All engines');
            var rows = store.list(modelSel.value, targetSel.value);
            if (frozen) rows = rows.filter(function (x) { return frozen[x.id]; });
            listEl.textContent = '';
            if (!rows.length) {
                listEl.appendChild(el('li', 'No requests yet. Send one to the API and it shows here.', 'traffic-empty'));
                return;
            }
            rows.forEach(function (x) {
                var li = el('li');
                var b = el('button');
                b.type = 'button';
                b.className = 'traffic-row' + (selected === x.id ? ' selected' : '') +
                    (x.done && x.status >= 400 ? ' failed' : '');
                b.title = (x.model || '(no model)') + ' · ' + targetLabel(x.target) + ' · ' + x.path;
                b.appendChild(el('span', shortTime(x.time), 'traffic-time'));
                b.appendChild(el('span', x.model || '(no model)', 'traffic-model'));
                b.appendChild(el('span', statusText(x), 'traffic-status'));
                b.appendChild(el('span', targetLabel(x.target) + ' · ' + x.path, 'traffic-path'));
                b.addEventListener('click', function () {
                    selected = x.id;
                    render();
                });
                li.appendChild(b);
                listEl.appendChild(li);
            });
        }

        function section(title, body, extra) {
            var s = el('section');
            var h = el('h6', title);
            if (extra) h.appendChild(extra);
            s.appendChild(h);
            if (body !== null) s.appendChild(el('pre', body));
            return s;
        }

        function copyButton(text) {
            var b = el('button', 'Copy', 'outline secondary traffic-small');
            b.type = 'button';
            b.addEventListener('click', function () {
                if (navigator.clipboard) navigator.clipboard.writeText(text());
            });
            return b;
        }

        function renderDetail() {
            var x = selected && store.byId[selected];
            detailEl.textContent = '';
            if (!x) {
                detailEl.appendChild(el('p', 'Select a request to see it.', 'traffic-empty'));
                return;
            }
            var a = assemble(x);
            detailEl.appendChild(el('p',
                (x.model || '(no model)') + ' · ' + targetLabel(x.target) + ' · ' + x.method + ' ' + x.path +
                ' · ' + statusText(x) + (a && a.finish ? ' · stopped: ' + a.finish : ''), 'traffic-meta'));

            var req = x.body ? pretty(x.body) : '(not recorded)';
            detailEl.appendChild(section('Request (as sent to the engine)', req, copyButton(function () { return req; })));
            if (x.note) detailEl.appendChild(el('p', 'Note: ' + x.note + '.', 'traffic-note'));

            if (a) {
                if (a.reasoning) detailEl.appendChild(section('Reasoning', a.reasoning));
                detailEl.appendChild(section('Answer', a.text || '(none)'));
                a.tools.forEach(function (t) {
                    detailEl.appendChild(section('Tool call: ' + (t.name || '?'), pretty(t.args)));
                });
            }
            var raw = x.contentType.indexOf('text/event-stream') >= 0 ? x.response : pretty(x.response);
            detailEl.appendChild(section('Response' + (x.contentType ? ' (' + x.contentType + ')' : ''),
                raw || '(nothing yet)', copyButton(function () { return x.response; })));
            if (x.cut) detailEl.appendChild(el('p', 'Note: the response was cut at 1 MiB.', 'traffic-note'));

            if (x.body && x.path.slice(-17) === '/chat/completions') {
                var btn = el('button', 'Show rendered prompt', 'outline traffic-small');
                btn.type = 'button';
                btn.addEventListener('click', function () { renderPrompt(x, btn); });
                var s = section('Rendered prompt', null, btn);
                var r = rendered[x.id];
                if (r) s.appendChild(el('pre', r));
                s.appendChild(el('p', 'The engine renders the prompt again when you press the button, from the ' +
                    'request above. If the chat template changed since, the result can differ from what the ' +
                    'engine used.' + (x.note ? ' Shortened or cut data in the request can change it too.' : ''),
                    'traffic-note'));
                detailEl.appendChild(s);
            }
        }

        function renderPrompt(x, btn) {
            btn.disabled = true;
            btn.setAttribute('aria-busy', 'true');
            fetch('/api/service/traffic/render', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target: x.target, body: JSON.parse(x.body) })
            }).then(function (resp) {
                return resp.json();
            }).then(function (j) {
                rendered[x.id] = j.prompt !== undefined ? j.prompt : 'Could not render: ' + (j.error || 'no answer');
            }).catch(function (e) {
                rendered[x.id] = 'Could not render: ' + e;
            }).then(function () {
                render();
            });
        }

        // render redraws a few times a second at most: a streamed answer
        // sends many small events, and the response can be large.
        function render() {
            if (pending) return;
            pending = true;
            setTimeout(function () {
                pending = false;
                renderList();
                var keep = detailEl.scrollTop;
                renderDetail();
                detailEl.scrollTop = keep;
            }, 150);
        }

        function open() {
            if (source) return;
            source = new LiveStream('/api/service/traffic');
            source.onopen = function () { stateEl.textContent = 'Recording while this window is open.'; };
            source.onerror = function () { stateEl.textContent = 'Connection lost; trying again…'; };
            source.onmessage = function (e) {
                var ev;
                try { ev = JSON.parse(e.data); } catch (err) { return; }
                store.apply(ev);
                render();
            };
            render();
        }

        function close() {
            if (source) source.close();
            source = null;
            stateEl.textContent = 'Not recording.';
        }

        document.getElementById('traffic-open').addEventListener('click', function () {
            dialog.showModal();
            open();
        });
        dialog.addEventListener('close', close);
        modelSel.addEventListener('change', function () { render(); });
        targetSel.addEventListener('change', function () { render(); });
        // Pause keeps the list to the requests it shows now, so it does
        // not move while you read. Their responses still come in.
        pauseBox.addEventListener('change', function () {
            frozen = null;
            if (pauseBox.checked) {
                frozen = {};
                store.order.forEach(function (x) { frozen[x.id] = true; });
            }
            render();
        });
        document.getElementById('traffic-clear').addEventListener('click', function () {
            store.clear();
            rendered = {};
            if (frozen) frozen = {};
            selected = null;
            render();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
})();
