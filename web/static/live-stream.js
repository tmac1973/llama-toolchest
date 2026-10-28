// LiveStream: a server-sent event stream (EventSource) that holds its
// connection only while it is needed.
//
// Over plain HTTP a browser opens at most 6 connections to one server, and
// an open stream holds one of them for as long as it lasts. A few tabs on
// pages with live logs used all 6, and every other request from the UI
// waited. So a LiveStream:
//   - closes its connection while the tab is hidden and opens it again
//     when the tab is shown, and
//   - closes for good on the server's "done" event (the build finished, the
//     router exited), where a plain EventSource would connect again.
//
// The server sends recent lines again to each new connection, so a panel
// that adds lines must start empty each time the stream opens: pages do
// that in onopen, and htmx elements with data-sse-replay are cleared below.
//
// It has the parts of the EventSource interface the pages and htmx's SSE
// extension use: onopen, onmessage, onerror, addEventListener,
// removeEventListener, close and readyState.
(function () {
    var Native = window.EventSource;
    if (!Native) return;
    var CONNECTING = 0, OPEN = 1, CLOSED = 2;

    function LiveStream(url, options) {
        this.url = url;
        this.readyState = CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this._options = options;
        this._source = null;
        this._listeners = {};
        this._finished = false;
        var self = this;
        this._onVisibility = function () { self._sync(); };
        document.addEventListener('visibilitychange', this._onVisibility);
        this._sync();
    }
    LiveStream.CONNECTING = CONNECTING;
    LiveStream.OPEN = OPEN;
    LiveStream.CLOSED = CLOSED;

    // Events the native stream reports through its on* properties; any
    // other name is a named server event and needs its own listener.
    var builtIn = { open: true, message: true, error: true };

    // _sync connects or disconnects to match the tab's visibility.
    LiveStream.prototype._sync = function () {
        if (this._finished) return;
        if (document.hidden) {
            this._disconnect();
            this.readyState = CONNECTING;
        } else if (!this._source) {
            this._connect();
        }
    };

    LiveStream.prototype._connect = function () {
        var self = this;
        var src = new Native(this.url, this._options);
        this._source = src;
        src.onopen = function (e) {
            if (self._source !== src) return;
            self.readyState = OPEN;
            self._emit('open', e);
        };
        src.onmessage = function (e) {
            if (self._source === src) self._emit('message', e);
        };
        src.onerror = function (e) {
            if (self._source !== src) return;
            if (src.readyState === CLOSED) {
                // The server refused the stream (not found, or an error);
                // the native stream does not try again.
                self._finished = true;
                self._disconnect();
                self._forget();
                self.readyState = CLOSED;
            } else {
                self.readyState = CONNECTING; // the native stream tries again
            }
            self._emit('error', e);
        };
        this._listenNative(src, 'done');
        Object.keys(this._listeners).forEach(function (type) {
            if (!builtIn[type] && type !== 'done') self._listenNative(src, type);
        });
    };

    LiveStream.prototype._listenNative = function (src, type) {
        var self = this;
        src.addEventListener(type, function (e) {
            if (self._source !== src) return;
            self._emit(type, e);
            if (type === 'done') self.close();
        });
    };

    LiveStream.prototype._disconnect = function () {
        if (this._source) {
            this._source.close();
            this._source = null;
        }
    };

    LiveStream.prototype._forget = function () {
        document.removeEventListener('visibilitychange', this._onVisibility);
    };

    LiveStream.prototype._emit = function (type, e) {
        var handler = builtIn[type] ? this['on' + type] : null;
        if (typeof handler === 'function') handler.call(this, e);
        (this._listeners[type] || []).slice().forEach(function (fn) { fn(e); });
    };

    LiveStream.prototype.addEventListener = function (type, fn) {
        var list = this._listeners[type];
        if (!list) {
            list = this._listeners[type] = [];
            if (this._source && !builtIn[type] && type !== 'done') this._listenNative(this._source, type);
        }
        if (list.indexOf(fn) < 0) list.push(fn);
    };

    LiveStream.prototype.removeEventListener = function (type, fn) {
        var list = this._listeners[type];
        if (list) {
            var i = list.indexOf(fn);
            if (i >= 0) list.splice(i, 1);
        }
    };

    LiveStream.prototype.close = function () {
        this._finished = true;
        this._disconnect();
        this._forget();
        this.readyState = CLOSED;
    };

    window.LiveStream = LiveStream;

    // htmx's SSE extension makes its streams through this function.
    if (window.htmx) {
        htmx.createEventSource = function (url) {
            return new LiveStream(url, { withCredentials: true });
        };
    }

    // An htmx log panel marked data-sse-replay starts empty each time its
    // stream opens, because the server sends the recent lines again.
    document.addEventListener('htmx:sseOpen', function (e) {
        var elt = e.target;
        if (elt && elt.hasAttribute && elt.hasAttribute('data-sse-replay')) elt.textContent = '';
    });
})();
