// Shows failed requests to the user.
//
// htmx does not swap a 4xx or 5xx response into the page, so without
// this a refused or failed action looked like a button that did nothing:
// a failed HuggingFace search, a download refused for lack of disk space,
// a router action refused while a benchmark job runs. Every such response
// now shows its message in one notice at the top of the page.
//
// The notice stays until it is closed, or until the action that failed
// is tried again and works. Background polling (hx-trigger "every …") is
// left out, so a page polling a restarting server does not flash errors;
// so is any element marked data-own-errors, which shows errors itself.
//
// apiCall is the same for fetch(): it resolves with the response when it
// is OK, and otherwise shows the notice and rejects.
(function () {
    var MAX_MESSAGE = 400;
    var notice = null;
    var source = null;

    // messageFrom turns an error response into one readable line: the
    // "error" field of a JSON body, otherwise the text the server sent.
    function messageFrom(status, text, contentType) {
        var msg = (text || '').trim();
        if (msg && /json/i.test(contentType || '')) {
            try {
                var body = JSON.parse(msg);
                var e = body && body.error;
                if (typeof e === 'string') msg = e;
                else if (e && typeof e.message === 'string') msg = e.message;
            } catch (_) { /* not JSON after all: show the text */ }
        }
        if (/^\s*</.test(msg)) msg = ''; // an HTML error page says nothing useful here
        if (!msg) msg = 'The server answered with status ' + status + '.';
        if (msg.length > MAX_MESSAGE) msg = msg.slice(0, MAX_MESSAGE) + '…';
        return msg;
    }

    function ensureNotice() {
        if (notice) return notice;
        notice = document.createElement('div');
        notice.id = 'error-notice';
        notice.setAttribute('role', 'alert');
        notice.hidden = true;
        var text = document.createElement('span');
        text.className = 'error-notice-text';
        var close = document.createElement('button');
        close.type = 'button';
        close.className = 'error-notice-close';
        close.setAttribute('aria-label', 'Close this message');
        close.title = 'Close this message';
        close.textContent = '×';
        close.addEventListener('click', hideError);
        notice.appendChild(text);
        notice.appendChild(close);
        var main = document.querySelector('main') || document.body;
        main.insertBefore(notice, main.firstChild);
        return notice;
    }

    // showError puts message in the notice. from is the element whose
    // request failed, so a later success of the same element clears it.
    function showError(message, from) {
        var n = ensureNotice();
        n.querySelector('.error-notice-text').textContent = message;
        n.hidden = false;
        source = from || null;
    }

    function hideError() {
        if (notice) notice.hidden = true;
        source = null;
    }

    function label(elt) {
        var withLabel = elt && elt.closest && elt.closest('[data-error-label]');
        return withLabel ? withLabel.getAttribute('data-error-label') : 'Request failed';
    }

    function ignored(elt) {
        if (!elt || !elt.closest) return false;
        if (elt.closest('[data-own-errors]')) return true;
        var trigger = elt.getAttribute && elt.getAttribute('hx-trigger');
        return !!trigger && /\bevery\b/.test(trigger);
    }

    document.addEventListener('htmx:responseError', function (evt) {
        var elt = evt.detail.elt;
        if (ignored(elt)) return;
        var xhr = evt.detail.xhr;
        var msg = messageFrom(xhr.status, xhr.responseText, xhr.getResponseHeader && xhr.getResponseHeader('Content-Type'));
        showError(label(elt) + ': ' + msg, elt);
    });

    document.addEventListener('htmx:sendError', function (evt) {
        var elt = evt.detail.elt;
        if (ignored(elt)) return;
        showError(label(elt) + ': the server could not be reached.', elt);
    });

    document.addEventListener('htmx:afterRequest', function (evt) {
        if (source && evt.detail.successful && evt.detail.elt === source) hideError();
    });

    // apiCall is fetch() that reports failure: it resolves with the
    // response when it is OK, and otherwise shows the server's message
    // (prefixed with what, e.g. "Cancel failed") and rejects, so the
    // caller's .then() only runs on success.
    function apiCall(url, options, what) {
        var prefix = (what || 'Request failed') + ': ';
        return fetch(url, options).then(function (r) {
            if (r.ok) return r;
            return r.text().then(function (t) {
                var msg = messageFrom(r.status, t, r.headers.get('Content-Type'));
                showError(prefix + msg);
                throw new Error(msg);
            });
        }, function (e) {
            showError(prefix + 'the server could not be reached.');
            throw e;
        });
    }

    window.showError = showError;
    window.hideError = hideError;
    window.apiCall = apiCall;
    window.errorMessageFrom = messageFrom;
})();
