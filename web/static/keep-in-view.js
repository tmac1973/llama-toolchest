// Keeps what the user is looking at in place when htmx swaps it.
//
// Two problems, both seen as "the page scrolls down by itself":
//
// 1. Scroll anchoring. When an element on screen is swapped, its old
//    nodes go and Chrome picks a new anchor, which can move the page by a
//    few hundred pixels: the reply to Measure on the Server page did, and
//    a row of the instance table that grew on a 5-second refresh went up
//    instead of growing down. So for a swap on screen, Chrome's anchoring
//    is off for two frames and this script anchors instead: it notes one
//    element (the one clicked, or the one at the top of what shows), finds
//    the same element in the new content by its place in the tree, and
//    scrolls by however far it moved. A row above it that gets shorter or
//    taller then does not move it. A swap above the screen keeps Chrome's
//    anchoring, so its change in height does not move what is on screen.
//
// 2. A panel that gets shorter. A button at the bottom of a tall panel
//    (Save and apply under the Autoconfigure review, for one) swaps the
//    panel for a short message. The message ends up above the screen and
//    the cards below move up into view. After such a swap, if the panel is
//    now wholly above the screen, scroll it back to the top of the screen.
//    Only for requests started by the user (a trusted click, change or
//    submit) and swapped into an element that holds what they acted on:
//    polling and htmx.ajax calls never do this. A long form that saves as
//    you type is left alone: it still covers the screen.
(function () {
    var pending = null;
    var anchor = null;

    // scrollerOf is the element that scrolls el: the nearest ancestor that
    // scrolls, or the page itself.
    function scrollerOf(el) {
        for (var n = el.parentElement; n && n !== document.body; n = n.parentElement) {
            var o = getComputedStyle(n).overflowY;
            if ((o === 'auto' || o === 'scroll') && n.scrollHeight > n.clientHeight) return n;
        }
        return document.scrollingElement || document.documentElement;
    }

    function onScreen(el) {
        var r = el.getBoundingClientRect();
        return r.bottom > 0 && r.top < window.innerHeight;
    }

    function pauseAnchoring(sc) {
        sc.style.overflowAnchor = 'none';
        // Two frames: the swap's layout happens in the next one.
        requestAnimationFrame(function () {
            requestAnimationFrame(function () { sc.style.overflowAnchor = ''; });
        });
    }

    // steps is the way from root down to el: at each level the child's
    // index and tag, and its top and height on screen.
    function steps(root, el) {
        var out = [];
        for (var n = el; n && n !== root; n = n.parentElement) {
            var p = n.parentElement;
            if (!p) return null;
            var r = n.getBoundingClientRect();
            out.unshift({i: Array.prototype.indexOf.call(p.children, n), tag: n.tagName,
                         top: r.top, h: r.bottom - r.top});
        }
        return n === root ? out : null;
    }

    // moved follows steps in the new content under root and says how far
    // the anchor moved on screen. The anchor is the outermost step that
    // fits on the screen: a table row, not the whole table (it does not
    // move when a row above it changes) and not a button in the row (a
    // row in another state has other buttons). 0 when the new content has
    // another shape.
    function moved(root, path) {
        var want = path.length - 1;
        for (var k = 0; k < path.length; k++) {
            if (path[k].h <= window.innerHeight) { want = k; break; }
        }
        var n = root;
        for (var j = 0; j <= want; j++) {
            var c = n.children && n.children[path[j].i];
            if (!c || c.tagName !== path[j].tag) return 0;
            n = c;
        }
        return n.getBoundingClientRect().top - path[want].top;
    }

    function pickAnchor(target, sc, elt) {
        if (elt && elt !== target && target.contains(elt)) return elt;
        if (!document.elementFromPoint) return null;
        var r = target.getBoundingClientRect();
        var top = sc === document.scrollingElement || sc === document.documentElement
            ? 0 : Math.max(0, sc.getBoundingClientRect().top);
        var y = Math.max(r.top, top) + 2;
        var x = Math.min(Math.max(r.left + r.width / 2, 0), window.innerWidth - 1);
        var hit = document.elementFromPoint(x, y);
        return hit && hit !== target && target.contains(hit) ? hit : null;
    }

    document.addEventListener('htmx:beforeSwap', function (e) {
        var d = e.detail || {};
        var cfg = d.requestConfig || {};
        var ev = cfg.triggeringEvent;
        var elt = cfg.elt || d.elt;
        pending = null;
        anchor = null;
        if (!d.target || !d.target.getBoundingClientRect) return;

        if (d.target.isConnected !== false && onScreen(d.target)) {
            var sc = scrollerOf(d.target);
            if (sc && sc.style) {
                pauseAnchoring(sc);
                var a = pickAnchor(d.target, sc, elt);
                var path = a && steps(d.target, a);
                if (path && path.length) anchor = {target: d.target, id: d.target.id, sc: sc, path: path};
            }
        }

        if (!elt || !ev || !ev.isTrusted) return;
        if (d.target !== elt && !d.target.contains(elt)) return;
        // An element that sets its own show: or scroll: (the Configure
        // form's show:none) has chosen where the page goes.
        var swapSpec = elt.closest && elt.closest('[hx-swap]');
        if (swapSpec && /\b(show|scroll):/.test(swapSpec.getAttribute('hx-swap'))) return;
        pending = {target: d.target, id: d.target.id};
    });

    // An outerHTML swap replaces the target: find the new one by id.
    function current(p) {
        return p.target.isConnected ? p.target : (p.id && document.getElementById(p.id));
    }

    document.addEventListener('htmx:afterSwap', function () {
        var a = anchor;
        anchor = null;
        if (a) {
            var root = current(a);
            var delta = root ? moved(root, a.path) : 0;
            if (delta) a.sc.scrollTop += delta;
        }

        var p = pending;
        pending = null;
        if (!p) return;
        var el = current(p);
        if (!el || !el.getBoundingClientRect) return;
        if (el.getBoundingClientRect().bottom > 0) return;
        el.scrollIntoView({block: 'start'});
    });
})();
