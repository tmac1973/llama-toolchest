// Keeps a panel in view when a click inside it replaces it with something
// shorter.
//
// A button at the bottom of a tall panel (Save and apply under the
// Autoconfigure review, for one) swaps the panel for a short message. The
// page keeps its scroll position, so the message ends up above the screen
// and the cards below the panel move up into view: it looks as if the
// page scrolled down. After such a swap, if the panel is now wholly above
// the screen, scroll it back to the top of the screen.
//
// Only requests started by the user (a trusted click, change or submit)
// and swapped into an element that holds what they acted on. Polling and
// htmx.ajax calls have no such event, so they never move the page. A long
// form that saves as you type is left alone: it still covers the screen,
// so it is not wholly above it.
(function () {
    var pending = null;

    document.addEventListener('htmx:beforeSwap', function (e) {
        var d = e.detail || {};
        var cfg = d.requestConfig || {};
        var ev = cfg.triggeringEvent;
        var elt = cfg.elt || d.elt;
        pending = null;
        if (!d.target || !elt || !ev || !ev.isTrusted) return;
        if (d.target !== elt && !d.target.contains(elt)) return;
        // An element that sets its own show: or scroll: (the Configure
        // form's show:none) has chosen where the page goes.
        var swapSpec = elt.closest && elt.closest('[hx-swap]');
        if (swapSpec && /\b(show|scroll):/.test(swapSpec.getAttribute('hx-swap'))) return;
        pending = {target: d.target, id: d.target.id};
    });

    document.addEventListener('htmx:afterSwap', function () {
        var p = pending;
        pending = null;
        if (!p) return;
        // An outerHTML swap replaces the target: find the new one by id.
        var el = p.target.isConnected ? p.target : (p.id && document.getElementById(p.id));
        if (!el || !el.getBoundingClientRect) return;
        if (el.getBoundingClientRect().bottom > 0) return;
        el.scrollIntoView({block: 'start'});
    });
})();
