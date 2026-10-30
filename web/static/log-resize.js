// Server log height: a grip on the log's bottom edge sets the height, this
// browser remembers it, and a double-click on the grip goes back to the
// default. With nothing saved, the log keeps its normal size: it fills the
// space the page leaves it (see .log-section in server.html).
(function () {
    var KEY = 'llama-toolchest.serverLogHeight';
    // The CSS minimum, and room kept below the log for the page around it.
    var MIN = 300;
    var MARGIN = 120;

    function clampHeight(px, viewportPx) {
        var max = Math.max(MIN, viewportPx - MARGIN);
        return Math.round(Math.min(Math.max(px, MIN), max));
    }

    // Storage can be missing or refuse access (private windows, blocked
    // site data); the log then just starts at its default size.
    function load(storage) {
        try {
            var v = parseInt(storage.getItem(KEY), 10);
            return isNaN(v) ? 0 : v;
        } catch (e) {
            return 0;
        }
    }

    function save(storage, px) {
        try { storage.setItem(KEY, String(px)); } catch (e) {}
    }

    function forget(storage) {
        try { storage.removeItem(KEY); } catch (e) {}
    }

    function init(section, grip, storage, win) {
        win = win || window;
        // wanted is the height the user chose; what is shown may be less
        // while the window is too small for it.
        var wanted = 0;

        function show(px) {
            section.style.height = clampHeight(px, win.innerHeight) + 'px';
            section.classList.add('sized');
        }

        function reset() {
            wanted = 0;
            section.style.height = '';
            section.classList.remove('sized');
        }

        wanted = load(storage);
        if (wanted > 0) show(wanted);

        var startY = 0;
        var startHeight = 0;
        var dragging = false;

        grip.addEventListener('pointerdown', function (e) {
            dragging = true;
            startY = e.clientY;
            startHeight = section.offsetHeight;
            if (grip.setPointerCapture) grip.setPointerCapture(e.pointerId);
            e.preventDefault();
        });
        grip.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            wanted = clampHeight(startHeight + e.clientY - startY, win.innerHeight);
            show(wanted);
        });
        function end() {
            if (!dragging) return;
            dragging = false;
            if (wanted > 0) save(storage, wanted);
        }
        grip.addEventListener('pointerup', end);
        grip.addEventListener('pointercancel', end);

        grip.addEventListener('dblclick', function () {
            forget(storage);
            reset();
        });

        // A smaller window must not leave the log taller than it allows.
        win.addEventListener('resize', function () {
            if (wanted > 0) show(wanted);
        });
    }

    window.LogResize = { clampHeight: clampHeight, init: init };

    document.addEventListener('DOMContentLoaded', function () {
        var section = document.querySelector('.server-page .log-section');
        var grip = section && section.querySelector('.log-grip');
        if (!section || !grip) return;
        var storage = null;
        try { storage = window.localStorage; } catch (e) {}
        init(section, grip, storage || { getItem: function () { return null; }, setItem: function () {}, removeItem: function () {} });
    });
})();
