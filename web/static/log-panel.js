// Log panel: live-tail auto-scroll + copy to clipboard + clear
document.addEventListener('DOMContentLoaded', function () {
    initLogPanels();
});

// Re-init after htmx swaps (for dynamically inserted build logs)
document.addEventListener('htmx:afterSettle', function () {
    initLogPanels();
});

function initLogPanels() {
    document.querySelectorAll('.log-panel').forEach(function (panel) {
        if (panel._logPanelInit) return;
        panel._logPanelInit = true;

        var pre = panel.querySelector('pre');
        var tailToggle = panel.querySelector('.log-tail-toggle');
        var copyBtn = panel.querySelector('.log-copy-btn');
        var clearBtn = panel.querySelector('.log-clear-btn');
        if (!pre) return;

        // Live tail: auto-scroll on new content. Pages that stream into the
        // panel only append text; scrolling is done here.
        var liveTail = tailToggle ? tailToggle.checked : true;

        if (tailToggle) {
            tailToggle.addEventListener('change', function () {
                liveTail = tailToggle.checked;
                if (liveTail) pre.scrollTop = pre.scrollHeight;
            });
        }

        var observer = new MutationObserver(function () {
            if (liveTail) pre.scrollTop = pre.scrollHeight;
        });
        observer.observe(pre, { childList: true, characterData: true, subtree: true });

        // Copy to clipboard with fallback
        if (copyBtn) {
            copyBtn.addEventListener('click', function () {
                var text = pre.textContent || pre.innerText;
                copyToClipboard(text).then(function () {
                    var orig = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    setTimeout(function () { copyBtn.textContent = orig; }, 1500);
                }).catch(function () {
                    copyBtn.textContent = 'Failed';
                    setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1500);
                });
            });
        }

        // Clear log contents (and the server-side buffer if data-clear-url
        // is set). With a server buffer, the panel is cleared only once the
        // server has cleared it; a failure shows the page's error notice.
        if (clearBtn) {
            var clearUrl = panel.dataset.clearUrl;
            clearBtn.addEventListener('click', function () {
                if (!clearUrl) {
                    pre.textContent = '';
                    return;
                }
                apiCall(clearUrl, { method: 'DELETE' }, 'Clearing the log failed').then(function () {
                    pre.textContent = '';
                }).catch(function () {});
            });
        }
    });
}

// Copy text to clipboard. The async clipboard API is missing or refused in
// a non-secure context (HTTP on a LAN address), so a hidden textarea and
// execCommand('copy') are used then.
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText && window.isSecureContext) {
        return navigator.clipboard.writeText(text).catch(function () {
            return copyWithTextarea(text);
        });
    }
    return copyWithTextarea(text);
}

function copyWithTextarea(text) {
    return new Promise(function (resolve, reject) {
        try {
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            var ok = document.execCommand('copy');
            document.body.removeChild(ta);
            if (ok) {
                resolve();
            } else {
                reject(new Error('copy was refused'));
            }
        } catch (e) {
            reject(e);
        }
    });
}
