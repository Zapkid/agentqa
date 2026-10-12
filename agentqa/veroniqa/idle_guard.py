"""A small script for the /talk/ page that stops an unattended tab from costing function CPU.

Why: Vercel closes a WebSocket when its function reaches the plan's maximum duration (5 minutes on
Hobby). Streamlit's page then reconnects by itself, and every reconnect wakes cold function
instances (it fetches /_stcore/health and /_stcore/host-config at once, so two start), imports the
whole app and runs a new session. A tab left open in the background does that every 5 minutes for
as long as it stays open, which on a free plan's CPU allowance adds up to weeks of CPU.

What: when the connection drops while the tab is hidden, or while nobody has touched it for two
minutes, the script lets Streamlit's own retry loop hang on its first request instead of reaching
the server, and shows a one-line notice. The first sign the person is back (the tab becomes
visible, or a click, key press, scroll or touch) reloads the page, which starts a fresh session. A
person who is actually using the app is unaffected: their reconnect proceeds as before.

It wraps `WebSocket` and `fetch` before Streamlit's own scripts run (it sits in the page's head,
and Streamlit's bundle is a deferred module), so Streamlit's code is not modified.
"""

from __future__ import annotations

IDLE_MS = 120_000

_TEMPLATE = """(function () {
  var IDLE_MS = __IDLE_MS__;
  var lastInput = Date.now();
  var paused = false;
  var NativeWebSocket = window.WebSocket;
  var nativeFetch = window.fetch.bind(window);

  function resume() { if (paused) { window.location.reload(); } }
  ['pointerdown', 'keydown', 'wheel', 'touchstart', 'scroll'].forEach(function (name) {
    window.addEventListener(name, function () { lastInput = Date.now(); resume(); },
      { capture: true, passive: true });
  });
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) { resume(); }
  });

  function pause() {
    if (paused) { return; }
    paused = true;
    var note = document.createElement('div');
    note.setAttribute('role', 'status');
    note.id = 'vq-paused';
    note.textContent = 'Paused while you were away. Click or press a key to continue.';
    note.style.cssText = 'position:fixed;left:50%;bottom:16px;transform:translateX(-50%);' +
      'z-index:99999;padding:8px 16px;border-radius:100px;background:#18181f;color:#e8e8f0;' +
      'border:1px solid #2a2a35;font:14px system-ui,sans-serif';
    (document.body || document.documentElement).appendChild(note);
  }

  window.WebSocket = class extends NativeWebSocket {
    constructor(url, protocols) {
      super(url, protocols);
      if (String(url).indexOf('/_stcore/stream') !== -1) {
        this.addEventListener('close', function () {
          if (document.hidden || Date.now() - lastInput > IDLE_MS) { pause(); }
        });
      }
    }
  };

  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input : (input && input.url) || String(input);
    if (paused && url.indexOf('/_stcore/') !== -1) { return new Promise(function () {}); }
    return nativeFetch(input, init);
  };
})();"""

SCRIPT = _TEMPLATE.replace("__IDLE_MS__", str(IDLE_MS))
