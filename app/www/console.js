
(function () {
  console.info("[console] Script running")

    // --- queue + helpers -------------------------------------------------------
  const queue = [];
  function canSendViaSet()  { return typeof window.Shiny?.setInputValue  === "function"; }
  function canSendViaOn()   { return typeof window.Shiny?.onInputChange === "function"; }
  function isShinyReady()   { return canSendViaSet() || canSendViaOn(); }

  function _sendNow(id, value) {
    if (canSendViaSet()) {
      // Shiny.setInputValue(id, value, options)
      Shiny.setInputValue(id, value, { priority: "event" });
    } else if (canSendViaOn()) {
      // Shiny.onInputChange(id, value)  (no options arg)
      Shiny.onInputChange(id, value);
    } else {
      // not ready yet
      queue.push([id, value]);
    }
  }

  function flushQueue() {
    if (!isShinyReady()) return;
    while (queue.length) {
      const [id, value] = queue.shift();
      _sendNow(id, value);
    }
  }

  // Try to flush when Shiny connects/initializes (R & Py variants), and also after DOM ready
  document.addEventListener("shiny:connected", flushQueue);
  document.addEventListener("shiny:sessioninitialized", flushQueue);
  document.addEventListener("DOMContentLoaded", flushQueue);

  // --- your log wiring -------------------------------------------------------
  const LOG_LEVELS = { log: 1, info: 2, warn: 3, error: 4 };
  let threshold = LOG_LEVELS.info;

  function fmtArgs(args) {
    return args.map(a => {
      try { return typeof a === "string" ? a : JSON.stringify(a); }
      catch { return String(a); }
    }).join(" ");
  }

  function sendConsole(level, args) {
    const payload = { level, text: fmtArgs(args), ts: Date.now() };
    _sendNow("Console_log", payload);
  }

  const origInfo = console.info;
  const origWarn = console.warn;
  const origErr  = console.error;

  console.info = function (...args) {
    origInfo.apply(console, args);
    if (LOG_LEVELS.info >= threshold) sendConsole("info", args);
  };
  console.warn = function (...args) {
    origWarn.apply(console, args);
    if (LOG_LEVELS.warn >= threshold) sendConsole("warn", args);
  };
  console.error = function (...args) {
    origErr.apply(console, args);
    if (LOG_LEVELS.error >= threshold) sendConsole("error", args);
  };

  // optional: expose a tiny control API
  window.PythagorasConsole = {
    setThreshold(name) { if (LOG_LEVELS[name]) threshold = LOG_LEVELS[name]; },
    getThreshold() { return Object.entries(LOG_LEVELS).find(([,v]) => v === threshold)?.[0]; }
  };

  // restore originals when the session ends / page unloads
  function restore() {
    console.info = origInfo; console.warn = origWarn; console.error = origErr;
  }
  document.addEventListener("shiny:sessionended", restore);
  document.addEventListener("shiny:disconnected", restore);
  window.addEventListener("beforeunload", restore);
})();