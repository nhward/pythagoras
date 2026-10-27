/* Opt-in, local-only activity sampling; no console forwarding or Shiny inputs. */
(() => {
    "use strict";
    if (window.PythagorasDiagnostics) return;
    let timer = null, observer = null, counts = new Map(), samples = [];
    let connected = null, last = 0;
    const label = node => {
        const el = node instanceof Element ? node : node?.parentElement;
        return (el?.closest('[id]')?.id || el?.tagName || 'document').slice(0, 160);
    };
    function count(key, amount = 1) {
        if (!counts.has(key) && counts.size >= 200) key = 'other';
        counts.set(key, (counts.get(key) || 0) + amount);
    }
    const events = 'shiny:connected shiny:disconnected shiny:busy shiny:idle shiny:inputchanged shiny:value shiny:recalculating shiny:recalculated';
    function onEvent(event) {
        if (event.type === 'shiny:connected') connected = true;
        if (event.type === 'shiny:disconnected') connected = false;
        // Names only: never record input/output values or dataset contents.
        count(`${event.type}:${String(event.name || event.target?.id || '').slice(0, 160)}`);
    }
    function sample() {
        const now = performance.now();
        samples.push({time: new Date().toISOString(), elapsedMs: Math.round(now - last),
            connected, visibility: document.visibilityState,
            activity: Object.fromEntries([...counts].sort((a, b) => b[1] - a[1]))});
        if (samples.length > 120) samples.shift();
        counts.clear();
        last = now;
    }
    function stop() {
        if (timer === null) return;
        clearInterval(timer);
        timer = null;
        observer.disconnect();
        observer = null;
        window.jQuery?.(document).off('.pythagorasDiagnostics');
        sample();
    }
    function start() {
        if (timer !== null) return;
        last = performance.now();
        window.jQuery?.(document).on(events.split(' ').map(e => e + '.pythagorasDiagnostics').join(' '), onEvent);
        observer = new MutationObserver(records => {
            for (const record of records) count(`DOM:${label(record.target)}:${record.attributeName || record.type}`);
        });
        observer.observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
        timer = setInterval(sample, 5000);
    }
    function snapshot() {
        return {running: timer !== null, samples: JSON.parse(JSON.stringify(samples)),
            pending: Object.fromEntries(counts),
            animations: (document.getAnimations?.() || []).slice(0, 100).map(a => ({
                target: label(a.effect?.target), state: a.playState,
                name: a.animationName || a.constructor.name
            }))};
    }
    window.PythagorasDiagnostics = {start, stop, snapshot, download() {
        const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(), null, 2)], {type: 'application/json'}));
        const a = document.createElement('a');
        a.href = url; a.download = 'pythagoras-diagnostics.json'; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }};
    window.addEventListener('pagehide', stop);
    if (new URLSearchParams(location.search).get('diagnostics') === '1') {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
        else start();
    }
})();
