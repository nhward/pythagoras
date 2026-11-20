"""Large portable search archives stage in IndexedDB, not Web Storage."""
from pathlib import Path

import pytest
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='scenarios/bookmark_default.py', scope='function')


@pytest.mark.ui
def test_large_configurations_stage_restore_and_switch_without_session_storage_payload(page, app):
    page.goto(app.url)
    page.wait_for_function('window.indexedDB && window.jQuery')
    script = (Path(__file__).parents[1] / 'app/www/pythagoras.js').read_text()
    script = script[script.index('    const bookmarkDatabaseName ='):script.rfind('\n});')]
    script = script.replace('window.jQuery(document).one("shiny:idle", publishBookmarkStartup);', '')
    script = script.replace('window.setTimeout(() => window.location.replace(url), 250);', 'window.__reloadUrl = url.href;')
    # Exercise actual staging/startup code; only replace the page-navigation and
    # Shiny event boundaries so assertions can inspect each side of the reload.
    harness = '''(function(Shiny) {
    ''' + script + '''
        window.__stageConfig = stageBookmarkAndReload;
        window.__restoreConfig = publishBookmarkStartup;
        window.__savedRecords = bookmarkRecords;
    })({addCustomMessageHandler: () => {}, setInputValue: (id, value) => {window.__startup = value;}});'''
    for marker, size in [('first', 6 * 1024 * 1024), ('second', 7 * 1024 * 1024)]:
        page.evaluate(harness)
        page.evaluate('''async ({marker, size}) => {
            await window.__stageConfig({version: 2, settings: {}, layout: [], marker,
                pending: {blobs: {fold: 'A'.repeat(size)}}});
        }''', {'marker': marker, 'size': size})
        token = page.evaluate("sessionStorage.getItem('pythagoras-bookmark-once')")
        assert len(token) < 200 and 'indexeddb-stage' in token
        assert page.evaluate("async () => (await window.__savedRecords()).some(r => r.configuration?.marker)") is False
        page.evaluate('async () => await window.__restoreConfig()')
        assert page.evaluate('window.__startup.source') == 'staged'
        assert page.evaluate('window.__startup.configuration.marker') == marker
        assert page.evaluate('window.__startup.configuration.pending.blobs.fold.length') == size
        assert page.evaluate("sessionStorage.getItem('pythagoras-bookmark-once')") is None
        assert page.evaluate('''async () => {
            const db = await new Promise((resolve, reject) => {
                const request = indexedDB.open('pythagoras-bookmarks', 2);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
            try { return await new Promise(resolve => {
                const request = db.transaction('staged-configurations').objectStore('staged-configurations').count();
                request.onsuccess = () => resolve(request.result);
            }); } finally { db.close(); }
        }''') == 0
