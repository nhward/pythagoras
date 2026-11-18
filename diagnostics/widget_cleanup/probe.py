"""Run explicitly with pytest -s; intentionally outside normal test discovery."""
import json
import os
from importlib.metadata import version
from pathlib import Path

import pytest
import shinywidgets
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='./app.py', scope='function')
EVIDENCE = Path('/tmp/pythagoras-widget-cleanup-evidence.jsonl')


@pytest.mark.parametrize('action', ['Update', 'Replace', 'Remove', 'Close'])
def test_lifecycle(page, app, action):
    errors, page_errors, removal_events = [], [], []
    page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
    page.on('console', lambda msg: removal_events.append(msg.text) if msg.text.startswith('WIDGET_REMOVAL ') else None)
    page.on('pageerror', lambda error: page_errors.append(str(error)))
    page.goto(app.url)
    expect(page.locator('#Chart .js-plotly-plot')).to_be_visible(timeout=30000)
    before = json.loads(page.locator('#Probe').inner_text())
    asset = page.locator('script[src$="/output.js"]').first.evaluate('el => el.src')
    asset_text = page.request.get(asset).text() if asset else ''
    manager_asset = page.locator('script[src$="/libembed-amd.js"]').first.evaluate('el => el.src')
    manager_text = page.request.get(manager_asset).text()
    guarded_registry = 't.model&&t.model.views&&delete t.model.views[r]' in manager_text
    if os.environ.get('WIDGET_CLEANUP_VARIANT', '').startswith('registry-guard'):
        assert guarded_registry
    page.locator('#' + action).click()
    expect(page.locator('#Probe')).to_contain_text('"action": "' + ('render' if action == 'Replace' else action.lower()) + '"')
    if action == 'Replace':
        expect(page.locator('#Probe')).not_to_contain_text(before['model_id'])
        expect(page.locator('#Chart .js-plotly-plot')).to_be_visible()
    # Closing is asynchronous; allow its caught console errors to settle.
    page.wait_for_timeout(1500)
    result = {'variant': os.environ.get('WIDGET_CLEANUP_VARIANT', 'installed'),
              'package_path': shinywidgets.__file__, 'browser': page.context.browser.browser_type.name, 'action': action,
              'versions': {p: version(p) for p in ['shiny', 'shinywidgets', 'ipywidgets', 'anywidget', 'plotly']},
              'before': before, 'after': json.loads(page.locator('#Probe').inner_text()),
              'console_errors': errors, 'page_errors': page_errors,
              'served_asset_deletes_model': 'delete v.model;' in asset_text, 'removal_events': removal_events, 'served_manager_guards_registry': guarded_registry}
    with EVIDENCE.open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    print(json.dumps(result, indent=2))
    assert not errors, errors
    assert not page_errors, page_errors
