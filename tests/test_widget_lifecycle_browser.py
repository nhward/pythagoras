"""Real widget teardown must preserve Jupyter bookkeeping and report failures."""
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

pytestmark = pytest.mark.ui
app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/widget_lifecycle_browser.py'), scope='function')


def capture_errors(page):
    errors = []

    def log(message):
        if message.type == 'error':
            errors.append([argument.evaluate('(e) => ({text:String(e),message:e?.message,stack:e?.stack})')
                           for argument in message.args])
    page.on('console', log)
    page.on('pageerror', lambda error: errors.append(str(error)))
    return errors


def test_replacement_and_close_release_views_without_javascript_errors(page, app):
    errors = capture_errors(page)
    page.goto(app.url)
    plot = page.locator('.js-plotly-plot')
    expect(plot).to_be_visible(timeout=30000)
    page.get_by_role('button', name='Update', exact=True).click()
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data?.[0]?.y?.[0] === 4")
    for _ in range(3):
        plot.evaluate('el => { window.previousPlot = el; }')
        page.get_by_role('button', name='Replace', exact=True).click()
        page.wait_for_function('''() => {
            const plot = document.querySelector('.js-plotly-plot');
            return plot && plot !== window.previousPlot && !window.previousPlot.isConnected;
        }''')
        expect(plot).to_be_visible()
    page.get_by_role('button', name='Close', exact=True).click()
    expect(plot).to_have_count(0)
    page.wait_for_timeout(100)  # Let the asynchronous close promise finish.
    assert not errors, errors


def test_unexpected_teardown_errors_are_not_suppressed(page, app):
    errors = capture_errors(page)
    page.goto(app.url)
    expect(page.locator('.js-plotly-plot')).to_be_visible(timeout=30000)
    page.evaluate('''() => new Promise(resolve => require(['@jupyter-widgets/base'], base => {
        base.DOMWidgetView.prototype.remove = function () { throw new Error('Deliberate teardown failure'); };
        resolve();
    }))''')
    page.get_by_role('button', name='Close', exact=True).click()
    page.wait_for_timeout(700)
    assert any('Deliberate teardown failure' in str(error) for error in errors), errors


mixed_app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/widget_binding_mixed.py'), scope='function')


def test_cached_and_new_outputs_share_widget_manager(page, mixed_app):
    errors = capture_errors(page)
    page.goto(mixed_app.url)
    expect(page.locator('#Early .js-plotly-plot')).to_be_visible(timeout=15000)
    page.get_by_role('button', name='Add chart', exact=True).click()
    expect(page.locator('#Later .js-plotly-plot')).to_be_visible(timeout=15000)
    page.get_by_role('button', name='Refresh charts', exact=True).click()
    page.wait_for_function("() => document.querySelector('#Early .js-plotly-plot')?.data?.[0]?.y?.[0] === 4")
    assert page.evaluate("() => Shiny.outputBindings.getBindings().filter(b => b.binding.name === 'shiny.IPyWidgetOutput').length") == 1
    assert not errors, errors
