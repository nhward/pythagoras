"""Preview-card charts keep their models during display and fullscreen changes."""
import re
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

SCENARIOS = Path(__file__).parent / 'scenarios'
for scenario in ('obs_duplicates', 'data_homogeneous', 'var_dissimilar'):
    globals()['app_' + scenario] = create_app_fixture(app=str(SCENARIOS / (scenario + '.py')), scope='function')


@pytest.mark.ui
@pytest.mark.parametrize('scenario', ['obs_duplicates', 'data_homogeneous', 'var_dissimilar'])
def test_preview_charts_render_and_update_without_missing_models(page, request, scenario):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    app = request.getfixturevalue('app_' + scenario)
    page.set_viewport_size({'width': 1600, 'height': 1100})
    page.goto(app.url)
    plot = page.locator('.card .front .js-plotly-plot').first
    expect(plot).to_be_visible(timeout=30000)
    plot.evaluate('el => { window.initialPreviewPlot = el; }')
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(page.locator('.card.fullscreen-active')).to_be_visible()
    page.wait_for_function('''() => document.querySelector('.card .front .js-plotly-plot')?._context?.displayModeBar === true''')
    expect(plot).to_be_visible()
    page.locator('[id$="-ContractButton"]').click()
    expect(page.locator('.card.fullscreen-active')).to_have_count(0)
    page.wait_for_function('''() => document.querySelector('.card .front .js-plotly-plot')?._context?.displayModeBar === false''')
    assert plot.evaluate('el => el === window.initialPreviewPlot')
    expect(plot).to_be_visible()
    expect(page.get_by_text(re.compile('No model found for id'))).to_have_count(0)
    assert not errors, errors
