"""Card-wide Plotly presentation survives updates, fullscreen and late widgets."""
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

pytestmark = pytest.mark.ui
app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/plotly_presentation.py'), scope='function')
models_app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/cand_models.py'), scope='function')
assessment_app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/cand_assessment.py'), scope='function')
resample_app = create_app_fixture(app=str(Path(__file__).parent / 'scenarios/obs_resample.py'), scope='function')


def check_presentation(page, full, count):
    page.wait_for_function('''({full, count}) => {
        const plots = [...document.querySelectorAll('.js-plotly-plot')];
        return plots.length === count && plots.every(p =>
            p._context?.displayModeBar === full && p._context?.displaylogo === false &&
            p._fullLayout?.modebar?.orientation === 'v' &&
            (full ? p.layout.margin.r >= 35 : p.layout.margin.r === 5));
    }''', arg={'full': full, 'count': count})
    if full:
        for bar in page.locator('.modebar').all():
            expect(bar).to_be_visible()
        expect(page.locator('.modebar-btn[data-title="Produced with Plotly.js"]')).to_have_count(0)
    else:
        for bar in page.locator('.modebar').all():
            expect(bar).not_to_be_visible()


def test_plotly_presentation_for_all_widget_lifecycles(page, app):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(app.url)
    check_presentation(page, False, 2)
    page.get_by_role('button', name='Add chart', exact=True).click()
    check_presentation(page, False, 3)
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click(force=True)
    check_presentation(page, True, 3)
    page.get_by_role('button', name='Update charts', exact=True).click()
    check_presentation(page, True, 3)
    page.locator('[id$="-ContractButton"]').click(force=True)
    check_presentation(page, False, 3)
    assert not errors, errors


@pytest.mark.parametrize('fixture', ['models_app', 'assessment_app', 'resample_app'])
def test_existing_card_charts_share_fullscreen_presentation(page, request, fixture):
    application = request.getfixturevalue(fixture)
    page.goto(application.url)
    expect(page.locator('.js-plotly-plot').first).to_be_visible(timeout=30000)
    count = page.locator('.js-plotly-plot').count()
    check_presentation(page, False, count)
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click(force=True)
    check_presentation(page, True, count)
    page.locator('[id$="-ContractButton"]').click(force=True)
    check_presentation(page, False, count)
