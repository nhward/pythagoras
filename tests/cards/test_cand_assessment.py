"""Assessment charts, restored display controls and diagnostic export behavior."""
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

SCENARIOS = Path(__file__).resolve().parents[1] / 'scenarios'
app = create_app_fixture(app=str(SCENARIOS / 'cand_assessment.py'), scope='function')
hard_app = create_app_fixture(app=str(SCENARIOS / 'cand_assessment_hard.py'), scope='function')
restored_app = create_app_fixture(app=str(SCENARIOS / 'cand_assessment_restored.py'), scope='function')
ensemble_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_assessment.py'), scope='function')


def settings(page):
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()


def plot(page):
    return page.locator('[id$="-Chart"] .js-plotly-plot')


@pytest.mark.ui
def test_unchecked_busy_ensemble_does_not_block_individual_assessment(page, ensemble_app):
    page.goto(ensemble_app.url)
    expect(page.locator('#Background')).to_have_text('ensemble running', timeout=30000)
    expect(page.locator('#cand_ensemble-Apply input')).not_to_be_checked()
    expect(page.locator('#cand_assessment-Status')).to_contain_text('40 observations', timeout=30000)
    expect(page.locator('#cand_assessment-Title')).to_have_text('Assessment of Ridge 1 model')
    page.wait_for_function("() => document.querySelector('#cand_assessment-Chart .js-plotly-plot')?.data?.some(t => t.name === '1:1')")
    expect(page.locator('#cand_ensemble-Busy')).to_contain_text('Evaluating member models and ensemble')


@pytest.mark.ui
@pytest.mark.parametrize(('fixture', 'title'), [
    ('app', 'Assessment of Ridge 1 model'),
    ('hard_app', 'Assessment of ensemble model'),
])
def test_title_names_assessed_model(page, request, fixture, title):
    application = request.getfixturevalue(fixture)
    page.goto(application.url)
    expect(page.locator('[id$="-Title"]')).to_have_text(title, timeout=30000)

#TODO this test is failing
@pytest.mark.ui
def test_regression_grouping_does_not_refit_and_missing_winner_waits(page, app):
    page.goto(app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('40 observations', timeout=30000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('unchanged')
    expect(page.locator('[id$="-Calls"]')).to_have_text('collections=1')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.some(t => t.name === '1:1')''')
    page.wait_for_function('''() => {
        const layout = document.querySelector('[id$="-Chart"] .js-plotly-plot')?._fullLayout;
        return layout && Math.abs(layout.xaxis._length - layout.yaxis._length) < 2;
    }''')
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.expand-btn').click()
    page.wait_for_function('''() => {
        const layout = document.querySelector('[id$="-Chart"] .js-plotly-plot')?._fullLayout;
        return layout && layout.font.size === 13 && Math.abs(layout.xaxis._length - layout.yaxis._length) < 2;
    }''')
    page.keyboard.press('Escape')
    settings(page)
    page.locator('[id$="-Color"]').select_option('stratum')
    page.locator('[id$="-Symbol"]').select_option('sensitive')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.some(t => t.name?.includes('stratum=A'))''')
    expect(page.locator('[id$="-Calls"]')).to_have_text('collections=1')
    page.get_by_role('button', name='Clear upstream winner').click()
    expect(status).to_contain_text('Waiting for an upstream', timeout=10000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('unchanged')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.length === 0''')


@pytest.mark.ui
def test_hard_vote_roc_and_confusion_switch_without_refitting(page, hard_app):
    page.goto(hard_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('ROC score: vote share', timeout=30000)
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.some(t => t.name?.includes('AUC'))''')
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('unchanged')
    settings(page)
    expect(page.locator('[id$="-Class"]')).to_have_value('1')
    page.locator('[id$="-Class"]').select_option('0')
    page.locator('[id$="-ChartType"]').select_option('confusion')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.[0]?.type === 'heatmap' ''')
    expect(page.locator('[id$="-Calls"]')).to_have_text('collections=1')


@pytest.mark.ui
def test_restored_groups_and_flip_table(page, restored_app):
    page.goto(restored_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('40 observations', timeout=30000)
    settings(page)
    expect(page.locator('[id$="-Color"]')).to_have_value('stratum')
    expect(page.locator('[id$="-Symbol"]')).to_have_value('sensitive')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.some(t => t.name?.includes('sensitive=G1'))''')
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.flip-btn').click()
    expect(page.locator('[id$="-Table2"]')).to_be_visible(timeout=15000)
    expect(page.locator('[id$="-Table"]')).to_contain_text('Mean error (bias)', timeout=15000)
    expect(page.locator('[id$="-Table"]')).to_contain_text('sensitive=G1')
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.flip-btn').click()
    expect(plot(page)).to_be_visible()
    expect(page.locator('[id$="-Calls"]')).to_have_text('collections=1')
