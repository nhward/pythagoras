"""Real browser evaluation, settings and opt-in exports."""
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

SCENARIOS = Path(__file__).resolve().parents[1] / 'scenarios'
app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble.py'), scope='function')
classification_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_classification.py'), scope='function')
restored_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_restored.py'), scope='function')


def settings(page):
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()


def plot(page):
    return page.locator('[id$="-Chart"] .js-plotly-plot')


@pytest.mark.ui
def test_five_default_members_metric_sorting_subset_and_export(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('5 members', timeout=60000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('incoming')
    expect(plot(page)).to_be_visible()
    assert plot(page).evaluate('el => el.data[0].x.length') == 6
    assert plot(page).evaluate('el => el.data[0].x.every((x,i,a) => !i || a[i-1] <= x)')
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=5;fitted=5', timeout=30000)
    settings(page)
    expect(page.locator('[id$="-Models"] input:checked')).to_have_count(5)
    page.locator('[id$="-Metric"]').select_option('r2')
    page.wait_for_function('''() => {
        const p = document.querySelector('[id$="-Chart"] .js-plotly-plot');
        return p?.layout.xaxis.title.text.startsWith('R²') && p.data[0].x.every((x,i,a) => !i || a[i-1] >= x);
    }''')
    page.locator('[id$="-Models"] input[value="method_4"]').uncheck()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=4;fitted=5', timeout=30000)
    page.locator('[id$="-Style"]').select_option('median')
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 members', timeout=30000)
    page.locator('[id$="-Apply"] input').uncheck()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('incoming', timeout=10000)


@pytest.mark.ui
def test_classification_and_hard_voting(page, classification_app):
    page.goto(classification_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('5 members', timeout=60000)
    settings(page)
    expect(page.locator('[id$="-Metric"]')).to_have_value('balanced_accuracy')
    expect(page.locator('[id$="-Style"]')).to_have_value('soft')
    page.locator('[id$="-Style"]').select_option('hard')
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=5;fitted=5', timeout=30000)


@pytest.mark.ui
def test_restored_subset_style_metric_and_enabled_export(page, restored_app):
    page.goto(restored_app.url)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2;fitted=5', timeout=60000)
    settings(page)
    expect(page.locator('[id$="-Models"] input:checked')).to_have_count(2)
    expect(page.locator('[id$="-Style"]')).to_have_value('median')
    expect(page.locator('[id$="-Metric"]')).to_have_value('r2')


progress_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_progress.py'), scope='function')


@pytest.mark.ui
def test_training_progress_does_not_restart_or_hide_unchanged_ensemble(page, progress_app):
    page.goto(progress_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    expect(plot(page)).to_be_visible()
    probe = page.locator('[id$="-ProgressProbe"]')
    page.wait_for_function(r'''() => {
        const text = document.querySelector('[id$="-ProgressProbe"]')?.textContent || '';
        return Number(text.match(/updates=(\d+)/)?.[1]) >= 15;
    }''')
    expect(probe).to_contain_text('fits=1')
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2', timeout=10000)
    settings(page)
    page.locator('[id$="-Style"]').select_option('median')
    expect(probe).to_contain_text('fits=2', timeout=10000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2', timeout=10000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members')
