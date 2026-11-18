import json

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/cand_train.py', scope='function')


@pytest.mark.ui
def test_search_progress_evidence_export_and_stale_settings(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1100})
    page.goto(app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    probe = page.locator('[id$="-ExportState"]')
    expect(probe).to_contain_text('accepted=False')
    expect(probe).to_contain_text('saved_acceptance=False')
    page.evaluate("() => { const id = document.querySelector('[id$=\"-Trials\"]').id; const ns = id.slice(0, -6); Shiny.setInputValue(id, 1, {priority: 'event'}); Shiny.setInputValue(ns + 'Seconds', 0.000001, {priority: 'event'}); }")
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('4 complete')
    best = page.locator('[id$="-Best"]')
    expect(best).to_contain_text('Best parameters found so far')
    expect(best).to_contain_text('Ridge (alpha)')
    expect(best).not_to_contain_text('cand_methods__')
    expect(page.locator('[id$="-History"] .plotly')).to_be_attached()
    plot = page.locator('[id$="-History"] .js-plotly-plot')
    page.wait_for_function("el => el.layout?.title?.text === 'Trial score'", arg=plot.element_handle())
    page.wait_for_function('''el => !(el.layout.annotations?.length) && !(el.layout.images?.length)
        && el._fullLayout.xaxis.visible && el._fullLayout.yaxis.visible''', arg=plot.element_handle())
    expect(plot.locator('.annotation')).to_have_count(0)
    expect(plot.locator('.imagelayer image')).to_have_count(0)
    assert plot.evaluate('el => el._context.displayModeBar') is False
    def expect_chart_fills_output():
        page.wait_for_function('''() => {
            const host = document.querySelector('[id$="-History"]');
            const plot = host?.querySelector('.js-plotly-plot');
            return plot?._fullLayout && Math.abs(plot._fullLayout.width - host.clientWidth) <= 4
                && Math.abs(plot._fullLayout.height - host.clientHeight) <= 4;
        }''')
    expect_chart_fills_output()
    plot.hover()
    expect(plot.locator('.modebar')).to_have_count(0)
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?._context?.displayModeBar === true""")
    assert plot.evaluate('el => el.layout.modebar.orientation') == 'v'
    assert plot.evaluate('el => el._context.displaylogo') is False
    expect(plot.locator('.modebar')).to_be_visible()
    expect_chart_fills_output()
    page.set_viewport_size({'width': 1400, 'height': 950})
    expect_chart_fills_output()
    expect(plot.locator('.plotly-logomark')).to_have_count(0)
    page.locator('[id$="-ContractButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?._context?.displayModeBar === false""")
    assert plot.evaluate("el => el.data.every(t => t.type === 'bar' && !t.error_y?.array)")
    expect(page.locator('[id$="-SDMultiplier"]')).to_have_count(0)
    expect(page.locator('[id$="-ChartView"]')).to_have_count(0)
    expect(progress).to_contain_text('Finished')
    expect(progress).to_contain_text('4 complete')
    expect(probe).to_contain_text('alpha=1.0')
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    table = page.locator('[id$="-Candidates"]')
    expect(table).to_contain_text('Candidate', timeout=30000)
    expect(table).not_to_contain_text('Complexity')
    page.wait_for_function("""el => el.getAnimations({subtree: true}).every(a =>
        a.effect.getTiming().iterations === Infinity || a.playState !== 'running')""",
        arg=page.locator('.card').first.element_handle())
    grid = table.locator('.shiny-data-grid')
    if grid.evaluate('el => el.scrollWidth > el.clientWidth'):
        grid.hover()
        page.mouse.wheel(600, 0)
        page.wait_for_function('el => el.scrollLeft > 0', arg=grid.element_handle())
    table.get_by_text('{"cand_methods__estimator__alpha": 0.01}', exact=True).click()
    expect(page.get_by_role('button', name='Accept selected tuning', exact=True)).to_have_count(0)
    expect(page.get_by_role('button', name='Revert acceptance', exact=True)).to_have_count(0)
    expect(probe).to_contain_text('accepted=False')
    expect(probe).to_contain_text('results=4')
    expect(probe).to_contain_text('alpha=1.0')
    # A changed seed blocks resume without discarding history; Cancel clears history before a new search.
    page.evaluate("""() => Shiny.setInputValue(document.querySelector('[id$="-Seed"]').id, 99, {priority:'event'})""")
    expect(progress).to_contain_text('stale')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('history retained')
    expect(probe).to_contain_text('results=4')
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('Fitting started')


@pytest.mark.ui
def test_flip_side_has_no_manual_checkpoint_or_download_area(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-Progress"]')).to_contain_text('Ready', timeout=30000)
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    expect(page.locator('[id$="-Candidates"]')).to_be_visible()
    expect(page.get_by_text('Search space and checkpoints', exact=True)).to_have_count(0)
    for suffix in ('Specification', 'Checkpoint', 'Results', 'Restore'):
        expect(page.locator(f'[id$="-{suffix}"]')).to_have_count(0)


implicit_app = create_app_fixture(app='../scenarios/cand_train_implicit.py', scope='function')


@pytest.mark.ui
def test_skopt_search_without_upstream_plan_uses_visible_implicit_cv(page, implicit_app):
    page.goto(implicit_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Implicit kfold resampling', timeout=30000)
    expect(progress).to_contain_text('5 folds')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('4 complete')
    expect(progress).to_contain_text('Implicit kfold resampling')
    expect(page.locator('[id$="-Best"]')).to_contain_text('Best parameters found so far')


repeated_app = create_app_fixture(app='../scenarios/cand_train_repeated.py', scope='function')


@pytest.mark.ui
def test_repeated_cv_exports_candidate_evidence_without_comparison_controls(page, repeated_app):
    page.goto(repeated_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('2 repeats × 3 folds')
    expect(page.locator('[id$="-ExportState"]')).to_contain_text('results=4')
    expect(page.locator('[id$="-ExportState"]')).to_contain_text('accepted=False')
    for name in ('EquivalenceTolerance', 'ComparisonLimit', 'DecisionProbability'):
        expect(page.locator(f'[id$="-{name}"]')).to_have_count(0)


multiple_app = create_app_fixture(app='../scenarios/cand_train_multiple.py', scope='function')


@pytest.mark.ui
def test_multiple_methods_publish_evidence_without_selecting_a_method(page, multiple_app):
    page.goto(multiple_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('2 starting-configuration evaluations', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('5 complete')
    plot = page.locator('[id$="-History"] .js-plotly-plot')
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?.data?.length === 2""")
    assert len(set(plot.evaluate('el => el.data.map(t => t.marker.color)'))) == 2
    assert plot.evaluate("el => el.data.every(t => t.type === 'bar' && !t.error_y?.array)")
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    table = page.locator('[id$="-Candidates"]')
    expect(table).to_contain_text('Method', timeout=30000)
    expect(table).to_contain_text('Ridge Regression')
    expect(table).to_contain_text('Decision Tree Regression')
    table.get_by_text('{"cand_methods__estimator__max_depth": 3}', exact=True).click()
    expect(page.get_by_role('button', name='Accept selected tuning', exact=True)).to_have_count(0)
    expect(page.locator('[id$="-MultiState"]')).to_have_text('No method accepted')



@pytest.mark.ui
def test_method_legend_is_only_visible_in_fullscreen(page, multiple_app):
    page.set_viewport_size({'width': 1200, 'height': 900})
    page.goto(multiple_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    plot = page.locator('[id$="-History"] .js-plotly-plot')
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?.layout?.showlegend === false""")
    expect(plot.locator('.legend')).to_have_count(0)
    colors = plot.evaluate('el => el.data.map(t => t.marker.color)')
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?.layout?.showlegend === true""")
    expect(plot.locator('.legend')).to_be_visible()
    assert plot.evaluate('el => el.data.map(t => t.marker.color)') == colors
    page.locator('[id$="-ContractButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?.layout?.showlegend === false""")
    expect(plot.locator('.legend')).to_have_count(0)


historical_app = create_app_fixture(app='../scenarios/cand_train_historical.py', scope='function')


@pytest.mark.ui
def test_completed_archive_can_be_reviewed_readonly_without_matching_search_settings(page, historical_app):
    page.goto(historical_app.url)
    expect(page.locator('[id$="-HistoricalState"]')).to_have_text('4 completed · readonly=True', timeout=30000)
    expect(page.locator('[id$="-Progress"]')).to_contain_text('stale')


restored_app = create_app_fixture(app='../scenarios/cand_train_restored.py', scope='function')
mismatched_app = create_app_fixture(app='../scenarios/cand_train_restored_mismatch.py', scope='function')


def plotted_trials(page):
    return page.locator('[id$="-History"] .js-plotly-plot').evaluate('el => el.data.flatMap(t => Array.from(t.x).map((x, i) => [x, t.y[i]]))')


@pytest.mark.ui
def test_restored_pause_resumes_and_appends_bars(page, restored_app):
    page.goto(restored_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('2 complete', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.[0]?.x?.length === 2")
    original = plotted_trials(page)
    expect(page.locator('[id$="-ExportState"]')).to_contain_text('saved_diagnostics=False')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('Fitting resumed')
    expect(progress).to_contain_text('4 complete', timeout=45000)
    page.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.[0]?.x?.length === 4")
    assert plotted_trials(page)[:2] == original


@pytest.mark.ui
def test_restored_hash_mismatch_keeps_bars_until_cancel(page, mismatched_app):
    page.goto(mismatched_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('stale', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.[0]?.x?.length === 2")
    original = plotted_trials(page)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-RestartDiagnostic"]')).to_have_count(0)
    expect(page.get_by_role('button', name='Start a new tuning search', exact=True)).to_have_count(0)
    expect(page.locator('[id$="-Message"]')).to_contain_text('history retained')
    assert plotted_trials(page) == original
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('Fitting started')
    expect(progress).to_contain_text('4 complete', timeout=45000)


session_restore_app = create_app_fixture(app='../scenarios/cand_train_session_restore.py', scope='function')


@pytest.mark.ui
def test_two_live_sessions_restore_imputation_name_and_resume_history(page, session_restore_app):
    page.goto(session_restore_app.url)
    page.get_by_role('button', name='Build workflow', exact=True).click()
    expect(page.locator('#Names')).to_have_text('miss_impute;cand_train')
    expect(page.locator('[id$="-Progress"]')).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Progress"]')).to_contain_text('2 complete', timeout=45000)
    saved = page.locator('#Snapshot').inner_text()
    saved_state = json.loads(saved)
    assert saved_state['search']['checkpoint']['specification']['fixed']['miss_impute__iterative__random_state'] == 1729
    original = plotted_trials(page)
    # Keep the first session alive: it must not reserve names in this session.
    restored = page.context.new_page()
    restored.goto(session_restore_app.url)
    restored.locator('#Saved').fill(saved)
    restored.get_by_role('button', name='Build workflow', exact=True).click()
    expect(restored.locator('#Names')).to_have_text('miss_impute;cand_train')
    expect(restored.locator('[id$="-Progress"]')).to_contain_text('2 complete', timeout=30000)
    restored.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.[0]?.x?.length === 2")
    assert plotted_trials(restored) == original
    restored.evaluate("() => Shiny.setInputValue('cand_train-Trials', 0, {priority:'event'})")
    restored.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(restored.locator('[id$="-Message"]')).to_contain_text('Fitting resumed')
    expect(restored.locator('[id$="-Progress"]')).to_contain_text('4 complete', timeout=45000)
    restored.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.[0]?.x?.length === 4")
    assert plotted_trials(restored)[:2] == original
    expect(restored.locator('[id$="-RestartDiagnostic"]')).to_have_count(0)
    restored.close()


@pytest.mark.ui
def test_cancel_stops_running_search_and_clears_history_before_fresh_play(page, app):
    page.goto(app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Running')
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready', timeout=15000)
    expect(page.locator('[id$="-Message"]')).to_contain_text('history cleared')
    expect(page.locator('[id$="-ExportState"]')).to_contain_text('results=0')
    page.wait_for_function("() => document.querySelector('[id$=\"-History\"] .js-plotly-plot')?.data?.every(t => t.type !== 'bar' || !t.x?.length)")
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('Fitting started')
    expect(progress).to_contain_text('4 complete', timeout=45000)


@pytest.mark.ui
def test_legacy_local_choices_block_until_committed_upstream(page, session_restore_app):
    page.goto(session_restore_app.url)
    page.locator('#Saved').fill(json.dumps({'inputs': {
        'ModelValues': '{"cand_methods__estimator__alpha": [99]}',
        'Method': 'grid', 'Seed': 1729, 'Execution': 'sequential', 'Workers': 1}}))
    page.get_by_role('button', name='Build workflow', exact=True).click()
    expect(page.locator('[id$="-Progress"]')).to_contain_text('Commit those choices in Tuning parameters', timeout=30000)
    saved = json.loads(page.locator('#Snapshot').inner_text())
    assert 'ModelValues' not in saved['inputs']
    assert json.loads(saved['legacy_model_values']) == {'cand_methods__estimator__alpha': [99]}
    assert page.get_by_label('Additional model choices (JSON)', exact=True).count() == 0


failure_app = create_app_fixture(app='../scenarios/cand_train_failures.py', scope='function')


@pytest.mark.ui
def test_first_ten_fit_failures_are_reported_in_one_persistent_notification(page, failure_app):
    page.goto(failure_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    notice = page.locator('.shiny-notification-error')
    expect(notice).to_contain_text('Training failures', timeout=60000)
    expect(notice).to_contain_text('alpha')
    expect(notice).to_contain_text('InvalidParameterError')
    expect(progress).to_contain_text('14 failed', timeout=60000)
    expect(notice).to_have_count(1)
    expect(notice.locator('li')).to_have_count(10)
    expect(notice).to_contain_text('Trial 2')
    expect(notice).not_to_contain_text('Trial 12')
    notice.locator('.shiny-notification-close').click()
    expect(notice).to_have_count(0)
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready')
    expect(notice).to_have_count(0)
