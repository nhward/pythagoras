"""Real browser evaluation, settings and opt-in exports."""
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

SCENARIOS = Path(__file__).resolve().parents[1] / 'scenarios'
app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble.py'), scope='function')
classification_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_classification.py'), scope='function')
restored_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_restored.py'), scope='function')
dynamic_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_dynamic.py'), scope='function')
training_app = create_app_fixture(app=str(SCENARIOS / 'cand_train_ensemble.py'), scope='function')
dormant_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_dormant.py'), scope='function')
saved_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_saved.py'), scope='function')
exhausted_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_exhausted.py'), scope='function')
diagnostics_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_diagnostics.py'), scope='function')
busy_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_busy.py'), scope='function')


def settings(page):
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()


def plot(page):
    return page.locator('[id$="-Chart"] .js-plotly-plot')


def expect_chart_to_fill_panel(page, output):
    measure = '''suffix => {
        const pane = document.querySelector('.ensemble-chart');
        const host = pane?.querySelector(`[id$="-${suffix}"]`);
        const plot = host?.querySelector('.js-plotly-plot');
        if (!plot?._fullLayout) return null;
        const a = pane.getBoundingClientRect(), b = host.getBoundingClientRect();
        const front = pane.closest('.front').getBoundingClientRect();
        return [a.width, a.height, b.width, b.height, plot._fullLayout.width, plot._fullLayout.height,
            a.bottom - front.bottom];
    }'''
    try:
        page.wait_for_function('suffix => { const d = (' + measure + ''')(suffix);
            return d && d[1] > 100 && Math.abs(d[1]-d[3]) < 4
                && Math.abs(d[0]-d[2]) < 4 && Math.abs(d[4]-d[2]) < 4
                && Math.abs(d[5]-d[3]) < 4 && Math.abs(d[6]) < 4; }''', arg=output)
    except Exception:
        ancestors = page.locator(f'[id$="-{output}"]').evaluate('''el => {
            const chain = [];
            for (let node = el; node && chain.length < 6; node = node.parentElement) {
                const css = getComputedStyle(node);
                chain.push([node.id, node.className, node.style.cssText, css.height, css.display, css.flex]);
            }
            return chain;
        }''')
        pytest.fail(f'{output} pane/host/plot dimensions: {page.evaluate(measure, output)}; ancestors: {ancestors}')


@pytest.mark.ui
def test_five_default_members_metric_sorting_subset_and_export(page, app):
    errors = []
    phase = 'startup'
    page.on('pageerror', lambda error: errors.append(f'{phase}: {error}'))
    page.on('console', lambda message: errors.append(f'{phase}: {message.text}') if message.type == 'error' else None)
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('5 members', timeout=60000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('incoming')
    expect(page.locator('[id$="-Status"]')).to_contain_text("Evaluation only; model 'Ridge 1' passed through")
    expect(plot(page)).to_be_visible()
    plot(page).evaluate('el => { window.initialEnsemblePlot = el; }')
    assert plot(page).evaluate('el => el.data[0].x.length') == 6
    assert plot(page).evaluate('el => el.data[0].x.every((x,i,a) => !i || a[i-1] <= x)')
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=5;fitted=5', timeout=30000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Ensemble model passed through')
    phase = 'sidebar'
    settings(page)
    expect(page.locator('[id$="-Models"] input:checked')).to_have_count(5)
    phase = 'metric'
    page.locator('[id$="-Metric"]').select_option('r2')
    page.wait_for_function('''() => {
        const p = document.querySelector('[id$="-Chart"] .js-plotly-plot');
        return p?.layout.xaxis2.title.text.startsWith('R²') && p.data[0].x.every((x,i,a) => !i || a[i-1] >= x);
    }''')
    phase = 'membership'
    page.locator('[id$="-Models"] input[value="method_4"]').uncheck()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=4;fitted=5', timeout=30000)
    phase = 'style'
    page.locator('[id$="-Style"]').select_option('median')
    # The previous mean result already says "4 members". Wait for the newly
    # evaluated median export before clicking a footer that may still move.
    expect(page.locator('[id$="-ExportStyle"]')).to_have_text('median', timeout=30000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 members', timeout=30000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Ensemble model passed through')
    page.locator('[id$="-Apply"] input').uncheck()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('incoming', timeout=10000)
    assert plot(page).evaluate('el => el === window.initialEnsemblePlot')
    page.locator('.card').first.hover()
    phase = 'expand'
    page.locator('[id$="-ExpandButton"]').click()
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?._context.displayModeBar === true''')
    phase = 'contract'
    page.locator('[id$="-ContractButton"]').click()
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?._context.displayModeBar === false''')
    assert plot(page).evaluate('el => el === window.initialEnsemblePlot')
    phase = 'flip'
    page.locator('[id$="-FlipButton"]').click(force=True)
    for name in ('MAE', 'RMSE', 'R²'):
        expect(page.locator('[id$="-Table"]').get_by_role('columnheader', name=name, exact=True)).to_be_visible()
    assert not errors, errors


@pytest.mark.ui
def test_settings_changed_before_enabling_ensemble_complete_export(page, dynamic_app):
    page.goto(dynamic_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('5 members', timeout=30000)
    settings(page)
    page.locator('[id$="-Models"] input[value="method_4"]').uncheck()
    page.locator('[id$="-Style"]').select_option('median')
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 members', timeout=30000)
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('#ExportProbe')).to_have_text('ensemble=4;fitted=5', timeout=10000)
    expect(page.locator('[id$="-PipelineBusy"]')).to_have_text('')


@pytest.mark.ui
def test_upstream_training_cancel_responds_with_ensemble_enabled(page, training_app):
    page.goto(training_app.url)
    progress = page.locator('#cand_train-Progress')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('#cand_ensemble-Status')).to_contain_text('2 members', timeout=60000)
    page.locator('#cand_ensemble-Apply input').check()
    expect(page.locator('#ExportProbe')).to_have_text('ensemble=2', timeout=15000)
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready', timeout=15000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Running', timeout=15000)
    page.get_by_role('button', name='Cancel current tuning', exact=True).click()
    expect(progress).to_contain_text('Ready', timeout=15000)
    expect(page.locator('#cand_ensemble-Apply input')).to_be_checked()


@pytest.mark.ui
def test_no_reviewed_exports_leave_ensemble_dormant_during_training(page, dormant_app):
    page.goto(dormant_app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('Choose weighted method winners', timeout=30000)
    page.wait_for_function("() => Number(document.querySelector('[id$=\"-Activity\"]')?.textContent.match(/updates=(\\d+)/)?.[1]) >= 10")
    expect(page.locator('[id$="-Activity"]')).to_contain_text('evaluations=0;cancellations=0')
    status.evaluate('''el => {
        window.footerChanges = 0;
        window.footerObserver = new MutationObserver(() => window.footerChanges++);
        window.footerObserver.observe(el, {childList: true, characterData: true, subtree: true, attributes: true});
    }''')
    page.wait_for_function("() => Number(document.querySelector('[id$=\"-Activity\"]').textContent.match(/updates=(\\d+)/)[1]) >= 20")
    assert page.evaluate('window.footerChanges') == 0
    expect(page.locator('[id$="-Busy"]')).to_have_text('')
    page.get_by_role('button', name='Export reviewed models', exact=True).click()
    expect(status).to_contain_text('2 members', timeout=30000)
    expect(page.locator('[id$="-Activity"]')).to_contain_text('evaluations=1;')


@pytest.mark.ui
def test_saved_chart_and_table_restore_without_new_evaluation(page, saved_app):
    page.goto(saved_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.[0]?.x?.length === 3")
    original = plot(page).evaluate('el => el.data[0].x')
    page.get_by_role('button', name='Save and restore', exact=True).click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Saved evaluation scores', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.[0]?.x?.length === 3")
    assert plot(page).evaluate('el => el.data[0].x') == original
    expect(page.locator('#Evaluations')).to_have_text('evaluations=1')
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click(force=True)
    for name in ('MAE', 'RMSE', 'R²'):
        expect(page.locator('[id$="-Table"]').get_by_role('columnheader', name=name, exact=True)).to_be_visible()
    expect(page.locator('#Evaluations')).to_have_text('evaluations=1')


@pytest.mark.ui
@pytest.mark.parametrize('restore_button', ['Restore legacy scores', 'Restore scores without diagnostics'])
def test_missing_saved_residuals_refresh_without_enabling_export(page, saved_app, restore_button):
    page.goto(saved_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.model_keys?.length === 3")
    expect(page.locator('#Evaluations')).to_have_text('evaluations=1')
    page.get_by_role('button', name=restore_button, exact=True).click()
    expect(page.locator('#Evaluations')).to_have_text('evaluations=2', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.representation === 'points'")
    expect(page.locator('[id$="-Apply"] input')).not_to_be_checked()
    # The refreshed aggregates are portable: a subsequent reload needs no work.
    page.get_by_role('button', name='Save and restore', exact=True).click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Saved evaluation scores', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.representation === 'density'")
    expect(page.locator('#Evaluations')).to_have_text('evaluations=2')


@pytest.mark.ui
def test_exhausted_training_restores_winners_and_triggers_ensemble_without_resume(page, exhausted_app):
    page.goto(exhausted_app.url)
    expect(page.locator('#cand_train-Progress')).to_contain_text('Finished', timeout=30000)
    expect(page.locator('#cand_ensemble-Status')).to_contain_text('2 members', timeout=30000)
    page.wait_for_function("() => document.querySelector('#cand_ensemble-Chart .js-plotly-plot')?.data?.[0]?.x?.length === 3")
    page.locator('#cand_ensemble-FlipButton').click(force=True)
    expect(page.locator('#cand_ensemble-Table')).to_contain_text('Ensemble', timeout=10000)
    expect(page.locator('#cand_train-Progress')).to_contain_text('Finished')


@pytest.mark.ui
def test_combined_diagnostics_use_ticked_models_fill_and_reuse_evaluation(page, diagnostics_app):
    page.goto(diagnostics_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('5 members', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.pair_count === 15")
    expect_chart_to_fill_panel(page, 'Chart')
    expect(page.locator('.ensemble-chart .navbar')).to_have_count(0)
    expect(page.locator('[id$="-PairsChart"]')).to_have_count(0)
    assert plot(page).evaluate('el => el.data.filter(t => t.type === "heatmap").every(t => t.showscale === false)')
    expect(page.locator('[id$="-EvaluationCalls"]')).to_have_text('1')
    assert plot(page).evaluate('''el => {
        const residuals = el.data.filter(t => t.meta?.role === 'residual-pair');
        const limit = Math.max(...residuals.flatMap(t => [...t.x, ...t.y]).filter(Number.isFinite).map(Math.abs)) * 1.05;
        return residuals.length === 15 && residuals.every(t => {
            const axis = el._fullLayout[t.xaxis.replace('x', 'xaxis')];
            return t.meta.row < t.meta.column && axis.autorange === false
                && Math.abs(axis.range[0] + limit) < 1e-6 && Math.abs(axis.range[1] - limit) < 1e-6;
        });
    }''')
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click(force=True)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?._context?.displayModeBar === true")
    expect_chart_to_fill_panel(page, 'Chart')
    expect(page.locator('.back')).to_have_css('visibility', 'visible')
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.some(t => t.meta?.role === 'correlation' && t.showscale === true)")
    page.screenshot(path='/tmp/cand-ensemble-combined-full.png')
    page.locator('[id$="-ContractButton"]').click(force=True)
    settings(page)
    page.locator('[id$="-Models"] input[value="method_4"]').uncheck()
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.pair_count === 10")
    assert 'method_4' not in plot(page).evaluate('el => el.layout.meta.model_keys')
    expect(page.locator('[id$="-EvaluationCalls"]')).to_have_text('2')
    page.locator('[id$="-ExpandButton"]').click(force=True)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?._context?.displayModeBar === true")
    expect_chart_to_fill_panel(page, 'Chart')
    height = plot(page).evaluate('el => el._fullLayout.height')
    viewport = page.viewport_size
    page.set_viewport_size({'width': viewport['width'], 'height': viewport['height'] + 250})
    expect_chart_to_fill_panel(page, 'Chart')
    page.wait_for_function('height => document.querySelector(\'[id$="-Chart"] .js-plotly-plot\')._fullLayout.height > height + 200', arg=height)
    page.locator('[id$="-Metric"]').select_option('r2')
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.xaxis2?.title?.text?.startsWith('R²')")
    expect(page.locator('[id$="-EvaluationCalls"]')).to_have_text('2')
    page.screenshot(path='/tmp/cand-ensemble-combined.png')


@pytest.mark.ui
def test_combined_bookmark_restores_density_without_fits(page, saved_app):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.goto(saved_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.representation === 'points'")
    page.get_by_role('button', name='Save and restore', exact=True).click()
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.representation === 'density'")
    expect(page.locator('[id$="-Status"]')).to_contain_text('Saved evaluation scores')
    expect(page.locator('#Evaluations')).to_have_text('evaluations=1')
    page.locator('[id$="-Apply"] input').check()
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.representation === 'points'")
    expect(page.locator('#Evaluations')).to_have_text('evaluations=2')
    assert not errors, errors


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
def test_busy_spinner_stays_visible_through_upstream_waits_during_one_evaluation(page, busy_app):
    page.goto(busy_app.url)
    busy = page.locator('[id$="-Busy"]')
    expect(busy).to_contain_text('Evaluating member models', timeout=30000)
    busy.evaluate('''el => {
        window.busyDisappeared = 0;
        window.busyObserver = new MutationObserver(() => {
            if (!el.querySelector('.spinner-border')) window.busyDisappeared++;
        });
        window.busyObserver.observe(el, {childList:true, subtree:true});
    }''')
    page.wait_for_function("() => Number(document.querySelector('[id$=\"-Activity\"]')?.textContent.match(/waits=(\\d+)/)?.[1]) >= 30")
    expect(busy).to_contain_text('Evaluating member models')
    assert page.evaluate('window.busyDisappeared') == 0
    page.evaluate('window.busyObserver.disconnect()')
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    expect(busy).to_have_text('')
    expect(page.locator('[id$="-Activity"]')).to_contain_text('evaluations=1;')
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.layout?.meta?.model_keys?.length === 3")


@pytest.mark.ui
def test_training_progress_does_not_restart_or_hide_unchanged_ensemble(page, progress_app):
    page.goto(progress_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members', timeout=30000)
    expect(plot(page)).to_be_visible()
    plot(page).evaluate('el => { window.ensemblePlot = el; }')
    probe = page.locator('[id$="-ProgressProbe"]')
    page.wait_for_function(r'''() => {
        const text = document.querySelector('[id$="-ProgressProbe"]')?.textContent || '';
        return Number(text.match(/updates=(\d+)/)?.[1]) >= 15;
    }''')
    expect(probe).to_contain_text('fits=1')
    assert plot(page).evaluate('el => el === window.ensemblePlot')
    page.locator('[id$="-Apply"] input').check()
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2', timeout=10000)
    settings(page)
    page.locator('[id$="-ExpandButton"]').click(force=True)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.some(t => t.meta?.role === 'correlation' && t.showscale === true)")
    page.locator('[id$="-Style"]').select_option('median')
    expect(probe).to_contain_text('fits=2', timeout=10000)
    page.locator('[id$="-ContractButton"]').click(force=True)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.some(t => t.meta?.role === 'correlation' && t.showscale === false)")
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2', timeout=10000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 members')


waiting_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_waiting.py'), scope='function')


@pytest.mark.ui
def test_waits_without_fitting_until_two_models_are_available(page, waiting_app):
    page.goto(waiting_app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('Waiting for at least two', timeout=30000)
    expect(page.locator('[id$="-FitCalls"]')).to_have_text('fits=0')
    expect(page.locator('[id$="-Busy"]')).to_have_text('')
    page.get_by_role('button', name='Add second model', exact=True).click()
    expect(status).to_contain_text('2 members', timeout=30000)
    expect(page.locator('[id$="-FitCalls"]')).to_have_text('fits=1')
    settings(page)
    page.locator('[id$="-Models"] input[value="method_1"]').uncheck()
    expect(status).to_contain_text('Waiting for at least two', timeout=10000)
    expect(page.locator('[id$="-FitCalls"]')).to_have_text('fits=1')
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data?.length === 0''')


changing_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_changing.py'), scope='function')


@pytest.mark.ui
def test_changing_winners_finish_snapshots_and_keep_results_visible(page, changing_app):
    page.goto(changing_app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('Showing an earlier evaluated snapshot', timeout=30000)
    page.wait_for_function('''() => document.querySelector('[id$="-Chart"] .js-plotly-plot')?.data[0]?.x?.length === 3''')
    expect(plot(page)).to_be_visible()
    page.locator('[id$="-Apply"] input').check()
    page.wait_for_function(r'''() => Number(document.querySelector('[id$="-ProgressProbe"]')?.textContent.match(/fits=(\d+)/)?.[1]) >= 2''')
    expect(status).to_contain_text('2 members')
    expect(page.locator('[id$="-ExportProbe"]')).not_to_have_text('ensemble=2')
    page.get_by_role('button', name='Pause winner updates', exact=True).click()
    expect(status).not_to_contain_text('Showing an earlier evaluated snapshot', timeout=15000)
    expect(page.locator('[id$="-ExportProbe"]')).to_have_text('ensemble=2', timeout=15000)


review_progress_app = create_app_fixture(app=str(SCENARIOS / 'cand_ensemble_review_progress.py'), scope='function')


@pytest.mark.ui
def test_continuous_publications_through_real_candidate_review(page, review_progress_app):
    page.goto(review_progress_app.url)
    expect(page.locator('[id="cand_ensemble-Status"]')).to_contain_text('2 members', timeout=30000)
    expect(plot(page)).to_be_visible()
    page.wait_for_function("() => Number(document.querySelector('#UpdateProbe')?.textContent) >= 10")
    expect(page.locator('[id="cand_ensemble-Status"]')).to_contain_text('2 members')
    page.wait_for_function("() => document.querySelector('#cand_ensemble-Chart .js-plotly-plot')?.data?.[0]?.x?.length === 3")


@pytest.mark.ui
def test_checkbox_completes_guarded_export_during_upstream_progress(page, review_progress_app):
    page.goto(review_progress_app.url)
    expect(page.locator('#cand_ensemble-Status')).to_contain_text('2 members', timeout=30000)
    page.locator('#cand_ensemble-Apply input').check()
    expect(page.locator('#ExportProbe')).to_have_text('ensemble=2', timeout=15000)
    expect(page.locator('#cand_ensemble-PipelineBusy')).to_have_text('', timeout=10000)
    expect(page.locator('#cand_ensemble-Status')).to_contain_text('Ensemble model passed through')
    # Startup can already exceed 15 publications. Let the readiness probe catch
    # up with the completed export before measuring subsequent invalidations.
    exported_at = int(page.locator('#UpdateProbe').inner_text())
    page.wait_for_function("minimum => Number(document.querySelector('#UpdateProbe').textContent) >= minimum", arg=exported_at + 5)
    pending = page.locator('#ReadinessProbe').inner_text()
    updates = int(page.locator('#UpdateProbe').inner_text())
    page.wait_for_function("minimum => Number(document.querySelector('#UpdateProbe').textContent) >= minimum", arg=updates + 5)
    expect(page.locator('#ReadinessProbe')).to_have_text(pending)
    page.locator('#cand_ensemble-Apply input').uncheck()
    expect(page.locator('#ExportProbe')).to_have_text('incoming')
    page.locator('#cand_ensemble-Apply input').check()
    expect(page.locator('#ExportProbe')).to_have_text('ensemble=2')
    expect(page.locator('#cand_ensemble-PipelineBusy')).to_have_text('')
