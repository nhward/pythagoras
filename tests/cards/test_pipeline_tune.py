import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/pipeline_tune.py', scope='function')


@pytest.mark.ui
def test_search_progress_selection_acceptance_revert_and_stale_settings(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1100})
    page.goto(app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    probe = page.locator('[id$="-ExportState"]')
    expect(probe).to_contain_text('accepted=False')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('4 complete')
    best = page.locator('[id$="-Best"]')
    expect(best).to_contain_text('Best parameters found so far')
    expect(best).to_contain_text('data_models__estimator__alpha')
    expect(best).to_contain_text('Improvement over starting configuration')
    expect(page.locator('[id$="-History"] .plotly')).to_be_attached()
    plot = page.locator('[id$="-History"] .js-plotly-plot')
    page.wait_for_function("el => el.layout?.title?.text === 'Mean score'", arg=plot.element_handle())
    assert plot.evaluate('el => el._context.displayModeBar') is False
    plot.hover()
    expect(plot.locator('.modebar')).to_have_count(0)
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?._context?.displayModeBar === true""")
    assert plot.evaluate('el => el.layout.modebar.orientation') == 'v'
    assert plot.evaluate('el => el._context.displaylogo') is False
    expect(plot.locator('.modebar')).to_be_visible()
    expect(plot.locator('.plotly-logomark')).to_have_count(0)
    page.locator('[id$="-ContractButton"]').click()
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?._context?.displayModeBar === false""")
    assert plot.evaluate("el => el.data.every(t => t.mode === 'markers' && !t.error_y?.array)")
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
    table.get_by_text('{"data_models__estimator__alpha": 0.01}', exact=True).click()
    page.get_by_role('button', name='Accept selected tuning', exact=True).click()
    expect(probe).to_contain_text('accepted=True', timeout=15000)
    expect(probe).to_contain_text('alpha=0.01')
    expect(page.locator('[id$="-Message"]')).to_contain_text('Trial 2 accepted')
    page.get_by_role('button', name='Revert acceptance', exact=True).click()
    expect(probe).to_contain_text('accepted=False')
    # A changed seed makes old results stale; play starts a fresh search.
    page.evaluate("""() => Shiny.setInputValue(document.querySelector('[id$="-Seed"]').id, 99, {priority:'event'})""")
    expect(progress).to_contain_text('stale')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('Fitting started')


@pytest.mark.ui
def test_checkpoint_download_restore_and_larger_budget_resume(page, app, tmp_path):
    page.goto(app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    def set_budget(value):
        page.evaluate("""value => Shiny.setInputValue(document.querySelector('[id$="-Trials"]').id,
            value, {priority:'event'})""", value)
    set_budget(1)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Halted', timeout=45000)
    expect(progress).to_contain_text('2 complete')
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    page.get_by_text('Search space and checkpoints', exact=True).click()
    with page.expect_download() as download:
        page.get_by_role('link', name='Save checkpoint', exact=True).click()
    checkpoint = tmp_path / 'search.json'
    download.value.save_as(checkpoint)
    page.locator('[id$="-Restore"]').set_input_files(checkpoint)
    expect(page.locator('[id$="-Message"]')).to_contain_text('Compatible checkpoint loaded')
    set_budget(0)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('4 complete')


implicit_app = create_app_fixture(app='../scenarios/pipeline_tune_implicit.py', scope='function')


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


repeated_app = create_app_fixture(app='../scenarios/pipeline_tune_repeated.py', scope='function')


@pytest.mark.ui
def test_bayesian_repeated_cv_comparison_and_display_settings(page, repeated_app):
    page.goto(repeated_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('2 repeats × 3 folds')
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    page.get_by_text('Bayesian practical-equivalence comparisons', exact=True).click()
    table = page.locator('[id$="-Comparisons"]')
    expect(table).to_contain_text('P(equivalent)', timeout=30000)
    expect(table).to_contain_text('Repeat advantage SD')
    note = page.locator('[id$="-ComparisonNote"]')
    expect(note).to_contain_text('All 6 pairs')
    expect(note).to_contain_text('not independent datasets')
    page.evaluate("""() => Shiny.setInputValue(document.querySelector('[id$="-ComparisonLimit"]').id, 2, {priority:'event'})""")
    expect(note).to_contain_text('All 1 pairs')
    expect(progress).to_contain_text('Finished')
    expect(page.locator('[id$="-ExportState"]')).to_contain_text('accepted=False')
    page.evaluate("""() => Shiny.setInputValue(document.querySelector('[id$="-EquivalenceTolerance"]').id, 0, {priority:'event'})""")
    expect(note).to_contain_text('positive practical-equivalence tolerance')
    expect(progress).to_contain_text('Finished')


multiple_app = create_app_fixture(app='../scenarios/pipeline_tune_multiple.py', scope='function')


@pytest.mark.ui
def test_multiple_methods_can_be_tuned_and_different_method_accepted(page, multiple_app):
    page.goto(multiple_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('2 starting-configuration evaluations', timeout=30000)
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=45000)
    expect(progress).to_contain_text('5 complete')
    plot = page.locator('[id$="-History"] .js-plotly-plot')
    page.wait_for_function("""() => document.querySelector('[id$="-History"] .js-plotly-plot')?.data?.length === 2""")
    assert len(set(plot.evaluate('el => el.data.map(t => t.marker.color)'))) == 2
    assert plot.evaluate("el => el.data.every(t => t.mode === 'markers' && !t.error_y?.array)")
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    table = page.locator('[id$="-Candidates"]')
    expect(table).to_contain_text('Method', timeout=30000)
    expect(table).to_contain_text('Ridge Regression')
    expect(table).to_contain_text('Decision Tree Regression')
    table.get_by_text('{"data_models__estimator__max_depth": 3}', exact=True).click()
    page.get_by_role('button', name='Accept selected tuning', exact=True).click()
    expect(page.locator('[id$="-MultiState"]')).to_have_text('DecisionTreeRegressor;depth=3', timeout=30000)
    page.get_by_role('button', name='Revert acceptance', exact=True).click()
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
