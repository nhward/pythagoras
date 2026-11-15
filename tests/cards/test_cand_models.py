import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/cand_models.py', scope='function')


def set_weights(page, a, b):
    page.locator('[id$="-Weights"]').evaluate('''(el, values) => {
        const slider = jQuery(el).data('ionRangeSlider');
        slider.update({from: values[0], to: values[1]});
        slider.options.onFinish(slider.result);
    }''', [a, b])


@pytest.mark.ui
def test_pareto_tabs_weights_export_inspection_and_widget_reuse(page, app):
    page.set_viewport_size({'width': 1300, 'height': 1050})
    page.goto(app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('2 methods', timeout=45000)
    state = page.locator('[id$="-SelectionState"]')
    expect(state).to_contain_text("records=['cand_models']", timeout=30000)
    plot = page.locator('[id$="-ReviewPlot"] .js-plotly-plot')
    page.wait_for_function('''() => document.querySelector('[id$="-ReviewPlot"] .js-plotly-plot')?.data?.some(t => t.type === 'scatter3d')''')
    assert plot.evaluate("el => el.data.every(t => t.type === 'scatter3d')")
    page.wait_for_function("() => !document.documentElement.classList.contains('shiny-busy') && !document.querySelector('[id$=\"-ReviewPlot\"]').classList.contains('recalculating')")
    page.screenshot(path='/tmp/cand_models.png')
    host = page.locator('[id$="-ReviewPlot"]')
    assert abs(plot.bounding_box()['width'] - host.bounding_box()['width']) < 5
    assert abs(plot.bounding_box()['height'] - host.bounding_box()['height']) < 5
    assert plot.locator('canvas').count() > 0
    assert page.locator('.back').evaluate("el => getComputedStyle(el).visibility") == 'hidden'
    default_selection = state.inner_text()
    # Exercise actual pointer dragging from overlapping 100/100 handles.
    track = page.locator('.cand-models-weights .irs-line').bounding_box()
    handle = page.locator('.cand-models-weights .irs-handle.from').bounding_box()
    page.mouse.move(handle['x'] + 2, handle['y'] + handle['height'] / 2)
    page.mouse.down()
    page.mouse.move(track['x'] + track['width'] * .4, handle['y'] + handle['height'] / 2, steps=8)
    expect(state).to_have_text(default_selection)
    page.mouse.up()
    expect(page.locator('[id$="-WeightSummary"]')).not_to_contain_text('Score 100%')
    set_weights(page, 100, 100)
    expect(state).to_have_text(default_selection)
    page.get_by_role('tab', name='Decision Tree Regression', exact=True).click()
    expect(status).to_contain_text('1 methods')
    expect(state).to_have_text(default_selection)
    page.get_by_role('tab', name='Overall', exact=True).click()
    expect(status).to_contain_text('2 methods')
    page.evaluate('''() => {
        window.newWidgets = [];
        jQuery(document).on('shiny:message.candtest', (event) => {
            for (const message of Object.values(event.message?.custom || {})) {
                if (message?.content?.data?.method === 'open') window.newWidgets.push(message);
            }
        });
    }''')
    plot.evaluate("el => Plotly.relayout(el, {'scene.camera.eye': {x: 2.3, y: 1.7, z: .9}})")
    set_weights(page, 0, 100)
    expect(page.locator('[id$="-WeightSummary"]')).to_contain_text('Throughput 100%')
    expect(state).to_contain_text('overall=4')
    assert plot.evaluate('el => el.layout.scene.camera.eye.x') == 2.3
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(page.locator('.card.fullscreen-active')).to_be_visible()
    page.wait_for_function("() => document.querySelector('[id$=\"-ReviewPlot\"] .js-plotly-plot').getBoundingClientRect().height > 500")
    page.wait_for_function("() => { const p = document.querySelector('[id$=\"-ReviewPlot\"] .js-plotly-plot'); return Math.abs(p._fullLayout.height - p.getBoundingClientRect().height) < 5; }")
    page.screenshot(path='/tmp/cand_models_fullscreen.png')
    page.locator('[id$="-ContractButton"]').click()
    fastest_selection = state.inner_text()
    # Tab browsing never changes the exported winner or method winners.
    page.get_by_role('tab', name='Ridge Regression', exact=True).click()
    expect(state).to_have_text(fastest_selection)
    page.get_by_role('tab', name='Overall', exact=True).click()
    expect(page.locator('[id$="-EligibleMethods"]')).to_have_count(0)
    assert page.evaluate('window.newWidgets.length') == 0
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    table = page.locator('[id$="-Candidates"]')
    expect(table).to_contain_text('Method winner')
    table.locator('tbody tr').first.click()
    expect(page.locator('[id$="-ComparisonDetails"]')).to_have_attribute('open', '')
    expect(state).to_contain_text('overall=4')
    expect(page.get_by_role('button', name='Accept selected model', exact=True)).to_have_count(0)
    download = page.locator('[id$="-ResultsCSV"]')
    expect(download).to_have_text('Download')
    assert abs(download.bounding_box()['width'] - 220) < 2


restored_app = create_app_fixture(app='../scenarios/cand_models_restored.py', scope='function')


@pytest.mark.ui
def test_restore_ignores_retired_method_filter(page, restored_app):
    page.goto(restored_app.url)
    expect(page.locator('[id$="-WeightSummary"]')).to_contain_text('Throughput 100%', timeout=45000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 methods')
    state = page.locator('[id$="-SelectionState"]')
    expect(state).not_to_contain_text('999')
    expect(state).to_contain_text("'tree':")
    expect(state).to_contain_text("'ridge':")
    expect(page.locator('[id$="-EligibleMethods"]')).to_have_count(0)
    expect(state).to_contain_text("records=['cand_models']")
    assert page.locator('[id$="-Weights"]').evaluate("el => jQuery(el).data('ionRangeSlider').result.from") == 0


historical_app = create_app_fixture(app='../scenarios/cand_models_historical.py', scope='function')


@pytest.mark.ui
def test_historical_evidence_is_reviewed_without_selecting_a_pipeline(page, historical_app):
    page.goto(historical_app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Historical review (read-only)', timeout=45000)
    expect(page.locator('[id$="-SelectionState"]')).to_contain_text('overall=None')
    expect(page.locator('[id$="-SelectionState"]')).to_contain_text('records=[]')
