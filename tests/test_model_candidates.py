import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='scenarios/model_candidates.py', scope='function')


@pytest.mark.ui
def test_repeated_chart_updates_reuse_one_widget(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 methods', timeout=45000)
    page.wait_for_function('''() => document.querySelector('[id$="-ReviewPlot"] .js-plotly-plot')?.layout?.title?.text''')
    page.evaluate('''() => {
        window.widgetLifecycle = {opened: [], closed: []};
        jQuery(document).on('shiny:message.widgetLifecycleTest', event => {
            for (const [type, payload] of Object.entries(event.message?.custom || {})) {
                if (!['shinywidgets_comm_open', 'shinywidgets_comm_close'].includes(type)) continue;
                const message = typeof payload === 'string' ? JSON.parse(payload) : payload;
                widgetLifecycle[type.endsWith('_open') ? 'opened' : 'closed'].push(message.content.comm_id);
            }
        });
    }''')
    for _ in range(4):
        for view, title in [('tradeoff', 'Method representatives'), ('errors', 'Error exceedance'),
                            ('within', 'Configurations within a method')]:
            page.evaluate('''view => Shiny.setInputValue(document.querySelector('[id$="-View"]').id,
                view, {priority: 'event'})''', view)
            page.wait_for_function('''title => document.querySelector('[id$="-ReviewPlot"] .js-plotly-plot')
                ?.layout?.title?.text?.includes(title)''', arg=title)
    assert page.evaluate('widgetLifecycle.opened.length') == 0
    assert page.evaluate('widgetLifecycle.closed.length') == 0
    expect(page.locator('[id$="-ReviewPlot"] .js-plotly-plot')).to_be_visible()


@pytest.mark.ui
def test_review_views_representative_acceptance_and_revert(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1100})
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 methods', timeout=45000)
    details = page.locator('details.model-candidates-differences')
    expect(details).not_to_have_attribute('open', '')
    plot = page.locator('[id$="-ReviewPlot"] .js-plotly-plot')
    expect(plot).to_be_attached(timeout=30000)
    point = plot.locator('.scatterlayer .point').first
    expect(point).to_be_visible()
    box = point.bounding_box()
    page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
    expect(details).to_have_attribute('open', '')
    page.get_by_role('button', name='Accept selected model', exact=True).click()
    expect(page.locator('[id$="-ReviewState"]')).to_contain_text('Ridge;evidence=5')
    page.get_by_role('button', name='Revert acceptance', exact=True).click()
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    table = page.locator('[id$="-Candidates"]')
    expect(table).to_contain_text('P95 absolute error', timeout=30000)
    expect(table).to_contain_text('Observed rows/s')
    expect(table).to_contain_text('Practically equivalent')
    reference = table.locator('tbody tr').filter(has_text='Best score (reference)').first
    expect(reference.locator('td.model-candidates-reference').first).to_be_visible()
    assert reference.locator('td').first.evaluate('el => getComputedStyle(el).fontWeight') == '600'
    download = page.get_by_role('link', name='Download', exact=True)
    expect(download).to_be_visible()
    assert download.evaluate('el => el.getBoundingClientRect().width') == 220
    expect(download.locator('svg')).to_have_count(1)
    table.get_by_text('Decision Tree Regression', exact=True).first.click()
    expect(details).to_have_attribute('open', '')
    differences = page.locator('[id$="-Differences"]')
    expect(differences).to_contain_text('Decision Tree: estimator-max-depth')
    parameter = differences.locator('tbody tr td').first
    assert parameter.evaluate('''el => {
        const style = getComputedStyle(el);
        return el.getBoundingClientRect().width >= 280
            && style.whiteSpace === 'normal' && style.textOverflow !== 'ellipsis';
    }''')
    expect(page.locator('[id$="-Note"]')).not_to_contain_text('Tuning comparisons are exploratory')
    tree = table.get_by_text('Decision Tree Regression', exact=True).first
    tree.click(modifiers=['ControlOrMeta'])
    expect(details).not_to_have_attribute('open', '')
    tree.click()
    expect(details).to_have_attribute('open', '')
    page.get_by_role('button', name='Use selected as representative', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('representative for Decision Tree')
    page.get_by_role('button', name='Accept selected model', exact=True).click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('accepted.')
    expect(page.locator('[id$="-ReviewState"]')).to_contain_text('DecisionTreeRegressor;evidence=5')
    page.get_by_role('button', name='Revert acceptance', exact=True).click()
    expect(page.locator('[id$="-ReviewState"]')).to_have_text('No review accepted')
    page.locator('[id$="-FlipButton"]').click()
    page.evaluate('''() => Shiny.setInputValue(document.querySelector('[id$="-View"]').id, 'tradeoff', {priority:'event'})''')
    page.wait_for_function('''() => document.querySelector('[id$="-ReviewPlot"] .js-plotly-plot')?.layout?.title?.text?.includes('Method representatives')''')
    assert plot.evaluate('el => el.data.length') == 2
    page.evaluate('''() => Shiny.setInputValue(document.querySelector('[id$="-View"]').id, 'errors', {priority:'event'})''')
    page.wait_for_function('''() => document.querySelector('[id$="-ReviewPlot"] .js-plotly-plot')?.layout?.title?.text?.includes('Error exceedance')''')
    assert plot.evaluate('el => el.data.length') == 2
