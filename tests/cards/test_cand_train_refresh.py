import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/cand_train_refresh.py', scope='function')
footer_app = create_app_fixture(app='../scenarios/cand_train_footer.py', scope='function')


@pytest.mark.ui
def test_empty_ui_outputs_do_not_gain_spinner_height_across_cards(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-Progress"]')).to_contain_text('Ready', timeout=30000)
    dimensions = page.evaluate('''() => {
        const outputs = [...document.querySelectorAll('.card .shiny-html-output:empty:not(.shiny-table-output)')]
            .filter(el => el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden');
        const result = outputs.map(el => {
            const before = el.getBoundingClientRect().height;
            el.classList.add('recalculating');
            const during = el.getBoundingClientRect().height;
            el.classList.remove('recalculating');
            return {id: el.id, before, during};
        });
        const notice = document.createElement('div');
        notice.className = 'shiny-html-output';
        document.querySelector('[id$="-Progress"]').parentElement.appendChild(notice);
        notice.innerHTML = '<div role="status">A real positioning notice</div>';
        notice.classList.add('recalculating');
        result.push({id: 'real notice', height: notice.firstElementChild.getBoundingClientRect().height,
            display: getComputedStyle(notice).display, overflow: getComputedStyle(notice).overflow});
        notice.remove();
        return result;
    }''')
    assert len(dimensions) >= 3  # Includes shared outputs in both tuning and review.
    for sample in dimensions[:-1]:
        assert abs(sample['during'] - sample['before']) < 1, sample
    assert dimensions[-1]['height'] > 0, dimensions[-1]
    assert dimensions[-1]['display'] != 'none'


@pytest.mark.ui
def test_summary_and_progress_scroll_without_moving_chart_or_footer(page, footer_app):
    page.set_viewport_size({'width': 500, 'height': 900})
    page.goto(footer_app.url)
    progress = page.locator('[id$="-Progress"]')
    expect(progress).to_contain_text('Ready', timeout=30000)
    page.wait_for_function('''() => document.querySelector('[id$="-History"] .js-plotly-plot')?._fullLayout''')
    page.evaluate('document.fonts.ready')
    page.evaluate('''() => {
        window.footerPositions = [];
        window.sampleFooter = true;
        const actions = document.querySelector('.pipeline-tune-actions');
        const region = document.querySelector('.pipeline-tune-progress');
        const summary = document.querySelector('.pipeline-tune-best');
        const chart = document.querySelector('[id$="-History"]');
        function sample() {
            footerPositions.push({top: actions.getBoundingClientRect().top,
                height: region.getBoundingClientRect().height,
                summaryTop: summary.getBoundingClientRect().top,
                summaryHeight: summary.getBoundingClientRect().height,
                chartTop: chart.getBoundingClientRect().top,
                chartHeight: chart.getBoundingClientRect().height,
                notice: [...document.querySelectorAll('[id$="-PositionNotice"], [id$="-PipelineBusy"]')]
                    .map(el => el.textContent.trim()).join('')});
            if (sampleFooter) requestAnimationFrame(sample);
        }
        sample();
    }''')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('Finished', timeout=30000)
    samples = page.evaluate('''() => {sampleFooter = false; return footerPositions;}''')
    assert max(s['height'] for s in samples) - min(s['height'] for s in samples) < 1
    assert max(s['top'] for s in samples) - min(s['top'] for s in samples) < 1
    # Real notices can legitimately occupy space; empty recalculating outputs cannot.
    samples = [s for s in samples if not s['notice']]
    assert samples
    for key in ('summaryTop', 'summaryHeight', 'chartTop', 'chartHeight'):
        assert max(s[key] for s in samples) - min(s[key] for s in samples) < 1, key
    region = page.get_by_role('region', name='Tuning progress')
    # Exercise overflow as well: the short fixture status fits at the app's
    # 650px minimum card width, so constrain just this region for the wheel check.
    region.evaluate("el => el.style.width = '220px'")
    assert region.evaluate('el => el.scrollHeight > el.clientHeight')
    region.hover()
    page.mouse.wheel(0, 300)
    page.wait_for_function("() => document.querySelector('.pipeline-tune-progress').scrollTop > 0")
    summary = page.get_by_role('region', name='Best tuning result')
    summary.evaluate("el => el.style.width = '180px'")
    assert summary.evaluate('el => el.scrollHeight > el.clientHeight')
    summary.hover()
    page.mouse.wheel(0, 300)
    page.wait_for_function("() => document.querySelector('.pipeline-tune-best').scrollTop > 0")


@pytest.mark.ui
def test_shared_refresh_interval_changes_live_and_final_results_bypass_it(page, app):
    page.goto(app.url)
    progress = page.locator('[id$="-Progress"]')
    published = page.locator('#PublishedTrials')
    expect(progress).to_contain_text('Ready', timeout=30000)
    expect(published).to_have_text('0')
    page.get_by_role('button', name='Start or resume tuning', exact=True).click()
    expect(progress).to_contain_text('1 complete', timeout=30000)
    expect(published).to_have_text('0')
    history = page.locator('[id$="-History"] .js-plotly-plot')
    assert history.evaluate('el => el.data.reduce((n, t) => n + (t.x?.length || 0), 0)') == 0
    flip = page.locator('[id^="cand_train"][id$="-FlipButton"]')
    flip.click()
    table = page.locator('[id^="cand_train"][id$="-Candidates"]')
    expect(table).to_be_visible()
    expect(table.locator('tbody tr')).to_have_count(0)
    flip.click()

    def refresh_interval(seconds):
        page.evaluate('''seconds => Shiny.setInputValue(
            document.querySelector('[id$="-RefreshSeconds"]').id,
            seconds, {priority: 'event'})''', seconds)

    refresh_interval(0)
    expect(published).not_to_have_text('0', timeout=10000)
    page.wait_for_function('''() => {
        const count = Number(document.querySelector('#PublishedTrials').textContent);
        const plot = document.querySelector('[id$="-History"] .js-plotly-plot');
        return count > 0 && plot?.data?.reduce((n,t) => n + (t.x?.length || 0), 0) === count;
    }''')
    refresh_interval(30)
    expect(progress).to_contain_text('Finished', timeout=30000)
    expect(published).to_have_text('4', timeout=10000)
    page.wait_for_function('''() => document.querySelector('[id$="-History"] .js-plotly-plot')
        ?.data?.reduce((n,t) => n + (t.x?.length || 0), 0) === 4''')
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 trials')
    flip.click()
    expect(table.locator('tbody tr')).to_have_count(4)
    flip.click()

    calls = page.locator('#ReviewCalls').inner_text()
    page.get_by_role('button', name='Repeat equivalent source', exact=True).click()
    # Acknowledge the round trip after export readiness and publication.
    expect(page.locator('#SourceEvent')).to_have_text('1')
    expect(published).to_have_text('4')
    expect(page.locator('#ReviewCalls')).to_have_text(calls)

    page.get_by_role('button', name='Change source data', exact=True).click()
    expect(progress).to_contain_text('Source data or roles changed')
    expect(published).to_have_text('0')
    expect(page.locator('[id$="-Status"]')).to_contain_text('Complete pipeline tuning upstream')
