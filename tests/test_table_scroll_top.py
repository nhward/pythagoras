"""Explicit grid-window resets survive late layout and respect user scrolling."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui


@pytest.fixture
def grid(page):
    source = (Path(__file__).parents[1] / 'app/www/pythagoras.js').read_text()
    code = source.split('    const tableScrollRequests =')[1].split('    // End table scroll reset.')[0]
    page.set_content('''<div id="table"><div class="shiny-data-grid" style="height:60px;overflow:auto"><div style="height:1000px">Rows</div></div></div>
        <div id="other"><div class="shiny-data-grid" style="height:60px;overflow:auto"><div style="height:1000px">Other rows</div></div></div>''')
    page.evaluate('window.handlers = {}; window.Shiny = {addCustomMessageHandler: (name, fn) => { handlers[name] = fn; }}')
    page.add_script_tag(content='const tableScrollRequests =' + code)
    return page.locator('#table .shiny-data-grid')


def test_scroll_reset_survives_late_layout_and_is_scoped(page, grid):
    grid.evaluate('el => el.scrollTop = 700')
    page.locator('#other .shiny-data-grid').evaluate('el => el.scrollTop = 500')
    page.evaluate("handlers.table_scroll_top({id: 'table'})")
    page.wait_for_function("document.querySelector('#table .shiny-data-grid').scrollTop === 0")
    grid.evaluate('el => setTimeout(() => { el.scrollTop = 700; }, 100)')
    page.wait_for_timeout(600)
    assert grid.evaluate('el => el.scrollTop') == 0
    assert page.locator('#other .shiny-data-grid').evaluate('el => el.scrollTop') == 500


def test_user_wheel_cancels_pending_reset(page, grid):
    page.evaluate("handlers.table_scroll_top({id: 'table'})")
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
    grid.dispatch_event('wheel', {'deltaY': 100})
    grid.evaluate('el => el.scrollTop = 300')
    page.wait_for_timeout(600)
    assert grid.evaluate('el => el.scrollTop') == 300
