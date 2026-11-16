"""Expanded navbar controls share a row without changing collapse semantics."""
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/navbar_inline.py', scope='function')


@pytest.mark.ui
@pytest.mark.parametrize('identifier', ['Forced', 'Responsive', 'Unclassed'])
def test_expanded_toggle_shares_tabs_row_and_wraps_without_overflow(page, app, identifier):
    page.set_viewport_size({'width': 600, 'height': 1100})
    page.goto(app.url)
    host = page.locator('#' + identifier + '-host')
    navbar = host.locator('.navbar')
    toggle = navbar.get_by_role('button', name='Toggle navigation')
    first = navbar.get_by_role('tab', name='First', exact=True)
    expect(first).to_be_hidden()
    toggle.click()
    expect(first).to_be_visible()
    page.wait_for_function('''id => {
        const host = document.getElementById(id + '-host');
        const button = host.querySelector('.navbar-toggler').getBoundingClientRect();
        const first = host.querySelector('[role=tab]').getBoundingClientRect();
        const collapse = host.querySelector('.navbar-collapse');
        return !collapse.classList.contains('collapsing') && first.left >= button.right
            && Math.abs(first.top - button.top) < 5;
    }''', arg=identifier)
    assert navbar.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    last = navbar.get_by_role('tab').last
    last.click()
    expect(last).to_have_attribute('aria-selected', 'true')
    assert last.bounding_box()['y'] > first.bounding_box()['y']
    # Close/open from the keyboard, preserving the selected tab.
    toggle.focus()
    page.keyboard.press('Enter')
    expect(first).to_be_hidden()
    expect(toggle).to_have_attribute('aria-expanded', 'false')
    page.keyboard.press('Space')
    expect(first).to_be_visible()
    expect(last).to_have_attribute('aria-selected', 'true')
    # A phone-width container must wrap even an unbroken tab label.
    host.evaluate("el => el.style.width = '180px'")
    assert navbar.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    assert navbar.get_by_role('tab').first.bounding_box()['x'] >= toggle.bounding_box()['x'] + toggle.bounding_box()['width']
    if identifier != 'Forced':
        page.set_viewport_size({'width': 1000, 'height': 1100})
        expect(toggle).to_be_hidden()
        expect(first).to_be_visible()
