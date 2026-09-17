"""Exercise production two-second delays (the normal pytest bypass is disabled)."""
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
variable_app=create_app_fixture(app='../scenarios/var_encode_settling.py',scope='function')
text_app=create_app_fixture(app='../scenarios/var_text_encode_settling.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1400}}

def check_settling(page,app,first,second,expected):
    page.goto(app.url)
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    def by_id(name):return page.locator(f'#{namespace}-{name}')
    expect(by_id('Probe')).to_contain_text('steps=0',timeout=30000)
    # Record actual client deliveries, including transient intermediate outputs.
    page.evaluate('''ns => {
        window.probeUpdates=[];
        new MutationObserver(() => window.probeUpdates.push(document.getElementById(ns+'-Probe').textContent))
            .observe(document.getElementById(ns+'-Probe'),{childList:true,subtree:true,characterData:true});
    }''',namespace)
    group=by_id('Encode')
    group.locator(f'input[value="{first}"]').check()
    page.wait_for_timeout(1100)
    group.locator(f'input[value="{second}"]').check()
    page.wait_for_timeout(1100)
    expect(by_id('Probe')).to_contain_text('steps=0')
    assert all('steps=0' in value for value in page.evaluate('window.probeUpdates'))
    expect(by_id('Probe')).to_contain_text(expected,timeout=30000)
    # Removing every selection must also wait rather than bypassing the debounce.
    group.locator(f'input[value="{first}"]').uncheck()
    group.locator(f'input[value="{second}"]').uncheck()
    page.wait_for_timeout(600)
    expect(by_id('Probe')).to_contain_text(expected)
    expect(by_id('Probe')).to_contain_text('steps=0',timeout=30000)


def test_variable_encoding_settles_combined_selection(page,variable_app):
    check_settling(page,variable_app,'nominal','code','steps=2')


def test_text_encoding_settles_combined_selection(page,text_app):
    check_settling(page,text_app,'sentiment','characteristics','steps=1; original=False')
