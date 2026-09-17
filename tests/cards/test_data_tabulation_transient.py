import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
app=create_app_fixture(app='../scenarios/data_tabulation_transient.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1600,'height':1100}}

def test_stable_table_containers_through_none_and_pending_sources(page,app):
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.goto(app.url)
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    def by_id(name):return page.locator(f'#{namespace}-{name}')
    expect(by_id('DataTable2')).to_contain_text('value',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.flip-btn').click()
    expect(by_id('Structure')).to_contain_text('value',timeout=30000)
    page.evaluate('''ns => {
        window.originalDataGrid=document.getElementById(ns+'-DataTable2');
        window.originalStructureGrid=document.getElementById(ns+'-Structure');
    }''',namespace)
    for _ in range(3):
        by_id('SourceState').select_option('missing')
        expect(by_id('SourceProbe')).to_have_text('missing')
        by_id('SourceState').select_option('pending')
        expect(by_id('Structure')).to_contain_text('new_column',timeout=30000)
        page.locator('.card').first.hover();page.locator('button.flip-btn').click()
        expect(by_id('DataTable2')).to_contain_text('101',timeout=30000)
        by_id('SourceState').select_option('ready')
        expect(by_id('DataTable2')).not_to_contain_text('new_column',timeout=30000)
        page.locator('.card').first.hover();page.locator('button.flip-btn').click()
        expect(by_id('Structure')).not_to_contain_text('new_column',timeout=30000)
    assert page.evaluate('''ns => window.originalDataGrid===document.getElementById(ns+'-DataTable2')
        && window.originalStructureGrid===document.getElementById(ns+'-Structure')''',namespace)
    assert not errors,errors
