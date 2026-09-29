import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/data_tabulation_transient.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1600,'height':1100}}

def test_single_tables_recover_through_moves_none_and_pending_sources(page,app):
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.goto(app.url)
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    def by_id(name):return page.locator(f'#{namespace}-{name}')
    expect(by_id('DataTable2')).to_contain_text('value',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.flip-btn').click()
    expect(by_id('Structure')).to_contain_text('value',timeout=30000)
    def expect_single_table(name):
        # Container replacement is allowed; duplicate outputs or internal
        # rendering roots are not. Content checks below establish recovery.
        expect(by_id(name)).to_have_count(1)
        expect(by_id(name).locator(':scope > .html-fill-container')).to_have_count(1)
        expect(by_id(name).locator('table')).to_have_count(1)

    expect_single_table('DataTable2')
    expect_single_table('Structure')
    for _ in range(3):
        # insertBefore/appendChild moves, as used by Sortable, disconnect and
        # reconnect custom elements even though their identity is unchanged.
        page.locator('.card').first.evaluate('card => card.parentElement.appendChild(card)')
        by_id('SourceState').select_option('missing')
        expect(by_id('SourceProbe')).to_have_text('missing')
        by_id('SourceState').select_option('pending')
        expect(by_id('Structure')).to_contain_text('new_column',timeout=30000)
        expect_single_table('Structure')
        page.locator('.card').first.hover();page.locator('button.flip-btn').click()
        expect(by_id('DataTable2')).to_contain_text('101',timeout=30000)
        expect_single_table('DataTable2')
        by_id('SourceState').select_option('ready')
        expect(by_id('DataTable2')).to_contain_text('value',timeout=30000)
        expect(by_id('DataTable2')).not_to_contain_text('101',timeout=30000)
        expect(by_id('DataTable2')).not_to_contain_text('new_column',timeout=30000)
        page.locator('.card').first.hover();page.locator('button.flip-btn').click()
        expect(by_id('Structure')).to_contain_text('value',timeout=30000)
        expect(by_id('Structure')).not_to_contain_text('new_column',timeout=30000)
        expect_single_table('DataTable2')
        expect_single_table('Structure')
    assert not errors,errors
