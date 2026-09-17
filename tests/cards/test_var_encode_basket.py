import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/var_encode_basket.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_encode_basket_restored.py',scope='function')
empty_app=create_app_fixture(app='../scenarios/var_encode_empty.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1400}}
def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')
def toggle(page):return by_id(page,'Encode').locator('input[value="basket"]')

def test_periods_and_reversible_encoding_without_target(page,app):
    page.goto(app.url)
    page.get_by_role('tab',name='Basket',exact=True).click()
    table=by_id(page,'BasketTable')
    expect(table).to_contain_text('Distinct items',timeout=30000)
    expect(table).to_contain_text('flag__apple')
    toggle(page).check()
    expect(by_id(page,'Status')).to_contain_text('Basket: 2 new predictors',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False')
    page.locator('.card').first.hover();page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'RemoveOriginal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)
    toggle(page).uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True',timeout=30000)


def test_restore_basket_tab_and_enabled_encoding(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'BasketTable')).to_be_visible()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    expect(toggle(page)).to_be_checked()


def test_no_basket_predictors_is_benign(page,empty_app):
    page.goto(empty_app.url)
    page.get_by_role('tab',name='Basket',exact=True).click()
    expect(by_id(page,'BasketMessage')).to_contain_text('No Basket predictors are available',timeout=30000)
    toggle(page).check()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=False; unchanged=True')
