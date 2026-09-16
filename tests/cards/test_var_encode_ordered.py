"""Ordered panel browser coverage, method changes and restoration."""
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
app=create_app_fixture(app='../scenarios/var_encode_ordered.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_encode_ordered_restored.py',scope='function')
empty_app=create_app_fixture(app='../scenarios/var_encode_empty.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1400}}
def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')
def toggle(page,name):return by_id(page,'Encode').locator(f'input[value="{name}"]')

def test_ordered_method_changes_and_three_independent_toggles(page,app):
    page.goto(app.url)
    page.get_by_role('tab',name='Ordered',exact=True).click()
    expect(by_id(page,'OrderedTable')).to_contain_text('low → medium → high → very high',timeout=30000)
    toggle(page,'ordered').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; ordered_original=False',timeout=30000)
    expect(by_id(page,'Status')).to_contain_text('Ordered encoding enabled: 1 Predictor features')
    page.locator('.card').first.hover();page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'OrderedMethod').select_option('polynomial')
    expect(by_id(page,'Status')).to_contain_text('Ordered encoding enabled: 3 Predictor features',timeout=30000)
    expect(by_id(page,'OrderedTable')).to_contain_text('ordered__L, ordered__Q, ordered__C')
    toggle(page,'nominal').check();toggle(page,'code').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=3;',timeout=30000)
    toggle(page,'ordered').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=2; ordered_original=True',timeout=30000)
    toggle(page,'nominal').uncheck();toggle(page,'code').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; ordered_original=True; unchanged=True',timeout=30000)


def test_restored_polynomial_degree_and_original_retention(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'OrderedTable')).to_be_visible()
    expect(by_id(page,'Status')).to_contain_text('Ordered encoding enabled: 2 Predictor features',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; ordered_original=True')
    expect(toggle(page,'ordered')).to_be_checked()


def test_no_ordered_predictors_is_benign(page,empty_app):
    page.goto(empty_app.url)
    page.get_by_role('tab',name='Ordered',exact=True).click()
    expect(by_id(page,'OrderedMessage')).to_contain_text('No Ordered Predictor',timeout=30000)
    toggle(page,'ordered').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=False; unchanged=True')
