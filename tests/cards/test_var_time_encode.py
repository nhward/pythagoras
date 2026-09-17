import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
app=create_app_fixture(app='../scenarios/var_time_encode.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_time_encode_restored.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1400}}
def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')

def test_preview_enable_audit_retention_and_undo(page,app):
    page.goto(app.url)
    expect(by_id(page,'TimeTable')).to_contain_text('flag__month',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True')
    toggle=by_id(page,'Encode').locator('input[value="time"]')
    toggle.check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.flip-btn').click()
    expect(by_id(page,'FeatureTable')).to_contain_text('cyclic',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.flip-btn').click()
    page.locator('.card').first.hover();page.locator('button.collapse-toggle').click()
    by_id(page,'RemoveOriginal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)
    toggle.uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True',timeout=30000)

def test_restored_feature_selection(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Encode').locator('input[value="time"]')).to_be_checked()
    expect(by_id(page,'Status')).to_contain_text('2 Predictor features',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False')
