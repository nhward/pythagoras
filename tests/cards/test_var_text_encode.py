import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/var_text_encode.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_text_encode_restored.py',scope='function')
empty_app=create_app_fixture(app='../scenarios/var_text_encode_empty.py',scope='function')
pytestmark=pytest.mark.ui
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1400}}
def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')
def toggle(page,method):return by_id(page,'Encode').locator(f'input[value="{method}"]')

def test_panels_independent_toggles_audit_and_undo(page,app):
    page.goto(app.url)
    expect(by_id(page,'Summary_bow')).to_contain_text('flag__bow__',timeout=30000)
    expect(by_id(page,'Detail_bow')).to_contain_text('Document frequency')
    toggle(page,'bow').check()
    toggle(page,'sentiment').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    page.get_by_role('tab',name='Sentiment',exact=True).click()
    expect(by_id(page,'Detail_sentiment')).to_contain_text('compound',timeout=30000)
    page.get_by_role('tab',name='Latent semantics',exact=True).click()
    expect(by_id(page,'Detail_lsa')).to_contain_text('Explained variance',timeout=30000)
    page.get_by_role('tab',name='Text characteristics',exact=True).click()
    expect(by_id(page,'Detail_characteristics')).to_contain_text('characters',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.flip-btn').click()
    expect(by_id(page,'AuditTable')).to_contain_text('Missing text indicator',timeout=30000)
    toggle(page,'bow').uncheck();toggle(page,'sentiment').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True',timeout=30000)


def test_embedding_upload_and_unavailable_method_retains_source(page,app,tmp_path):
    page.goto(app.url)
    page.get_by_role('tab',name='Word embeddings',exact=True).click()
    expect(by_id(page,'Message_embedding')).to_contain_text('Load a pretrained',timeout=30000)
    toggle(page,'embedding').check();toggle(page,'sentiment').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)
    page.locator('.card').first.hover();page.locator('button.collapse-toggle').click()
    p=tmp_path/'test.vec';p.write_text('3 2\ngood 1 0\nbad 0 1\nproduct 1 1\n')
    by_id(page,'EmbeddingFile').set_input_files(str(p))
    expect(by_id(page,'Summary_embedding')).to_contain_text('test.vec',timeout=30000)
    expect(by_id(page,'Detail_embedding')).to_contain_text('coverage',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    by_id(page,'RemoveOriginal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)


def test_restored_combination_tab_and_retention(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Summary_sentiment')).to_be_visible()
    expect(toggle(page,'sentiment')).to_be_checked()
    expect(toggle(page,'characteristics')).to_be_checked()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)


def test_no_text_predictors_benign(page,empty_app):
    page.goto(empty_app.url)
    expect(by_id(page,'Message_bow')).to_contain_text('No Text predictors are available',timeout=30000)
    toggle(page,'bow').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True')
