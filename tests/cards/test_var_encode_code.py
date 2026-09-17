"""Code panel browser workflows and independent encoding toggles."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/var_encode_code.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_encode_code_restored.py',scope='function')
no_target_app=create_app_fixture(app='../scenarios/var_encode.py',scope='function')
pytestmark=pytest.mark.ui

@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1200}}


def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')


def toggle(page,value):return by_id(page,'Encode').locator(f'input[value="{value}"]')


def test_code_and_nominal_toggle_independently_and_settings_rebuild(page,app):
    page.goto(app.url)

    page.get_by_role('tab',name='Code',exact=True).click()
    expect(by_id(page,'CodeMessage')).to_contain_text("Code: Unseen codes use the 'target' mean or class proportions.")
    expect(by_id(page,'CodeTable')).to_contain_text('Cardinality')
    toggle(page,'code').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; code_original=False; nominal_original=True',timeout=30000)
    toggle(page,'nominal').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=2; code_original=False; nominal_original=False',timeout=30000)
    page.locator('.card').first.hover();page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'CodeMethod').select_option('smooth')
    expect(by_id(page,'CodeTable')).to_contain_text('fixed strength',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=2; code_original=False')
    by_id(page,'RemoveOriginal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=2; code_original=True; nominal_original=True',timeout=30000)
    toggle(page,'nominal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=1;',timeout=30000)
    toggle(page,'code').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0;',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('unchanged=True')


def test_restoration_and_target_changes_preserve_nominal_encoding(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'CodeTable')).to_be_visible()
    expect(by_id(page,'Probe')).to_contain_text('steps=2; code_original=False; nominal_original=False',timeout=30000)
    expect(toggle(page,'code')).to_be_checked()
    expect(by_id(page,'Status')).to_contain_text('Code: 3 new predictors')
    by_id(page,'NoTarget').click()
    expect(by_id(page,'CodeMessage')).to_contain_text('exactly one Target',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; code_original=True; nominal_original=False',timeout=30000)
    by_id(page,'RestoreTarget').click()
    expect(by_id(page,'Probe')).to_contain_text('steps=2; code_original=False; nominal_original=False',timeout=30000)


def test_missing_target_message_and_toggle_are_benign(page,no_target_app):
    page.goto(no_target_app.url)
    page.get_by_role('tab',name='Code',exact=True).click()
    expect(by_id(page,'CodeMessage')).to_contain_text('Assign a Target upstream',timeout=30000)
    toggle(page,'code').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True; unchanged=True',timeout=30000)
    expect(by_id(page,'Status')).to_contain_text('Code encoding unavailable',timeout=30000)
    toggle(page,'nominal').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
