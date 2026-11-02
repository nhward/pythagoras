"""Real-browser commitment, independent method tabs, flip evidence and reset."""
import os
import sys
from pathlib import Path
APP_ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(APP_ROOT)
sys.path.insert(0,str(APP_ROOT))
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/var_select.py',scope='function')
stability_app=create_app_fixture(app='../scenarios/var_select_stability.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_select_restored.py',scope='function')


def by_id(page,name):
    return page.locator(f'[id$="-{name}"]')


@pytest.mark.ui
def test_radio_selection_flip_and_automatic_source_refresh(page,app):
    page.goto(app.url)
    expect(by_id(page,'Status')).to_contain_text('selected;',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=none')
    expect(by_id(page,'Apply')).to_have_count(0)
    assert page.get_by_label('Boruta',exact=True).get_attribute('type')=='radio'
    page.get_by_label('Boruta',exact=True).check()
    expect(page.get_by_label('None',exact=True)).not_to_be_checked()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select',timeout=30000)
    expect(by_id(page,'ExportProbe')).not_to_contain_text('columns=signal,noise,category,empty')
    page.get_by_role('tab',name='GrootCV',exact=True).click()
    expect(by_id(page,'Chart_GrootCV')).to_contain_text('TreeSHAP',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select')
    page.locator('.card').first.hover()
    by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'EvidenceTitle')).to_contain_text('GrootCV')
    expect(by_id(page,'Evidence')).to_contain_text('Variable',timeout=30000)

    grid=by_id(page,'Evidence').locator('.shiny-data-grid')
    expect(grid).to_be_visible()
    assert grid.evaluate('(el) => el.scrollHeight > el.clientHeight && el.clientHeight > 0')
    grid.hover()
    page.mouse.wheel(0,600)
    page.wait_for_function('id => document.getElementById(id).querySelector(".shiny-data-grid").scrollTop > 0',
                           arg=by_id(page,'Evidence').get_attribute('id'))
    page.get_by_label('None',exact=True).check()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=none')
    page.get_by_label('Boruta',exact=True).check()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select')
    by_id(page,'ChangeSource').click()
    expect(by_id(page,'ExportProbe')).to_contain_text('name=Updated source',timeout=60000)
    expect(by_id(page,'Status')).to_contain_text('Applied automatically',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select',timeout=30000)


@pytest.mark.ui
def test_restored_commitment_recomputes_before_export(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Status')).to_contain_text('selected;',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select',timeout=30000)
    expect(page.get_by_label('Boruta',exact=True)).to_be_checked()
    page.get_by_role('tab',name='GrootCV',exact=True).click()
    expect(by_id(page,'Chart_GrootCV')).to_contain_text('TreeSHAP',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select')


@pytest.mark.ui
def test_settings_recompute_and_empty_recommendation_blocks_export(page,app):
    page.goto(app.url)
    page.get_by_label('Boruta',exact=True).check()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select',timeout=60000)
    by_id(page,'FewIterations').click()
    expect(by_id(page,'Status')).to_contain_text('No variables were selected',timeout=60000)
    expect(by_id(page,'Status')).to_contain_text('Downstream export paused')
    expect(by_id(page,'ExportProbe')).to_have_text('waiting')
    page.get_by_label('None',exact=True).check()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=none')
    page.get_by_label('GrootCV',exact=True).check()
    expect(by_id(page,'Status')).to_contain_text('GrootCV:',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=var_select',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_contain_text('selected=GrootCV; iterations=100')
    by_id(page,'InvalidWeights').click()
    expect(by_id(page,'Status')).to_contain_text('nonmissing',timeout=60000)
    expect(by_id(page,'ExportProbe')).to_have_text('waiting')
    page.get_by_label('None',exact=True).check()
    expect(by_id(page,'ExportProbe')).to_contain_text('pipeline=none')


@pytest.mark.unit
def test_checkbox_bookmarks_migrate_to_scalar_radio_without_saved_masks():
    from copy import deepcopy
    from cards.var_select import recommendation_configuration
    state={'inputs':{'Recommendation':['Boruta'],'Trees':40},
           'selection':{'method':'Boruta','identity':'old'},'unrelated':{'keep':True}}
    original=deepcopy(state)
    restored=recommendation_configuration(state)
    assert restored=={'inputs':{'Recommendation':'Boruta','Trees':40},'unrelated':{'keep':True}}
    assert state==original
    assert recommendation_configuration({'inputs':{'Recommendation':'GrootCV'}})['inputs']['Recommendation']=='GrootCV'
    assert recommendation_configuration({'inputs':{'Recommendation':[]}})['inputs']['Recommendation']=='None'
    assert recommendation_configuration({'inputs':{},'selection':{'method':'Boruta'}})['inputs']['Recommendation']=='Boruta'


@pytest.mark.ui
def test_stability_is_optional_and_never_changes_exported_recommendation(page,stability_app):
    page.goto(stability_app.url)
    expect(by_id(page,'ExportProbe')).to_contain_text('selected=Boruta',timeout=60000)
    expect(by_id(page,'Chart_GrootCV')).to_contain_text('Selection frequency',timeout=60000)
    before=by_id(page,'ExportProbe').inner_text()
    page.locator('.card').first.hover()
    by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'StabilityStatus')).to_contain_text('3 / 3 training folds succeeded',timeout=30000)
    expect(by_id(page,'Evidence')).to_contain_text('Eligible folds',timeout=30000)
    page.get_by_text('Fold-by-fold stability',exact=True).click()
    expect(by_id(page,'StabilityDetails')).to_contain_text('CV1',timeout=30000)
    expect(by_id(page,'ExportProbe')).to_have_text(before)
    by_id(page,'DisableStability').click()
    expect(by_id(page,'ExportProbe')).to_have_text(before)
    by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'Chart_GrootCV')).to_contain_text('TreeSHAP',timeout=30000)


@pytest.mark.ui
def test_evidence_lists_protected_target_as_pass_through(page, restored_app):
    page.goto(restored_app.url)
    expect(by_id(page, 'ExportProbe')).to_contain_text('pipeline=var_select', timeout=60000)
    by_id(page, 'FlipButton').click(force=True)
    evidence = by_id(page, 'Evidence')
    expect(evidence.locator('.shiny-data-grid')).to_be_visible()
    page.wait_for_function('''id => {
        const grid = document.getElementById(id)?.querySelector('.shiny-data-grid');
        if (!grid) return false;
        grid.scrollTop = grid.scrollHeight;
        return grid.textContent.includes('Pass through');
    }''', arg=evidence.get_attribute('id'))
    row = evidence.locator('tbody tr').filter(has_text='Pass through')
    expect(row).to_have_count(1)
    expect(row).to_contain_text('target')
