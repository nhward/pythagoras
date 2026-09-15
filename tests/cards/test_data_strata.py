"""Distribution facets, original-unit ANOVA, role filtering and browser workflows."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import pytest
from cards import data_strata as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from scipy.stats import f_oneway
from shiny.playwright import controller
from shiny.pytest import create_app_fixture

app=create_app_fixture(app='../scenarios/data_strata.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/data_strata_restored.py',scope='function')
@pytest.fixture(scope='session')
def browser_context_args():return {'viewport':{'width':1800,'height':1100}}

def source():
    rng=np.random.default_rng(2025)
    frame=pd.DataFrame({'x':np.r_[rng.normal(0,1,30),rng.normal(5,2,30)],'y':rng.normal(size=60),
                        'group':['a']*30+['b']*30,'target':np.arange(60,dtype=float),'id':np.arange(60),'weight':np.arange(60)+1.})
    roles=RoleMap()
    for c in ['x','y']:roles.set_roles(c,[Role.PREDICTOR])
    for c,r in [('group',Role.STRATIFIER),('target',Role.TARGET),('id',Role.IDENTIFIER),('weight',Role.WEIGHTING)]:roles.set_roles(c,[r])
    return proxy_data(_df=frame,_roles=roles)

@pytest.mark.unit
class TestDistributions:
    def test_roles(self):
        d=source()
        assert m._variables(d)==['x','y']
        assert m._variables(d,True)==['x','y','target']
        assert m._stratifiers(d)==['group']
    @pytest.mark.parametrize('method',['ordinary','welch'])
    def test_anova_matches_scipy(self,method):
        d=source();groups=[d.frame.x.iloc[:30],d.frame.x.iloc[30:]]
        r=m._anova(groups,method);expected=f_oneway(*groups,equal_var=method=='ordinary')
        assert r['F']==pytest.approx(expected.statistic)
        assert r['p']==pytest.approx(expected.pvalue)
        assert 0<r['Eta squared']<1
    def test_normalization_preserves_stratum_differences_and_anova(self):
        d=source();a=m._analyze(d,['x'],'group');b=m._analyze(d,['x'],'group',normalization='zscore')
        assert a.anova.equals(b.anova)
        assert a.summary.equals(b.summary)
        assert b.values['x',0].mean()<0<b.values['x',1].mean()
        assert np.concatenate(list(b.values.values())).std(ddof=1)==pytest.approx(1)
    def test_facets_share_scale_and_types_switch(self):
        r=m._analyze(source(),['x','y'],'group')
        for kind in ['box','violin']:
            f=m._figure(r,kind=kind,points=True,mean=True,notches=True)
            assert len(f.data)==4 and all(t.type==kind for t in f.data)
            assert f.layout.yaxis.range==f.layout.yaxis2.range
            assert f.layout.yaxis3.range==f.layout.yaxis4.range
            assert all(t.customdata is not None for t in f.data)
    def test_empty_constant_singleton_and_missing(self):
        d=source();d.frame.loc[0,'group']=None;d.frame.loc[1,'x']=np.inf
        r=m._analyze(d,['x'],'group')
        assert r.missing_strata==1 and r.rows==59
        assert r.summary['Missing/nonfinite'].sum()==1
        assert r.anova.N.iloc[0]==58
        d.frame['x']=1
        r=m._analyze(d,['x'],'group')
        assert 'identical' in r.anova.Status.iloc[0]
        assert all(t.type=='box' for t in m._figure(r).data)
        assert 'two finite' in m._anova([[1],[2,3]],'ordinary')['Status']
        assert 'positive variance' in m._anova([[1,1],[2,3]],'welch')['Status']
    def test_no_stratifier_or_selection(self):
        r=m._analyze(source(),['x'])
        assert r.levels==['All observations'] and 'Choose a Stratifier' in r.anova.Status.iloc[0]
        empty=m._analyze(source(),[])
        assert empty.error and len(m._figure(empty).layout.images)==1
    def test_sampling_reproducible_and_retains_rare_strata(self):
        codes=np.r_[np.zeros(1000,dtype=int),np.ones(2,dtype=int)]
        selected=m._sample(codes,100)
        assert len(selected)==100 and set(codes[selected])=={0,1}
        np.testing.assert_array_equal(selected,m._sample(codes,100))
    def test_too_many_strata_not_silently_dropped(self):
        d=source();d.frame['group']=np.arange(60)
        assert '60 strata exceed' in m._analyze(d,['x'],'group').error
    def test_weights_explicitly_not_applied_and_input_unchanged(self):
        d=source();before=d.clone();a=m._analyze(d,['x'],'group')
        assert d.equals(before) and 'weighting column is not applied' in a.note
        d.frame['weight']*=100
        assert a.anova.equals(m._analyze(d,['x'],'group').anova)
    def test_bh_adjustment(self):
        np.testing.assert_allclose(m._adjust_p([.01,.04,.03,np.nan]),[.03,.04,.04,np.nan],equal_nan=True)
    def test_identifier_hover_and_missing_level_unused_categories(self):
        d=source();d.frame.index=[0]*60;d.frame['id']=d.frame.id.astype(object);d.frame.iloc[0,d.frame.columns.get_loc('id')]='<case>'
        d.frame['group']=pd.Categorical(d.frame.group,categories=['a','b','unused'])
        r=m._analyze(d,['x'],'group')
        assert r.levels==['a','b']
        assert r.hover['x',0][0]=='id: &lt;case&gt;'


def by_id(page,name):
    ns=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{ns}-{name}')
def settings(page):
    page.locator('.card').first.hover();page.locator('.card').first.locator('button.collapse-toggle').click()

@pytest.mark.ui
def test_initial_selection_facets_anova_and_type_switch(page,app):
    page.goto(app.url)
    expect(by_id(page,'Status')).to_contain_text('1 variables; 2 strata',timeout=60000)
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')
    chart=by_id(page,'Plots').locator('.js-plotly-plot');expect(chart).to_be_visible()
    assert chart.evaluate('el=>el.data.map(t=>t.type)')==['violin','violin']
    settings(page)
    by_id(page,'Kind').locator('input[value="box"]').check()
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data?.every(t=>t.type==='box')")
    settings(page)
    page.locator('.card').first.hover();by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'Anova')).to_contain_text('Eta squared')
    expect(by_id(page,'Anova')).to_contain_text('BH adjusted p')
    page.get_by_role('tab',name='Group summaries',exact=True).click()
    expect(by_id(page,'Summaries')).to_contain_text('Median')

@pytest.mark.ui
def test_multiple_variables_target_and_no_facet(page,app):
    page.goto(app.url)
    expect(by_id(page,'Status')).to_contain_text('1 variables; 2 strata',timeout=60000)
    settings(page)
    by_id(page,'IncludeTarget').check()
    selector=controller.InputSelectize(page,by_id(page,'Variables').get_attribute('id'))
    selector.set(['x','y','target'])
    expect(by_id(page,'Status')).to_contain_text('3 variables; 2 strata',timeout=60000)
    by_id(page,'Stratifier').select_option('')
    expect(by_id(page,'Status')).to_contain_text('3 variables; 1 strata',timeout=60000)
    selector.set([])
    expect(by_id(page,'Status')).to_contain_text('Select at least one numeric',timeout=60000)
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')

@pytest.mark.ui
def test_restored_choices_and_welch(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Status')).to_contain_text('2 variables; 2 strata',timeout=60000)
    page.locator('.card').first.hover();by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'Anova')).to_contain_text('Welch')


restoring_app=create_app_fixture(app='../scenarios/data_strata_restoring.py',scope='function')

@pytest.mark.ui
def test_variables_survive_late_upstream_type_restore(page,restoring_app):
    page.goto(restoring_app.url)
    expect(by_id(page,'Status')).to_contain_text('1 variables; 2 strata',timeout=60000)
    by_id(page,'FinishRestore').click()
    expect(by_id(page,'Status')).to_contain_text('2 variables; 2 strata',timeout=60000)
    assert by_id(page,'Variables').evaluate('el=>el.selectize.getValue()') == ['x','y']
