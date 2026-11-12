"""Surrogate-tree fidelity, role filtering, importance, and browser interactions."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import pytest
from cards import obs_cluster_profile as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/obs_cluster_profile.py',scope='function')
@pytest.fixture(scope='session')
def browser_context_args():
    return {'viewport':{'width':1800,'height':1100}}

def source():
    x = np.arange(60,dtype=float)
    frame = pd.DataFrame({'x':x,'category':np.resize(['a','b'],60),
        'cluster__partition':pd.Categorical(np.where(x<30,'c1','c2'),categories=['c1','c2','unused']),
        'cluster__mixture_2':np.where(x<30,'c1','c2'), 'weight':np.resize([.5,1.,3.],60), 'id':x, 'target':x})
    roles = RoleMap()
    for col in ['x','category','cluster__mixture_2']:roles.set_roles(col,[Role.PREDICTOR])
    roles.set_roles('cluster__partition',[Role.STRATIFIER])
    roles.set_roles('weight',[Role.WEIGHTING]);roles.set_roles('id',[Role.IDENTIFIER]);roles.set_roles('target',[Role.TARGET])
    return proxy_data(_df=frame,_roles=roles)

def analyze(data=None,**kwargs):
    return m._analyze(source() if data is None else data,target='cluster__partition',**kwargs)

@pytest.mark.unit
class TestProfile:
    @pytest.mark.parametrize('name', ['cluster__partition', 'cluster__mixture', 'cluster__density_2'])
    def test_current_generated_names_are_found_without_membership_leakage(self, name):
        data = source()
        data.frame[name] = data.frame['cluster__partition']
        data.role_map.set_roles(name, [Role.STRATIFIER])
        data.frame['cluster__mixture_3'] = data.frame['cluster__partition']
        data.role_map.set_roles('cluster__mixture_3', [Role.PREDICTOR])
        assert m._targets(data) == list(dict.fromkeys(['cluster__partition', name]))
        result = m._analyze(data, target=name)
        assert not result.message
        assert result.predictors == ['x', 'category']
        assert result.balanced > .9

    def test_exported_membership_is_discovered_and_profiled(self):
        from cards.obs_clusters import _export_memberships, _membership_name
        from ClusterMembershipTransformer import ClusterMembershipTransformer
        data = source()
        name = _membership_name(data.columns, 'cluster__partition')
        transformer = ClusterMembershipTransformer(('x',), output_column=name).fit(data.frame)
        exported = _export_memberships(data, {'Partition': transformer})
        assert name == 'cluster__partition_2'
        assert name in m._targets(exported)
        result = m._analyze(exported, target=name)
        assert not result.message
        assert result.predictors == ['x', 'category']

    def test_role_and_name_required(self):
        data=source()
        assert m._targets(data)==['cluster__partition']
        data.role_map.set_roles('cluster__mixture_2',[Role.STRATIFIER])
        data.role_map.set_roles('id',[Role.STRATIFIER])
        assert m._targets(data)==['cluster__partition','cluster__mixture_2']
        assert m._analyze(data,'id').message
    def test_explains_labels_and_excludes_leaking_roles(self):
        data=source();r=analyze(data)
        assert not r.message
        assert r.balanced>.9 and r.accuracy>.9
        assert 0 <= r.baseline_balanced < r.balanced
        assert r.predictors==['x','category']
        assert len(r.rules)>=2 and set(r.recall.Cluster)=={'c1','c2'}
        assert r.confusion.drop(columns='Actual').to_numpy().sum()==pytest.approx(60)
        assert not data.has_pipeline
        assert m._figure(r).layout.annotations
    def test_weighting_zero_rows_duplicate_index_and_scale_invariance(self):
        data=source();data.frame.index=[0]*60;data.frame.iloc[0,data.frame.columns.get_loc('weight')]=0
        a=analyze(data)
        data.frame['weight']*=100
        b=analyze(data)
        assert a.observations==59 and a.weighted
        assert a.accuracy==pytest.approx(b.accuracy)
        assert analyze(data,use_weights=False).observations==60
    @pytest.mark.parametrize('bad',[np.nan,-1,np.inf])
    def test_invalid_importance_is_not_silently_repaired(self,bad):
        data=source();data.frame.loc[0,'weight']=bad
        with pytest.warns(UserWarning, match='Using unweighted'):
            result = analyze(data)
        expected = analyze(data, use_weights=False)
        assert not result.message and not result.weighted
        assert result.accuracy == pytest.approx(expected.accuracy)
        assert not analyze(data,use_weights=False).message
    def test_missing_predictors_are_imputed_and_missing_labels_excluded(self):
        data=source();data.frame.loc[0,'x']=np.nan;data.frame.loc[1,'category']=None;data.frame.loc[2,'cluster__partition']=None
        result=analyze(data)
        assert not result.message and result.observations==59
    def test_unallocated_and_single_class(self):
        data=source();data.frame['cluster__partition']=np.where(data.frame.x<30,'unallocated','c2')
        assert not analyze(data).message
        assert 'two clusters' in analyze(data,include_unallocated=False).message
    def test_stratified_cap_and_fold_cap(self):
        result=analyze(limit=12,folds=10)
        assert not result.message and result.observations==12 and result.eligible==60 and result.folds==6
    def test_no_predictors_and_rare_class(self):
        data=source()
        for c in ['x','category']:data.role_map.set_roles(c,[Role.NONE])
        assert 'No Predictor-role' in analyze(data).message
        data=source();data.frame['cluster__partition']=['c1']*59+['c2']
        assert 'two analyzed observations' in analyze(data).message
    def test_fold_pipeline_handles_unseen_category(self):
        model=m._pipeline(['x'],['category'],3,.02)
        model.fit(pd.DataFrame({'x':[0.,1.,2.,3.],'category':['a']*4}),['c1','c1','c2','c2'])
        assert len(model.predict(pd.DataFrame({'x':[np.nan],'category':['new']})))==1
    def test_importance_changes_fitting_and_accuracy(self):
        data=source();data.frame['x']=1;data.frame['category']='same'
        data.frame['weight']=np.where(data.frame.cluster__partition=='c1',1.,9.)
        weighted=analyze(data);equal=analyze(data,use_weights=False)
        assert weighted.accuracy==pytest.approx(.9)
        assert equal.accuracy==pytest.approx(.5)
        assert set(weighted.rules['Predicted cluster'])=={'c2'}
        assert weighted.balanced==pytest.approx(.5)
    def test_stump_is_a_valid_explanation(self):
        data=source();data.frame['x']=1;data.frame['category']='same'
        r=analyze(data)
        assert not r.message and len(r.rules)==1
        assert r.rules.Rule.iloc[0]=='All observations'
        assert len(m._figure(r).layout.annotations)==1


def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')

@pytest.mark.ui
def test_profile_chart_fills_panel_and_grows_in_fullscreen(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Accuracy')).to_contain_text('CV accuracy', timeout=60000)
    tree = by_id(page, 'Tree_cluster__partition')
    plot = tree.locator('.js-plotly-plot')
    expect(plot).to_be_visible()
    def fills_panel():
        page.wait_for_function('''el => {
            const pane = el.closest('.tab-pane');
            const svg = el.querySelector('.main-svg');
            return svg && Math.abs(el.getBoundingClientRect().height - pane.getBoundingClientRect().height) < 2
                && Math.abs(svg.getBoundingClientRect().height - el.getBoundingClientRect().height) < 2;
        }''', arg=tree.element_handle())
    fills_panel()
    assert plot.evaluate('el => el.layout.margin.t') == 0
    before = tree.bounding_box()['height']
    page.locator('.card').first.hover()
    by_id(page, 'ExpandButton').click()
    page.wait_for_function('el => el.getBoundingClientRect().height > ' + str(before + 100), arg=tree.element_handle())
    fills_panel()
    page.get_by_role('tab', name='Density_2', exact=True).click()
    other = by_id(page, 'Tree_cluster__density_2')
    expect(other.locator('.js-plotly-plot')).to_be_visible()
    assert other.evaluate("el => el.closest('.tab-pane').classList.contains('html-fill-container')")

@pytest.mark.ui
def test_tree_metrics_tables_and_pass_through(page,app):
    page.goto(app.url)
    expect(by_id(page,'Accuracy')).to_contain_text('CV accuracy',timeout=60000)
    expect(by_id(page,'Accuracy')).to_contain_text('59 of 59 rows; 2 predictors')
    expect(by_id(page,'Accuracy')).to_contain_text('Observation importance applied')
    expect(by_id(page,'Tree_cluster__partition').locator('.js-plotly-plot')).to_be_visible()
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')
    page.locator('.card').first.hover();by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'Comparison')).to_contain_text('Partition')
    expect(by_id(page,'Comparison')).to_contain_text('Density_2')
    page.get_by_role('tab',name='Rules',exact=True).click()
    expect(by_id(page,'Rules')).to_contain_text('Training purity')
    page.get_by_role('tab',name='Cluster accuracy',exact=True).click()
    expect(by_id(page,'Recall')).to_contain_text('CV Recall')
    page.get_by_role('tab',name='Confusion',exact=True).click()
    expect(by_id(page,'Confusion')).to_contain_text('Predicted c1')

@pytest.mark.ui
def test_target_switch_and_weighting_option(page,app):
    page.goto(app.url)
    expect(by_id(page,'Accuracy')).to_contain_text('CV accuracy',timeout=60000)
    page.get_by_role('tab',name='Density_2',exact=True).click()
    expect(by_id(page,'Accuracy')).to_contain_text('Density_2:',timeout=60000)
    page.locator('.card').first.hover();page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'UseWeights').uncheck()
    expect(by_id(page,'Accuracy')).to_contain_text('60 of 60 rows',timeout=60000)
    by_id(page,'Unallocated').uncheck()
    expect(by_id(page,'Accuracy')).to_contain_text('50 of 50 rows',timeout=60000)
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')


@pytest.mark.ui
def test_dynamic_membership_tabs_and_empty_state(page,app):
    page.goto(app.url)
    expect(by_id(page,'Accuracy')).to_contain_text('CV accuracy',timeout=60000)
    expect(page.get_by_role('tab',name='Partition',exact=True)).to_be_visible()
    page.get_by_role('tab',name='Density_2',exact=True).click()
    expect(by_id(page,'Tree_cluster__density_2').locator('.js-plotly-plot')).to_be_visible()
    expect(by_id(page,'Accuracy')).to_contain_text('Density_2:')
    by_id(page,'RemoveMemberships').click()
    expect(page.get_by_role('tab',name='No memberships',exact=True)).to_be_visible()
    expect(page.get_by_role('tab',name='Partition',exact=True)).to_have_count(0)
    expect(by_id(page,'Accuracy')).to_contain_text('Assign a cluster-named column the Stratifier role.',timeout=60000)
    empty = by_id(page,'EmptyTree').locator('.js-plotly-plot')
    expect(empty).to_be_visible()
    state = empty.evaluate('el => ({images: el.layout.images.length, message: el.layout.annotations[0].text})')
    assert state['images'] == 1
    assert 'No cluster membership' in state['message']
    by_id(page,'RestoreMemberships').click()
    expect(page.get_by_role('tab',name='Partition',exact=True)).to_be_visible()
    expect(page.get_by_role('tab',name='No memberships',exact=True)).to_have_count(0)
    expect(by_id(page,'Accuracy')).to_contain_text('Partition:',timeout=60000)
    page.get_by_role('tab',name='Density_2',exact=True).click()
    expect(by_id(page,'Tree_cluster__density_2').locator('.js-plotly-plot')).to_be_visible()
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')

restored_app = create_app_fixture(app='../scenarios/obs_cluster_profile_restored.py',scope='function')

@pytest.mark.ui
def test_restored_membership_tab_is_selected(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Accuracy')).to_contain_text('Density_2:',timeout=60000)
    expect(page.get_by_role('tab',name='Density_2',exact=True)).to_have_attribute('aria-selected','true')
    expect(by_id(page,'Tree_cluster__density_2').locator('.js-plotly-plot')).to_be_visible()


@pytest.mark.unit
def test_unavailable_tree_uses_standard_empty_figure():
    figure = m._figure(m.Profile('cluster__partition', message='Too few observations'))
    assert len(figure.layout.images) == 1
    assert figure.layout.annotations[0].text == 'Too few observations'
    assert not figure.layout.xaxis.visible
