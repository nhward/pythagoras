"""Nominal encoding, training isolation, pipeline schema and footer workflows."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import pytest
from cards import var_encode as m
from NominalEncodingTransformer import NominalEncodingTransformer
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.pytest import create_app_fixture
from sklearn.base import BaseEstimator, TransformerMixin

app=create_app_fixture(app='../scenarios/var_encode.py',scope='function')
restored_app=create_app_fixture(app='../scenarios/var_encode_restored.py',scope='function')
empty_app=create_app_fixture(app='../scenarios/var_encode_empty.py',scope='function')
@pytest.fixture(scope='session')
def browser_context_args():
    return {'viewport':{'width':1700,'height':1000}}


def source():
    frame=pd.DataFrame({'binary':pd.Categorical(['yes','no','yes','no',None,'yes']),
        'color':pd.Categorical(['red','blue','green','red','red','blue']),
        'constant':pd.Categorical(['a']*6),'empty':pd.Categorical([None]*6),
        'ordered':pd.Categorical(['a','b']*3,ordered=True),'numeric':np.arange(6,dtype=float),
        'text':['a','b']*3,'logical':[True,False]*3,'group':pd.Categorical(['g','h']*3),
        'weight':[1.,2.,3.,4.,5.,6.]},index=[4,4,7,8,9,10])
    roles=RoleMap()
    for c in frame:roles.set_roles(c,[Role.PREDICTOR])
    roles.set_roles('group',[Role.STRATIFIER]);roles.set_roles('weight',[Role.WEIGHTING])
    return proxy_data(_df=frame,_roles=roles,_cluster_count=3)


class ScaleNumeric(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):return self
    def transform(self,X):
        out=X.copy();out['numeric']*=2
        return out


@pytest.mark.unit
class TestEncoding:
    def test_only_semantic_nominal_predictors(self):
        assert m._nominal_predictors(source())==['binary','color','constant','empty']

    def test_binary_multilevel_constant_and_missing(self):
        s=source();before=s.clone();r=m._analyze(s);assert not r.error
        assert len(r.transformer.output_columns_)==4
        assert r.frame['binary__no'].tolist()[:4]==[0,1,0,1]
        assert pd.isna(r.frame['binary__no'].iloc[4])
        assert r.frame.filter(like='color__').sum(axis=1).eq(1).all()
        assert 'constant' not in r.frame and 'empty' in r.frame
        assert r.frame.index.equals(s.frame.index) and s.equals(before)
        assert r.table.set_index('Variable').loc['binary','Reference']=='yes'

    def test_retained_originals_and_role_updates(self):
        s=source()
        for remove in [False,True]:
            r=m._analyze(s,remove_original=remove);out=m._apply(s,r)
            assert ('binary' in out.columns)==(not remove)
            assert bool(out.role_map.roles_for('binary'))==(not remove)
            assert all(out.role_map.roles_for(c)=={Role.PREDICTOR} for c in r.transformer.output_columns_)
            assert out.role_map.roles_for('group')=={Role.STRATIFIER}
            assert out.cluster_count==3 and out.pipeline_steps==('var_encode',)
            pd.testing.assert_frame_equal(out.clean_frame,s.frame)

    def test_training_ignores_preview_and_unused_category_metadata(self):
        s=source();out=m._apply(s,m._analyze(s))
        recipe=out.pipeline_for_training().steps[0][1]
        assert not hasattr(recipe,'encodings_')
        training=s.frame.iloc[:2]
        fitted=out.pipeline_for_training().fit(training)
        model=fitted.steps[0][1]
        assert 'green' not in model.encodings_['color'][0]
        with pytest.warns(UserWarning):
            held=fitted.transform(s.frame.iloc[2:3])
        assert held.filter(like='color__').to_numpy().sum()==0
        assert list(held.columns)==list(model.get_feature_names_out())

    def test_grouping_and_unknown_infrequent(self):
        frame=pd.DataFrame({'c':pd.Categorical(['a']*4+['b']*3+['c','d'])})
        model=NominalEncodingTransformer(['c'],max_categories=2,handle_unknown='infrequent_if_exist').fit(frame)
        assert len(model.output_columns_)==1
        assert model.summary_[0]['Pooled levels']=='b, c, d'
        with pytest.warns(UserWarning):
            actual=model.transform(pd.DataFrame({'c':['new']}))
        assert actual.iloc[0,0]==1
        model=NominalEncodingTransformer(['c'],min_frequency=2).fit(frame)
        assert len(model.output_columns_)==3
        assert model.summary_[0]['Pooled levels']=='c, d'
        model=NominalEncodingTransformer(['c'],min_frequency=20).fit(frame)
        assert model.output_columns_==[]

    def test_unknown_error_missing_and_empty_batch(self):
        frame=pd.DataFrame({'c':pd.Categorical(['a','b','c'])})
        model=NominalEncodingTransformer(['c'],handle_unknown='error').fit(frame)
        with pytest.raises(ValueError,match='unknown'):
            model.transform(pd.DataFrame({'c':['z']}))
        assert model.transform(pd.DataFrame({'c':[None]})).isna().all().all()
        assert model.transform(frame.iloc[:0]).shape==(0,3)

    def test_collision_safe_and_mixed_labels(self):
        frame=pd.DataFrame({'c':pd.Categorical([1,'1',2]),'c__1':[9,8,7]})
        model=NominalEncodingTransformer(['c']).fit(frame)
        out=model.transform(frame)
        assert out.columns.is_unique and out['c__1'].tolist()==[9,8,7]
        assert len(model.output_columns_)==3
        np.testing.assert_array_equal(out[model.output_columns_].to_numpy(),np.eye(3))

    def test_composes_after_prior_pipeline(self):
        s=source();step=ScaleNumeric().fit(s.frame)
        upstream=s.with_pipeline_step(step,name='scale',preview_frame=step.transform(s.frame))
        out=m._apply(upstream,m._analyze(upstream))
        assert out.pipeline_steps==('scale','var_encode')
        pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
        assert len(out.processing_records)==2
        assert not hasattr(out.pipeline_for_training().steps[-1][1],'encodings_')

    def test_no_eligible_and_all_missing_are_benign(self):
        s=proxy_data(pd.DataFrame({'n':[1.,2.]}))
        assert m._apply(s,m._analyze(s)) is s
        s=proxy_data(pd.DataFrame({'c':pd.Categorical([None,None])}))
        assert m._apply(s,m._analyze(s)) is s

    def test_explicit_removal_validation(self):
        s=source();step=ScaleNumeric()
        with pytest.raises(ValueError,match='preserve DataFrame columns'):
            s.with_pipeline_step(step,name='bad',preview_frame=s.frame.drop(columns='binary'))
        with pytest.raises(ValueError,match='unique existing'):
            s.with_pipeline_step(step,name='bad',preview_frame=s.frame,removed_columns=['absent'])
        with pytest.raises(ValueError,match='preserve the DataFrame index'):
            s.with_pipeline_step(step,name='bad',preview_frame=s.frame.reset_index(drop=True))

    def test_weights_do_not_change_frequency_counts(self):
        s=source();a=m._analyze(s,min_frequency=2)
        s.frame['weight']*=100
        b=m._analyze(s,min_frequency=2)
        pd.testing.assert_frame_equal(a.frame.drop(columns='weight'),b.frame.drop(columns='weight'))
        pd.testing.assert_frame_equal(a.table,b.table)


def by_id(page,name):
    namespace=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')


@pytest.mark.ui
def test_toggle_reversible_and_settings_rebuild_single_step(page,app):
    page.goto(app.url)
    # expect(by_id(page,'Status')).to_contain_text('Preview: 4 indicators',timeout=30000)
    expect(by_id(page,'NominalTable')).to_contain_text('Cardinality')
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True')
    toggle=by_id(page,'Encode').locator('input[value="nominal"]')
    toggle.check()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    expect(by_id(page,'Status')).to_contain_text('Nominal: 4 new predictors; 2 original predictors removed.')
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'RemoveOriginal').uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=True',timeout=30000)
    toggle.uncheck()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=True',timeout=30000)
    expect(by_id(page,'Probe')).to_contain_text('unchanged=True')


@pytest.mark.ui
def test_restored_enabled_encoding(page,restored_app):
    page.goto(restored_app.url)
    expect(by_id(page,'Probe')).to_contain_text('steps=1; original=False',timeout=30000)
    expect(by_id(page,'Status')).to_contain_text('Nominal: 4 new predictors; 2 original predictors removed.')
    expect(by_id(page,'Encode').locator('input[value="nominal"]')).to_be_checked()


@pytest.mark.ui
def test_empty_panel_toggle_is_benign(page,empty_app):
    page.goto(empty_app.url)
    expect(by_id(page,'NominalMessage')).to_contain_text('No nominal predictors are available.',timeout=30000)
    by_id(page,'Encode').locator('input[value="nominal"]').check()
    expect(by_id(page,'Probe')).to_contain_text('steps=0; original=False; unchanged=True')
