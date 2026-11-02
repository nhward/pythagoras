"""Training-local resampling and diagnostic stability without consensus leakage."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from evaluation_plan import build_plan, level_key
from proxy_data import proxy_data
from roles import RoleMap
from selection_analysis import analyze_selection, export_selection, selection_stability, selection_identity
from selection_resampling import selection_inner_splits

pytestmark=pytest.mark.unit


def source(n=240, **plan_options):
    rng=np.random.default_rng(12)
    frame=pd.DataFrame({'signal':rng.normal(size=n),'noise':rng.normal(size=n),
        'nominal':pd.Categorical(rng.choice(['a','b',None],n)),
        'id':np.arange(n),'group':pd.Categorical(np.repeat(np.arange(3),n//3)),
        'time':np.repeat(np.arange(24),n//24)})
    frame['outcome']=frame.signal*7+rng.normal(size=n)*.01
    roles=RoleMap.from_primitive({'predictor':['signal','noise','nominal'],'target':['outcome']})
    strategy=plan_options.get('strategy','kfold')
    if strategy in ('grouped','logo','stratified'):
        roles.set_roles('group',['stratifier'])
    if strategy=='temporal':
        roles.set_roles('time',['sequence'])
    data=proxy_data(frame,_roles=roles)
    plan=build_plan(data,**(dict(strategy='kfold',objective='independent',folds=3,shuffle=False)|plan_options))
    assert plan.ready,plan.errors
    return data.with_evaluation_plan(plan,card='obs_resample')


def options(**kwargs):
    return dict(trees=30,depth=4,iterations=18,alpha=.05,percentile=100,two_step=True,
        folds=5,repeats=1,cutoff=1.,policy='confirmed',random_state=2025,n_jobs=1,use_weights=True)|kwargs


def analyze(data, **kwargs):
    result=analyze_selection(data,'GrootCV',options(**kwargs))
    assert not result['error'],result['error']
    return result


def test_upstream_fold_count_is_inherited_but_memberships_are_not():
    data=source()
    result=analyze(data,repeats=2)
    model=result['model']
    assert len(model.inner_splits_)==6 # upstream K=3, not card fallback K=5
    assert 'codes' not in result['recipe'].resampling
    assert 'development' not in result['recipe'].resampling
    train,_=next(data.evaluation_splits())
    fitted=clone(result['recipe']).fit(data.clean_frame.iloc[train],data.clean_frame.outcome.iloc[train])
    assert len(fitted.inner_splits_)==6
    for a,b in fitted.inner_splits_:
        assert max(np.r_[a,b])<len(train)
        assert not set(a)&set(b)
    assert not hasattr(clone(fitted),'inner_splits_')


@pytest.mark.parametrize('strategy',['grouped','logo'])
def test_inner_groups_are_disjoint_and_rebuilt_on_outer_training_rows(strategy):
    data=source(strategy=strategy,objective='unseen_strata',group='group')
    result=analyze(data)
    train,valid=next(data.evaluation_splits())
    raw=data.clean_frame.iloc[train]
    fitted=clone(result['recipe']).fit(raw,raw.outcome)
    assert len(fitted.inner_splits_)==2 # only two outer-training groups remain
    for a,b in fitted.inner_splits_:
        assert not set(raw.group.iloc[a]) & set(raw.group.iloc[b])
        assert not set(raw.group.iloc[np.r_[a,b]]) & set(data.clean_frame.group.iloc[valid])
    if strategy=='grouped':
        assert any('Using K=2' in note for note in fitted.inner_resampling_['notes'])


@pytest.mark.parametrize('expanding',[True,False])
def test_inner_temporal_boundaries_keep_gap_and_horizon_and_reduce_window(expanding):
    data=source(strategy='temporal',objective='future',sequence='time',window=8,horizon=3,gap=1,step=3,expanding=expanding)
    result=analyze(data)
    outer_train,outer_valid=next(data.evaluation_splits())
    raw=data.clean_frame.iloc[outer_train]
    fitted=clone(result['recipe']).fit(raw,raw.outcome)
    assert len(fitted.inner_splits_)==1
    assert any('reduced from 8 to 4' in note for note in fitted.inner_resampling_['notes'])
    for a,b in fitted.inner_splits_:
        assert raw.time.iloc[a].max()+1 < raw.time.iloc[b].min()
        assert raw.time.iloc[b].nunique()==3
        assert raw.time.iloc[np.r_[a,b]].max() < data.clean_frame.time.iloc[outer_valid].min()
    with pytest.raises(ValueError,match='Too few distinct training times'):
        clone(result['recipe']).fit(raw.iloc[:30],raw.outcome.iloc[:30])


def test_stratification_uses_upstream_variable_instead_of_outcome():
    data=source(strategy='stratified',stratify='group',shuffle=True,seed=9)
    result=analyze(data)
    for _,valid in result['model'].inner_splits_:
        assert data.clean_frame.group.iloc[valid].nunique()==3
    with pytest.raises(ValueError,match='metadata missing'):
        clone(result['recipe']).fit(data.clean_frame.drop(columns='group'),data.clean_frame.outcome)


def test_predefined_validation_is_not_used_as_an_inner_holdout():
    data=source()
    frame=data.frame.copy();frame['partition']=np.where(frame.id<160,'dev','validation')
    roles=RoleMap.from_primitive(data.role_map.to_primitive());roles.set_roles('partition',['partition'])
    data=proxy_data(frame,_roles=roles)
    plan=build_plan(data,strategy='predefined',objective='independent',
        partition_map={level_key('dev'):'development',level_key('validation'):'validation'})
    assert plan.ready,plan.errors
    result=analyze(data.with_evaluation_plan(plan,card='obs_resample'),folds=2)
    assert result['observations']==160
    assert len(result['model'].inner_splits_)==2
    assert all(max(np.r_[a,b])<160 for a,b in result['model'].inner_splits_)


class DropNoiseInOneFold(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):
        self.drop_=X.id.mean()>120
        return self
    def transform(self,X):
        return X.drop(columns='noise') if self.drop_ else X.copy()


@pytest.mark.parametrize('method',['Boruta','GrootCV'])
def test_stability_refits_pipeline_and_does_not_modify_selection(method):
    data=source()
    transform=DropNoiseInOneFold().fit(data.frame)
    data=data.with_pipeline_step(transform,name='drop_noise',preview_frame=transform.transform(data.frame))
    before=selection_identity(data)
    base=analyze_selection(data,method,options())
    assert not base['error']
    mask=base['model'].get_support()
    diagnostic=selection_stability(data,method,options())
    assert not diagnostic['error']
    assert diagnostic['successful']==3
    noise=diagnostic['summary'].set_index('Variable').loc['noise']
    assert noise['Eligible folds']==2 and noise['Unavailable folds']==1
    assert (diagnostic['details']['Validation rows']==80).all()
    np.testing.assert_array_equal(mask,base['model'].get_support())
    assert selection_identity(data)==before
    exported=export_selection(base)
    assert exported.pipeline.named_steps.var_select.resampling==base['recipe'].resampling
    assert 'stability' not in exported.pipeline.named_steps.var_select.get_params()


def test_stability_reports_failed_folds_without_changing_the_denominator(monkeypatch):
    from FeatureSelectionTransformer import FeatureSelectionTransformer
    data=source()
    original=FeatureSelectionTransformer.fit
    calls=[]
    def checked_fit(self,X,y=None,**kwargs):
        calls.append(set(X.id))
        if len(calls)==1:
            raise ValueError('deliberate unavailable fit')
        return original(self,X,y,**kwargs)
    monkeypatch.setattr(FeatureSelectionTransformer,'fit',checked_fit)
    result=selection_stability(data,'Boruta',options())
    assert not result['error']
    assert result['successful']==2 and result['planned']==3
    assert len(result['failures'])==1
    assert (result['summary']['Eligible folds']==2).all()
    assert (result['summary']['Failed folds']==1).all()
    for seen,(train,valid) in zip(calls,data.evaluation_splits()):
        assert seen==set(data.clean_frame.id.iloc[train])
        assert not seen&set(data.clean_frame.id.iloc[valid])


def test_stability_excludes_reserved_observations_and_requires_a_plan():
    data=source()
    bare=proxy_data(data.frame,_roles=data.role_map)
    assert 'upstream obs_resample' in selection_stability(bare,'Boruta',options())['error']
    frame=data.frame.copy();frame['partition']=np.where(frame.id<180,'dev','holdout')
    roles=RoleMap.from_primitive(data.role_map.to_primitive());roles.set_roles('partition',['partition'])
    data=proxy_data(frame,_roles=roles)
    plan=build_plan(data,strategy='kfold',objective='independent',folds=3,
        partition_map={level_key('dev'):'development',level_key('holdout'):'holdout'})
    assert plan.ready
    result=selection_stability(data.with_evaluation_plan(plan,card='obs_resample'),'Boruta',options())
    assert not result['error']
    assert (result['details']['Training rows']==120).all()
    assert (result['details']['Validation rows']==60).all()


def test_predefined_future_design_derives_inner_forward_blocks_locally():
    data=source()
    frame=data.frame.copy();frame['partition']=np.where(frame.time<18,'dev','validation')
    roles=RoleMap.from_primitive(data.role_map.to_primitive())
    roles.set_roles('partition',['partition']);roles.set_roles('time',['sequence'])
    data=proxy_data(frame,_roles=roles)
    plan=build_plan(data,strategy='predefined',objective='future',sequence='time',
        partition_map={level_key('dev'):'development',level_key('validation'):'validation'})
    assert plan.ready,plan.errors
    result=analyze(data.with_evaluation_plan(plan,card='obs_resample'),folds=3)
    assert result['observations']==180
    assert len(result['model'].inner_splits_)==3
    raw=data.clean_frame.iloc[:180]
    for train,valid in result['model'].inner_splits_:
        assert raw.time.iloc[train].max()<raw.time.iloc[valid].min()
        assert raw.time.iloc[valid].max()<18
    assert any('derived 3 local forward splits' in note for note in result['model'].inner_resampling_['notes'])


@pytest.mark.asyncio
async def test_stability_subprocess_transports_only_diagnostic_results():
    from selection_jobs import run_selection
    data=source()
    result=await run_selection(data,'GrootCV',options(),diagnostic=True)
    assert not result['error']
    assert result['successful']==3
    assert not any(key in result for key in ('recipe','model','source'))
