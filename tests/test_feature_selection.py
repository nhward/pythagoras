"""Original-variable selection, training isolation, and native mixed-data adapters."""
import asyncio
import pickle

import numpy as np
import pandas as pd
import pytest
from FeatureSelectionTransformer import FeatureSelectionTransformer
from feature_selection import encode_selection_frame
from model_commit import build_model
from proxy_data import proxy_data
from roles import RoleMap
from selection_analysis import analyze_selection, export_selection, selection_identity
from selection_jobs import run_selection
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.unit


def example(n=240):
    rng = np.random.default_rng(83)
    signal = rng.normal(size=n)
    f = pd.DataFrame({'signal':signal, 'noise':rng.normal(size=n),
        'nominal':pd.Categorical(rng.choice(['apple', 'pear', None], n)),
        'flag':pd.array(rng.choice([True, False, None], n), dtype='boolean'),
        'empty':pd.array([None]*n, dtype='Float64'), 'id':np.arange(n),
        'y':signal * 5 + rng.normal(size=n)*.1}, index=np.zeros(n, dtype=int))
    return proxy_data(f, _roles=RoleMap.from_primitive(
        {'predictor':['signal','noise','nominal','flag','empty'], 'identifier':['id'], 'target':['y']}))


def options(**kwargs):
    o = dict(trees=40, depth=4, iterations=18, alpha=.05, percentile=100,
        two_step=True, folds=3, repeats=1, cutoff=1., policy='confirmed',
        random_state=2025, n_jobs=1, use_weights=True)
    o.update(kwargs)
    return o


def fit(source=None, method='Boruta', **kwargs):
    return analyze_selection(example() if source is None else source, method, options(**kwargs))


@pytest.mark.parametrize('method', ['Boruta','GrootCV'])
def test_mixed_missing_inputs_select_and_preserve_raw_schema(method):
    source = example()
    result = fit(source, method)
    assert not result['error']
    m = result['model']
    assert 'signal' in m.selected_columns_
    assert 'empty' not in m.selected_columns_
    out = export_selection(result)
    pd.testing.assert_frame_equal(out.frame, source.frame[[c for c in source.frame if c not in m.removed_columns_]])
    pd.testing.assert_frame_equal(out.clean_frame, source.clean_frame)
    assert out.pipeline_steps == ('var_select',)
    assert not hasattr(out.pipeline.named_steps.var_select, 'selected_columns_')
    assert not hasattr(result['recipe'], 'selected_columns_')
    assert len(m.details_['real']) > 0
    assert all(len(v) > 0 for v in m.details_['shadows'])
    assert m.get_support().shape == (5,)
    assert list(m.get_feature_names_out()) == list(out.frame)
    assert list(m.get_support(indices=True)) == list(np.flatnonzero(m.support_))
    assert len(pickle.dumps(m)) > 0


def test_category_codebooks_do_not_learn_validation_categories():
    train = pd.DataFrame({'category':pd.Categorical(['a',None,'b']), 'number':pd.array([1,None,3],dtype='Int64')})
    encoded, vocabulary = encode_selection_frame(train)
    assert vocabulary == {'category':['a','b']}
    query = pd.DataFrame({'category':['novel','b',None], 'number':[4.,5.,6.]})
    seen, _ = encode_selection_frame(query, vocabulary)
    assert seen.category.isna().tolist() == [True, False, True]
    assert encoded.number.isna().tolist() == [False, True, False]
    assert vocabulary == {'category':['a','b']}


@pytest.mark.parametrize('method',['Boruta','GrootCV'])
def test_whole_nominal_variable_and_missingness_can_be_selected(method):
    rng = np.random.default_rng(10);n=360
    categories = rng.choice(['left','middle','right',None],n)
    frame = pd.DataFrame({'c':pd.Categorical(categories),'noise':rng.normal(size=n)})
    frame['y'] = pd.Series(categories).map({'left':0.,'middle':10.,'right':-10.}).fillna(25.).to_numpy()
    source = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor':['c','noise'],'target':['y']}))
    result = fit(source,method)
    assert not result['error']
    assert 'c' in result['model'].selected_columns_
    pd.testing.assert_series_equal(export_selection(result).frame.c, source.frame.c)


def test_preview_is_not_a_training_mask_and_missing_y_preserves_rows():
    source = example()
    result = fit(source)
    original_selection = list(result['model'].selected_columns_)
    recipe = export_selection(result).pipeline_for_training()
    train = source.clean_frame.iloc[:180].copy()
    # Reverse the target relationship so a fresh fit must select noise.
    y = train.noise.to_numpy()*8
    y[:2] = np.nan
    recipe.fit(train, y)
    fitted = recipe.named_steps.var_select
    assert 'noise' in fitted.selected_columns_
    assert not np.isin([0,1],fitted.fit_positions_).any()
    assert len(recipe.transform(source.clean_frame.iloc[180:])) == 60
    assert result['model'].selected_columns_ == original_selection
    pd.testing.assert_index_equal(recipe.transform(train).index, train.index)


def test_held_out_target_perturbation_does_not_affect_fit():
    source = example()
    a, b = source.clean_frame.copy(), source.clean_frame.copy()
    b.iloc[180:, b.columns.get_loc('y')] = -999999
    recipe = FeatureSelectionTransformer(('signal','noise','nominal'),trees=30,iterations=15)
    left = clone(recipe).fit(a.iloc[:180],a.y.iloc[:180])
    right = clone(recipe).fit(b.iloc[:180],b.y.iloc[:180])
    np.testing.assert_array_equal(left.get_support(),right.get_support())
    np.testing.assert_allclose(left.details_['real'],right.details_['real'])


def test_training_can_select_surviving_candidates_after_upstream_removal():
    frame = example().frame.drop(columns=['noise'])
    m = FeatureSelectionTransformer(('signal','noise','nominal'),trees=25,iterations=15).fit(frame,frame.y)
    assert m.candidate_columns_ == ['signal','nominal']
    with pytest.raises(ValueError, match='Missing fitted'):
        m.transform(frame.drop(columns='signal'))


@pytest.mark.parametrize('method', ['Boruta', 'GrootCV'])
@pytest.mark.parametrize('protected_roles', [['treatment'], ['sensitive'], ['predictor', 'treatment']])
def test_nonpredictor_roles_pass_through_selection_and_evidence(method, protected_roles):
    from selection_analysis import selection_evidence
    source = example()
    source.role_map.set_roles('nominal', protected_roles)
    result = fit(source, method)
    assert not result['error'], result['error']
    model = result['model']
    protected = ['nominal', 'id', 'y']
    assert not set(protected).intersection(model.candidate_columns_)
    exported = export_selection(result)
    training = exported.pipeline_for_training().fit(source.frame, source.frame.y.to_numpy())
    transformed = training.transform(source.frame)
    table = selection_evidence(result)
    assert 'Rank' not in table.columns
    assert set(table.Decision) <= ({'Confirmed', 'Tentative', 'Rejected', 'Pass through'}
                                   if method == 'Boruta' else {'Selected', 'Rejected', 'Pass through'})
    assert list(table.Variable) == list(source.columns)
    evidence = table.set_index('Variable')
    for column in protected:
        pd.testing.assert_series_equal(exported.frame[column], source.frame[column])
        pd.testing.assert_series_equal(transformed[column], source.frame[column])
        assert exported.role_map.roles_for(column) == source.role_map.roles_for(column)
        assert evidence.loc[column, 'Decision'] == 'Pass through'
        assert evidence.loc[column, 'Included']
        assert pd.isna(evidence.loc[column, 'Mean importance'])
    assert evidence.loc['empty', 'Decision'] == 'Rejected'
    assert not evidence.loc['empty', 'Included']
    assert evidence.loc['signal', 'Decision'] != 'Pass through'


def test_clone_identity_and_input_schema_errors():
    source = example()
    assert selection_identity(source) == selection_identity(source.clone())
    result = fit(source)
    assert not hasattr(clone(result['model']), 'selected_columns_')
    for values in [[{'nested':1}]*len(source), [complex(1,2)]*len(source)]:
        source.frame['signal'] = values
        source.clean_frame['signal'] = values
        assert 'Unsupported' in fit(source)['error']


@pytest.mark.parametrize('method',['Boruta','GrootCV'])
def test_all_missing_and_no_signal_are_not_forced_to_select(method):
    source = example()
    source.frame['signal'] = np.nan
    source.clean_frame['signal'] = np.nan
    # An entirely empty candidate has no importance; no arbitrary keep-one rule.
    recipe = FeatureSelectionTransformer(('signal',),method=method,trees=20,iterations=15,folds=3,repeats=1)
    model = recipe.fit(source.frame, source.frame.y)
    assert model.selected_columns_ == []
    assert len(model.transform(source.frame)) == len(source)


def test_weights_are_strict_and_structure_never_silently_randomizes():
    source=example()
    source.frame['weight']=np.nan;source.clean_frame['weight']=np.nan
    source.role_map.set_roles('weight',['weighting'])
    assert 'nonmissing' in fit(source)['error']
    assert not fit(source,use_weights=False)['error']
    source.role_map.set_roles('id',['sequence'])
    assert 'group/time' in fit(source,'GrootCV',use_weights=False)['error']


def test_multiclass_nominal_missing_inputs():
    source=example(300)
    source.frame['y']=pd.Categorical(np.where(source.frame.signal> .5,'up',np.where(source.frame.signal<-.5,'down','middle')))
    source.clean_frame['y']=source.frame.y
    for method in ('Boruta','GrootCV'):
        result=fit(source,method)
        assert not result['error']
        assert 'signal' in result['model'].selected_columns_


def test_preview_refits_upstream_instead_of_slicing_global_preview():
    source=example()
    # No evaluation plan: simulate development input directly with a smaller source.
    scaler=StandardScaler().set_output(transform='pandas')
    numeric=source.frame[['signal','noise','y']]
    raw=proxy_data(numeric,_roles=RoleMap.from_primitive({'predictor':['signal','noise'],'target':['y']}))
    upstream=raw.with_pipeline_step(scaler,name='scale',preview_frame=scaler.fit_transform(numeric))
    result=fit(upstream)
    assert not result['error']
    assert 'signal' in result['model'].selected_columns_


@pytest.mark.asyncio
async def test_subprocess_roundtrip_and_cancellation_reaps_worker(monkeypatch):
    source=example()
    result=await run_selection(source,'Boruta',options())
    assert not result['error']
    assert 'signal' in result['model'].selected_columns_
    original=asyncio.create_subprocess_exec
    processes=[]
    async def tracked(*args,**kwargs):
        process=await original(*args,**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio,'create_subprocess_exec',tracked)
    task=asyncio.create_task(run_selection(source,'GrootCV',options(trees=2000,repeats=20)))
    for _ in range(100):
        if processes: break
        await asyncio.sleep(.01)
    assert processes
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert processes[0].returncode is not None


def test_groot_category_dictionaries_are_fitted_inside_each_inner_fold(monkeypatch):
    import feature_selection
    seen=[]
    original=feature_selection.encode_selection_frame
    def capture(frame, reference=None):
        result=original(frame,reference)
        seen.append((len(frame),reference,result[1]))
        return result
    monkeypatch.setattr(feature_selection,'encode_selection_frame',capture)
    result=fit(example(90),'GrootCV')
    assert not result['error']
    assert len(seen)==6
    assert [rows for rows,_,_ in seen] == [60,30]*3
    for train, validation in zip(seen[::2],seen[1::2]):
        assert train[1] is None
        assert validation[1] == train[2]


def test_development_preview_excludes_reserved_targets_and_refits_upstream():
    from evaluation_plan import build_plan, level_key
    from sklearn.base import BaseEstimator, TransformerMixin
    class CenterSignal(TransformerMixin,BaseEstimator):
        def fit(self,X,y=None):
            self.mean_=X.signal.mean()
            return self
        def transform(self,X):
            out=X.copy();out['signal']=out.signal-self.mean_
            return out
    data=example()
    data.frame['partition']=np.where(np.arange(len(data))<180,'dev','held')
    data.role_map.set_roles('partition',['partition'])
    data.frame.iloc[180:,data.frame.columns.get_loc('signal')]+=1000
    data=proxy_data(data.frame,_roles=data.role_map)
    def planned(source):
        plan=build_plan(source,strategy='kfold',objective='independent',folds=3,
            purpose='holdout',development_levels=[level_key('dev')],reserved_levels=[level_key('held')])
        assert plan.ready
        return source.with_evaluation_plan(plan,card='obs_resample')
    data=planned(data)
    transform=CenterSignal().fit(data.frame)
    source=data.with_pipeline_step(transform,name='center',preview_frame=transform.transform(data.frame))
    result=fit(source)
    assert not result['error']
    assert result['observations']==180
    exported=export_selection(result)
    assert exported.evaluation_plan.ready
    exported.evaluation_plan.require(exported,supervised=True)
    assert exported.evaluation_plan.development==source.evaluation_plan.development
    assert exported.evaluation_plan.holdout==source.evaluation_plan.holdout
    manual=FeatureSelectionTransformer(tuple(result['recipe'].columns),**{
        k:v for k,v in result['recipe'].get_params(deep=False).items() if k!='columns'})
    expected=CenterSignal().fit_transform(data.clean_frame.iloc[:180])
    manual.fit(expected, data.clean_frame.y.iloc[:180])
    np.testing.assert_allclose(result['model'].details_['real'],manual.details_['real'])
    perturbed=data.clone()
    perturbed.clean_frame.iloc[180:,perturbed.clean_frame.columns.get_loc('y')]=-99999
    perturbed=planned(perturbed)
    other=fit(perturbed.with_pipeline_step(transform,name='center',preview_frame=transform.transform(perturbed.frame)))
    assert not other['error']
    np.testing.assert_allclose(result['model'].details_['real'],other['model'].details_['real'])


def test_all_tentative_policy_and_empty_export_status():
    source=example()
    result=fit(source,iterations=3,policy='tentative')
    assert not result['error']
    assert result['model'].selected_columns_ == result['model'].candidate_columns_
    pd.testing.assert_frame_equal(export_selection(result).frame,source.frame)
    result=fit(source,iterations=3,policy='confirmed')
    assert not result['error']
    assert not result['model'].selected_columns_
    with pytest.raises(ValueError,match='No variables'):
        export_selection(result)


@pytest.mark.parametrize('model_first',[False,True])
def test_downstream_model_uses_fitted_subset_not_preview_columns(model_first):
    # A model built using a preview containing signal alone must still accept
    # noise when a fresh training fit selects it instead (and vice versa).
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from model_commit import prepare_predictors
    source=example()
    result=fit(source)
    estimator=Pipeline([('predictors',FunctionTransformer(prepare_predictors,
        kw_args={'columns':['signal'],'preparation':'none'})),('estimator',Ridge())])
    if model_first:
        source=source.with_estimator(estimator,label='Ridge',recipe_id='test')
        result=fit(source)
        selected=export_selection(result)
    else:
        selected=export_selection(result).with_estimator(estimator,label='Ridge',recipe_id='test')
    y=source.clean_frame.noise.to_numpy()*5
    trained=selected.pipeline_for_training().fit(source.clean_frame,y)
    assert 'noise' in trained.named_steps.var_select.selected_columns_
    assert np.isfinite(trained.predict(source.clean_frame)).all()
    old=estimator.named_steps.predictors.kw_args
    assert old['columns']==['signal'] and 'selected_schema' not in old


def test_recorded_transformer_source_replays_with_explicit_namespace():
    from code_recording import recording_context
    from module import Module
    from sklearn.base import BaseEstimator,TransformerMixin
    from sklearn.utils.validation import check_is_fitted
    registry={}
    source=example(120)
    model=FeatureSelectionTransformer(('signal','noise'),trees=25,iterations=15)
    recording_context(registry,model.fit)(source.frame,source.frame.y)
    namespace=dict(np=np,pd=pd,BaseEstimator=BaseEstimator,TransformerMixin=TransformerMixin,
                   check_is_fitted=check_is_fitted)
    exec(Module.clean_code(registry['FeatureSelectionTransformer']),namespace)
    replay=namespace['FeatureSelectionTransformer'](('signal','noise'),trees=25,iterations=15)
    replay.fit(source.frame,source.frame.y)
    assert replay.selected_columns_==model.selected_columns_


def test_native_numeric_adapter_preserves_borutapy_decisions():
    from boruta import BorutaPy
    from lightgbm import LGBMRegressor
    from feature_selection import fit_boruta
    source=example()
    frame=source.frame[['signal','noise']]
    y=source.frame.y.to_numpy()
    native=fit_boruta(frame,y,weights=None,task='regression',trees=40,depth=4,
        iterations=18,alpha=.05,percentile=100,two_step=True,seed=2025,jobs=1)
    forest=LGBMRegressor(boosting_type='rf',n_estimators=40,max_depth=4,
        num_leaves=16,subsample=.7,subsample_freq=1,colsample_bytree=.7,
        min_child_samples=5,importance_type='gain',n_jobs=1,verbosity=-1,
        reg_alpha=0.,reg_lambda=0.)
    class ReferenceBoruta(BorutaPy):
        def _do_tests(self,decisions,hits,iteration):
            result=super()._do_tests(decisions,hits,iteration)
            self.terminal_decisions=result.copy()
            return result
    stock=ReferenceBoruta(forest,n_estimators=40,max_iter=18,perc=100,alpha=.05,
        two_step=True,random_state=2025).fit(frame.to_numpy(),y)
    # Compare terminal testing states, not upstream's post-rough-fix mask.
    np.testing.assert_array_equal(native['decisions'],stock.terminal_decisions)
    np.testing.assert_array_equal(native['ranking'],stock.ranking_)


@pytest.mark.asyncio
async def test_identity_and_worker_accept_upstream_provenance_and_preserve_it():
    from dataclasses import replace
    from sklearn.preprocessing import FunctionTransformer
    source=example()
    source=source.with_cleaned_data(source.frame.copy(),card='clean',operation='Record cleaning',
        parameters={'nested':{'columns':['signal']}})
    source=source.with_pipeline_step(FunctionTransformer(),name='upstream',preview_frame=source.frame.copy())
    assert source.processing_records and source.cleaning_records
    identity=selection_identity(source)
    assert identity==selection_identity(source.clone())
    last=source.processing_records[-1]
    changed=replace(source,_processing_records=(*source.processing_records[:-1],
        replace(last,parameters={'changed':True})))
    assert identity!=selection_identity(changed)
    result=await run_selection(source,'Boruta',options())
    assert not result['error']
    assert result['source'].processing_records==source.processing_records
    assert result['source'].cleaning_records==source.cleaning_records
    assert identity==selection_identity(result['source'])
    selected=export_selection(result)
    assert selected.pipeline_steps==('upstream','var_select')
    assert selected.processing_records[:-1]==source.processing_records
    assert selected.cleaning_records==source.cleaning_records
    with pytest.raises(TypeError):
        result['source'].processing_records[-1].parameters['changed']=True
