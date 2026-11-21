import numpy as np
import pandas as pd
import pytest
from cards.cand_methods import _analyze
from estimator_catalogue import installed_versions, load_catalogue
from model_commit import build_model
from pandas.testing import assert_frame_equal
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.preprocessing import FunctionTransformer

pytestmark = pytest.mark.unit


def fixture():
    frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'y': [2., 4., 6., 8.]})
    source = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    from evaluation_plan import build_plan
    source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=2), card='obs_resample')
    rows = _analyze(source, load_catalogue(), installed_versions())['rows']
    row = next(r for r in rows if r['estimator'] == 'sklearn.linear_model.Ridge'
               and r['profile']['construction']['preparation'] == 'impute_onehot')
    return source, row


def test_commit_clones_preserves_data_replaces_model_and_fits_without_target_leakage():
    source, row = fixture()
    source = source.with_pipeline_step(FunctionTransformer(), name='upstream', preview_frame=source.frame)
    model = build_model(row, source.model_predictors)
    committed = source.with_estimator(model, label='Linear Regression', recipe_id=row['id'])
    assert committed.pipeline_steps == ('upstream', 'cand_methods')
    assert_frame_equal(committed.frame, source.frame)
    assert_frame_equal(committed.clean_frame, source.clean_frame)
    assert not hasattr(committed.pipeline.named_steps.cand_methods.named_steps.estimator, 'coef_')
    trained = committed.pipeline_for_training().fit(source.clean_frame, source.frame.y)
    assert np.isfinite(trained.predict(source.clean_frame)).all()
    assert trained.named_steps.cand_methods.named_steps.estimator.n_features_in_ == 1
    again = committed.with_estimator(model, label='Linear Regression', recipe_id=row['id'])
    assert again.pipeline_steps == committed.pipeline_steps
    assert len([r for r in again.processing_records if r.card == 'cand_methods']) == 1
    assert again.processing_records[-1].operation == 'Committed method: Linear Regression'
    after = again.with_pipeline_step(FunctionTransformer(), name='later', preview_frame=source.frame)
    assert after.pipeline_steps == ('upstream', 'later', 'cand_methods')


def test_unvalidated_recipe_is_not_constructed():
    source, row = fixture()
    with pytest.raises(ValueError, match='not validated'):
        build_model(dict(row, selectable=False), source.model_predictors)


@pytest.mark.parametrize('name', ['CatBoostRegressor','CatBoostClassifier'])
def test_catboost_recipe_handles_categories_missing_values_and_cloning(name):
    from copy import deepcopy

    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.base import clone, is_classifier, is_regressor
    from sklearn.model_selection import KFold, cross_val_score
    catalogue=load_catalogue()
    profile=deepcopy(next(p for p in catalogue['profiles'] if p['estimator']=='catboost.'+name))
    profile['construction']['parameters'].update(iterations=8,thread_count=1)
    row={'selectable':True,'estimator':'catboost.'+name,'profile':profile}
    X=pd.DataFrame({'x':np.arange(24,dtype=float),
                    'cat':pd.Series(['a',None,'__pythagoras_missing__',1,'1',True]*4,dtype=object),
                    'flag':pd.Series([True,False,None]*8,dtype='boolean')})
    X.loc[::7,'x']=np.nan
    before=X.copy(deep=True)
    y=np.arange(24,dtype=float) if name.endswith('Regressor') else np.where(np.arange(24)%2,'yes','no')
    model=build_model(row,list(X))
    assert ('estimator__iterations' in model.get_params())
    assert is_classifier(model)==name.endswith('Classifier')
    assert is_regressor(model)==name.endswith('Regressor')
    trained=clone(model).fit(X,y,estimator__sample_weight=np.linspace(1,2,len(X)))
    estimator=trained.named_steps.estimator
    assert isinstance(estimator,(CatBoostClassifier,CatBoostRegressor))
    assert estimator.get_cat_feature_indices()==[1,2]
    prepared=trained.named_steps.predictors.transform(X)
    assert len(set(prepared.cat.iloc[:6]))==6  # No missing/string/type collisions.
    unseen=X.iloc[:3].copy();unseen.loc[0,'cat']='unseen'
    assert len(trained.predict(unseen))==3
    scores=cross_val_score(model,X,y,cv=KFold(3,shuffle=True,random_state=1729),error_score='raise')
    assert np.isfinite(scores).all()
    assert_frame_equal(X,before)
    assert not model.named_steps.estimator.is_fitted()


def test_catboost_category_names_follow_actual_fold_selection():
    from copy import deepcopy

    from model_commit import with_selection_schema
    profile=deepcopy(next(p for p in load_catalogue()['profiles'] if p['estimator']=='catboost.CatBoostRegressor'))
    profile['construction']['parameters'].update(iterations=5,thread_count=1)
    model=build_model({'selectable':True,'estimator':'catboost.CatBoostRegressor','profile':profile},['x','cat'])
    model=with_selection_schema(model,['x','cat'])
    X=pd.DataFrame({'x':np.arange(12,dtype=float)})
    model.fit(X,2*X.x)
    assert model.named_steps.estimator.get_cat_feature_indices()==[]
    assert np.isfinite(model.predict(X)).all()


def test_cubist_recipe_preserves_feature_names_during_prediction():
    import warnings

    pytest.importorskip('cubist')
    from sklearn.base import clone

    profile = next(p for p in load_catalogue()['profiles']
                   if p['id'] == 'cubist.Cubist:audited_tabular_recipe')
    row = {'id': profile['id'], 'estimator': profile['estimator'],
           'profile': profile, 'selectable': True}
    frame = pd.DataFrame({'x': np.arange(30, dtype=float),
                          'category': ['a', 'b', 'c'] * 10})
    target = 2 * frame.x + np.sin(frame.x)
    model = clone(build_model(row, list(frame.columns)))
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='X does not have valid feature names')
        model.fit(frame.iloc[:24], target.iloc[:24])
        predictions = model.predict(frame.iloc[24:])
    assert np.isfinite(predictions).all()
    prepared = model.named_steps['prepare'].transform(frame.iloc[24:])
    assert isinstance(prepared, pd.DataFrame)
    assert list(prepared.columns) == list(model.named_steps['estimator'].feature_names_in_)


@pytest.mark.parametrize('train', [[0, 1], [0, 2], [0, 2, 3]])
@pytest.mark.parametrize('multiple', [False, True])
def test_fold_nominal_schema_uses_fitted_indicators_and_preserves_roles(train, multiple):
    from NominalEncodingTransformer import NominalEncodingTransformer
    from roles import Role

    _, row = fixture()
    frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'Utilities': ['AllPub', 'AllPub', 'NoSeWa', 'Other'],
                          'Utilities__AllPub': [90., 91., 92., 93.], 'y': [2., 4., 6., 8.]})
    source = proxy_data(frame, _roles=RoleMap.from_primitive({
        'predictor': ['x', 'Utilities'], 'target': ['y'], 'stratifier': ['Utilities__AllPub']}))
    encoder = NominalEncodingTransformer(columns=['Utilities'])
    preview = encoder.fit_transform(frame)
    roles = RoleMap()
    for name in encoder.output_columns_:
        roles.set_roles(name, {Role.PREDICTOR})
    source = source.with_pipeline_step(encoder, name='var_encode', preview_frame=preview,
        added_roles=roles, removed_columns=encoder.removed_columns_)
    model = build_model(row, source.model_predictors)
    if multiple:
        source = source.with_model_candidates([
            {'id': str(i), 'label': f'Ridge {i}', 'estimator': model} for i in range(2)])
    else:
        source = source.with_estimator(model, label='Ridge', recipe_id='ridge')
    for candidate in (['0', '1'] if multiple else [None]):
        pipeline = source.pipeline_for_training(candidate)
        pipeline.fit(frame.iloc[train], frame.y.iloc[train])
        encoding = pipeline.named_steps.var_encode
        preparation = pipeline.named_steps.cand_methods.named_steps.predictors
        encoded = encoding.transform(frame)
        selected = preparation.transform(encoded)
        assert list(selected) == ['x', *encoding.output_columns_]
        assert 'Utilities__AllPub' not in selected  # Raw stratifier name collides with a possible indicator.
        assert 'y' not in selected
        assert np.isfinite(pipeline.predict(frame)).all()  # Includes levels unseen during fold fitting.
        assert not hasattr(source.pipeline.named_steps.var_encode, 'encodings_')
        # Refitting the same pipeline must not retain the previous fold's names.
        pipeline.fit(frame.iloc[:2], frame.y.iloc[:2])
        assert pipeline.named_steps.cand_methods.named_steps.estimator.n_features_in_ == 1
        assert np.isfinite(pipeline.predict(frame)).all()


def test_fold_encoding_keeps_missing_fixed_predictors_as_errors():
    from model_commit import prepare_predictors
    with pytest.raises(KeyError, match='missing'):
        prepare_predictors(pd.DataFrame({'x': [1]}), columns=['x', 'missing'], preparation='numeric')


@pytest.mark.parametrize('name', ['CatBoostRegressor', 'CatBoostClassifier', 'PreparedCatBoostRegressor', 'PreparedCatBoostClassifier'])
def test_recorded_catboost_none_defaults_remain_cloneable(name):
    import catboost
    import catboost_model
    from model_commit import configured_pipeline
    from model_review import CandidatePipelines
    from fit_service import fit_key
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    factory = getattr(catboost_model if name.startswith('Prepared') else catboost, name)
    template = Pipeline([('model', factory(depth=4, verbose=False, allow_writing_files=False))])
    # Search baselines record unset native parameters as None. Explicitly
    # resetting depth must use the constructor default rather than retain 4.
    parameters = {'model__iterations': None, 'model__depth': None}
    mapping = CandidatePipelines({'catboost': template}, {'0': ('catboost', parameters)})
    materialized = mapping['0']
    for model in (materialized, clone(materialized), configured_pipeline(template, parameters)):
        settings = model.named_steps['model'].get_params()
        assert 'iterations' not in settings
        assert 'depth' not in settings
        assert settings['verbose'] is False
        default = Pipeline([('model', factory(verbose=False, allow_writing_files=False))])
        assert fit_key('context', model, [0, 1], [2]) == fit_key('context', default, [0, 1], [2])
    assert template.named_steps['model'].get_params()['depth'] == 4
