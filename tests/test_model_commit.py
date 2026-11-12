import numpy as np
import pandas as pd
import pytest
from cards.data_models import _analyze
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
    assert committed.pipeline_steps == ('upstream', 'data_models')
    assert_frame_equal(committed.frame, source.frame)
    assert_frame_equal(committed.clean_frame, source.clean_frame)
    assert not hasattr(committed.pipeline.named_steps.data_models.named_steps.estimator, 'coef_')
    trained = committed.pipeline_for_training().fit(source.clean_frame, source.frame.y)
    assert np.isfinite(trained.predict(source.clean_frame)).all()
    assert trained.named_steps.data_models.named_steps.estimator.n_features_in_ == 1
    again = committed.with_estimator(model, label='Linear Regression', recipe_id=row['id'])
    assert again.pipeline_steps == committed.pipeline_steps
    assert len([r for r in again.processing_records if r.card == 'data_models']) == 1
    assert again.processing_records[-1].operation == 'Committed method: Linear Regression'
    after = again.with_pipeline_step(FunctionTransformer(), name='later', preview_frame=source.frame)
    assert after.pipeline_steps == ('upstream', 'later', 'data_models')


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
