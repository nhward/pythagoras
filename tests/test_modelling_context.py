import numpy as np
import pandas as pd
import pytest
from evaluation_plan import build_plan
from modelling_context import ModellingContext
from proxy_data import proxy_data
from roles import RoleMap

pytestmark = pytest.mark.unit


def survival_source():
    frame = pd.DataFrame({'x': np.arange(24), 'duration': np.arange(24) + 1.,
                          'event': [1, 0] * 12, 'subject': np.arange(24)})
    return proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': ['x', 'duration', 'event'],
        'identifier': ['subject'], 'stratifier': ['event']})).with_modelling_context(
            ModellingContext(task='survival', duration='duration', event='event'))


def test_survival_context_and_memberships():
    data = survival_source()
    plan = build_plan(data, goal='observations', folds=3)
    assert plan.ready and plan.supervised
    assert plan.recipe['task'] == 'survival'
    assert data.model_predictors == ['x']
    assert data.clone().modelling_context == data.modelling_context
    assert 'Train events' in plan.diagnostics
    for train, validation in plan.splits(data):
        assert not set(train) & set(validation)
        outcome = data.modelling_context.survival_outcome(data, train)
        assert outcome.dtype.names == ('event', 'time')
        assert outcome['event'].any()
    configured = data.with_evaluation_plan(plan, card='obs_resample')
    changed = configured.with_modelling_context(ModellingContext(task='regression'))
    assert not changed.evaluation_plan.ready
    assert not changed.equals(configured)


@pytest.mark.parametrize('change, message', [
    ({'duration': -1}, 'nonnegative'), ({'event': 2}, 'codes'),
    ({'event': np.nan}, 'codes'), ({'duration': np.inf}, 'finite')])
def test_invalid_survival_outcomes(change, message):
    data = survival_source()
    for column, value in change.items():
        data.clean_frame.loc[0, column] = value
    plan = build_plan(data, goal='observations', stratifier=None, folds=3)
    assert not plan.ready and message in plan.errors[0]


def test_unsupported_designs_and_eventless_folds():
    data = survival_source()
    assert 'administrative censoring' in build_plan(data, strategy='temporal', objective='future').errors[0]
    data.clean_frame['event'] = 0
    assert 'no observed events' in build_plan(data, goal='observations', folds=3).errors[0]
    data.clean_frame['subject'] = 1
    assert 'repeated records' in build_plan(data, strategy='kfold', objective='independent').errors[0]


def test_outcome_inference():
    data = proxy_data(pd.DataFrame({'y': [1., 2.]}), _roles=RoleMap.from_primitive({'target': ['y']}))
    assert data.modelling_context.resolved_task(data) == 'regression'
    data.clean_frame['y'] = pd.Categorical(['low', 'high'], categories=['low', 'high'], ordered=True)
    assert data.modelling_context.resolved_task(data) == 'ordinal'
    assert ModellingContext(task='classification').resolved_task(data) == 'classification'


def test_unknown_prediction_outcomes_are_allowed_and_context_survives_sampling():
    data = survival_source()
    frame = data.clean_frame.copy()
    frame['partition'] = ['development'] * 20 + ['prediction'] * 4
    frame.loc[20:, ['duration', 'event']] = np.nan
    data = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': ['x'],
        'partition': ['partition']}), _modelling_context=data.modelling_context)
    plan = build_plan(data, goal='observations', folds=2, partition_map={
        'str:development': 'development', 'str:prediction': 'prediction'})
    assert plan.ready
    assert plan.prediction == (20, 21, 22, 23)
    assert data.sample(n=10).modelling_context == data.modelling_context
