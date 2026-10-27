"""Fixed, reviewed mappings replay in cloned training pipelines."""
import functools
import operator
import pickle

import pandas as pd
import pytest
from cards import var_nominal as card
from NominalReconciliationTransformer import NominalReconciliationTransformer
from pandas.testing import assert_frame_equal
from proxy_data import proxy_data
from roles import Role, RoleMap
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError


def batch(column, source, destination):
    return [card.proposal(column, [source], destination, 'reviewed')]


@pytest.mark.unit
def test_clone_and_serialization_replay_absent_labels_and_sequential_batches():
    transformer = NominalReconciliationTransformer([
        batch('x', 'a', 'b'), batch('x', 'b', 'c')])
    train = pd.DataFrame({'x': pd.Categorical(['c', None]), 'n': [1, 2]})
    pipeline = Pipeline([('reconcile', clone(transformer))]).fit(train)
    future = pd.DataFrame({'x': pd.Categorical(['a', 'new', None]), 'n': [3, 4, 5]}, index=[8, 8, 9])
    before = future.copy(deep=True)
    result = pickle.loads(pickle.dumps(pipeline)).transform(future)
    assert result.x.iloc[:2].tolist() == ['c', 'new']
    assert pd.isna(result.x.iloc[2])
    assert result.index.equals(future.index)
    assert isinstance(result.x.dtype, pd.CategoricalDtype)
    assert_frame_equal(future, before)
    assert pipeline.get_feature_names_out().tolist() == ['x', 'n']
    with pytest.raises(NotFittedError):
        transformer.transform(future)


@pytest.mark.unit
def test_cycles_and_conflicts_validated_even_without_observed_sources():
    X = pd.DataFrame({'x': pd.Categorical(['z'])})
    for items, message in [
        (batch('x', 'a', 'b') + batch('x', 'b', 'a'), 'cycle'),
        (batch('x', 'a', 'b') + batch('x', 'a', 'c'), 'Conflicting'),
    ]:
        with pytest.raises(ValueError, match=message):
            NominalReconciliationTransformer([items]).fit(X)


@pytest.mark.unit
def test_string_inputs_missing_column_and_schema_validation():
    X = pd.DataFrame({'x': ['a', None], 'n': [1, 2]})
    fitted = NominalReconciliationTransformer([batch('x', 'a', 'b')]).fit(X)
    assert fitted.transform(X).x.iloc[0] == 'b'
    with pytest.raises(ValueError, match='Missing nominal'):
        fitted.transform(X[['n']])
    with pytest.raises(ValueError, match='schema'):
        fitted.transform(X[['n', 'x']])


@pytest.mark.unit
def test_export_appends_once_preserves_upstream_and_clean_data():
    X = pd.DataFrame({'x': pd.Categorical(['a', 'z', None])})
    source = proxy_data(X)
    upstream = NominalReconciliationTransformer([batch('x', 'a', 'b')])
    source = source.with_pipeline_step(upstream, name='upstream', preview_frame=upstream.fit_transform(X))
    state = card.fresh_state()
    state['batches'] = [batch('x', 'b', 'c'), batch('x', 'c', 'd')]
    result = card.export_reconciliation(source, state)
    assert result.pipeline_steps == ('upstream', 'var_nominal')
    assert_frame_equal(result.clean_frame, X)
    assert_frame_equal(result.pipeline_for_training().fit_transform(X), result.frame)
    assert card.export_reconciliation(source, state).pipeline_steps == result.pipeline_steps
    assert card.export_reconciliation(source, card.fresh_state()) is source


@pytest.mark.unit
def test_targets_cleaned_separately_and_stratifiers_replayed():
    X = pd.DataFrame({c: pd.Categorical(['a', 'b']) for c in ['x', 'target', 'stratum']})
    roles = RoleMap()
    for column, role in [('x', Role.PREDICTOR), ('target', Role.TARGET), ('stratum', Role.STRATIFIER)]:
        roles.set_roles(column, [role])
    source = proxy_data(_df=X, _roles=roles)
    state = card.fresh_state()
    state['batches'] = [functools.reduce(operator.iadd, (batch(c, 'a', 'b') for c in X), [])]
    result = card.export_reconciliation(source, state)
    assert result.clean_frame.target.tolist() == ['b', 'b']
    assert result.clean_frame.x.tolist() == ['a', 'b']
    assert result.clean_frame.stratum.tolist() == ['a', 'b']
    assert set(result._pipeline.steps[-1][1].columns) == {'x', 'stratum'}
    assert_frame_equal(result.pipeline_for_training().fit_transform(result.clean_frame), result.frame)
    with pytest.raises(ValueError, match='Target-label'):
        card.export_reconciliation(result, state)
