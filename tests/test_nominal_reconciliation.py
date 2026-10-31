"""Fixed, reviewed mappings replay in cloned training pipelines."""
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
@pytest.mark.parametrize('with_pipeline', [False, True])
def test_targets_excluded_from_restored_mappings_and_stratifiers_replayed(with_pipeline):
    X = pd.DataFrame({c: pd.Categorical(['a', 'b']) for c in ['x', 'target', 'stratum', 'dual']})
    roles = RoleMap()
    for column, role in [('x', Role.PREDICTOR), ('target', Role.TARGET), ('stratum', Role.STRATIFIER)]:
        roles.set_roles(column, [role])
    roles.set_roles('dual', [Role.PREDICTOR, Role.TARGET])
    source = proxy_data(_df=X, _roles=roles)
    if with_pipeline:
        upstream = NominalReconciliationTransformer([])
        source = source.with_pipeline_step(upstream, name='upstream', preview_frame=upstream.fit_transform(X))
    assert card.eligible_columns(source) == ['x']
    assert card.eligible_columns(source, True) == ['x', 'stratum']
    state = card.fresh_state(card.source_fingerprint(source))
    state['batches'] = [sum((batch(c, 'a', 'b') for c in X), [])]
    state['manual'] = batch('target', 'a', 'b')
    target_item = state['manual'][0]
    state['ledger'] = [dict(target_item, decision='accept')]
    state['decisions'] = {target_item['key']: 'accept'}
    state['expected'] = {'target': {'levels': ['b']}, 'x': {'levels': ['b']}}
    cleaned_state = card.without_target_decisions(source, state)
    assert not cleaned_state['manual'] and not cleaned_state['ledger'] and not cleaned_state['decisions']
    assert set(cleaned_state['expected']) == {'x'}
    assert state['manual']  # State sanitization does not mutate the saved object.
    result = card.export_reconciliation(source, state)
    assert_frame_equal(result.clean_frame, X)
    assert result.frame.target.tolist() == ['a', 'b']
    assert result.frame.dual.tolist() == ['a', 'b']
    assert result.frame.x.tolist() == ['b', 'b']
    assert result.frame.stratum.tolist() == ['b', 'b']
    assert set(result._pipeline.steps[-1][1].columns) == {'x', 'stratum'}
    assert_frame_equal(card.committed_frame(source, state), result.frame)
    assert_frame_equal(result.pipeline_for_training().fit_transform(result.clean_frame), result.frame)
    target_only = card.fresh_state()
    target_only['batches'] = [batch('target', 'a', 'b')]
    assert card.export_reconciliation(source, target_only) is source
    assert_frame_equal(card.committed_frame(source, target_only), source.frame)
