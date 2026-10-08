import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import numpy as np
import pandas as pd
import pytest
from card import Card
from proxy_data import proxy_data
from roles import Role, RoleMap, weighting_purpose
from weighting import ExcessiveZeroWeightsError, WeightingError

pytestmark = pytest.mark.unit


def source(frame):
    return proxy_data(_df=frame, _roles=RoleMap.from_primitive({"weighting": list(frame)}))


def test_product_purpose_and_duplicate_index():
    frame = pd.DataFrame({"importance": [2., 3.], "other": [4., 5.],
                          "balance__target": [0.5, 2.]}, index=[7, 7])
    data = source(frame)
    weights = Card.effective_weights(data)
    assert weights.tolist() == [4., 30.]
    assert weights.index.equals(frame.index)
    assert Card.effective_weights(data, purpose="test").tolist() == [8., 15.]
    assert Card.effective_weights(data, purpose="test", test_policy="uniform") is None
    assert weighting_purpose("balance__target") == "balance"
    assert weighting_purpose("target") == "importance"
    pd.testing.assert_frame_equal(data.frame, frame)


def test_missing_training_only_column_is_not_required_for_importance_test_weights():
    frame = pd.DataFrame({"importance": [2., 3.]})
    columns = ("importance", "balance__target")
    assert Card.effective_weights(frame, columns, purpose="test").tolist() == [2., 3.]
    with pytest.raises(WeightingError, match="absent"):
        Card.effective_weights(frame, columns)
    assert Card.effective_weights(frame, ()) is None


def test_exact_ten_percent_and_disjoint_components():
    frame = pd.DataFrame({"a": [0.] + [1.] * 9, "b": [1.] * 10})
    assert Card.effective_weights(source(frame)).iloc[0] == 0
    frame.loc[1, "b"] = 0
    with pytest.raises(ExcessiveZeroWeightsError, match="2/10"):
        Card.effective_weights(source(frame))
    # The policy is applied to the actual subset supplied by the caller.
    with pytest.raises(ExcessiveZeroWeightsError):
        Card.effective_weights(frame.iloc[:5], ("a",))


@pytest.mark.parametrize("values", [[-1., 1.], [np.nan, 1.], [np.inf, 1.],
                                     [True, False], [1j, 2j], ["1", "2"]])
def test_invalid_components(values):
    with pytest.raises(WeightingError):
        Card.effective_weights(source(pd.DataFrame({"bad": values})))


@pytest.mark.parametrize("value", [1e300, 1e-300])
def test_product_overflow_and_underflow(value):
    with pytest.raises(WeightingError, match="overflow or underflow"):
        Card.effective_weights(source(pd.DataFrame({"a": [value], "b": [value]})))


def test_multiple_weighting_roles_validate_and_roundtrip():
    data = source(pd.DataFrame({"a": [1., 2.], "balance__b": [2., 1.]}))
    data.frame["predictor"] = [3., 4.]
    data.role_map.set_roles("predictor", [Role.PREDICTOR])
    assert not data.validate()
    assert RoleMap.from_primitive(data.role_map.to_primitive()) == data.role_map


def test_consumers_and_saved_transformers_use_the_product():
    from cards import var_correlation, var_transform, obs_clusters
    from ClusterMembershipTransformer import ClusterMembershipTransformer
    from sklearn.base import clone
    frame = pd.DataFrame({"x": np.arange(20.) ** 2, "y": np.sin(np.arange(20.)),
                          "a": np.linspace(1., 2., 20), "balance__b": np.linspace(2., 4., 20)})
    roles = RoleMap.from_primitive({"predictor": ["x", "y"], "weighting": ["a", "balance__b"]})
    data = proxy_data(_df=frame, _roles=roles)
    combined = frame.a * frame.balance__b
    single_frame = frame.drop(columns=["a", "balance__b"]).assign(weight=combined)
    single_roles = RoleMap.from_primitive({"predictor": ["x", "y"], "weighting": ["weight"]})
    single = proxy_data(_df=single_frame, _roles=single_roles)
    _, weights, _ = var_correlation._analysis_frame(data, ["x", "y"], "pearson", 100)
    np.testing.assert_allclose(weights, combined)
    multiple_result = var_transform._analyse_distribution(data, ["Center", "Scale"], use_weights=True)
    single_result = var_transform._analyse_distribution(single, ["Center", "Scale"], use_weights=True)
    np.testing.assert_allclose(multiple_result.frame[["x", "y"]], single_result.frame[["x", "y"]])
    fitted = clone(multiple_result.transformer).fit(frame.iloc[:10])
    # A fitted preprocessing step can transform Test rows without training weights.
    fitted.transform(frame.iloc[10:].drop(columns=["a", "balance__b"]))
    prepared = obs_clusters._prepare(data, limit=100, standardize=True, use_weights=True)
    expected = obs_clusters._prepare(single, limit=100, standardize=True, use_weights=True)
    np.testing.assert_allclose(prepared[0], expected[0])
    np.testing.assert_allclose(prepared[2], expected[2])
    model = ClusterMembershipTransformer(columns=("x", "y"), weighting=("a", "balance__b"), n_clusters=2)
    clone(model).fit(frame).transform(frame.drop(columns=["a", "balance__b"]))


@pytest.mark.parametrize('values', [[-1., 1.], [np.nan, 1.], [np.inf, 1.], [0., 1.], [0., 0.]])
def test_optional_weighting_falls_back_and_disabled_skips_validation(values):
    from weighting import WeightingFallbackWarning
    data = source(pd.DataFrame({'weight': values}))
    with pytest.warns(WeightingFallbackWarning, match='Using unweighted'):
        assert Card.observation_weights(data) is None
    assert Card.observation_weights(data, enabled=False) is None
    with pytest.raises(WeightingError):
        Card.effective_weights(data)


def test_optional_weighting_preserves_valid_product():
    data = source(pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}))
    pd.testing.assert_series_equal(Card.observation_weights(data), Card.effective_weights(data))


@pytest.mark.parametrize('name', [
    'miss_informative', 'miss_type', 'var_correlation', 'var_transform',
    'data_coverage', 'obs_clusters', 'obs_k_clusters', 'obs_cluster_profile', 'targ_balance',
])
def test_weight_consumers_expose_setting_and_fallback_notice(name):
    from importlib import import_module
    from shiny import reactive
    card = import_module(f'cards.{name}').instance()
    with reactive.isolate():
        settings = str(card.settings)
    assert 'UseWeights' in settings
    assert 'Use observation weightings' in settings
    assert 'WeightingNotice' in settings
