"""Missingness recipes refit inside folds and preserve paired row alignment."""

import numpy as np
import pandas as pd
import pytest
from cards import miss_map, var_transform
from feature_engine.imputation import DropMissingData
from MissingnessTransformer import MissingnessTransformer
from pipeline import Pipeline
from proxy_data import proxy_data
from sklearn.base import clone
from sklearn.dummy import DummyClassifier


@pytest.mark.unit
@pytest.mark.parametrize("width", [3, 50])
def test_all_slider_boundaries_match_strict_missingness(width):
    X = pd.DataFrame([
        [np.nan] * missing + [1.] * (width - missing)
        for missing in range(width + 1)
    ], columns=[f"x{i}" for i in range(width)])
    for percentage in range(101):
        threshold = percentage / 100
        transformer = MissingnessTransformer(
            remove_observations=True, observation_threshold=threshold,
        )
        result = transformer.fit_transform(X)
        pd.testing.assert_frame_equal(result, X.loc[X.isna().mean(axis=1) <= threshold])
        if threshold < 1:
            assert isinstance(transformer.row_filter_, DropMissingData)


@pytest.mark.unit
def test_column_selection_is_fitted_on_training_rows_only():
    X = pd.DataFrame({"a": [1., 2., np.nan, np.nan, np.nan], "b": [1., 2., 3., 4., 5.]})
    result = miss_map._remove_excessive_variables(proxy_data(X), 0.5)
    assert list(result.frame) == ["b"]
    pipe = result.pipeline_for_training()
    assert not hasattr(pipe.steps[0][1], "removed_columns_")
    pipe.fit(X.iloc[:2])
    assert list(pipe.transform(X.iloc[2:])) == ["a", "b"]
    pd.testing.assert_frame_equal(result.clean_frame, X)


@pytest.mark.unit
@pytest.mark.parametrize("target_kind", ["series", "array", "none"])
def test_duplicate_indices_keep_targets_aligned(target_kind):
    X = pd.DataFrame({"x": [1., np.nan, 3., 4.]}, index=[7, 7, 2, 7])
    y = pd.Series([10, 20, 30, 40], index=X.index)
    if target_kind == "array":
        y = y.to_numpy()
    elif target_kind == "none":
        y = None
    transformer = MissingnessTransformer(remove_observations=True).fit(X)
    Xt, yt = transformer.transform_training_x_y(X, y)
    pd.testing.assert_frame_equal(Xt, X.iloc[[0, 2, 3]])
    if y is None:
        assert yt is None
    else:
        assert np.asarray(yt).tolist() == [10, 30, 40]


@pytest.mark.unit
def test_miss_map_after_learning_and_before_an_estimator():
    X = pd.DataFrame({"x": [1., np.nan, 3., 4.]})
    source = proxy_data(X)
    upstream = var_transform._apply_analysis(
        source, var_transform._analyse_distribution(source, ["Scale"]),
    )
    result = miss_map._remove_excessive_observations(upstream, 0.5)
    assert result.pipeline_steps == ("var_transform", "miss_map")
    pd.testing.assert_frame_equal(result.clean_frame, X)
    fitted = result.pipeline_for_training().fit(X)
    pd.testing.assert_frame_equal(fitted.transform(X), upstream.frame)
    pipe = Pipeline([*result.pipeline.steps, ("model", DummyClassifier())])
    pipe.fit(X, pd.Series([0, 1, 0, 1]))
    assert pipe.named_steps.model.class_prior_ == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.unit
def test_empty_columns_and_empty_rows():
    X = pd.DataFrame({"x": [np.nan, np.nan]}, index=[4, 8])
    result = miss_map._transform_data(
        proxy_data(X), remove_variables=True, remove_observations=True,
        variable_threshold=0.5, observation_threshold=0.5,
    )
    assert result.frame.shape == (2, 0)
    assert miss_map._issues_table(result.frame, result.role_map, 0.5, 0.5).empty
    row_filter = MissingnessTransformer(remove_observations=True).fit(X)
    pd.testing.assert_frame_equal(row_filter.transform(X), X)
    assert row_filter.transform(X.iloc[:0]).empty
    assert clone(row_filter).fit_transform(X.iloc[:0]).empty


@pytest.mark.unit
def test_row_removal_requires_explicit_permission_and_ordered_subset():
    X = pd.DataFrame({"x": [1., np.nan, 3.]}, index=[5, 8, 2])
    source = proxy_data(X)
    transformer = MissingnessTransformer(remove_observations=True)
    preview = transformer.fit_transform(X)
    with pytest.raises(ValueError, match="preserve the DataFrame index"):
        source.with_pipeline_step(transformer, name="test", preview_frame=preview)
    with pytest.raises(ValueError, match="source row order"):
        source.with_pipeline_step(transformer, name="test", preview_frame=preview.iloc[::-1], allow_row_removal=True)
    result = source.with_pipeline_step(transformer, name="test", preview_frame=preview, allow_row_removal=True)
    assert len(result.frame) == 2


@pytest.mark.unit
@pytest.mark.parametrize('threshold', [0, .5, 1])
@pytest.mark.parametrize('cached', [False, True])
def test_training_filter_preserves_every_test_row_and_score_target(threshold, cached, tmp_path):
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    X = pd.DataFrame({'x': [1., 2., np.nan, 4.],
                      'drop': [np.nan, np.nan, np.nan, 1.]})
    y = pd.Series([0, 0, 1, 1])
    pipe = Pipeline([
        ('missing', MissingnessTransformer(remove_variables=True,
            remove_observations=True, observation_threshold=threshold)),
        ('impute', SimpleImputer()),
        ('model', DummyClassifier(strategy='most_frequent')),
    ], memory=str(tmp_path) if cached else None).fit(X, y)
    expected_prior = [2/3, 1/3] if threshold < 1 else [.5, .5]
    assert pipe.named_steps.model.class_prior_ == pytest.approx(expected_prior)
    test = pd.DataFrame({'x': [np.nan, 6., np.nan], 'drop': [9., 9., 9.]}, index=[7, 7, 2])
    target = pd.Series([1, 0, 1], index=test.index)
    filtered, paired = pipe[:1].transform_x_y(test, target)
    pd.testing.assert_frame_equal(filtered, test[['x']])
    pd.testing.assert_series_equal(paired, target)
    predictions = pipe.predict(test)
    assert len(predictions) == len(test)
    assert pipe.score(test, target) == accuracy_score(target, predictions)
