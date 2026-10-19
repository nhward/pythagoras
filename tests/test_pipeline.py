"""Compatibility of mixed resampling and row-removing pipelines."""

import pandas as pd
import pytest
from feature_engine.imputation import DropMissingData
from feature_engine.pipeline import Pipeline as FeatureEnginePipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@pytest.mark.unit
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("sample_first", [False, True])
def test_mixed_pipeline_training_and_scoring(tmp_path, cached, sample_first):
    X = pd.DataFrame({"x": [0., 1., 2., 3., 4., 5., 6., float("nan")]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1])
    steps = [("drop", DropMissingData()),
             ("sample", RandomOverSampler(random_state=1))]
    if sample_first:
        steps.reverse()
    pipe = Pipeline(steps + [("scale", StandardScaler()),
                             ("model", LogisticRegression())],
                    memory=str(tmp_path) if cached else None)
    pipe = clone(pipe)
    assert isinstance(pipe, (ImbalancedPipeline, FeatureEnginePipeline))
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (7,)
    assert pipe.score(X, y) == pipe.named_steps.model.score(
        pipe.named_steps.scale.transform(X.dropna()), y[X.x.notna()]
    )
    assert pipe.get_params()["sample__random_state"] == 1


@pytest.mark.unit
def test_joint_transform_skips_sampler():
    X = pd.DataFrame({"x": [0., 1., 2., 3., float("nan")]})
    y = pd.Series([0, 0, 0, 1, 1])
    pipe = Pipeline([("drop", DropMissingData()),
                     ("sample", RandomOverSampler(random_state=1)),
                     ("end", "passthrough")])
    assert len(pipe.fit_transform(X, y)) == 6
    Xt, yt = pipe.transform_x_y(X, y)
    pd.testing.assert_frame_equal(Xt, X.dropna())
    pd.testing.assert_series_equal(yt, y[X.x.notna()])


@pytest.mark.unit
def test_final_sampler():
    X = pd.DataFrame({"x": [0., 1., 2., 3., float("nan")]})
    y = pd.Series([0, 0, 0, 1, 1])
    pipe = Pipeline([("noop", None), ("drop", DropMissingData()),
                     ("sample", RandomOverSampler(random_state=1))])
    Xt, yt = pipe.fit_resample(X, y)
    assert len(Xt) == len(yt) == 6
    assert yt.value_counts().tolist() == [3, 3]


@pytest.mark.unit
@pytest.mark.parametrize("estimator", ["classifier", "regressor", LogisticRegression()])
def test_skrub_factory_preserves_preprocessing_and_predicts(estimator, tmp_path):
    from pipeline import tabular_pipeline
    from skrub import tabular_pipeline as skrub_tabular_pipeline

    X = pd.DataFrame({
        "number": [1., 2., 3., 4., 5., float("nan")],
        "category": ["a", "b", "a", "b", "a", None],
    })
    y = [0, 0, 0, 1, 1, 1]
    expected = skrub_tabular_pipeline(estimator, n_jobs=1)
    actual = tabular_pipeline(estimator, n_jobs=1, memory=str(tmp_path))
    assert type(actual) is Pipeline
    assert [(n, type(s)) for n, s in actual.steps] == [
        (n, type(s)) for n, s in expected.steps
    ]
    assert actual.named_steps.tablevectorizer.n_jobs == 1
    actual = clone(actual)
    actual.fit(X, y)
    assert actual.predict(X).shape == (6,)
    assert actual.memory == str(tmp_path)
    # Unknown categories at inference must be handled by fitted preprocessing.
    assert actual.predict(pd.DataFrame({"number": [7.], "category": ["new"]})).shape == (1,)


@pytest.mark.unit
def test_skrub_preprocessing_with_row_filter_and_sampler():
    from MissingnessTransformer import MissingnessTransformer

    X = pd.DataFrame({
        "number": [1., 2., 3., 4., 5., 6., float("nan")],
        "category": ["a", "b", "a", "b", "a", "b", "a"],
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1])
    # Skrub accepts our subclass as a pipeline and flattens its supplied steps.
    pipe = Pipeline.from_tabular(Pipeline([
        ("sample", RandomOverSampler(random_state=1)),
        ("model", LogisticRegression()),
    ]), n_jobs=1)
    pipe.steps.insert(0, ("drop", MissingnessTransformer(
        remove_observations=True, observation_threshold=0,
    )))
    pipe = clone(pipe).fit(X, y)
    sampler = next(step for _, step in pipe.steps if isinstance(step, RandomOverSampler))
    assert len(sampler.sample_indices_) == 8
    assert pipe.predict(X).shape == (6,)
    assert 0 <= pipe.score(X, y) <= 1
