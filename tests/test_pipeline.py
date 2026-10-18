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
