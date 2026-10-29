"""Approved placeholder rules replay without learning from each new batch."""
import pickle

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pipeline import Pipeline
from PlaceholderTransformer import PlaceholderTransformer, placeholder_rules
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import NotFittedError

pytestmark = pytest.mark.unit


def test_fixed_extrema_rules_survive_cloning_serialization_and_new_batches():
    review = pd.DataFrame({'x': [-99, 1, 2], 'interior': [-100, -99, 5]})
    rules = placeholder_rules(review, ['int: -99'])
    assert rules == [('x', 'int', '-99')]
    step = PlaceholderTransformer(rules)
    with pytest.raises(NotFittedError):
        step.transform(review)
    pipe = SklearnPipeline([('placeholders', clone(step))]).fit(review.iloc[1:])
    pipe = pickle.loads(pickle.dumps(pipe))
    future = pd.DataFrame({'x': [-100, -99, 5], 'interior': [-99, 2, 5]}, index=[8, 8, 9])
    before = future.copy()
    result = pipe.transform(future)
    assert result.x.isna().tolist() == [False, True, False]
    assert result.interior.tolist() == [-99, 2, 5]
    assert_frame_equal(future, before)
    assert result.index.equals(future.index)
    assert pipe.get_feature_names_out().tolist() == list(future)
    assert_frame_equal(pipe.transform(future.iloc[[1]]), result.iloc[[1]])


class CaptureEstimator(BaseEstimator):
    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        return self


@pytest.mark.parametrize('container', ['series', 'frame', 'array', 'matrix'])
@pytest.mark.parametrize('cached', [False, True])
def test_project_pipeline_transforms_target_during_fit_and_paired_transform(tmp_path, container, cached):
    X = pd.DataFrame({'x': ['NA', 'ok'], 'target': [-99, 1]}, index=[4, 4])
    y = X.target.copy()
    if container == 'frame':
        y = y.to_frame()
    elif container == 'array':
        y = y.to_numpy()
    elif container == 'matrix':
        y = y.to_numpy().reshape(-1, 1)
    step = PlaceholderTransformer([('x', 'str', 'NA'), ('target', 'int', '-99')],
                                  target_columns=['target'])
    pipe = clone(Pipeline([('placeholder', step), ('model', CaptureEstimator())],
                         memory=str(tmp_path) if cached else None)).fit(X, y)
    model = pipe.named_steps.model
    assert model.X_.isna().iloc[0].all()
    assert pd.isna(np.asarray(model.y_).ravel()).tolist() == [True, False]
    Xt, yt = pipe[:-1].transform_x_y(X, y)
    assert_frame_equal(Xt, model.X_)
    assert pd.isna(np.asarray(yt).ravel()).tolist() == [True, False]
    assert X.target.tolist() == [-99, 1]
    assert np.asarray(y).ravel().tolist() == [-99, 1]
    if container == 'series':
        assert_series_equal(yt, model.y_)
    assert_frame_equal(pipe[:-1].transform(X), Xt)


def test_separate_target_and_multiple_named_targets():
    X = pd.DataFrame({'x': [1, 2]})
    step = PlaceholderTransformer([('a', 'str', 'NA'), ('b', 'int', '-99')],
                                  target_columns=['a', 'b']).fit(X)
    y = pd.Series([-99, 1], name='b')
    _, result = step.transform_x_y(X, y)
    assert result.isna().tolist() == [True, False]
    assert result.name == 'b'
    assert_frame_equal(step.transform(X), X)
    with pytest.raises(ValueError, match='width'):
        step.transform_x_y(X, y.to_numpy())
    with pytest.raises(ValueError, match='same number'):
        step.transform_x_y(X, y.iloc[:1])


def test_extrema_approval_does_not_cascade_with_sentinel_order():
    X = pd.DataFrame({'x': [-999, -99, 1]})
    for values in [['int: -999', 'int: -99'], ['int: -99', 'int: -999']]:
        assert placeholder_rules(X, values) == [('x', 'int', '-999')]


def test_target_replacement_composes_with_missingness_removal():
    from MissingnessTransformer import MissingnessTransformer

    X = pd.DataFrame({'x': [1, 2, 3], 'target': [-99, 0, 1]}, index=[7, 7, 8])
    step = PlaceholderTransformer([('target', 'int', '-99')], target_columns=['target'])
    pipe = Pipeline([
        ('placeholder', step),
        ('missing', MissingnessTransformer(remove_observations=True, observation_threshold=0)),
        ('model', CaptureEstimator()),
    ]).fit(X, X.target)
    model = pipe.named_steps.model
    assert model.X_.index.tolist() == [7, 8]
    assert model.y_.tolist() == [0, 1]
    assert model.y_.index.tolist() == [7, 8]
