"""Integration tests for composing learning cards through ProxyData."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cards import miss_impute, var_transform
from proxy_data import proxy_data


def _source() -> proxy_data:
    return proxy_data(pd.DataFrame({
        "skewed": [1.0, 2.0, np.nan, 4.0, 8.0, 16.0, 32.0, 64.0],
        "companion": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
        "category": pd.Categorical(
            ["a", "a", None, "b", "b", "a", "b", "a"],
        ),
    }))


def _imputation(data: proxy_data):
    return miss_impute._analyse(
        data, "simple", 3, 3, 1, 0.25, 11, 1,
    )


@pytest.mark.unit
def test_transform_then_impute_builds_one_trainable_pipeline():
    source = _source()
    transformed = var_transform._apply_analysis(
        source,
        var_transform._analyse_distribution(source, ["Scale"]),
    )
    result = miss_impute._apply_analysis(transformed, _imputation(transformed))

    assert result.pipeline_steps == ("var_transform", "miss_impute")
    pd.testing.assert_frame_equal(result.clean_frame, source.frame)
    assert result.frame.isna().sum().sum() == 0

    fitted = result.pipeline_for_training().fit(result.clean_frame.iloc[:6])
    held_out = fitted.transform(result.clean_frame.iloc[6:])
    assert isinstance(held_out, pd.DataFrame)
    assert list(held_out.columns) == list(source.columns)


@pytest.mark.unit
def test_impute_then_transform_builds_one_trainable_pipeline():
    source = _source()
    imputed = miss_impute._apply_analysis(source, _imputation(source))
    result = var_transform._apply_analysis(
        imputed,
        var_transform._analyse_distribution(imputed, ["Scale", "Center"]),
    )

    assert result.pipeline_steps == ("miss_impute", "var_transform")
    pd.testing.assert_frame_equal(result.clean_frame, source.frame)
    assert result.frame.isna().sum().sum() == 0

    fitted = result.pipeline_for_training().fit(result.clean_frame.iloc[:6])
    held_out = fitted.transform(result.clean_frame.iloc[6:])
    assert isinstance(held_out, pd.DataFrame)
    assert list(held_out.columns) == list(source.columns)
