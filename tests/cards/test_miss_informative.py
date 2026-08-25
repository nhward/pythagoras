from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent.parent / 'app'
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import pytest
from card import Card
from shiny import reactive
from shinywidgets._serialization import json_packer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def miss_informative():
    return importlib.import_module("cards.miss_informative")


@pytest.fixture
def serial_permutation_importance(miss_informative, monkeypatch):
    """Keep unit tests deterministic and avoid spawning joblib worker processes."""
    original = miss_informative.permutation_importance

    def serial(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(miss_informative, "permutation_importance", serial)


def informative_classification_frame(rows: int = 80) -> pd.DataFrame:
    target = np.tile([0, 1], rows // 2)
    rng = np.random.default_rng(17)
    order = rng.permutation(rows)
    target = target[order]
    return pd.DataFrame({
        # Median imputation makes x constant; only its shadow retains the signal.
        "x": np.where(target == 1, np.nan, 0.0),
        "noise": rng.normal(size=rows),
        "target": target,
    })


def informative_regression_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    missing = np.tile([False, True], rows // 2)
    rng.shuffle(missing)
    return pd.DataFrame({
        "x": np.where(missing, np.nan, 0.0),
        "noise": rng.normal(size=rows),
        "target": missing.astype(float) * 10 + rng.normal(scale=0.05, size=rows),
    })


def importance_table(miss_informative) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Variable": "ordinary",
            "Variable Type": "Predictor",
            "Source Variable": None,
            "Missing Proportion": np.nan,
            "Importance": 0.10,
            "Importance SD": 0.02,
            "Positive Fraction": 0.8,
            "Interpretation": "Uninformative",
        },
        {
            "Variable": f"{Card.SHADOW_PREFIX}x",
            "Variable Type": "Shadow",
            "Source Variable": "x",
            "Missing Proportion": 0.25,
            "Importance": 0.20,
            "Importance SD": 0.03,
            "Positive Fraction": 1.0,
            "Interpretation": "Informative",
        },
    ], columns=miss_informative.IMPORTANCE_COLUMNS)


class TestCardDefinition:
    @pytest.mark.unit
    def test_metadata_and_regions(self, miss_informative):
        card = miss_informative.this
        assert card.name == "miss_informative"
        assert card.long_name == "Informative Missingness"
        assert card.mutable
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_expected_outputs_and_controls_are_present(self, miss_informative):
        front = str(miss_informative.this.front.tagify())
        back = str(miss_informative.this.back.tagify())
        footer = str(miss_informative.this.footer.tagify())
        settings = str(miss_informative.this.settings)
        assert 'id="Importance"' in front
        assert 'id="Table"' in back
        assert 'id="Significance"' in footer
        assert 'id="Shadow"' in footer
        for control in ("CVFolds", "MinMissProp", "MinBalancedAccuracy", "MaxObs"):
            assert f'id="{control}"' in settings

    @pytest.mark.unit
    def test_test_mode_seeds_expected_frame(self, miss_informative):
        with reactive.isolate():
            proxy = miss_informative.this._imports.get()
        frame = proxy.to_native()
        assert frame.shape == (8, 3)
        assert frame.columns.tolist() == ["age", "group", "income"]
        assert frame["income"].isna().sum() == 3


class TestPreparation:
    @pytest.mark.unit
    def test_empty_analysis_has_stable_schema(self, miss_informative):
        analysis = miss_informative._empty_analysis(None, "No target")
        assert analysis.target is None
        assert analysis.task is None
        assert analysis.message == "No target"
        assert analysis.folds == 0
        assert analysis.importance.columns.tolist() == miss_informative.IMPORTANCE_COLUMNS

    @pytest.mark.unit
    def test_feature_frame_classifies_and_normalises_mixed_types(
        self, miss_informative
    ):
        frame = pd.DataFrame({
            "numeric": [1.0, np.inf, np.nan],
            "datetime": pd.to_datetime(["2025-01-01", None, "2025-01-03"]),
            "boolean": pd.Series([True, False, pd.NA], dtype="boolean"),
            "category": pd.Series(["A", "B", None], dtype="category"),
        }, index=[3, 5, 7])
        features, numeric, categorical = miss_informative._feature_frame(
            frame, frame.columns.tolist()
        )
        assert features.index.tolist() == [3, 5, 7]
        assert numeric == ["numeric", "datetime"]
        assert categorical == ["boolean", "category"]
        assert np.isnan(features.loc[5, "numeric"])
        assert pd.isna(features.loc[7, "category"])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("task", "model_type"),
        [
            ("classification", RandomForestClassifier),
            ("regression", RandomForestRegressor),
        ],
    )
    def test_pipeline_uses_expected_default_forest(
        self, miss_informative, task, model_type
    ):
        pipeline = miss_informative._forest_pipeline(
            task=task,
            numeric=["number"],
            categorical=["group"],
            random_state=31,
        )
        forest = pipeline.named_steps["forest"]
        assert isinstance(forest, model_type)
        assert forest.n_estimators == 100
        assert forest.max_depth is None
        assert forest.random_state == 31
        assert forest.n_jobs == -1

    @pytest.mark.unit
    def test_pipeline_handles_missing_and_unseen_categories(self, miss_informative):
        pipeline = miss_informative._forest_pipeline(
            task="classification",
            numeric=["number"],
            categorical=["group"],
            random_state=31,
        )
        train = pd.DataFrame({
            "number": [1.0, np.nan, 2.0, 3.0],
            "group": ["A", "A", "B", np.nan],
        })
        pipeline.fit(train, [0, 0, 1, 1])
        prediction = pipeline.predict(pd.DataFrame({
            "number": [np.nan], "group": ["unseen"]
        }))
        assert prediction.shape == (1,)


class TestValidationPaths:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("target", "predictors", "missing_variables", "message"),
        [
            (None, ["x"], ["x"], "The Target role is not assigned"),
            ("absent", ["x"], ["x"], "The target 'absent' is not in the data"),
            ("target", [], ["x"], "No predictor variables are assigned"),
            (
                "target", ["x"], [],
                "No predictors exceed the minimum missing-value proportion",
            ),
        ],
    )
    def test_early_validation_messages(
        self, miss_informative, target, predictors, missing_variables, message
    ):
        result = miss_informative._fit_forest_importance(
            informative_classification_frame(8),
            target=target,
            predictors=predictors,
            missing_variables=missing_variables,
        )
        assert result.message == message
        assert result.importance.empty

    @pytest.mark.unit
    def test_too_few_observed_targets_returns_message(self, miss_informative):
        frame = informative_classification_frame(8)
        frame.loc[3:, "target"] = np.nan
        result = miss_informative._fit_forest_importance(
            frame, target="target", predictors=["x"], missing_variables=["x"]
        )
        assert result.message == "Too few observations have a recorded target"

    @pytest.mark.unit
    def test_classification_requires_two_cases_per_class(self, miss_informative):
        frame = informative_classification_frame(8)
        frame["target"] = [0, 0, 0, 0, 0, 0, 0, 1]
        result = miss_informative._fit_forest_importance(
            frame, target="target", predictors=["x"], missing_variables=["x"]
        )
        assert result.message == (
            "Classification requires at least two observations in every class"
        )

    @pytest.mark.unit
    def test_unsupported_target_type_returns_message(self, miss_informative):
        frame = pd.DataFrame({
            "x": [np.nan, 0.0, np.nan, 0.0],
            "target": [[1, 0], [0, 1], [1, 0], [0, 1]],
        })
        result = miss_informative._fit_forest_importance(
            frame, target="target", predictors=["x"], missing_variables=["x"]
        )
        assert result.message is not None
        assert result.message.startswith("Unsupported target type:")


class TestForestAnalysis:
    @pytest.mark.unit
    def test_classification_identifies_informative_shadow(
        self, miss_informative, serial_permutation_importance
    ):
        result = miss_informative._fit_forest_importance(
            informative_classification_frame(),
            target="target",
            predictors=["x", "noise"],
            missing_variables=["x"],
            cv_folds=4,
        )
        shadow = result.importance.loc[
            result.importance["Variable"].eq(f"{Card.SHADOW_PREFIX}x")
        ].iloc[0]
        assert result.message is None
        assert result.task == "classification"
        assert result.score_name == "CV Balanced Accuracy"
        assert result.score == pytest.approx(1.0)
        assert result.folds == 4
        assert result.observations == 80
        assert shadow["Variable Type"] == "Shadow"
        assert shadow["Source Variable"] == "x"
        assert shadow["Missing Proportion"] == pytest.approx(0.5)
        assert shadow["Positive Fraction"] == pytest.approx(1.0)
        assert shadow["Interpretation"] == "Informative"

    @pytest.mark.unit
    def test_classification_threshold_can_make_shadow_uninformative(
        self, miss_informative, serial_permutation_importance
    ):
        result = miss_informative._fit_forest_importance(
            informative_classification_frame(40),
            target="target",
            predictors=["x"],
            missing_variables=["x"],
            cv_folds=2,
            minimum_balanced_accuracy=1.01,
        )
        shadow = result.importance.loc[
            result.importance["Variable Type"].eq("Shadow")
        ].iloc[0]
        assert shadow["Interpretation"] == "Uninformative"

    @pytest.mark.unit
    def test_cv_folds_are_capped_by_minority_class(
        self, miss_informative, serial_permutation_importance
    ):
        frame = informative_classification_frame(12)
        frame["target"] = [0] * 9 + [1] * 3
        frame["x"] = np.where(frame["target"].eq(1), np.nan, 0.0)
        result = miss_informative._fit_forest_importance(
            frame,
            target="target",
            predictors=["x"],
            missing_variables=["x"],
            cv_folds=10,
        )
        assert result.folds == 3

    @pytest.mark.unit
    def test_regression_identifies_informative_shadow(
        self, miss_informative, serial_permutation_importance
    ):
        result = miss_informative._fit_forest_importance(
            informative_regression_frame(),
            target="target",
            predictors=["x", "noise"],
            missing_variables=["x"],
            cv_folds=4,
        )
        shadow = result.importance.loc[
            result.importance["Variable Type"].eq("Shadow")
        ].iloc[0]
        assert result.message is None
        assert result.task == "regression"
        assert result.score_name == "CV R-squared"
        assert result.score > 0.95
        assert shadow["Importance"] > 0
        assert shadow["Interpretation"] == "Informative"

    @pytest.mark.unit
    def test_invalid_weights_are_cleaned_without_failure(
        self, miss_informative, serial_permutation_importance
    ):
        frame = informative_classification_frame(40)
        weights = pd.Series(
            [np.nan, np.inf, -1, 0, 1] * 8,
            index=frame.index,
        )
        result = miss_informative._fit_forest_importance(
            frame,
            target="target",
            predictors=["x"],
            missing_variables=["x"],
            sample_weight=weights,
            cv_folds=2,
        )
        assert result.message is None
        assert np.isfinite(result.score)


class TestImportanceFigure:
    @pytest.mark.unit
    def test_message_analysis_returns_empty_figure(self, miss_informative):
        figure = miss_informative._importance_figure(
            miss_informative._empty_analysis(None, "The Target role is not assigned")
        )
        assert len(figure.layout.annotations) == 1
        assert figure.layout.annotations[0].text == "The Target role is not assigned"
        assert len(figure.layout.images) == 1

    @pytest.mark.unit
    def test_importance_figure_labels_shadows_and_predictors(self, miss_informative):
        analysis = miss_informative.ForestAnalysis(
            target="target",
            task="classification",
            score_name="CV Balanced Accuracy",
            score=0.8,
            score_sd=0.1,
            folds=5,
            observations=100,
            importance=importance_table(miss_informative),
        )
        figure = miss_informative._importance_figure(analysis)
        assert len(figure.data) == 1
        assert figure.data[0].orientation == "h"
        assert set(figure.data[0].y) == {"ordinary", "Missing: x"}
        assert set(figure.data[0].marker.color) == {"#0d6efd", "#ffc107"}
        assert [row[2] for row in figure.data[0].customdata] == [
            "Not applicable", "25.0%"
        ]

    @pytest.mark.unit
    def test_figure_filters_nonfinite_importance_and_serializes_strictly(
        self, miss_informative
    ):
        table = importance_table(miss_informative)
        table.loc[0, "Importance"] = np.nan
        table.loc[1, "Importance SD"] = np.inf
        table.loc[1, "Positive Fraction"] = np.nan
        analysis = miss_informative.ForestAnalysis(
            "target", "classification", "CV Balanced Accuracy",
            0.8, 0.1, 5, 100, table,
        )
        figure = miss_informative._importance_figure(analysis)
        assert list(figure.data[0].y) == ["Missing: x"]
        packed = json_packer(figure.to_plotly_json())
        assert "NaN" not in packed
        assert "Infinity" not in packed

    @pytest.mark.unit
    def test_maximum_variables_limits_bars(self, miss_informative):
        table = pd.concat(
            [importance_table(miss_informative)] * 4,
            ignore_index=True,
        )
        table["Variable"] = [f"v{index}" for index in range(len(table))]
        analysis = miss_informative.ForestAnalysis(
            "target", "classification", "CV Balanced Accuracy",
            0.8, 0.1, 5, 100, table,
        )
        figure = miss_informative._importance_figure(
            analysis, maximum_variables=3
        )
        assert len(figure.data[0].x) == 3
