from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import pytest
from card import Card
from cyclic_pandas import as_cyclic
from list_pandas import as_list
from playwright.sync_api import Page, expect
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

_HELPER_CARDS = {}

app = create_app_fixture(app="../scenarios/miss_type.py", scope="function")

@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}

@pytest.fixture
def miss_type():
    """Import the card module from the app package."""
    return importlib.import_module("cards.miss_type")


@pytest.fixture
def card(miss_type):
    return miss_type.instance()


def get_card(page: Page):
    return page.locator(".card").first


def get_namespace(page: Page) -> str:
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return card_id.partition("-")[0]


def namespaced_id(page: Page, local_id: str) -> str:
    return f"{get_namespace(page)}-{local_id}"


def by_id(page: Page, local_id: str):
    return page.locator(f"#{namespaced_id(page, local_id)}")


def set_shiny_input(page: Page, local_id: str, value):
    page.wait_for_function("() => !!window.Shiny?.setInputValue")
    page.evaluate(
        """
        ([inputId, inputValue]) => window.Shiny.setInputValue(
            inputId, inputValue, {priority: "event"}
        )
        """,
        [namespaced_id(page, local_id), value],
    )


def classification_frame(rows: int = 80) -> pd.DataFrame:
    group = np.tile([0, 1], rows // 2)
    target = np.arange(rows, dtype=float)
    target[group == 1] = np.nan
    return pd.DataFrame({
        "target": target,
        "group": group,
        "noise": np.random.default_rng(17).normal(size=rows),
    })


def regression_frame(rows: int = 80) -> pd.DataFrame:
    group = np.tile([0, 1, 2, 3], rows // 4)
    first = np.arange(rows, dtype=float)
    second = np.arange(rows, dtype=float) + 100
    first[group >= 1] = np.nan
    second[group >= 3] = np.nan
    return pd.DataFrame({"first": first, "second": second, "group": group})


class TestInstance:
    @pytest.mark.unit
    def test_metadata_and_regions(self, card):
        assert card.name == "miss_type"
        assert card.long_name == "Missingness Type"
        assert "decision trees" in card.description
        assert not card.mutable
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert card.hasFooter()


    @pytest.mark.unit
    def test_expected_outputs_and_settings_are_present(self, card):
        front = str(card.front.tagify())
        back = str(card.back.tagify())
        footer = str(card.footer.tagify())
        settings = str(card.settings)
        assert 'id="Target"' in front
        assert 'id="Tree"' in front
        assert 'id="Busy"' in footer
        assert 'id="Summary"' in footer
        assert 'id="Table"' in back
        for control in (
            "MaxTreeDepth", "MinLeafSamples", "MinMissProp", "CVFolds",
            "Permutations", "AdjustFDR", "Alpha", "MinImprovement",
            "MinBalancedAccuracy", "MinRSquared", "MinFoldFraction",
            "MinClassCount", "AddSeq", "MaxObs",
        ):
            assert f'id="{control}"' in settings


class TestSmallHelpers:
    @pytest.mark.unit
    def test_benjamini_hochberg_preserves_index_and_missing_values(self, miss_type):
        values = pd.Series([0.01, 0.04, 0.03, np.nan], index=list("abcd"))
        adjusted = miss_type._benjamini_hochberg(values)
        assert adjusted.index.tolist() == list("abcd")
        assert adjusted.iloc[:3].to_numpy() == pytest.approx([0.03, 0.04, 0.04])
        assert np.isnan(adjusted.iloc[3])

    @pytest.mark.unit
    def test_missing_variables_only_returns_incomplete_columns(self, miss_type):
        frame = pd.DataFrame({"complete": [1, 2], "missing": [1, np.nan]})
        assert miss_type._missing_variables(frame) == ["missing"]

    @pytest.mark.unit
    def test_eligible_predictors_excludes_unsafe_and_high_cardinality_columns(
        self, miss_type
    ):
        frame = pd.DataFrame({
            "target": [1, np.nan, 3, np.nan, 5, 6],
            "numeric": range(6),
            "category": pd.Categorical(["A", "A", "B", "B", "A", "B"]),
            "identifier": [f"id-{value}" for value in range(6)],
            "collection": [[value] for value in range(6)],
            f"{Card.SHADOW_PREFIX}target": [False, True, False, True, False, False],
            "excluded": range(10, 16),
        })
        predictors = miss_type._eligible_predictors(
            frame, target="target", excluded={"excluded"}
        )
        assert predictors == ["numeric", "category"]

    @pytest.mark.unit
    def test_row_styles_assign_semantic_css_classes(self, miss_type):
        table = pd.DataFrame({
            "Missingness Type": [
                "Random", "Uncertain", "Patterned", "Insufficient data", "Random"
            ]
        })
        styles = miss_type._missingness_row_styles(table)
        assert styles == [
            {"rows": [0, 4], "class": "miss-type-random-row"},
            {"rows": [1], "class": "miss-type-uncertain-row"},
            {"rows": [2], "class": "miss-type-patterned-row"},
            {"rows": [3], "class": "miss-type-insufficient-row"},
        ]
        assert miss_type._missingness_row_styles(pd.DataFrame({"x": [1]})) == []


class TestDesignMatrix:
    @pytest.mark.unit
    def test_mixed_design_matrix_is_numeric_and_imputed(self, miss_type):
        cyclic = as_cyclic(pd.Series([23.0, 1.0, np.nan]), period=24)
        basket = as_list(pd.Series([["red", "blue"], ["green"], pd.NA]))
        frame = pd.DataFrame({
            "number": [1.0, np.nan, np.inf],
            "boolean": pd.Series([True, False, pd.NA], dtype="boolean"),
            "ordered": pd.Series(pd.Categorical(
                ["low", "high", None], categories=["low", "high"], ordered=True
            )),
            "category": pd.Series(["A", "B", None], dtype="category"),
            "date": pd.to_datetime(["2025-01-01", None, "2025-01-03"]),
            "cyclic": cyclic,
            "basket": basket,
        })
        matrix, specifications = miss_type._fit_design_matrix(
            frame, frame.columns.tolist()
        )
        assert matrix.shape[0] == len(frame)
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in matrix.dtypes)
        assert np.isfinite(matrix.to_numpy()).all()
        assert matrix["number"].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert matrix["boolean"].tolist() == pytest.approx([1.0, 0.0, 0.0])
        assert matrix["ordered"].tolist() == pytest.approx([0.0, 1.0, 0.0])
        assert matrix.loc[1, "date"] == pytest.approx(
            (matrix.loc[0, "date"] + matrix.loc[2, "date"]) / 2
        )
        assert matrix["cyclic"].tolist() == pytest.approx([23.0, 1.0, 1.0])
        category_columns = matrix.filter(like="category =")
        assert category_columns.columns.tolist() == ["category = A", "category = B"]
        assert category_columns.loc[2].tolist() == pytest.approx([1.0, 0.0])
        basket_columns = matrix.filter(like="basket contains")
        assert basket_columns.shape[1] == 3
        assert any(
            basket_columns.loc[2].equals(basket_columns.loc[position])
            for position in (0, 1)
        )
        assert not any("<missing>" in column for column in matrix.columns)
        assert {specification["kind"] for specification in specifications} == {
            "numeric", "boolean", "ordered", "categorical", "datetime",
            "cyclic", "list",
        }

    @pytest.mark.unit
    def test_held_out_transform_reuses_training_imputation_and_categories(
        self, miss_type
    ):
        train = pd.DataFrame({"number": [1.0, np.nan], "category": ["A", "B"]})
        test = pd.DataFrame({
            "number": [np.nan, 2.0],
            "category": [None, "unseen"],
        })
        train_matrix, specifications = miss_type._fit_design_matrix(
            train, ["number", "category"]
        )
        test_matrix = miss_type._transform_design_matrix(test, specifications)
        assert test_matrix.columns.tolist() == train_matrix.columns.tolist()
        assert test_matrix.loc[0, "number"] == pytest.approx(1.0)
        assert test_matrix.filter(like="category =").iloc[0].tolist() == [1.0, 0.0]
        assert test_matrix.filter(like="category =").iloc[1].eq(0).all()
        assert np.isfinite(test_matrix.to_numpy()).all()
        assert not any("<missing>" in column for column in test_matrix.columns)

    @pytest.mark.unit
    def test_held_out_list_imputation_uses_only_training_donors(self, miss_type):
        train = pd.DataFrame({
            "basket": as_list(pd.Series([["A", "B"], ["C"], pd.NA])),
        })
        test = pd.DataFrame({
            "basket": as_list(pd.Series([pd.NA, ["unseen"]])),
        })
        train_matrix, specifications = miss_type._fit_design_matrix(
            train, ["basket"]
        )
        test_matrix = miss_type._transform_design_matrix(test, specifications)
        assert test_matrix.columns.tolist() == train_matrix.columns.tolist()
        assert test_matrix.iloc[0].sum() in {1.0, 2.0}
        assert test_matrix.iloc[1].sum() == 0.0
        assert np.isfinite(test_matrix.to_numpy()).all()


class TestTreePreparationAndFit:
    @pytest.mark.unit
    def test_prepare_classification_data_cleans_weights_and_adds_sequence(
        self, miss_type
    ):
        frame = classification_frame(8)
        weights = pd.Series([1, np.nan, -2, np.inf, 2, 3, 4, 5])
        matrix, truth, task, cleaned, working, predictors = miss_type._prepare_tree_data(
            frame,
            target="target",
            add_sequence=True,
            sample_weight=weights,
            excluded={"noise"},
        )
        assert task == "classification"
        assert truth.dtype == bool
        assert truth.sum() == 4
        assert "rownum" in working and "rownum" in predictors
        assert matrix.columns.tolist() == ["group", "rownum"]
        assert cleaned is not None and np.isfinite(cleaned).all()
        assert (cleaned >= 0).all()

    @pytest.mark.unit
    def test_prepare_observation_count_is_regression(self, miss_type):
        frame = regression_frame(8)
        _, truth, task, _, _, _ = miss_type._prepare_tree_data(
            frame,
            target=miss_type.OBS_COUNT,
            add_sequence=False,
            sample_weight=None,
            excluded=None,
        )
        assert task == "regression"
        assert truth.tolist() == frame[["first", "second"]].isna().sum(axis=1).tolist()

    @pytest.mark.unit
    def test_prepare_rejects_unknown_target(self, miss_type):
        with pytest.raises(KeyError, match="Unknown missingness target"):
            miss_type._prepare_tree_data(
                classification_frame(8),
                target="unknown",
                add_sequence=False,
                sample_weight=None,
                excluded=None,
            )

    @pytest.mark.unit
    def test_classification_tree_finds_pattern(self, miss_type):
        analysis = miss_type._fit_missingness_tree(
            classification_frame(), target="target", add_sequence=False
        )
        assert analysis.task == "classification"
        assert analysis.model is not None
        assert analysis.branches >= 1
        assert np.array_equal(analysis.truth, analysis.prediction)

    @pytest.mark.unit
    def test_tree_without_predictors_returns_constant_analysis(self, miss_type):
        frame = pd.DataFrame({"target": [1.0, np.nan, 2.0, np.nan]})
        analysis = miss_type._fit_missingness_tree(
            frame, target="target", add_sequence=False
        )
        assert analysis.model is None
        assert analysis.branches == 0
        assert analysis.feature_names == []
        assert analysis.prediction.tolist() == [True, True, True, True]


class TestDiagnosticsAndInterpretation:
    @pytest.mark.unit
    def test_classification_diagnostics_report_held_out_pattern(self, miss_type):
        result = miss_type._classification_diagnostics(
            classification_frame(),
            target="target",
            add_sequence=False,
            cv_folds=4,
            permutations=3,
        )
        assert result["Missing Count"] == 40
        assert result["Observed Count"] == 40
        assert result["CV Folds"] == 4
        assert result["CV Balanced Accuracy"] == pytest.approx(1.0)
        assert result["Improvement"] > 0
        assert 0 < result["Permutation p-value"] <= 1

    @pytest.mark.unit
    def test_classification_diagnostics_handles_tiny_minority(self, miss_type):
        frame = classification_frame(8)
        frame["target"] = np.arange(8, dtype=float)
        frame.loc[0, "target"] = np.nan
        result = miss_type._classification_diagnostics(
            frame, target="target", add_sequence=False, permutations=0
        )
        assert result["Missing Count"] == 1
        assert result["CV Folds"] == 0
        assert np.isnan(result["CV Balanced Accuracy"])

    @pytest.mark.unit
    def test_regression_diagnostics_report_expected_fields(self, miss_type):
        result = miss_type._regression_diagnostics(
            regression_frame(), add_sequence=False, cv_folds=4, permutations=2
        )
        assert result["Observations"] == 80
        assert result["CV Folds"] == 4
        assert np.isfinite(result["CV R-squared"])
        assert np.isfinite(result["Null R-squared"])
        assert 0 < result["Permutation p-value"] <= 1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("diagnostics", "expected"),
        [
            ({
                "CV R-squared": np.nan, "Permutation p-value": np.nan,
                "Improvement": np.nan, "Fraction Folds Above Null": np.nan,
            }, "Insufficient data"),
            ({
                "CV R-squared": 0.7, "Permutation p-value": 0.01,
                "Improvement": 0.2, "Fraction Folds Above Null": 1.0,
            }, "Patterned"),
            ({
                "CV R-squared": 0.3, "Permutation p-value": 0.01,
                "Improvement": 0.02, "Fraction Folds Above Null": 0.4,
            }, "Uncertain"),
            ({
                "CV R-squared": -0.1, "Permutation p-value": 0.8,
                "Improvement": -0.01, "Fraction Folds Above Null": 0.0,
            }, "Random"),
        ],
    )
    def test_regression_interpretation(self, miss_type, diagnostics, expected):
        assert miss_type._interpret_regression_model(
            diagnostics,
            alpha=0.05,
            minimum_improvement=0.05,
            minimum_r_squared=0.5,
            minimum_fold_fraction=0.8,
        ) == expected

    @pytest.mark.unit
    def test_classification_interpretation_covers_all_outcomes(self, miss_type):
        table = pd.DataFrame({
            "Permutation p-value": [0.001, 0.01, 0.9, 0.01],
            "Missing Count": [30, 30, 30, 1],
            "Observed Count": [70, 70, 70, 99],
            "CV Balanced Accuracy": [0.9, 0.52, 0.5, 0.9],
            "Improvement": [0.4, 0.02, 0.0, 0.4],
            "Fraction Folds Above Null": [1.0, 0.4, 0.0, 1.0],
        })
        result = miss_type._interpret_missingness_models(
            table,
            adjust_fdr=False,
            alpha=0.05,
            minimum_improvement=0.05,
            minimum_balanced_accuracy=0.55,
            minimum_fold_fraction=0.8,
            minimum_class_count=20,
        )
        assert result["Missingness Type"].tolist() == [
            "Patterned", "Uncertain", "Random", "Insufficient data"
        ]


class TestTableAndPlot:
    @pytest.mark.unit
    def test_empty_missingness_table_has_interpreted_schema(self, miss_type):
        table = miss_type._missingness_table(
            classification_frame(8), targets=[], processes=1
        )
        assert table.empty
        assert "Adjusted p-value" in table
        assert "Missingness Type" in table

    @pytest.mark.unit
    def test_single_process_missingness_table_is_sorted_and_interpreted(
        self, miss_type
    ):
        frame = classification_frame()
        second = np.arange(len(frame), dtype=float)
        second[np.arange(len(frame)) % 4 == 0] = np.nan
        frame["second"] = second
        table = miss_type._missingness_table(
            frame,
            targets=["target", "second"],
            add_sequence=False,
            cv_folds=3,
            permutations=0,
            minimum_class_count=2,
            processes=1,
        )
        assert set(table["Variable"]) == {"target", "second"}
        assert table["Adjusted p-value"].isna().all()
        assert table["Missingness Type"].isin(
            ["Random", "Uncertain", "Patterned", "Insufficient data"]
        ).all()
        assert table["CV Balanced Accuracy"].notna().all()

    @pytest.mark.unit
    def test_empty_tree_figure_contains_explanation(self, miss_type):
        analysis = miss_type.TreeAnalysis(
            None, [], "target", "classification",
            np.array([False, True]), np.array([False, False]), 0,
        )
        figure = miss_type._tree_figure(analysis)
        assert len(figure.layout.annotations) == 1
        assert "no decision-tree structure" in figure.layout.annotations[0].text

    @pytest.mark.unit
    def test_fitted_tree_figure_contains_nodes_and_edges(self, miss_type):
        analysis = miss_type._fit_missingness_tree(
            classification_frame(), target="target", add_sequence=False
        )
        figure = miss_type._tree_figure(analysis)
        assert len(figure.layout.annotations) >= 3
        assert len(figure.data) == 1
        assert figure.data[0].mode == "lines"
        assert sum(value is None for value in figure.data[0].x) >= 2
        assert figure.layout.xaxis.visible is False
        assert figure.layout.yaxis.visible is False


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_targets_and_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Missingness Type", exact=True)).to_be_visible()
        expect(page.get_by_role("tab", name="Obs-count", exact=True)).to_be_visible()
        expect(page.get_by_role("tab", name="target", exact=True)).to_be_visible(
            timeout=30_000
        )
        for control in (
            "MaxTreeDepth", "MinLeafSamples", "MinMissProp", "CVFolds",
            "Permutations", "AdjustFDR", "Alpha", "MinImprovement",
            "MinBalancedAccuracy", "MinRSquared", "MinFoldFraction",
            "MinClassCount", "AddSeq", "MaxObs",
        ):
            expect(by_id(page, control)).to_be_attached()

    @pytest.mark.ui
    def test_variable_tree_and_summary_complete(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        target_tab = page.get_by_role("tab", name="target", exact=True)
        target_tab.click(timeout=30_000)
        expect(target_tab).to_have_attribute("aria-selected", "true")
        expect(by_id(page, "target__Tree")).to_be_visible()
        expect(by_id(page, "target__Tree").locator(".plotly")).to_be_attached(
            timeout=30_000
        )
        expect(by_id(page, "Summary")).to_contain_text(
            "Interpretation: Patterned.", timeout=30_000
        )

    @pytest.mark.ui
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The dynamically inserted target tab becomes active in Bootstrap but "
            "does not update Shiny input.Target; the summary remains Obs-count."
        ),
    )
    def test_selecting_variable_tab_changes_summary_to_classification(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        target_tab = page.get_by_role("tab", name="target", exact=True)
        target_tab.click(timeout=30_000)
        expect(target_tab).to_have_attribute("aria-selected", "true")
        expect(by_id(page, "Summary")).to_contain_text(
            "Interpretation: Patterned.", timeout=30_000
        )
        expect(by_id(page, "Summary")).to_contain_text(
            "CV balanced accuracy is", timeout=3_000
        )

    @pytest.mark.ui
    def test_flip_displays_missingness_diagnostics(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        target_tab = page.get_by_role("tab", name="target", exact=True)
        target_tab.click(timeout=30_000)
        expect(target_tab).to_have_attribute("aria-selected", "true")
        expect(by_id(page, "Summary")).to_contain_text(
            "Interpretation: Patterned.", timeout=30_000
        )
        by_id(page, "FlipButton").click(force=True)
        table = controller.OutputDataFrame(page, namespaced_id(page, "Table2"))
        table.expect_nrow(1)
        table.expect_column_labels([
            "Variable",
            "Branches",
            "CV Balanced Accuracy",
            "Null Balanced Accuracy",
            "Improvement",
            "Fraction Folds Above Null",
            "Permutation p-value",
            "Missing Count",
            "Observed Count",
            "CV Folds",
            "Adjusted p-value",
            "Missingness Type",
        ])
        table.expect_cell("target", row=0, col=0)
        table.expect_cell("Patterned", row=0, col=11)

    @pytest.mark.ui
    def test_missing_proportion_change_rebuilds_target_tabs(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(page.get_by_role("tab", name="target", exact=True)).to_be_visible(
            timeout=30_000
        )

        set_shiny_input(page, "MinMissProp", 0.5)
        expect(by_id(page, "Summary")).to_contain_text(
            "The data does not contain significantly missing variables.",
            timeout=30_000,
        )
        expect(page.get_by_role("tab", name="target", exact=True)).to_have_count(0)

        set_shiny_input(page, "MinMissProp", 0.49)
        expect(page.get_by_role("tab", name="target", exact=True)).to_be_visible(
            timeout=30_000
        )
