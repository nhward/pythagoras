from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/miss_impute.py", scope="function")


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.miss_impute")


def mixed_frame():
    return pd.DataFrame({
        "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        "ordered": pd.Categorical(
            ["low", "low", None, "middle", "middle", "high", "high", "high"],
            categories=["low", "middle", "high"], ordered=True,
        ),
        "nominal": pd.Categorical(["a", "a", None, "b", "b", "b", "a", "a"]),
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", None, "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"]),
    })


@pytest.mark.unit
def test_card_is_mutable_with_apply_control_and_performance_side(card_module):
    card = card_module.instance()
    assert card.mutable is True
    assert card.long_name == "Learned Imputation"
    assert 'id="MissingChart"' in str(card.front)
    assert 'id="Performance"' in str(card.back)
    assert 'id="Apply"' in str(card.footer)
    assert "shiny-input-checkboxgroup" in str(card.footer)
    assert 'id="Method"' in str(card.settings)
    assert 'id="Iterations"' in str(card.settings)
    assert 'id="MinImprovement"' in str(card.settings)
    assert 'id="Trees"' not in str(card.settings)


@pytest.mark.unit
def test_simple_imputation_uses_dtype_appropriate_statistics(card_module):
    frame = mixed_frame()
    result, methods = card_module._simple_impute(frame, list(frame.columns))
    assert result.isna().sum().sum() == 0
    assert result.loc[2, "numeric"] == pytest.approx(5.0)
    assert result.loc[2, "ordered"] == "middle"
    assert result.loc[2, "nominal"] == "a"
    assert result.loc[2, "date"] == pd.Timestamp("2025-01-05")
    assert methods == {
        "numeric": "Median",
        "ordered": "Median",
        "nominal": "Mode",
        "date": "Median",
    }


@pytest.mark.unit
def test_only_supported_predictors_are_eligible(card_module):
    from cyclic_pandas import as_cyclic
    from list_pandas import as_list
    from proxy_data import proxy_data
    from roles import Role, RoleMap

    frame = mixed_frame()
    frame["cycle"] = as_cyclic(pd.Series([0, 1, None, 3, 4, 5, 6, 7]), period=12)
    frame["basket"] = as_list(pd.Series([[1], [2], None, [1], [2], [1], [2], [1]]))
    frame["target"] = [1, 2, np.nan, 4, 5, 6, 7, 8]
    roles = RoleMap()
    for column in frame.columns:
        roles.set_roles(column, [Role.TARGET if column == "target" else Role.PREDICTOR])
    data = proxy_data(_df=frame, _roles=roles)

    eligible, excluded = card_module._eligible_columns(frame, data.role_map)

    assert set(eligible) == {"numeric", "ordered", "nominal", "date"}
    assert "Cyclic" in excluded["cycle"]
    assert "List" in excluded["basket"]
    assert excluded["target"] == "Not assigned the predictor role"


@pytest.mark.unit
def test_knn_imputes_continuous_and_falls_back_to_mode(card_module):
    frame = mixed_frame()
    result, methods = card_module._knn_impute(
        frame, list(frame.columns), list(frame.columns), neighbours=3,
    )
    assert result.isna().sum().sum() == 0
    assert methods["numeric"] == "Nearest neighbours (3)"
    assert methods["ordered"] == "Nearest neighbours (3)"
    assert methods["date"] == "Nearest neighbours (3)"
    assert methods["nominal"] == "Mode"


@pytest.mark.unit
def test_iterative_imputation_preserves_categorical_dtype(card_module):
    frame = mixed_frame()
    result, methods = card_module._iterative_impute(
        frame, list(frame.columns), list(frame.columns), iterations=5, seed=7,
    )
    assert result.isna().sum().sum() == 0
    assert isinstance(result["nominal"].dtype, pd.CategoricalDtype)
    assert methods["numeric"] == "Iterative prediction (5 iterations)"
    assert methods["nominal"] == "Mode"


@pytest.mark.unit
def test_random_baseline_draws_only_from_remaining_observed_values(card_module):
    trial = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    held = np.array([1, 3])

    result = card_module._random_donor_baseline(
        trial, held, np.random.default_rng(42)
    )

    assert set(result.iloc[held]).issubset({1.0, 3.0, 5.0})
    assert result.iloc[[0, 2, 4]].equals(trial.iloc[[0, 2, 4]])


@pytest.mark.unit
def test_analysis_reports_exclusions_and_cross_validated_metrics(card_module):
    from cyclic_pandas import as_cyclic
    from proxy_data import proxy_data
    frame = mixed_frame()
    frame["cycle"] = as_cyclic(pd.Series([0, 1, None, 3, 4, 5, 6, 7]), period=12)
    result = card_module._analyse(
        proxy_data(_df=frame), "simple", 3, 10, 2, 0.25, 11, 2,
    )
    assert result.frame["numeric"].isna().sum() == 0
    assert result.frame["cycle"].isna().sum() == 1
    cycle = result.summary.loc[result.summary["Predictor"] == "cycle"].iloc[0]
    assert "not imputed" in cycle["Status"]
    assert set(result.evaluation["Status"]) == {"Assessed"}
    assert set(result.evaluation["Metric"]) == {"MAE", "Ordinal MAE", "Accuracy", "MAE (days)"}
    assert "Random baseline" in result.evaluation


@pytest.mark.unit
@pytest.mark.parametrize("method", ["simple", "iterative", "knn"])
def test_analysis_supports_all_three_sklearn_imputers(card_module, method):
    from proxy_data import proxy_data

    result = card_module._analyse(
        proxy_data(_df=mixed_frame()), method, 3, 3, 1, 0.25, 11, 1,
    )

    assert result.frame.isna().sum().sum() == 0
    assert not result.evaluation.empty


@pytest.mark.unit
def test_performance_is_classified_and_sorted_by_improvement(card_module):
    evaluation = pd.DataFrame({
        "Predictor": ["weak", "unknown", "strong"],
        "Improvement": [0.05, np.nan, 0.40],
        "Status": ["Assessed", "Too few observed values", "Assessed"],
    })

    table = card_module._performance_table(evaluation, 0.10)

    assert list(table["Predictor"]) == ["strong", "weak", "unknown"]
    assert list(table["Status"]) == [
        "Strong", "Weak", "Too few observed values",
    ]


@pytest.mark.unit
def test_performance_threshold_is_strict_and_styles_are_semantic(card_module):
    evaluation = pd.DataFrame({
        "Predictor": ["equal", "above"],
        "Improvement": [0.10, 0.1001],
        "Status": ["Assessed", "Assessed"],
    })
    table = card_module._performance_table(evaluation, 0.10)
    styles = card_module._performance_row_styles(table)

    assert table.set_index("Predictor").loc["equal", "Status"] == "Weak"
    assert table.set_index("Predictor").loc["above", "Status"] == "Strong"
    assert {style["class"] for style in styles} == {
        "miss-impute-strong-row", "miss-impute-weak-row",
    }


@pytest.mark.unit
def test_chart_changes_after_counts_only_when_applied(card_module):
    summary = pd.DataFrame({
        "Predictor": ["x", "excluded"], "Before": [3, 2], "After": [0, 2],
        "Status": ["Median", "List variables are not imputed"],
    })
    preview = card_module._missingness_figure(summary, applied=False)
    applied = card_module._missingness_figure(summary, applied=True)
    assert list(preview.data[1].x) == [2, 3]
    assert list(applied.data[1].x) == [2, 0]
    assert applied.layout.legend.orientation == "h"
    assert applied.layout.legend.y < 0
    assert applied.layout.legend.xanchor == "center"
    assert applied.layout.margin.b == 90


def get_card(page: Page):
    return page.locator(".card").first


def by_id(page: Page, local_id: str):
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return page.locator(f"#{card_id.partition('-')[0]}-{local_id}")


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_and_apply_control_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "MissingChart").locator(".plotly")).to_be_attached(timeout=20_000)
        expect(by_id(page, "Check")).to_contain_text("Ready to impute 3", timeout=20_000)
        expect(by_id(page, "Apply")).to_be_attached()
        expect(by_id(page, "MinImprovement")).to_be_attached()

    @pytest.mark.ui
    def test_apply_control_changes_export_status(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("Ready to impute", timeout=20_000)
        by_id(page, "Apply").locator("input[type=checkbox]").check(force=True)
        expect(by_id(page, "Check")).to_contain_text("Imputed 3 predictors", timeout=20_000)

    @pytest.mark.ui
    def test_flipside_performance_rows_are_classified_and_coloured(
        self, page: Page, app: ShinyAppProc,
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("Ready to impute", timeout=20_000)
        by_id(page, "FlipButton").click(force=True)

        table = by_id(page, "PerformanceTable")
        expect(table).to_contain_text("Improvement", timeout=15_000)
        # At the current 0.25 default threshold this deterministic scenario's
        # best improvement (about 0.21) is intentionally classified Weak.
        expect(table).to_contain_text("Weak")
        expect(table.locator(".miss-impute-weak-row").first).to_be_attached()
