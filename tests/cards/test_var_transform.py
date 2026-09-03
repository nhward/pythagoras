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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/var_transform.py", scope="function")


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.var_transform")


def distribution_data():
    from proxy_data import proxy_data
    from roles import Role, RoleMap

    rng = np.random.default_rng(47)
    frame = pd.DataFrame({
        "skewed": np.r_[rng.exponential(2, 79), np.nan],
        "large": rng.normal(100, 20, 80),
        "outcome": rng.exponential(5, 80),
        "category": pd.Categorical(np.tile(["A", "B"], 40)),
        "identifier": [f"id-{index}" for index in range(80)],
        "constant": np.ones(80),
    })
    roles = RoleMap()
    for column in frame.columns:
        roles.set_roles(
            column,
            [
                Role.IDENTIFIER if column == "identifier"
                else Role.TARGET if column == "outcome"
                else Role.PREDICTOR
            ],
        )
    return proxy_data(_df=frame, _roles=roles), frame


@pytest.mark.unit
def test_card_is_mutable_with_transform_controls_and_statistics(card_module):
    card = card_module.instance()

    assert card.mutable is True
    assert card.long_name == "Variable Transform"
    assert 'id="DistributionChart"' in str(card.front)
    assert 'id="Statistics"' in str(card.back)
    footer = str(card.footer)
    assert 'id="Transform"' in footer
    assert "shiny-input-checkboxgroup" in footer
    assert "Mean centre" in footer
    assert "Common spread" in footer
    assert "Reduce skew" in footer
    assert 'id="IncludeTarget"' in str(card.settings)
    assert 'id="Labels"' in str(card.settings)


@pytest.mark.unit
def test_pipeline_uses_sklearn_and_keeps_controls_independent(card_module):
    pipeline = card_module._build_pipeline(["Deskew", "Center", "Scale"])

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["deskew", "common_spread", "centre"]
    assert isinstance(pipeline.named_steps["deskew"], PowerTransformer)
    assert pipeline.named_steps["deskew"].method == "yeo-johnson"
    assert pipeline.named_steps["deskew"].standardize is False
    assert isinstance(
        pipeline.named_steps["common_spread"], card_module.CommonSpreadScaler,
    )
    assert isinstance(pipeline.named_steps["centre"], StandardScaler)
    assert pipeline.named_steps["centre"].with_mean is True
    assert pipeline.named_steps["centre"].with_std is False
    assert card_module._transform_name(["Center", "Deskew", "Scale"]) == (
        "Reduce skew → Common spread → Mean centre"
    )
    assert card_module._build_pipeline([]) is None


@pytest.mark.unit
def test_eligible_columns_are_numeric_nonconstant_predictors(card_module):
    data, _ = distribution_data()

    eligible, excluded = card_module._eligible_columns(data)

    assert eligible == ["skewed", "large"]
    assert excluded["outcome"] == "Continuous target transformation is not enabled"
    assert "Categorical" in excluded["category"]
    assert excluded["identifier"] == "Not assigned the predictor role"
    assert excluded["constant"] == "Constant predictor"

    table = card_module._analyse_distribution(data, []).statistics.set_index("Variable")
    assert table.loc["identifier", "Role"] == "Identifier"
    assert table.loc["category", "Role"] == "Predictor"
    assert table.loc["outcome", "Role"] == "Target"


@pytest.mark.unit
def test_flip_table_reports_all_assigned_roles(card_module):
    from roles import Role

    data, _ = distribution_data()
    data.role_map.set_roles("identifier", [Role.IDENTIFIER, Role.SENSITIVE])

    table = card_module._analyse_distribution(data, []).statistics.set_index("Variable")

    assert table.loc["identifier", "Role"] == "Identifier, Sensitive"


@pytest.mark.unit
def test_continuous_target_is_separately_opted_in(card_module):
    data, _ = distribution_data()

    assert card_module._continuous_target(data) == "outcome"
    eligible, excluded = card_module._eligible_columns(data, include_target=True)

    assert eligible == ["skewed", "large", "outcome"]
    assert "outcome" not in excluded


@pytest.mark.unit
def test_target_option_is_a_no_op_when_no_target_is_assigned(card_module):
    from roles import Role

    data, source = distribution_data()
    data.role_map.set_roles("outcome", [Role.NONE])

    result = card_module._analyse_distribution(
        data, ["Scale"], include_target=True,
    )

    assert result.target is None
    assert result.eligible == ["skewed", "large"]
    pd.testing.assert_series_equal(result.frame["outcome"], source["outcome"])


@pytest.mark.unit
def test_target_is_transformed_independently_and_can_be_inverted(card_module):
    data, source = distribution_data()
    result = card_module._analyse_distribution(
        data, ["Deskew", "Scale", "Center"], include_target=True,
    )

    table = result.statistics.set_index("Variable")
    assert result.target == "outcome"
    assert table.loc["outcome", "Role"] == "Target"
    assert "outcome" in result.pipelines
    restored = result.inverse_target(result.frame["outcome"])
    np.testing.assert_allclose(restored, source["outcome"].to_numpy())


@pytest.mark.unit
def test_centering_and_scaling_use_standard_scaler(card_module):
    data, source = distribution_data()

    result = card_module._analyse_distribution(data, ["Center", "Scale"])

    assert result.transforms == ("Center", "Scale")
    for column in result.eligible:
        observed = result.frame[column].dropna().to_numpy()
        assert observed.mean() == pytest.approx(0, abs=1e-12)
        assert observed.std(ddof=0) == pytest.approx(1, abs=1e-12)
    pd.testing.assert_series_equal(result.frame["category"], source["category"])
    pd.testing.assert_series_equal(result.frame["identifier"], source["identifier"])
    # Analysis must never mutate its incoming frame.
    pd.testing.assert_frame_equal(data.to_native(), source)


@pytest.mark.unit
def test_common_spread_preserves_mean_and_shape_but_sets_unit_sd(card_module):
    data, _ = distribution_data()

    result = card_module._analyse_distribution(data, ["Scale"])
    table = result.statistics.set_index("Variable")

    for column in ("skewed", "large"):
        row = table.loc[column]
        assert row["Mean after"] == pytest.approx(row["Mean before"], abs=1e-12)
        assert row["SD after"] == pytest.approx(1.0, abs=1e-12)
        assert row["Skew after"] == pytest.approx(row["Skew before"], abs=1e-12)
        assert row["Kurtosis after"] == pytest.approx(
            row["Kurtosis before"], abs=1e-12,
        )


@pytest.mark.unit
def test_yeo_johnson_reduces_skew_and_preserves_missing_values(card_module):
    data, source = distribution_data()

    result = card_module._analyse_distribution(data, ["Deskew"])
    row = result.statistics.set_index("Variable").loc["skewed"]

    assert abs(row["Skew after"]) < abs(row["Skew before"])
    assert np.isfinite(row["Yeo-Johnson lambda"])
    assert result.frame["skewed"].isna().equals(source["skewed"].isna())
    assert row["Transformation"] == "Reduce skew"


@pytest.mark.unit
def test_no_selection_returns_same_values_and_reports_exclusions(card_module):
    data, source = distribution_data()

    result = card_module._analyse_distribution(data, [])

    assert result.transforms == ()
    pd.testing.assert_frame_equal(result.frame, source)
    table = result.statistics.set_index("Variable")
    assert table.loc["skewed", "Transformation"] == "None"
    assert "not transformed" in table.loc["category", "Transformation"]


@pytest.mark.unit
def test_describe_reports_ordinary_not_excess_kurtosis(card_module):
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    summary = card_module._describe(values)

    assert summary["Kurtosis"] == pytest.approx(values.kurt() + 3.0)


@pytest.mark.unit
def test_card_figure_has_only_location_and_spread_panel(card_module):
    data, _ = distribution_data()
    result = card_module._analyse_distribution(data, ["Deskew", "Center", "Scale"])

    figure = card_module._distribution_figure(result.statistics, transformed=True)

    assert figure.layout.xaxis.title.text == "Mean"
    assert figure.layout.yaxis.title.text == "Standard deviation"
    assert figure.layout.xaxis.range[0] < 0
    assert figure.layout.yaxis.range[0] < 0
    assert "xaxis2" not in figure.layout
    assert [trace.name for trace in figure.data].count("Before") == 1
    assert [trace.name for trace in figure.data].count("After") == 1
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y < 0


@pytest.mark.unit
def test_full_screen_figure_has_three_axis_sharing_panels(card_module):
    data, _ = distribution_data()
    result = card_module._analyse_distribution(data, ["Deskew", "Center", "Scale"])

    figure = card_module._distribution_figure(
        result.statistics, transformed=True, full_screen=True,
    )

    # Top-left: skew/SD; top-right: mean/SD; bottom-left: skew/kurtosis.
    assert figure.layout.xaxis.title.text == ""
    assert figure.layout.yaxis.title.text == "Standard deviation"
    assert figure.layout.xaxis2.title.text == "Mean"
    assert figure.layout.yaxis2.matches == "y"
    assert figure.layout.xaxis3.title.text == "Skew"
    assert figure.layout.xaxis.matches == "x3"
    assert figure.layout.yaxis3.title.text == "Kurtosis"
    horizontal_gap = figure.layout.xaxis2.domain[0] - figure.layout.xaxis.domain[1]
    vertical_gap = figure.layout.yaxis.domain[0] - figure.layout.yaxis3.domain[1]
    assert horizontal_gap == pytest.approx(
        card_module.FULL_SCREEN_HORIZONTAL_SPACING,
    )
    assert vertical_gap == pytest.approx(card_module.FULL_SCREEN_VERTICAL_SPACING)
    assert horizontal_gap < vertical_gap
    assert [trace.name for trace in figure.data].count("Before") == 3
    assert [trace.name for trace in figure.data].count("After") == 3


@pytest.mark.unit
def test_transformations_do_not_redefine_original_axis_ranges(card_module):
    data, _ = distribution_data()
    original = card_module._analyse_distribution(data, [])
    transformed = card_module._analyse_distribution(
        data, ["Deskew", "Center", "Scale"],
    )

    original_figure = card_module._distribution_figure(
        original.statistics, transformed=False, full_screen=True,
    )
    transformed_figure = card_module._distribution_figure(
        transformed.statistics, transformed=True, full_screen=True,
    )

    for axis in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
        assert list(getattr(original_figure.layout, axis).range) == pytest.approx(
            list(getattr(transformed_figure.layout, axis).range)
        )


def get_card(page: Page):
    return page.locator(".card").first


def by_id(page: Page, local_id: str):
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return page.locator(f"#{card_id.partition('-')[0]}-{local_id}")


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_and_transform_controls_render(
        self, page: Page, app: ShinyAppProc,
    ):
        page.goto(app.url)

        expect(by_id(page, "DistributionChart").locator(".plotly")).to_be_attached(
            timeout=20_000,
        )
        expect(by_id(page, "Check")).to_contain_text(
            "3 predictors are continuous numeric variables", timeout=20_000,
        )
        expect(by_id(page, "Transform")).to_be_attached()
        expect(by_id(page, "IncludeTarget")).to_be_attached()

    @pytest.mark.ui
    def test_selecting_transforms_updates_status_and_flipside(
        self, page: Page, app: ShinyAppProc,
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "3 predictors are continuous numeric variables", timeout=20_000,
        )

        by_id(page, "Transform").locator('input[value="Deskew"]').check(force=True)
        by_id(page, "Transform").locator('input[value="Center"]').check(force=True)
        expect(by_id(page, "Check")).to_contain_text(
            "Transformed 3 predictors using Reduce skew → Mean centre",
            timeout=20_000,
        )

        by_id(page, "FlipButton").click(force=True)
        table = by_id(page, "StatisticsTable")
        expect(table).to_be_visible(timeout=10_000)
        for heading in ("Variable", "Skew before", "Skew after", "Yeo-Johnson lambda"):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("right_skewed")

    @pytest.mark.ui
    def test_target_is_explicitly_included_and_identified(
        self, page: Page, app: ShinyAppProc,
    ):
        page.goto(app.url)
        expect(by_id(page, "IncludeTarget")).to_be_attached(timeout=20_000)

        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page).locator(".collapse-toggle")).to_be_visible()
        get_card(page).locator(".collapse-toggle").click()
        expect(by_id(page, "IncludeTarget")).to_be_visible()
        by_id(page, "IncludeTarget").check()
        expect(by_id(page, "Check")).to_contain_text(
            "3 predictors and 1 target are continuous numeric variables",
            timeout=20_000,
        )
        by_id(page, "Transform").locator('input[value="Scale"]').check(force=True)
        expect(by_id(page, "Check")).to_contain_text(
            "Transformed 3 predictors and 1 target using Common spread",
            timeout=20_000,
        )

        by_id(page, "FlipButton").click(force=True)
        table = by_id(page, "StatisticsTable")
        expect(table).to_contain_text("outcome", timeout=10_000)
        expect(table).to_contain_text("Target")
