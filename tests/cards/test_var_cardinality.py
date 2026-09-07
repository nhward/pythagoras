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

app = create_app_fixture(app="../scenarios/var_cardinality.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.var_cardinality")


@pytest.fixture(scope="module")
def card(card_module):
    return card_module.instance()


def profile_data():
    from proxy_data import proxy_data
    from roles import Role, RoleMap
    from text_pandas import as_text

    frame = pd.DataFrame({
        "constant": [2.5] * 8,
        "decimal": np.resize([0.1, 0.2, 0.3], 8),
        "integer": np.arange(8),
        "category": pd.Categorical(np.resize(["A", "B"], 8)),
        "code": pd.Series([f"id-{index}" for index in range(8)], dtype="string"),
        "all_missing": np.full(8, np.nan),
        "metadata": [{"source": index % 2} for index in range(8)],
        "shadow__decimal": np.arange(8, dtype=float),
    })
    frame["prose"] = as_text([
        f"This is distinct explanatory prose number {index}." for index in range(8)
    ])
    roles = RoleMap()
    for column in frame.columns:
        roles.set_roles(column, [Role.PREDICTOR])
    roles.set_roles("code", [Role.IDENTIFIER, Role.SENSITIVE])
    return proxy_data(_df=frame, _roles=roles, _name="Cardinality unit test")


def analyze(card_module, data=None, maximum=100):
    return card_module._analyse_cardinality(
        profile_data() if data is None else data,
        maximum_observations=maximum,
        low_threshold=4,
        high_threshold=50,
    )


def get_card(page: Page):
    return page.locator(".card").first


def namespaced_id(page: Page, local_id: str) -> str:
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return f"{card_id.partition('-')[0]}-{local_id}"


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


@pytest.mark.unit
def test_card_is_immutable_with_chart_table_footer_and_settings(card):
    assert card.mutable is False
    assert card.long_name == "Variable cardinality"
    assert card.hasFlipSide()
    assert card.hasFooter()
    assert card.hasSidebar()
    assert 'id="Chart"' in str(card.front)
    assert 'id="Profile"' in str(card.back)
    assert 'id="Check"' in str(card.footer)
    settings = str(card.settings)
    for control in (
        "ShowThresholds", "Logarithmic", "Thresholds", "MaxObs", "Ordering",
    ):
        assert f'id="{control}"' in settings


@pytest.mark.unit
def test_sample_positions_are_deterministic_sorted_and_unique(card_module):
    first = card_module._sample_positions(1_000, 25)
    second = card_module._sample_positions(1_000, 25)

    assert first.tolist() == second.tolist()
    assert len(first) == len(np.unique(first)) == 25
    assert np.all(np.diff(first) > 0)
    assert card_module._sample_positions(4, 10).tolist() == [0, 1, 2, 3]


@pytest.mark.unit
def test_bounded_cardinality_is_exact_only_within_the_limit(card_module):
    exact = card_module._bounded_cardinality(pd.Series(["a", "b", "a"]), 2)
    bounded = card_module._bounded_cardinality(pd.Series(range(20)), 4)

    assert exact == (2, True)
    assert bounded == (5, False)


@pytest.mark.unit
def test_assessment_respects_type_cardinality_and_identifier_role(card_module):
    arguments = {
        "cardinality_exact": True,
        "uniqueness": 1.0,
        "exact_unique": True,
        "low_threshold": 4,
        "high_threshold": 50,
    }

    decimal = card_module._assessment(
        kind="decimal", cardinality=3, role="Predictor", **arguments,
    )
    nominal_predictor = card_module._assessment(
        kind="nominal", cardinality=100, role="Predictor", **arguments,
    )
    nominal_identifier = card_module._assessment(
        kind="nominal", cardinality=100, role="Identifier", **arguments,
    )

    assert decimal[0] == "Strong review"
    assert nominal_predictor[0] == "Strong review"
    assert "identifier" in nominal_predictor[2].lower()
    assert nominal_identifier[0] == "Expected"


@pytest.mark.unit
def test_analysis_profiles_roles_types_missingness_and_omits_shadows(card_module):
    result = analyze(card_module)
    table = result.profiles.set_index("Variable")

    assert "shadow__decimal" not in table.index
    assert table.loc["code", "Role"] == "Identifier, Sensitive"
    assert table.loc["prose", "Semantic type"] == "free text"
    assert table.loc["all_missing", "Missing"] == 8
    assert table.loc["all_missing", "Finding"] == "Not assessed"
    assert table.loc["metadata", "Semantic type"] == "object"
    assert table.loc["metadata", "Finding"] == "Not assessed"
    assert table.loc["constant", "Finding"] == "Strong review"
    assert table.loc["category", "Finding"] == "Expected"
    assert table.loc["code", "Finding"] == "Expected"


@pytest.mark.unit
def test_large_data_uses_stable_sample_and_reports_a_lower_bound(card_module):
    from proxy_data import proxy_data

    frame = pd.DataFrame({"value": np.arange(1_000)})
    data = proxy_data.from_native(frame)
    first = analyze(card_module, data, maximum=40)
    second = analyze(card_module, data, maximum=40)
    row = first.profiles.iloc[0]

    assert first.sampled is True
    assert first.sampled_observations == 40
    assert first.source_observations == 1_000
    assert row["Distinct"] == "≥51"
    assert row["Basis"] == "Sample/lower bound"
    pd.testing.assert_frame_equal(first.profiles, second.profiles)


@pytest.mark.unit
def test_profile_ordering_and_row_styles_follow_findings(card_module):
    profiles = pd.DataFrame({
        "Variable": ["expected", "review", "strong", "unassessed"],
        "Distinct value": [2, 10, 1, 0],
        "Finding": ["Expected", "Review", "Strong review", "Not assessed"],
    })

    ordered = card_module._ordered_profiles(profiles, "finding")
    styles = card_module._profile_row_styles(ordered)

    assert ordered["Variable"].tolist() == [
        "strong", "review", "expected", "unassessed",
    ]
    assert styles == [
        {"rows": [0], "class": "var-cardinality-strong-review-row"},
        {"rows": [1], "class": "var-cardinality-review-row"},
        {"rows": [2], "class": "var-cardinality-expected-row"},
        {"rows": [3], "class": "var-cardinality-not-assessed-row"},
    ]


@pytest.mark.unit
def test_figure_contains_roles_palette_hover_and_bounded_log_range(card_module):
    result = analyze(card_module)
    figure = card_module._cardinality_figure(
        result,
        ordering="original",
        logarithmic=True,
        show_thresholds=True,
        low_threshold=4,
        high_threshold=50,
        full_screen=False,
    )
    bars = figure.data[0]
    table = result.profiles

    assert bars.type == "bar"
    assert list(bars.y) == table["Variable"].tolist()
    assert list(bars.marker.color) == table["Finding"].map(
        card_module.STATUS_COLOURS,
    ).tolist()
    assert list(bars.customdata[:, 4]) == table["Role"].tolist()
    assert "Role: %{customdata[4]}" in bars.hovertemplate
    assert figure.layout.xaxis.type == "log"
    assert list(figure.layout.xaxis.range) == [0.0, 1.0]
    assert len(figure.layout.shapes) == 2
    assert {trace.name for trace in figure.data[1:]} == set(
        card_module.STATUS_COLOURS,
    )


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_settings_and_hover_data_render(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)

        plot = by_id(page, "Chart").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=20_000)
        expect(by_id(page, "Check")).to_contain_text(
            "Strong review: 3; review: 1; expected: 3; not assessed: 2",
            timeout=20_000,
        )
        for control in (
            "ShowThresholds", "Logarithmic", "Thresholds", "MaxObs", "Ordering",
        ):
            expect(by_id(page, control)).to_be_attached()

        plot_data = plot.evaluate(
            """
            element => ({
                variables: Array.from(element.data[0].y),
                roles: Array.from(element.data[0].customdata, row => row[4]),
                hovertemplate: element.data[0].hovertemplate,
                axisType: element.layout.xaxis.type,
            })
            """
        )
        code_position = plot_data["variables"].index("record_code")
        assert plot_data["roles"][code_position] == "Identifier"
        assert "Role: %{customdata[4]}" in plot_data["hovertemplate"]
        assert plot_data["axisType"] == "log"

    @pytest.mark.ui
    def test_flip_table_has_roles_findings_and_contrast_safe_colors(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "Strong review: 3", timeout=20_000,
        )

        by_id(page, "FlipButton").click(force=True)
        table = by_id(page, "ProfileTable")
        expect(table).to_be_visible(timeout=10_000)
        for heading in (
            "Variable", "Role", "Semantic type", "Distinct", "Finding",
            "Explanation", "Role consideration",
        ):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("Identifier")
        expect(table).to_contain_text("Sensitive")

        strong_cell = table.locator(
            ".var-cardinality-strong-review-row"
        ).first
        expected_cell = table.locator(
            ".var-cardinality-expected-row"
        ).first
        expect(strong_cell).to_be_visible()
        expect(expected_cell).to_be_visible()
        assert strong_cell.evaluate(
            "element => getComputedStyle(element).backgroundColor"
        ) == "rgb(217, 95, 2)"
        assert strong_cell.evaluate(
            "element => getComputedStyle(element).color"
        ) == "rgb(0, 0, 0)"
        assert expected_cell.evaluate(
            "element => getComputedStyle(element).backgroundColor"
        ) == "rgb(57, 120, 168)"
        assert expected_cell.evaluate(
            "element => getComputedStyle(element).color"
        ) == "rgb(255, 255, 255)"

    @pytest.mark.ui
    def test_ordering_setting_reorders_chart_without_changing_findings(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        plot = by_id(page, "Chart").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=20_000)

        set_shiny_input(page, "Ordering", "original")
        page.wait_for_function(
            """
            selector => document.querySelector(selector)?.data?.[0]?.y?.[0]
                === "constant_decimal"
            """,
            arg=f"#{namespaced_id(page, 'Chart')} .js-plotly-plot",
            timeout=20_000,
        )
        assert plot.evaluate(
            "element => Array.from(element.data[0].y)"
        )[:3] == ["constant_decimal", "few_decimal", "count"]
        expect(by_id(page, "Check")).to_contain_text("Strong review: 3")
