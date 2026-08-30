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

app = create_app_fixture(app="../scenarios/miss_sets.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


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


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.miss_sets")


def patterned_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "A": [np.nan, np.nan, 3.0, np.nan, 5.0, 6.0],
        "B": [np.nan, np.nan, 3.0, 4.0, np.nan, 6.0],
        "C": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "complete": range(6),
    })


@pytest.mark.unit
def test_card_is_immutable_and_has_complete_ui(card_module):
    card = card_module.instance()

    assert card.mutable is False
    assert card.long_name == "Missingness sets"
    assert 'id="Upset"' in str(card.front)
    assert 'id="Table"' in str(card.back)
    assert 'id="Check"' in str(card.footer)
    settings = str(card.settings)
    assert 'id="MaxIntersections"' in settings
    assert 'id="MaxVariables"' in settings
    assert 'id="MinCount"' in settings
    assert 'id="MaxObs"' in settings


@pytest.mark.unit
def test_missing_variables_are_ordered_by_frequency_then_name(card_module):
    assert card_module._missing_variables(patterned_frame()) == ["A", "B", "C"]


@pytest.mark.unit
def test_intersections_count_exact_missingness_patterns(card_module):
    result = card_module._intersection_counts(patterned_frame(), ["A", "B", "C"])

    assert list(result["Observations"]) == [2, 1, 1, 1]
    assert result.iloc[0]["Missing variables"] == "A, B"
    assert result.iloc[0]["_membership"] == ("A", "B")
    assert result.iloc[0]["Degree"] == 2
    assert result.iloc[0]["Proportion"] == pytest.approx(2 / 6)
    assert result["Observations"].sum() == 5


@pytest.mark.unit
def test_complete_rows_are_not_intersections(card_module):
    frame = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    result = card_module._intersection_counts(frame, ["A", "B"])

    assert result.empty
    assert list(result.columns) == card_module.INTERSECTION_COLUMNS


@pytest.mark.unit
def test_intersection_selection_applies_count_and_display_limits(card_module):
    counts = card_module._intersection_counts(patterned_frame(), ["A", "B", "C"])

    selected = card_module._select_intersections(
        counts, maximum=1, minimum_count=2
    )

    assert len(selected) == 1
    assert selected.iloc[0]["Missing variables"] == "A, B"


@pytest.mark.unit
def test_upset_figure_contains_bars_matrix_and_active_memberships(card_module):
    frame = patterned_frame()
    variables = card_module._missing_variables(frame)
    intersections = card_module._intersection_counts(frame, variables)

    figure = card_module._upset_figure(frame, variables, intersections)

    trace_types = [trace.type for trace in figure.data]
    assert trace_types.count("bar") == 2
    assert trace_types.count("scatter") >= 1 + len(intersections)
    assert list(figure.data[0].y) == list(intersections["Observations"])
    assert figure.layout.xaxis.title.text is None
    assert figure.layout.yaxis3.showticklabels is False
    assert tuple(figure.layout.yaxis2.range) == tuple(figure.layout.yaxis3.range)
    assert list(figure.data[1].customdata) == variables


@pytest.mark.unit
def test_upset_memberships_grow_in_full_screen(card_module):
    frame = patterned_frame()
    variables = card_module._missing_variables(frame)
    intersections = card_module._intersection_counts(frame, variables)

    card_figure = card_module._upset_figure(frame, variables, intersections)
    full_figure = card_module._upset_figure(
        frame, variables, intersections, full_screen=True
    )

    assert full_figure.data[2].marker.size > card_figure.data[2].marker.size
    card_active = next(
        trace for trace in card_figure.data
        if trace.type == "scatter" and trace.mode == "markers"
        and trace.marker.color == "#7b3f00"
    )
    full_active = next(
        trace for trace in full_figure.data
        if trace.type == "scatter" and trace.mode == "markers"
        and trace.marker.color == "#7b3f00"
    )
    assert full_active.marker.size > card_active.marker.size
    card_connector = next(
        trace for trace in card_figure.data
        if trace.type == "scatter" and trace.mode == "lines"
    )
    full_connector = next(
        trace for trace in full_figure.data
        if trace.type == "scatter" and trace.mode == "lines"
    )
    assert full_connector.line.width > card_connector.line.width


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"A": [1, 2], "B": [3, 4]}), "At least two"),
        (pd.DataFrame({"A": [np.nan, 2], "B": [3, 4]}), "At least two"),
    ],
)
def test_upset_figure_has_informative_empty_state(card_module, frame, message):
    variables = card_module._missing_variables(frame)
    intersections = card_module._intersection_counts(frame, variables)

    figure = card_module._upset_figure(frame, variables, intersections)

    assert message in figure.layout.annotations[0].text


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_and_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)

        expect(by_id(page, "Upset").locator(".plotly")).to_be_attached(
            timeout=15_000
        )
        expect(by_id(page, "Check")).to_contain_text(
            "Showing 4 intersections across 3 missing variables",
            timeout=15_000,
        )
        expect(by_id(page, "MaxIntersections")).to_be_attached()
        expect(by_id(page, "MaxVariables")).to_be_attached()
        expect(by_id(page, "MinCount")).to_be_attached()

    @pytest.mark.ui
    def test_flip_displays_intersection_table(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("Showing 4", timeout=15_000)

        by_id(page, "FlipButton").click(force=True)

        expect(by_id(page, "Table2")).to_be_visible(timeout=10_000)
        table = by_id(page, "Table2")
        for heading in (
            "Missing variables", "Degree", "Observations", "Proportion"
        ):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("A, B")

    @pytest.mark.ui
    def test_minimum_count_filters_intersections(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("Showing 4", timeout=15_000)

        set_shiny_input(page, "MinCount", 2)

        expect(by_id(page, "Check")).to_contain_text(
            "Showing 1 intersection", timeout=15_000
        )

    @pytest.mark.ui
    def test_fullscreen_keeps_upset_chart_available(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Upset").locator(".plotly")).to_be_attached(
            timeout=15_000
        )

        by_id(page, "ExpandButton").click(force=True)

        expect(get_card(page)).to_be_visible()
        expect(by_id(page, "Upset")).to_be_visible()
