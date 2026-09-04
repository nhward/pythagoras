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

from proxy_data import proxy_data
from roles import Role, RoleMap

app = create_app_fixture(app="../scenarios/obs_duplicates.py", scope="function")


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
    return importlib.import_module("cards.obs_duplicates")


def duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "A": [1, 1, 1, 1, 9],
        "B": ["x", "x", "x", "y", "z"],
        "C": [10, 10, 11, 12, 99],
    })


@pytest.mark.unit
def test_card_is_mutable_and_has_expected_regions(card_module):
    card = card_module.instance()

    assert card.mutable is True
    assert card.long_name == "Observation duplicates"
    assert 'id="BarChart"' in str(card.front)
    assert 'id="Table"' in str(card.back)
    footer = str(card.footer)
    assert 'id="Busy"' in footer
    assert 'id="RemoveExact"' in footer
    assert 'id="Check"' in footer
    settings = str(card.settings)
    assert 'id="SignificantFigures"' in settings
    assert 'id="MaxDifferences"' in settings


@pytest.mark.unit
def test_eligible_columns_exclude_roles_and_shadow_variables(card_module):
    frame = pd.DataFrame({
        "value": [1],
        "target": [2],
        "id": [3],
        "weight": [1.0],
        "geometry": ["POINT (0 0)"],
        "unused": [4],
        "shadow__value": [False],
    })
    roles = RoleMap()
    roles.set_roles("value", [Role.PREDICTOR])
    roles.set_roles("target", [Role.TARGET])
    roles.set_roles("id", [Role.IDENTIFIER])
    roles.set_roles("weight", [Role.WEIGHTING])
    roles.set_roles("geometry", [Role.GEOMETRY])
    roles.set_roles("unused", [Role.NONE])
    roles.set_roles("shadow__value", [Role.PREDICTOR])
    proxy = proxy_data(_df=frame, _roles=roles)

    assert card_module._eligible_columns(proxy) == ["value", "target"]


@pytest.mark.unit
def test_deduplicate_proxy_removes_later_exact_rows_and_preserves_metadata(
    card_module,
):
    frame = duplicate_frame()
    proxy = proxy_data(_df=frame, _name="Duplicates")

    result = card_module._deduplicate_proxy(proxy, significant_figures=16)

    assert result is not proxy
    assert result.name == "Duplicates"
    assert result.role_map == proxy.role_map
    assert result.frame.index.tolist() == [0, 2, 3, 4]
    assert len(proxy.frame) == 5
    assert len(result.cleaning_records) == 1
    record = result.cleaning_records[0]
    assert record.card == "obs_duplicates"
    assert record.parameters == {"significant_figures": 16}
    assert record.input_shape == (5, 3)
    assert record.output_shape == (4, 3)
    recalculated = card_module._duplicate_results(
        result.frame, maximum_differences=2
    )
    assert recalculated.loc[0, "Count"] == 0


@pytest.mark.unit
def test_deduplication_respects_significant_figure_rounding(card_module):
    proxy = proxy_data(_df=pd.DataFrame({
        "value": [1234.4, 1234.5, 9999.0],
        "group": ["A", "A", "B"],
    }))

    precise = card_module._deduplicate_proxy(proxy, significant_figures=16)
    rounded = card_module._deduplicate_proxy(proxy, significant_figures=3)

    assert len(precise.frame) == 3
    assert len(rounded.frame) == 2


@pytest.mark.unit
def test_significant_rounding_changes_floats_but_not_integers(card_module):
    frame = pd.DataFrame({
        "float": [1234.5, 0.012345],
        "integer": pd.Series([1234, 12], dtype="int64"),
        "text": ["a", "b"],
    })

    result = card_module._round_significant(frame, 3)

    assert result["float"].tolist() == pytest.approx([1230.0, 0.0123])
    assert result["integer"].tolist() == [1234, 12]
    assert result["text"].tolist() == ["a", "b"]


@pytest.mark.unit
def test_duplicate_rows_are_assigned_to_minimum_difference(card_module):
    result = card_module._duplicate_results(
        duplicate_frame(), maximum_differences=2
    )

    assert result["Differences tolerated"].tolist() == [0, 1, 2]
    assert result["Count"].tolist() == [1, 1, 1]
    assert result["Redundant row numbers"].tolist() == ["2", "3", "4"]


@pytest.mark.unit
def test_container_values_can_be_compared(card_module):
    frame = pd.DataFrame({
        "items": [[1, 2], [1, 2], [3]],
        "metadata": [{"a": 1}, {"a": 1}, {"a": 2}],
    })

    result = card_module._duplicate_results(frame, maximum_differences=0)

    assert result.iloc[0]["Count"] == 1
    assert result.iloc[0]["Redundant row numbers"] == "2"


@pytest.mark.unit
def test_combination_count_and_guard(card_module):
    assert card_module._combination_count(5, 2) == 16

    with pytest.raises(ValueError, match="column combinations"):
        card_module._duplicate_results(
            pd.DataFrame(np.zeros((2, 10))),
            maximum_differences=3,
            maximum_combinations=10,
        )


@pytest.mark.unit
def test_empty_comparison_returns_schema(card_module):
    result = card_module._duplicate_results(
        pd.DataFrame(index=range(3)), maximum_differences=2
    )

    assert result.empty
    assert list(result.columns) == card_module.RESULT_COLUMNS


@pytest.mark.unit
def test_bar_chart_represents_each_tolerance_level(card_module):
    results = card_module._duplicate_results(
        duplicate_frame(), maximum_differences=2
    )

    figure = card_module._duplicates_figure(results)

    assert len(figure.data) == 1
    assert figure.data[0].type == "bar"
    assert list(figure.data[0].x) == ["0", "1", "2"]
    assert list(figure.data[0].y) == [1, 1, 1]
    assert figure.layout.xaxis.title.text == "Number of differences tolerated"


@pytest.mark.unit
def test_chart_is_empty_when_no_duplicates_or_near_duplicates(card_module):
    results = card_module._duplicate_results(
        pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
        }),
        maximum_differences=1,
    )

    figure = card_module._duplicates_figure(results)

    assert len(figure.data) == 0
    assert len(figure.layout.annotations) == 1
    assert (
        figure.layout.annotations[0].text
        == "No duplicates or near duplicates"
    )


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_and_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)

        expect(by_id(page, "BarChart").locator(".plotly")).to_be_attached(
            timeout=15_000
        )
        expect(by_id(page, "Check")).to_contain_text(
            "1 redundant exact-duplicate row", timeout=15_000
        )
        expect(by_id(page, "SignificantFigures")).to_be_attached()
        expect(by_id(page, "MaxDifferences")).to_be_attached()
        expect(by_id(page, "RemoveExact")).to_be_attached()

    @pytest.mark.ui
    def test_flip_displays_duplicate_row_table(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "1 redundant exact-duplicate row", timeout=15_000
        )

        by_id(page, "FlipButton").click(force=True)

        expect(by_id(page, "Table2")).to_be_visible(timeout=10_000)
        table = by_id(page, "Table2")
        for heading in (
            "Differences tolerated", "Redundant row numbers", "Count"
        ):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("2")

    @pytest.mark.ui
    def test_remove_exact_duplicates_recalculates_outputs(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "1 redundant exact-duplicate row", timeout=15_000
        )

        set_shiny_input(page, "RemoveExact", ["Exact duplicates"])

        expect(by_id(page, "Check")).to_contain_text(
            "Removed 1 redundant exact-duplicate row; 4 observations remain",
            timeout=15_000,
        )
        by_id(page, "FlipButton").click(force=True)
        expect(by_id(page, "Table2")).to_be_visible(timeout=10_000)
        expect(by_id(page, "Table2")).to_contain_text("0")

    @pytest.mark.ui
    def test_fullscreen_keeps_duplicate_chart_available(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "BarChart").locator(".plotly")).to_be_attached(
            timeout=15_000
        )

        by_id(page, "ExpandButton").click(force=True)

        expect(get_card(page)).to_be_visible()
        expect(by_id(page, "BarChart")).to_be_visible()
