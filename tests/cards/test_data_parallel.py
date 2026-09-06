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

app = create_app_fixture(app="../scenarios/data_parallel.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.data_parallel")


@pytest.fixture(scope="module")
def card(card_module):
    return card_module.instance()


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "number": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
        "integer": [10, 20, 30, 40, 50, 60],
        "group": pd.Categorical(
            ["low", "high", "middle", "low", "high", "middle"],
            categories=["low", "middle", "high"],
            ordered=True,
        ),
        "when": pd.date_range("2025-01-01", periods=6, freq="D"),
        "flag": [False, True, False, True, False, True],
    })


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
def test_card_is_immutable_front_only_and_has_complete_ui(card):
    assert card.mutable is False
    assert card.long_name == "Parallel coordinates"
    assert card.hasSidebar()
    assert card.hasFooter()
    assert not card.hasFlipSide()
    assert 'id="Chart"' in str(card.front)
    assert 'id="Check"' in str(card.footer)
    settings = str(card.settings)
    for control in ("Variables", "Colour", "MaxObs"):
        assert f'id="{control}"' in settings


@pytest.mark.unit
def test_eligible_columns_include_supported_scalar_types(card_module):
    frame = sample_frame()
    frame["too_many"] = [f"value-{index}" for index in range(len(frame))]
    frame["shadow__number"] = frame["number"]
    frame["all_missing"] = np.nan

    eligible, excluded = card_module._eligible_columns(
        frame, maximum_levels=3,
    )

    assert eligible == ["number", "integer", "group", "when", "flag"]
    assert "3 observed levels" in excluded["too_many"]
    assert excluded["shadow__number"] == "Shadow variable"
    assert excluded["all_missing"] == "No observed values"


@pytest.mark.unit
def test_complete_cases_remove_missing_and_infinite_values(card_module):
    frame = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, np.inf, 5.0],
        "group": ["A", "A", None, "B", "B"],
    })

    result = card_module._finite_complete_cases(frame)

    assert result.index.tolist() == [0, 4]


@pytest.mark.unit
def test_ordered_categories_and_booleans_receive_labelled_ticks(card_module):
    frame = sample_frame()

    category, category_values = card_module._encode_series(
        frame["group"], "group",
    )
    boolean, boolean_values = card_module._encode_series(
        frame["flag"], "flag",
    )

    assert category["ticktext"] == ["low", "middle", "high"]
    assert category["tickvals"] == [0, 1, 2]
    assert category_values.tolist() == [0, 2, 1, 0, 2, 1]
    assert boolean["ticktext"] == ["False", "True"]
    assert boolean_values.tolist() == [0, 1, 0, 1, 0, 1]


@pytest.mark.unit
def test_datetime_axis_has_readable_ticks_and_numeric_values(card_module):
    dimension, values = card_module._encode_series(
        sample_frame()["when"], "when",
    )

    assert dimension["label"] == "when"
    assert len(dimension["tickvals"]) == 5
    assert dimension["ticktext"][0].startswith("2025-01-01")
    assert np.issubdtype(values.dtype, np.floating)
    assert np.all(np.diff(values) > 0)


@pytest.mark.unit
def test_preparation_uses_colour_for_complete_cases_and_samples_stably(card_module):
    from proxy_data import proxy_data

    frame = sample_frame()
    first = card_module._prepare_parallel_data(
        proxy_data.from_native(frame),
        ["integer", "group"],
        "number",
        maximum_observations=3,
        show_colour_scale=True,
    )
    second = card_module._prepare_parallel_data(
        proxy_data.from_native(frame),
        ["integer", "group"],
        "number",
        maximum_observations=3,
        show_colour_scale=True,
    )

    assert first.source_observations == 6
    assert first.complete_observations == 5
    assert len(first.frame) == 3
    assert first.frame.index.tolist() == second.frame.index.tolist()
    assert len(first.dimensions) == 2
    assert first.colour is not None
    assert first.colour["showscale"] is True


@pytest.mark.unit
def test_categorical_colour_has_discrete_scale_and_labelled_colorbar(card_module):
    line = card_module._colour_line(
        sample_frame()["group"], "group", show_scale=True,
    )

    assert line["showscale"] is True
    assert line["colorbar"]["title"]["text"] == "group"
    assert line["colorbar"]["ticktext"] == ["low", "middle", "high"]
    assert len(line["colorscale"]) == 6
    assert line["colorscale"][0][0] == 0.0
    assert line["colorscale"][-1][0] == 1.0


@pytest.mark.unit
def test_identity_uses_assigned_key_with_row_number_fallback(card_module):
    from proxy_data import proxy_data
    from roles import Role, RoleMap

    frame = pd.DataFrame({
        "site": ["A", "A", None],
        "record": [10, 11, 12],
        "value": [1.0, 2.0, 3.0],
    }, index=[4, 4, 9])
    roles = RoleMap()
    roles.set_roles("site", [Role.IDENTIFIER])
    roles.set_roles("record", [Role.IDENTIFIER])
    data = proxy_data(_df=frame, _roles=roles)

    labels = card_module._observation_identities(data)

    assert labels == [
        "site = A; record = 10",
        "site = A; record = 11",
        "Row 3",
    ]

    roles.set_roles("record", [Role.PREDICTOR])
    without_key = proxy_data(_df=frame, _roles=roles)
    without_key.role_map.clear_roles("site")
    assert card_module._observation_identities(without_key) == [
        "Row 1", "Row 2", "Row 3",
    ]


@pytest.mark.unit
def test_parallel_figure_contains_one_axis_per_selected_variable(card_module):
    from proxy_data import proxy_data

    prepared = card_module._prepare_parallel_data(
        proxy_data.from_native(sample_frame()),
        ["number", "integer", "group", "when", "flag"],
        card_module.NO_COLOUR,
        maximum_observations=100,
        show_colour_scale=False,
    )

    figure = card_module._parallel_figure(prepared, full_screen=False)

    assert len(figure.data) == 1
    assert figure.data[0].type == "parcoords"
    assert list(figure.data[0].customdata) == [
        "Row 1", "Row 2", "Row 3", "Row 4", "Row 6",
    ]
    assert [axis.label for axis in figure.data[0].dimensions] == [
        "number", "integer", "group", "when", "flag",
    ]
    assert figure.layout.font.size == 10


@pytest.mark.unit
def test_parallel_figure_reports_too_few_axes_and_no_complete_rows(card_module):
    from proxy_data import proxy_data

    one_axis = card_module._prepare_parallel_data(
        proxy_data.from_native(sample_frame()),
        ["number"],
        card_module.NO_COLOUR,
        maximum_observations=100,
        show_colour_scale=False,
    )
    incompatible = pd.DataFrame({
        "x": [1.0, np.nan],
        "y": [np.nan, 2.0],
    })
    no_rows = card_module._prepare_parallel_data(
        proxy_data.from_native(incompatible),
        ["x", "y"],
        card_module.NO_COLOUR,
        maximum_observations=100,
        show_colour_scale=False,
    )

    one_axis_figure = card_module._parallel_figure(one_axis, full_screen=False)
    no_rows_figure = card_module._parallel_figure(no_rows, full_screen=False)

    assert "at least two" in one_axis_figure.layout.annotations[0].text.lower()
    assert "no observations" in no_rows_figure.layout.annotations[0].text.lower()


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_status_and_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)

        expect(by_id(page, "Chart").locator(".plotly")).to_be_attached(
            timeout=15_000,
        )
        expect(by_id(page, "Check")).to_contain_text(
            "Showing 17 observations; 1 incomplete observation omitted.",
            timeout=15_000,
        )
        for control in ("Variables", "Colour", "MaxObs"):
            expect(by_id(page, control)).to_be_attached()
        expect(get_card(page).locator(".parallel-axis-mode")).to_have_count(0)
        expect(get_card(page).locator(".parallel-reset-zoom")).to_have_count(0)
        expect(by_id(page, "FlipButton")).to_have_count(0)

    @pytest.mark.ui
    def test_hovering_a_line_displays_its_identifier(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        plot = by_id(page, "Chart").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=15_000)

        point = plot.evaluate(
            """
            plot => {
                const trace = plot.data[0];
                const dimensions = trace.dimensions;
                const byLabel = new Map(
                    dimensions.map(dimension => [String(dimension.label), dimension])
                );
                const axes = Array.from(
                    plot.querySelectorAll('.parcoords-control-view .y-axis')
                ).map(axis => {
                    const title = axis.querySelector('.axis-title');
                    const brush = axis.querySelector('.axis-brush .background');
                    const label = title.getAttribute('data-unformatted')
                        || title.textContent;
                    const rectangle = brush.getBoundingClientRect();
                    return {
                        dimension: byLabel.get(String(label)),
                        x: rectangle.left + rectangle.width / 2,
                        top: rectangle.top,
                        bottom: rectangle.bottom,
                    };
                }).sort((left, right) => left.x - right.x);
                const row = 0;
                const ordinate = axis => {
                    const [low, high] = axis.dimension.range;
                    const value = Number(axis.dimension.values[row]);
                    return axis.bottom - (value - low) / (high - low)
                        * (axis.bottom - axis.top);
                };
                const left = axes[0];
                const right = axes[1];
                const clientX = (left.x + right.x) / 2;
                const clientY = (ordinate(left) + ordinate(right)) / 2;
                plot.dispatchEvent(new MouseEvent('mousemove', {
                    bubbles: true,
                    clientX,
                    clientY,
                }));
                return {
                    identity: trace.customdata[row],
                    x: clientX,
                    y: clientY,
                };
            }
            """
        )

        tooltip = get_card(page).locator(".parallel-hover-tooltip")
        expect(tooltip).to_be_visible(timeout=5_000)
        expect(tooltip).to_have_text(point["identity"])
        assert point["identity"] == "high_cardinality = id-01"

        plot.dispatch_event(
            "click",
            {"clientX": point["x"], "clientY": point["y"]},
        )

        comparison = get_card(page).locator(".parallel-comparison-label")
        expect(comparison).to_be_visible(timeout=5_000)
        expect(comparison).to_contain_text(point["identity"])
        expect(get_card(page).locator(".parallel-comparison-overlay")).to_be_visible()

        get_card(page).locator(".parallel-clear-comparison").click()
        expect(comparison).to_be_hidden()
        expect(get_card(page).locator(".parallel-comparison-overlay")).to_be_hidden()

    @pytest.mark.ui
    def test_variable_selection_changes_complete_case_population(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "Showing 17 observations", timeout=15_000,
        )

        set_shiny_input(page, "Variables", ["score", "group"])

        expect(by_id(page, "Check")).to_contain_text(
            "Showing 18 observations.", timeout=15_000,
        )

    @pytest.mark.ui
    def test_colour_scale_is_shown_after_expanding(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(by_id(page, "Chart").locator(".plotly")).to_be_attached(
            timeout=15_000,
        )

        set_shiny_input(page, "Colour", "group")
        by_id(page, "ExpandButton").click(force=True)

        plot = by_id(page, "Chart").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=15_000)
        selector = f"#{namespaced_id(page, 'Chart')} .js-plotly-plot"
        page.wait_for_function(
            """
            selector => document.querySelector(selector)?.data?.[0]
                ?.line?.showscale === true
            """,
            arg=selector,
            timeout=15_000,
        )
        assert plot.evaluate("element => element.data[0].line.showscale") is True
        expect(get_card(page)).to_be_visible()
