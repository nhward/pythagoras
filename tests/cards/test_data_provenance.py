from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc
from sklearn.preprocessing import StandardScaler


path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/data_provenance.py", scope="function")


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.data_provenance")


def journey_proxy():
    from proxy_data import proxy_data

    source_frame = pd.DataFrame({
        "value": [1.0, 2.0, 3.0],
        "discard": [9, 9, 9],
    })
    source = proxy_data(_df=source_frame, _name="Example")
    cleaned = source.with_cleaned_data(
        source_frame.drop(columns="discard"),
        card="var_modify",
        operation="Remove unused variable",
        parameters={"variable": "discard", "reason": "Unused"},
    )
    scaler = StandardScaler().set_output(transform="pandas")
    preview = scaler.fit_transform(cleaned.frame)
    return cleaned.with_pipeline_step(
        scaler,
        name="var_transform",
        preview_frame=preview,
    )


@pytest.mark.unit
def test_card_is_immutable_with_chart_table_and_status(card_module):
    card = card_module.instance()

    assert card.mutable is False
    assert card.long_name == "Data journey"
    assert 'id="JourneyChart"' in str(card.front)
    assert 'class="journey-chart-scroll html-fill-item"' in str(card.front)
    assert 'id="JourneyTable"' in str(card.back)
    assert 'id="Status"' in str(card.footer)
    assert card.hasSidebar()
    assert 'id="HideInactive"' in str(card.settings)
    assert 'checked="checked"' in str(card.settings).lower()


@pytest.mark.unit
def test_table_has_source_cleaning_learning_and_preview_rows(card_module):
    table = card_module._journey_table(journey_proxy())

    assert table.columns.tolist() == card_module.JOURNEY_COLUMNS
    assert table["Stage"].tolist() == [
        "Source", "Cleaning", "Learning", "Preview",
    ]
    assert table["Attempted"].tolist() == ["", "Yes", "Yes", ""]
    assert table["Step"].tolist() == [0, 1, 2, 3]
    cleaning = table.loc[table["Stage"].eq("Cleaning")].iloc[0]
    assert cleaning["Card"] == "var_modify"
    assert cleaning["Variables"] == "discard"
    assert cleaning["Input shape"] == "3 × 2"
    assert cleaning["Output shape"] == "3 × 1"
    assert cleaning["Variable change"] == -1
    assert '"reason": "Unused"' in cleaning["Parameters"]

    learning = table.loc[table["Stage"].eq("Learning")].iloc[0]
    assert learning["Card"] == "var_transform"
    assert learning["Method"] == "StandardScaler"
    assert learning["Input shape"] == learning["Output shape"] == "3 × 1"
    assert learning["Row change"] == 0


@pytest.mark.unit
def test_filter_uses_attempted_status_not_shape_change(card_module):
    from proxy_data import proxy_data

    source = proxy_data(pd.DataFrame({"value": [1, 2]}))
    attempted_no_op = source.with_cleaned_data(
        source.frame,
        card="obs_duplicates",
        operation="Remove exact duplicate observations",
    )
    with_inactive = attempted_no_op.with_inactive_step(
        stage="Cleaning",
        card="miss_placeholders",
        operation="Replace missing-value placeholders",
    )
    table = card_module._journey_table(with_inactive)

    visible = card_module._visible_journey(table, hide_inactive=True)
    all_steps = card_module._visible_journey(table, hide_inactive=False)

    assert "Remove exact duplicate observations" in set(visible["Operation"])
    assert "Replace missing-value placeholders" not in set(visible["Operation"])
    assert "Replace missing-value placeholders" in set(all_steps["Operation"])


@pytest.mark.unit
def test_variable_modifications_are_formatted_as_a_readable_list(card_module):
    text = card_module._parameters_text({
        "changes": [
            {
                "variable": "age",
                "original_type": "str",
                "new_type": "int",
                "original_order": None,
                "new_order": None,
            },
            {"variable": "code", "new_name": "region_code"},
        ],
    })

    assert text.splitlines() == [
        "• age: type str → int",
        "• code: rename to region_code",
    ]
    assert "{" not in text


@pytest.mark.unit
def test_empty_history_still_explains_source_and_preview(card_module):
    from proxy_data import proxy_data

    table = card_module._journey_table(
        proxy_data(pd.DataFrame({"value": [1, 2]}), _name="Plain"),
    )

    assert table["Stage"].tolist() == ["Source", "Preview"]
    assert table.iloc[0]["Output shape"] == "2 × 1"
    assert table.iloc[-1]["Method"] == "Full-data preview only"


@pytest.mark.unit
def test_chart_colours_stages_and_hover_contains_table_fields(card_module):
    table = card_module._journey_table(journey_proxy())
    figure = card_module._journey_figure(table)

    traces = {trace.name: trace for trace in figure.data if trace.name}
    assert set(traces) == {"Source", "Cleaning", "Learning", "Preview"}
    for stage, colour in card_module.STAGE_COLOURS.items():
        assert traces[stage].marker.color == colour
    learning_hover = traces["Learning"].hovertemplate
    for column in ("Stage", "Card", "Operation", "Method", "Variables"):
        assert f"{column}:" in learning_hover
    assert len(figure.layout.annotations) == 2 * len(table) - 1
    assert figure.layout.legend.orientation == "h"


@pytest.mark.unit
def test_chart_abbreviates_long_hover_values_without_changing_table(card_module):
    table = card_module._journey_table(journey_proxy())
    learning_row = table.index[table["Stage"].eq("Learning")][0]
    full_variables = ", ".join(f"variable_{number}" for number in range(20))
    table.loc[learning_row, "Variables"] = full_variables

    figure = card_module._journey_figure(table)

    learning_trace = next(
        trace for trace in figure.data if trace.name == "Learning"
    )
    variable_index = [
        column for column in card_module.JOURNEY_COLUMNS if column != "Step"
    ].index("Variables")
    hover_variables = learning_trace.customdata[0][variable_index]
    assert len(hover_variables) <= card_module.HOVER_VALUE_MAX_LENGTH
    assert hover_variables.endswith("...")
    assert table.loc[learning_row, "Variables"] == full_variables


@pytest.mark.unit
def test_full_screen_allows_larger_nodes(card_module):
    table = card_module._journey_table(journey_proxy())

    card_figure = card_module._journey_figure(table, full_screen=False)
    full_figure = card_module._journey_figure(table, full_screen=True)
    card_nodes = next(trace for trace in card_figure.data if trace.name == "Cleaning")
    full_nodes = next(trace for trace in full_figure.data if trace.name == "Cleaning")

    assert full_nodes.marker.size > card_nodes.marker.size


@pytest.mark.unit
def test_chart_wraps_after_six_steps_and_reverses_each_row(card_module):
    x, y = card_module._journey_positions(
        14, steps_per_row=card_module.CARD_STEPS_PER_ROW,
    )

    assert x.tolist() == [
        0, 1, 2, 3, 4, 5,
        5, 4, 3, 2, 1, 0,
        0, 1,
    ]
    assert y.tolist() == [
        0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1,
        -2, -2,
    ]


@pytest.mark.unit
def test_full_screen_uses_fewer_rows(card_module):
    table = pd.concat(
        [card_module._journey_table(journey_proxy())] * 4,
        ignore_index=True,
    )
    table["Step"] = range(len(table))

    card_figure = card_module._journey_figure(table, full_screen=False)
    full_figure = card_module._journey_figure(table, full_screen=True)
    card_path = card_figure.data[0]
    full_path = full_figure.data[0]

    assert len(set(card_path.y)) == 3
    assert len(set(full_path.y)) == 2
    assert full_figure.layout.height < card_figure.layout.height


def get_card(page: Page):
    return page.locator(".card").first


def by_id(page: Page, local_id: str):
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return page.locator(f"#{card_id.partition('-')[0]}-{local_id}")


class TestWebKitUI:
    @pytest.mark.ui
    def test_flow_chart_and_status_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)

        expect(by_id(page, "JourneyChart").locator(".plotly")).to_be_attached(
            timeout=20_000,
        )
        expect(by_id(page, "Status")).to_contain_text(
            "1 cleaning step; 1 learning step", timeout=20_000,
        )

    @pytest.mark.ui
    def test_flip_shows_the_corresponding_rows(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "JourneyChart").locator(".plotly")).to_be_attached(
            timeout=20_000,
        )
        by_id(page, "FlipButton").click(force=True)

        table = by_id(page, "JourneyTable")
        expect(table).to_be_visible(timeout=10_000)
        for value in ("Source", "Cleaning", "Learning", "Preview"):
            expect(table).to_contain_text(value)
        expect(table).to_contain_text("Remove unused variable")
        expect(table).to_contain_text("VariableTransformStep")
        expect(table).not_to_contain_text("Remove exact duplicate observations")

        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page).locator(".collapse-toggle")).to_be_visible()
        get_card(page).locator(".collapse-toggle").click()
        by_id(page, "HideInactive").uncheck()
        expect(table).to_contain_text(
            "Remove exact duplicate observations", timeout=10_000,
        )
