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

app = create_app_fixture(app="../scenarios/miss_map.py", scope="function")


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.miss_map")


def frame():
    return pd.DataFrame({
        "A": [np.nan, 2, np.nan, 4],
        "B": [np.nan, 2, np.nan, 4],
        "C": [np.nan, 2, 3, 4],
    })


@pytest.mark.unit
def test_card_is_mutable_and_uses_checkbox_group(card_module):
    card = card_module.instance()
    assert card.mutable is True
    assert card.long_name == "Excessive missingness"
    assert 'id="Map"' in str(card.front)
    assert 'id="Table"' in str(card.back)
    assert 'id="Remove"' in str(card.footer)
    assert "shiny-input-checkboxgroup" in str(card.footer)
    assert 'value="Variables"' in str(card.footer)
    assert 'value="Observations"' in str(card.footer)


@pytest.mark.unit
def test_excessive_observations_use_strict_threshold(card_module):
    result = card_module._excessive_observation_mask(frame(), 0.5)
    assert result.tolist() == [True, False, True, False]


@pytest.mark.unit
def test_removal_preserves_proxy_metadata(card_module):
    from proxy_data import proxy_data
    source = proxy_data(_df=frame(), _name="sample")
    result = card_module._remove_excessive_observations(source, 0.5)
    assert result.name == "sample"
    assert list(result.frame.index) == [1, 3]
    assert result.role_map == source.role_map
    assert len(source.frame) == 4
    assert len(result.cleaning_records) == 1
    record = result.cleaning_records[0]
    assert record.card == "miss_map"
    assert record.parameters == {"threshold": 0.5}
    assert record.input_shape == (4, 3)
    assert record.output_shape == (2, 3)


@pytest.mark.unit
def test_variable_removal_preserves_remaining_roles(card_module):
    from proxy_data import proxy_data
    from roles import Role
    source = proxy_data(_df=frame(), _name="sample")

    result = card_module._remove_excessive_variables(source, 0.4)

    assert list(result.frame.columns) == ["C"]
    assert result.name == "sample"
    assert result.role_map.roles_for("C") == {Role.PREDICTOR}
    assert not result.role_map.roles_for("A")
    assert result.cleaning_records[-1].operation == (
        "Remove excessively incomplete variables"
    )


@pytest.mark.unit
def test_combined_removal_processes_variables_before_observations(card_module):
    from proxy_data import proxy_data
    source = proxy_data(_df=pd.DataFrame({
        # This column is removed at 50%; without it, no row exceeds 50% missingness.
        "mostly_missing": [np.nan, np.nan, np.nan, 4],
        "kept": [np.nan, 2, 3, 4],
        "complete": [1, 2, 3, 4],
    }))

    result = card_module._transform_data(
        source,
        remove_variables=True,
        remove_observations=True,
        variable_threshold=0.5,
        observation_threshold=0.5,
    )

    assert list(result.frame.columns) == ["kept", "complete"]
    assert len(result.frame) == 4
    assert len(result.cleaning_records) == 1
    assert len(result.processing_records) == 1
    record = result.cleaning_records[0]
    assert record.operation == "Remove excessive missingness"
    assert dict(record.parameters) == {
        "remove_variables": True,
        "variable_threshold": 0.5,
        "remove_observations": True,
        "observation_threshold": 0.5,
        "execution_order": "variables before observations",
    }


@pytest.mark.unit
def test_disabled_removals_produce_one_inactive_card_step(card_module):
    from proxy_data import proxy_data

    source = proxy_data(_df=frame())
    result = card_module._transform_data(
        source,
        remove_variables=False,
        remove_observations=False,
        variable_threshold=0.4,
        observation_threshold=0.5,
    )

    assert result.cleaning_records == ()
    assert len(result.processing_records) == 1
    record = result.processing_records[0]
    assert record.card == "miss_map"
    assert record.operation == "Remove excessive missingness"
    assert record.attempted is False


@pytest.mark.unit
def test_issues_include_variable_and_observation_problems(card_module):
    from proxy_data import proxy_data
    source = proxy_data(_df=frame())
    result = card_module._issues_table(source.frame, source.role_map, 0.4, 0.5)
    assert "Excessive variable missingness" in set(result["Issue"])
    assert "Excessive observation missingness" in set(result["Issue"])


@pytest.mark.unit
def test_full_screen_margins_align_with_heatmap(card_module):
    figure = card_module._missingness_figure(
        frame(), hide_complete=False, sort_variables=True,
        variable_threshold=0.5, observation_threshold=0.5,
        show_thresholds=True, full_screen=True,
    )
    assert [trace.type for trace in figure.data] == ["heatmap", "bar", "bar"]
    assert tuple(figure.layout.yaxis.range) == tuple(figure.layout.yaxis2.range)
    assert tuple(figure.layout.xaxis.range) == tuple(figure.layout.xaxis3.range)
    assert len(figure.layout.shapes) == 2


@pytest.mark.unit
def test_observation_bars_use_all_variables_when_complete_ones_are_hidden(
    card_module,
):
    data = pd.DataFrame({
        "incomplete": [np.nan, 2],
        "complete_1": [1, 2],
        "complete_2": [1, 2],
    })

    figure = card_module._missingness_figure(
        data,
        hide_complete=True,
        sort_variables=True,
        variable_threshold=0.5,
        observation_threshold=0.5,
        show_thresholds=True,
        full_screen=True,
    )

    # The heatmap displays one variable, but the observation denominator is
    # all three variables: 1/3 missing, not 1/1 missing.
    assert list(figure.data[2].y) == pytest.approx([1 / 3, 0])


def get_card(page: Page):
    return page.locator(".card").first


def by_id(page: Page, local_id: str):
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    namespace = card_id.partition("-")[0]
    return page.locator(f"#{namespace}-{local_id}")


class TestWebKitUI:
    @pytest.mark.ui
    def test_map_and_control_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Map").locator(".plotly")).to_be_attached(timeout=15_000)
        expect(by_id(page, "Remove")).to_be_attached()
        expect(by_id(page, "Check")).to_contain_text("observations exceed", timeout=15_000)

    @pytest.mark.ui
    def test_removal_updates_map_status(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("observations exceed", timeout=15_000)
        by_id(page, "Remove").locator(
            'input[type=checkbox][value="Observations"]'
        ).check(force=True)
        expect(by_id(page, "Check")).to_contain_text("removed 1", timeout=15_000)
