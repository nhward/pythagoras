from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

path = Path(__file__).resolve().parents[2]/ "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/system_log.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def system_log():
    return importlib.import_module("cards.system_log")


@pytest.fixture
def card(system_log):
    return system_log.instance()


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


class TestInstance:
    @pytest.mark.unit
    def test_metadata_and_regions(self, card):
        assert card.name == "system_log"
        assert card.long_name == "System log"
        assert not card.mutable
        assert card.hasSidebar()
        assert card.hasFooter()
        assert not card.hasFlipSide()

    @pytest.mark.unit
    def test_ui_contains_table_filters_and_refresh(self, card):
        front = str(card.front)
        settings = str(card.settings)
        footer = str(card.footer)
        assert 'id="LogTable"' in front
        for control in ("Levels", "Search", "Maximum", "AutoRefresh"):
            assert f'id="{control}"' in settings
        assert 'id="Refresh"' in footer
        assert 'id="Status"' in footer

    @pytest.mark.unit
    def test_complete_card_ui_contains_table_output(self, card):
        card_ui = str(card.call_ui())
        assert "<shiny-data-frame" in card_ui
        assert '-LogTable"' in card_ui


class TestLogFrame:
    @pytest.mark.unit
    def test_message_column_is_allocated_extra_width(self, system_log):
        style = system_log.MESSAGE_COLUMN_STYLE
        assert style["cols"] == "Message"
        assert style["style"]["width"] == "50%"
        assert style["style"]["min-width"] == "30rem"

    @pytest.fixture
    def records(self):
        return [
            {
                "Time": datetime(2026, 1, 1, 10, 0, 0),  # noqa: DTZ001
                "Level": "INFO",
                "Logger": "import",
                "Message": "Loaded passenger data",
                "Source": "/app/cards/data_import.py",
                "Line": 10,
                "Thread": "MainThread",
            },
            {
                "Time": datetime(2026, 1, 1, 10, 0, 1),  # noqa: DTZ001
                "Level": "ERROR",
                "Logger": "model",
                "Message": "Could not fit tree",
                "Source": "/app/cards/miss_type.py",
                "Line": 20,
                "Thread": "worker-1",
            },
            {
                "Time": datetime(2026, 1, 1, 10, 0, 2),  # noqa: DTZ001
                "Level": "DEBUG",
                "Logger": "table",
                "Message": "Rendered rows",
                "Source": "/app/cards/data_tabulation.py",
                "Line": 30,
                "Thread": "MainThread",
            },
        ]

    @pytest.mark.unit
    def test_filters_levels_searches_and_orders_newest_first(
        self, system_log, records
    ):
        result = system_log._log_frame(
            records,
            levels=["INFO", "ERROR"],
            query="tree",
            maximum=100,
        )
        assert result["Logger"].tolist() == ["model"]
        assert result["Level"].tolist() == ["ERROR"]

    @pytest.mark.unit
    def test_limits_after_filtering_and_returns_expected_schema(
        self, system_log, records
    ):
        result = system_log._log_frame(
            records,
            levels=system_log.LOG_LEVELS,
            maximum=2,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == system_log.LOG_COLUMNS
        assert result["Logger"].tolist() == ["table", "model"]
        assert result["Source"].tolist() == [
            "data_tabulation.py",
            "miss_type.py",
        ]
        assert result["Time"].str.match(r"\d{4}-\d{2}-\d{2} ").all()

    @pytest.mark.unit
    def test_no_selected_levels_returns_empty_frame(self, system_log, records):
        result = system_log._log_frame(records, levels=[], maximum=100)
        assert result.empty
        assert result.columns.tolist() == system_log.LOG_COLUMNS


class TestApplicationLogHandler:
    @pytest.mark.unit
    def test_handler_is_bounded_and_returns_copies(self):
        from module import ApplicationLogHandler

        handler = ApplicationLogHandler(capacity=2)
        logger = logging.getLogger("system-log-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.info("first")
        logger.warning("second")
        logger.error("third")

        records = handler.snapshot()
        assert [record["Message"] for record in records] == ["second", "third"]
        records[0]["Message"] = "changed"
        assert handler.snapshot()[0]["Message"] == "second"


class TestWebKitUI:
    @pytest.mark.ui
    def test_log_grid_finishes_rendering_with_seeded_records(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("System log", exact=True)).to_be_visible()
        grid = controller.OutputDataFrame(page, namespaced_id(page, "LogTable"))
        grid.expect_nrow(3)
        grid.expect_column_labels([
            "Time", "Level", "Logger", "Message", "Source", "Line", "Thread"
        ])
        expect(by_id(page, "Status")).to_have_text("Showing 3 records")
        expect(by_id(page, "LogTable")).not_to_have_class("recalculating")

    @pytest.mark.ui
    def test_grid_shows_compact_sources_and_wide_message_column(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "LogTable"))
        grid.expect_cell("Scenario table rendered", row=0, col=3)
        grid.expect_cell("data_tabulation.py", row=0, col=4)
        message_cell = by_id(page, "LogTable").locator("tbody td:nth-child(4)").first
        expect(message_cell).to_have_css("min-width", "480px")

    @pytest.mark.ui
    def test_search_and_level_filters_update_grid(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "Search", "tree")
        expect(by_id(page, "Status")).to_have_text("Showing 1 record")
        grid = controller.OutputDataFrame(page, namespaced_id(page, "LogTable"))
        grid.expect_nrow(1)
        grid.expect_cell("ERROR", row=0, col=1)
        grid.expect_cell("Scenario could not fit tree", row=0, col=3)
        grid.expect_cell("miss_type.py", row=0, col=4)

        set_shiny_input(page, "Search", "")
        set_shiny_input(page, "Levels", ["INFO"])
        expect(by_id(page, "Status")).to_have_text("Showing 1 record")
        grid.expect_nrow(1)
        grid.expect_cell("Scenario passenger data loaded", row=0, col=3)

    @pytest.mark.ui
    def test_limit_refresh_and_auto_refresh_controls_remain_usable(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "Maximum", 2)
        expect(by_id(page, "Status")).to_have_text("Showing 2 records")
        grid = controller.OutputDataFrame(page, namespaced_id(page, "LogTable"))
        grid.expect_nrow(2)

        by_id(page, "Refresh").click()
        expect(by_id(page, "Status")).to_have_text("Showing 2 records")
        page.get_by_role("button", name="Toggle sidebar").click()
        expect(by_id(page, "AutoRefresh")).to_be_visible()
        by_id(page, "AutoRefresh").check()
        expect(by_id(page, "AutoRefresh")).to_be_checked()
        grid.expect_nrow(2)
