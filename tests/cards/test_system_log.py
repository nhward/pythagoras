from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parent.parent.parent / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))


@pytest.fixture
def system_log():
    module = importlib.import_module("cards.system_log")
    module.this = module.instance()
    return module


class TestInstance:
    @pytest.mark.unit
    def test_metadata_and_regions(self, system_log):
        card = system_log.this
        assert card.name == "system_log"
        assert card.long_name == "System log"
        assert not card.mutable
        assert card.hasSidebar()
        assert card.hasFooter()
        assert not card.hasFlipSide()

    @pytest.mark.unit
    def test_ui_contains_table_filters_and_refresh(self, system_log):
        front = str(system_log.this.front)
        settings = str(system_log.this.settings)
        footer = str(system_log.this.footer)
        assert 'id="LogTable"' in front
        for control in ("Levels", "Search", "Maximum", "AutoRefresh"):
            assert f'id="{control}"' in settings
        assert 'id="Refresh"' in footer
        assert 'id="Status"' in footer

    @pytest.mark.unit
    def test_complete_card_ui_contains_table_output(self, system_log):
        card_ui = str(system_log.this.call_ui())
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
                "Time": datetime(2026, 1, 1, 10, 0, 0),
                "Level": "INFO",
                "Logger": "import",
                "Message": "Loaded passenger data",
                "Source": "/app/cards/data_import.py",
                "Line": 10,
                "Thread": "MainThread",
            },
            {
                "Time": datetime(2026, 1, 1, 10, 0, 1),
                "Level": "ERROR",
                "Logger": "model",
                "Message": "Could not fit tree",
                "Source": "/app/cards/miss_type.py",
                "Line": 20,
                "Thread": "worker-1",
            },
            {
                "Time": datetime(2026, 1, 1, 10, 0, 2),
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
