"""Seeded Shiny scenario for system-log browser tests."""

import logging
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.system_log import instance
from module import ApplicationLogHandler, Module

# Isolate this scenario from records produced while pytest imports other cards.
old_handler = Module.log_handler
handler = ApplicationLogHandler()
Module.log_handler = handler
if old_handler in Module.log.handlers:
    Module.log.removeHandler(old_handler)
Module.log.addHandler(handler)

this = instance()
app = this.application()


def add_record(name: str, level: int, message: str, source: str, line: int) -> None:
    record = logging.LogRecord(name, level, source, line, message, (), None)
    handler.handle(record)


add_record(
    "import",
    logging.INFO,
    "Scenario passenger data loaded",
    "/app/cards/data_import.py",
    10,
)
add_record(
    "model",
    logging.ERROR,
    "Scenario could not fit tree",
    "/app/cards/miss_type.py",
    20,
)
add_record(
    "table",
    logging.DEBUG,
    "Scenario table rendered",
    "/app/cards/data_tabulation.py",
    30,
)
