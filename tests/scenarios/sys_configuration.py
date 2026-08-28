"""Test-owned Shiny application for configuration-card browser tests."""

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.sys_configuration import instance

this = instance()
app = this.application()
