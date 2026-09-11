"""Full application fixture forced into remote-server runtime mode."""

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from module import Module

Module.runtime_mode = classmethod(lambda cls, session: "server")
