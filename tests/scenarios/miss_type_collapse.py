"""Exercise late navbar collapse with a configurable one-tab limit."""

import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name("miss_type.py")))
scenario["this"].config.setdefault("settings", {})["max_collapse"] = 1
app = scenario["app"]
