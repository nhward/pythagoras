"""A saved shadow selection that is unavailable in the current source."""
from pathlib import Path
import runpy

scenario = runpy.run_path(str(Path(__file__).with_name("miss_informative.py")))
scenario["this"].restore_configuration_state({"inputs": {"Shadow": ["unavailable"]}})
app = scenario["this"].application()
