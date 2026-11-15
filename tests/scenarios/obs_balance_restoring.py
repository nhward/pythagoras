"""Restore the target-balancing action before dynamic footer inputs exist."""
import runpy
from pathlib import Path

app = runpy.run_path(str(Path(__file__).with_name('obs_balance.py')),
                    init_globals={'RESTORED': True})['app']
