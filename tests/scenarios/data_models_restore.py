"""Restore an actual downloaded card configuration in a fresh app."""
import json
import os
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('data_models.py')))
this = scenario['this']
with open(os.environ['PYTHAGORAS_TEST_MODEL_CONFIG']) as stream:
    this.restore_configuration_state(json.load(stream))
app = this.application()
