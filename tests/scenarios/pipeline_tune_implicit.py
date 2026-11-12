"""Bayesian skopt search without a committed upstream evaluation plan."""
import runpy
from dataclasses import replace
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('pipeline_tune.py')))
this = scenario['this']
this._imports.set(replace(scenario['source'], _evaluation_plan=None))
this.restore_configuration_state({'inputs': {'Method': 'skopt'}})
app = this.application()
