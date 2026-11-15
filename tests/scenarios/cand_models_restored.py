"""Restore a nondefault allocation and ignore a retired method filter."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_models.py')))
this = scenario['this']
this.restore_configuration_state({'inputs': {'Weights': [0, 100], 'EligibleMethods': ['ridge'],
    'ErrorFloor': 1e-9, 'EquivalenceTolerance': None, 'DecisionProbability': .95},
    'selection': {'version': 1, 'overall': 999, 'methods': {'tree': 999}}})
app = this.application()
