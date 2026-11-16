"""Restore the intrinsic transparency constraint from bookmark inputs."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_models.py')))
this = scenario['this']
this.restore_configuration_state({'inputs': {
    'UsuallyTransparentOnly': True, 'Weights': [0, 100],
    'ErrorFloor': 1e-12, 'EquivalenceTolerance': None, 'DecisionProbability': .95}})
app = this.application()
