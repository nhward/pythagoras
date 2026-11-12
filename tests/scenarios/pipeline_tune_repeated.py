"""Repeated-CV tuning with display-only Bayesian comparison settings."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('pipeline_tune.py')))
from evaluation_plan import build_plan

this = scenario['this']
source = scenario['source']
source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent',
                                               folds=3, repeats=2), card='obs_resample')
this._imports.set(source)
this.restore_configuration_state({'inputs': {'Method': 'grid', 'Execution': 'sequential',
                                             'EquivalenceTolerance': 0.1}})
app = this.application()
