"""Only conditional tree trials are complete under a strict transparency requirement."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_models_transparent.py')))
this = scenario['this']
source = scenario['scenario']['source']
results = source.tuning_results
results['state']['trials'] = [trial for trial in results['state']['trials'] if trial['model_id'] == 'tree']
this._imports.set(source.with_tuning_results(results))
app = this.application()
