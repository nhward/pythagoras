"""Finite search whose candidate fits fail parameter validation."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train.py')))
this = scenario['this']
source = scenario['source']
spec = source.tuning_spec
spec['grid'] = {'cand_methods__estimator__alpha': [-float(i) for i in range(1, 15)]}
spec['combinations'] = 14
this._imports.set(source.with_tuning_spec(spec))
app = this.application()
