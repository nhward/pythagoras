"""Review-only evidence must not become a selected downstream pipeline."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_models.py')))
this, source = scenario['this'], scenario['source']
this._imports.set(source.with_tuning_results({**source.tuning_results, 'read_only': True}))
app = this.application()
