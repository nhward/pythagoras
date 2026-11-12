"""Method-specific settings in a bundled learned imputer."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('tuning_parameters.py')))
from cards.miss_impute import ImputationStep

this = scenario['this']
source = scenario['source']
source = source.with_pipeline_step(ImputationStep(('x',), ('x',)), name='miss_impute', preview_frame=source.frame)
this._imports.set(source)
app = this.application()
