"""Review and commit nominal mappings after an existing pipeline step."""
import runpy
from pathlib import Path

context = runpy.run_path(str(Path(__file__).with_name('var_nominal.py')))
this, app = context['this'], context['app']
from NominalReconciliationTransformer import NominalReconciliationTransformer
from shiny import reactive

with reactive.isolate():
    source = this._imports.get()
transformer = NominalReconciliationTransformer([])
this._imports.set(source.with_pipeline_step(transformer, name='upstream',
    preview_frame=transformer.fit_transform(source.frame)))
