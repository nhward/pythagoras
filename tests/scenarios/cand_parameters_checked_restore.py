"""Restore checked parameter drafts without a saved presentation order."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_parameters.py')))
from tuning import discover_source

this, source = scenario['this'], scenario['source']
rows = discover_source(source)
chosen = [row for row in rows if row['path'] == 'miss_map__observation_threshold'
          or row['path'].endswith('__max_depth')]
assert len(chosen) == 2
this.restore_configuration_state({'tuning': {'drafts': {
    row['path']: dict(row['definition'], owner=row['owner'], active=True) for row in chosen}}})
app = this.application()
