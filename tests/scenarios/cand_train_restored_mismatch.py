"""A restored opaque fingerprint mismatch must preserve all trial history."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train_restored.py')))
this, restoration = scenario['this'], scenario['restoration']
restoration['search']['checkpoint']['specification']['fingerprint'] = 'saved-opaque-fingerprint'
restoration['search']['checkpoint']['specification'].pop('training_compatibility', None)
this.restore_configuration_state(restoration)
app = this.application()
