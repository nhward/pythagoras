"""Live tuning replay with the production footer, without diagnostic probes."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train_refresh.py')))
this = scenario['tune']
this.footer = scenario['scenario']['original_footer']
app = this.application()
