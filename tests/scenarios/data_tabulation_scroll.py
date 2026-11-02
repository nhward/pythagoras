"""A long metadata listing that overflows in both directions."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import pandas as pd
from cards.data_tabulation import instance
from proxy_data import proxy_data

this = instance()
this._imports.set(proxy_data(pd.DataFrame({f'variable_{i:03}':range(12) for i in range(160)})))
app = this.application()
