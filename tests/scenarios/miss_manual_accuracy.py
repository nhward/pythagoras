"""One 80%-missing leaf, one 40%-missing leaf, and a weak variable."""
import os
import sys
from pathlib import Path
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import miss_manual as m
from proxy_data import proxy_data

frame = pd.DataFrame({
    'value': pd.Series([None] * 8 + [1] * 2 + [None] * 4 + [1] * 6, dtype='Int64'),
    'group': pd.Categorical(['A'] * 10 + ['B'] * 10),
    'weak': pd.Series([None if i in [0, 1, 8, 10, 14, 15] else 1 for i in range(20)], dtype='Int64'),
})
this = m.instance()
this.restore_configuration_state({'inputs': {'MaxTreeDepth': 1, 'MinLeafAccuracy': 0.8, 'ExtraLevel': ''}})
this._imports.set(proxy_data(_df=frame, _name='Leaf accuracy example'))
app = this.application()
