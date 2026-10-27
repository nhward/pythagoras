"""Large membership rendering fixture; bypass fitting to isolate browser work."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import obs_clusters as m
from proxy_data import proxy_data

frame = pd.DataFrame(np.random.default_rng(123).normal(size=(2000, 50)),
                     columns=[f'x{i}' for i in range(50)])

def analysis(source, **options):
    return m.ClusterViews(source=source, options=options,
        positions=np.arange(len(source)), coordinates=source.frame.iloc[:, :2].to_numpy(),
        labels={method: np.repeat('c1', len(source)) for method in m.METHODS},
        predictors=list(source.columns))

m._analyze = analysis
this = m.instance()
this.restore_configuration_state({'inputs': {'Projection': 'pca'}})
this._imports.set(proxy_data(_df=frame, _cluster_count=1, _name='Large membership'))
app = this.application()
