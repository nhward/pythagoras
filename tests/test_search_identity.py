"""Search identity must survive reconstruction under another Python hash seed."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROBE = """
import json
import numpy as np
import pandas as pd
from proxy_data import proxy_data
from roles import RoleMap
from FeatureSelectionTransformer import FeatureSelectionTransformer
from model_commit import prepare_predictors
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from pipeline_search import prepare
columns = ['zeta','alpha','gamma','beta','delta','epsilon','theta','kappa']
x = np.arange(30, dtype=float)
frame = pd.DataFrame({**{c: x + i for i,c in enumerate(columns)}, 'y': x * 2})
source = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor':columns,'target':['y']}))
source = source.with_pipeline_step(FeatureSelectionTransformer(tuple(columns), predictors=tuple(source.model_predictors)),
                                  name='var_select', preview_frame=source.frame)
model = Pipeline([('predictors', FunctionTransformer(prepare_predictors, kw_args={'columns': source.model_predictors, 'preparation':'numeric'})), ('estimator',Ridge())])
source = source.with_estimator(model, label='Ridge', recipe_id='ridge')
problem = prepare(source)
print(json.dumps({'predictors':source.model_predictors, 'fingerprint':problem.specification['fingerprint']}))
"""


def test_selection_and_model_search_identity_are_independent_of_python_hash_seed():
    root = Path(__file__).resolve().parents[1]
    results = []
    for seed in ('11', '31'):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(root / 'app'))
        process = subprocess.run([sys.executable, '-c', PROBE], cwd=root, env=environment,
                                 check=True, capture_output=True, text=True, timeout=30)
        results.append(json.loads(process.stdout))
    assert results[0] == results[1]
    assert results[0]['predictors'] == ['zeta', 'alpha', 'gamma', 'beta', 'delta', 'epsilon', 'theta', 'kappa']
