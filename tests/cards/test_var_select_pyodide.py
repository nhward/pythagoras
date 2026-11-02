"""Real WebAssembly selection/pipeline regression using cached Shinylive assets.

Setup (no network is used during the test):
  .venv/bin/python -m pip download --no-deps -d /tmp/var-select-pyodide-wheels \
      Boruta==0.4.3 feature-engine==1.9.4 imbalanced-learn==0.14.2
Override that wheel directory with PYTHAGORAS_WASM_WHEELS if desired.
"""
import functools
import json
import os
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selection_wasm_site(tmp_path):
    from shinylive._assets import pyodide_lock_json_file
    assets = pyodide_lock_json_file().parent
    wheels = Path(os.environ.get('PYTHAGORAS_WASM_WHEELS', '/tmp/var-select-pyodide-wheels'))
    required = ['Boruta-0.4.3-py3-none-any.whl', 'feature_engine-1.9.4-py3-none-any.whl',
                'imbalanced_learn-0.14.2-py3-none-any.whl']
    if not (assets / 'pyodide.mjs').exists() or not all((wheels / n).exists() for n in required):
        pytest.skip('Requires cached Shinylive assets and the pure-Python wheels listed in this test')
    (tmp_path / 'pyodide').symlink_to(assets, target_is_directory=True)
    (tmp_path / 'wheels').symlink_to(wheels, target_is_directory=True)
    (tmp_path / 'source.json').write_text(json.dumps({p.name:p.read_text() for p in (ROOT / 'app').glob('*.py')}))
    (tmp_path / 'requirements.txt').write_text((ROOT / 'app/requirements.txt').read_text())
    (tmp_path / 'index.html').write_text('<!doctype html><html><body>Pyodide selection regression</body></html>')
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', required
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


@pytest.mark.ui
def test_selection_and_exported_pipeline_in_pyodide(page, selection_wasm_site):
    url, wheels = selection_wasm_site
    page.goto(url)
    page.set_default_timeout(180000)
    result = page.evaluate('''async ({url, wheels}) => {
        const {loadPyodide} = await import(url + '/pyodide/pyodide.mjs');
        const py = await loadPyodide({indexURL: url + '/pyodide/'});
        await py.loadPackage(['ssl', 'lightgbm', 'pandas', 'micropip', 'geopandas', 'statsmodels', 'shiny']);
        py.globals.set('wheel_urls', py.toPy(wheels.map(n => url + '/wheels/' + n)));
        await py.runPythonAsync('import micropip\\nawait micropip.install(list(wheel_urls))');
        const sources = await (await fetch(url + '/source.json')).json();
        py.FS.mkdirTree('/app');
        for (const [name, text] of Object.entries(sources)) py.FS.writeFile('/app/' + name, text);
        py.FS.writeFile('/app/requirements.txt', await (await fetch(url + '/requirements.txt')).text());
        return await py.runPythonAsync(`
import json, sys, importlib.metadata
sys.path.insert(0, '/app')
import numpy as np
import pandas as pd
from packaging.requirements import Requirement
from FeatureSelectionTransformer import FeatureSelectionTransformer
from proxy_data import proxy_data
from roles import RoleMap
from selection_analysis import analyze_selection, export_selection

assert sys.platform == 'emscripten'
# Resolve the actual dependency markers in the browser environment.
requirements = [Requirement(line) for line in open('/app/requirements.txt')
                if line.strip() and not line.startswith('#')]
for package in ('lightgbm', 'boruta'):
    applicable = [r for r in requirements if r.name.lower() == package
                  and (r.marker is None or r.marker.evaluate())]
    assert len(applicable) == 1
    assert applicable[0].specifier.contains(importlib.metadata.version(package))

rng = np.random.default_rng(83)
n = 240
signal = rng.normal(size=n)
frame = pd.DataFrame({'signal':signal, 'noise':rng.normal(size=n),
    'nominal':pd.Categorical(rng.choice(['a','b',None], n)),
    'flag':pd.array(rng.choice([True,False,None], n),dtype='boolean'),
    'empty':pd.array([None]*n,dtype='Float64'),
    'treatment':pd.Categorical(rng.choice(['treated','control',None], n)),
    'id':np.arange(n)})
frame.loc[frame.index[::17], 'signal'] = np.nan
roles=RoleMap.from_primitive({'predictor':['signal','noise','nominal','flag','empty'],
    'treatment':['treatment'], 'identifier':['id'], 'target':['outcome']})
options=dict(trees=40,depth=4,iterations=18,alpha=.05,percentile=100,
    two_step=True,folds=3,repeats=1,cutoff=1.,policy='confirmed',
    random_state=2025,n_jobs=4,use_weights=True)
reports=[]
for task in ('regression','classification'):
    data=frame.copy()
    values=np.nan_to_num(data.signal.to_numpy())
    data['outcome']=(values*5+rng.normal(size=n)*.1 if task=='regression'
                     else pd.Categorical(np.where(values>.5,'up',np.where(values<-.5,'down','middle'))))
    # Duplicate indices must survive selection exactly.
    data.index=np.zeros(n,dtype=int)
    source=proxy_data(data,_roles=roles)
    for method in ('Boruta','GrootCV'):
        result=analyze_selection(source,method,options)
        assert not result['error'], result['error']
        model=result['model']
        assert model.effective_n_jobs_ == 1
        assert 'signal' in model.selected_columns_
        assert 'empty' not in model.selected_columns_
        assert 'treatment' not in model.candidate_columns_
        pd.testing.assert_series_equal(export_selection(result).frame.treatment, data.treatment)
        exported=export_selection(result)
        expected=data.drop(columns=model.removed_columns_)
        pd.testing.assert_frame_equal(exported.frame,expected)
        pd.testing.assert_frame_equal(exported.clean_frame,data)
        assert not hasattr(exported.pipeline.named_steps.var_select,'selected_columns_')
        pipeline=exported.pipeline_for_training()
        pipeline.fit(data.iloc[:180],data.outcome.iloc[:180].to_numpy())
        assert pipeline.named_steps.var_select.effective_n_jobs_ == 1
        observed=pipeline.transform(data.iloc[180:])
        assert len(observed)==60 and observed.index.equals(data.index[180:])
        assert all(c in observed for c in ('id','outcome'))
        pd.testing.assert_frame_equal(observed,data.iloc[180:][observed.columns])
        reports.append({'task':task,'method':method,'selected':model.selected_columns_,
                        'workers':model.effective_n_jobs_})
# Exercise inherited structured designs and the diagnostic in WebAssembly too.
from evaluation_plan import build_plan
from selection_analysis import selection_stability
structured=[]
for strategy in ('grouped','temporal'):
    data=frame.copy()
    data['outcome']=np.nan_to_num(data.signal.to_numpy())*5+rng.normal(size=n)*.01
    data['group']=pd.Categorical(np.repeat(np.arange(3),80))
    data['time']=np.repeat(np.arange(24),10)
    local_roles=RoleMap.from_primitive(roles.to_primitive())
    if strategy=='grouped':
        local_roles.set_roles('group',['stratifier'])
        settings=dict(strategy='grouped',objective='unseen_strata',group='group',folds=3)
    else:
        local_roles.set_roles('time',['sequence'])
        settings=dict(strategy='temporal',objective='future',sequence='time',window=8,horizon=3,gap=1,step=3)
    source=proxy_data(data,_roles=local_roles)
    plan=build_plan(source,**settings)
    assert plan.ready,plan.errors
    source=source.with_evaluation_plan(plan,card='obs_resample')
    result=analyze_selection(source,'GrootCV',options)
    assert not result['error'],result['error']
    model=result['model']
    assert model.inner_resampling_['strategy']==strategy
    assert model.effective_n_jobs_==1
    for a,b in model.inner_splits_:
        if strategy=='grouped':
            assert not set(data.group.iloc[a]) & set(data.group.iloc[b])
        else:
            assert data.time.iloc[a].max()+1<data.time.iloc[b].min()
    if strategy=='grouped':
        diagnostic=selection_stability(source,'GrootCV',options)
        assert not diagnostic['error'],diagnostic['error']
        assert diagnostic['successful']==3
    structured.append(strategy)
json.dumps({'structured':structured,'platform':sys.platform,'versions':{p:importlib.metadata.version(p)
            for p in ('lightgbm','Boruta','scikit-learn')},'cases':reports})
`);
    }''', {'url':url,'wheels':wheels})
    report=json.loads(result)
    assert report['platform']=='emscripten'
    assert report['versions']['lightgbm']=='4.5.0'
    assert len(report['cases'])==4
    assert report['structured']==['grouped','temporal']
    print(json.dumps(report,indent=2))
