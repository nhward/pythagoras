"""Real spawned fits: resource bounds, concurrency and restartable batches."""
import copy
import json
import os
import threading
import time
from pathlib import Path

import joblib
import pytest
from pipeline_search import SearchWorker, fresh, prepare, validate_checkpoint
from sklearn.linear_model import Ridge
from test_pipeline_search import finish, gp_problem, source

pytestmark = pytest.mark.unit


class AuditRidge(Ridge):
    def __init__(self, alpha=1., log_path='', delay=.25):
        super().__init__(alpha=alpha)
        self.log_path = log_path
        self.delay = delay

    def fit(self, X, y, **kwargs):
        rows = list(map(int, y.index))
        def log(event):
            if self.log_path:
                with open(self.log_path, 'a') as stream:
                    stream.write(json.dumps({'event': event,'pid': os.getpid(),'alpha': self.alpha,
                        'rows': rows,'time': time.monotonic()})+'\n')
        log('start')
        if not self.log_path or not Path(self.log_path+'.fast').exists():
            # The first fold's training rows start later, deliberately finishing out of order.
            time.sleep(self.delay * (3 if rows[0] else 1))
        result = super().fit(X,y,**kwargs)
        log('end')
        return result


def parallel_problem(mode, *, log_path='', delay=.25, method='grid'):
    data = source()
    model = data.pipeline.named_steps.data_models
    model.set_params(estimator=AuditRidge(log_path=log_path,delay=delay))
    data = data.with_estimator(model,label='Audited Ridge',recipe_id='parallel-test')
    return prepare(data, model_values={'data_models__estimator__alpha':[.1,10.]},
                   method=method,execution=mode,workers=2)


def normalized(state):
    return [(t['params'],t['fold_ids'],t['folds'],t['state']) for t in state['trials']]


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
@pytest.mark.parametrize('mode',['folds','trials','auto'])
def test_actual_parallel_fits_match_serial_scores_and_preserve_holdout(mode,tmp_path):
    path = str(tmp_path/'fits.jsonl')
    p = parallel_problem(mode,log_path=path)
    assert p.specification['execution']['workers']==2
    state = finish(p)
    assert state['status']=='exhausted'
    serial = finish(parallel_problem('sequential',delay=0))
    assert normalized(state)==normalized(serial)
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    starts=[r for r in records if r['event']=='start']
    assert len({r['pid'] for r in starts})==2
    assert all(not set(r['rows']) & set(range(20,24)) for r in starts)
    intervals=[]
    for first in starts:
        last=next(r for r in records if r['event']=='end' and
                  (r['pid'],r['alpha'],r['rows'])==(first['pid'],first['alpha'],first['rows']) and r['time']>=first['time'])
        intervals.append((first,last))
    overlaps=[(a,c) for a,b in intervals for c,d in intervals
              if a['pid']!=c['pid'] and max(a['time'],c['time'])<min(b['time'],d['time'])]
    assert overlaps
    if mode=='trials':
        assert any(a['alpha']!=b['alpha'] for a,b in overlaps)
    else:
        assert any(a['alpha']==b['alpha'] for a,b in overlaps)  # Includes parallel baseline folds.


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
def test_out_of_order_fold_pause_and_checkpoint_validation(tmp_path):
    p=parallel_problem('folds',log_path=str(tmp_path/'fits.jsonl'),delay=.4)
    stop=threading.Event()
    published=[]
    def publish(state):
        if state['trials'] and state['trials'][0]['fold_ids']:
            published.append(state)
            stop.set()
    paused=finish(p,publish=publish,pause=stop.is_set)
    assert paused['status']=='paused'
    assert published[0]['trials'][0]['fold_ids']==[1]
    assert paused['trials'][0]['fold_ids']==[0,1]
    checkpoint=json.loads(json.dumps(paused))
    validate_checkpoint(p,checkpoint)
    bad=copy.deepcopy(checkpoint)
    bad['trials'][0]['fold_ids']=[0,0]
    with pytest.raises(ValueError,match='indexed fold'):validate_checkpoint(p,bad)
    resumed=finish(p,checkpoint)
    assert resumed['trials'][0]['fold_ids']==[0,1,2]


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
@pytest.mark.parametrize('method',['random','grid','tpe','skopt'])
def test_parallel_batch_resume_matches_uninterrupted_search(method):
    data=source()
    p=prepare(data,model_values={'data_models__estimator__alpha':[.01,.1,1.,10.]},
              method=method,execution='trials',workers=2)
    stop=threading.Event()
    def publish(state):
        if any(t['folds'] for t in state['trials'][1:]):stop.set()
    paused=finish(p,publish=publish,pause=stop.is_set)
    assert paused['status']=='paused'
    assert paused['batch']==[1,2]
    assert sum(t['state']=='running' for t in paused['trials'])==2
    resumed=finish(p,json.loads(json.dumps(paused)))
    complete=finish(p)
    assert normalized(resumed)==normalized(complete)
    assert len({joblib.hash(t['params']) for t in complete['trials']})==4


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
def test_cancel_cleans_fit_processes_and_retries_missing_work(tmp_path):
    import psutil
    path=str(tmp_path/'cancel.jsonl')
    p=parallel_problem('folds',log_path=path,delay=10)
    worker=SearchWorker();checkpoint=fresh(p)
    try:
        worker.start(p,checkpoint,trials=1,seconds=0)
        deadline=time.monotonic()+25
        records=[]
        while time.monotonic()<deadline:
            checkpoint=worker.poll() or checkpoint
            if Path(path).exists():records=[json.loads(line) for line in Path(path).read_text().splitlines()]
            if len({r['pid'] for r in records})==2:break
            time.sleep(.03)
        pids={r['pid'] for r in records}
        assert len(pids)==2
        tick=time.monotonic();worker.close()
        assert time.monotonic()-tick<4
        assert not worker.running
        assert all(not psutil.pid_exists(pid) for pid in pids)
        Path(path+'.fast').touch()
        resumed=finish(p,checkpoint,trials=1)
        assert all(t['state']=='complete' for t in resumed['trials'])
    finally:
        worker.close()


def test_resource_bounds_and_operational_identity(monkeypatch):
    from types import SimpleNamespace

    import psutil
    monkeypatch.setattr(joblib,'cpu_count',lambda:8)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=1024**3))
    p=prepare(source(),execution='auto',workers=64)
    assert p.specification['execution']['workers']==1
    p2=prepare(source(),execution='sequential',workers=1)
    assert p.specification['fingerprint']!=p2.specification['fingerprint']
    with pytest.raises(ValueError,match='does not match'):validate_checkpoint(p2,fresh(p))


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
def test_parallel_gp_chart_updates_only_on_completion():
    from cards.pipeline_tune import chart_snapshot
    data=source()
    model=gp_problem().pipeline.named_steps.data_models
    data=data.with_estimator(model,label='GP',recipe_id='parallel-gp')
    p=prepare(data,model_values={'data_models__estimator__alpha':[.2,.3,.4]},execution='trials',workers=2,method='grid')
    changes=[];last=None
    def publish(state):
        nonlocal last
        snapshot=chart_snapshot(state)
        if snapshot!=last:
            changes.append(len(snapshot['trials']))
            last=snapshot
    result=finish(p,publish=publish)
    assert all(t['state'] == 'complete' for t in result['trials'])
    assert changes==[0,1,2,3,4]


class CrashOnceRidge(AuditRidge):
    def fit(self,X,y,**kwargs):
        marker=Path(self.log_path+'.once')
        try:
            with marker.open('x'):
                pass
        except FileExistsError:
            return super().fit(X,y,**kwargs)
        os._exit(7)


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
def test_crashed_fit_worker_reports_error_and_checkpoint_can_resume(tmp_path):
    data=source()
    model=data.pipeline.named_steps.data_models
    model.set_params(estimator=CrashOnceRidge(log_path=str(tmp_path/'crash.jsonl'),delay=0))
    data=data.with_estimator(model,label='Crash test',recipe_id='crash-once')
    p=prepare(data,execution='folds',workers=2,method='grid')
    failed=finish(p)
    assert failed['status']=='error'
    assert 'worker exited unexpectedly' in failed['error']
    validate_checkpoint(p,failed)
    resumed=finish(p,json.loads(json.dumps(failed)))
    assert resumed['status']=='exhausted'
    assert resumed['trials'][0]['state']=='complete'


@pytest.mark.parametrize('cpus,expected', [(1,1),(2,1),(3,1),(4,2),(10,5),(16,8)])
def test_auto_worker_count_tracks_available_cpus(monkeypatch,cpus,expected):
    from types import SimpleNamespace

    import psutil
    monkeypatch.setattr(joblib,'cpu_count',lambda:cpus)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(total=128*1024**3))
    assert prepare(source(),execution='auto',workers=0).specification['execution']['workers']==expected
    assert prepare(source(),execution='sequential',workers=0).specification['execution']['workers']==1
    assert prepare(source(),execution='auto',workers=3).specification['execution']['workers']==min(3,cpus)


@pytest.mark.skipif(joblib.cpu_count()<2,reason='Requires two available CPUs')
def test_committed_catboost_adapter_runs_in_spawned_tuning_workers():
    from estimator_catalogue import load_catalogue
    from model_commit import build_model
    from test_pipeline_search import source_without_plan
    data=source_without_plan()
    profile=copy.deepcopy(next(p for p in load_catalogue()['profiles'] if p['estimator']=='catboost.CatBoostRegressor'))
    profile['construction']['parameters'].update(iterations=5,thread_count=1)
    model=build_model({'selectable':True,'estimator':'catboost.CatBoostRegressor','profile':profile},data)
    data=data.with_estimator(model,label='CatBoost',recipe_id=profile['id'])
    p=prepare(data,execution='folds',workers=2,method='grid')
    assert 'catboost' in p.specification['versions']
    assert 'catboost_model.py' in p.specification['implementation']
    result=finish(p,trials=0)
    assert result['status']=='exhausted'
    assert result['trials'][0]['state']=='complete'
    assert len(result['trials'][0]['folds'])==5
    assert not data.pipeline.named_steps.data_models.named_steps.estimator.is_fitted()
