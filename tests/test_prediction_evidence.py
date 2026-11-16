"""Memory growth, portable configuration switching and browser-safe diagnostics."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import prediction_evidence as store
import pytest
from model_review import completed_evidence, review_frame
from pipeline_search import (
    SearchWorker,
    _UpdatePublisher,
    fresh,
    prepare,
    run,
    validate_checkpoint,
)
from test_pipeline_search import committed_grid, source

pytestmark = pytest.mark.unit


def paused_problem(seed=1729):
    problem = prepare(source(), method='grid', seed=seed)
    paused = [False]
    def publish(state):
        if state['trials'] and state['trials'][0]['folds']:
            paused[0] = True
    state = run(problem, fresh(problem), trials=0, publish=publish, pause=lambda: paused[0])
    assert state['status'] == 'paused'
    assert store.references(state)
    return problem, state


def test_portable_paused_checkpoint_resumes_after_original_files_are_removed():
    problem, state = paused_problem()
    portable = json.loads(json.dumps(store.portable_checkpoint(state)))
    assert portable['pending_evidence']['blobs']
    assert store.directory() not in json.dumps(portable)
    assert 'observed' not in json.dumps(portable) and 'predicted' not in json.dumps(portable)
    original_keys = set(store.references(state))
    store.remove(original_keys)
    restored = store.restore_checkpoint(portable)
    assert not original_keys & set(store.references(restored))
    validate_checkpoint(problem, restored)
    result = run(problem, restored, trials=0, publish=lambda s: None, pause=lambda: False)
    assert result['trials'][0]['state'] == 'complete'
    assert not store.references(result)
    assert 'pending_evidence' not in store.portable_checkpoint(result)
    uninterrupted = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    assert result['trials'][0]['diagnostics'] == uninterrupted['trials'][0]['diagnostics']


def test_configuration_switches_import_only_their_own_arrays():
    a, state_a = paused_problem(1729)
    b, state_b = paused_problem(1730)
    saved_a, saved_b = store.portable_checkpoint(state_a), store.portable_checkpoint(state_b)
    store.remove(store.references(state_a)); store.remove(store.references(state_b))
    restored_a = store.restore_checkpoint(saved_a)
    restored_b = store.restore_checkpoint(saved_b)
    assert set(store.references(restored_a)).isdisjoint(store.references(restored_b))
    validate_checkpoint(a, restored_a); validate_checkpoint(b, restored_b)
    values_b = [store.read_losses(ref).copy() for ref in store.references(restored_b).values()]
    store.remove(store.references(restored_a))
    for expected, ref in zip(values_b, store.references(restored_b).values()):
        np.testing.assert_array_equal(expected, store.read_losses(ref))
    again_a = store.restore_checkpoint(saved_a)
    validate_checkpoint(a, again_a)
    store.remove(store.references(restored_b)); store.remove(store.references(again_a))


def test_corrupt_or_missing_pending_archive_is_explicit():
    _, state = paused_problem()
    portable = store.portable_checkpoint(state)
    key = next(iter(portable['pending_evidence']['blobs']))
    portable['pending_evidence']['blobs'][key] = 'AAAA'
    with pytest.raises(ValueError, match='checksum'):
        store.restore_checkpoint(portable)
    store.remove(store.references(state))
    with pytest.raises(ValueError, match='unavailable'):
        store.portable_checkpoint(state)


def test_shinylive_restore_and_completed_reviews_do_not_use_native_files(monkeypatch):
    monkeypatch.setattr(store, 'sys', SimpleNamespace(platform='emscripten'))
    monkeypatch.setattr(store, '_path', lambda *a, **k: pytest.fail('Host filesystem accessed'))
    ref = {**store.write_losses([1., 2., 3.]), 'version': 2, 'count': 3, 'prediction_seconds': .01,
           'loss_kind': 'absolute_error', 'maximum': 3.}
    state = {'trials': [{'id': 0, 'fold_ids': [0], 'folds': [{'evidence': ref}]}]}
    portable = store.portable_checkpoint(state)
    store.remove(store.references(state))
    restored = store.restore_checkpoint(portable)
    trial = restored['trials'][0]
    retired = store.summarize_trial(trial, {'task': 'regression', 'folds': 1})
    store.remove(retired)
    assert trial['diagnostics']['p95_absolute_error'] == pytest.approx(2.9)
    assert 'pending_evidence' not in store.portable_checkpoint(restored)
    assert not store.references(restored)
    assert not store._MEMORY


def test_exact_repeat_tail_and_sampled_curves_are_preserved():
    arrays = [[1., 2.], [3., 4.], [10., 20.], [30., 40.]]
    folds = [{'evidence': {**store.write_losses(a), 'version': 2, 'count': 2,
                          'prediction_seconds': .01, 'loss_kind': 'absolute_error', 'maximum': max(a)}} for a in arrays]
    trial = {'fold_ids': [0, 1, 2, 3], 'folds': folds}
    spec = {'task': 'regression', 'folds': 4, 'evaluation': {'split_descriptors': [{'repeat': r} for r in (1, 1, 2, 2)]}}
    retired = store.summarize_trial(trial, spec)
    store.remove(retired)
    summary = trial['diagnostics']
    assert summary['p95_absolute_error'] == pytest.approx((np.quantile([1,2,3,4], .95) + np.quantile([10,20,30,40], .95)) / 2)
    for x, rate in zip(np.linspace(0., summary['maximum'], 101), summary['exceedance']):
        expected = (np.mean(np.asarray([1,2,3,4]) > x) + np.mean(np.asarray([10,20,30,40]) > x)) / 2
        assert rate == pytest.approx(expected)
    assert not store.references({'trials': [trial]})


def test_completed_trial_storage_does_not_scale_with_observation_count():
    from model_commit import prepare_predictors
    from proxy_data import proxy_data
    from roles import RoleMap
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    sizes = []
    before = set(Path(store.directory()).glob('*.npz'))
    for count in (200, 20000):
        x = np.arange(count, dtype=float)
        data = proxy_data(pd.DataFrame({'x': x, 'y': x + np.sin(x)}),
                          _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
        model = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
                          kw_args={'columns': ['x'], 'preparation': 'numeric'})), ('estimator', Ridge())])
        data = data.with_estimator(model, label='Ridge', recipe_id='ridge')
        problem = prepare(committed_grid(data, {'cand_methods__estimator__alpha': [.1, 10.]}), method='grid')
        state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
        assert all(t['state'] == 'complete' for t in state['trials'])
        sizes.append(len(json.dumps(state['trials'])))
        assert not store.references(state)
        assert all(len(t['diagnostics']['exceedance']) == 101 for t in state['trials'])
        assert len(completed_evidence(problem, state)['pipelines'].templates) == 1
    assert sizes[1] < 2 * sizes[0]  # 100× more observations, bounded persisted trial size.
    assert set(Path(store.directory()).glob('*.npz')) == before


def test_progress_publishes_only_changed_trials_and_no_large_specification():
    messages = []
    sender = _UpdatePublisher(SimpleNamespace(put=messages.append))
    state = {'specification': {'large': 'x' * 100000}, 'status': 'running', 'elapsed': 0,
             'trials': [{'id': n, 'state': 'complete', 'folds': [], 'fold_ids': [], 'elapsed': 0, 'error': ''} for n in range(100)]}
    sender(state)
    state['trials'][-1]['elapsed'] = 1
    sender(state)
    assert len(messages[-1]['trials']) == 1
    assert 'specification' not in messages[-1]['header']
    assert len(json.dumps(messages[-1])) < 1000


def test_legacy_configuration_drops_redundant_rows_but_preserves_scores():
    original = {'specification': {'prediction_evidence': 1}, 'trials': [
        {'id': 0, 'state': 'complete', 'fold_ids': [0], 'folds': [
            {'metric': .8, 'evidence': {'observed': [1.] * 10000, 'predicted': [2.] * 10000, 'positions': list(range(10000))}}]}]}
    restored = store.restore_checkpoint(original)
    assert restored['trials'][0]['folds'][0]['metric'] == .8
    assert len(json.dumps(restored)) < 1000
    assert original['trials'][0]['folds'][0]['evidence']['observed']
    assert restored['legacy_evidence_removed']


def test_checkpoint_rehydrates_in_a_new_native_cache_after_restart(monkeypatch, tmp_path):
    import tempfile
    problem, state = paused_problem()
    saved = json.loads(json.dumps(store.portable_checkpoint(state)))
    store.remove(store.references(state))
    with tempfile.TemporaryDirectory(dir=tmp_path) as new_cache:  # noqa: SIM117
        with monkeypatch.context() as context:
            context.setattr(store, '_TEMP', SimpleNamespace(name=new_cache))
            rebuilt = prepare(source(), method='grid')
            assert rebuilt.evidence_directory != problem.evidence_directory
            assert rebuilt.specification == problem.specification
            restored = store.restore_checkpoint(saved)
            validate_checkpoint(rebuilt, restored)
            result = run(rebuilt, restored, trials=0, publish=lambda s: None, pause=lambda: False)
            assert result['trials'][0]['state'] == 'complete'
            assert not store.references(result)
            assert not list(Path(new_cache).glob('*.npz'))


def test_candidate_methods_share_one_evaluation_membership_record():
    from test_multi_model_search import configured
    problem = prepare(configured(), method='grid')
    for key, child in problem.branches.items():
        assert child.specification['evaluation'] is problem.specification['evaluation']
        assert 'evaluation' not in problem.specification['branches'][key]
        assert 'split_sizes' not in problem.specification['branches'][key]


def test_completed_historical_review_is_file_free_and_cannot_accept(monkeypatch):
    from model_review import archived_evidence
    problem = prepare(source(), method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    saved = json.loads(json.dumps(store.portable_checkpoint(state)))
    monkeypatch.setattr(store, '_path', lambda *a, **k: pytest.fail('Completed review accessed native files'))
    restored = store.restore_checkpoint(saved)
    results = archived_evidence(source(), restored)
    frame, note = review_frame(results)
    assert len(frame) == 1 and frame['P95 absolute error'].notna().all()
    assert 'read-only' in note
    data = source().with_tuning_results(results)
    with pytest.raises(ValueError, match='read-only'):
        data.with_selected_model(0)


def test_native_worker_queue_is_bounded_and_completed_files_are_retired():
    import time
    problem = prepare(committed_grid(source(), {'cand_methods__estimator__alpha': [.1, 10.]}), method='grid')
    worker = SearchWorker()
    try:
        worker.start(problem, fresh(problem), 0, 0)
        namespace = worker._namespace
        assert worker.updates._maxsize == 2
        deadline = time.monotonic() + 30
        result = None
        while time.monotonic() < deadline:
            latest = worker.poll()
            if latest is not None:
                result = latest
                if result['status'] != 'running':
                    break
            time.sleep(.02)
        assert result is not None and result['status'] == 'exhausted'
        assert len(result['trials']) == 3
        assert not store.references(result)
        assert not list(Path(store.directory()).glob(namespace + '_*.npz'))
    finally:
        worker.close()
