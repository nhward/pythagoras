"""Native/shared and browser reuse without portable fitted artifacts."""
import copy
import threading
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
from candidate_selection import candidate_frame, decisions
from ensemble import evaluate_ensemble, representatives
from ensemble_scenario import example_source
from fit_responses import collect_responses, score_responses
from fit_service import FitService, MemoryFitService, fit_key
from model_review import completed_evidence
from pipeline_search import METRICS, SearchWorker, fresh, prepare
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer


@pytest.fixture(scope='module')
def source():
    return example_source(count=5)


@pytest.fixture
def service(tmp_path):
    value = FitService(cpus=3, directory=tmp_path)
    value.enable()
    value.runtime.register_review('review-test')
    yield value
    value.close()
    assert not Path(value.root).exists()


def test_classification_cached_scorers_match_sklearn():
    X, y = make_classification(n_samples=60, random_state=42)
    model = LogisticRegression().fit(X[:40], y[:40])
    responses = collect_responses(model, X[40:], 'classification')
    scores, errors = score_responses(responses, X[40:], y[40:], 'classification', METRICS['classification'])
    assert not errors
    for metric, value in scores.items():
        assert value == pytest.approx(get_scorer(metric)(model, X[40:], y[40:]))


def test_cold_warm_style_subset_and_one_changed_member(source, service):
    keys = tuple(representatives(source)[1])
    cold = evaluate_ensemble(source, keys, 'mean', fit_service=service)
    service.publish_snapshot('card', cold.cache_owner)
    assert cold.diagnostics['new_fits'] == 25
    assert cold.diagnostics['reused_folds'] == 0
    warm = evaluate_ensemble(source, keys, 'median', fit_service=service)
    service.publish_snapshot('card', warm.cache_owner)
    assert warm.diagnostics['new_fits'] == 0
    assert warm.diagnostics['reused_folds'] == 25
    assert warm.fitted[0].estimators_[0][1] is cold.fitted[0].estimators_[0][1]
    subset = evaluate_ensemble(source, keys[:2], 'mean', fit_service=service)
    assert subset.diagnostics['new_fits'] == 0
    changed = copy.deepcopy(source.tuning_results)
    changed['pipelines'].templates[keys[0]].set_params(cand_methods__estimator__alpha=123)
    updated = source.with_tuning_results(changed)
    one = evaluate_ensemble(updated, keys, 'mean', fit_service=service)
    assert one.diagnostics['new_fits'] == 5
    assert one.diagnostics['reused_folds'] == 20
    assert service.runtime.snapshot()['ensemble_fits'] == 30
    assert service.runtime.snapshot()['context_bytes'] == 0
    # Display aggregation consumes responses; ordinary downstream prediction works.
    X = source.clean_frame.iloc[list(one.splits[0][1])]
    assert np.isfinite(one.fitted[0].predict(X)).all()


def test_training_fits_reused_without_ensemble_fits_or_checkpoint_paths(source, service):
    raw = replace(source, _selected_candidate=None)
    problem = prepare(raw, method='grid')
    worker = SearchWorker(fit_service=service)
    try:
        worker.start(problem, fresh(problem), trials=0, seconds=0)
        deadline = time.monotonic() + 60
        state = None
        while worker.running and time.monotonic() < deadline:
            state = worker.poll() or state
            time.sleep(.03)
        state = worker.poll() or state
        assert not worker.running
        assert state['status'] == 'exhausted', state
        evidence = completed_evidence(problem, state)
        frame, _ = candidate_frame(evidence)
        chosen = decisions(frame, problem.specification['metric'], [1, 0, 0], 1e-12)
        winners = {k: v - 1 for k, v in chosen['methods'].items()}
        evidence['review'] = {'representatives': winners}
        service.runtime.pin_review('review', problem.specification['fingerprint'],
            [joblib.hash(t['params']) for t in state['trials'] if t['id'] in winners.values()],
            [joblib.hash(t['params']) for t in state['trials']])
        worker.close()
        reviewed = raw.with_tuning_results(evidence)
        expected_fits = len(winners) * len(problem.splits)
        assert expected_fits > 0
        assert service.runtime.snapshot()['train_fits'] == expected_fits
        result = evaluate_ensemble(reviewed, tuple(winners), 'mean', fit_service=service)
        assert result.diagnostics['new_fits'] == 0
        assert result.diagnostics['reused_folds'] == expected_fits
        assert service.runtime.snapshot().get('ensemble_fits', 0) == 0
        assert service.runtime.snapshot()['train_fits'] == expected_fits
        assert service.root not in repr(state)
        assert service.root not in repr(evidence)
    finally:
        worker.close()


def test_browser_sequential_cache(source):
    service = MemoryFitService()
    keys = tuple(representatives(source)[1])
    first = evaluate_ensemble(source, keys, 'mean', fit_service=service)
    second = evaluate_ensemble(source, keys, 'median', fit_service=service)
    assert first.diagnostics['new_fits'] == 25
    assert second.diagnostics['new_fits'] == 0
    assert len(service.entries) == 25
    evaluate_ensemble(source, keys[:2], 'mean', fit_service=service)
    assert len(service.entries) == 10
    service.close()
    assert not service.entries


def test_training_priority_nonpreemptive_limit_and_expansion(tmp_path):
    service = FitService(cpus=10, directory=tmp_path)
    runtime = service.runtime
    release = threading.Event()
    entered = [threading.Event() for _ in range(3)]
    threads = []
    try:
        runtime.register_search('search')
        assert runtime.ensemble_width() == 2
        def ensemble(i):
            with runtime.lease('ensemble'):
                entered[i].set()
                release.wait(5)
        for i in range(3):
            thread = threading.Thread(target=ensemble, args=(i,))
            thread.start(); threads.append(thread)
        deadline = time.monotonic() + 3
        while sum(e.is_set() for e in entered) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        assert sum(e.is_set() for e in entered) == 2
        # Existing fits continue; a fully stopped search expands spare capacity.
        runtime.finish_search('search')
        assert runtime.ensemble_width() == 9
        assert all(e.wait(2) for e in entered)
        assert not release.is_set()
    finally:
        release.set()
        for thread in threads: thread.join(5)
        service.close()


def test_pins_eviction_corruption_and_shutdown(source, tmp_path):
    service = FitService(cpus=2, cache_budget=0, directory=tmp_path)
    runtime = service.runtime
    try:
        problem = prepare(source)
        train, validation = problem.splits[0]
        model = problem.pipeline.fit(problem.frame.iloc[train], problem.frame[problem.target].iloc[train])
        response = collect_responses(model, problem.frame.iloc[validation], 'regression')
        key = 'a' * 32
        assert runtime.put(key, model, response, 'current') is None
        assert runtime.snapshot()['pinned_bytes'] > 0
        runtime.prune()
        assert runtime.path(key).exists()
        runtime.release('current')
        assert not runtime.path(key).exists()
        runtime.put(key, model, response, 'corrupt')
        runtime.path(key).write_bytes(b'not a joblib file')
        assert runtime.get(key, 'reader') is None
        assert runtime.snapshot()['cache_misses_corrupt'] == 1
        runtime.register_search('active')
        service.stop()
        with pytest.raises(InterruptedError), runtime.lease('ensemble'): pass
    finally:
        service.close()
    assert not Path(service.root).exists()


def test_waiting_training_precedes_new_ensemble_work_and_cancel_releases_lease(tmp_path):
    service = FitService(cpus=2, memory_budget=100, directory=tmp_path)
    runtime = service.runtime
    trained, ensemble, release, cancel = (threading.Event() for _ in range(4))
    threads = []
    def train():
        with runtime.lease('train', 80):
            trained.set()
            release.wait(5)
    def evaluate():
        with runtime.lease('ensemble', 80):
            ensemble.set()
    try:
        with runtime.lease('ensemble', 80):
            thread = threading.Thread(target=train); thread.start(); threads.append(thread)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with runtime.database() as db:
                    waiting = db.execute('SELECT count(*) FROM leases WHERE kind="train" AND state="waiting"').fetchone()[0]
                if waiting: break
                time.sleep(.01)
            assert waiting
            thread = threading.Thread(target=evaluate); thread.start(); threads.append(thread)
        assert trained.wait(2)
        assert not ensemble.wait(.1)
        cancel.set()
        with pytest.raises(InterruptedError), runtime.lease('ensemble', 80, cancel): pass
        release.set()
        assert ensemble.wait(2)
    finally:
        release.set()
        for thread in threads: thread.join(5)
        service.close()


class UnserializableRidge(Ridge):
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.native_resource_ = threading.Lock()
        return self


def test_unserializable_fitted_provider_still_evaluates_and_reuses_live_models(source, service):
    evidence = source.tuning_results
    keys = tuple(representatives(source)[1])[:2]
    evidence['pipelines'].templates[keys[0]].set_params(cand_methods__estimator=UnserializableRidge())
    updated = source.with_tuning_results(evidence)
    first = evaluate_ensemble(updated, keys, 'mean', fit_service=service)
    assert first.diagnostics['serialization_fallbacks']
    assert first.table('r2').Score.notna().all()
    second = evaluate_ensemble(updated, keys, 'median', fit_service=service)
    assert second.diagnostics['new_fits'] == 0
    assert second.fitted[0].estimators_[0][1] is first.fitted[0].estimators_[0][1]


def test_fit_identity_changes_with_data_splits_parameters_and_seed(source):
    problem = prepare(source)
    train, validation = problem.splits[0]
    key = fit_key(problem.fit_context, problem.pipeline, train, validation)
    assert key != fit_key(problem.fit_context, problem.pipeline, train[::-1], validation)
    assert key != fit_key(problem.fit_context, problem.pipeline, train, validation[::-1])
    assert key != fit_key(prepare(source, seed=1).fit_context, problem.pipeline, train, validation)
    changed_frame = source.clean_frame.assign(x=source.clean_frame.x + 1)
    changed = replace(source, _df=changed_frame, _clean_df=changed_frame)
    assert prepare(changed).fit_context != problem.fit_context
    pipeline = copy.deepcopy(problem.pipeline).set_params(cand_methods__estimator__alpha=123)
    assert key != fit_key(problem.fit_context, pipeline, train, validation)


def test_fit_identity_ignores_duplicate_distribution_paths(source, monkeypatch):
    import fit_service

    problem = prepare(source)
    train, validation = problem.splits[0]
    distributions = list(fit_service.metadata.distributions())
    monkeypatch.setattr(fit_service, '_VERSIONS', None)
    monkeypatch.setattr(fit_service.metadata, 'distributions', lambda: iter(distributions))
    key = fit_key(problem.fit_context, problem.pipeline, train, validation)
    # A spawned IDE runner adds the same bundled libraries to sys.path again.
    monkeypatch.setattr(fit_service, '_VERSIONS', None)
    monkeypatch.setattr(fit_service.metadata, 'distributions', lambda: iter(distributions * 2))
    assert key == fit_key(problem.fit_context, problem.pipeline, train, validation)


class SlowRidge(Ridge):
    def fit(self, X, y, **kwargs):
        time.sleep(.2)
        return super().fit(X, y, **kwargs)


def test_cancellation_releases_late_worker_pins_and_data(source, service):
    from concurrent.futures import ThreadPoolExecutor
    evidence = source.tuning_results
    keys = tuple(representatives(source)[1])[:2]
    evidence['pipelines'].templates[keys[0]].set_params(cand_methods__estimator=SlowRidge())
    updated = source.with_tuning_results(evidence)
    cancel = threading.Event()
    with ThreadPoolExecutor(1) as thread:
        future = thread.submit(evaluate_ensemble, updated, keys, 'mean', cancel, fit_service=service)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if service.runtime.snapshot()['ensemble_running']: break
            time.sleep(.01)
        cancel.set()
        with pytest.raises(InterruptedError): future.result(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = service.runtime.snapshot()
        if status['pinned_bytes'] == status['context_bytes'] == 0: break
        time.sleep(.03)
    assert status['pinned_bytes'] == status['context_bytes'] == 0


def test_abandoned_session_sweep_keeps_live_sessions(tmp_path):
    import json

    from fit_service import cleanup_abandoned
    service = FitService(cpus=2, directory=tmp_path)
    dead = tmp_path / 'pythagoras-fit-abandoned'
    dead.mkdir()
    (dead / 'owner.json').write_text(json.dumps([99999999, 0]))
    try:
        cleanup_abandoned(tmp_path)
        assert not dead.exists()
        assert Path(service.root).exists()
    finally:
        service.close()


def test_collects_each_response_once_and_keeps_probability_scoring_when_labels_fail(monkeypatch):
    X, y = make_classification(n_samples=60, random_state=42)
    model = LogisticRegression().fit(X[:40], y[:40])
    calls = {'predict': 0, 'predict_proba': 0, 'decision_function': 0}
    for name in calls:
        original = getattr(model, name)
        def response(frame, name=name, original=original):
            calls[name] += 1
            # LogisticRegression.predict calls its decision_function internally;
            # count externally requested responses rather than that internal call.
            if name == 'predict':
                raise ValueError('Labels unavailable')
            return original(frame)
        monkeypatch.setattr(model, name, response)
    responses = collect_responses(model, X[40:], 'classification')
    scores, errors = score_responses(responses, X[40:], y[40:], 'classification', METRICS['classification'])
    assert calls == {'predict': 1, 'predict_proba': 1, 'decision_function': 2}
    assert scores['neg_log_loss'] is not None
    assert scores['balanced_accuracy'] is None
    assert 'Labels unavailable' in errors['balanced_accuracy']


def test_pending_artifacts_need_an_active_review_consumer(source, tmp_path):
    service = FitService(cpus=2, cache_budget=0, directory=tmp_path)
    runtime = service.runtime
    try:
        problem = prepare(source)
        train, validation = problem.splits[0]
        model = problem.pipeline.fit(problem.frame.iloc[train], problem.frame[problem.target].iloc[train])
        response = collect_responses(model, problem.frame.iloc[validation], 'regression')
        key = 'b' * 32
        runtime.put(key, model, response, 'pending:search:trial', 'search', 'trial')
        runtime.retire_unreviewed('search', 'trial')
        assert not runtime.path(key).exists()
        runtime.register_review('review')
        runtime.put(key, model, response, 'pending:search:trial', 'search', 'trial')
        runtime.retire_unreviewed('search', 'trial')
        assert runtime.path(key).exists()
        runtime.enable()
        runtime.pin_review('review', 'search', ['trial'], ['trial'])
        assert runtime.path(key).exists()
        runtime.remove_review('review')
        assert not runtime.path(key).exists()
    finally:
        service.close()
