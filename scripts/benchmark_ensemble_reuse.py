"""Reproducible native cache benchmark (not a general model speed benchmark).

Run from the project root:
    PYTHONPATH=app .venv/bin/python scripts/benchmark_ensemble_reuse.py
The printed JSON contains observed timings, fit counts and sampled process-tree
RSS. Timings exclude plan preparation; cold timings include process startup.
Temporary fitted artifacts are removed.
"""
import argparse
import json
import time

import joblib
import pandas as pd
import psutil
from candidate_selection import candidate_frame, decisions
from ensemble import evaluate_ensemble
from fit_service import FitService
from model_commit import prepare_predictors
from model_review import completed_evidence
from pipeline_search import SearchWorker, fresh, prepare
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def source(rows):
    X, y = make_regression(n_samples=rows, n_features=6, noise=10, random_state=42)
    frame = pd.DataFrame(X, columns=[f'x{i}' for i in range(6)])
    columns = list(frame)
    frame['y'] = y
    data = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': columns, 'target': ['y']}))
    estimators = [Ridge(), DecisionTreeRegressor(max_depth=7),
        RandomForestRegressor(n_estimators=60, max_depth=7),
        ExtraTreesRegressor(n_estimators=60, max_depth=7), HistGradientBoostingRegressor(max_iter=40)]
    return data.with_model_candidates([{'id': f'method_{i}', 'label': type(model).__name__,
        'estimator': Pipeline([('predictors', FunctionTransformer(prepare_predictors,
            kw_args={'columns': columns, 'preparation': 'numeric'})), ('scale', StandardScaler()), ('estimator', model)])}
        for i, model in enumerate(estimators)])


def rss():
    process = psutil.Process()
    total = 0
    for child in [process, *process.children(recursive=True)]:
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def training(data, service):
    problem = prepare(data, method='grid', execution='folds', workers=2)
    worker = SearchWorker(fit_service=service)
    start, peak, state = time.monotonic(), rss(), None
    try:
        worker.start(problem, fresh(problem), trials=0, seconds=0)
        while worker.running:
            if time.monotonic() - start > 120:
                raise TimeoutError('Training benchmark exceeded two minutes.')
            state = worker.poll() or state
            peak = max(peak, rss())
            time.sleep(.02)
        state = worker.poll() or state
        if state['status'] != 'exhausted' or any(t['state'] != 'complete' for t in state['trials']):
            raise RuntimeError(f"Benchmark search failed: {state['status']}")
        seconds = time.monotonic() - start
        evidence = completed_evidence(problem, state)
        frame, _ = candidate_frame(evidence)
        selected = decisions(frame, problem.specification['metric'], [1, 0, 0], 1e-12)['methods']
        winners = {key: trial - 1 for key, trial in selected.items()}
        evidence['review'] = {'representatives': winners}
        service.runtime.pin_review('benchmark-review', problem.specification['fingerprint'],
            [joblib.hash(t['params']) for t in state['trials'] if t['id'] in winners.values()],
            [joblib.hash(t['params']) for t in state['trials']])
        return data.with_tuning_results(evidence), {'seconds': seconds, 'sampled_peak_rss_bytes': peak,
            'folds_per_second': sum(len(t['folds']) for t in state['trials']) / seconds}
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--cpus', type=int, default=4)
    args = parser.parse_args()
    data = source(args.rows)
    report = {'rows': args.rows, 'configured_cpus': args.cpus}
    services = []
    try:
        baseline = FitService(cpus=args.cpus); services.append(baseline)
        reviewed, report['training_without_artifacts'] = training(data, baseline)
        shared = FitService(cpus=args.cpus); services.append(shared); shared.enable()
        reviewed, report['training_with_artifacts'] = training(data, shared)
        keys = tuple(reviewed.tuning_results['review']['representatives'])
        cold = FitService(cpus=args.cpus); services.append(cold); cold.enable()
        result = evaluate_ensemble(reviewed, keys, 'mean', fit_service=cold)
        cold.publish_snapshot('card', result.cache_owner)
        report['cold_ensemble'] = result.diagnostics
        warm = evaluate_ensemble(reviewed, keys, 'median', fit_service=cold)
        cold.publish_snapshot('card', warm.cache_owner)
        report['warm_style_change'] = warm.diagnostics
        evidence = reviewed.tuning_results
        evidence['pipelines'].templates[keys[0]].set_params(cand_methods__estimator__alpha=2.)
        changed = reviewed.with_tuning_results(evidence)
        updated = evaluate_ensemble(changed, keys, 'mean', fit_service=cold)
        cold.publish_snapshot('card', updated.cache_owner)
        report['one_changed_member'] = updated.diagnostics
        reused = evaluate_ensemble(reviewed, keys, 'mean', fit_service=shared)
        shared.publish_snapshot('card', reused.cache_owner)
        report['reuse_training'] = reused.diagnostics
        report['shared_resources'] = shared.runtime.snapshot()
        print(json.dumps(report, indent=2))
    finally:
        for service in services:
            service.close()


if __name__ == '__main__':
    main()
