"""Failure feedback stays separate from the objective and replays from history."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from search_feasibility import failure_kind, feasibility_model, weighted_proposal


def history():
    spec = {'grid': {'region': ['good', 'bad'], 'extra': list(range(30))}, 'seed': 17}
    trials = [{'params': {'region': region, 'extra': i}, 'state': state, 'failure_kind': 'fit'}
              for region, state in [('good', 'complete'), ('bad', 'failed')] for i in range(6)]
    return spec, trials


def test_feasibility_learns_across_combinations_without_objective_values():
    spec, trials = history()
    model = feasibility_model(spec, trials)
    assert model({'region': 'good', 'extra': 29}) > model({'region': 'bad', 'extra': 29}) + .5
    # Running, legacy, and execution failures are not negative evidence.
    for trial in trials:
        if trial['state'] == 'failed':
            trial['failure_kind'] = 'execution'
    assert feasibility_model(spec, trials) is None
    for trial in trials:
        trial.pop('failure_kind', None)
    assert feasibility_model(spec, trials) is None
    assert feasibility_model(spec, [t for t in trials if t['state'] == 'failed']) is None


def test_weighting_prefers_feasibility_and_preserves_exploration_and_replay():
    spec, trials = history()
    model = feasibility_model(spec, trials)
    def propose(problem, state):
        rng = np.random.default_rng(problem.specification['seed'])
        return {'region': str(rng.choice(['good', 'bad'])), 'extra': 29}
    choices = []
    for seed in range(100):
        problem = SimpleNamespace(specification=dict(spec, seed=seed))
        state = {'rng': np.random.default_rng(seed).bit_generator.state}
        restored = json.loads(json.dumps(state))
        first = weighted_proposal(problem, state, propose, model)
        assert first == weighted_proposal(problem, restored, propose, model)
        assert state == restored
        choices.append(first['region'])
    assert 75 < choices.count('good') < 100
    # No feasible points still terminates and returns a candidate.
    assert weighted_proposal(problem, state, propose, lambda _: 0) is not None


@pytest.mark.parametrize('error, expected', [
    (ValueError('invalid parameter'), 'fit'), (ArithmeticError('singular'), 'fit'),
    (MemoryError(), 'execution'), (OSError(), 'execution'), (KeyError('schema'), 'execution'),
    (ValueError('CUDA out of memory'), 'execution'),
    (RuntimeError('unknown'), 'unknown')])
def test_failure_classification(error, expected):
    assert failure_kind(error) == expected


@pytest.mark.parametrize('method', ['tpe', 'skopt'])
def test_real_search_resume_and_failure_scores_stay_truthful(method, monkeypatch):
    import search_feasibility
    calls = []
    original = search_feasibility.weighted_proposal
    def tracked(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(search_feasibility, "weighted_proposal", tracked)
    from test_pipeline_search import source, committed_grid, finish
    from pipeline_search import prepare, validate_checkpoint
    data = source()
    grid = {'cand_methods__estimator__alpha': [-1., 1.],
            'cand_methods__estimator__tol': [.001, .002, .003, .004, .005, .006]}
    problem = prepare(committed_grid(data, grid), method=method)
    partial = finish(problem, trials=6)
    checkpoint = json.loads(json.dumps(partial))
    validate_checkpoint(problem, checkpoint)
    first = finish(problem, checkpoint)
    second = finish(problem, copy.deepcopy(checkpoint))
    assert [t['params'] for t in first['trials']] == [t['params'] for t in second['trials']]
    failures = [t for t in first['trials'] if t['state'] == 'failed']
    assert failures and all(t['failure_kind'] == 'fit' for t in failures)
    assert all(not t['folds'] for t in failures)
    assert first['status'] == 'exhausted'
    assert calls  # Both real optimizers actually used the feasibility layer.
    legacy = copy.deepcopy(first)
    for trial in legacy['trials']:
        trial.pop('failure_kind', None)
    validate_checkpoint(problem, legacy)
    assert feasibility_model(problem.specification, legacy['trials']) is None
    broken = copy.deepcopy(first)
    next(t for t in broken['trials'] if t['state'] == 'failed')['failure_kind'] = 'invalid'
    with pytest.raises(ValueError, match='failure classification'):
        validate_checkpoint(problem, broken)
