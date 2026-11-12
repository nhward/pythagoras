"""Finite, fold-fitted search with JSON checkpoints and isolated native workers.

No full-data preview is fitted, no holdout is scored, and no fitted estimator
is exported. Optimizers propose configurations; this module owns evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from roles import Role
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    StratifiedKFold,
)
from tuning import (
    applicable,
    condition_terms,
    conditional_grids,
    discover,
    search_values,
)

METRICS = {
    'regression': {'neg_root_mean_squared_error': 'RMSE · minimize',
                   'neg_mean_absolute_error': 'MAE · minimize', 'r2': 'R² · maximize'},
    'classification': {'balanced_accuracy': 'Balanced accuracy · maximize',
                       'accuracy': 'Accuracy · maximize', 'roc_auc': 'ROC AUC · maximize',
                       'neg_log_loss': 'Log loss · minimize'},
}
EXECUTIONS = {'auto': 'Auto', 'trials': 'Trials', 'folds': 'Folds', 'sequential': 'Sequential'}

METHODS = {'random': 'Random', 'grid': 'Exhaustive', 'tpe': 'Bayesian TPE', 'skopt': 'Bayesian GP'}


def displayed(score, metric):
    return None if score is None else (-score if metric.startswith('neg_') else score)


def terminal_model(pipeline):
    model = pipeline
    while hasattr(model, 'steps'):
        model = model.steps[-1][1]
    return model


def model_task(source):
    if len(source.model_candidates) > 1 and source.selected_model_candidate is None:
        tasks = {model_task(source.with_estimator(c['estimator'], label=c['label'], recipe_id=c['id']))
                 for c in source.model_candidates}
        if len(tasks) != 1:
            raise ValueError('All committed methods must match the same outcome task.')
        return tasks.pop()
    pipeline = source.pipeline
    if pipeline is None or not pipeline.steps or pipeline.steps[-1][0] != 'data_models':
        raise ValueError('Commit a model in Model method candidates first.')
    task = source.modelling_context.resolved_task(source)
    if task == 'regression' and is_regressor(pipeline):
        return task
    if task in {'classification', 'ordinal'} and is_classifier(pipeline):
        return 'classification'
    raise ValueError('This version supports committed classification and regression models matching the outcome task.')


@dataclass
class SearchProblem:
    pipeline: object
    frame: object
    target: str
    splits: list
    specification: dict
    branches: dict = field(default_factory=dict)


def implicit_splits(source, target, task, seed):
    """Independent-row CV only; never infer partition/time/group semantics."""
    roles = source.role_map
    if any(roles.columns_with_role(role) for role in (Role.PARTITION, Role.SEQUENCE, Role.GEOMETRY)):
        raise ValueError('Implicit CV cannot infer partition, temporal or spatial boundaries. Commit an upstream evaluation plan.')
    for name in roles.columns_with_role(Role.IDENTIFIER):
        if not source.clean_frame[name].is_unique:
            raise ValueError('Repeated identifiers require an explicit upstream evaluation plan for grouped or temporal resampling.')
    if roles.columns_with_role(Role.STRATIFIER):
        raise ValueError('Assigned stratifiers require an explicit upstream evaluation plan.')
    y = source.clean_frame[target]
    observed = np.flatnonzero(y.notna().to_numpy())
    prediction = np.flatnonzero(y.isna().to_numpy()).tolist()
    folds = min(5, len(observed))
    strategy = 'kfold'
    if task == 'classification':
        counts = y.iloc[observed].value_counts()
        counts = counts[counts > 0]
        if len(counts) < 2:
            raise ValueError('Implicit classification CV requires at least two observed classes.')
        folds = min(folds, int(counts.min()))
        strategy = 'stratified'
    if folds < 2:
        raise ValueError('Implicit CV requires at least two observations (and two per class for classification).')
    splitter = (StratifiedKFold if strategy == 'stratified' else KFold)(
        n_splits=folds, shuffle=True, random_state=seed)
    iterator = splitter.split(observed, y.iloc[observed]) if strategy == 'stratified' else splitter.split(observed)
    splits = [(observed[a], observed[b]) for a, b in iterator]
    evaluation = {'origin': 'implicit', 'strategy': strategy, 'folds': folds, 'seed': seed,
                  'prediction': prediction, 'holdout': []}
    return splits, evaluation


def execution_settings(pipeline, frame, splits, mode, workers):
    """Stable resource bounds; available RAM is too volatile for search identity."""
    if mode not in EXECUTIONS or isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or not 0 <= workers <= 64:
        raise ValueError('Choose an execution mode and 0–64 workers (zero means automatic).')
    cpus = max(1, joblib.cpu_count())
    try:
        import psutil
        budget = int(psutil.virtual_memory().total // 4)
    except ImportError:
        budget = 2 * 1024**3
    estimate = 256 * 1024**2 + 6 * int(frame.memory_usage(deep=True).sum())
    if isinstance(terminal_model(pipeline), GaussianProcessRegressor):
        estimate += 8 * 6 * max(len(train) for train, _ in splits)**2
    requested = int(workers)
    count = requested or max(1, cpus // 2)
    count = 1 if mode == 'sequential' else max(1, min(count, cpus, budget // estimate))
    return {'mode': mode, 'requested_workers': requested, 'workers': count,
            'memory_budget_bytes': budget, 'estimated_fit_bytes': estimate}


def prepare(source, *, model_values=None, metric=None, secondary=(), method='random', seed=1729, execution='sequential', workers=1):
    if len(source.model_candidates) > 1 and source.selected_model_candidate is None:
        return prepare_candidates(source, model_values=model_values, metric=metric, secondary=secondary,
                                  method=method, seed=seed, execution=execution, workers=workers)
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**31:
        raise ValueError('Choose an integer seed between 0 and 2147483647.')
    task = model_task(source)
    if method not in METHODS:
        raise ValueError('Choose Random, Exhaustive, Bayesian TPE or Bayesian GP.')
    if method == 'tpe':
        try:
            import optuna  # noqa: F401
        except ImportError as exc:
            raise ValueError('Bayesian TPE requires Optuna. Restore project dependencies with uv sync --inexact.') from exc
    if method == 'skopt':
        try:
            import skopt  # noqa: F401
        except ImportError as exc:
            raise ValueError('Bayesian GP requires scikit-optimize. Restore project dependencies with uv sync --inexact.') from exc
    plan = source.evaluation_plan
    if plan is not None:
        plan.require(source, supervised=True)
    if source.role_map.columns_with_role(Role.WEIGHTING):
        raise ValueError('Weighted fitting/scoring needs a row-aligned metadata adapter; this version cannot tune weighted data.')
    targets = list(source.role_map.columns_with_role(Role.TARGET))
    if len(targets) != 1:
        raise ValueError('Assign exactly one Target before tuning.')
    pipeline = source.pipeline_for_training()
    grid = copy.deepcopy((source.tuning_spec or {}).get('grid', {}))
    if source._tuning_spec is not None and source.tuning_spec is None:
        raise ValueError('The committed tuning specification is stale. Review and commit it again.')
    known = {r['path']: r for r in discover(pipeline)}
    model_values = {} if model_values is None else model_values
    if not isinstance(model_values, dict):
        raise TypeError('Model parameter choices must be a JSON object of full parameter paths and lists.')
    for path, values in model_values.items():
        row = known.get(path)
        if row is None or not path.startswith('data_models__') or row['fixed']:
            raise ValueError(f'Not a tunable model parameter: {path}')
        values = search_values({'kind': row['definition']['kind'] or 'categorical', 'values': values}, row['metadata'])
        if path in grid and grid[path] != values:
            raise ValueError(f'{path} already has committed choices. Change them in Tuning parameters.')
        grid[path] = values
    # Validate incoming grids too: never permit fixed configuration or topology tuning.
    for path, values in grid.items():
        row = known.get(path)
        if row is None or row['fixed']:
            raise ValueError(f'Unavailable or fixed parameter: {path}')
        search_values({'kind': row['definition']['kind'] or 'categorical', 'values': values}, row['metadata'])
    conditions = {p: known[p]['condition'] for p in grid if known[p].get('condition')}
    parameter_grids = conditional_grids(grid, conditions)
    combinations = len(ParameterGrid(parameter_grids))
    metric = metric or next(iter(METRICS[task]))
    secondary = [s for s in METRICS[task] if s != metric]
    if any(s not in METRICS[task] for s in [metric, *secondary]):
        raise ValueError('Choose metrics compatible with the committed outcome task.')
    model = pipeline.steps[-1][1]
    for score in [metric]:
        if score == 'roc_auc' and not any(callable(getattr(model, a, None)) for a in ('predict_proba', 'decision_function')):
            raise ValueError('ROC AUC requires predict_proba or decision_function.')
        if score == 'neg_log_loss' and not callable(getattr(model, 'predict_proba', None)):
            raise ValueError('Log loss requires predict_proba.')
    if plan is None:
        splits, evaluation = implicit_splits(source, targets[0], task, int(seed))
    else:
        splits = [(np.asarray(a), np.asarray(b)) for a, b in plan.splits(source)]
        evaluation = {'origin': 'upstream', 'strategy': plan.recipe['strategy'],
                      'repeats': plan.repeats, 'folds_per_repeat': plan.splits_per_repeat,
                      'split_descriptors': list(plan.split_descriptors),
                      'plan': plan.bookmark(), 'holdout': list(plan.holdout), 'prediction': list(plan.prediction)}
    forbidden = set(evaluation['holdout']) | set(evaluation['prediction'])
    used = set()
    for train, validation in splits:
        if not len(train) or not len(validation) or set(train) & set(validation):
            raise ValueError('Evaluation splits must have nonempty disjoint training and validation sets.')
        if (set(train) | set(validation)) & forbidden:
            raise ValueError('Evaluation splits overlap final holdout or prediction rows.')
        used.update(train); used.update(validation)
    frame = source.clean_frame.copy(deep=True)
    if frame[targets[0]].iloc[sorted(used)].isna().any():
        raise ValueError('Evaluation rows must have observed Target values.')
    # Fix otherwise-unseeded estimators, and parallelize only at the worker level.
    fixed = {k: int(seed) for k, v in pipeline.get_params(deep=True).items()
             if k.endswith('__random_state') and v is None}
    fixed.update({k: 1 for k in pipeline.get_params(deep=True)
                  if k.endswith(('__n_jobs', '__thread_count', '__nthread', '__num_threads'))})
    pipeline.set_params(**fixed)
    resources = execution_settings(pipeline, frame, splits, execution, workers)
    records = [r for r in source.processing_records if r.card == 'data_models' and r.attempted]
    selected = next((c for c in source.model_candidates if c['id'] == source.selected_model_candidate), None)
    model_id = selected['id'] if selected else str(records[-1].parameters.get('recipe_id', '')) if records else type(model).__qualname__
    method_label = selected['label'] if selected else records[-1].method if records else type(model).__qualname__
    packages = {'scikit-learn', 'numpy', 'pandas', 'scipy', 'feature-engine', 'imbalanced-learn', 'joblib'}
    owners = {cls.__module__.split('.')[0] for v in pipeline.get_params(deep=True).values()
              if hasattr(v, 'get_params') for cls in type(v).__mro__}
    distributions = importlib.metadata.packages_distributions()
    packages.update(p for owner in owners for p in distributions.get(owner, []))
    if method == 'tpe': packages.add('optuna')
    if method == 'skopt': packages.add('scikit-optimize')
    versions = {p: importlib.metadata.version(p) for p in sorted(packages)}
    root = Path(__file__).resolve().parent
    files = {Path(__file__), root / 'pipeline.py', root / 'tuning.py', root / 'model_commit.py',
             root / 'bundled_parameters.py'}
    for value in pipeline.get_params(deep=True).values():
        try:
            file = inspect.getsourcefile(value if callable(value) else type(value))
            if file and Path(file).resolve().is_relative_to(root): files.add(Path(file))
        except TypeError: pass
    implementation = {str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    specification = {'version': 4, 'model_id': model_id, 'method_label': method_label, 'task': task, 'grid': grid, 'combinations': combinations,
                         'metric': metric, 'secondary': secondary, 'method': method, 'seed': int(seed),
                         'fixed': fixed, 'execution': resources, 'folds': len(splits),
                         'split_sizes': [[len(a), len(b)] for a, b in splits], 'evaluation': evaluation, 'versions': versions, 'implementation': implementation}
    if conditions:
        specification.update(conditions=conditions, parameter_grids=parameter_grids)
    specification['fingerprint'] = joblib.hash((pipeline, frame, source.role_map.to_primitive(),
        source.modelling_context.to_primitive(), evaluation, splits, specification))
    return SearchProblem(pipeline, frame, targets[0], splits, specification)


def fresh(problem):
    spec = copy.deepcopy(problem.specification)
    return {'version': 4, 'specification': spec, 'status': 'ready', 'trials': [], 'elapsed': 0.,
                'rng': np.random.default_rng(spec['seed']).bit_generator.state, 'batch': []}


def baseline_parameters(problem):
    spec = problem.specification
    current = {k: problem.pipeline.get_params().get(k) for k in spec['grid']}
    return {k: v for k, v in current.items() if applicable(spec.get('conditions', {}).get(k), current)}


def in_search_grid(spec, parameters):
    return any(set(parameters) == set(grid) and all(
        any(type(value) is type(choice) and value == choice for choice in grid[path])
        for path, value in parameters.items())
        for grid in spec.get('parameter_grids', [spec['grid']]))


def _validate_single_checkpoint(problem, checkpoint):
    """Accept JSON data only; reject stale or structurally invalid evaluations."""
    state = json.loads(json.dumps(checkpoint, allow_nan=False))
    if state.get('version') != 4 or state.get('specification') != problem.specification:
        raise ValueError('Checkpoint does not match the current data, pipeline, search space, splits, objective or software.')
    spec = problem.specification
    if not isinstance(state.get('trials'), list) or not isinstance(state.get('elapsed'), (int, float)) or state['elapsed'] < 0:
        raise ValueError('Invalid search checkpoint.')
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng']
    seen = set()
    for i, trial in enumerate(state['trials']):
        if trial['id'] != i or trial['model_id'] != spec['model_id'] or trial['kind'] != ('baseline' if i == 0 else 'candidate'):
            raise ValueError('Invalid trial identity.')
        if i and not in_search_grid(spec, trial['params']):
            raise ValueError('Trial values are outside the search space.')
        if not i and trial['params'] != baseline_parameters(problem):
            raise ValueError('Baseline parameters do not match.')
        identity = joblib.hash(trial['params'])
        if identity in seen: raise ValueError('Duplicate trial configuration.')
        seen.add(identity)
        if trial['state'] not in {'running', 'complete', 'failed'} or len(trial['folds']) > spec['folds']:
            raise ValueError('Invalid trial state.')
        if trial['state'] == 'complete' and len(trial['folds']) != spec['folds']:
            raise ValueError('A completed candidate must include every evaluation fold.')
        if trial['state'] == 'running' and len(trial['folds']) == spec['folds']:
            raise ValueError('A fully evaluated trial must be complete or failed.')
        ids = trial['fold_ids']
        if (len(ids) != len(trial['folds']) or ids != sorted(set(ids))
                or any(type(j) is not int or not 0 <= j < spec['folds'] for j in ids)):
            raise ValueError('Invalid indexed fold results.')
        for fold in trial['folds']:
            metrics = {spec['metric'], *spec['secondary']}
            if set(fold) != metrics | {'metric_errors'}:
                raise ValueError('Invalid fold scores.')
            errors = fold['metric_errors']
            if not isinstance(errors, dict) or not set(errors) <= set(spec['secondary']) or not all(
                    isinstance(v, str) and v for v in errors.values()):
                raise ValueError('Invalid secondary metric errors.')
            for metric in metrics:
                value = fold[metric]
                if value is None and metric in errors:
                    continue
                if metric in errors or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid fold scores.')
    batch = state['batch']
    if (not isinstance(batch, list) or batch != sorted(set(batch))
            or any(type(i) is not int or not 0 <= i < len(state['trials']) for i in batch)
            or (batch and batch != list(range(batch[0], len(state['trials']))))
            or any(t['state'] == 'running' and t['id'] not in batch
                   for t in state['trials'])):
        raise ValueError('Invalid search batch.')
    state['status'] = 'paused'
    return state


METHOD_PARAMETER = '__method__'


def branch_state(problem, state, candidate):
    child = problem.branches[candidate]
    selected = [t for t in state['trials'] if t['params'].get(METHOD_PARAMETER) == candidate]
    ids = {t['id']: i for i, t in enumerate(selected)}
    trials = [{**t, 'id': i, 'params': {k: v for k, v in t['params'].items() if k != METHOD_PARAMETER}}
              for i, t in enumerate(selected)]
    return {**state, 'specification': child.specification, 'trials': trials,
            'batch': [ids[i] for i in state['batch'] if i in ids]}


def _validate_checkpoint(problem, checkpoint):
    if not problem.branches:
        return _validate_single_checkpoint(problem, checkpoint)
    state = json.loads(json.dumps(checkpoint, allow_nan=False))
    if state.get('version') != 4 or state.get('specification') != problem.specification:
        raise ValueError('Checkpoint does not match the current candidate methods or search settings.')
    batch = state['batch']
    if (batch != sorted(set(batch)) or (batch and batch != list(range(batch[0], len(state['trials']))))
            or any(type(i) is not int or i < 0 or i >= len(state['trials']) for i in batch)):
        raise ValueError('Invalid search batch.')
    for i, trial in enumerate(state['trials']):
        candidate = trial['params'].get(METHOD_PARAMETER)
        if trial['id'] != i or candidate not in problem.branches:
            raise ValueError('Invalid candidate method identity.')
        if trial.get('method') != problem.branches[candidate].specification['method_label']:
            raise ValueError('Invalid candidate method label.')
        if trial['state'] == 'running' and i not in batch:
            raise ValueError('Invalid search batch.')
    for candidate, child in problem.branches.items():
        _validate_single_checkpoint(child, branch_state(problem, state, candidate))
    state['status'] = 'paused'
    return state


def prepare_candidates(source, **options):
    from dataclasses import replace

    from tuning import candidate_prefix, discover_source
    candidates = source.model_candidates
    committed = source.tuning_spec
    if source._tuning_spec is not None and committed is None:
        raise ValueError('The committed tuning specification is stale. Review and commit it again.')
    choices = options.pop('model_values')
    choices = {} if choices is None else choices
    if not isinstance(choices, dict):
        raise TypeError('Model choices must be an object of candidate-specific parameter paths.')
    known = {r['path']: r for r in discover_source(source)}
    if any(path not in known or not known[path].get('candidate_id') or known[path]['fixed'] for path in choices):
        raise ValueError('Use candidate-specific tunable model parameter paths from Tuning parameters.')
    grids = dict((committed or {}).get('grid', {}))
    for path, values in choices.items():
        if path in grids and grids[path] != values:
            raise ValueError('Change already committed choices in Tuning parameters.')
        grids[path] = values
    branches = {}
    for c in candidates:
        prefix = candidate_prefix(c['id'])
        grid = {path.removeprefix(prefix): values for path, values in grids.items()
                if not path.startswith('candidate_') or path.startswith(prefix)}
        branch_source = replace(source.with_estimator(c['estimator'], label=c['label'], recipe_id=c['id']), _tuning_spec=None)
        if grid:
            from tuning import fingerprint
            branch_source = branch_source.with_tuning_spec({'grid': grid,
                'fingerprint': fingerprint(discover_source(branch_source)),
                'combinations': len(ParameterGrid(grid))})
        child = prepare(branch_source, model_values=None, **options)
        child.specification['method_label'] = c['label']
        branches[c['id']] = child
    first = next(iter(branches.values()))
    if any(child.specification['task'] != first.specification['task'] for child in branches.values()):
        raise ValueError('Candidate methods must match the outcome task.')
    for child in branches.values():
        if len(child.splits) != len(first.splits) or any(not np.array_equal(a, x) or not np.array_equal(b, y)
                for (a, b), (x, y) in zip(child.splits, first.splits)):
            raise ValueError('All methods must use identical evaluation memberships.')
        child.frame = first.frame
        child.splits = first.splits
    spec = copy.deepcopy(first.specification)
    spec.update(model_id='candidate-set', method_label='Multiple methods', grid={},
                combinations=sum(c.specification['combinations'] for c in branches.values()),
                branches={key: child.specification for key, child in branches.items()},
                fixed={}, versions={key: value for child in branches.values() for key, value in child.specification['versions'].items()})
    spec['execution']['workers'] = min(c.specification['execution']['workers'] for c in branches.values())
    spec['execution']['estimated_fit_bytes'] = max(c.specification['execution']['estimated_fit_bytes'] for c in branches.values())
    spec['fingerprint'] = joblib.hash(spec)
    return SearchProblem(None, first.frame, first.target, first.splits, spec, branches)


def candidate_proposal(problem, state):
    # Independent method optimizers with deterministic balanced allocation.
    candidates = sorted(problem.branches, key=lambda key: (
        sum(t['params'].get(METHOD_PARAMETER) == key for t in state['trials']), list(problem.branches).index(key)))
    for candidate in candidates:
        local = branch_state(problem, state, candidate)
        proposed = _proposal(problem.branches[candidate], local)
        state['rng'] = local['rng']
        if proposed is not None:
            return {METHOD_PARAMETER: candidate, **proposed}
    return None


def validate_checkpoint(problem, checkpoint):
    try:
        return _validate_checkpoint(problem, checkpoint)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError('Malformed search checkpoint.') from exc


def mean_score(trial, metric):
    values = [f[metric] for f in trial['folds']]
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def skopt_proposals(problem, state, count=None):
    """Rebuild a GP surrogate from actual completed finite-grid evaluations."""
    from skopt import Optimizer
    from skopt.space import Categorical
    spec = problem.specification
    if spec.get('conditions'):
        # A GP has a fixed-dimensional input space. Fit a surrogate per active
        # branch instead of assigning fictitious values to inactive dimensions.
        branches = spec['parameter_grids']
        reservations = []
        result = []
        for _ in range(count or 1):
            recorded = [*state['trials'], *reservations]
            seen = {joblib.hash(t['params']) for t in recorded}
            available = [(i, g) for i, g in enumerate(branches)
                         if any(joblib.hash(p) not in seen for p in ParameterGrid(g))]
            if not available:
                break
            _, grid = min(available, key=lambda item, recorded=recorded: sum(in_search_grid({'grid': item[1]}, t['params'])
                                                          for t in recorded))
            child = copy.copy(problem)
            child.specification = {**spec, 'grid': grid}
            child.specification.pop('conditions', None)
            child.specification.pop('parameter_grids', None)
            history = {**state, 'trials': [t for t in state['trials'] if in_search_grid({'grid': grid}, t['params'])]}
            proposal = skopt_proposals(child, history)[0]
            if joblib.hash(proposal) in seen:
                proposal = next(p for p in ParameterGrid(grid) if joblib.hash(p) not in seen)
            result.append(proposal)
            reservations.append({'params': proposal})
        return result
    grid = spec['grid']
    paths = [path for path in sorted(grid) if len(grid[path]) > 1]
    if not paths:
        point = {path: values[0] for path, values in grid.items()}
        return [point] * (count or 1)
    optimizer = Optimizer([Categorical(list(range(len(grid[path])))) for path in paths],
        base_estimator='GP', acq_optimizer='sampling', n_jobs=1, model_queue_size=1,
        random_state=(spec['seed'] + len(state['trials'])) % (2**32))
    points, losses = [], []
    for trial in state['trials']:
        if trial['state'] != 'complete':
            continue  # Failures remain excluded from proposals, without inventing scores.
        indices = {}
        for path, values in grid.items():
            matches = [i for i, value in enumerate(values)
                       if type(value) is type(trial['params'][path]) and value == trial['params'][path]]
            if not matches:
                break  # An out-of-grid baseline cannot be surrogate evidence.
            indices[path] = matches[0]
        else:
            points.append([indices[path] for path in paths])
            losses.append(-mean_score(trial, spec['metric']))  # skopt minimizes.
    if points:
        optimizer.tell(points, losses)
    points = [optimizer.ask()] if count is None else optimizer.ask(n_points=count, strategy='cl_min')
    result = []
    for point in points:
        proposed = {path: values[0] for path, values in grid.items()}
        proposed.update({path: grid[path][int(index)] for path, index in zip(paths, point)})
        result.append(proposed)
    return result


def skopt_proposal(problem, state):
    proposals = skopt_proposals(problem, state)
    return proposals[0] if proposals else None


def _proposal(problem, state):
    spec = problem.specification
    seen = {joblib.hash(t['params']) for t in state['trials']}
    grid = ParameterGrid(spec.get('parameter_grids', spec['grid']))
    if grid_coverage(state)['visited'] == spec['combinations']:
        return None
    if spec['method'] == 'grid':
        return next((p for p in grid if joblib.hash(p) not in seen), None)
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng']
    if spec['method'] == 'skopt':
        proposed = skopt_proposal(problem, state)
        if proposed is not None and joblib.hash(proposed) not in seen:
            return proposed
    if spec['method'] == 'tpe':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # A proposal is a seeded function of the recorded history. Recreate the
        # study from actual choices (including duplicate-proposal fallbacks),
        # never executable pickle or a fictitious sequence of past suggestions.
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
            seed=(spec['seed'] + len(state['trials'])) % (2**32), constant_liar=True))
        distributions = {k: optuna.distributions.CategoricalDistribution(tuple(range(len(v)))) for k, v in spec['grid'].items()}
        for old in state['trials']:
            indices = {}
            for path, value in old['params'].items():
                values = spec['grid'][path]
                matches = [i for i, v in enumerate(values) if type(v) is type(value) and v == value]
                if not matches: break
                indices[path] = matches[0]
            else:
                study.add_trial(optuna.trial.create_trial(params=indices, distributions={k: distributions[k] for k in indices},
                    value=mean_score(old, spec['metric']) if old['state'] == 'complete' else None,
                    state={'complete': optuna.trial.TrialState.COMPLETE, 'running': optuna.trial.TrialState.RUNNING,
                           'failed': optuna.trial.TrialState.FAIL}[old['state']]))
        trial = study.ask()
        proposed = {}
        selectors = {t['selector'] for c in spec.get('conditions', {}).values() for t in condition_terms(c)}
        paths = sorted(spec['grid'], key=lambda k: k not in selectors)
        for k in paths:
            if applicable(spec.get('conditions', {}).get(k), proposed):
                values = spec['grid'][k]
                proposed[k] = values[trial.suggest_categorical(k, list(range(len(values))))]
        # A duplicate is evaluated under a new random unseen configuration, while
        # still recording the actual values so restored studies remain truthful.
        if joblib.hash(proposed) not in seen: return proposed
    return random_unseen(problem, state)


def random_unseen(problem, state):
    spec = problem.specification
    seen = {joblib.hash(t['params']) for t in state['trials']}
    grid = ParameterGrid(spec.get('parameter_grids', spec['grid']))
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng']
    for _ in range(100):
        proposed = grid[int(rng.integers(len(grid)))]
        state['rng'] = rng.bit_generator.state
        if joblib.hash(proposed) not in seen: return proposed
    return next((p for p in grid if joblib.hash(p) not in seen), None)


def fit_fold(problem, params, fold):
    train, validation = problem.splits[fold]  # noqa: RUF059
    if problem.branches:
        candidate = params[METHOD_PARAMETER]
        return fit_fold(problem.branches[candidate], {k: v for k, v in params.items() if k != METHOD_PARAMETER}, fold)
    estimator = clone(problem.pipeline).set_params(**params)
    X = problem.frame
    estimator.fit(X.iloc[train].copy(), X[problem.target].iloc[train].copy())
    result = {'metric_errors': {}}
    return estimator, result


def evaluate_fold(problem, params, fold):
    estimator, result = fit_fold(problem, params, fold)
    _, validation = problem.splits[fold]
    X = problem.frame
    for metric in [problem.specification['metric'], *problem.specification['secondary']]:
        try:
            value = float(get_scorer(metric)(estimator, X.iloc[validation].copy(), X[problem.target].iloc[validation].copy()))
            if not math.isfinite(value):
                raise ValueError('Scorer returned a nonfinite value.')
            result[metric] = value
        except Exception as exc:
            if metric == problem.specification['metric']:
                raise
            result[metric] = None
            result['metric_errors'][metric] = f'{type(exc).__name__}: {exc}'
    return result


def branch_state_from_spec(state, candidate):
    return {**state, 'specification': state['specification']['branches'][candidate],
            'trials': [{**t, 'params': {k: v for k, v in t['params'].items() if k != METHOD_PARAMETER}}
                       for t in state['trials'] if t['params'].get(METHOD_PARAMETER) == candidate]}


def grid_coverage(state):
    """Count grid configurations separately from the baseline, including overlap."""
    if state['specification'].get('branches'):
        totals = [grid_coverage(branch_state_from_spec(state, candidate)) for candidate in state['specification']['branches']]
        return {key: sum(t[key] for t in totals) for key in ('complete', 'failed', 'visited', 'total')}
    visited = [t for t in state['trials'] if in_search_grid(state['specification'], t['params'])]
    return {'complete': sum(t['state'] == 'complete' for t in visited),
            'failed': sum(t['state'] == 'failed' for t in visited),
            'visited': sum(t['state'] in {'complete', 'failed'} for t in visited),
            'total': state['specification']['combinations']}


class _FitRunner:
    """Evaluate fits locally; fitted estimators never travel through IPC."""
    def __init__(self, problem):
        self.problem = problem

    def execute(self, job):
        params, fold = job
        tick = time.monotonic()
        try:
            value = evaluate_fold(self.problem, params, fold)
            return value, '', time.monotonic() - tick
        except Exception as exc:  # noqa: BLE001
            return None, f'{type(exc).__name__}: {exc}', time.monotonic() - tick


_PROCESS_RUNNER = None


def _initialize_fit_process(problem):
    global _PROCESS_RUNNER
    _PROCESS_RUNNER = _FitRunner(problem)


def _execute_fit_process(job):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _PROCESS_RUNNER.execute(job)


class _ImmediateResult:
    def __init__(self, value):
        self.value = value

    def ready(self):
        return True

    def get(self):
        return self.value


def _store_fold(trial, fold, result):
    indexed = dict(zip(trial['fold_ids'], trial['folds']))
    indexed[fold] = result
    trial['fold_ids'] = sorted(indexed)
    trial['folds'] = [indexed[i] for i in trial['fold_ids']]


def _new_trial(problem, state, params):
    trial_id = len(state['trials'])
    branch = problem.branches.get(params.get(METHOD_PARAMETER)) if problem.branches else None
    identity = branch.specification['model_id'] if branch else problem.specification['model_id']
    baseline = not any(t['model_id'] == identity for t in state['trials']) if branch else trial_id == 0
    trial = {'id': trial_id, 'model_id': identity,
             'method': (branch.specification.get('method_label', identity) if branch else
                        problem.specification.get('method_label', identity)),
             'kind': 'baseline' if baseline else 'candidate', 'params': params,
             'state': 'running', 'fold_ids': [], 'folds': [], 'elapsed': 0., 'error': ''}
    state['trials'].append(trial)
    return trial_id


def run(problem, checkpoint, *, trials=None, seconds=0, publish, pause, cancel=lambda: False, worker_registry=None):
    """Coordinate deterministic batches, indexed fold results and bounded fits."""
    if (trials is not None and (isinstance(trials, bool) or not isinstance(trials, int) or trials < 0)
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Supply a nonnegative candidate budget (zero means unlimited) and a finite nonnegative time budget.')
    if trials == 0:
        trials = None
    state = validate_checkpoint(problem, checkpoint)
    spec = problem.specification
    execution = spec['execution']
    workers, mode = execution['workers'], execution['mode']
    start, prior = time.monotonic(), state['elapsed']
    pool = None
    local = _FitRunner(problem) if workers == 1 else None
    inflight = {}
    stopping = None

    def emit(status):
        state.update(status=status, elapsed=prior + time.monotonic() - start)
        publish(copy.deepcopy(state))

    def submit(trial, fold):
        job = (trial['params'], fold)
        result = _ImmediateResult(local.execute(job)) if local is not None else pool.apply_async(_execute_fit_process, (job,))
        inflight[(trial['id'], fold)] = result

    def allocate_batch():
        if problem.branches:
            existing = {t['params'][METHOD_PARAMETER] for t in state['trials']}
            for candidate, child in problem.branches.items():
                if candidate not in existing:
                    params = baseline_parameters(child)
                    return [_new_trial(problem, state, {METHOD_PARAMETER: candidate, **params})]
            width = workers if mode in {'trials', 'auto'} else 1
            if trials is not None:
                width = min(width, max(0, trials - sum(t['kind'] == 'candidate' for t in state['trials'])))
            batch = []
            for _ in range(width):
                params = candidate_proposal(problem, state)
                if params is None:
                    break
                batch.append(_new_trial(problem, state, params))
            return batch
        if not state['trials']:
            return [_new_trial(problem, state, baseline_parameters(problem))]
        limit = workers if mode in {'trials', 'auto'} else 1
        if trials is not None:
            limit = min(limit, max(0, trials - (len(state['trials']) - 1)))
        limit = min(limit, spec['combinations'] - grid_coverage(state)['visited'])
        proposed = skopt_proposals(problem, state, limit) if spec['method'] == 'skopt' and limit else []
        batch = []
        for j in range(limit):
            if proposed:
                params = proposed[j]
                if joblib.hash(params) in {joblib.hash(t['params']) for t in state['trials']}:
                    params = random_unseen(problem, state)
            else:
                params = _proposal(problem, state)
            if params is None:
                break
            batch.append(_new_trial(problem, state, params))
        return batch

    emit('running')
    try:
        if workers > 1:
            pool = mp.get_context('spawn').Pool(workers, initializer=_initialize_fit_process, initargs=(problem,))
            pool_processes = list(pool._pool)
            if worker_registry is not None:
                import psutil
                for i, process in enumerate(pool_processes):
                    worker_registry[2*i] = process.pid
                    worker_registry[2*i+1] = psutil.Process(process.pid).create_time()
        while True:
            if cancel():
                stopping = 'paused'
                break  # Terminate outstanding fits; only published results survive.
            if pause():
                stopping = 'paused'
            elif seconds and prior + time.monotonic() - start >= seconds:
                stopping = stopping or 'budget reached'

            if pool is not None and any(process.exitcode is not None for process in pool_processes):
                raise RuntimeError('A fit worker exited unexpectedly; resume to retry missing folds.')

            # Completion order may differ from fold order; checkpoints always use IDs.
            for key, future in list(inflight.items()):
                if not future.ready():
                    continue
                trial_id, fold = key
                value, error, elapsed = future.get()
                del inflight[key]
                trial = state['trials'][trial_id]
                trial['elapsed'] += elapsed
                if error:
                    trial.update(state='failed', error=error)
                else:
                    _store_fold(trial, fold, value)
                    if trial['state'] != 'failed' and len(trial['folds']) == spec['folds']:
                        trial['state'] = 'complete'
                emit('running')

            # A progress subscriber can request pause as soon as a fold is published.
            if cancel():
                stopping = 'paused'
                break
            if pause():
                stopping = 'paused'
            elif seconds and prior + time.monotonic() - start >= seconds:
                stopping = stopping or 'budget reached'
            if stopping:
                if not inflight:
                    break
                time.sleep(.02)
                continue

            if not state['batch']:
                state['batch'] = allocate_batch()
                if not state['batch']:
                    stopping = 'exhausted' if grid_coverage(state)['visited'] == spec['combinations'] else 'budget reached'
                    break
                emit('running')

            batch = [state['trials'][i] for i in state['batch']]
            scores_done = all(t['state'] != 'running' for t in batch)
            if scores_done and not inflight:
                state['batch'] = []
                continue

            if not scores_done:
                parallel_folds = mode == 'folds' or (mode == 'auto' and len(batch) == 1)
                for trial in batch:
                    if trial['state'] != 'running':
                        continue
                    trial_jobs = sum(k[0] == trial['id'] for k in inflight)
                    for fold in range(spec['folds']):
                        if len(inflight) >= workers or (trial_jobs and not parallel_folds):
                            break
                        key = (trial['id'], fold)
                        if fold not in trial['fold_ids'] and key not in inflight:
                            submit(trial, fold)
                            trial_jobs += 1
            if inflight:
                time.sleep(.02)
        emit(stopping or 'paused')
    except Exception as exc:  # noqa: BLE001
        state['error'] = f'{type(exc).__name__}: {exc}'
        emit('error')
    finally:
        if pool is not None:
            if inflight or state['status'] == 'error':
                pool.terminate()
            else:
                pool.close()
            pool.join()
    return state


def _worker(problem, state, trials, seconds, updates, pause_event, cancel_event, worker_registry):
    if os.name == 'posix':
        os.setsid()  # Isolate this coordinator and its children for bounded cancellation.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(problem, state, trials=trials, seconds=seconds, publish=updates.put,
            pause=pause_event.is_set, cancel=cancel_event.is_set, worker_registry=worker_registry)


class SearchWorker:
    """Non-daemon coordinator owns a bounded pool; cancellation cleans its tree."""
    def __init__(self):
        self.process = None
        self.updates = None
        self.pause_event = None
        self.cancel_event = None
        self.worker_registry = None
        self.started = None
        self.prior_elapsed = 0.

    @property
    def running(self):
        return self.process is not None and self.process.is_alive()

    def start(self, problem, state, trials, seconds):
        if self.running:
            raise ValueError('Pause the current search before starting another.')
        self.close()
        ctx = mp.get_context('spawn')
        self.updates, self.pause_event, self.cancel_event = ctx.Queue(), ctx.Event(), ctx.Event()
        self.worker_registry = ctx.Array('d', 128)
        self.process = ctx.Process(target=_worker,
            args=(problem, state, trials, seconds, self.updates, self.pause_event, self.cancel_event, self.worker_registry))
        self.prior_elapsed, self.started = state['elapsed'], time.monotonic()
        self.process.start()

    @property
    def elapsed(self):
        return self.prior_elapsed + (time.monotonic() - self.started if self.started is not None else 0.)

    def poll(self):
        latest = None
        if self.updates is not None:
            while True:
                try:
                    latest = self.updates.get_nowait()
                except queue.Empty:
                    break
        return latest

    def pause(self):
        if self.pause_event is not None:
            self.pause_event.set()

    def close(self):
        if self.process is not None:
            if self.running:
                self.cancel_event.set()
                self.process.join(timeout=.5)
            import psutil
            owned = []
            if self.worker_registry is not None:
                for i in range(0, len(self.worker_registry), 2):
                    pid, created = self.worker_registry[i:i+2]
                    if not pid or not created:
                        continue
                    try:
                        child = psutil.Process(int(pid))
                        if child.create_time() == created:
                            owned.append(child)
                    except psutil.NoSuchProcess:
                        pass
            group = self.process.pid
            def matches_group(pid):
                try:
                    return os.getpgid(pid) == group
                except ProcessLookupError:
                    return False
            group_owned = os.name == 'posix' and (
                (self.process.is_alive() and matches_group(group)) or
                any(matches_group(child.pid) for child in owned if child.is_running()))
            if group_owned:
                try:
                    os.killpg(group, signal.SIGTERM)
                    self.process.join(timeout=2)
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                for child in owned:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                _, alive = psutil.wait_procs(owned, timeout=1)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.process.close()
            self.process = None
        if self.updates is not None:
            self.updates.close()
            self.updates = None


def accepted_parameters(problem, checkpoint, trial_id):
    state = validate_checkpoint(problem, checkpoint)
    selected = next((t for t in state['trials'] if t['id'] == trial_id and t['state'] == 'complete'), None)
    if selected is None: raise ValueError('Select a completed candidate evaluated on every fold.')
    if problem.branches:
        candidate = selected['params'][METHOD_PARAMETER]
        return {**problem.branches[candidate].specification['fixed'],
                **{k: v for k, v in selected['params'].items() if k != METHOD_PARAMETER}}
    return {**problem.specification['fixed'], **selected['params']}
