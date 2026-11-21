"""Fold-fitted ensembles of complete reviewed pipelines; no final fitting."""
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import nullcontext
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from code_recording import recordable
from fit_responses import collect_responses, combine_responses, score_responses
from fit_service import fit_key
from pipeline_search import METRICS, displayed, prepare
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


class _Members(BaseEstimator):
    def __init__(self, members, style):
        self.members = members
        self.style = style

    def fit(self, X, y):
        self._validate()
        self.estimators_ = tuple((name, clone(model).fit(X.copy(), y.copy())) for name, model in self.members)
        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y)
        return self

    def _validate(self):
        if not self.members:
            raise ValueError('Select at least one model.')
        if self.style not in self.styles:
            raise ValueError('Choose an ensemble style compatible with the task.')
        if self.style == 'soft' and any(not callable(getattr(model, 'predict_proba', None)) for _, model in self.members):
            raise ValueError('Soft voting requires probabilities from every member; choose hard voting instead.')


@recordable
class RegressionEnsemble(RegressorMixin, _Members):
    styles = ('mean', 'median')

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        predictions = np.asarray([model.predict(X.copy()) for _, model in self.estimators_])
        return (np.mean if self.style == 'mean' else np.median)(predictions, axis=0)


@recordable
class ClassificationEnsemble(ClassifierMixin, _Members):
    styles = ('soft', 'hard')

    @available_if(lambda self: self.style == 'soft')
    def predict_proba(self, X):
        check_is_fitted(self, 'estimators_')
        result = np.zeros((len(X), len(self.classes_)))
        for _, model in self.estimators_:
            probabilities = model.predict_proba(X.copy())
            for i, label in enumerate(model.classes_):
                result[:, np.flatnonzero(self.classes_ == label)[0]] += probabilities[:, i]
        return result / len(self.estimators_)

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        if self.style == 'soft':
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        predictions = np.asarray([model.predict(X.copy()) for _, model in self.estimators_])
        counts = np.asarray([(predictions == label).sum(axis=0) for label in self.classes_])
        return self.classes_[np.argmax(counts, axis=0)]


def representatives(source):
    results = source.tuning_results
    if results is None or results.get('read_only'):
        raise ValueError('Complete and review compatible model training upstream.')
    if results['state']['specification']['task'] not in METRICS:
        raise ValueError('Ensembles currently support regression and classification.')
    winners = results.get('review', {}).get('representatives', {})
    if not winners:
        raise ValueError('Choose weighted method winners in Candidate models upstream.')
    trials = {t['id']: t for t in results['state']['trials'] if t['state'] == 'complete'}
    members = {}
    for method, trial_id in winners.items():
        trial = trials.get(trial_id)
        if trial is None or trial['model_id'] != method or str(trial_id) not in results['pipelines']:
            raise ValueError(f'Winning model for {method} is unavailable; review models upstream.')
        members[method] = {'trial': trial_id, 'label': trial.get('method', method),
                           'pipeline': results['pipelines'][str(trial_id)]}
    return results, members




def evaluation_context_signature(source):
    """Data and resampling identity shared by successive winner snapshots."""
    results, _ = representatives(source)
    spec = results['state']['specification']
    return joblib.hash((source.clean_frame, source.role_map.to_primitive(), source.modelling_context,
                       source.evaluation_plan, spec['task'], spec['seed']))


def evaluation_signature(source, selected, style):
    """Only selected fitting inputs, independent of unrelated search progress."""
    results, winners = representatives(source)
    if len(set(selected)) < 2 or any(key not in winners for key in selected):
        raise ValueError('Waiting for at least two selected, available method winners.')
    spec = results['state']['specification']
    return joblib.hash((source.clean_frame, source.role_map.to_primitive(), source.modelling_context,
        source.evaluation_plan, spec['task'], spec['seed'], style,
        tuple((key, winners[key]['trial'], winners[key]['pipeline']) for key in selected)))


def source_signature(source):
    return joblib.hash((source.clean_frame, source.role_map.to_primitive(), source.modelling_context,
                       source.evaluation_plan, source.pipeline,
                       (source._tuning_results or {}).get('results', {}).get('review')))


@dataclass(frozen=True)
class EnsembleEvaluation:
    """Runtime evidence. Fitted models are per split, never a final fitted model.

    Treat models as read-only. Cloning proxy data shares this potentially large
    immutable result instead of copying every fitted member.
    """
    definition: object
    fitted: tuple
    splits: tuple
    scores: tuple
    members: tuple
    task: str
    evaluation: dict
    input_signature: str
    search_fingerprint: str
    trial_ids: tuple
    review: dict
    cache_owner: str | None = None
    cache_keys: tuple = ()
    diagnostics: dict = field(default_factory=dict)

    def __deepcopy__(self, memo):
        return self

    @property
    def split_count(self):
        return len(self.splits)

    def table(self, metric):
        rows = []
        for key, label in (*self.members, ('ensemble', 'Ensemble')):
            values = [row[metric] for row in self.scores if row['model'] == key]
            valid = [v for v in values if v is not None]
            complete = len(valid) == len(self.splits)
            score = float(np.mean(valid)) if complete else np.nan
            errors = sorted({row.get('errors', {}).get(metric, '') for row in self.scores if row['model'] == key} - {''})
            rows.append({'Model': label, 'Key': key, 'Score': displayed(score, metric),
                         'SD': float(np.std(valid)) if complete else np.nan,
                         'Splits': len(valid), 'Rank score': score, 'Note': '; '.join(errors)})
        return pd.DataFrame(rows).sort_values('Rank score', ascending=False, kind='stable', na_position='last')


    def metric_table(self, preferred):
        """All feasible metric means, ordered by the preferred metric."""
        table = self.table(preferred).set_index('Key')
        label = METRICS[self.task][preferred].split(' · ')[0]
        summary = table[['Model']].copy()
        notes = {key: [] for key in table.index}
        for metric, description in METRICS[self.task].items():
            values = self.table(metric).set_index('Key').reindex(table.index)
            name = description.split(' · ')[0]
            if values.Score.notna().any():
                summary[name] = values.Score
            for key, note in values.Note.items():
                if note:
                    notes[key].append(f'{name}: {note}')
        summary[f'{label} SD'] = table.SD
        summary[f'{label} splits'] = table.Splits
        summary['Note'] = ['; '.join(notes[key]) for key in summary.index]
        return summary.reset_index(drop=True)


@dataclass(frozen=True)
class EnsembleDisplay:
    """Portable aggregate scores; never sufficient for fitted-model export."""
    task: str
    members: tuple
    split_count: int
    evaluation: dict
    input_signature: str
    tables: dict

    def table(self, metric):
        return pd.DataFrame(self.tables[metric]).replace({None: np.nan}).sort_values(
            'Rank score', ascending=False, kind='stable', na_position='last')

    def metric_table(self, preferred):
        return EnsembleEvaluation.metric_table(self, preferred)


def portable_ensemble_display(result):
    """Retain only the aggregate statistics used by the chart and table."""
    return {'version': 1, 'task': result.task, 'members': [list(member) for member in result.members],
            'split_count': result.split_count, 'origin': result.evaluation['origin'],
            'input_signature': result.input_signature,
            'tables': {metric: result.table(metric).astype(object).where(
                pd.notna(result.table(metric)), None).to_dict('records') for metric in METRICS[result.task]}}


def restore_ensemble_display(saved):
    if saved is None:
        return None
    try:
        if saved['version'] != 1 or saved['task'] not in METRICS or saved['split_count'] < 1:
            raise ValueError('Invalid saved ensemble summary.')
        result = EnsembleDisplay(saved['task'], tuple(tuple(member) for member in saved['members']),
            int(saved['split_count']), {'origin': saved['origin']}, saved['input_signature'], saved['tables'])
        keys = {key for key, _ in result.members} | {'ensemble'}
        for metric in METRICS[result.task]:
            table = result.table(metric)
            if set(table.Key) != keys or not {'Model', 'Score', 'SD', 'Splits', 'Note'}.issubset(table.columns):
                raise ValueError('Invalid saved ensemble summary.')
        return result
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError('Invalid saved ensemble summary.') from exc


@recordable
def evaluate_ensemble(source, selected, style, cancelled=None, *, fit_service=None):
    results, winners = representatives(source)
    if len(set(selected)) < 2 or any(key not in winners for key in selected):
        raise ValueError('Waiting for at least two selected, available method winners.')
    task = results['state']['specification']['task']
    factory = RegressionEnsemble if task == 'regression' else ClassificationEnsemble
    members = tuple((key, winners[key]['pipeline']) for key in selected)
    definition = factory(members, style)
    definition._validate()
    cancelled = cancelled or threading.Event()
    # prepare() owns the same plan validation and implicit-CV defaults as cand_train.
    first = source.with_selected_model(winners[selected[0]]['trial'])
    problem = prepare(first, seed=results['state']['specification']['seed'])
    frame, target, splits = problem.frame, problem.target, problem.splits
    runtime = getattr(fit_service, 'runtime', None)
    memory = getattr(fit_service, 'entries', None)
    owner = 'ensemble:' + uuid.uuid4().hex
    jobs, resolved, futures = [], {}, {}
    diagnostics = {'reused_folds': 0, 'new_fits': 0, 'serialization_fallbacks': [], 'seconds': 0.}
    started = time.monotonic()
    metrics = METRICS[task]
    for fold, (train, validation) in enumerate(splits):
        for key, pipeline in members:
            cache_key = fit_key(problem.fit_context, pipeline, train, validation)
            cached = runtime.get(cache_key, owner) if runtime else memory.get(cache_key) if memory is not None else None
            if cached is not None:
                resolved[fold, key] = cached
                diagnostics['reused_folds'] += 1
            else:
                jobs.append((fold, key, pipeline, cache_key))

    def fit_local(job):
        fold, key, pipeline, cache_key = job
        train, validation = splits[fold]
        if cancelled.is_set():
            raise InterruptedError('Ensemble evaluation superseded or cancelled.')
        # Local fallback also obeys the shared CPU/memory lease.
        lease = runtime.lease('ensemble', estimate(pipeline), cancelled) if runtime else nullcontext()
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(1) if runtime else nullcontext()
        with lease, limits:
            if runtime:
                cached = runtime.get(cache_key, owner)
                if cached is not None:
                    resolved[fold, key] = cached
                    diagnostics['reused_folds'] += 1
                    return
            model = clone(pipeline).fit(frame.iloc[train].copy(), frame[target].iloc[train].copy())
            response = collect_responses(model, frame.iloc[validation], task)
            if runtime:
                runtime.count('ensemble_fits')
                runtime.count('prediction_batches')
                error = None if joblib.hash(pipeline) in fit_service.local_only else runtime.put(cache_key, model, response, owner)
                if error:
                    diagnostics['serialization_fallbacks'].append(error)
                runtime.remember(cache_key, model, response)
            elif memory is not None:
                memory[cache_key] = model, response
            resolved[fold, key] = model, response
            diagnostics['new_fits'] += 1

    def estimate(pipeline):
        from pipeline_search import execution_settings
        return execution_settings(pipeline, frame, splits, 'auto', 0)['estimated_fit_bytes']

    try:
        if runtime is None:
            for job in jobs:
                fit_local(job)
        elif jobs:
            context_path = runtime.put_context(problem.fit_context, frame, target, owner)
            pending = iter(jobs)
            exhausted = False
            while futures or not exhausted:
                if cancelled.is_set():
                    raise InterruptedError('Ensemble evaluation superseded or cancelled.')
                width = runtime.ensemble_width()
                while not exhausted and len(futures) < width:
                    job = next(pending, None)
                    if job is None:
                        exhausted = True
                        break
                    fold, key, pipeline, cache_key = job
                    if joblib.hash(pipeline) in fit_service.local_only:
                        fit_local(job)
                        continue
                    train, validation = splits[fold]
                    future = fit_service.submit(context_path, cache_key, pipeline, train, validation, task, estimate(pipeline), owner)
                    futures[future] = job
                if not futures:
                    continue
                done, _ = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        cache_key, error, fitted_now = future.result()
                    except Exception as error:
                        if 'pickle' not in str(error).lower():
                            raise
                        diagnostics['serialization_fallbacks'].append(str(error))
                        fit_service.local_only.add(joblib.hash(job[2]))
                        fit_local(job)
                        continue
                    diagnostics['new_fits'] += int(fitted_now)
                    diagnostics['reused_folds'] += int(not fitted_now)
                    if error:
                        diagnostics['serialization_fallbacks'].append(error)
                        fit_service.local_only.add(joblib.hash(job[2]))
                        fit_local(job)
                        continue
                    cached = runtime.get(cache_key, owner)
                    if cached is None:
                        raise ValueError('A newly fitted ensemble artifact could not be loaded.')
                    resolved[job[0], job[1]] = cached
        fitted, scores = [], []
        for fold, (_, validation) in enumerate(splits):
            models = [(key, resolved[fold, key][0]) for key, _ in members]
            responses = [(key, resolved[fold, key][1]) for key, _ in members]
            ensemble = clone(definition)
            ensemble.estimators_ = tuple(models)
            classes = np.unique(frame[target].iloc[splits[fold][0]]) if task == 'classification' else None
            if task == 'classification':
                ensemble.classes_ = classes
            combined = combine_responses([r for _, r in responses], style, task, classes)
            for key, response in (*responses, ('ensemble', combined)):
                values, errors = score_responses(response, frame.iloc[validation], frame[target].iloc[validation], task, metrics)
                scores.append({'model': key, 'fold': fold, 'errors': errors, **values})
            fitted.append(ensemble)
        if cancelled.is_set():
            raise InterruptedError('Ensemble evaluation superseded or cancelled.')
        if runtime and not diagnostics['serialization_fallbacks'] and any(joblib.hash(pipeline) in fit_service.local_only for _, pipeline in members):
            diagnostics['serialization_fallbacks'].append('A fitted provider is reusable only while its session model remains in memory.')
        diagnostics['seconds'] = time.monotonic() - started
        keys = tuple(fit_key(problem.fit_context, pipeline, train, validation)
                     for train, validation in splits for _, pipeline in members)
        if memory is not None:
            # Browser storage stays bounded to the selected fitted members.
            for key in list(memory):
                if key not in keys:
                    del memory[key]
        return EnsembleEvaluation(definition, tuple(fitted),
            tuple((tuple(a.tolist()), tuple(b.tolist())) for a, b in splits), tuple(scores),
            tuple((key, winners[key]['label']) for key in selected), task, problem.specification['evaluation'],
            evaluation_signature(source, selected, style), results['state']['specification']['fingerprint'],
            tuple((key, winners[key]['trial']) for key in selected), results['review'],
            owner if runtime else None, keys, diagnostics)
    except BaseException:
        for future in futures:
            future.cancel()
        # Running native jobs finish their lease, then release this abandoned
        # snapshot. Cancellation must not leave their late artifacts pinned.
        if runtime:
            runtime.release(owner)
            for future in futures:
                def release_when_done(_future, runtime=runtime, owner=owner):
                    try:
                        runtime.release(owner)
                    except Exception:  # noqa: BLE001, S110
                        pass  # The session may already have removed its cache.
                future.add_done_callback(release_when_done)
        raise
    finally:
        if runtime:
            # Queued workers may not have read the shared data yet. Keep it until
            # all outstanding jobs finish, including after card cancellation.
            outstanding = tuple(futures)
            def release_context(_future=None):
                if all(f.done() for f in outstanding):
                    try:
                        runtime.release_context(owner)
                    except Exception:  # noqa: BLE001, S110
                        pass  # Session destruction already removed the directory.
            for future in outstanding:
                future.add_done_callback(release_context)
            release_context()
