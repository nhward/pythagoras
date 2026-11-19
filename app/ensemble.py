"""Fold-fitted ensembles of complete reviewed pipelines; no final fitting."""
import threading
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from code_recording import recordable
from pipeline_search import METRICS, displayed, prepare
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import get_scorer
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

    def __deepcopy__(self, memo):
        return self

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


@recordable
def evaluate_ensemble(source, selected, style, cancelled=None):
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
    fitted, scores = [], []
    metrics = METRICS[task]
    for fold, (train, validation) in enumerate(splits):
        models = []
        for key, pipeline in members:
            if cancelled.is_set():
                raise InterruptedError('Ensemble evaluation superseded or cancelled.')
            model = clone(pipeline).fit(frame.iloc[train].copy(), frame[target].iloc[train].copy())
            models.append((key, model))
        ensemble = clone(definition)
        ensemble.estimators_ = tuple(models)
        if task == 'classification':
            ensemble.classes_ = np.unique(frame[target].iloc[train])
        for key, model in (*models, ('ensemble', ensemble)):
            row = {'model': key, 'fold': fold, 'errors': {}}
            for metric in metrics:
                try:
                    value = float(get_scorer(metric)(model, frame.iloc[validation].copy(), frame[target].iloc[validation].copy()))
                    if not np.isfinite(value):
                        raise ValueError('Scorer returned a nonfinite value.')
                    row[metric] = value
                except (ValueError, AttributeError) as exc:
                    row[metric] = None
                    row['errors'][metric] = str(exc)
            scores.append(row)
        fitted.append(ensemble)
    if cancelled.is_set():
        raise InterruptedError('Ensemble evaluation superseded or cancelled.')
    return EnsembleEvaluation(definition, tuple(fitted),
        tuple((tuple(a.tolist()), tuple(b.tolist())) for a, b in splits), tuple(scores),
        tuple((key, winners[key]['label']) for key in selected), task, problem.specification['evaluation'],
        evaluation_signature(source, selected, style), results['state']['specification']['fingerprint'],
        tuple((key, winners[key]['trial']) for key in selected), results['review'])
