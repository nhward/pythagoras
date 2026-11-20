"""Collect held-out model responses once, then preserve sklearn scorer semantics."""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.utils.metaestimators import available_if


@dataclass
class Responses:
    values: dict
    classes: object = None
    seconds: float = 0.
    errors: dict = field(default_factory=dict)


def collect_responses(model, frame, task):
    values, errors = {}, {}
    batch = frame.copy()
    started = time.perf_counter()
    try:
        values['predict'] = np.asarray(model.predict(batch))
    except Exception as error:  # noqa: BLE001
        errors['predict'] = f'{type(error).__name__}: {error}'
    seconds = time.perf_counter() - started
    classes = np.asarray(model.classes_) if task == 'classification' else None
    if task == 'classification':
        for name in ('predict_proba', 'decision_function'):
            if callable(getattr(model, name, None)):
                try:
                    values[name] = np.asarray(getattr(model, name)(frame.copy()))
                except Exception as error:  # Provider responses can fail independently.  # noqa: BLE001
                    errors[name] = f'{type(error).__name__}: {error}'
    return Responses(values, classes, seconds, errors)


class _CachedResponses(BaseEstimator):
    def __init__(self, responses):
        self.responses = responses
        self.fitted_ = True

    def predict(self, X):
        if 'predict' not in self.responses.values:
            raise ValueError(self.responses.errors.get('predict', 'Prediction response unavailable.'))
        result = self.responses.values['predict']
        if len(result) != len(X):
            raise ValueError('Cached responses do not match the validation rows.')
        return result


class _RegressorResponses(RegressorMixin, _CachedResponses):
    pass


class _ClassifierResponses(ClassifierMixin, _CachedResponses):
    def __init__(self, responses):
        super().__init__(responses)
        self.classes_ = responses.classes

    @available_if(lambda self: 'predict_proba' in self.responses.values)
    def predict_proba(self, X):
        result = self.responses.values['predict_proba']
        if len(result) != len(X):
            raise ValueError('Cached probabilities do not match the validation rows.')
        return result

    @available_if(lambda self: 'decision_function' in self.responses.values)
    def decision_function(self, X):
        result = self.responses.values['decision_function']
        if len(result) != len(X):
            raise ValueError('Cached decisions do not match the validation rows.')
        return result


def response_estimator(responses, task):
    return (_ClassifierResponses if task == 'classification' else _RegressorResponses)(responses)


def score_responses(responses, frame, observed, task, metrics):
    proxy = response_estimator(responses, task)
    scores, errors = {}, {}
    for metric in metrics:
        try:
            value = float(get_scorer(metric)(proxy, frame, observed))
            if not np.isfinite(value):
                raise ValueError('Scorer returned a nonfinite value.')
            scores[metric] = value
        except Exception as error:  # noqa: BLE001
            scores[metric] = None
            errors[metric] = f'{type(error).__name__}: {error}'
    return scores, errors


def combine_responses(members, style, task, classes=None):
    if task == 'regression' or style == 'hard':
        for response in members:
            if 'predict' not in response.values:
                raise ValueError(response.errors.get('predict', 'A member prediction is unavailable.'))
    if task == 'regression':
        values = np.asarray([r.values['predict'] for r in members])
        return Responses({'predict': (np.mean if style == 'mean' else np.median)(values, axis=0)})
    if style == 'soft':
        if any('predict_proba' not in response.values for response in members):
            raise ValueError('Soft voting requires valid probabilities from every member.')
        probabilities = np.zeros((len(members[0].values['predict_proba']), len(classes)))
        for response in members:
            for i, label in enumerate(response.classes):
                probabilities[:, np.flatnonzero(classes == label)[0]] += response.values['predict_proba'][:, i]
        probabilities /= len(members)
        return Responses({'predict': classes[np.argmax(probabilities, axis=1)], 'predict_proba': probabilities}, classes)
    labels = np.asarray([r.values['predict'] for r in members])
    counts = np.asarray([(labels == label).sum(axis=0) for label in classes])
    return Responses({'predict': classes[np.argmax(counts, axis=0)]}, classes)
