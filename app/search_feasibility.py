"""Score-free, fold-search feasibility evidence and proposal weighting."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


def failure_kind(error):
    """Conservatively distinguish estimator rejection from execution/schema faults."""
    if any(token in str(error).casefold() for token in (
            'out of memory', 'bad_alloc', 'cannot allocate memory', 'resource temporarily unavailable')):
        return 'execution'
    if isinstance(error, (MemoryError, OSError, TimeoutError, KeyError, AttributeError, ImportError)):
        return 'execution'
    if isinstance(error, (ValueError, TypeError, ArithmeticError)) or type(error).__name__ in {
            'XGBoostError', 'LightGBMError', 'CatBoostError'}:
        return 'fit'
    return 'unknown'


def feasibility_model(spec, trials):
    """Return success probabilities, or None until both outcomes are observed.

    Old failures without a classification are deliberately not inferred from text.
    Categorical tokens avoid inventing distances between finite grid choices.
    """
    from pipeline_search import in_search_grid

    evidence = [t for t in trials if (t['state'] == 'complete' or
                (t['state'] == 'failed' and t.get('failure_kind') == 'fit'))
                and in_search_grid(spec, t['params'])]
    labels = [int(t['state'] == 'complete') for t in evidence]
    if len(evidence) < 4 or len(set(labels)) < 2:
        return None
    paths = sorted(spec['grid'])
    def features(params):
        return {path: joblib.hash(params[path]) if path in params else '__inactive__' for path in paths}
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform([features(t['params']) for t in evidence])
    forest = RandomForestClassifier(n_estimators=48, max_depth=5, n_jobs=1, random_state=spec['seed'])
    forest.fit(matrix, labels)
    def probability(params):
        return float(forest.predict_proba(vectorizer.transform([features(params)]))[0, 1])
    return probability


def weighted_proposal(problem, state, propose, probability):
    """Rejection-weight optimizer proposals, with an exploration floor.

    Scores never enter this model. Each retry uses the same real history and a
    derived seed. Bounded retries keep even all-infeasible searches progressing.
    """
    import copy
    # A separate stream avoids correlating acceptance with the optimizer's own
    # seeded draw (or its random unseen fallback).
    rng = np.random.default_rng(int(joblib.hash(('feasibility', state['rng']))[:16], 16))
    best, best_probability = None, -1.
    for attempt in range(8):
        local = copy.copy(problem)
        local.specification = dict(problem.specification,
            seed=(problem.specification['seed'] + attempt * 104729) % (2**32))
        candidate = propose(local, state)
        # propose() may use the checkpoint RNG for unseen-point fallbacks.
        progress_rng = np.random.default_rng()
        progress_rng.bit_generator.state = state['rng']
        progress_rng.random()
        state['rng'] = progress_rng.bit_generator.state
        if candidate is None:
            return best
        chance = .1 + .9 * probability(candidate)
        if chance > best_probability:
            best, best_probability = candidate, chance
        accepted = rng.random() < chance
        if accepted:
            return candidate
    return best
