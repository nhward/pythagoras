"""Exploratory paired Bayesian comparisons following sklearn's CV statistics example.

No refitting, p-values, multiplicity adjustment, or automatic trial acceptance.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t

SOURCE = 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#pairwise-comparison-of-all-models-bayesian-approach'
COLUMNS = ['Trial A', 'Trial B', 'Mean advantage A', '95% lower', '95% upper',
           'P(A better)', 'P(equivalent)', 'P(A worse)', 'Repeats', 'Paired splits',
           'Repeat advantage SD', 'Assessment']


def paired_distribution(differences, ratios):
    """Corrected Student-t posterior for a mean paired score difference."""
    differences = np.asarray(differences, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if (differences.ndim != 1 or len(differences) < 2 or ratios.shape != differences.shape
            or not np.isfinite(differences).all() or not np.isfinite(ratios).all()
            or (ratios <= 0).any()):
        raise ValueError('Paired finite scores and positive split-size ratios are required.')
    mean = float(differences.mean())
    scale = float(np.sqrt((1 / len(differences) + ratios.mean()) * differences.var(ddof=1)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError('Zero or invalid paired-score variability: a nondegenerate posterior cannot be estimated.')
    return t(df=len(differences) - 1, loc=mean, scale=scale)


def paired_posterior(differences, ratios, tolerance, probability=0.95):
    """Positive differences mean A is better, including negative sklearn losses.

    Use the mean validation/training size ratio for near-balanced folds. The
    variance correction is retained even though there is no Bonferroni correction.
    """
    posterior = paired_distribution(differences, ratios)
    if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Set a positive practical-equivalence tolerance in primary-metric units.')
    if not np.isfinite(probability) or not 0.5 < probability < 1:
        raise ValueError('Decision probability must be greater than 0.5 and less than 1.')
    mean = float(np.mean(differences))
    worse = float(posterior.cdf(-tolerance))
    better = float(posterior.sf(tolerance))
    equivalent = float(np.clip(1 - worse - better, 0, 1))
    low, high = posterior.interval(0.95)
    assessment = ('A better' if better >= probability else 'A worse' if worse >= probability
                  else 'Practically equivalent' if equivalent >= probability else 'Inconclusive')
    return {'Mean advantage A': mean, '95% lower': float(low), '95% upper': float(high),
            'P(A better)': better, 'P(equivalent)': equivalent, 'P(A worse)': worse,
            'Assessment': assessment}


def comparison_data(state):
    """Validated paired split scores, shared by the table and chart."""
    if state is None:
        return [], [], 'Complete at least two trials to compare them.'
    spec = state['specification']
    evaluation = spec.get('evaluation', {})
    recipe = evaluation.get('plan', {}).get('recipe', {})
    if (evaluation.get('origin') != 'upstream' or evaluation.get('strategy') not in {'kfold', 'stratified'}
            or evaluation.get('repeats', 1) < 2 or not recipe.get('shuffle', False)):
        return [], [], 'Bayesian comparisons require upstream repeated shuffled ordinary or stratified K-fold CV.'
    sizes = spec.get('split_sizes', [])
    n = spec['folds']
    if len(sizes) != n or any(a <= 0 or b <= 0 for a, b in sizes):
        return [], [], 'Split sizes are unavailable; start a new search to enable comparisons.'
    descriptors = evaluation.get('split_descriptors', [])
    repeats = evaluation['repeats']
    k = evaluation.get('folds_per_repeat', 0)
    expected = [(r, f) for r in range(1, repeats + 1) for f in range(1, k + 1)]
    if (len(descriptors) != n or len(expected) != n
            or [(d.get('repeat'), d.get('fold')) for d in descriptors] != expected):
        return [], [], 'Complete, ordered repeat/fold identities are required for paired comparisons.'
    metric = spec['metric']
    completed = [trial for trial in state['trials'] if trial['state'] == 'complete'
                 and trial.get('fold_ids') == list(range(n)) and len(trial['folds']) == n
                 and all(isinstance(f.get(metric), (int, float)) and np.isfinite(f[metric]) for f in trial['folds'])]
    completed.sort(key=lambda trial: (-np.mean([f[metric] for f in trial['folds']]), trial['id']))
    return completed, [b / a for a, b in sizes], ''


def comparison_frame(state, tolerance, limit=5, probability=0.95):
    """All pairs within the top completed trials; never pair different split IDs."""
    empty = pd.DataFrame(columns=COLUMNS)
    completed, ratios, note = comparison_data(state)
    if note:
        return empty, note
    if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
        return empty, 'Set a positive practical-equivalence tolerance in primary-metric units.'
    if type(limit) is not int or not 2 <= limit <= 20:
        return empty, 'Choose a comparison shortlist size from 2 to 20.'
    if not np.isfinite(probability) or not 0.5 < probability < 1:
        return empty, 'Decision probability must be greater than 0.5 and less than 1.'
    spec = state['specification']
    evaluation = spec['evaluation']
    repeats, k, n = evaluation['repeats'], evaluation['folds_per_repeat'], spec['folds']
    metric = spec['metric']
    shortlist = completed[:limit]
    if len(shortlist) < 2:
        return empty, 'Complete at least two trials on every evaluation split to compare them.'
    rows = []
    for a, b in combinations(shortlist, 2):
        row = {'Trial A': a['id'] + 1, 'Trial B': b['id'] + 1, 'Repeats': repeats, 'Paired splits': n}
        differences = np.array([f[metric] for f in a['folds']]) - np.array([f[metric] for f in b['folds']])
        row['Repeat advantage SD'] = float(differences.reshape(repeats, k).mean(axis=1).std(ddof=1))
        try:
            row.update(paired_posterior(differences, ratios, tolerance, probability))
        except ValueError as exc:
            row['Assessment'] = str(exc)
        rows.append(row)
    note = (f'All {len(rows)} pairs among the top {len(shortlist)} of {len(completed)} completed trials. '
            'Positive advantage means A performs better; tolerance is an absolute metric difference. '
            'Probabilities and 95% credible intervals use the corrected Student-t approximation, '
            'with the mean validation/training size ratio. Repeat advantage SD describes variation across repeat means; '
            'it is not a standard error and the repeats are not independent datasets. Pairwise probabilities are not probabilities '
            'of being the best trial. Exploratory comparison of tuning results: selection bias is not removed. '
            'No multiple-comparison p-value correction is applied; no trial is accepted automatically.')
    return pd.DataFrame(rows, columns=COLUMNS), note
