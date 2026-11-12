import copy

import numpy as np
import pytest
from pipeline_comparison import comparison_frame, paired_posterior
from scipy.stats import t

pytestmark = pytest.mark.unit


def state():
    metric = 'neg_mean_absolute_error'
    return {'specification': {'metric': metric, 'folds': 6, 'split_sizes': [[20, 10]] * 6,
        'evaluation': {'origin': 'upstream', 'strategy': 'kfold', 'repeats': 2, 'folds_per_repeat': 3,
            'plan': {'recipe': {'shuffle': True}},
            'split_descriptors': [{'repeat': r, 'fold': f} for r in (1, 2) for f in (1, 2, 3)]}},
        'trials': [{'id': i, 'state': 'complete', 'fold_ids': list(range(6)),
                    'folds': [{metric: value} for value in scores]} for i, scores in enumerate([
                        [-1, -1.1, -.9, -1.2, -1, -.8],
                        [-2, -2.2, -2, -2, -2.2, -1.8],
                        [-1.01, -1.11, -.92, -1.18, -1.02, -.79]])]}


def test_matches_corrected_student_t_example_and_direction():
    d = np.array([.1, .2, -.1, .1, .4, .2])
    result = paired_posterior(d, [.5] * 6, .05)
    posterior = t(5, loc=d.mean(), scale=np.sqrt((1/6 + .5) * d.var(ddof=1)))
    assert result['P(A worse)'] == pytest.approx(posterior.cdf(-.05))
    assert result['P(A better)'] == pytest.approx(posterior.sf(.05))
    assert result['P(equivalent)'] == pytest.approx(posterior.cdf(.05) - posterior.cdf(-.05))
    assert [result['95% lower'], result['95% upper']] == pytest.approx(posterior.interval(.95))
    reverse = paired_posterior(-d, [.5] * 6, .05)
    assert reverse['P(A better)'] == pytest.approx(result['P(A worse)'])
    assert sum(result[k] for k in ('P(A better)', 'P(equivalent)', 'P(A worse)')) == pytest.approx(1)
    assert posterior.std() > d.std(ddof=1) / np.sqrt(6)


def test_all_pairs_loss_orientation_repeat_summaries_and_shortlist():
    data = state()
    original = copy.deepcopy(data)
    table, note = comparison_frame(data, .1)
    assert len(table) == 3
    row = table.loc[(table['Trial A'] == 1) & (table['Trial B'] == 2)].iloc[0]
    assert row['Mean advantage A'] > 0  # Smaller MAE is better.
    assert row['P(A better)'] > .95
    difference = np.array([f['neg_mean_absolute_error'] for f in data['trials'][0]['folds']]) - np.array(
        [f['neg_mean_absolute_error'] for f in data['trials'][1]['folds']])
    assert row['Repeat advantage SD'] == pytest.approx(difference.reshape(2, 3).mean(axis=1).std(ddof=1))
    assert row['Repeats'] == 2 and row['Paired splits'] == 6
    assert 'not independent datasets' in note and 'selection bias' in note
    assert len(comparison_frame(data, .1, limit=2)[0]) == 1
    assert data == original


@pytest.mark.parametrize('tolerance', [None, 0, -1, np.inf, np.nan])
def test_requires_explicit_positive_tolerance(tolerance):
    assert comparison_frame(state(), tolerance)[0].empty


def test_rejects_unsupported_plans_and_unpaired_trials():
    for strategy in ('grouped', 'temporal', 'predefined'):
        data = state()
        data['specification']['evaluation']['strategy'] = strategy
        assert comparison_frame(data, .1)[0].empty
    data = state()
    data['specification']['evaluation']['repeats'] = 1
    assert comparison_frame(data, .1)[0].empty
    data = state()
    data['specification']['evaluation']['split_descriptors'][0]['fold'] = 2
    assert comparison_frame(data, .1)[0].empty
    data = state()
    data['trials'][0]['fold_ids'].reverse()
    data['trials'][1]['state'] = 'failed'
    assert comparison_frame(data, .1)[0].empty


def test_degenerate_differences_do_not_claim_certain_equivalence():
    with pytest.raises(ValueError, match='variability'):
        paired_posterior([0.] * 6, [.5] * 6, .1)
    data = state()
    data['trials'][1]['folds'] = copy.deepcopy(data['trials'][0]['folds'])
    table, _ = comparison_frame(data, .1)
    row = table.loc[(table['Trial A'] == 1) & (table['Trial B'] == 2)].iloc[0]
    assert 'variability' in row['Assessment']
    assert np.isnan(row['P(equivalent)'])
