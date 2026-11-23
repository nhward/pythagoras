"""Residual pairing, repeated-validation coverage and portable diagnostic views."""
import json
from dataclasses import replace

import numpy as np
import pytest
from ensemble import evaluate_ensemble, portable_ensemble_display, representatives, restore_ensemble_display
from ensemble_diagnostics import build_residual_diagnostics, performance_figure, residual_block
from ensemble_scenario import example_source
from fit_responses import Responses

pytestmark = pytest.mark.unit


def test_residuals_align_within_split_and_repeats_are_not_unique_observations():
    members = (('a', 'A'), ('b', 'B'))
    blocks = []
    for offset in (0, 1):
        a = Responses({'predict': np.array([11., 12., 13.]) + offset})
        b = Responses({'predict': np.array([9., 8., 7.]) - offset})
        combined = Responses({'predict': np.full(3, 10.)})
        blocks.append(residual_block([2, 0, 1], np.full(3, 10.), [('a', a), ('b', b)], combined, 'regression', ()))
    result = build_residual_diagnostics(members, blocks, 'regression')
    view = result.summary['views'][0]
    assert view['events'] == 6 and view['observations'] == 3
    pair = view['pairs'][0]
    assert pair['correlation'] == pytest.approx(-1)
    assert pair['events'] == 6 and pair['observations'] == 3
    assert sum(n for _, _, n in pair['bins']) == 6
    assert all(p['correlation'] is None for p in view['pairs'][1:])  # constant ensemble residuals
    np.testing.assert_array_equal(result.rows, [2, 0, 1, 2, 0, 1])
    np.testing.assert_array_equal(result.folds, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result.samples[0, :, 0], [1, 2, 3, 2, 3, 4])
    np.testing.assert_array_equal(result.samples[0, :, 1], [-1, -2, -3, -2, -3, -4])


def test_missing_residuals_reduce_pair_coverage_without_dropping_selected_models():
    values = np.array([[[1., 3., 0.], [2., np.nan, 0.], [3., 5., 0.]]])
    result = build_residual_diagnostics((('a', 'A'), ('b', 'B')), [(np.arange(3), values)], 'regression')
    assert [m['key'] for m in result.summary['models']] == ['a', 'b', 'ensemble']
    pair = result.summary['views'][0]['pairs'][0]
    assert pair['events'] == pair['observations'] == 2
    assert pair['correlation'] == pytest.approx(1)


def test_classification_aligns_probability_columns_and_does_not_encode_class_labels():
    response = Responses({'predict_proba': np.array([[.8, .2], [.3, .7]])}, classes=np.array(['yes', 'no']))
    hard = Responses({'predict': np.array(['no', 'yes'])}, classes=np.array(['no', 'yes']))
    block = residual_block([0, 1], ['no', 'yes'], [('a', response)], hard, 'classification', np.array(['no', 'yes']))
    np.testing.assert_allclose(block[1][0, :, 0], [-.8, .7])
    np.testing.assert_allclose(block[1][1, :, 0], [.8, -.7])
    assert np.isnan(block[1][:, :, 1]).all()


@pytest.fixture(scope='module')
def result():
    source = example_source(count=3)
    return evaluate_ensemble(source, tuple(representatives(source)[1]), 'mean')


def test_heatmap_and_bars_share_the_selected_model_order(result):
    figure = performance_figure(result, 'neg_root_mean_squared_error')
    keys = result.table('neg_root_mean_squared_error').Key.tolist()
    correlation = next(t for t in figure.data if t.meta and t.meta.get('role') == 'correlation')
    assert list(figure.data[0].y) == list(correlation.y) == list(correlation.x) == keys
    assert all(correlation.z[i][j] is None for i in range(len(keys)) for j in range(i+1, len(keys)))
    assert figure.layout.yaxis2.matches == 'y'
    assert correlation.zmin == -1 and correlation.zmax == 1
    assert correlation.showscale is False
    assert any(t.meta and t.meta.get('role') == 'correlation' and t.showscale for t in performance_figure(result, 'r2', full_screen=True).data)
    assert figure.layout.xaxis2.domain[0] - figure.layout.xaxis.domain[1] == pytest.approx(.015)
    assert figure.layout.annotations[1].text == 'Resampled performance'
    assert [m['key'] for m in result.residuals.summary['models']] == ['method_0', 'method_1', 'method_2', 'ensemble']


def test_every_selected_pair_is_drawn_and_portable_density_preserves_counts(result):
    saved = json.loads(json.dumps(portable_ensemble_display(result), allow_nan=False))
    restored = restore_ensemble_display(saved)
    assert restored.residuals.samples is None
    assert restored.residuals.summary == result.residuals.summary
    points = performance_figure(result, 'r2')
    density = performance_figure(restored, 'r2')
    assert points.layout.meta['pair_count'] == density.layout.meta['pair_count'] == 6
    assert sum(bool(t.meta) and t.meta.get('role') == 'residual-pair' for t in points.data) == 6
    assert points.layout.autosize and points.layout.width is None and points.layout.height is None
    assert sum(bool(t.meta) and t.meta.get('role') == 'residual-pair' for t in density.data) == 6
    assert points.layout.meta['model_keys'] == density.layout.meta['model_keys']
    for trace in density.data:
        if trace.meta and trace.meta.get('role') == 'residual-pair':
            assert np.nansum(np.asarray(trace.z, dtype=float)) == 40
    saved['residuals']['views'][0]['pairs'][0]['bins'] = [[32, 0, 1]]
    with pytest.raises(ValueError, match='Invalid saved ensemble'):
        restore_ensemble_display(saved)


def test_shared_range_uses_actual_residuals_even_with_large_target_values(result):
    observed = np.array([100000., 500000.])
    responses = [(key, Responses({'predict': observed + errors})) for (key, _), errors in zip(
        result.members, ([1., -2.], [-3., 4.], [5., -7.]))]
    block = residual_block([0, 1], observed, responses, Responses({'predict': observed + [1., -1.]}), 'regression', ())
    diagnostics = build_residual_diagnostics(result.members, [block], 'regression')
    actual = replace(result, residuals=diagnostics)
    limit = 7 * 1.05
    assert diagnostics.summary['views'][0]['limit'] == pytest.approx(limit)
    restored = restore_ensemble_display(portable_ensemble_display(actual))
    for evaluated in (actual, restored):
        figure = performance_figure(evaluated, 'r2')
        for trace in figure.data:
            if trace.meta and trace.meta.get('role') == 'residual-pair':
                assert trace.meta['row'] < trace.meta['column']
                x = figure.layout[trace.xaxis.replace('x', 'xaxis')]
                y = figure.layout[trace.yaxis.replace('y', 'yaxis')]
                assert list(x.range) == pytest.approx([-limit, limit])
                assert list(y.range) == pytest.approx([-limit, limit])
                assert x.autorange is y.autorange is False



def test_phase_one_bookmarks_remain_readable(result):
    saved = portable_ensemble_display(result)
    saved['version'] = 1
    saved.pop('residuals')
    restored = restore_ensemble_display(saved)
    assert restored.residuals is None
    assert len(performance_figure(restored, 'r2').data[0].x) == 4


def test_unsampled_extreme_does_not_expand_the_plotted_pairs_range(result):
    values = np.array([[[1., -2., 3., 0.], [600000., 1., 2., 0.]]])
    diagnostics = build_residual_diagnostics(result.members, [(np.arange(2), values)], 'regression')
    sampled = replace(diagnostics, samples=diagnostics.samples[:, :1], rows=diagnostics.rows[:1], folds=diagnostics.folds[:1])
    evaluated = replace(result, residuals=sampled)
    figure = performance_figure(evaluated, 'r2')
    assert [-figure.layout.meta['residual_limit'], figure.layout.meta['residual_limit']] == pytest.approx([-3.15, 3.15])
    assert figure.layout.plot_bgcolor == performance_figure(evaluated, 'r2').layout.plot_bgcolor == '#bbd6f8'
    assert sum(trace.type == 'bar' for trace in figure.data) == 1
    # Portable density still represents all held-out events, including the tail.
    restored = restore_ensemble_display(portable_ensemble_display(evaluated))
    assert [-performance_figure(restored, 'r2').layout.meta['residual_limit'], performance_figure(restored, 'r2').layout.meta['residual_limit']] == pytest.approx([-630000., 630000.])


def test_upper_pairs_follow_column_x_row_y_after_metric_reordering(result):
    for metric in ('r2', 'neg_root_mean_squared_error'):
        figure = performance_figure(result, metric)
        original = [m['key'] for m in result.residuals.summary['models']]
        for trace in figure.data:
            if trace.meta and trace.meta.get('role') == 'residual-pair':
                a = original.index(trace.meta['y_key'])
                b = original.index(trace.meta['x_key'])
                mask = np.isfinite(result.residuals.samples[0, :, a]) & np.isfinite(result.residuals.samples[0, :, b])
                np.testing.assert_array_equal(trace.x, result.residuals.samples[0, mask, b])
                np.testing.assert_array_equal(trace.y, result.residuals.samples[0, mask, a])
                x = figure.layout[trace.xaxis.replace('x', 'xaxis')].domain
                y = figure.layout[trace.yaxis.replace('y', 'yaxis')].domain
                n = len(original)
                left, right = figure.layout.xaxis.domain
                assert (left+(right-left)*trace.meta['column']/n) < x[0] < x[1]
                assert (1-(trace.meta['row']+1)/n) < y[0] < y[1]


def test_restored_upper_density_axes_keep_their_model_orientation(result):
    values = np.array([[[-2., 1., -4., 0.], [-1., 2., -3., 1.], [0., 3., -2., 2.], [1., 4., -1., 3.]]])
    diagnostics = build_residual_diagnostics(result.members, [(np.arange(4), values)], 'regression')
    restored = restore_ensemble_display(portable_ensemble_display(replace(result, residuals=diagnostics)))
    figure = performance_figure(restored, 'r2')
    original = [m['key'] for m in diagnostics.summary['models']]
    bin_width = 2 * diagnostics.summary['views'][0]['limit'] / 32
    for trace in figure.data:
        if trace.meta and trace.meta.get('role') == 'residual-pair':
            weights = np.nan_to_num(np.asarray(trace.z, dtype=float))
            mean_x = (weights.sum(axis=0) * trace.x).sum() / weights.sum()
            mean_y = (weights.sum(axis=1) * trace.y).sum() / weights.sum()
            assert mean_x == pytest.approx(values[0, :, original.index(trace.meta['x_key'])].mean(), abs=bin_width)
            assert mean_y == pytest.approx(values[0, :, original.index(trace.meta['y_key'])].mean(), abs=bin_width)
