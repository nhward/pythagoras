"""Kernel families remain scalar, conditional, cloneable and fold-fitted."""
import json
import pickle

import numpy as np
import pytest
from estimator_catalogue import load_catalogue
from kernel_search import SearchablePairwiseKernel, kernel_parameters
from model_commit import build_model
from pipeline_search import fresh, prepare, run, validate_checkpoint
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from test_pipeline_search import source
from tuning import (
    compile_spec,
    conditional_grids,
    discover,
    discover_source,
    restore_commitment,
)

pytestmark = pytest.mark.unit


def gp_source():
    data = source()
    profile = next(p for p in load_catalogue()['profiles']
                   if p['id'] == 'sklearn.gaussian_process.GaussianProcessRegressor:searchable_kernels_v1')
    model = build_model({'selectable': True, 'profile': profile, 'estimator': profile['estimator']}, data)
    # Explicit observation noise for a noisy fixture; no automatic scaling/noise changes.
    model.set_params(estimator__alpha=.01)
    return data.with_estimator(model, label='GP', recipe_id=profile['id'])


def commitment(data):
    rows = discover_source(data)
    values = {'metric': ['rbf', 'poly', 'linear'], 'gamma': [.1, 1.], 'degree': [2, 3]}
    drafts = {r['path']: dict(r['definition'], values=values[r['path'].split('__')[-1]],
              active=True, owner=r['owner']) for r in rows
              if '__kernel__' in r['path'] and r['path'].split('__')[-1] in values}
    return rows, drafts, compile_spec(rows, drafts)


@pytest.mark.parametrize('family', ['rbf', 'poly', 'linear'])
def test_native_kernel_equivalence_clone_diagonal_gradient_and_pickle(family):
    X = np.array([[0., 1.], [1., 2.], [2., 3.]])
    kernel = SearchablePairwiseKernel(metric=family, gamma=.3, degree=2, coef0=.7)
    reference = PairwiseKernel(metric=family, gamma=.3, gamma_bounds='fixed' if family == 'linear' else (1e-5, 1e5),
                              pairwise_kernels_kwargs={'degree': 2, 'coef0': .7})
    K, gradient = kernel(X, eval_gradient=True)
    expected, expected_gradient = reference(X, eval_gradient=True)
    np.testing.assert_allclose(K, expected)
    np.testing.assert_allclose(gradient, expected_gradient)
    np.testing.assert_allclose(kernel.diag(X), np.diag(K))
    np.testing.assert_allclose(pickle.loads(pickle.dumps(clone(kernel)))(X), K)
    assert kernel.n_dims == (0 if family == 'linear' else 1)
    if family != 'linear':
        moved = kernel.clone_with_theta(np.log([.8]))
        assert moved.gamma == pytest.approx(.8)
    if family == 'poly':
        assert not np.allclose(K, clone(kernel).set_params(degree=3)(X))


def test_default_covariance_matches_legacy_and_optimizer_none_means_exact_gamma():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0])
    kernel = SearchablePairwiseKernel()
    np.testing.assert_allclose(kernel(X), RBF(1.)(X))
    model = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=.01).fit(X, y)
    assert model.kernel_.gamma == .5
    mean, std = model.predict(X, return_std=True)
    _, covariance = model.predict(X, return_cov=True)
    assert np.isfinite(mean).all()
    np.testing.assert_allclose(std ** 2, np.diag(covariance), atol=1e-12)
    optimized = GaussianProcessRegressor(kernel=kernel, alpha=.01).fit(X, y)
    assert optimized.alpha == .01  # Alpha is fixed, not an optimizer starting value.
    assert optimized.kernel_.metric == 'rbf'
    assert optimized.kernel_.gamma != pytest.approx(.5)


@pytest.mark.parametrize('parameters', [{'metric': 'unknown'}, {'gamma': 0}, {'metric': 'poly', 'degree': 1.5},
                                        {'metric': 'poly', 'degree': 0}, {'metric': 'poly', 'coef0': -1}])
def test_invalid_kernel_configuration_fails_clearly(parameters):
    with pytest.raises(ValueError):
        SearchablePairwiseKernel(**parameters)(np.ones((3, 1)))


def test_safe_descriptor_and_legacy_recipe_construction():
    for cls in ('GaussianProcessRegressor', 'GaussianProcessClassifier'):
        name = 'sklearn.gaussian_process.' + cls
        assert kernel_parameters(name, {'kernel': None}) == {'kernel': None}
        with pytest.raises(ValueError):
            kernel_parameters(name, {'kernel': {'arbitrary_import': 'bad'}})
    with pytest.raises(ValueError):
        kernel_parameters('sklearn.svm.SVR', {'kernel': {'metric': 'rbf'}})
    profile = next(p for p in load_catalogue()['profiles'] if p['id'].endswith('Regressor:exact_gp'))
    model = build_model({'selectable': True, 'profile': profile, 'estimator': profile['estimator']}, ['x'])
    assert model.named_steps.estimator.kernel is None


def test_conditional_gp_grid_restores_and_counts_only_applicable_parameters():
    data = gp_source()
    rows, drafts, spec = commitment(data)
    assert spec['combinations'] == 7  # Two RBF, four polynomial, one linear.
    restored = restore_commitment(rows, json.loads(json.dumps({'spec': spec, 'drafts': drafts})))
    assert restored['spec'] == spec
    from sklearn.model_selection import ParameterGrid
    conditions = {r['path']: r['condition'] for r in rows if r['condition'] and r['path'] in spec['grid']}
    configurations = list(ParameterGrid(conditional_grids(spec['grid'], conditions)))
    linear = next(c for c in configurations if c['cand_methods__estimator__kernel__metric'] == 'linear')
    assert not any(k.endswith(('__gamma', '__degree')) for k in linear)


@pytest.mark.parametrize('estimator', [SVR(), NuSVR(), SVC(), NuSVC(), KernelRidge()])
def test_native_kernel_methods_use_conditional_gamma_degree_and_offset(estimator):
    rows = discover(Pipeline([('model', estimator)]))
    values = {'kernel': ['rbf', 'poly', 'linear'], 'gamma': [.1, 1.], 'degree': [2, 3], 'coef0': [0., 1.]}
    drafts = {r['path']: dict(r['definition'], values=values[r['path'].split('__')[-1]], active=True, owner=r['owner'])
              for r in rows if r['path'].split('__')[-1] in values}
    assert compile_spec(rows, drafts)['combinations'] == 11


@pytest.mark.parametrize('method', ['random', 'grid', 'tpe', 'skopt'])
def test_all_search_methods_visit_three_gp_kernel_families_and_resume(method):
    data = gp_source()
    _, _, spec = commitment(data)
    # Keep this search focused on family/degree; exact scale is also covered above.
    spec['grid'].pop('cand_methods__estimator__kernel__gamma')
    spec['combinations'] = 4
    data = data.with_tuning_spec(spec)
    problem = prepare(data, method=method)
    stopped = False
    def publish(state):
        nonlocal stopped
        stopped = sum(t['state'] == 'complete' for t in state['trials']) >= 2
    partial = run(problem, fresh(problem), publish=publish, pause=lambda: stopped)
    assert partial['status'] == 'paused'
    resumed = run(problem, json.loads(json.dumps(partial)), publish=lambda s: None, pause=lambda: False)
    assert resumed['status'] == 'exhausted', resumed.get('error')
    validate_checkpoint(problem, resumed)
    assert all(t['state'] == 'complete' for t in resumed['trials']), resumed['trials']
    assert {t['params'].get('cand_methods__estimator__kernel__metric', 'rbf') for t in resumed['trials']} == {'rbf', 'poly', 'linear'}
    for trial in resumed['trials']:
        if trial['params'].get('cand_methods__estimator__kernel__metric') == 'poly':
            assert trial['params']['cand_methods__estimator__kernel__degree'] in (2, 3)


def test_new_recipe_is_preferred_without_changing_legacy_identity():
    from cards.cand_methods import _analyze, _method_groups
    from estimator_catalogue import installed_versions
    data = source()
    result = _analyze(data, load_catalogue(), installed_versions())
    rows = _method_groups(result, show_all=True)['sklearn.gaussian_process.GaussianProcessRegressor']
    assert rows[0]['id'].endswith(':searchable_kernels_v1')
    assert any(row['id'].endswith(':exact_gp') for row in rows)


def test_gp_refits_scale_within_training_folds_and_replays_construction():
    from sklearn.model_selection import KFold, cross_validate
    X = np.linspace(-1, 1, 18).reshape(-1, 1)
    y = np.sin(3 * X[:, 0])
    model = GaussianProcessRegressor(kernel=SearchablePairwiseKernel(), alpha=.01)
    splits = list(KFold(3).split(X))
    results = cross_validate(model, X, y, cv=splits, return_estimator=True)
    for (train, held), fitted in zip(splits, results['estimator']):
        np.testing.assert_array_equal(fitted.X_train_, X[train])
        assert len(fitted.X_train_) == 12
    replay = eval(repr(SearchablePairwiseKernel(metric='poly', degree=4)),
                  {'SearchablePairwiseKernel': SearchablePairwiseKernel})
    assert replay.metric == 'poly' and replay.degree == 4


def test_gp_and_svm_alternatives_keep_separate_conditional_kernel_grids():
    from model_commit import prepare_predictors
    from sklearn.preprocessing import FunctionTransformer
    data = gp_source()
    gp = data.pipeline.named_steps.cand_methods
    svr = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': ['x'], 'preparation': 'numeric'})), ('estimator', SVR())])
    data = data.with_model_candidates([
        {'id': 'gp', 'label': 'GP', 'estimator': gp},
        {'id': 'svr', 'label': 'SVR', 'estimator': svr}])
    rows = discover_source(data)
    drafts = {}
    for row in rows:
        name = row['path'].split('__')[-1]
        if row.get('candidate_id') and name in {'kernel', 'metric', 'degree'} and not row['fixed']:
            drafts[row['path']] = dict(row['definition'], active=True, owner=row['owner'],
                values=[2, 3] if name == 'degree' else ['rbf', 'poly', 'linear'])
    spec = compile_spec(rows, drafts)
    assert spec['combinations'] == 8
    problem = prepare(data.with_tuning_spec(spec), method='grid')
    result = run(problem, fresh(problem), publish=lambda s: None, pause=lambda: False)
    assert result['status'] == 'exhausted'
    assert all(t['state'] == 'complete' for t in result['trials']), [(t['params'], t['error']) for t in result['trials']]
    from pipeline_search import accepted_parameters
    trial = next(t for t in result['trials'] if t['model_id'] == 'gp'
                 and t['params'].get('cand_methods__estimator__kernel__metric') == 'poly')
    selected = data.with_tuned_parameters(accepted_parameters(problem, result, trial['id']),
        fingerprint=problem.specification['fingerprint'], trial=trial['id'],
        metric=problem.specification['metric'], score=0, candidate_id='gp')
    kernel = selected.pipeline_for_training().named_steps.cand_methods.named_steps.estimator.kernel
    assert kernel.metric == 'poly' and kernel.degree in (2, 3)


@pytest.mark.parametrize(('name', 'values'), [('gamma', [0.]), ('gamma', [-1.]), ('degree', [0]),
                                             ('coef0', [-1.]), ('gamma', [None])])
def test_invalid_gp_choices_are_rejected_before_training(name, values):
    data = gp_source()
    rows = discover_source(data)
    row = next(r for r in rows if r['path'] == 'cand_methods__estimator__kernel__' + name)
    drafts = {row['path']: dict(row['definition'], values=values, active=True, owner=row['owner'])}
    with pytest.raises(ValueError):
        compile_spec(rows, drafts)


@pytest.mark.skipif(__import__('joblib').cpu_count() < 2, reason='Requires two available CPUs')
def test_kernel_search_runs_in_parallel_fold_workers():
    data = gp_source()
    _, _, spec = commitment(data)
    problem = prepare(data.with_tuning_spec(spec), method='grid', execution='folds', workers=2)
    result = run(problem, fresh(problem), publish=lambda s: None, pause=lambda: False)
    assert result['status'] == 'exhausted'
    assert all(t['state'] == 'complete' for t in result['trials']), [(t['params'], t['error']) for t in result['trials']]
