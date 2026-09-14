"""Cluster membership calculations, learned exports, and real-browser workflows."""
from __future__ import annotations
import os
import sys
from pathlib import Path
APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
from proxy_data import proxy_data
from roles import Role, RoleMap
from cards import obs_clusters as m
from threadpoolctl import threadpool_limits

app = create_app_fixture(app="../scenarios/obs_clusters.py", scope="function")
learned_app = create_app_fixture(app="../scenarios/obs_clusters_learned.py", scope="function")
weighted_app = create_app_fixture(app="../scenarios/obs_clusters_weighted.py", scope="function")
collision_app = create_app_fixture(app="../scenarios/obs_clusters_collision.py", scope="function")

@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield

@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1800, "height": 1100}}

def source(weighted=False):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(np.vstack([rng.normal(-4, .2, (12, 2)), rng.normal(4, .2, (12, 2))]), columns=["x", "y"])
    roles = RoleMap()
    for col in frame:
        roles.set_roles(col, [Role.PREDICTOR])
    if weighted:
        frame['w'] = np.resize([.5, 1., 2.], len(frame))
        roles.set_roles('w', [Role.WEIGHTING])
    return proxy_data(_df=frame, _roles=roles, _cluster_count=2)

def analyze(data=None, **kwargs):
    options = dict(projection='pca', neighbours=3, min_points=3)
    options.update(kwargs)
    return m._analyze(source() if data is None else data, **options)

@pytest.mark.unit
class TestClusters:
    def test_all_methods_and_shared_coordinates(self):
        r = analyze()
        assert not r.error
        assert set(r.labels) == set(m.METHODS)
        assert r.coordinates.shape == (24, 2)
        for method, labels in r.labels.items():
            if method != 'Density':
                assert len(set(labels)) == 2
                assert len(set(labels[:12])) == len(set(labels[12:])) == 1
                assert labels[0] != labels[-1]
            chart = m._figure(r, method)
            reference = m._figure(r, 'Partition')
            assert list(chart.layout.xaxis.range) == list(reference.layout.xaxis.range)
            assert list(chart.layout.yaxis.range) == list(reference.layout.yaxis.range)

    def test_k_one(self):
        r = analyze(source().with_cluster_count(1))
        for method in set(m.METHODS) - {'Density'}:
            assert set(r.labels[method]) == {'c1'}

    def test_k_exceeds_sample_is_not_silently_changed(self):
        r = analyze(source().with_cluster_count(25))
        assert 'exceeds' in r.error
        assert not r.labels

    @pytest.mark.parametrize('centre', ['centroids', 'medoids'])
    def test_importance_and_disable(self, centre):
        data = source(True)
        data.frame.loc[0, 'w'] = 0
        weighted = analyze(data, centre=centre)
        assert set(weighted.labels) == {'Partition', 'Density'}
        assert 0 not in weighted.positions
        assert 'unequal observation importance' in weighted.notes['Spectral']
        equal = analyze(data, use_weights=False, centre=centre)
        assert set(equal.labels) == set(m.METHODS)
        assert len(equal.positions) == 24

    def test_predictors_only(self):
        data = source()
        roles = RoleMap.from_primitive(data.role_map.to_primitive())
        frame = data.frame.copy()
        frame['target'] = np.nan
        roles.set_roles('target', [Role.TARGET])
        frame['shadow__x'] = np.nan
        roles.set_roles('shadow__x', [Role.PREDICTOR])
        r = analyze(proxy_data(_df=frame, _roles=roles, _cluster_count=2))
        np.testing.assert_allclose(r.coordinates, analyze().coordinates)
        assert r.predictors == ['x', 'y']

    def test_export_is_nominal_predictor_and_preserves_positions(self):
        data = source(True)
        data.frame.index = [0] * len(data)
        data.frame.iloc[0, 2] = 0
        data.frame.iloc[1, 0] = np.nan
        r = analyze(data, limit=10)
        out = m._export(r, 'Partition', 'membership')
        assert out.cluster_count == 2
        assert isinstance(out.frame.membership.dtype, pd.CategoricalDtype)
        assert not out.frame.membership.cat.ordered
        assert out.role_map.roles_for('membership') == {Role.PREDICTOR}
        assert out.frame.membership.notna().sum() == 23
        assert out.has_pipeline
        assert out.processing_records[-1].stage == "Learning"
        np.testing.assert_array_equal(out.frame.membership.iloc[r.positions].astype(str), r.labels['Partition'])
        assert_frame_equal(out.frame.drop(columns='membership'), data.frame)
        assert 'membership' not in data.columns
        assert out.processing_records[-1].parameters['method'] == 'Partition'

    @pytest.mark.parametrize('name', ['', 'x', 'shadow__cluster'])
    def test_export_rejects_invalid_or_existing_name(self, name):
        with pytest.raises(ValueError):
            m._export(analyze(), 'Partition', name)

    def test_invalid_weights_are_explained(self):
        data = source(True)
        data.frame.loc[0, 'w'] = -1
        assert 'Importance weights' in analyze(data).error

    def test_single_numeric_predictor_projection(self):
        data = proxy_data(_df=pd.DataFrame({'x': [0., .1, .2, 10., 10.1, 10.2]}), _cluster_count=2)
        r = analyze(data, neighbours=2)
        assert not r.error
        assert (r.coordinates[:, 1] == 0).all()

    def test_tsne_small_sample_is_finite(self):
        r = analyze(projection='tsne')
        assert not r.error
        assert np.isfinite(r.coordinates).all()
        baseline = analyze()
        for method in r.labels:
            np.testing.assert_array_equal(r.labels[method], baseline.labels[method])

    def test_density_noise_exports_as_category(self):
        r = analyze(min_points=100)
        assert set(r.labels['Density']) == {'unallocated'}
        assert 'found 0 groups' in r.notes['Density']
        out = m._export(r, 'Density', 'density')
        assert set(out.frame.density) == {'unallocated'}


def by_id(page, name):
    namespace = page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{namespace}-{name}')

def toggle(page, method):
    return by_id(page, 'IncludeMembership').locator(f'input[value="{method}"]')

@pytest.mark.ui
def test_tabs_table_export_and_reset(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    page.get_by_role('tab', name='Divisive', exact=True).click()
    page.locator('.card').first.hover()
    by_id(page, 'FlipButton').click(force=True)
    expect(by_id(page, 'TableTitle')).to_contain_text('Divisive membership')
    expect(by_id(page, 'Membership')).to_contain_text('Membership')
    by_id(page, 'FlipButton').click(force=True)
    toggle(page, 'Partition').check()
    toggle(page, 'Mixture').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_partition,cluster_mixture; methods=Partition,Mixture; nominal=True; predictors=True')
    toggle(page, 'Partition').uncheck()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_mixture; methods=Mixture')
    toggle(page, 'Mixture').uncheck()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=none')
    expect(by_id(page, 'ExportStatus')).to_have_count(0)

@pytest.mark.ui
def test_existing_columns_get_an_unused_membership_name(page, collision_app):
    page.goto(collision_app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    toggle(page, 'Partition').check()
    toggle(page, 'Mixture').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_partition_3,cluster_mixture_2')
    expect(by_id(page, 'VariableName')).to_have_count(0)

@pytest.mark.ui
def test_export_persists_across_tabs_but_resets_on_upstream_k(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    toggle(page, 'Partition').check()
    toggle(page, 'Mixture').check()
    page.get_by_role('tab', name='Spectral', exact=True).click()
    expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Mixture')
    by_id(page, 'ChangeK').click()
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=3', timeout=60000)
    for method in ['Partition', 'Mixture']:
        expect(toggle(page, method)).not_to_be_checked()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=none')

@pytest.mark.ui
def test_membership_after_learned_step_on_any_tab(page, learned_app):
    page.goto(learned_app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    page.get_by_role('tab', name='Density', exact=True).click()
    toggle(page, 'Mixture').check()
    toggle(page, 'Partition').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Mixture')
    expect(by_id(page, 'ExportProbe')).to_contain_text('clean_columns=x,y')
    toggle(page, 'Mixture').uncheck()
    expect(by_id(page, 'ExportProbe')).to_contain_text('pipeline=scale,obs_clusters; clean_columns=x,y')
    expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition;')
    toggle(page, 'Partition').uncheck()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=none')

@pytest.mark.unit
@pytest.mark.parametrize('border', [True, False])
def test_density_prediction_is_batch_independent_and_handles_missing(border):
    from ClusterMembershipTransformer import ClusterMembershipTransformer
    from sklearn.base import clone
    data = source(True)
    model = ClusterMembershipTransformer(('x', 'y'), method='Density', weighting='w', min_points=3, border=border).fit(data.frame)
    query = pd.DataFrame({'x': [-4., 10000., np.nan], 'y': [-4., 10000., 0.]})
    predicted = model.transform(query).cluster
    assert predicted.iloc[0] != 'unallocated'
    assert predicted.iloc[1] == 'unallocated'
    assert pd.isna(predicted.iloc[2])
    assert 'unallocated' in predicted.cat.categories
    assert model.transform(query.iloc[[0]]).cluster.iloc[0] == predicted.iloc[0]
    assert not hasattr(clone(model), 'core_points_')
    assert model.weighting == 'w'

@pytest.mark.ui
def test_density_membership_can_be_added_and_removed(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    toggle(page, 'Density').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_density; methods=Density; nominal=True; predictors=True')
    toggle(page, 'Partition').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Density')
    toggle(page, 'Density').uncheck()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_partition; methods=Partition;')

@pytest.mark.unit
def test_density_border_rule_and_training_pipeline():
    from sklearn.base import clone
    r = analyze(source(True), min_points=3)
    model = r.transformers['Density']
    # A synthetic fixed training reference isolates the border prediction rule.
    model.standardize = False
    model.predictors_ = ['x', 'y']
    model.core_points_ = np.array([[0., 0.]])
    model.core_labels_ = np.array([0])
    model.density_points_ = np.array([[0., 0.], [.1, 0.], [-.1, 0.]])
    model.density_weights_ = np.ones(3)
    model.radius_ = 1.
    query = pd.DataFrame({'x': [0., .95, 2.], 'y': [0., 0., 0.]})
    assert model.transform(query).cluster.iloc[1] != 'unallocated'
    model.border = False
    assert list(model.transform(query).cluster) == [model.label_map_[0], 'unallocated', 'unallocated']
    r = analyze(source(True), min_points=3)
    out = m._export(r, 'Density', 'density')
    recipe = out.pipeline_for_training()
    assert not hasattr(recipe.steps[-1][1], 'core_points_')
    train = out.clean_frame.iloc[:20]
    recipe.fit(train)
    fitted = recipe.steps[-1][1]
    assert len(fitted.density_points_) == len(train)
    assert recipe.transform(out.clean_frame.iloc[20:]).density.notna().all()
    scaled = source(True)
    scaled.frame['w'] *= 100
    other = analyze(scaled, min_points=3)
    np.testing.assert_array_equal(r.labels['Density'], other.labels['Density'])

@pytest.mark.ui
def test_k_one_disables_visible_membership_controls(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    toggle(page, 'Density').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_density')
    by_id(page, 'KOne').click()
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=1', timeout=60000)
    for method in ['Partition', 'Mixture', 'Density']:
        expect(toggle(page, method)).to_be_visible()
        expect(toggle(page, method)).to_be_disabled()
        expect(toggle(page, method)).not_to_be_checked()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=none')
    by_id(page, 'ChangeK').click()
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=3', timeout=60000)
    for method in ['Partition', 'Mixture', 'Density']:
        expect(toggle(page, method)).to_be_enabled()
    toggle(page, 'Partition').check()
    expect(by_id(page, 'ExportProbe')).to_contain_text('added=cluster_partition')


@pytest.mark.unit
class TestMembershipCoverage:
    @pytest.mark.parametrize('method', ['Partition', 'Mixture', 'Density'])
    def test_export_levels_roles_and_unfitted_recipe(self, method):
        data = source()
        result = analyze(data)
        exported = m._export(result, method, 'members')
        levels = list(exported.frame.members.cat.categories)
        assert set(levels) == ({'c1', 'c2', 'unallocated'} if method == 'Density' else {'c1', 'c2'})
        assert exported.role_map.roles_for('members') == {Role.PREDICTOR}
        assert not exported.frame.members.cat.ordered
        assert not hasattr(exported.pipeline.steps[-1][1], 'label_map_')
        assert_frame_equal(exported.clean_frame, data.frame)
        assert_frame_equal(exported.frame.drop(columns='members'), data.frame)

    def test_multiple_exports_compose_without_using_membership_as_coordinate(self):
        from copy import deepcopy
        data = source()
        result = analyze(data)
        plans = {}
        for method in ['Density', 'Mixture', 'Partition']:
            plans[method] = deepcopy(result.transformers[method]).set_params(output_column=method.lower())
        exported = m._export_memberships(data, plans)
        assert list(exported.columns) == ['x', 'y', 'partition', 'mixture', 'density']
        training = exported.pipeline_for_training().fit(data.frame)
        for _, transformer in training.steps:
            assert transformer.predictors_ == ['x', 'y']
        assert list(training.transform(data.frame).columns) == list(exported.columns)
        assert not data.has_pipeline
        assert len(exported.processing_records) == 3

    def test_identifier_hover_is_escaped_and_missing_identifier_falls_back_to_position(self):
        data = source()
        frame = data.frame.copy()
        frame.index = [7] * len(frame)
        frame['id'] = ['<case & one>', None] + [f'case-{i}' for i in range(2, len(frame))]
        roles = RoleMap.from_primitive(data.role_map.to_primitive())
        roles.set_roles('id', [Role.IDENTIFIER])
        result = analyze(proxy_data(_df=frame, _roles=roles, _cluster_count=2))
        hover = [item[0] for trace in m._figure(result, 'Partition').data for item in trace.customdata]
        assert 'id: &lt;case &amp; one&gt;' in hover
        assert 'Row 2' in hover
        assert not any('<case' in item for item in hover)
        assert result.predictors == ['x', 'y']

    def test_non_numeric_constant_and_nonpredictor_columns_are_excluded(self):
        data = source()
        frame = data.frame.copy()
        frame['boolean'] = np.arange(len(frame)) % 2 == 0
        frame['complex'] = np.arange(len(frame)) + 1j
        frame['category'] = pd.Categorical(np.resize(['a', 'b'], len(frame)))
        frame['constant'] = 7
        roles = RoleMap.from_primitive(data.role_map.to_primitive())
        for name in ['boolean', 'complex', 'category', 'constant']:
            roles.set_roles(name, [Role.PREDICTOR])
        frame['id'] = np.arange(len(frame)) * 1000
        roles.set_roles('id', [Role.IDENTIFIER])
        result = analyze(proxy_data(_df=frame, _roles=roles, _cluster_count=2))
        assert result.predictors == ['x', 'y']
        np.testing.assert_allclose(result.coordinates, analyze().coordinates)

    def test_no_varying_predictors_has_explanation(self):
        result = analyze(proxy_data(_df=pd.DataFrame({'constant': [1., 1., 1.]})))
        assert result.error
        assert not result.labels
        assert m._figure(result, 'Partition').layout.annotations[0].text == result.error

    def test_spectral_disconnected_graph_is_explained(self):
        result = analyze(neighbours=1)
        assert 'disconnected components' in result.notes['Spectral']
        assert 'Spectral' not in result.labels

    def test_sampled_hover_uses_original_rows(self):
        result = analyze(limit=10)
        expected = {f'Row {position + 1}' for position in result.positions}
        actual = {item[0] for trace in m._figure(result, 'Partition').data for item in trace.customdata}
        assert actual == expected
        assert len(result.positions) == 10


def open_settings(page):
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()


def payload(page):
    import json
    return json.loads(by_id(page, 'PayloadProbe').inner_text())


@pytest.mark.ui
class TestWeightedWeb:
    def test_real_weights_and_membership_levels(self, page, weighted_app):
        page.goto(weighted_app.url)
        expect(by_id(page, 'Status')).to_contain_text('23 of 24 rows analyzed; 2 numeric predictors', timeout=60000)
        for method in ['Partition', 'Density']:
            toggle(page, method).check()
        expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Density')
        expect(by_id(page, 'PayloadProbe')).to_contain_text('"cluster_density": 24')
        data = payload(page)
        assert data['unchanged'] and data['rows'] == 24
        assert data['weighting'] == ['importance']
        assert data['levels']['cluster_partition'] == ['c1', 'c2']
        assert set(data['levels']['cluster_density']) == {'c1', 'c2', 'unallocated'}
        assert all(recipe['weighting'] == 'importance' for recipe in data['recipes'].values())
        toggle(page, 'Mixture').check()
        expect(by_id(page, 'Status')).to_contain_text('unequal observation importance')
        expect(toggle(page, 'Mixture')).not_to_be_checked()
        expect(toggle(page, 'Partition')).to_be_checked()
        expect(toggle(page, 'Density')).to_be_checked()

    def test_disabling_weighting_allows_mixture_and_preserves_existing_recipe(self, page, weighted_app):
        page.goto(weighted_app.url)
        expect(by_id(page, 'Status')).to_contain_text('23 of 24', timeout=60000)
        toggle(page, 'Partition').check()
        expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition;')
        open_settings(page)
        by_id(page, 'UseWeights').uncheck()
        by_id(page, 'Centre').select_option('medoids')
        expect(by_id(page, 'Status')).to_contain_text('24 of 24', timeout=60000)
        open_settings(page)
        assert payload(page)['recipes']['Partition'] == {'centre': 'centroids', 'weighting': 'importance'}
        toggle(page, 'Mixture').check()
        expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Mixture')
        assert payload(page)['recipes']['Mixture']['weighting'] is None
        toggle(page, 'Partition').uncheck()
        expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Mixture;')
        toggle(page, 'Partition').check()
        expect(by_id(page, 'ExportProbe')).to_contain_text('methods=Partition,Mixture')
        assert payload(page)['recipes']['Partition'] == {'centre': 'medoids', 'weighting': None}

    def test_identifier_hover_and_active_membership_table(self, page, weighted_app):
        page.goto(weighted_app.url)
        expect(by_id(page, 'Status')).to_contain_text('23 of 24', timeout=60000)
        chart = by_id(page, 'Chart_Partition').locator('.js-plotly-plot')
        expect(chart).to_be_visible()
        page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data?.length > 0")
        hover = chart.evaluate('el => el.data.flatMap(t => t.customdata.map(row => row[0]))')
        assert 'identifier: case-02' in hover
        assert not any(item.startswith('Row ') for item in hover)
        page.get_by_role('tab', name='Density', exact=True).click()
        page.locator('.card').first.hover()
        by_id(page, 'FlipButton').click(force=True)
        expect(by_id(page, 'TableTitle')).to_contain_text('Density membership')
        expect(by_id(page, 'Membership')).to_contain_text('Data: identifier')
        expect(by_id(page, 'Membership')).to_contain_text('case-02')
        expect(by_id(page, 'Membership')).to_contain_text('c1')


@pytest.mark.ui
def test_every_tab_has_comparable_plot_and_does_not_export_implicitly(page, app):
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('Incoming K=2', timeout=60000)
    ranges = []
    for method in m.METHODS:
        page.get_by_role('tab', name=method, exact=True).click()
        chart = by_id(page, f'Chart_{method}').locator('.js-plotly-plot')
        expect(chart).to_be_visible()
        chart_id = by_id(page, f'Chart_{method}').get_attribute('id')
        page.wait_for_function("id => document.getElementById(id)?.querySelector('.js-plotly-plot')?.data?.length > 0", arg=chart_id)
        state = chart.evaluate("el => ({x: el.layout.xaxis.range, y: el.layout.yaxis.range, count: el.data.reduce((n,t) => n + t.x.length, 0), names: el.data.map(t=>t.name)})")
        assert state['count'] == 24
        assert all(name.startswith('Cluster c') or name == 'unallocated' for name in state['names'])
        ranges.append((state['x'], state['y']))
        expect(by_id(page, 'ExportProbe')).to_contain_text('added=none')
    assert all(value == ranges[0] for value in ranges)
