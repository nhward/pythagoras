"""Sorted-rank curves, descriptive gaps, and immutable card behavior."""
from threading import Event

import numpy as np
import pandas as pd
import pytest
from cards import var_continuous as m
from code_recording import recording_context
from cyclic_pandas import as_cyclic
from module import Module
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.playwright import controller
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/var_continuous.py', scope='function')
restoring_app = create_app_fixture(app='../scenarios/var_continuous_restoring.py', scope='function')


def source(frame):
    roles = RoleMap()
    for column in frame:
        roles.set_roles(column, [Role.PREDICTOR])
    return proxy_data(_df=frame, _roles=roles)


@pytest.mark.unit
def test_eligibility_respects_decimal_semantics_and_special_roles():
    data = source(pd.DataFrame({
        'decimal': [1., 2., 3.], 'nullable': pd.Series([1., None, 3.], dtype='Float64'),
        'integer': [1, 2, 3], 'logical': [True, False, True],
        'category': pd.Categorical([1., 2., 3.]),
        'cyclic': as_cyclic([1., 2., 3.], period=24),
        'target': [1., 2., 3.], 'identifier': [1., 2., 3.], 'weight': [1., 1., 1.],
        'shadow__decimal': [1., 2., 3.], 'ignored': [1., 2., 3.],
    }))
    for column, role in [('target', Role.TARGET), ('identifier', Role.IDENTIFIER), ('weight', Role.WEIGHTING)]:
        data.role_map.set_roles(column, [Role.PREDICTOR, role])
    data.role_map.set_roles('ignored', [Role.NONE])
    assert m._eligible_columns(data) == ['decimal', 'nullable']


@pytest.mark.unit
def test_treatment_and_optional_targets_precede_predictors():
    data = source(pd.DataFrame({**{f'p{i}': [1., 2.] for i in range(16)},
                                'treatment': [3., 4.], 'target': [5., 6.],
                                'integer_treatment': [1, 2]}))
    data.role_map.set_roles('treatment', [Role.TREATMENT])
    data.role_map.set_roles('integer_treatment', [Role.TREATMENT])
    data.role_map.set_roles('target', [Role.TARGET, Role.PREDICTOR])
    default = m._eligible_columns(data)
    enabled = m._eligible_columns(data, use_target=True)
    assert default[:15] == ['treatment', *[f'p{i}' for i in range(14)]]
    assert enabled[:15] == ['target', 'treatment', *[f'p{i}' for i in range(13)]]
    assert 'target' not in default and 'integer_treatment' not in enabled
    assert list(m._analyze(data, ['target', 'treatment']).curves) == ['treatment']
    assert list(m._analyze(data, ['target', 'treatment'], use_target=True).curves) == ['target', 'treatment']


@pytest.mark.unit
def test_sorted_values_ranks_ties_and_per_variable_omissions():
    data = source(pd.DataFrame({'x': [3., np.nan, 1., 1., np.inf], 'y': [2., 4., -np.inf, 6., 8.]}))
    data.frame.index = [0] * 5
    before = data.clone()
    result = m._analyze(data, ['x', 'y'])
    np.testing.assert_array_equal(result.curves['x'], [1., 1., 3.])
    np.testing.assert_array_equal(result.curves['y'], [2., 4., 6., 8.])
    figure = m._figure(result)
    np.testing.assert_allclose(figure.data[0].x, [100/3, 200/3, 100])
    np.testing.assert_allclose(figure.data[0].y, [0, 0, 1])
    np.testing.assert_allclose(figure.data[1].y, [0, 1/3, 2/3, 1])
    np.testing.assert_array_equal(figure.data[0].customdata, [[1, 1], [1, 2], [3, 3]])
    assert 'Value: %{customdata[0]' in figure.data[0].hovertemplate
    assert figure.layout.yaxis.showticklabels is False
    assert figure.layout.yaxis.ticks == ''
    assert result.table['Omitted'].tolist() == [2, 1]
    assert not figure.layout.showlegend
    assert m._figure(result, True).layout.showlegend
    assert data.equals(before)


@pytest.mark.unit
def test_descriptive_gap_and_low_cardinality_suppression():
    data = source(pd.DataFrame({'gap': [0., 1., 2., 20., 21.], 'repeated': [0., 0., 0., 10., 10.]}))
    table = m._analyze(data, ['gap', 'repeated']).table.set_index('Variable')
    assert table.loc['gap', 'Gap from'] == 2
    assert table.loc['gap', 'Gap to'] == 20
    assert table.loc['gap', 'Largest gap'] == 18
    assert table.loc['gap', 'Gap / range %'] == pytest.approx(100*18/21)
    assert table.loc['gap', 'Distinct %'] == 100
    assert table.loc['repeated', 'Distinct %'] == 40
    assert pd.isna(table.loc['repeated', 'Largest gap'])
    assert 'Withheld' in table.loc['repeated', 'Summary']


@pytest.mark.unit
def test_limit_is_applied_before_sorting_and_sampling_is_shared(monkeypatch):
    data = source(pd.DataFrame({'x': np.arange(1000, dtype=float), 'y': np.arange(1000, dtype=float)+10}))
    original_sort = np.sort
    sizes = []

    def bounded_sort(values, *args, **kwargs):
        sizes.append(len(values))
        return original_sort(values, *args, **kwargs)

    monkeypatch.setattr(m.np, 'sort', bounded_sort)
    first = m._analyze(data, ['x', 'y'], limit=17)
    second = m._analyze(data, ['x', 'y'], limit=17)
    assert sizes and max(sizes) == 17
    assert first.sampled == 17 and first.total == 1000
    np.testing.assert_array_equal(first.curves['x'], second.curves['x'])
    np.testing.assert_array_equal(first.curves['y'], first.curves['x']+10)


@pytest.mark.unit
def test_empty_constant_singleton_extreme_and_cancelled_cases():
    data = source(pd.DataFrame({'empty': [np.nan]*3, 'constant': [2.]*3, 'single': [np.nan, 1., np.inf]}))
    result = m._analyze(data, ['empty', 'constant', 'single'])
    assert len(m._figure(result).data) == 2
    assert m._figure(result).data[1].mode == 'markers'
    np.testing.assert_array_equal(m._figure(result).data[0].y, [0, 0, 0])
    np.testing.assert_array_equal(m._figure(result).data[1].y, [0])
    assert 'Select at least one' in m._analyze(data, []).message
    assert 'No finite' in m._analyze(data, ['empty']).message
    assert 'No eligible decimal' in m._analyze(source(pd.DataFrame({'int': [1, 2]})), ['int']).message
    assert m._figure(m._analyze(data, [])).layout.images
    extreme = source(pd.DataFrame({'x': [-1e308, 1e308]}))
    assert 'floating-point range' in m._analyze(extreme, ['x']).table.iloc[0]['Summary']
    np.testing.assert_array_equal(m._figure(m._analyze(extreme, ['x'])).data[0].y, [0, 1])
    cancel = Event(); cancel.set()
    assert 'superseded' in m._analyze(data, ['constant'], cancel=cancel).message
    assert m._analyze(source(pd.DataFrame({'x': pd.Series([], dtype=float)})), ['x']).sampled == 0


@pytest.mark.unit
def test_pipeline_unchanged_and_recorded_calculations_replay():
    from sklearn.preprocessing import FunctionTransformer
    data = source(pd.DataFrame({'x': [3., 1., 2.]}))
    data = data.with_pipeline_step(FunctionTransformer(), name='identity', preview_frame=data.frame.copy())
    before = data.clone()
    registry = {}
    result = recording_context(registry, m._analyze)(data, ['x'])
    recording_context(registry, m._figure)(result)
    assert {'_analyze', '_gap_summary', '_figure', 'ContinuousAnalysis'} <= registry.keys()
    namespace = {k: v for k, v in vars(m).items() if not k.startswith('_')}
    # Define the result container first, then functions whose annotations/defaults
    # can refer to it. Imports and constants are supplied as in a document setup.
    for name in ['ContinuousAnalysis', *[name for name in registry if name != 'ContinuousAnalysis']]:
        exec(Module.clean_code(registry[name]), namespace)
    replay = namespace['_analyze'](data, ['x'])
    pd.testing.assert_frame_equal(result.table, replay.table)
    assert data.equals(before) and data.has_pipeline


def by_id(page, name):
    return page.locator(f'[id$="-{name}"]')


@pytest.mark.ui
def test_default_selection_limit_legend_flip_and_code(page, app):
    page.set_viewport_size({'width': 1800, 'height': 1100})
    page.goto(app.url)
    expect(by_id(page, 'Status')).to_contain_text('15 variables plotted', timeout=60000)
    expect(by_id(page, 'PassThrough')).to_have_text('unchanged=True')
    chart = by_id(page, 'Chart').locator('.js-plotly-plot')
    expect(chart).to_be_visible()
    assert chart.evaluate('el => el.data.length') == 15
    assert chart.evaluate('el => el.layout.showlegend') is False
    by_id(page, 'ExpandButton').click(force=True)
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout.showlegend === true")
    chart.locator('.legendtoggle').first.click()
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data[0].visible === 'legendonly'")
    by_id(page, 'FlipButton').click(force=True)
    expect(by_id(page, 'Summary')).to_contain_text('Largest gap')
    expect(by_id(page, 'Summary')).to_contain_text('Withheld')
    by_id(page, 'CodeButton').click(force=True)
    dialog = page.get_by_role('dialog')
    expect(dialog).to_contain_text('def _gap_summary(')
    expect(dialog).not_to_contain_text('@recordable')
    dialog.get_by_role('button', name='Dismiss').click()
    expect(dialog).to_have_count(0)
    by_id(page, 'FlipButton').click(force=True)
    page.locator('.card').first.locator('button.collapse-toggle').click()
    variables = controller.InputSelectize(page, by_id(page, 'Variables').get_attribute('id'))
    variables.set(['gapped'])
    expect(by_id(page, 'Status')).to_contain_text('1 variable plotted', timeout=30000)
    controller.InputSlider(page, by_id(page, 'Limit').get_attribute('id')).set('10^2')
    expect(by_id(page, 'Status')).to_contain_text('100 of 200 rows assessed', timeout=30000)
    variables.set([])
    expect(by_id(page, 'Status')).to_contain_text('Select at least one', timeout=30000)
    expect(by_id(page, 'PassThrough')).to_have_text('unchanged=True')


@pytest.mark.ui
def test_saved_selection_survives_late_upstream_types(page, restoring_app):
    page.goto(restoring_app.url)
    expect(by_id(page, 'Status')).to_contain_text('1 variable plotted', timeout=60000)
    by_id(page, 'FinishRestore').click()
    expect(by_id(page, 'Status')).to_contain_text('2 variables plotted', timeout=30000)
    chart = by_id(page, 'Chart').locator('.js-plotly-plot')
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data.length === 2")
    assert chart.evaluate('el => el.data.map(trace => trace.name)') == ['gapped', 'later']
