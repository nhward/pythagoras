import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from cards import data_structure as m
from proxy_data import proxy_data
from roles import Role, RoleMap
from text_pandas import as_text


def source(frame, **roles):
    return proxy_data(_df=frame, _roles=RoleMap.from_primitive(roles))


def longitudinal():
    n = 60
    return source(pd.DataFrame({'id': np.repeat(['A', 'B'], n),
        'time': list(pd.date_range('2025-01-01', periods=n))*2,
        'value': np.tile(np.sin(np.arange(n)/8), 2)}),
        identifier=['id'], sequence=['time'], predictor=['value'])


@pytest.mark.unit
def test_dates_do_not_define_time_and_no_mutation():
    data = source(pd.DataFrame({'id':[1,2], 'birth':pd.to_datetime(['2000-01-01', '2001-01-01'])}), identifier=['id'], predictor=['birth'])
    before = data.clone()
    result = m._analyze(data)
    assert result['conclusions']['temporal'] == 'No temporal ordering declared'
    assert 'cross-sectional-compatible' in result['summary']
    assert 'date attributes' in result['summary']
    assert data.equals(before)
    assert m.instance().mutable is False
    assert m.instance().hasFlipSide()


@pytest.mark.unit
def test_dependence_and_structure_are_separate_and_match_shared_analyzer():
    data = longitudinal()
    result = m._analyze(data)
    assert 'Longitudinal-compatible' in result['summary']
    assert 'detected in' in result['dependency']
    pd.testing.assert_frame_equal(result['dependence'], m.obs_dependence._analyze(data, imputation='none').table)
    data.frame['value'] = 1.
    constant = m._analyze(data)
    assert constant['summary'] == result['summary']
    assert constant['dependency'] == 'Serial dependence not assessable'
    assert constant['dependence']['Status'].str.contains('Constant').all()


@pytest.mark.unit
@pytest.mark.parametrize('problem', ['irregular', 'missing', 'short'])
def test_unavailable_is_not_independence(problem):
    data = longitudinal()
    if problem == 'irregular':
        data.frame.loc[59, 'time'] += pd.Timedelta(days=1)
    elif problem == 'missing':
        data.frame.loc[0, 'value'] = np.nan
    else:
        data = source(data.frame.iloc[:10].copy(), identifier=['id'], sequence=['time'], predictor=['value'])
    result = m._analyze(data)
    assert 'Longitudinal-compatible' in result['summary']
    assert result['dependence']['Adjusted p'].isna().any()
    assert not result['dependence']['Imputed'].any()


@pytest.mark.unit
@pytest.mark.parametrize('problem,expected', [('duplicate','duplicate'), ('missing','missing'), ('multiple','Ambiguous'), ('numeric','time interpretation unconfirmed')])
def test_sequence_problems(problem, expected):
    data = longitudinal()
    if problem == 'duplicate':
        data.frame.loc[1,'time'] = data.frame.loc[0,'time']
    elif problem == 'missing':
        data.frame.loc[1,'time'] = pd.NaT
    elif problem == 'multiple':
        data.role_map.set_roles('value', [Role.SEQUENCE])
    else:
        data.frame['time'] = np.tile(np.arange(60), 2)
    result = m._analyze(data)
    assert expected in result['conclusions']['temporal']


@pytest.mark.unit
def test_composite_row_id_prevents_invented_longitudinal_structure():
    data = longitudinal()
    data.frame['row'] = np.arange(len(data.frame))
    data.role_map.set_roles('row', [Role.IDENTIFIER])
    result = m._analyze(data)
    assert 'no within-entity history' in result['summary']
    assert len(result['sizes']) == len(data.frame)


@pytest.mark.unit
def test_text_and_location_candidates_remain_distinct():
    frame = pd.DataFrame({'text':as_text(['你好', '', None, '   ']), 'lat':[1.,2.,3.,4.], 'long':[2.,3.,4.,5.]})
    result = m._analyze(source(frame))
    assert 'textual content' in result['summary']
    assert 'spatial geometry' not in result['summary']
    assert 'Candidate only' in result['findings']['spatial']['State'].tolist()
    assert result['texts']['text'][0] == 2
    assert np.isnan(result['texts']['text'][2])
    assert '2 empty/whitespace' in result['findings']['textual'].iloc[0]['Finding']


@pytest.mark.unit
def test_geometry_metadata_and_unknown_crs():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import Point
    frame = pd.DataFrame({'geometry':gpd.GeoSeries([Point(1,2,3), None, Point()])})
    data = source(frame, geometry=['geometry'])
    result = m._analyze(data)
    assert result['geometries']['geometry']['crs'] == 'Unknown'
    assert '1/3 valid' in result['findings']['spatial'].iloc[0]['Finding']
    assert 'Z present: True' in result['findings']['spatial'].iloc[0]['Evidence']
    assert m._figure(data, result, 'spatial').data


@pytest.mark.unit
def test_empty_and_all_figures_and_display_invariance():
    data = longitudinal()
    result = m._analyze(data)
    summary = result['summary']
    for panel in m.PANELS:
        for full in (False, True):
            m._figure(data, result, panel, limit=10, full=full).to_json()
            assert result['summary'] == summary
    empty = m._analyze(source(pd.DataFrame({'time':pd.Series([], dtype='datetime64[ns]')}), sequence=['time']))
    assert 'no observations' in empty['summary']


@pytest.mark.unit
def test_limits_and_target_selection(monkeypatch):
    data = longitudinal()
    data.role_map.set_roles('value', [Role.TARGET])
    assert m._analyze(data, use_target=False)['dependence'].empty
    monkeypatch.setattr(m.obs_dependence, 'MAX_ROWS', 10)
    result = m._analyze(data)
    assert result['dependence'].empty
    assert 'Analysis limit exceeded' in result['findings']['temporal'].iloc[-1]['Evidence']

from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/data_structure.py', scope='function')


@pytest.mark.ui
def test_panels_footer_details_and_export(page, app):
    page.set_viewport_size({'width':1600,'height':1100})
    page.goto(app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('Longitudinal-compatible', timeout=30000)
    expect(page.locator('[id$="-temporalSummary"]')).to_contain_text('Serial dependence detected')
    before = status.inner_text()
    for panel, label in m.PANELS.items():
        page.get_by_role('tab', name=label, exact=True).click()
        expect(status).to_have_text(before)
        expect(page.locator(f'[id$="-{panel}Chart"] .js-plotly-plot')).to_be_visible(timeout=15000)
    expect(page.locator('[id$="-Probe"]')).to_have_text('unchanged=True')
    page.get_by_role('tab', name='Temporal', exact=True).click()
    page.wait_for_function("document.querySelector('.hidden > .contract-btn') !== null")
    page.locator('[id$="-ExpandButton"]').click(force=True)
    expect(status).to_have_text(before)
    page.locator('[id$="-FlipButton"]').click(force=True)
    expect(page.locator('[id$="-Findings"]')).to_contain_text('Serial dependence', timeout=15000)
    expect(page.locator('[id$="-Dependence"]')).to_contain_text('Lag 1', timeout=15000)


@pytest.mark.unit
def test_no_detection_does_not_mean_independence():
    rng = np.random.default_rng(739)
    data = source(pd.DataFrame({'time':np.arange(100), 'x':rng.normal(size=100)}), sequence=['time'], predictor=['x'])
    result = m._analyze(data)
    assert 'independence is not established' in result['dependency']
    assert 'time interpretation unconfirmed' in result['summary']
    assert m._figure(data, result, 'temporal', traces=['x'], limit=10).data[0].mode == 'markers'


@pytest.mark.unit
def test_mixed_strata_and_nullable_numeric():
    data = longitudinal()
    data.frame['group'] = pd.Series(['x', None]*60, dtype='string')
    data.role_map.set_roles('group', [Role.STRATIFIER])
    data.frame['value'] = data.frame['value'].astype('Float64')
    data.frame.loc[0, 'value'] = pd.NA
    result = m._analyze(data)
    figure = m._figure(data, result, 'temporal', stratifier='group')
    assert all(trace.name == 'Changing stratum' for trace in figure.data)
    assert len({trace.line.color for trace in figure.data}) == 1
    assert m._strata(data, 'group')[1] == '(Missing)'


@pytest.mark.ui
@pytest.mark.parametrize("finish", ["complete", "cancel"])
def test_guide_closes_settings_sidebar(page, app, finish):
    page.set_viewport_size({'width':1600, 'height':1100})
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Longitudinal-compatible', timeout=30000)
    page.locator('[id$="-GuideButton"]').click(force=True)
    page.wait_for_function("window.myShepherdTour?.isActive()")
    # Jump to a real settings step to cover ending while the sidebar is open.
    page.evaluate("""() => {
        const tour = window.myShepherdTour;
        const step = tour.steps.find(step => {
            const selector = step.options.attachTo?.element;
            return typeof selector === 'string' && document.querySelector(selector)?.closest('.sidebar');
        });
        if (!step) throw new Error('No sidebar Guide step');
        tour.show(step.id);
    }""")
    layout = page.locator('.bslib-sidebar-layout').filter(has=page.locator('[id$="-Stratifier"]'))
    expect(layout).not_to_have_class(re.compile(r'.*sidebar-collapsed.*'))
    expect(page.locator('.shepherd-element:not([hidden])')).to_be_visible()
    page.evaluate('(finish) => window.myShepherdTour[finish]()', finish)
    expect(layout).to_have_class(re.compile(r'.*sidebar-collapsed.*'))
    expect(page.locator('.shepherd-element')).to_have_count(0)
