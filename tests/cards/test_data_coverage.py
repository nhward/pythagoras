"""Coverage calculations, mosaic geometry and card browser workflows."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pytest
from cards import data_coverage as m
from proxy_data import proxy_data
from roles import Role, RoleMap
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
from shiny.playwright import controller

app = create_app_fixture(app='../scenarios/data_coverage.py', scope='function')
@pytest.fixture(scope='session')
def browser_context_args():
    return {'viewport':{'width':1800,'height':1100}}


def source():
    frame = pd.DataFrame([('a','x','yes')]*30+[('b','y','no')]*30,columns=['stratum','treatment','sensitive'])
    frame['predictor'] = np.arange(60) % 2
    frame['weight'] = np.arange(60)+1.
    roles = RoleMap()
    for c,r in [('stratum',Role.STRATIFIER),('treatment',Role.TREATMENT),('sensitive',Role.SENSITIVE),('predictor',Role.PREDICTOR),('weight',Role.WEIGHTING)]:
        roles.set_roles(c,[r])
    return proxy_data(_df=frame,_roles=roles)


@pytest.mark.unit
class TestCoverage:
    def test_roles_and_cardinality(self):
        d=source()
        assert m._variables(d)==['stratum','treatment','sensitive']
        d.frame['stratum']=[[1]]*60
        d.frame['treatment']=range(60)
        assert m._variables(d)==['sensitive']

    def test_expected_counts_and_zero_intersections(self):
        r=m._analyze(source(),['stratum','treatment'])
        assert not r.error and r.rows==60
        np.testing.assert_array_equal(r.observed,[[30,0],[0,30]])
        np.testing.assert_array_equal(r.expected,[[15,15],[15,15]])
        assert len(r.table)==4 and r.table.Empty.sum()==2
        assert r.table['Pearson residual'].iloc[0]==pytest.approx(-np.sqrt(15))
        assert r.table['Shortfall'].max()==15
        assert r.table['Observed %'].sum()==100

    def test_independent_nonuniform_counts_are_grey(self):
        frame=pd.DataFrame([('a','x')]*20+[('a','y')]*10+[('b','x')]*40+[('b','y')]*20,columns=['stratum','treatment'])
        d=proxy_data(_df=frame,_roles=source().role_map)
        r=m._analyze(d,list(frame))
        np.testing.assert_allclose(r.expected,r.observed)
        assert set(r.table.Coverage)=={'Near expectation'}
        assert all(m._band(v)==2 for v in r.table['Pearson residual'])

    def test_multiway_independence_and_weights_ignored(self):
        d=source(); before=d.clone()
        r=m._analyze(d,['stratum','treatment','sensitive'])
        assert r.expected.shape==(2,2,2) and np.all(r.expected==7.5)
        assert r.table.Empty.sum()==6 and r.expected.sum()==r.observed.sum()==60
        assert d.equals(before) and 'importance column is not applied' in r.note
        d.frame['weight']*=100
        pd.testing.assert_frame_equal(r.table,m._analyze(d,r.variables).table)

    def test_missing_and_unused_categories(self):
        d=source();d.frame.loc[0,'stratum']=None
        d.frame['treatment']=pd.Categorical(d.frame.treatment,categories=['x','y','unused'])
        r=m._analyze(d,['stratum','treatment'])
        assert r.rows==59 and r.omitted==1 and r.observed.shape==(2,2)
        r=m._analyze(d,['stratum','treatment'],missing=True)
        assert r.rows==60 and r.omitted==0 and r.observed.shape==(3,2)
        assert r.table['Expected < 5'].any()

    def test_limits_and_empty_outputs(self):
        for variables,kw in [([],{}),(['stratum'],{}),(['stratum','predictor'],{}),(['stratum','treatment'],{'max_cells':3})]:
            r=m._analyze(source(),variables,**kw)
            assert r.error and len(m._figure(r).layout.images)==1
        d=source();d.frame.loc[:29,'treatment']=None;d.frame.loc[30:,'stratum']=None
        assert 'No observations' in m._analyze(d,['stratum','treatment']).error or 'eligible' in m._analyze(d,['stratum','treatment']).error

    @pytest.mark.parametrize('area',['observed','expected'])
    def test_recursive_tile_areas(self,area):
        r=m._analyze(source(),['stratum','treatment','sensitive'])
        values=getattr(r,area)
        for index,(x0,y0,x1,y1) in m._rectangles(values).items():
            assert (x1-x0)*(y1-y0)==pytest.approx(values[index]/values.sum())
        f=m._figure(r,area=area,full_screen=True)
        assert f.layout.showlegend and f.layout.modebar.orientation=='v'
        assert f.layout.xaxis.title.text == 'stratum → sensitive'
        assert f.layout.yaxis.title.text == 'treatment'
        assert f.layout.xaxis.visible and not f.layout.xaxis.showticklabels
        assert not m._figure(r).layout.xaxis.visible
        assert not m._figure(r).layout.showlegend
        assert not m._figure(r,shade=False,full_screen=True).layout.showlegend

    def test_escaped_hover_and_labels(self):
        d=source();d.frame['stratum']=d.frame.stratum.replace('a','<b>')
        f=m._figure(m._analyze(d,['stratum','treatment']))
        assert '&lt;b&gt;' in f.data[0].hovertemplate


def by_id(page,name):
    ns=page.locator('.card').first.get_attribute('id').partition('-')[0]
    return page.locator(f'#{ns}-{name}')


@pytest.mark.ui
def test_restored_mosaic_table_and_full_screen_legend(page,app):
    page.goto(app.url)
    expect(by_id(page,'Status')).to_contain_text('8 intersections',timeout=60000)
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')
    chart=by_id(page,'Mosaic').locator('.js-plotly-plot')
    expect(chart).to_be_visible()
    assert chart.evaluate('el=>el.layout.showlegend') is False
    assert 'expected-counts' in chart.evaluate('el=>el.layout.title.text')
    assert chart.evaluate('el=>el.layout.modebar.orientation')=='v'
    tile=chart.locator('.scatterlayer .trace').first.locator('.js-fill')
    bounds=tile.bounding_box()
    page.mouse.move(bounds['x']+10,bounds['y']+10)
    expect(chart.locator('.hoverlayer')).to_contain_text('Expected:')
    expect(chart.locator('.hoverlayer')).to_contain_text('Pearson residual:')
    by_id(page,'ExpandButton').click(force=True)
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.showlegend === true")
    expect(chart.locator('.xtitle')).to_have_text('Region → Group')
    expect(chart.locator('.ytitle')).to_have_text('Treatment')
    by_id(page,'FlipButton').click(force=True)
    expect(by_id(page,'Table')).to_contain_text('Pearson residual',timeout=15000)
    expect(by_id(page,'Table')).to_contain_text('Strongly under-represented')


@pytest.mark.ui
def test_selection_clear_and_area_switch(page,app):
    page.goto(app.url)
    expect(by_id(page,'Status')).to_contain_text('8 intersections',timeout=60000)
    page.locator('.card').first.hover()
    page.locator('.card').first.locator('button.collapse-toggle').click()
    by_id(page,'Area').locator('input[value="observed"]').check()
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.title?.text?.includes('observed-counts')")
    selector=controller.InputSelectize(page,by_id(page,'Variables').get_attribute('id'))
    selector.set(['Region','Treatment'])
    expect(by_id(page,'Status')).to_contain_text('4 intersections',timeout=30000)
    selector.set([])
    expect(by_id(page,'Status')).to_contain_text('Select between two and five',timeout=30000)
    expect(by_id(page,'PassThrough')).to_contain_text('unchanged=True')
