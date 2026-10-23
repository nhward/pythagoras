import numpy as np
import pandas as pd
import pytest
from cards.data_models import _analyze, _table
from estimator_catalogue import load_catalogue
from modelling_context import ModellingContext
from proxy_data import proxy_data
from roles import RoleMap


def source():
    return proxy_data(pd.DataFrame({'x': [1., np.nan, 3.], 'cat': pd.Categorical(['a', 'b', 'a']),
                                   'y': [1., 2., 3.], 'id': [1, 2, 3]}),
                      _roles=RoleMap.from_primitive({'predictor': ['x', 'cat'], 'target': ['y'], 'identifier': ['id']}))


@pytest.mark.unit
def test_mixed_missing_matching_and_passthrough():
    data = source()
    before = data.clone()
    result = _analyze(data, load_catalogue(), {})
    assert result['task'] == 'regression'
    assert result['summary']['predictors'] == 2
    table = _table(result)
    assert not table.empty
    assert not any('Ridge' in name for name in table.Model)
    assert any('Ridge' in name for name in _table(result, show_all=True).Model)
    assert all(table.Readiness == 'Meets requirements')
    assert _table(result, installed_only=True).empty
    assert len(_table(result, show_all=True)) > len(table)
    assert data.equals(before)
    assert 'Evaluation not ready' in result['notes'][0]


@pytest.mark.unit
def test_empty_unsupported_infinity_and_unresolved():
    data = source()
    data.frame.loc[0, 'x'] = np.inf
    assert _table(_analyze(data, load_catalogue(), {})).empty
    data = proxy_data(pd.DataFrame({'x': pd.to_datetime(['2020-01-01']), 'y': [1.]}),
                      _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    assert _table(_analyze(data, load_catalogue(), {})).empty
    assert _analyze(proxy_data(pd.DataFrame({'x': [1.]})), load_catalogue(), {})['task'] == 'unresolved'
    assert _table(_analyze(proxy_data(source().frame.iloc[:0], _roles=source().role_map), load_catalogue(), {})).empty


@pytest.mark.unit
def test_survival_and_temporal_requirements():
    data = source().with_modelling_context(ModellingContext(task='survival', duration='y', event='id',
                                                           event_value='1', censored_value='2'))
    assert _table(_analyze(data, load_catalogue(), {})).empty  # invalid event code 3
    data.clean_frame.loc[2, 'id'] = 1
    result = _analyze(data, load_catalogue(), {})
    assert 'longitudinal' in result['structure']
    assert all(r['compatibility'] != 'compatible' for r in result['rows'])
    result = _analyze(source(), load_catalogue(), {}, forecasting=True)
    assert all('Requires time_index' in r['reasons'] for r in result['rows'])


from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/data_models.py', scope='function')


@pytest.mark.ui
def test_models_table_filters_details_and_pass_through(page, app):
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id('Summary')).to_contain_text('Regression', timeout=30000)
    expect(by_id('Table')).to_contain_text('Meets requirements', timeout=30000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    by_id('ShowAll').check()
    card.locator('button.collapse-toggle').click()
    expect(by_id('Table')).to_contain_text('Ridge', timeout=30000)
    expect(by_id('ExportState')).to_have_text('unchanged=True;pipeline=False')
    card = page.locator('.card').first
    by_id('Table').get_by_text('Ridge', exact=True).click()
    card.hover()
    by_id('FlipButton').click()
    expect(by_id('RecipeDetails')).to_contain_text('sklearn.linear_model.Ridge')
    expect(by_id('RecipeDetails')).to_contain_text('impute_onehot')
    by_id('FlipButton').click()
    card.hover()
    card.locator('button.collapse-toggle').click()
    by_id('ShowAll').check()
    expect(by_id('Table')).to_contain_text('Preprocessing:', timeout=15000)
    card.hover()
    by_id('FlipButton').click()
    expect(by_id('Notes')).to_contain_text('Evaluation not ready')
    expect(by_id('Notes')).to_contain_text('Catalogue')


@pytest.mark.unit
def test_weights_and_valid_survival_are_explained():
    data = source()
    data.frame['weights'] = [1., -1., 2.]
    data = proxy_data(data.frame, _roles=RoleMap.from_primitive({
        'predictor': ['x'], 'target': ['y'], 'weighting': ['weights']}))
    result = _analyze(data, load_catalogue(), {}, use_weights=True)
    assert 'unweighted requirements' in result['notes'][0]
    assert 'sample_weight_fit' not in result['summary']['required']
    data = source().with_modelling_context(ModellingContext(task='survival', duration='y', event='cat',
        event_value='a', censored_value='b'))
    result = _analyze(data, load_catalogue(), {})
    assert _table(result).empty
    table = _table(result, show_all=True)
    assert not table.empty
    assert set(table.Evidence) == {'documentation only'}
    assert all(not r['selectable'] for r in result['rows'])


@pytest.mark.unit
def test_readiness_combines_preparation_and_broad_requirements():
    from cards.data_models import _readiness
    row = {'input_assessment': {'actions': [], 'readiness': 'Meets reviewed input requirements'}, 'reasons': []}
    assert _readiness(row) == 'Meets requirements'
    row['reasons'] = ['sample_weight_fit: outside this recipe', 'temporal structure: unreviewed']
    row['input_assessment']['actions'] = ['Resolve missing predictor values']
    value = _readiness(row)
    assert 'Does not support weighting' in value
    assert 'Untested: temporal structure' in value
    assert 'Preprocessing: Resolve missing predictor values' in value


@pytest.mark.ui
def test_selected_rows_override_striping_and_hover(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Meets requirements', timeout=30000)
    rows = table.locator('tbody tr')
    for index in (0, 1):
        row = rows.nth(index)
        cell = row.locator('td').first
        cell.click()
        expect(row).to_have_attribute('aria-selected', 'true')
        for hovered in (False, True):
            if hovered:
                cell.hover()
            else:
                page.locator('[id$="-Summary"]').hover()
            assert cell.evaluate('''el => {
                const style = getComputedStyle(el);
                const probe = document.createElement('span');
                probe.style.backgroundColor = style.getPropertyValue('--shiny-datagrid-grid-body-selected-bgcolor');
                el.appendChild(probe);
                const expected = getComputedStyle(probe).backgroundColor;
                probe.remove();
                return style.backgroundColor === expected;
            }''')


@pytest.mark.unit
@pytest.mark.parametrize('identifier, expected', [
    ('sklearn.ensemble.RandomForestRegressor', ('Random Forest Regressor', 'Ensemble', 'sklearn')),
    ('lightgbm.GBMRegressor', ('GBM Regressor', 'Lightgbm', 'lightgbm')),
    ('sklearn.linear_model.Ridge', ('Ridge', 'Linear model', 'sklearn')),
])
def test_model_display_labels(identifier, expected):
    from cards.data_models import _model_labels
    labels = _model_labels(identifier)
    assert (labels['Model'], labels['Type'], labels['Package']) == expected
