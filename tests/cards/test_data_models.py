import json

import numpy as np
import pandas as pd
import pytest
from cards.data_models import _analyze, _table
from estimator_catalogue import load_catalogue
from modelling_context import ModellingContext
from proxy_data import proxy_data
from roles import RoleMap
from shiny.pytest import create_app_fixture

restored_app = create_app_fixture(app='../scenarios/data_models_restore.py', scope='function')


def source():
    return proxy_data(pd.DataFrame({'x': [1., np.nan, 3.], 'cat': pd.Categorical(['a', 'b', 'a']),
                                   'y': [1., 2., 3.], 'id': [1, 2, 3]}),
                      _roles=RoleMap.from_primitive({'predictor': ['x', 'cat'], 'target': ['y'], 'identifier': ['id']}))


@pytest.mark.unit
@pytest.mark.parametrize('task', ['regressor', 'classifier'])
def test_benchmark_models_never_appear_in_candidates(task):
    from estimator_catalogue import assess_profiles
    catalogue = load_catalogue()
    profile = next(p for p in catalogue['profiles']
                   if p['estimator'].startswith('sklearn.dummy.') and p['task'] == task)
    rows = assess_profiles(catalogue, task=task,
                           package_versions=profile['inspected_dependencies'])
    baseline = next(row for row in rows if row['id'] == profile['id'])
    assert baseline['purpose'] == 'benchmark_only'
    assert baseline['evidence'] == 'probe_passed'
    assert not baseline['selectable']
    for show_all in (False, True):
        for installed_only in (False, True):
            assert baseline['estimator'] not in set(_table({'rows': rows}, show_all=show_all,
                                                    installed_only=installed_only).ID)


@pytest.mark.unit
def test_mixed_missing_matching_and_passthrough():
    data = source()
    before = data.clone()
    result = _analyze(data, load_catalogue(), {})
    assert result['task'] == 'regression'
    assert result['summary']['predictors'] == 2
    table = _table(result)
    assert table.empty  # Internal-CV recipes also require preparation for this input.
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
    expect(by_id('Table').get_by_text('Pred throughput', exact=True)).to_be_attached()
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
    expect(by_id('RecipeDetails')).to_contain_text('Broader method: sklearn.linear_model.ElasticNet')
    expect(by_id('RecipeDetails')).to_contain_text('dedicated Ridge solver')
    by_id('FlipButton').click()
    card.hover()
    card.locator('button.collapse-toggle').click()
    by_id('ShowAll').check()
    expect(by_id('Table')).to_contain_text('Preprocessing:', timeout=15000)
    card.hover()
    by_id('FlipButton').click()
    expect(by_id('Notes')).to_have_count(0)
    expect(by_id('RecipeDetails')).to_contain_text('Catalogue')


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


@pytest.mark.unit
def test_superseded_ready_recipe_hidden_and_restored_without_changing_evidence():
    data = proxy_data(pd.DataFrame({'x': [1., 2., 3.], 'y': [2., 4., 6.]}),
                      _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    result = _analyze(data, load_catalogue(), {})
    ridge = next(r for r in result['rows'] if r['estimator'] == 'sklearn.linear_model.Ridge')
    assert ridge['compatibility'] == 'compatible'
    assert ridge['evidence'] == 'probe_passed'
    assert ridge['input_assessment']['actions'] == []
    assert ridge['estimator'] not in set(_table(result).ID)
    restored = _table(result, show_all=True).set_index('ID').loc[ridge['estimator']]
    assert 'Superseded in catalogue by sklearn.linear' in restored.Readiness
    assert restored.Evidence == 'probe passed'
    assert _table(result, show_all=True, installed_only=True).empty


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


@pytest.mark.ui
def test_recipe_panel_compact_and_fullscreen_layout(page, app):
    page.set_viewport_size({'width': 1100, 'height': 850})
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id('Table')).to_contain_text('Meets requirements', timeout=30000)
    by_id('Table').locator('tbody tr').first.click()
    card = page.locator('.card').first
    card.hover()
    by_id('FlipButton').click()
    panel = page.locator('.model-recipe')
    expect(panel).to_be_visible()
    expect(panel.locator('.model-recipe-title')).not_to_be_empty()
    expect(by_id('Notes')).to_have_count(0)
    def assert_layout():
        assert panel.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        title = panel.locator('.model-recipe-title').bounding_box()
        path = panel.locator('.model-recipe-path').bounding_box()
        grid = panel.locator('.model-recipe-grid').bounding_box()
        header = panel.locator('.model-recipe-header').bounding_box()
        assert title['y'] + title['height'] <= path['y'] + 1
        assert header['y'] + header['height'] <= grid['y'] + 1
        footer = panel.locator('.model-recipe-footer')
        footer.scroll_into_view_if_needed()
        expect(footer).to_be_in_viewport()
    page.wait_for_timeout(1500)  # shared card flip animation
    by_id('Back').evaluate('(el) => { el.scrollTop = 0; }')
    panel.locator('.model-recipe-title').hover()
    page.mouse.wheel(0, 350)
    page.wait_for_function("document.querySelector('[id$=\"-Back\"]').scrollTop > 0")
    assert_layout()
    by_id('Back').evaluate('(el) => { el.scrollTop = 0; }')
    page.screenshot(path='/tmp/model-recipe-compact.png')
    card.hover()
    by_id('ExpandButton').click(force=True)
    expect(card).to_have_class(__import__('re').compile('fullscreen-active'))
    page.wait_for_timeout(600)
    bounds = panel.bounding_box()
    face = by_id('Back').bounding_box()
    assert bounds['width'] >= face['width'] * .95
    assert_layout()
    by_id('Back').evaluate('(el) => { el.scrollTop = 0; }')
    page.screenshot(path='/tmp/model-recipe-fullscreen.png')
    page.set_viewport_size({'width': 430, 'height': 850})
    assert_layout()


@pytest.mark.unit
def test_method_table_groups_recipes_without_merging_evidence():
    from copy import deepcopy
    data = proxy_data(pd.DataFrame({'x': [1., 2., 3.], 'y': [2., 4., 6.]}),
                      _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    result = _analyze(data, load_catalogue(), {})
    before = deepcopy(result['rows'])
    table = _table(result, show_all=True)
    method = 'sklearn.ensemble.RandomForestRegressor'
    assert table.ID.is_unique
    row = table.set_index('ID').loc[method]
    assert row.Recipes == 2
    from cards.data_models import _method_groups
    recipes = _method_groups(result, show_all=True)[method]
    assert list(table.columns) == [
        'Model', 'Core mechanism', 'Composition', 'Fitting criterion', 'Transparency',
        'Learning strategy', 'Prediction outputs', 'Pred throughput', 'Scale', 'Input requirements',
        'Implicit resampling', 'Readiness', 'Evidence', 'Package', 'Installation', 'Recipes', 'ID']
    assert row.Evidence == recipes[0]['evidence'].replace('_', ' ')
    assert row['Input requirements'] == recipes[0]['input_assessment']['requirements'].replace('_', ' ')
    assert result['rows'] == before
    # Filtering happens per recipe before grouping; a hidden recipe cannot
    # contribute evidence or capabilities to the visible method summary.
    recipes[0]['availability'] = 'not_installed'
    recipes[1]['availability'] = 'version_matches'
    filtered = _method_groups(result, show_all=True, installed_only=True)[method]
    assert len(filtered) == 1 and filtered[0] is recipes[1]


@pytest.mark.ui
def test_multiple_recipes_for_one_method(page, app):
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id('Table')).to_contain_text('Meets requirements', timeout=30000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    by_id('ShowAll').check()
    card.locator('button.collapse-toggle').click()
    model = by_id('Table').get_by_text('Random Forest Regressor', exact=True)
    expect(model).to_have_count(1)
    model.click()
    card.hover()
    by_id('FlipButton').click()
    recipes = by_id('RecipeDetails').locator('.model-method-recipe')
    expect(recipes).to_have_count(2)
    expect(recipes.first).to_have_attribute('open', '')
    recipes.nth(1).locator('summary').click()
    expect(recipes.nth(1).locator('.model-recipe')).to_be_visible()
    expect(by_id('RecipeDetails')).to_contain_text('numeric missing')
    expect(by_id('RecipeDetails')).to_contain_text('mixed missing via preprocessing')


@pytest.mark.unit
def test_characteristics_compare_recipes_without_pooling_capabilities():
    from copy import deepcopy

    from cards.data_models import _recipe_details
    from model_taxonomy import LABELS
    result = _analyze(source(), load_catalogue(), {})
    table = _table(result, show_all=True)
    assert set(LABELS.values()) <= set(table.columns)
    assert 'Capacity growth' not in table.columns
    method = 'sklearn.ensemble.RandomForestRegressor'
    recipes = [deepcopy(r) for r in result['rows'] if r['estimator'] == method]
    assert len(recipes) >= 2
    assert _table({'rows': recipes}, show_all=True).iloc[0]['Core mechanism'] == 'Tree'
    recipes[1]['profile']['taxonomy']['core_mechanism']['values'] = ['generalized_linear']
    recipes[1]['profile']['outputs'] = ['point_prediction', 'predictive_standard_deviation']
    summary = _table({'rows': recipes}, show_all=True).iloc[0]
    assert summary['Core mechanism'] == 'Varies by recipe'
    assert summary['Prediction outputs'] == 'Varies by recipe'
    assert summary['Transparency'] == 'Usually opaque'
    details = str(_recipe_details(recipes[0], 'test'))
    assert 'Model characteristics' in details
    assert 'expert guidance' in details
    assert 'stage 2' in details
    assert 'Usually opaque' in details


@pytest.mark.ui
def test_characteristic_sort_preserves_selected_recipe_and_hides_front(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Meets requirements', timeout=30000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    page.locator('[id$="-ShowAll"]').check()
    card.locator('button.collapse-toggle').click()
    # Column filters are disabled; exercise the supported sorting control.
    expect(table.locator('thead tr.filters input')).to_have_count(0)
    heading = table.locator('thead').get_by_text('Learning strategy', exact=True)
    heading.click()
    heading.click()  # Descending places Lazy before Eager.
    selected = table.get_by_text('K Neighbors Regressor', exact=True)
    expect(selected).to_be_visible()
    leading_rows = [table.locator('tbody tr').nth(i).text_content() for i in range(2)]
    # The virtualized DOM contains only a subset of the methods.
    _ = len(_table(_analyze(source(), load_catalogue(), {}), show_all=True))
    expect(page.locator('[id$="-Status"]')).to_have_text(
        'No estimator committed. Select methods and Commit.')
    selected.click()
    card.hover()
    page.locator('[id$="-FlipButton"]').click()
    details = page.locator('[id$="-RecipeDetails"]')
    expect(details).to_contain_text('sklearn.neighbors.KNeighborsRegressor')
    expect(details).to_contain_text('Model characteristics')
    expect(details).to_contain_text('Lazy')
    expect(card.locator('.flippable > .front')).to_have_css('visibility', 'hidden')
    page.locator('[id$="-FlipButton"]').click()
    expect(card.locator('.flippable > .front')).to_have_css('visibility', 'visible')
    # Flipping can change the number of virtualized rows; compare the leading
    # sorted records rather than the entire mounted viewport.
    for i, text in enumerate(leading_rows):
        expect(table.locator('tbody tr').nth(i)).to_have_text(text)
    # Reopening the details also verifies selection survived the round trip.
    page.locator('[id$="-FlipButton"]').click()
    expect(details).to_contain_text('sklearn.neighbors.KNeighborsRegressor')
    expect(page.locator('[id$="-ExportState"]')).to_have_text('unchanged=True;pipeline=False')


@pytest.mark.ui
def test_gp_prediction_spread_replaces_formulation(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Meets requirements', timeout=30000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    page.locator('[id$="-ShowAll"]').check()
    card.locator('button.collapse-toggle').click()
    expect(table.locator('thead')).not_to_contain_text('Formulation')
    row = table.locator('tbody tr').filter(has_text='Gaussian Process Regressor')
    expect(row).to_contain_text('Point prediction & spread')
    row.get_by_text('Gaussian Process Regressor', exact=True).click()
    card.hover()
    page.locator('[id$="-FlipButton"]').click()
    details = page.locator('[id$="-RecipeDetails"]')
    expect(details).to_contain_text('Point prediction & spread')
    expect(details).to_contain_text('Predictive standard deviation')
    expect(details).to_contain_text('return_std=True')
    expect(details).not_to_contain_text('Discriminative')
    expect(details).not_to_contain_text('Formulation')


@pytest.mark.unit
def test_learning_strategy_column_and_recipe_disagreement():
    from copy import deepcopy

    from cards.data_models import _recipe_details
    result = _analyze(source(), load_catalogue(), {})
    table = _table(result, show_all=True).set_index('ID')
    assert table.loc['sklearn.neighbors.KNeighborsRegressor', 'Learning strategy'] == 'Lazy'
    assert table.loc['sklearn.gaussian_process.GaussianProcessRegressor', 'Learning strategy'] == 'Eager'
    rows = [deepcopy(r) for r in result['rows'] if r['estimator'] == 'sklearn.ensemble.RandomForestRegressor']
    rows[1]['profile']['taxonomy']['learning_strategy']['values'] = ['hybrid']
    assert _table({'rows': rows}, show_all=True).iloc[0]['Learning strategy'] == 'Varies by recipe'
    assert 'Hybrid' in str(_recipe_details(rows[1], 'test'))


@pytest.mark.ui
def test_learning_strategy_column_and_details(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Meets requirements', timeout=30000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    page.locator('[id$="-ShowAll"]').check()
    card.locator('button.collapse-toggle').click()
    heading = table.locator('thead').get_by_text('Learning strategy', exact=True)
    heading.click()
    heading.click()  # Descending order places Lazy before Eager.
    row = table.locator('tbody tr').filter(has_text='K Neighbors Regressor')
    expect(row).to_contain_text('Lazy')
    table.get_by_text('K Neighbors Regressor', exact=True).click()
    card.hover()
    page.locator('[id$="-FlipButton"]').click()
    details = page.locator('[id$="-RecipeDetails"]')
    expect(details).to_contain_text('Learning strategy')
    expect(details).to_contain_text('Lazy')
    expect(details).to_contain_text('query-specific neighbor retrieval')


@pytest.mark.ui
def test_commit_selected_model_exports_pipeline_and_status(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Hist Gradient Boosting Regressor', timeout=30000)
    table.get_by_text('Hist Gradient Boosting Regressor', exact=True).click()
    expect(page.locator('[id$="-ExportState"]')).to_have_text('unchanged=True;pipeline=False')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_have_text('Committed method: Hist Gradient Boosting Regressor')
    expect(page.locator('[id$="-ExportState"]')).to_have_text('unchanged=False;pipeline=True')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: Hist Gradient Boosting Regressor')


@pytest.mark.ui
def test_method_choice_can_change_after_selection_and_commit(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    first = table.get_by_text('Hist Gradient Boosting Regressor', exact=True)
    second = table.get_by_text('LGBM Regressor', exact=True)
    expect(first).to_be_visible(timeout=30000)
    first.click()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: Hist Gradient Boosting Regressor')
    second.click()
    expect(second.locator('..')).to_have_attribute('aria-selected', 'true')
    expect(page.locator('[id$="-Status"]')).to_contain_text('Selected method: LGBM Regressor; press Commit')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: LGBM Regressor')
    first.click()
    expect(first.locator('..')).to_have_attribute('aria-selected', 'true')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: Hist Gradient Boosting Regressor')


@pytest.mark.ui
def test_json_restores_selected_row_separately_from_committed_model(page, app, request, monkeypatch, tmp_path):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    first = table.get_by_text('Hist Gradient Boosting Regressor', exact=True)
    second = table.get_by_text('LGBM Regressor', exact=True)
    expect(first).to_be_visible(timeout=30000)
    first.click()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: Hist Gradient Boosting Regressor')
    table.locator('thead').get_by_text('Model', exact=True).click()
    second.click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Selected method: LGBM Regressor')
    with page.expect_download() as download:
        page.get_by_role('link', name='Configuration JSON', exact=True).click()
    saved = tmp_path / 'model.json'
    download.value.save_as(saved)
    choice = json.loads(saved.read_text())['model_choice']
    assert choice['selected_method'] == 'lightgbm.LGBMRegressor'
    assert choice['selected_recipe'] != choice['committed_recipe']
    monkeypatch.setenv('PYTHAGORAS_TEST_MODEL_CONFIG', str(saved))
    resumed = request.getfixturevalue('restored_app')
    page.goto(resumed.url)
    restored_row = page.locator('[id$="-Table"]').get_by_text('LGBM Regressor', exact=True)
    expect(restored_row.locator('..')).to_have_attribute('aria-selected', 'true', timeout=30000)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: Hist Gradient Boosting Regressor')
    expect(page.locator('[id$="-Status"]')).to_contain_text('Selected method: LGBM Regressor')
    expect(page.locator('[id$="-ExportState"]')).to_have_text('unchanged=False;pipeline=True')
    # Restoring a row must not keep forcing it after a subsequent user choice.
    page.locator('[id$="-Table"]').get_by_text('Hist Gradient Boosting Regressor', exact=True).click()
    expect(page.locator('[id$="-Table"]').get_by_text('Hist Gradient Boosting Regressor', exact=True).locator('..')).to_have_attribute('aria-selected', 'true')
    restored_row.click()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed method: LGBM Regressor')
    expect(page.locator('[id$="-ExportState"]')).to_have_text('unchanged=False;pipeline=True')


@pytest.mark.ui
def test_candidates_table_grows_to_fullscreen_height(page, app):
    page.set_viewport_size({'width': 1400, 'height': 1000})
    page.goto(app.url)
    table = page.locator('.model-candidates-table')
    expect(table).to_contain_text('Meets requirements', timeout=30000)
    before = table.bounding_box()['height']
    card = page.locator('.card').first
    card.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(card).to_have_class(__import__('re').compile('fullscreen-active'))
    page.wait_for_timeout(600)
    after = table.bounding_box()['height']
    assert after > before + 200
    grid = table.locator('shiny-data-frame')
    assert grid.bounding_box()['height'] > after * .9


@pytest.mark.unit
def test_implicit_resampling_visibility_commit_guard_and_upstream_switch():
    from evaluation_plan import build_plan
    from model_commit import build_model
    catalogue = load_catalogue()
    methods = {'sklearn.linear_model.ElasticNet', 'sklearn.linear_model.ElasticNetCV'}
    for entry in catalogue['estimators']:
        if entry['id'] in methods:
            entry.pop('superseded_by', None)
    data = proxy_data(pd.DataFrame({'x': np.arange(20.), 'y': np.arange(20.) * 2}),
                      _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    profile = next(p for p in catalogue['profiles'] if p['estimator'] == 'sklearn.linear_model.ElasticNetCV')
    versions = dict(profile['inspected_dependencies'])
    upstream = data.with_evaluation_plan(build_plan(data, strategy='kfold', objective='independent', folds=3), card='obs_resample')
    for value, allowed, excluded in (
            (data, 'sklearn.linear_model.ElasticNetCV', 'sklearn.linear_model.ElasticNet'),
            (upstream, 'sklearn.linear_model.ElasticNet', 'sklearn.linear_model.ElasticNetCV')):
        result = _analyze(value, catalogue, versions)
        visible = _table(result)
        assert allowed in set(visible.ID)
        assert excluded not in set(visible.ID)
        assert excluded in set(_table(result, show_all=True).ID)
        row = next(r for r in result['rows'] if r['estimator'] == excluded)
        assert not row['selectable']
        assert row['compatibility'] == 'compatible'  # Input capability evidence stays intact.
        with pytest.raises(ValueError, match='resampling'):
            build_model(row, value)
        assert {label for cell in _table(result, show_all=True)['Implicit resampling']
                for label in cell.split(', ')} <= {'Yes','No','Conditional','Unreviewed'}


@pytest.mark.unit
def test_non_suffix_internal_calibration_bootstrap_and_unknown():
    from copy import deepcopy

    from estimator_catalogue import implicit_resampling
    catalogue = load_catalogue()
    def profile(name): return next(p for p in catalogue['profiles'] if p['estimator'] == name)
    assert implicit_resampling(profile('sklearn.svm.NuSVC'))['status'] == 'yes'
    assert implicit_resampling(profile('sklearn.ensemble.RandomForestRegressor'))['status'] == 'no'
    changed = deepcopy(profile('sklearn.svm.NuSVC'))
    changed['construction']['parameters']['probability'] = False
    assert implicit_resampling(changed)['status'] == 'unknown'
    changed['id'] = 'new.Method:unreviewed'
    assert implicit_resampling(changed)['status'] == 'unknown'


@pytest.mark.unit
@pytest.mark.parametrize('count,expected', [(10000, 'No'), (10001, 'Yes')])
def test_automatic_validation_split_boundary(count, expected):
    from evaluation_plan import build_plan
    data = proxy_data(pd.DataFrame({'x': np.arange(count, dtype=float), 'y': np.arange(count, dtype=float)}),
                      _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    catalogue = load_catalogue()
    def histogram(result):
        return next(r for r in result['rows'] if r['estimator'] == 'sklearn.ensemble.HistGradientBoostingRegressor')
    row = histogram(_analyze(data, catalogue, {}))
    assert row['resampling_label'] == expected
    upstream = data.with_evaluation_plan(build_plan(data, strategy='kfold', objective='independent', folds=3), card='obs_resample')
    row = histogram(_analyze(upstream, catalogue, {}))
    assert row['resampling_label'] == ('No' if count == 10000 else 'Conditional')
    assert bool(row.get('resampling_exclusion')) == (count > 10000)


@pytest.mark.unit
def test_prediction_throughput_is_ranked_recipe_guidance_and_preserves_eligibility():
    result = _analyze(source(), load_catalogue(), {})
    before = [(r['id'], r['compatibility'], r['evidence'], r['selectable']) for r in result['rows']]
    table = _table(result, show_all=True)
    assert set(table['Pred throughput']) <= {'Low', 'Medium', 'High'}
    from cards.data_models import _method_groups
    for method, recipes in _method_groups(result, show_all=True).items():
        assert table.set_index('ID').loc[method, 'Pred throughput'] == recipes[0]['prediction_throughput']['rating']
    assert [(r['id'], r['compatibility'], r['evidence'], r['selectable']) for r in result['rows']] == before


@pytest.mark.ui
def test_change_committed_method_after_restoring_same_selected_and_committed(page,app,request,monkeypatch,tmp_path):
    page.goto(app.url)
    table=page.locator('[id$="-Table"]')
    status=page.locator('[id$="-Status"]')
    expect(table).to_contain_text('Hist Gradient Boosting Regressor',timeout=30000)
    table.get_by_text('Hist Gradient Boosting Regressor',exact=True).click()
    page.locator('[id$="-Commit"]').click()
    expect(status).to_have_text('Committed method: Hist Gradient Boosting Regressor')
    with page.expect_download() as download:
        page.get_by_role('link',name='Configuration JSON',exact=True).click()
    saved=tmp_path/'same-choice.json'
    download.value.save_as(saved)
    choice=json.loads(saved.read_text())['model_choice']
    assert choice['selected_recipe']==choice['committed_recipe']
    monkeypatch.setenv('PYTHAGORAS_TEST_MODEL_CONFIG',str(saved))
    resumed=request.getfixturevalue('restored_app')
    page.goto(resumed.url)
    table=page.locator('[id$="-Table"]')
    status=page.locator('[id$="-Status"]')
    expect(status).to_have_text('Committed method: Hist Gradient Boosting Regressor',timeout=30000)
    table.get_by_text('LGBM Regressor',exact=True).click()
    expect(status).to_contain_text('Selected method: LGBM Regressor; press Commit')
    page.locator('[id$="-Commit"]').click()
    expect(status).to_have_text('Committed method: LGBM Regressor')
    with page.expect_download() as download:
        page.get_by_role('link',name='Configuration JSON',exact=True).click()
    changed=tmp_path/'changed-choice.json'
    download.value.save_as(changed)
    choice=json.loads(changed.read_text())['model_choice']
    assert choice['selected_recipe']==choice['committed_recipe']
    assert choice['committed_recipe'].startswith('lightgbm.LGBMRegressor:')


@pytest.mark.ui
def test_restored_gaussian_process_can_be_replaced_by_catboost(page,request,monkeypatch,tmp_path):
    monkeypatch.setenv('PYTHAGORAS_TEST_MODEL_NUMERIC','1')
    initial=request.getfixturevalue('app')
    page.goto(initial.url)
    table=page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Gaussian Process Regressor',timeout=30000)
    table.get_by_text('Gaussian Process Regressor',exact=True).click()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_have_text('Committed method: Gaussian Process Regressor')
    expect(page.locator('[id$="-ExportedMethod"]')).to_have_text('GaussianProcessRegressor')
    with page.expect_download() as download:
        page.get_by_role('link',name='Configuration JSON',exact=True).click()
    saved=tmp_path/'gp-committed.json'
    download.value.save_as(saved)
    monkeypatch.setenv('PYTHAGORAS_TEST_MODEL_CONFIG',str(saved))
    resumed=request.getfixturevalue('restored_app')
    page.goto(resumed.url)
    expect(page.locator('[id$="-Status"]')).to_have_text('Committed method: Gaussian Process Regressor',timeout=30000)
    page.locator('[id$="-Table"]').get_by_text('Cat Boost Regressor',exact=True).click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Selected method: Cat Boost Regressor')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_have_text('Committed method: Cat Boost Regressor')
    expect(page.locator('[id$="-ExportedMethod"]')).to_contain_text('CatBoostRegressor')
    with page.expect_download() as download:
        page.get_by_role('link',name='Configuration JSON',exact=True).click()
    changed=tmp_path/'catboost-committed.json'
    download.value.save_as(changed)
    choice=json.loads(changed.read_text())['model_choice']
    assert choice['selected_recipe']==choice['committed_recipe']
    assert choice['committed_recipe'].startswith('catboost.CatBoostRegressor:')


@pytest.mark.unit
def test_skglm_elastic_net_alternative_is_hidden_but_retained_with_reason():
    from cards.data_models import _recipe_details
    from estimator_catalogue import assess_profiles
    catalogue = load_catalogue()
    rows = assess_profiles(catalogue, task='regressor')
    row = next(r for r in rows if r['estimator'] == 'skglm.ElasticNet')
    reason = 'Alternative implementation of sklearn Elastic Net with the same statistical objective; retained for solver-performance comparisons.'
    assert row['superseded_by']['relationship'] == 'alternative_implementation'
    assert 'skglm.ElasticNet' not in set(_table({'rows': rows}).ID)
    shown = _table({'rows': rows}, show_all=True).set_index('ID')
    assert reason in shown.loc['skglm.ElasticNet', 'Readiness']
    assert 'sklearn.linear_model.ElasticNet' in set(_table({'rows': rows}).ID)
    details = str(_recipe_details(row, catalogue['catalogue_version']))
    assert reason in details
    assert 'Representative method: sklearn.linear_model.ElasticNet' in details


@pytest.mark.ui
def test_commit_multiple_methods_and_restore_candidate_set(page, app, request, monkeypatch, tmp_path):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    first = table.get_by_text('Hist Gradient Boosting Regressor', exact=True)
    second = table.get_by_text('LGBM Regressor', exact=True)
    expect(first).to_be_visible(timeout=30000)
    first.click()
    second.click(modifiers=['ControlOrMeta'])
    expect(first.locator('..')).to_have_attribute('aria-selected', 'true')
    expect(second.locator('..')).to_have_attribute('aria-selected', 'true')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed methods:')
    expect(page.locator('[id$="-ExportedCandidates"]')).to_contain_text('Hist Gradient Boosting Regressor')
    expect(page.locator('[id$="-ExportedCandidates"]')).to_contain_text('LGBM Regressor')
    with page.expect_download() as download:
        page.get_by_role('link', name='Configuration JSON', exact=True).click()
    path = tmp_path / 'multiple-models.json'
    download.value.save_as(path)
    state = json.loads(path.read_text())['model_choice']
    assert len(state['committed_recipes']) == 2
    assert len(state['selected_methods']) == 2
    monkeypatch.setenv('PYTHAGORAS_TEST_MODEL_CONFIG', str(path))
    resumed = request.getfixturevalue('restored_app')
    page.goto(resumed.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('Committed methods:', timeout=30000)
    expect(page.locator('[id$="-Table"] tbody tr[aria-selected="true"]')).to_have_count(2)
