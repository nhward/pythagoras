"""Reviewed rules stay fixed across folds and are editable at tree leaves."""
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from ManualImputationTransformer import ManualImputationTransformer, manual_kind, manual_value, manual_mask
from cards import miss_manual as m
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.pytest import create_app_fixture
from playwright.sync_api import expect

app = create_app_fixture(app='../scenarios/miss_manual.py', scope='function')


def active_tree(page):
    return page.locator('#miss_manual-Variables .tab-pane.active')


def select_variable(page, name):
    page.get_by_role('tab', name=name, exact=True).click()
    expect(page.get_by_role('tab', name=name, exact=True)).to_have_attribute('aria-selected', 'true')


def source():
    frame = pd.DataFrame({'cost': pd.Series([None, None, 20, 40], dtype='Int64'),
                          'basis': pd.Categorical(['FREE', 'FREE', 'PAID', 'PAID']),
                          'outcome': [0, 0, 1, 1]})
    roles = RoleMap()
    roles.set_roles('basis', [Role.IDENTIFIER])
    roles.set_roles('outcome', [Role.TARGET])
    return proxy_data(_df=frame, _roles=roles)


def rule_for(proposal, value=0):
    leaf = next(n for n in proposal['nodes'] if n['leaf'] and n['missing'])
    return {'column': proposal['column'], 'kind': proposal['kind'], 'conditions': leaf['conditions'],
            'value': value, 'extra_level': ''}


@pytest.mark.unit
def test_reviewed_healthcare_rule_survives_clone_pickle_folds_and_pipeline():
    data = source()
    original = data.frame.copy(deep=True)
    proposal = m.propose_tree(data, 'cost')
    rule = rule_for(proposal)
    assert {c['feature']['column'] for c in rule['conditions']} == {'basis'}
    transformer = ManualImputationTransformer([rule])
    train = data.frame.iloc[2:]
    fitted = clone(transformer).fit(train)
    test = data.frame.iloc[:2].copy()
    test.index = [7, 7]
    assert fitted.transform(test).cost.tolist() == [0, 0]
    assert pickle.loads(pickle.dumps(fitted)).transform(test).cost.tolist() == [0, 0]
    result = m.apply_rules(data, [rule])
    assert result.frame.cost.tolist() == [0, 0, 20, 40]
    assert result.frame.cost.dtype == original.cost.dtype
    assert_frame_equal(result.clean_frame, original)
    assert_frame_equal(data.frame, original)
    pipeline = result.pipeline_for_training()
    assert not hasattr(pipeline.steps[-1][1], 'rules_')
    pipeline.fit(train)
    assert pipeline.transform(test).cost.tolist() == [0, 0]
    assert pipeline.steps[-1][1].get_params()['rules'] == [rule]
    json.dumps(rule)


@pytest.mark.unit
def test_target_exclusion_and_skipped_dtypes():
    data = source()
    data.frame['outcome'] = np.nan
    data.frame['ordered'] = pd.Categorical([None, 'a', 'b', 'a'], ordered=True)
    data.frame['complex'] = [[1], None, [2], [3]]
    allowed, skipped = m.eligible_columns(data)
    assert allowed == ['cost']
    assert set(skipped) == {'ordered', 'complex'}
    with pytest.raises(ValueError, match='eligible'):
        m.propose_tree(data, 'outcome')
    rule = rule_for(m.propose_tree(data, 'cost'))
    bad = deepcopy(rule)
    bad['conditions'][0]['feature']['column'] = 'outcome'
    with pytest.raises(ValueError, match='Target'):
        m.apply_rules(data, [bad])


@pytest.mark.unit
@pytest.mark.parametrize('depth', [1, 2, 3])
def test_native_missing_routing_matches_tree_leaves(depth):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({'x': rng.normal(size=100), 'category': pd.Categorical(np.resize(['a', 'b', None], 100))})
    frame.loc[::4, 'x'] = np.nan
    frame['cost'] = np.where(frame.x.isna() | frame.category.eq('a'), np.nan, 10.)
    proposal = m.propose_tree(proxy_data(_df=frame), 'cost', depth)
    json.dumps(proposal, allow_nan=False)
    masks = []
    for leaf in (n for n in proposal['nodes'] if n['leaf']):
        mask = manual_mask(frame, leaf['conditions'])
        assert mask.sum() == leaf['rows']
        assert frame.cost.isna().to_numpy()[mask].sum() == leaf['missing']
        masks.append(mask)
        assert leaf['depth'] <= depth
    assert np.all(np.sum(masks, axis=0) == 1)


@pytest.mark.unit
@pytest.mark.parametrize('dtype,value', [('Int64', 1.5), ('UInt8', -1), ('UInt8', 256), ('Float64', float('inf'))])
def test_invalid_numeric_replacements(dtype, value):
    with pytest.raises(ValueError):
        manual_value(pd.Series([None, 1], dtype=dtype), value)


@pytest.mark.unit
def test_categories_unseen_levels_empty_tree_and_simultaneous_rules():
    frame = pd.DataFrame({'a': pd.Categorical([None, 'known']), 'b': pd.Series([None, 1], dtype='Int64')})
    categorical = {'column': 'a', 'kind': 'categorical', 'conditions': [], 'value': 'Other', 'extra_level': 'Other', 'levels': ['known']}
    condition = {'feature': {'column': 'a', 'kind': 'categorical', 'level': 'Other', 'levels': ['Other', 'known']}, 'threshold': .5, 'missing_left': True, 'left': False}
    numeric = {'column': 'b', 'kind': 'integer', 'conditions': [condition], 'value': 0}
    for rules in ([categorical, numeric], [numeric, categorical]):
        out = ManualImputationTransformer(rules).fit_transform(frame)
        assert out.a.iloc[0] == 'Other'
        assert pd.isna(out.b.iloc[0])  # Does not see the replacement in a.
        assert out.a.cat.categories.tolist() == ['known', 'Other']
    categorical.update(value='known', extra_level='')
    # Reviewed category remains legal when absent from the training partition.
    text = pd.DataFrame({'a': pd.Series([None], dtype='string')})
    assert ManualImputationTransformer([categorical]).fit_transform(text).a.iloc[0] == 'known'
    with pytest.raises(ValueError, match='category'):
        manual_value(frame.a, 'new')
    alone = proxy_data(_df=pd.DataFrame({'cost': pd.Series([None, None], dtype='Int64')}))
    proposal = m.propose_tree(alone, 'cost')
    assert proposal['nodes'][0]['leaf']
    assert m.apply_rules(alone, [rule_for(proposal)]).frame.cost.tolist() == [0, 0]
    assert_frame_equal(m.apply_rules(alone, []).frame, alone.frame)


@pytest.mark.unit
def test_boolean_date_validation_and_conflicts():
    assert manual_value(pd.Series([None, True], dtype='boolean'), 'false') is False
    dates = pd.Series(pd.to_datetime([None, '2025-01-01']))
    assert manual_value(dates, '2025-03-02') == '2025-03-02T00:00:00'
    rule = rule_for(m.propose_tree(source(), 'cost'))
    with pytest.raises(ValueError, match='Overlapping'):
        ManualImputationTransformer([rule, rule]).fit_transform(source().frame)


@pytest.mark.ui
def test_leaf_editors_commit_validation_depth_and_reset(page, app):
    page.goto(app.url)
    status = page.locator('#miss_manual-Status')
    expect(status).to_contain_text('0 committed rules', timeout=60000)
    leaf = active_tree(page).locator('.miss-manual-leaf').filter(has=page.get_by_label('Fill missing values'))
    expect(leaf).to_have_count(1)
    expect(active_tree(page)).to_contain_text('Skipped: ORDERED')
    expect(leaf).to_contain_text('10 missing / 10 rows (100%)')
    leaf.get_by_label('Fill missing values').check()
    leaf.get_by_label('Whole-number replacement').fill('1.5')
    page.locator('#miss_manual-Commit').click()
    expect(page.get_by_text('HEALTHCARE_COST: Enter a whole number.', exact=True)).to_be_visible()
    expect(status).to_contain_text('0 committed rules')
    leaf.get_by_label('Whole-number replacement').fill('0')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('1 committed rule · 10 missing values filled')
    expect(page.locator('#miss_manual-ExportProbe')).to_contain_text('missing=0; first=0; clean_missing=10; steps=identity,miss_manual')
    select_variable(page, 'LABEL')
    expect(active_tree(page)).to_contain_text('Fill missing values')
    select_variable(page, 'HEALTHCARE_COST')
    expect(leaf.get_by_label('Fill missing values')).to_be_checked()
    expect(leaf.get_by_label('Whole-number replacement')).to_have_value('0')
    previous_id = leaf.get_by_label('Fill missing values').get_attribute('id')
    page.locator('#miss_manual-MaxTreeDepth').evaluate("el => { window.jQuery(el).data('ionRangeSlider').update({from: 1}); window.jQuery(el).trigger('change'); }")
    expect(leaf.get_by_label('Fill missing values')).not_to_have_attribute('id', previous_id)
    expect(leaf.get_by_label('Fill missing values')).not_to_be_checked()
    expect(status).to_contain_text('1 committed rule · 10 missing values filled')
    page.locator('#miss_manual-Reset').click()
    expect(status).to_contain_text('0 committed rules · 0 missing values filled')
    expect(leaf.get_by_label('Fill missing values')).not_to_be_checked()
    expect(leaf.get_by_label('Whole-number replacement')).to_have_value('')
    select_variable(page, 'LABEL')
    for checkbox in active_tree(page).get_by_label('Fill missing values').all():
        expect(checkbox).not_to_be_checked()
    select_variable(page, 'HEALTHCARE_COST')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('0 committed rules · 0 missing values filled')


restoring_app = create_app_fixture(app='../scenarios/miss_manual_restoring.py', scope='function')


@pytest.mark.ui
def test_restore_committed_rules_and_leaf_draft(page, restoring_app):
    page.goto(restoring_app.url)
    expect(page.locator('#miss_manual-Status')).to_contain_text('1 committed rule · 10 missing values filled', timeout=60000)
    expect(active_tree(page).get_by_label('Fill missing values')).to_be_checked()
    expect(active_tree(page).get_by_label('Whole-number replacement')).to_have_value('0')
    expect(page.locator('#miss_manual-ExportProbe')).to_contain_text('steps=identity,miss_manual')


@pytest.mark.ui
def test_category_extension_depth_and_real_leaf_scrolling(page, app):
    page.goto(app.url)
    status = page.locator('#miss_manual-Status')
    expect(status).to_contain_text('0 committed rules', timeout=60000)
    card = page.locator('.card').first
    card.hover()
    page.locator('#miss_manual-ExpandButton').click(force=True)
    page.wait_for_function("document.querySelector('.card.fullscreen-active')?.getBoundingClientRect().height >= innerHeight * .95")
    card.hover()
    card.locator('button.collapse-toggle').click()
    previous_editor = active_tree(page).get_by_label('Fill missing values').get_attribute('id')
    page.locator('#miss_manual-ExtraLevel').fill('Other')
    # Wait for the server to install the new proposal before switching tabs.
    expect(active_tree(page).get_by_label('Fill missing values')).not_to_have_attribute('id', previous_editor)
    select_variable(page, 'LABEL')
    leaf = active_tree(page).locator('.miss-manual-leaf').filter(has=page.get_by_label('Fill missing values')).first
    expect(leaf.get_by_label('Replacement', exact=True).locator('option')).to_have_text(['Choose a category', 'known', 'Other'])
    leaf.get_by_label('Fill missing values').check()
    leaf.get_by_label('Replacement', exact=True).select_option(label='Other')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('1 committed rule · 5 missing values filled')
    select_variable(page, 'HEALTHCARE_COST')
    expect(active_tree(page).get_by_label('Whole-number replacement')).to_be_visible()
    scroll = active_tree(page).locator('.miss-manual-scroll')
    # Compact cards need real wheel scrolling; fullscreen may already fit.
    card.hover()
    page.locator('#miss_manual-ContractButton').click(force=True)
    scroll.hover()
    page.mouse.wheel(700, 700)
    expect(active_tree(page).get_by_label('Whole-number replacement')).to_be_in_viewport()
    assert scroll.evaluate('el => el.scrollHeight <= el.clientHeight || el.scrollTop > 0')


@pytest.mark.unit
def test_missing_only_split_is_finite_json_and_replayable():
    frame = pd.DataFrame({'split': [None, None, 20., 40.], 'fill': pd.Categorical([None, 'known', 'known', 'known'])})
    data = proxy_data(_df=frame)
    proposal = m.propose_tree(data, 'fill')
    json.dumps(proposal, allow_nan=False)
    rule = rule_for(proposal, 'Other')
    rule.update(extra_level='Other', levels=['known'])
    assert any(c.get('missing_split') for c in rule['conditions'])
    out = m.apply_rules(data, [rule])
    assert out.frame['fill'].tolist() == ['Other', 'known', 'known', 'known']


@pytest.mark.unit
def test_fingerprint_ignores_target_values_and_detects_input_semantics():
    data = source()
    before = m.source_fingerprint(data)
    data.frame['outcome'] = [9, 8, 7, 6]
    assert m.source_fingerprint(data) == before
    data.role_map.set_roles('basis', [Role.TARGET])
    assert m.source_fingerprint(data) != before
    text = proxy_data(_df=pd.DataFrame({'label': pd.Series([None, 'None'], dtype=object)}))
    before = m.source_fingerprint(text)
    text.frame.iloc[0, 0] = 'None'
    assert m.source_fingerprint(text) != before


@pytest.mark.ui
def test_date_picker_and_time_are_typed_and_optional(page, app):
    page.goto(app.url)
    status = page.locator('#miss_manual-Status')
    expect(status).to_contain_text('0 committed rules', timeout=60000)
    select_variable(page, 'VISIT_DATE')
    leaf = active_tree(page).locator('.miss-manual-leaf').filter(has=page.get_by_label('Fill missing values')).first
    expect(leaf.get_by_label('Replacement date', exact=True)).to_be_visible()
    leaf.get_by_label('Fill missing values').check()
    leaf.get_by_label('Replacement date', exact=True).press_sequentially('2025-06-07')
    leaf.get_by_label('Replacement date', exact=True).press('Tab')
    leaf.get_by_label('Replacement time', exact=True).fill('14:30:00')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('1 committed rule · 5 missing values filled')


@pytest.mark.unit
def test_cyclic_destinations_are_skipped():
    from cyclic_pandas import as_cyclic
    data = source()
    data.frame['cycle'] = as_cyclic(pd.Series([0, 1, None, 3]), period=12)
    assert manual_kind(data.frame['cycle']) == 'unsupported'
    assert 'cycle' in m.eligible_columns(data)[1]


@pytest.mark.ui
def test_refresh_has_no_stale_export_callbacks(page, app):
    page.goto(app.url)
    expect(page.locator('#miss_manual-Status')).to_contain_text('0 committed rules', timeout=60000)
    page.reload()
    expect(page.locator('#miss_manual-Status')).to_contain_text('0 committed rules', timeout=60000)
    expect(page.locator('#miss_manual-PositionNotice')).to_be_empty()
    expect(page.locator('.shiny-output-error')).to_have_count(0)


@pytest.mark.unit
def test_readable_conditions_and_singular_status():
    condition = {'feature': {'column': 'HEALTHCARE_BASIS', 'kind': 'categorical', 'level': 'FREE'},
                 'left': False, 'missing_left': False, 'threshold': .5}
    assert m.condition_sentence(condition) == 'HEALTHCARE_BASIS equals "FREE"'
    assert m.condition_sentence(dict(condition, left=True)) == 'HEALTHCARE_BASIS does not equal "FREE"'
    assert m.status_text(1, 1, 1) == '1 committed rule · 1 missing value filled · 1 remains. Unchecked leaves stay missing.'
    assert m.status_text(2, 2, 0).startswith('2 committed rules · 2 missing values filled · 0 remain.')


@pytest.mark.ui
def test_commit_all_tabs_atomically_and_reset_all_fields(page, app):
    page.goto(app.url)
    status = page.locator('#miss_manual-Status')
    expect(status).to_contain_text('0 committed rules', timeout=60000)
    cost = active_tree(page)
    cost.get_by_label('Fill missing values').check()
    cost.get_by_label('Whole-number replacement').fill('0')
    select_variable(page, 'LABEL')
    label = active_tree(page)
    expect(label.get_by_label('Replacement', exact=True)).to_be_visible()
    label.get_by_label('Fill missing values').check()
    # The invalid categorical draft blocks the valid cost draft too.
    page.locator('#miss_manual-Commit').click()
    expect(page.get_by_text('LABEL: Choose a replacement category.', exact=True)).to_be_visible()
    expect(status).to_contain_text('0 committed rules')
    label.get_by_label('Replacement', exact=True).select_option(label='known')
    # Commit from a third tab, so neither completed draft is the current one.
    select_variable(page, 'VISIT_DATE')
    expect(active_tree(page).get_by_label('Replacement date', exact=True)).to_be_visible()
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('2 committed rules · 15 missing values filled')
    select_variable(page, 'HEALTHCARE_COST')
    page.locator('.card').first.hover()
    page.locator('#miss_manual-FlipButton').click(force=True)
    expect(page.locator('#miss_manual-Rules')).to_contain_text('missing HEALTHCARE_COST ← 0 when')
    expect(page.locator('#miss_manual-Rules')).to_contain_text('HEALTHCARE_BASIS equals "FREE"')
    expect(page.locator('#miss_manual-Rules')).not_to_contain_text('missing destination cells only')
    page.locator('#miss_manual-FlipButton').click(force=True)
    page.locator('#miss_manual-Reset').click()
    expect(status).to_contain_text('0 committed rules · 0 missing values filled')
    expect(active_tree(page).get_by_label('Fill missing values')).not_to_be_checked()
    expect(active_tree(page).get_by_label('Whole-number replacement')).to_have_value('')
    select_variable(page, 'LABEL')
    expect(active_tree(page).get_by_label('Fill missing values')).not_to_be_checked()
    expect(active_tree(page).get_by_label('Replacement', exact=True)).to_have_value('')
    select_variable(page, 'VISIT_DATE')
    expect(active_tree(page).get_by_label('Fill missing values')).not_to_be_checked()
    expect(active_tree(page).get_by_label('Replacement date', exact=True)).to_have_value('')
    expect(active_tree(page).get_by_label('Replacement time', exact=True)).to_have_value('')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('0 committed rules · 0 missing values filled')


@pytest.mark.unit
def test_placeholder_pipeline_preview_excludes_unused_replacement_levels():
    from PlaceholderTransformer import PlaceholderTransformer
    from ManualImputationTransformer import manual_levels

    frame = pd.DataFrame({'label': pd.Categorical(['--', 'known', None], categories=['unused', '--', 'known']),
                          'basis': pd.Categorical(['group', 'group', 'group'])})
    source = proxy_data(_df=frame)
    step = PlaceholderTransformer([('label', 'str', '--')])
    cleaned = source.with_pipeline_step(step, name='miss_placeholders', preview_frame=step.fit_transform(frame))
    assert cleaned.clean_frame.label.iloc[0] == '--'
    assert pd.isna(cleaned.frame.label.iloc[0])
    assert '--' in cleaned.frame.label.cat.categories  # Metadata, not current values.
    assert manual_levels(cleaned.frame.label) == ['known']
    proposal = m.propose_tree(cleaned, 'label')
    proposal['extra_level'] = ''
    leaf = next(n for n in proposal['nodes'] if n['leaf'] and n['missing'])
    draft = {f"{m.proposal_key(proposal)}_{leaf['id']}": {'enabled': True, 'value': '0'}}
    rules = m.rules_from_drafts(cleaned, proposal, draft)
    assert rules[0]['value'] == 'known'
    assert rules[0]['levels'] == ['known']
    with pytest.raises(ValueError, match='category'):
        manual_value(cleaned.frame.label, '--')
    result = m.apply_rules(cleaned, rules)
    assert result.frame.label.tolist() == ['known', 'known', 'known']
    # The recorded vocabulary remains valid even if a fold has no known labels.
    pipeline = result.pipeline_for_training().fit(frame.iloc[[0]])
    assert pipeline.transform(frame).label.tolist() == ['known', 'known', 'known']
    assert_frame_equal(source.frame, frame)


placeholder_app = create_app_fixture(app='../scenarios/miss_manual_placeholders.py', scope='function')


@pytest.mark.ui
def test_dropdown_uses_observed_pipeline_labels(page, placeholder_app):
    page.goto(placeholder_app.url)
    expect(page.locator('#miss_manual-Status')).to_contain_text('0 committed rules', timeout=60000)
    select_variable(page, 'LABEL')
    choices = active_tree(page).get_by_label('Replacement', exact=True)
    expect(choices.locator('option')).to_have_text(['Choose a category', 'known'])
    active_tree(page).get_by_label('Fill missing values').check()
    choices.select_option(label='known')
    page.locator('#miss_manual-Commit').click()
    expect(page.locator('#miss_manual-Status')).to_contain_text('1 committed rule · 5 missing values filled')


@pytest.mark.unit
@pytest.mark.parametrize('missing,rows,minimum,expected', [
    (8, 10, .8, True), (8, 10, .85, False), (17, 20, .85, True),
    (10, 10, 1., True), (9, 10, 1., False),
    (5, 10, .5, False), (4, 10, .5, False), (0, 10, .5, False), (0, 0, .5, False),
])
def test_leaf_accuracy_requires_missingness_prediction(missing, rows, minimum, expected):
    assert m.leaf_qualifies({'leaf': True, 'missing': missing, 'rows': rows}, minimum) is expected


@pytest.mark.unit
def test_commit_ignores_unreliable_leaf_drafts():
    data = proxy_data(_df=pd.DataFrame({'value': pd.Series([None, 1], dtype='Int64')}))
    proposal = {'column': 'value', 'kind': 'integer', 'nodes': [
        {'id': 1, 'leaf': True, 'missing': 8, 'rows': 10, 'conditions': []},
        {'id': 2, 'leaf': True, 'missing': 4, 'rows': 10, 'conditions': []},
    ]}
    key = m.proposal_key(proposal)
    drafts = {f'{key}_1': {'enabled': True, 'value': 0},
              f'{key}_2': {'enabled': True, 'value': 'invalid'}}
    assert [r['value'] for r in m.rules_from_drafts(data, proposal, drafts, .8)] == [0]
    assert m.rules_from_drafts(data, proposal, drafts, .85) == []


accuracy_app = create_app_fixture(app='../scenarios/miss_manual_accuracy.py', scope='function')


@pytest.mark.ui
def test_accuracy_filters_fields_tabs_and_preserves_commit(page, accuracy_app):
    page.goto(accuracy_app.url)
    status = page.locator('#miss_manual-Status')
    expect(page.get_by_role('tab', name='value', exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role('tab', name='weak', exact=True)).to_have_count(0)
    expect(active_tree(page).get_by_label('Fill missing values')).to_have_count(1)
    expect(active_tree(page)).to_contain_text('8 missing / 10 rows (80%)')
    expect(active_tree(page)).to_contain_text('4 missing / 10 rows (40%)')
    active_tree(page).get_by_label('Fill missing values').check()
    active_tree(page).get_by_label('Whole-number replacement').fill('0')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('1 committed rule · 8 missing values filled')
    slider = page.locator('#miss_manual-MinLeafAccuracy')
    assert float(slider.get_attribute('data-min')) == .5
    assert float(slider.get_attribute('data-max')) == 1.
    assert float(slider.get_attribute('data-step')) == .05
    slider.evaluate("el => { window.jQuery(el).data('ionRangeSlider').update({from: .85}); window.jQuery(el).trigger('change'); }")
    expect(page.get_by_role('tab', name='value', exact=True)).to_have_count(0)
    expect(page.locator('#miss_manual-Variables')).to_contain_text('No variables have a leaf predicting missingness with accuracy ≥ 85%')
    expect(status).to_contain_text('1 committed rule · 8 missing values filled')
    page.locator('#miss_manual-Commit').click()
    expect(status).to_contain_text('1 committed rule · 8 missing values filled')
    slider.evaluate("el => { window.jQuery(el).data('ionRangeSlider').update({from: .5}); window.jQuery(el).trigger('change'); }")
    expect(page.get_by_role('tab', name='value', exact=True)).to_be_visible()
    expect(page.get_by_role('tab', name='weak', exact=True)).to_have_count(0)
    expect(active_tree(page).get_by_label('Fill missing values')).to_have_count(1)
    expect(active_tree(page).get_by_label('Fill missing values')).to_be_checked()
