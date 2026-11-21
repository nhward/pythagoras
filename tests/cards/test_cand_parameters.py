import json

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/cand_parameters.py', scope='function')
restore_app = create_app_fixture(app='../scenarios/cand_parameters_restore.py', scope='function')


def parameter_row(page, name):
    return page.locator('[id$="-Table"]').get_by_text(name, exact=True).locator('xpath=ancestor::tr')


def tune(page, name):
    return parameter_row(page, name).get_by_role('checkbox')


def edit(page, name):
    parameter_row(page, name).get_by_role('button', name='Edit ' + name, exact=True).click()
    expect(page.locator('#shiny-modal')).to_be_visible()
    page.wait_for_function("() => { const el = document.querySelector('#shiny-modal'); const m = bootstrap.Modal.getInstance(el); return m && !m._isTransitioning; }")


def ok(page):
    page.locator('#shiny-modal').get_by_role('button', name='OK', exact=True).click()
    expect(page.locator('#shiny-modal')).to_have_count(0)


@pytest.mark.ui
def test_restore_replays_commit_without_committing_later_drafts_or_future_pipeline_changes(page, restore_app):
    page.goto(restore_app.url)
    exported = page.locator('[id$="-ExportState"]')
    status = page.locator('[id$="-Status"]')
    expect(exported).to_have_text('Committed combinations: 4', timeout=30000)
    expect(status).to_contain_text('1 possible combinations')
    expect(status).to_contain_text('Uncommitted changes')
    expect(page.locator('[id$="-SavedState"]')).to_have_text('Saved commitment is current')
    page.locator('button[id$="-Revert"]').click()
    expect(status).to_contain_text('4 possible combinations')
    expect(status).to_contain_text('Committed')
    expect(exported).to_have_text('Committed combinations: 4')
    page.get_by_role('button', name='Change pipeline', exact=True).click()
    expect(exported).to_have_text('No specification')
    expect(status).to_contain_text('Uncommitted changes')

@pytest.mark.ui
def test_enable_commit_and_revert_preserves_current_pipeline(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    row = parameter_row(page, 'Observation missingness proportion')
    before_order = table.locator('tbody tr').all_text_contents()
    before_y = row.bounding_box()['y']
    active = tune(page, 'Observation missingness proportion')
    active.check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')
    expect(table.locator('thead tr').first.locator('th')).to_have_text(
        ['Tune', 'Edit', 'Parameter', 'Source', 'Search values', 'Current value', 'Reason', 'Purpose'])
    assert table.locator('tbody tr').all_text_contents() == before_order
    assert abs(row.bounding_box()['y'] - before_y) < 2
    expect(active).to_be_checked()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-ExportState"]')).to_have_text('Committed combinations: 4')
    active.uncheck()
    expect(page.locator('[id$="-Status"]')).to_contain_text('1 possible combinations')
    expect(page.locator('[id$="-ExportState"]')).to_have_text('Committed combinations: 4')
    page.locator('button[id$="-Revert"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')

@pytest.mark.ui
def test_missing_definition_must_be_completed_before_enabling(page, app):
    page.goto(app.url)
    card = page.locator('.card').first
    card.hover(); card.locator('button.collapse-toggle').click()
    page.locator('[id$="-Visibility"]').get_by_label('All tunable (including unreviewed)', exact=True).check()
    card.locator('button.collapse-toggle').click()
    page.locator('[id$="-ExpandButton"]').click()
    page.locator('[id$="-Search"]').fill('unknown_count')
    expect(tune(page, 'Unknown count')).to_be_disabled()
    edit(page, 'Unknown count')
    page.locator('[id$="-Kind"]').select_option('integer')
    page.locator('[id$="-Values"]').fill('[2, 3, 4, null]')
    ok(page)
    expect(tune(page, 'Unknown count')).to_be_enabled()
    tune(page, 'Unknown count').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')

@pytest.mark.ui
def test_numeric_range_validation_and_review(page, app):
    page.goto(app.url)
    edit(page, 'Observation missingness proportion')
    page.locator('[id$="-Increment"]').fill('0')
    page.get_by_role('button', name='OK', exact=True).click()
    expect(page.locator('[id$="-ModalError"]')).to_contain_text('positive step')
    expect(page.locator('#shiny-modal')).to_be_visible()
    page.wait_for_function("() => { const el = document.querySelector('#shiny-modal'); const m = bootstrap.Modal.getInstance(el); return m && !m._isTransitioning; }")
    page.locator('[id$="-Increment"]').fill('0.1')
    ok(page)
    tune(page, 'Observation missingness proportion').check()
    page.locator('[id$="-Commit"]').click()
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    expect(page.locator('[id$="-Review"]')).to_contain_text('4 candidate combinations')
    expect(page.locator('[id$="-Review"]')).to_contain_text('miss_map__observation_threshold')


@pytest.mark.ui
def test_fullscreen_search_and_horizontal_editor(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(app.url)
    search = page.locator('[id$="-Search"]')
    expect(search).to_be_hidden()
    expect(page.locator('[id$="-Editor"]')).to_have_count(0)
    edit(page, 'Observation missingness proportion')
    controls = page.locator('#shiny-modal .tuning-editor-controls')
    assert controls.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    page.get_by_role('button', name='Cancel', exact=True).click()
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(search).to_be_visible()
    edit(page, 'Observation missingness proportion')
    assert controls.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    page.keyboard.press('Escape')
    expect(page.locator('#shiny-modal')).to_have_count(0)
    page.locator('[id$="-ContractButton"]').click()
    expect(search).to_be_hidden()


@pytest.mark.ui
def test_parameter_visibility_radio_levels(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    expect(table).not_to_contain_text('Fixed configuration')
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    setting = page.locator('[id$="-Visibility"]')
    expect(setting.get_by_label('Recommended', exact=True)).to_be_checked()
    expect(table.get_by_text('Bootstrap', exact=True)).to_have_count(0)
    expect(table.get_by_text('Max samples', exact=True)).to_have_count(0)
    setting.get_by_label('Recommended + advanced', exact=True).check()
    expect(table.get_by_text('Max samples', exact=True)).to_be_visible()
    expect(table.get_by_text('Bootstrap', exact=True)).to_be_visible()
    expect(table.get_by_text('Unknown flag', exact=True)).to_have_count(0)
    setting.get_by_label('All tunable (including unreviewed)', exact=True).check()
    expect(table.get_by_text('Bootstrap', exact=True)).to_be_visible()
    expect(table.get_by_text('Unknown flag', exact=True)).to_be_visible()
    expect(table).not_to_contain_text('Fixed configuration')
    setting.get_by_label('Everything (including operational and recipe)', exact=True).check()
    expect(table).to_contain_text('Fixed configuration')
    setting.get_by_label('Recommended', exact=True).check()
    expect(table).not_to_contain_text('Fixed configuration')
    expect(table).to_contain_text('Observation missingness proportion')


@pytest.mark.ui
def test_enabled_advanced_parameter_stays_visible_in_recommended_view(page, app):
    page.goto(app.url)
    card = page.locator('.card').first
    card.hover(); card.locator('button.collapse-toggle').click()
    visibility = page.locator('[id$="-Visibility"]')
    visibility.get_by_label('Recommended + advanced', exact=True).check()
    card.locator('button.collapse-toggle').click()
    tune(page, 'Remove observations').check()
    card.hover(); card.locator('button.collapse-toggle').click()
    visibility.get_by_label('Recommended', exact=True).check()
    card.locator('button.collapse-toggle').click()
    expect(page.locator('[id$="-Table"]')).to_contain_text('Remove observations')
    expect(page.locator('[id$="-Status"]')).to_contain_text('1 selected for auto-tuning')


@pytest.mark.ui
def test_technical_specification_scrolls_independently(page, app):
    page.goto(app.url)
    tune(page, 'Observation missingness proportion').check()
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    page.get_by_text('Technical search specification', exact=True).click()
    specification = page.locator('.tuning-specification-scroll')
    expect(specification).to_be_visible()
    assert specification.evaluate('el => el.scrollHeight > el.clientHeight')
    page.wait_for_timeout(1400)
    specification.hover(); page.mouse.wheel(0, 700)
    page.wait_for_function("document.querySelector('.tuning-specification-scroll').scrollTop > 0", timeout=3000)
    assert specification.evaluate('el => el.scrollTop > 0 && Math.abs(el.scrollHeight - el.clientHeight - el.scrollTop) < 2')


@pytest.mark.ui
def test_tuning_grid_fills_fullscreen_space_beside_editor(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(app.url)
    table = page.locator('.tuning-parameters-table')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    before = table.bounding_box()['height']
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(page.locator('[id$="-Search"]')).to_be_visible()
    page.wait_for_function("document.querySelector('.tuning-parameters-table').getBoundingClientRect().height > 600")
    after = table.bounding_box()
    assert after['height'] > before + 200
    assert table.locator('shiny-data-frame').bounding_box()['height'] > after['height'] * .9
    expect(page.locator('.front .tuning-editor-controls')).to_have_count(0)


@pytest.mark.ui
def test_boolean_editor_only_requires_auto_tune_checkbox(page, app):
    page.goto(app.url)
    card = page.locator('.card').first
    card.hover(); card.locator('button.collapse-toggle').click()
    page.locator('[id$="-Visibility"]').get_by_label('Recommended + advanced', exact=True).check()
    card.locator('button.collapse-toggle').click()
    tune(page, 'Remove observations').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 possible combinations')
    edit(page, 'Remove observations')
    expect(page.locator('[id$="-ChoiceValues"]')).to_be_attached()
    expect(page.locator('#shiny-modal')).to_contain_text('False')
    expect(page.locator('#shiny-modal')).to_contain_text('True')
    expect(page.locator('#shiny-modal input[type=checkbox]')).to_have_count(0)
    page.get_by_role('button', name='Cancel', exact=True).click()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-ExportState"]')).to_have_text('Committed combinations: 2')


@pytest.mark.ui
def test_explicit_choices_hide_numeric_range_fields(page, app):
    page.goto(app.url)
    edit(page, 'Observation missingness proportion')
    mode = page.locator('[id$="-Mode"]')
    mode.select_option('values')
    for name in ('Minimum', 'Maximum', 'Increment'):
        expect(page.locator(f'[id$="-{name}"]')).to_be_hidden()
    expect(page.locator('[id$="-Values"]')).to_be_visible()
    mode.select_option('range')
    for name in ('Minimum', 'Maximum', 'Increment'):
        expect(page.locator(f'[id$="-{name}"]')).to_be_visible()
    expect(page.locator('[id$="-Values"]')).to_be_hidden()


multiple_app = create_app_fixture(app='../scenarios/cand_parameters_multiple.py', scope='function')


@pytest.mark.ui
def test_method_sources_and_conditional_combination_count(page, multiple_app):
    page.set_viewport_size({'width':1600, 'height':1100})
    page.goto(multiple_app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Ridge Regression', timeout=30000)
    expect(table).to_contain_text('Decision Tree Regression')
    page.locator('.card').first.hover(); page.locator('[id$="-ExpandButton"]').click()
    page.locator('[id$="-Search"]').fill('alpha')
    tune(page, 'Alpha').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('6 possible combinations')
    page.locator('[id$="-Search"]').fill('max_depth')
    expect(tune(page, 'Max depth')).not_to_be_checked()
    tune(page, 'Max depth').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('9 possible combinations')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-MultiSpec"]')).to_have_text('9')


bundled_app = create_app_fixture(app='../scenarios/cand_parameters_bundled.py', scope='function')


@pytest.mark.ui
def test_bundled_method_settings_count_only_active_branches(page, bundled_app):
    page.set_viewport_size({'width':1600, 'height':1100})
    page.goto(bundled_app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Neighbors', timeout=30000)
    page.locator('.card').first.hover(); page.locator('[id$="-ExpandButton"]').click()
    page.locator('[id$="-Search"]').fill('miss_impute__method')
    tune(page, 'Method').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('3 possible combinations')
    page.locator('[id$="-Search"]').fill('miss_impute__knn__n_neighbors')
    edit(page, 'Neighbors')
    expect(page.locator('#shiny-modal')).to_contain_text('Applies when method includes knn')
    page.get_by_role('button', name='Cancel', exact=True).click()
    expect(tune(page, 'Neighbors')).not_to_be_checked()
    tune(page, 'Neighbors').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('6 possible combinations')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-ExportState"]')).to_have_text('Committed combinations: 6')


@pytest.mark.ui
@pytest.mark.parametrize('dismiss', ['cancel', 'escape', 'backdrop'])
def test_modal_dismissal_discards_edits_and_does_not_bookmark_transient_inputs(page, app, dismiss):
    page.goto(app.url)
    edit(page, 'Observation missingness proportion')
    original = page.locator('[id$="-Maximum"]').input_value()
    before = json.loads(page.locator('[id$="-ConfigurationState"]').inner_text())['tuning']['drafts']
    page.locator('[id$="-Maximum"]').fill('9')
    if dismiss == 'cancel':
        page.get_by_role('button', name='Cancel', exact=True).click()
    elif dismiss == 'escape':
        page.keyboard.press('Escape')
    else:
        page.locator('#shiny-modal').click(position={'x': 5, 'y': 5})
    expect(page.locator('#shiny-modal')).to_have_count(0)
    edit(page, 'Observation missingness proportion')
    expect(page.locator('[id$="-Maximum"]')).to_have_value(original)
    state = json.loads(page.locator('[id$="-ConfigurationState"]').inner_text())
    assert state['tuning']['drafts'] == before
    assert not set(state['inputs']).intersection({'RowEvent', 'Kind', 'Mode', 'Values', 'Minimum', 'Maximum', 'Increment'})


@pytest.mark.ui
def test_fixed_parameters_have_disabled_checkbox_and_read_only_details(page, app):
    page.goto(app.url)
    card = page.locator('.card').first
    card.hover(); card.locator('button.collapse-toggle').click()
    page.locator('[id$="-Visibility"]').get_by_label('Everything (including operational and recipe)', exact=True).check()
    card.locator('button.collapse-toggle').click()
    page.locator('[id$="-ExpandButton"]').click()
    page.locator('[id$="-Search"]').fill('n_jobs')
    expect(tune(page, 'N jobs')).to_be_disabled()
    edit(page, 'N jobs')
    expect(page.locator('#shiny-modal')).to_contain_text('Fixed configuration')
    expect(page.locator('#shiny-modal input, #shiny-modal select')).to_have_count(0)
    ok(page)
    expect(tune(page, 'N jobs')).to_be_disabled()


@pytest.mark.ui
def test_modal_refuses_stale_upstream_parameter_changes(page, restore_app):
    page.goto(restore_app.url)
    edit(page, 'Observation missingness proportion')
    page.locator('[id$="-Maximum"]').fill('9')
    page.evaluate("() => Shiny.setInputValue(document.querySelector('[id$=\"-ChangePipeline\"]').id, 1, {priority: 'event'})")
    expect(page.locator('[id$="-ExportState"]')).to_have_text('No specification')
    page.get_by_role('button', name='OK', exact=True).click()
    expect(page.locator('[id$="-ModalError"]')).to_contain_text('Upstream parameters changed')
    expect(page.locator('#shiny-modal')).to_be_visible()
    page.wait_for_function("() => { const el = document.querySelector('#shiny-modal'); const m = bootstrap.Modal.getInstance(el); return m && !m._isTransitioning; }")


@pytest.mark.ui
def test_row_actions_preserve_sorting_and_checkbox_state_after_scrolling(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    table.get_by_text('Parameter', exact=True).click()
    paths = lambda: table.locator('input.tuning-row-toggle').evaluate_all('els => els.map(el => el.dataset.tuningPath)')
    before = paths()
    tune(page, 'Observation missingness proportion').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')
    assert paths() == before
    table.get_by_text('Tune', exact=True).click()
    assert paths() == before  # HTML action columns are not sortable.
    grid = table.locator('.shiny-data-grid')
    grid.evaluate('el => el.scrollTop = el.scrollHeight')
    grid.evaluate('el => el.scrollTop = 0')
    expect(tune(page, 'Observation missingness proportion')).to_be_checked()


double_app = create_app_fixture(app='../scenarios/cand_parameters_double.py', scope='function')


@pytest.mark.ui
def test_row_actions_are_namespaced_between_card_instances(page, double_app):
    page.goto(double_app.url)
    cards = page.locator('.card')
    first, second = cards.nth(0), cards.nth(1)
    row = first.get_by_text('Observation missingness proportion', exact=True).locator('xpath=ancestor::tr')
    row.get_by_role('checkbox').check()
    expect(first.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')
    expect(second.locator('[id$="-Status"]')).to_contain_text('1 possible combinations')
    other = second.get_by_text('Observation missingness proportion', exact=True).locator('xpath=ancestor::tr')
    expect(other.get_by_role('checkbox')).not_to_be_checked()
    other.get_by_role('button', name='Edit Observation missingness proportion', exact=True).click()
    page.locator('[id$="-Maximum"]').fill('0.5')
    ok(page)
    other.get_by_role('checkbox').check()
    expect(second.locator('[id$="-Status"]')).to_contain_text('3 possible combinations')
    expect(first.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')


@pytest.mark.ui
def test_commit_promotes_checked_parameters_and_focuses_first_row(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    table.get_by_text('Parameter', exact=True).click()
    table.get_by_text('Parameter', exact=True).click()
    boxes = table.locator('input.tuning-row-toggle:enabled')
    chosen = [boxes.nth(i).get_attribute('data-tuning-path') for i in (1, 3)]
    labels = [boxes.nth(i).get_attribute('aria-label') for i in (1, 3)]
    before = table.locator('tbody tr').all_text_contents()
    for label in labels:
        table.get_by_role('checkbox', name=label, exact=True).check()
    assert table.locator('tbody tr').all_text_contents() == before
    table.locator('.shiny-data-grid').evaluate('el => el.scrollTop = el.scrollHeight')
    page.locator('[id$="-Commit"]').evaluate('el => el.addEventListener("click", () => setTimeout(() => { const g = document.querySelector(".tuning-parameters-table .shiny-data-grid"); g.scrollTop = g.scrollHeight; }, 100), {once: true})')
    page.locator('[id$="-Commit"]').click()
    expect(table.locator('tbody tr').nth(0).get_by_role('checkbox')).to_be_checked()
    expect(table.locator('tbody tr').nth(1).get_by_role('checkbox')).to_be_checked()
    expect(table.locator('tbody tr').first).to_have_attribute('aria-selected', 'true')
    expect(table.locator('tbody tr').first).to_be_in_viewport()
    assert {table.locator('tbody tr').nth(i).get_by_role('checkbox').get_attribute('data-tuning-path') for i in (0, 1)} == set(chosen)
    expect(table.locator('input.tuning-row-toggle:checked')).to_have_count(2)


    page.wait_for_timeout(650)
    assert table.locator('.shiny-data-grid').evaluate('el => el.scrollTop') == 0


checked_restore_app = create_app_fixture(app='../scenarios/cand_parameters_checked_restore.py', scope='function')


@pytest.mark.ui
def test_restored_checked_parameters_are_promoted_and_scrolled_to_top(page, checked_restore_app):
    page.add_init_script("""(() => {
        const observer = new MutationObserver(() => {
            const grid = document.querySelector('.tuning-parameters-table .shiny-data-grid');
            if (!grid || !grid.querySelector('tbody tr')) return;
            observer.disconnect();
            grid.scrollTop = grid.scrollHeight;
            setTimeout(() => { grid.scrollTop = grid.scrollHeight; }, 50);
        });
        observer.observe(document, {subtree: true, childList: true});
    })();""")
    page.goto(checked_restore_app.url)
    table = page.locator('[id$="-Table"]')
    expect(table.locator('input.tuning-row-toggle:checked')).to_have_count(2, timeout=30000)
    page.wait_for_timeout(650)
    expect(table.locator('tbody tr').nth(0).get_by_role('checkbox')).to_be_checked()
    expect(table.locator('tbody tr').nth(1).get_by_role('checkbox')).to_be_checked()
    assert table.locator('.shiny-data-grid').evaluate('el => el.scrollTop') == 0


defaults_app = create_app_fixture(app='../scenarios/cand_parameters_defaults.py', scope='function')


@pytest.mark.ui
def test_native_defaults_show_documented_values_without_setting_model_parameters(page, defaults_app):
    page.goto(defaults_app.url)
    row = page.locator('button.tuning-row-edit[data-tuning-path$="__iterations"]').locator('xpath=ancestor::tr')
    expect(row).to_contain_text('Default: 1000', timeout=30000)
    expect(page.locator('[id$="-DefaultsProbe"]')).to_have_text('iterations=None;depth=None')
    row.get_by_role('button', name='Edit').click()
    modal = page.locator('#shiny-modal')
    expect(modal).to_contain_text('Method default: 1000')
    expect(modal.get_by_role('link', name='Default documentation', exact=True)).to_have_attribute('href',
        'https://catboost.ai/docs/en/references/training-parameters/common#iterations')
    expect(page.locator('[id$="-DefaultsProbe"]')).to_have_text('iterations=None;depth=None')
