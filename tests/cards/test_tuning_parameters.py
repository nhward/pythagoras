import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/tuning_parameters.py', scope='function')

@pytest.mark.ui
def test_enable_commit_and_revert_preserves_current_pipeline(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    table.get_by_text('Observation missingness proportion', exact=True).click()
    active = page.locator('[id$="-Active"]')
    expect(active).to_be_enabled()
    active.check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')
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
    expect(page.locator('[id$="-Table"]')).to_contain_text('Observation missingness proportion', timeout=30000)
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    page.locator('[id$="-Search"]').fill('max_samples')
    page.locator('[id$="-Table"]').get_by_text('Max samples', exact=True).click()
    expect(page.locator('[id$="-Active"]')).to_be_disabled()
    page.locator('[id$="-Kind"]').select_option('integer')
    page.locator('[id$="-Values"]').fill('[2, 3, 4, null]')
    page.locator('[id$="-SaveDefinition"]').click()
    expect(page.locator('[id$="-Active"]')).to_be_enabled()
    page.locator('[id$="-Active"]').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('4 possible combinations')

@pytest.mark.ui
def test_numeric_range_validation_and_review(page, app):
    page.goto(app.url)
    page.locator('[id$="-Table"]').get_by_text('Observation missingness proportion', exact=True).click()
    page.locator('[id$="-Increment"]').fill('0')
    page.locator('[id$="-SaveDefinition"]').click()
    expect(page.locator('[id$="-Message"]')).to_contain_text('positive step')
    page.locator('[id$="-Increment"]').fill('0.1')
    page.locator('[id$="-SaveDefinition"]').click()
    page.locator('[id$="-Active"]').check()
    page.locator('[id$="-Commit"]').click()
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    expect(page.locator('[id$="-Review"]')).to_contain_text('4 candidate combinations')
    expect(page.locator('[id$="-Review"]')).to_contain_text('miss_map__observation_threshold')


@pytest.mark.ui
def test_fullscreen_search_and_horizontal_editor(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    search = page.locator('[id$="-Search"]')
    expect(search).to_be_hidden()
    table.get_by_text('Observation missingness proportion', exact=True).click()
    controls = page.locator('.tuning-editor-controls')
    assert controls.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(search).to_be_visible()
    page.wait_for_timeout(600)
    positions = [page.locator(f'[id$="-{name}"]').bounding_box()['y']
                 for name in ['Kind', 'Mode', 'Minimum', 'Maximum', 'Increment']]
    assert max(positions) - min(positions) < 3
    assert controls.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    page.locator('[id$="-ContractButton"]').click()
    expect(search).to_be_hidden()


@pytest.mark.ui
def test_fixed_configuration_hidden_unless_show_everything(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    expect(table).not_to_contain_text('Fixed configuration')
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    setting = page.locator('[id$="-ShowEverything"]')
    expect(setting).not_to_be_checked()
    setting.check()
    expect(table).to_contain_text('Fixed configuration')
    setting.uncheck()
    expect(table).not_to_contain_text('Fixed configuration')
    expect(table).to_contain_text('Observation missingness proportion')


@pytest.mark.ui
def test_technical_specification_scrolls_independently(page, app):
    page.goto(app.url)
    page.locator('[id$="-Table"]').get_by_text('Observation missingness proportion', exact=True).click()
    page.locator('[id$="-Active"]').check()
    page.locator('.card').first.hover()
    page.locator('[id$="-FlipButton"]').click()
    page.get_by_text('Technical search specification', exact=True).click()
    specification = page.locator('.tuning-specification-scroll')
    expect(specification).to_be_visible()
    assert specification.evaluate('el => el.scrollHeight > el.clientHeight')
    page.wait_for_timeout(1400)  # Wait for the card flip animation.
    specification.hover()
    page.mouse.wheel(0, 700)
    page.wait_for_function("document.querySelector('.tuning-specification-scroll').scrollTop > 0", timeout=3000)
    assert specification.evaluate('el => el.scrollTop > 0 && Math.abs(el.scrollHeight - el.clientHeight - el.scrollTop) < 2')
    expect(page.get_by_text('Technical search specification', exact=True)).to_be_in_viewport()


@pytest.mark.ui
def test_tuning_grid_fills_fullscreen_space_beside_editor(page, app):
    page.set_viewport_size({'width': 1600, 'height': 1000})
    page.goto(app.url)
    table = page.locator('.tuning-parameters-table')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    table.get_by_text('Observation missingness proportion', exact=True).click()
    before = table.bounding_box()['height']
    page.locator('.card').first.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(page.locator('[id$="-Search"]')).to_be_visible()
    page.wait_for_timeout(600)
    after = table.bounding_box()
    assert after['height'] > before + 200
    assert table.locator('shiny-data-frame').bounding_box()['height'] > after['height'] * .9
    editor = page.locator('.tuning-editor-controls')
    expect(editor).to_be_in_viewport()
    assert after['y'] + after['height'] <= editor.bounding_box()['y'] + 2


@pytest.mark.ui
def test_boolean_editor_only_requires_auto_tune_checkbox(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Remove observations', timeout=30000)
    table.get_by_text('Remove observations', exact=True).click()
    editor = page.locator('[id$="-Editor"]')
    expect(editor.locator('input')).to_have_count(1)
    expect(editor.locator('select, button')).to_have_count(0)
    active = editor.locator('[id$="-Active"]')
    expect(active).to_be_enabled()
    active.check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('2 possible combinations')
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-ExportState"]')).to_have_text('Committed combinations: 2')


@pytest.mark.ui
def test_explicit_choices_hide_numeric_range_fields(page, app):
    page.goto(app.url)
    table = page.locator('[id$="-Table"]')
    expect(table).to_contain_text('Observation missingness proportion', timeout=30000)
    table.get_by_text('Observation missingness proportion', exact=True).click()
    mode = page.locator('[id$="-Mode"]')
    mode.select_option('values')
    for name in ('Minimum', 'Maximum', 'Increment'):
        expect(page.locator(f'[id$="-{name}"]')).to_be_hidden()
    expect(page.locator('[id$="-Values"]')).to_be_visible()
    mode.select_option('range')
    for name in ('Minimum', 'Maximum', 'Increment'):
        expect(page.locator(f'[id$="-{name}"]')).to_be_visible()
    expect(page.locator('[id$="-Values"]')).to_be_hidden()
