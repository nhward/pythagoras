import pandas as pd
import pytest
from cards import var_nominal as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='../scenarios/var_nominal.py', scope='function')
learned_app = create_app_fixture(app='../scenarios/var_nominal_learned.py', scope='function')

@pytest.mark.unit
def test_similarity_and_frequency_direction():
    assert m.similarity('Mon.', 'Monday')[1].startswith('Abbreviation')
    assert m.similarity('Mondya', 'Monday')[1].startswith('Typo')
    frame = pd.DataFrame({'day': pd.Categorical(['Monday'] * 30 + ['Mon.'] * 3 + ['Mondya'])})
    items = m.proposals(frame, 'day')
    assert any(p['sources'] == ['Mondya'] and p['destination'] == 'Monday' for p in items)
    assert not any('Monday' in p['sources'] for p in items)

@pytest.mark.unit
def test_suppression_and_change_of_mind():
    pairs = [('a','b'), ('c','b'), ('c','a'), ('b','a'), ('a','c'), ('b','c')]
    items = [m.proposal('x',[a],b,'test') for a,b in pairs]
    decisions = {items[0]['key']:'accept'}
    visible = m.visible_proposals(items, decisions)
    assert visible == [items[0],items[1],items[5]]
    assert m.visible_proposals(items,{items[0]['key']:'undecided'}) == items
    pool = m.proposal('x',['a','z'],'Other','rare',kind='pool')
    assert pool not in m.visible_proposals(items+[pool],decisions)
    # Identical labels in other variables are unrelated.
    other = m.proposal('y',['a'],'b','test')
    assert other in m.visible_proposals(items+[other],decisions)

@pytest.mark.unit
def test_commit_is_simultaneous_preserves_missing_and_can_reset():
    frame = pd.DataFrame({'x':pd.Categorical(['a','b','c',None],categories=['a','b','c','unused'])},index=[4,4,7,9])
    source = proxy_data(frame)
    batch = [m.proposal('x',['a'],'b','test'),m.proposal('x',['b'],'c','test')]
    state = m.fresh_state()
    state['batches']=[batch]
    result=m.committed_frame(source,state)
    assert result.x.iloc[:3].tolist()==['c','c','c']
    assert pd.isna(result.x.iloc[3])
    assert result.index.equals(frame.index)
    assert list(result.x.cat.categories)==['c','unused']
    pd.testing.assert_frame_equal(m.committed_frame(source,m.fresh_state()),frame)
    with pytest.raises(ValueError,match='cycle'):
        m.apply_merges(frame,[m.proposal('x',['a'],'b','test'),m.proposal('x',['b'],'a','test')])

@pytest.mark.unit
def test_expected_destination_and_role_exclusions():
    source=proxy_data(pd.DataFrame({'x':pd.Categorical(['Sept']*3), 'ordered':pd.Categorical(['a']*3,ordered=True)}))
    assert m.eligible_columns(source)==['x']
    assert any(p['destination']=='September' for p in m.proposals(source.frame,'x',['September']))
    assert 'September' in str(m.bar_chart(source.frame.x,['September']))

@pytest.mark.ui
@pytest.mark.parametrize("fixture_name", ["app", "learned_app"])
def test_decisions_commit_reset_and_tabs(page, request, fixture_name):
    app = request.getfixturevalue(fixture_name)
    page.goto(app.url)
    panels=page.locator('[id$="-Panels"]')
    expect(panels.get_by_role('tab',name='Day',exact=True)).to_be_visible(timeout=20000)
    row=panels.locator('tr').filter(has_text='Mon. → Monday')
    expect(row).to_be_visible(timeout=10000)
    preview = page.locator('[id$="-Preview"]')
    expect(preview).to_have_text('Monday count: 30')
    typo = panels.locator('tr').filter(has_text='Mondya → Monday')
    pooling = panels.locator('tr').filter(has_text='Mondya → Other')
    expect(pooling).to_be_visible()
    typo.locator('input[value="accept"]').check()
    expect(pooling).to_have_count(0)
    expect(row).to_be_visible()  # The shared destination remains available.
    typo.locator('input[value="undecided"]').check()
    expect(pooling).to_be_visible()
    row.locator('input[value="accept"]').check()
    expect(page.locator('[id$="-Status"]')).to_contain_text('1 accepted proposals pending')
    expect(preview).to_have_text('Monday count: 30')
    page.locator('[id$="-Commit"]').click()
    expect(preview).to_have_text('Monday count: 33')
    steps = 'upstream,var_nominal' if fixture_name == 'learned_app' else 'var_nominal'
    expect(page.locator('[id$="-PipelineProbe"]')).to_have_text(f'steps={steps}; replay=True')
    expect(row).to_have_count(0)
    expect(page.locator('[id$="-Status"]')).to_contain_text('1 recorded decisions')
    panels.get_by_role('tab',name='Colour',exact=True).click()
    expect(panels.locator('tr').filter(has_text='Blu → Blue')).to_be_visible()
    page.locator('[id$="-Reset"]').click()
    expect(preview).to_have_text('Monday count: 30')
    panels.get_by_role('tab',name='Day',exact=True).click()
    expect(row).to_be_visible()
    expect(preview).to_have_text('Monday count: 30')
    expect(page.locator('[id$="-Status"]')).to_contain_text('0 recorded decisions')
    expected = 'steps=upstream; replay=True' if fixture_name == 'learned_app' else 'steps=none'
    expect(page.locator('[id$="-PipelineProbe"]')).to_have_text(expected)

@pytest.mark.ui
def test_fullscreen_editors_are_scoped_to_variable_tabs(page, app):
    page.goto(app.url)
    card = page.locator('.card').first
    panels = page.locator('[id$="-Panels"]')
    expect(panels.get_by_role('tab', name='Day', exact=True)).to_be_visible(timeout=20000)
    expect(panels.locator('details')).to_have_count(0)
    expect(card.locator('.hidden > .contract-btn')).to_have_count(1)
    card.hover()
    page.locator('[id$="-ExpandButton"]').click()
    expect(card).to_have_attribute('aria-expanded', 'true')
    editor = panels.locator('details:visible')
    editor.locator('summary').click()
    expect(panels.get_by_role('button', name='Save expectations')).to_have_count(0)
    editor.locator('[id$="_ExpectedCount"]').fill('5')
    expect(panels.locator('.tab-pane.active')).to_contain_text('Expected: 5')
    editor.locator('summary').click()
    editor.locator('[id$="_ExpectedCount"]').fill('0')
    expect(panels.locator('.tab-pane.active')).to_contain_text('Expected: unspecified')
    editor.locator('summary').click()
    editor.locator('[id$="_ExpectedLevels"]').fill('Monday\nTuesday\nWednesday')
    expect(panels.locator('.tab-pane.active')).to_contain_text('Expected: 3')
    editor.locator('summary').click()
    editor.locator('[id$="_Destination"]').fill('Weekday')
    panels.get_by_role('tab', name='Colour', exact=True).click()
    editor.locator('summary').click()
    expect(editor.locator('[id$="_Destination"]')).to_have_value('')
    editor.locator('[id$="_Destination"]').fill('Colour group')
    panels.get_by_role('tab', name='Day', exact=True).click()
    expect(editor.locator('[id$="_Destination"]')).to_have_value('Weekday')
    editor.locator('[id$="_ExpectedLevels"]').fill('Monday\nTuesday\nWednesday\nThursday')
    expect(panels.locator('.tab-pane.active')).to_contain_text('Expected: 4')
    # Automatic saving rebuilds the front; the manual entry must survive.
    editor.locator('summary').click()
    expect(editor.locator('[id$="_Destination"]')).to_have_value('Weekday')
    source_id = editor.locator('select[id$="_ManualSources"]').get_attribute('id')
    page.evaluate("id => document.getElementById(id).selectize.setValue(['Monday'])", source_id)
    editor.locator('[id$="_AddProposal"]').click()
    expect(panels.locator('tr').filter(has_text='Monday → Weekday')).to_be_visible()
    panels.get_by_role('tab', name='Colour', exact=True).click()
    editor.locator('summary').click()
    expect(editor.locator('[id$="_Destination"]')).to_have_value('Colour group')


@pytest.mark.ui
@pytest.mark.parametrize('fixture_name', ['app', 'learned_app'])
def test_stratifier_setting_never_exposes_targets(page, request, fixture_name):
    application = request.getfixturevalue(fixture_name)
    page.goto(application.url)
    panels = page.locator('[id$="-Panels"]')
    expect(panels.get_by_role('tab', name='Day', exact=True)).to_be_visible(timeout=20000)
    card = page.locator('.card').first
    card.hover()
    card.locator('button.collapse-toggle').click()
    page.get_by_label('Include nominal Stratifiers', exact=True).check()
    expect(panels.get_by_role('tab', name='Stratum', exact=True)).to_be_visible()
    expect(panels.get_by_role('tab', name='Outcome', exact=True)).to_have_count(0)
    expect(panels.get_by_role('tab', name='Dual', exact=True)).to_have_count(0)
    panels.get_by_role('tab', name='Stratum', exact=True).click()
    panels.locator('.tab-pane.active tr').filter(has_text='Mon. → Monday').locator('input[value="accept"]').check()
    page.locator('[id$="-Commit"]').click()
    expect(page.locator('[id$="-Status"]')).to_contain_text('1 recorded decisions')
    expect(page.locator('[id$="-TargetProbe"]')).to_have_text('Outcome Mon.: 3; Dual Mon.: 3')
    expect(page.locator('[id$="-PipelineProbe"]')).to_contain_text('replay=True')
    expect(page.locator('[id$="-Status"]')).not_to_contain_text('Target-label')
