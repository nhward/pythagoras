import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app='scenarios/late_cleaning.py', scope='function')


@pytest.mark.ui
def test_late_cleaning_passes_through_and_recovers(page, app):
    page.goto(app.url)
    notice = page.locator('[id$="-PositionNotice"]')
    result = page.locator('[id$="-Result"]')
    expect(notice).to_contain_text('Card disabled in this position', timeout=20000)
    expect(notice).to_contain_text('Move this card before the first learning step')
    expect(result).to_have_text("columns=['x']; pipeline=True")
    page.get_by_label('Learning step upstream').uncheck()
    expect(notice).to_have_text('')
    expect(result).to_have_text("columns=['x', 'shadow__x']; pipeline=False")
    page.get_by_label('Learning step upstream').check()
    expect(notice).to_contain_text('Card disabled in this position')
    expect(result).to_have_text("columns=['x']; pipeline=True")


incompatible_app = create_app_fixture(app='scenarios/pipeline_compatibility.py', scope='function')


@pytest.mark.ui
def test_declared_incompatibility_prevents_pipeline_replacement(page, incompatible_app):
    page.goto(incompatible_app.url)
    notice = page.locator('[id$="-PositionNotice"]')
    result = page.locator('[id$="-Result"]')
    calls = page.locator('[id$="-Calls"]')
    expect(notice).to_contain_text('not pipeline-compatible', timeout=20000)
    expect(result).to_have_text("columns=['x']; pipeline=True")
    expect(calls).to_have_text('0')
    page.get_by_label('Learning step upstream').uncheck()
    expect(notice).to_be_empty()
    expect(result).to_have_text("columns=['replacement']; pipeline=False")
    expect(calls).to_have_text('1')
    page.get_by_label('Learning step upstream').check()
    expect(notice).to_contain_text('not pipeline-compatible')
    expect(result).to_have_text("columns=['x']; pipeline=True")
    expect(calls).to_have_text('1')
