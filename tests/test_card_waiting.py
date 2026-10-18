import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app="scenarios/card_waiting.py", scope="function")


@pytest.mark.ui
def test_waiting_status_recovers_suspends_and_preserves_errors(page, app):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(app.url)
    status = page.locator('[id$="-PipelineBusy"]')
    print(status)
    expect(status).to_contain_text("Awaiting upstream data", timeout=15000)
    expect(status.locator(".spinner-border")).to_have_count(1)
    page.get_by_label("ready", exact=True).check()
    expect(page.locator('[id$="-Result"]')).to_have_text("rows=1")
    expect(status).to_have_text("")
    page.get_by_label("pending", exact=True).check()
    expect(status).to_contain_text("Awaiting upstream data")
    page.get_by_label("Active", exact=True).uncheck()
    expect(status).to_have_text("")
    page.get_by_label("Active", exact=True).check()
    expect(status).to_contain_text("Awaiting upstream data")
    page.get_by_label("error", exact=True).check()
    expect(page.locator('[id$="-Result"]')).to_contain_text("upstream failed")
    expect(status).to_have_text("")
    assert errors == []
