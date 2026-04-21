from shiny.pytest import create_app_fixture
from cards.Configuration import instance
from shiny.run import ShinyAppProc
from playwright.sync_api import Page
from shiny.playwright import controller
from tests.utils import find_id, hover_solo_card
import pytest
import re

card = instance()
card_name = card.long_name
app = create_app_fixture(app = "../cards/Configuration.py")

@pytest.fixture(scope="session")
def browser_context_args():
    return {
        "viewport": {"width": 1600, "height": 1000},
    }

@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {
        "headless": False,
        "slow_mo": 200,
    }

@pytest.mark.unit
def test_get_loaded_packages_non_empty():
    """get_loaded_packages() should return a non-empty data frame."""
    # grab the inner helper via record_code or refactor it out
    funcs = {}
    def record_code(fn):
        funcs[fn.__name__] = fn
        return fn

    card.record_code = record_code
    # call card.server with simple fakes
    card.server(None, lambda f: f, None)
    df = funcs["get_loaded_packages"]()
    assert not df.empty


@pytest.mark.ui
def test_summary_table_has_expected_properties(page: Page, app: ShinyAppProc):
    """Summary table should show exactly two columns: Property and Value."""
    page.goto(app.url)
    page.get_by_role("tab", name="Summary").click()
    sum_tbl = controller.OutputTable(page, find_id(page, "Summary"))
    sum_tbl.expect_ncol(2)  # Property / Value
    sum_tbl.expect_column_labels(["Property", "Value"])


@pytest.mark.ui
def test_url_table_has_expected_properties(page: Page, app: ShinyAppProc):
    """URL table should show exactly two columns: Property and Value."""
    page.goto(app.url)
    page.get_by_role("tab", name="Url").click()
    url_tbl = controller.OutputTable(page, find_id(page, "Url"))
    url_tbl.expect_ncol(2)  # Property / Value
    url_tbl.expect_column_labels(["Property", "Value"])


@pytest.mark.ui
def test_packages_table_structure(page: Page, app: ShinyAppProc):
    """URL table should show exactly two columns: Loaded package, Version."""
    page.goto(app.url)
    page.get_by_role("tab", name="Packages").click()
    pkg_tbl = controller.OutputTable(page, find_id(page, "Packages"))
    pkg_tbl.expect_ncol(2)
    pkg_tbl.expect_column_labels(["Loaded package", "Version"])


@pytest.mark.ui
def test_folders_table_has_name_path_files(page: Page, app: ShinyAppProc):
    """Folders table should show exactly three columns: Name, Path, Files."""
    page.goto(app.url)
    page.get_by_role("tab", name="Folders").click()
    fld_tbl = controller.OutputTable(page, find_id(page, "Folders"))
    fld_tbl.expect_ncol(3)
    fld_tbl.expect_column_labels(["Name", "Path", "Files"])


@pytest.mark.ui
def test_flip_button_switches_to_back(page, app):
    """Flipping the card should show the expected text"""
    page.goto(app.url)
    hover_solo_card(page)
    page.get_by_role("button", name = "Flip this card").click()
    loc = page.get_by_text(re.compile(r"Session information updated at"))
    loc.wait_for(state="visible")  # make Playwright wait for the text to appear
    assert loc.is_visible()
    page.get_by_role("button", name = "Flip this card").click()
    # loc.wait_for(state="hidden")  # make Playwright wait for the text to appear
    # assert loc.is_hidden()

@pytest.mark.ui
def test_info_button_opens_modal_if_markdown_exists(page, app):
    page.goto(app.url)
    hover_solo_card(page)
    btn = page.get_by_role("button", name="Information about this card")
    if btn.count() == 0:
        pytest.skip("No markdown file present → no Info button")
    btn.click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(state="visible")
    assert dialog.is_visible()
    assert "Configuration" in dialog.inner_text()
    page.get_by_role("button", name = "Dismiss").click()


def test_code_button_opens_code_modal(page, app):
    page.goto(app.url)
    hover_solo_card(page)
    page.get_by_role("button", name="View the code associated with this card").click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(state="visible")
    assert dialog.is_visible()
    assert f"{card_name} code" in dialog.inner_text()
    page.get_by_role("button", name = "Dismiss").click()


def test_expand_and_contract_keep_card_working(page, app):
    page.goto(app.url)
    hover_solo_card(page)
    page.get_by_role("button", name="Expand this card").click()
    contract_btn = page.get_by_role("button", name="Restore this card")
    contract_btn.wait_for(state="visible")
    assert contract_btn.is_visible()
    contract_btn.click()
    # Check Summary still functioning
    sum_tbl = controller.OutputTable(page, find_id(page, "Summary"))
    sum_tbl.expect_ncol(2)  # Property / Value
    sum_tbl.expect_column_labels(["Property", "Value"])


def test_close_button_hides_card(page, app):
    page.goto(app.url)
    hover_solo_card(page)
    close_btn = page.get_by_role("button", name="Close this card")
    close_btn.click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(state="visible")
    yes = dialog.get_by_role("button", name = "Yes, remove")
    yes.click()
    assert(page.locator('#cards-container > [id$="Card"]').count() == 0)
