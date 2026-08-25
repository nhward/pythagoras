import importlib
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

path = Path(__file__).resolve().parent.parent.parent / 'app'
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../../app/cards/sys_configuration.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.sys_configuration")


def get_card(page: Page):
    return page.locator(".card").first


def get_namespace(page: Page) -> str:
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return card_id.partition("-")[0]


def namespaced_id(page: Page, local_id: str) -> str:
    return f"{get_namespace(page)}-{local_id}"


def by_id(page: Page, local_id: str):
    return page.locator(f"#{namespaced_id(page, local_id)}")


def hover_card(page: Page):
    get_card(page).hover()


def recorded_helpers(card_module):
    """Expose server helpers without starting a reactive session."""
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def record_code(function):
        functions[function.__name__] = function
        return function

    card.record_code = record_code
    card.capture_print = lambda function: function
    inputs = SimpleNamespace(Refresh=lambda: 0)
    card.server(inputs, lambda function: function, None)
    return card, functions


class TestInstance:
    @pytest.mark.unit
    def test_metadata(self, card_module):
        card = card_module.this
        assert card.name == "sys_configuration"
        assert card.long_name == "Configuration"
        assert "host-system configuration" in card.description
        assert not card.mutable

    @pytest.mark.unit
    def test_expected_ui_regions(self, card_module):
        card = card_module.this
        assert card.front is not None
        assert card.back is not None
        assert card.footer is not None
        assert card.settings is None
        assert card.hasFlipSide()
        assert card.hasFooter()
        assert not card.hasSidebar()

    @pytest.mark.unit
    def test_front_contains_all_tabs_and_outputs(self, card_module):
        html = str(card_module.this.front.tagify())
        for label in ("Summary", "Url", "Packages", "Folders"):
            assert label in html
        for output_id in ("Summary", "Url", "Packages", "Folders"):
            assert f'id="{output_id}"' in html

    @pytest.mark.unit
    def test_back_contains_session_output(self, card_module):
        html = str(card_module.this.back)
        assert 'id="Session"' in html

    @pytest.mark.unit
    def test_footer_contains_refresh_control(self, card_module):
        html = str(card_module.this.footer)
        assert 'id="Refresh"' in html
        assert "Refresh" in html


class TestRecordedHelpers:
    @pytest.mark.unit
    def test_loaded_packages_is_nonempty_dataframe(self, card_module):
        _, functions = recorded_helpers(card_module)
        result = functions["get_loaded_packages"]()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.columns.tolist() == ["Loaded package", "Version"]
        assert result["Version"].map(lambda value: isinstance(value, str)).all()

    @pytest.mark.unit
    def test_loaded_packages_are_sorted_case_insensitively(self, card_module):
        _, functions = recorded_helpers(card_module)
        result = functions["get_loaded_packages"]()
        names = result["Loaded package"].tolist()
        assert names == sorted(names, key=str.casefold)
        assert result.index.equals(pd.RangeIndex(len(result)))

    @pytest.mark.unit
    def test_packages_output_delegates_to_loaded_packages(self, card_module):
        _, functions = recorded_helpers(card_module)
        result = functions["Packages"]()
        assert result.columns.tolist() == ["Loaded package", "Version"]
        assert not result.empty

    @pytest.mark.unit
    def test_folders_output_has_expected_schema_and_rows(self, card_module):
        _, functions = recorded_helpers(card_module)
        result = functions["Folders"]()
        assert result.columns.tolist() == ["Name", "Path", "Files"]
        assert result["Name"].tolist() == ["home", "www", "markdown", "cards"]
        assert result["Path"].map(lambda value: isinstance(value, str)).all()
        assert result["Files"].map(lambda value: isinstance(value, int)).all()


class TestWebKitTables:
    @pytest.mark.ui
    def test_card_and_summary_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Configuration", exact=True)).to_be_visible()
        table = controller.OutputTable(page, namespaced_id(page, "Summary"))
        table.expect_ncol(2)
        table.expect_column_labels(["Property", "Value"])

    @pytest.mark.ui
    def test_summary_contains_core_properties(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        summary = by_id(page, "Summary")
        for label in (
            "Running locally",
            "Python executable",
            "Python version",
            "Platform",
            "Installed packages",
            "Loaded packages",
        ):
            expect(summary).to_contain_text(label)

    @pytest.mark.ui
    def test_url_table_structure(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        page.get_by_role("tab", name="Url", exact=True).click()
        table = controller.OutputTable(page, namespaced_id(page, "Url"))
        table.expect_ncol(2)
        table.expect_column_labels(["Property", "Value"])
        expect(by_id(page, "Url")).to_contain_text("Host name")
        expect(by_id(page, "Url")).to_contain_text("Protocol")

    @pytest.mark.ui
    def test_packages_table_structure(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        page.get_by_role("tab", name="Packages", exact=True).click()
        table = controller.OutputTable(page, namespaced_id(page, "Packages"))
        table.expect_ncol(2)
        table.expect_column_labels(["Loaded package", "Version"])

    @pytest.mark.ui
    def test_folders_table_structure(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        page.get_by_role("tab", name="Folders", exact=True).click()
        table = controller.OutputTable(page, namespaced_id(page, "Folders"))
        table.expect_ncol(3)
        table.expect_column_labels(["Name", "Path", "Files"])
        expect(by_id(page, "Folders")).to_contain_text("markdown")
        expect(by_id(page, "Folders")).to_contain_text("cards")

    @pytest.mark.ui
    def test_switching_between_tabs_keeps_outputs_available(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        for tab, output_id in (
            ("Url", "Url"),
            ("Packages", "Packages"),
            ("Folders", "Folders"),
            ("Summary", "Summary"),
        ):
            page.get_by_role("tab", name=tab, exact=True).click()
            expect(by_id(page, output_id)).to_be_visible()


class TestWebKitCardControls:
    @pytest.mark.ui
    def test_refresh_keeps_summary_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "Refresh").click()
        table = controller.OutputTable(page, namespaced_id(page, "Summary"))
        table.expect_ncol(2)

    @pytest.mark.ui
    def test_flip_reveals_session_information(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        hover_card(page)
        by_id(page, "FlipButton").click(force=True)
        session = by_id(page, "Session")
        expect(session).to_be_visible()
        expect(session).to_contain_text(re.compile(r"Session information updated at"))

    @pytest.mark.ui
    def test_code_button_opens_modal(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        hover_card(page)
        by_id(page, "CodeButton").click(force=True)
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        expect(dialog).to_contain_text("Configuration code")
        dialog.get_by_role("button", name="Dismiss").click()
        expect(dialog).to_be_hidden()

    @pytest.mark.ui
    def test_information_button_when_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        hover_card(page)
        button = by_id(page, "InfoButton")
        if button.count() == 0:
            pytest.skip("No configuration markdown file is installed")
        button.click(force=True)
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        expect(dialog).to_contain_text("Configuration")

    @pytest.mark.ui
    def test_expand_and_restore_keep_summary_working(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        hover_card(page)
        by_id(page, "ExpandButton").click(force=True)
        restore = by_id(page, "ContractButton")
        expect(restore).to_be_visible()
        restore.click(force=True)
        table = controller.OutputTable(page, namespaced_id(page, "Summary"))
        table.expect_ncol(2)

    @pytest.mark.ui
    def test_close_confirmation_removes_card(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        hover_card(page)
        by_id(page, "CloseButton").click(force=True)
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        dialog.get_by_role("button", name="Yes, remove").click()
        expect(page.locator('#cards-container > [id$="Card"]')).to_have_count(0)
