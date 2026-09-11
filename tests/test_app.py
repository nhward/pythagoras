from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import ValidationError
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

APP_DIR = Path(__file__).resolve().parent.parent / "app"
APP_FILE = APP_DIR / "app.py"
CONFIG_FILE = APP_DIR / "default.pythagoras.json"
MODULE_NAME = "pythagoras_app_under_test"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


WEB_TEST_ENV = {
    "SHINY_TESTMODE": "1",
    "PYTHAGORAS_TEST_SHOW_START": "false",
}
START_PAGE_TEST_ENV = {
    "SHINY_TESTMODE": "1",
    "PYTHAGORAS_TEST_SHOW_START": "true",
}
TEST_STATE_ROOT = Path(tempfile.mkdtemp(prefix="pythagoras-app-tests-"))
BOOKMARK_TEST_DIR = TEST_STATE_ROOT / "saved-bookmarks"
BOOKMARK_RESTORE_DIR = TEST_STATE_ROOT / "restored-bookmarks"
WEB_TEST_ENV["PYTHAGORAS_TEST_BOOKMARK_DIR"] = str(
    TEST_STATE_ROOT / "ordinary-bookmarks"
)
START_PAGE_TEST_ENV["PYTHAGORAS_TEST_BOOKMARK_DIR"] = str(
    TEST_STATE_ROOT / "start-page-bookmarks"
)
SAVE_TEST_ENV = {
    "SHINY_TESTMODE": "1",
    "PYTHAGORAS_TEST_SHOW_START": "false",
    "PYTHAGORAS_TEST_BOOKMARK_DIR": str(BOOKMARK_TEST_DIR),
}
BOOKMARK_RESTORE_ENV = {
    "SHINY_TESTMODE": "1",
    "PYTHAGORAS_TEST_SHOW_START": "false",
    "PYTHAGORAS_TEST_BOOKMARK_DIR": str(BOOKMARK_RESTORE_DIR),
}

app = create_app_fixture(
    app="../app/app.py",
    scope="function",
    env=WEB_TEST_ENV,
)
start_app = create_app_fixture(
    app="../app/app.py",
    scope="function",
    env=START_PAGE_TEST_ENV,
)
save_app = create_app_fixture(
    app="../app/app.py",
    scope="function",
    env=SAVE_TEST_ENV,
)
bookmark_restore_app = create_app_fixture(
    app="../app/app.py",
    scope="function",
    env=BOOKMARK_RESTORE_ENV,
)
server_bookmark_app = create_app_fixture(
    app="scenarios/bookmark_server.py",
    scope="function",
    env=WEB_TEST_ENV,
)


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_state_root():
    """Remove the unique filesystem state allocated for this test run."""
    yield
    shutil.rmtree(TEST_STATE_ROOT, ignore_errors=True)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "reactive-flow.csv"
    path.write_text(
        "id,value,group\n1,10.25,A\n2,20.75,B\n",
        encoding="utf-8",
    )
    return path


def wait_for_shiny_ready(page: Page) -> None:
    """Wait until bookmark startup has completed and Shiny is idle."""
    page.wait_for_function(
        """
        () => Boolean(
            window.Shiny?.shinyapp?.$inputValues
            && Object.prototype.hasOwnProperty.call(
                window.Shiny.shinyapp.$inputValues,
                "BookmarkStartup",
            )
            && !document.documentElement.classList.contains("shiny-busy")
        )
        """,
        timeout=20_000,
    )


@pytest.fixture(scope="module")
def app_module() -> ModuleType:
    """Load app/app.py explicitly, avoiding the app directory namespace package."""
    spec = importlib.util.spec_from_file_location(MODULE_NAME, APP_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {APP_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_config() -> dict[str, object]:
    return {
        "version": 1,
        "settings": {
            "section_style": "tab",
            "reuse_cards": True,
            "max_card_height": "400px",
            "max_dupl_cards": 10,
        },
        "layout": [
            {
                "section": "Data prep",
                "cards": [
                    {"module": "data_import"},
                    {"module": "data_tabulation"},
                ],
            },
            {
                "section": "Missing values",
                "cards": [{"module": "miss_placeholders"}],
            },
        ],
    }


class TestApplicationHelpers:
    @pytest.mark.unit
    def test_section_normalise_surrounding_and_internal_spaces(self, app_module):
        assert app_module.Module.section_normalise("Data prep") == "Data_prep"
        assert app_module.Module.section_normalise("  Missing values  ") == "Missing_values"

    @pytest.mark.unit
    def test_sections_preserves_configured_order(
        self, app_module, monkeypatch, sample_config
    ):
        monkeypatch.setattr(app_module, "config", sample_config)
        assert app_module.sections() == ["Data prep", "Missing values"]

    @pytest.mark.unit
    def test_start_page_override_is_available_only_in_test_mode(
        self, app_module, monkeypatch, sample_config
    ):
        sample_config["settings"]["show_start"] = True
        monkeypatch.setattr(app_module, "config", sample_config)
        monkeypatch.setenv("PYTHAGORAS_TEST_SHOW_START", "false")
        monkeypatch.delenv("SHINY_TESTMODE", raising=False)
        assert app_module.show_start_enabled() is True
        monkeypatch.setenv("SHINY_TESTMODE", "1")
        assert app_module.show_start_enabled() is False
        monkeypatch.setenv("PYTHAGORAS_TEST_SHOW_START", "true")
        assert app_module.show_start_enabled() is True

    @pytest.mark.unit
    def test_default_configuration_is_reread_from_disk(
        self, app_module, monkeypatch, tmp_path
    ):
        configuration = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        path = tmp_path / "default.pythagoras.json"
        path.write_text(json.dumps(configuration), encoding="utf-8")
        monkeypatch.setattr(app_module, "CONFIG_PATH", path)

        assert app_module.load_default_configuration()["settings"][
            "show_start"
        ] is configuration["settings"]["show_start"]

        configuration["settings"]["show_start"] = not configuration[
            "settings"
        ]["show_start"]
        path.write_text(json.dumps(configuration), encoding="utf-8")

        assert app_module.load_default_configuration()["settings"][
            "show_start"
        ] is configuration["settings"]["show_start"]

    @pytest.mark.unit
    def test_welcome_document_is_embedded(self, app_module):
        markup = str(app_module.welcome())

        assert 'id="welcome-to-pythagoras"' in markup
        assert "Welcome to Pythagoras" in markup
        assert "Pythagoras is the scaffold that holds the cards together." in markup
        assert "<html" not in markup.casefold()
        assert "<head" not in markup.casefold()
        assert "<script" not in markup.casefold()
        assert "FontAwesomeKitConfig" not in markup
        assert '<i class="fa-solid' not in markup
        assert markup.count("<svg") == 4
        assert markup.count("<path") == 4

    @pytest.mark.unit
    def test_unknown_welcome_icon_is_left_unchanged(self, app_module):
        placeholder = '<i class="fa-solid fa-not-a-real-icon"></i>'

        assert app_module.replace_welcome_icons(placeholder) == placeholder

    @pytest.mark.unit
    def test_section_name_is_canonical_unique_and_schema_safe(self, app_module):
        assert app_module.validated_section_name(
            "  Model   review  ", ["Data prep"]
        ) == "Model review"

        for invalid in ("", "Start", "Data prep", "Data-prep", "Data_preparation"):
            with pytest.raises(ValueError):
                app_module.validated_section_name(
                    invalid, ["Data prep", "Data preparation"]
                )

    @pytest.mark.unit
    def test_section_order_inserts_next_to_current(self, app_module):
        original = ("First", "Third")
        assert app_module.inserted_section_order(
            original, current="Third", new="Second", position="before"
        ) == ("First", "Second", "Third")
        assert app_module.inserted_section_order(
            original, current="First", new="Second", position="after"
        ) == ("First", "Second", "Third")

    @pytest.mark.unit
    def test_tab_sections_create_one_panel_and_container_per_section(
        self, app_module, monkeypatch, sample_config
    ):
        monkeypatch.setattr(app_module, "config", sample_config)

        panels = app_module.create_sections()
        markup = "".join(str(panel.content) for panel in panels)

        assert len(panels) == 2
        assert 'id="section_0-cards-container"' in markup
        assert 'id="section_1-cards-container"' in markup

    @pytest.mark.unit
    def test_empty_section_has_creation_invitation(self, app_module):
        panel = app_module.section_panel(
            "section_7", "New section", group_style="tab", empty=True
        )
        markup = str(panel.content)

        assert "This section has no cards." in markup
        assert 'class="btn btn-primary btn-sm section-add-card"' in markup
        assert 'data-section-id="section_7"' in markup

    @pytest.mark.unit
    def test_accordion_sections_share_one_navigation_panel(
        self, app_module, monkeypatch, sample_config
    ):
        sample_config["settings"]["section_style"] = "accordion"
        monkeypatch.setattr(app_module, "config", sample_config)

        panels = app_module.create_sections()
        markup = str(panels[0].content)

        assert len(panels) == 1
        assert 'id="Accordion"' in markup
        assert 'id="section_0-cards-container"' in markup
        assert 'id="section_1-cards-container"' in markup

    @pytest.mark.unit
    def test_unknown_section_style_creates_no_panels(
        self, app_module, monkeypatch, sample_config
    ):
        sample_config["settings"]["section_style"] = "unsupported"
        monkeypatch.setattr(app_module, "config", sample_config)
        assert app_module.create_sections() == []

    @pytest.mark.unit
    def test_application_returns_a_shiny_app(
        self, app_module, monkeypatch, sample_config
    ):
        monkeypatch.setattr(app_module, "config", sample_config)
        shiny_app = app_module.application()
        assert shiny_app.__class__.__name__ == "App"

    @pytest.mark.unit
    def test_configuration_from_card_state_updates_only_visited_sections(
        self, app_module, sample_config
    ):
        candidate = app_module.configuration_from_card_state(
            sample_config,
            visited_sections=["Data prep"],
            section_orders={
                "Data prep": ("data_tabulation", "data_import", "data_import_0"),
            },
            card_modules={
                "data_import": "data_import",
                "data_import_0": "data_import",
                "data_tabulation": "data_tabulation",
            },
        )

        assert candidate["layout"][0]["cards"] == [
            {"module": "data_tabulation"},
            {"module": "data_import"},
            {"module": "data_import"},
        ]
        assert candidate["layout"][1] == sample_config["layout"][1]
        assert sample_config["layout"][0]["cards"][0] == {
            "module": "data_import"
        }

    @pytest.mark.unit
    def test_configuration_save_keeps_data_import_state_with_its_card(
        self, app_module, sample_config
    ):
        state = {
            "inputs": {
                "Navset": "Dataset based",
                "ServerFile": None,
                "LocalFilePath": "",
                "FName": "",
                "Dataset": "sklearn::iris",
                "DName": "iris analysis",
                "Url": "https://example.test/data.csv",
                "UName": "web draft",
                "UciDataset": None,
                "IName": "",
                "Separator": ",",
                "Sheet": 1,
            },
            "last_committed_tab": "Dataset based",
        }
        candidate = app_module.configuration_from_card_state(
            sample_config,
            visited_sections=["Data prep"],
            section_orders={
                "Data prep": ("data_tabulation", "data_import"),
            },
            card_modules={
                "data_import": "data_import",
                "data_tabulation": "data_tabulation",
            },
            card_states={"data_import": state},
        )

        assert candidate["layout"][0]["cards"] == [
            {"module": "data_tabulation"},
            {"module": "data_import", "state": state},
        ]

    @pytest.mark.unit
    def test_bookmark_restore_applies_settings_and_data_import_state(
        self, app_module, sample_config
    ):
        saved = {
            **sample_config,
            "settings": {
                **sample_config["settings"],
                "section_style": "accordion",
                "reuse_cards": False,
                "max_card_height": "725px",
                "max_dupl_cards": 4,
            },
            "layout": [
                {
                    "section": "Different saved layout",
                    "cards": [
                        {
                            "module": "data_import",
                            "state": {
                                "inputs": {"Dataset": "sklearn::iris"},
                                "last_committed_tab": "Dataset based",
                            },
                        },
                        {
                            "module": "miss_map",
                            "state": {"ignored": True},
                        },
                    ],
                }
            ],
        }

        restored = app_module.restore_data_import_only(sample_config, saved)

        assert restored["layout"][0]["section"] == "Data prep"
        assert restored["layout"][0]["cards"][0]["state"] == (
            saved["layout"][0]["cards"][0]["state"]
        )
        assert restored["layout"][1] == sample_config["layout"][1]
        assert restored["settings"] == saved["settings"]

    @pytest.mark.unit
    def test_configuration_writer_validates_before_atomic_replacement(
        self, app_module, sample_config, tmp_path
    ):
        config_path = tmp_path / "pythagoras.json"
        app_module.write_validated_configuration(
            sample_config,
            config_path=config_path,
            schema_path=app_module.SCHEMA_PATH,
        )
        assert json.loads(config_path.read_text(encoding="utf-8")) == sample_config

        original = config_path.read_text(encoding="utf-8")
        invalid = {
            **sample_config,
            "layout": [{"section": "Data prep", "cards": [{"module": "bad-name"}]}],
        }
        with pytest.raises(ValidationError):
            app_module.write_validated_configuration(
                invalid,
                config_path=config_path,
                schema_path=app_module.SCHEMA_PATH,
            )
        assert config_path.read_text(encoding="utf-8") == original

    @pytest.mark.unit
    def test_configuration_save_orders_new_sections_and_drops_empty_ones(
        self, app_module, sample_config
    ):
        candidate = app_module.configuration_from_card_state(
            sample_config,
            section_order=(
                "Data prep",
                "Model review",
                "Empty scratchpad",
                "Missing values",
            ),
            visited_sections=("Model review", "Empty scratchpad"),
            section_orders={
                "Model review": ("data_tabulation_0",),
                "Empty scratchpad": (),
            },
            card_modules={"data_tabulation_0": "data_tabulation"},
        )

        assert [group["section"] for group in candidate["layout"]] == [
            "Data prep",
            "Model review",
            "Missing values",
        ]
        assert candidate["layout"][1] == {
            "section": "Model review",
            "cards": [{"module": "data_tabulation"}],
        }

    @pytest.mark.unit
    def test_configuration_save_persists_a_renamed_section(
        self, app_module, sample_config, tmp_path
    ):
        section_definitions = {
            "section_0": {
                "section": "Input data",
                "cards": sample_config["layout"][0]["cards"],
            },
            "section_1": sample_config["layout"][1],
        }
        candidate = app_module.configuration_from_section_state(
            sample_config,
            section_order=("section_0", "section_1"),
            section_definitions=section_definitions,
            visited_sections=("section_0",),
            section_orders={
                "section_0": ("data_import", "data_tabulation"),
            },
            card_modules={
                "data_import": "data_import",
                "data_tabulation": "data_tabulation",
            },
            show_start=False,
            section_style="accordion",
        )

        config_path = tmp_path / "pythagoras.json"
        app_module.write_validated_configuration(
            candidate,
            config_path=config_path,
            schema_path=app_module.SCHEMA_PATH,
        )
        written = json.loads(config_path.read_text(encoding="utf-8"))

        assert written["layout"][0] == {
            "section": "Input data",
            "cards": [
                {"module": "data_import"},
                {"module": "data_tabulation"},
            ],
        }
        assert written["layout"][1]["section"] == "Missing values"
        assert written["settings"]["section_style"] == "accordion"
        assert sample_config["layout"][0]["section"] == "Data prep"

    @pytest.mark.unit
    def test_configuration_save_persists_show_start(
        self, app_module, sample_config
    ):
        candidate = app_module.configuration_from_card_state(
            sample_config,
            visited_sections=(),
            section_orders={},
            card_modules={},
            show_start=True,
        )

        assert candidate["settings"]["show_start"] is True
        assert "show_start" not in sample_config["settings"]


class TestApplicationBrowser:
    @pytest.mark.ui
    def test_shell_loads(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        expect(page.get_by_text("Pythagoras", exact=True)).to_be_visible()

    @pytest.mark.ui
    def test_configured_tabs_are_visible(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        expect(page.get_by_role("tab", name="Start", exact=True)).to_have_count(0)
        expect(page.locator("#welcome-to-pythagoras")).to_have_count(0)
        configured = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        for group in configured["layout"]:
            expect(page.get_by_role(
                "tab", name=group["section"], exact=True
            )).to_be_visible()

    @pytest.mark.ui
    def test_start_page_renders_welcome_and_defers_card_creation(
        self, page: Page, start_app: ShinyAppProc
    ):
        page.goto(start_app.url)
        wait_for_shiny_ready(page)

        start_tab = page.get_by_role("tab", name="Start", exact=True)
        expect(start_tab).to_be_visible()
        expect(start_tab).to_have_attribute("aria-selected", "true")
        expect(page.locator("#welcome-to-pythagoras")).to_be_visible()
        expect(page.get_by_text(
            "Pythagoras is the scaffold that holds the cards together."
        )).to_be_visible()
        expect(page.locator("#GuideButton")).to_be_visible()
        expect(page.locator("#data_import-Card")).to_have_count(0)

        icon_paths = page.locator(
            "#ManageCardSection svg path, #SaveConfiguration svg path, "
            "#FullScreen svg path, #Quit svg path"
        )
        expect(icon_paths).to_have_count(4)
        assert icon_paths.evaluate_all(
            "elements => elements.every(element => element.getAttribute('d'))"
        )
        welcome_icon_paths = page.locator("#Start-cards-container ol svg path")
        expect(welcome_icon_paths).to_have_count(4)
        assert welcome_icon_paths.evaluate_all(
            "elements => elements.every(element => element.getAttribute('d'))"
        )

        page.locator("#ManageCardSection").click()
        expect(page.locator("#ShowStartSection")).to_be_checked()
        page.locator("#CardPicker_cancel").click()
        expect(page.locator("#data_import-Card")).to_have_count(0)

        page.get_by_role("tab", name="Data prep", exact=True).click()
        expect(page.locator("#data_import-Card")).to_be_attached(timeout=20_000)

    @pytest.mark.ui
    def test_show_start_checkbox_adds_and_removes_start_panel(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        page.locator("#ManageCardSection").click()

        checkbox = page.locator("#ShowStartSection")
        expect(checkbox).not_to_be_checked()
        checkbox.check()
        start_tab = page.get_by_role("tab", name="Start", exact=True)
        expect(start_tab).to_be_visible()
        page.locator("#CardPicker_cancel").click()

        start_tab.click()
        expect(page.locator("#welcome-to-pythagoras")).to_be_visible()
        page.locator("#ManageCardSection").click()
        checkbox = page.locator("#ShowStartSection")
        expect(checkbox).to_be_checked()
        checkbox.uncheck()

        expect(start_tab).to_have_count(0)
        expect(page.get_by_role("tab", name="Data prep", exact=True)).to_have_attribute(
            "aria-selected", "true"
        )

    @pytest.mark.ui
    def test_navigation_actions_are_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        expect(page.locator("#ManageCardSection")).to_be_visible()
        expect(page.locator("#SaveConfiguration")).to_be_visible()
        expect(page.locator("#FullScreen")).to_be_visible()
        expect(page.locator("#Quit")).to_be_visible()
        action_ids = page.locator(
            "#ManageCardSection, #SaveConfiguration, #FullScreen"
        ).evaluate_all("elements => elements.map(element => element.id)")
        assert action_ids == [
            "ManageCardSection",
            "SaveConfiguration",
            "FullScreen",
        ]

    @pytest.mark.ui
    def test_bookmark_button_opens_fixed_manager_card(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        page.locator("#SaveConfiguration").click()
        dialog = page.get_by_role("dialog")
        bookmark_card = dialog.locator("[id$='-Card']")
        expect(bookmark_card).to_be_visible()
        expect(bookmark_card).to_contain_text("Bookmarks")
        expect(bookmark_card.locator(".drag-handle")).to_have_count(0)
        expect(bookmark_card.locator(".close-btn")).to_have_count(0)
        expect(dialog.locator("[id$='-SaveBookmark']")).to_be_visible()

    @pytest.mark.ui
    def test_local_bookmarks_are_selected_by_data_name_then_time(
        self, page: Page, bookmark_restore_app: ShinyAppProc
    ):
        shutil.rmtree(BOOKMARK_RESTORE_DIR, ignore_errors=True)
        BOOKMARK_RESTORE_DIR.mkdir(parents=True)

        def write_bookmark(filename, data_label, *, section_style="tab"):
            candidate = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            candidate["settings"]["section_style"] = section_style
            if section_style == "accordion":
                candidate["settings"]["max_card_height"] = "725px"
            imported = next(
                card
                for group in candidate["layout"]
                for card in group["cards"]
                if card["module"] == "data_import"
            )
            state = imported["state"]
            state["last_committed_tab"] = "Dataset based"
            state["inputs"]["Dataset"] = "sklearn::iris"
            state["inputs"]["DName"] = data_label
            candidate["bookmark"] = {
                "filename": filename,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            (BOOKMARK_RESTORE_DIR / filename).write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            time.sleep(0.05)

        try:
            write_bookmark(
                "Other--20260910T090000+1200.pythagoras.json",
                "older configuration",
                section_style="accordion",
            )
            write_bookmark(
                "Assmnt--20260911T140000+1200.pythagoras.json",
                "earlier Assmnt",
            )
            write_bookmark(
                "Assmnt--20260911T154501+1200.pythagoras.json",
                "latest Assmnt",
            )

            page.goto(bookmark_restore_app.url)
            wait_for_shiny_ready(page)
            expect(page.locator(
                "[id^='data_import'][id$='-Name']"
            ).first).to_contain_text("latest Assmnt", timeout=20_000)
            page.locator("#SaveConfiguration").click()
            dialog = page.get_by_role("dialog")
            data_names = dialog.locator("[id$='-SelectedDataName']")
            saved_times = dialog.locator("[id$='-SelectedBookmarkTime']")

            expect(data_names).to_have_value("Assmnt")
            assert data_names.locator("option").all_text_contents() == [
                "Assmnt",
                "Other",
            ]
            expect(saved_times).to_have_value("20260911T154501+1200")
            assert saved_times.locator("option").all_text_contents()[0] != (
                "20260911T154501+1200"
            )

            data_names.select_option("Other")
            expect(saved_times).to_have_value("20260910T090000+1200")
            dialog.locator("[id$='-LoadBookmark']").click()
            expect(page.locator(
                "[id^='data_import'][id$='-Name']"
            ).first).to_contain_text(
                "older configuration", timeout=20_000
            )
            expect(page.locator("#Accordion")).to_be_visible()
            expect(page.locator(
                "[id^='data_import'][id$='-Card']"
            ).first).to_have_attribute("style", re.compile(r"725px"))
        finally:
            shutil.rmtree(BOOKMARK_RESTORE_DIR, ignore_errors=True)

    @pytest.mark.ui
    def test_refresh_reads_newest_local_bookmark_by_creation_time(
        self, page: Page, bookmark_restore_app: ShinyAppProc
    ):
        shutil.rmtree(BOOKMARK_RESTORE_DIR, ignore_errors=True)
        BOOKMARK_RESTORE_DIR.mkdir(parents=True)

        def write_bookmark(dataset, name, filename, *, section_style="tab"):
            candidate = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            candidate["settings"]["section_style"] = section_style
            imported = next(
                card
                for group in candidate["layout"]
                for card in group["cards"]
                if card["module"] == "data_import"
            )
            state = imported["state"]
            state["last_committed_tab"] = "Dataset based"
            state["inputs"]["Dataset"] = f"sklearn::{dataset}"
            state["inputs"]["DName"] = name
            candidate["bookmark"] = {
                "filename": filename,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            (BOOKMARK_RESTORE_DIR / filename).write_text(
                json.dumps(candidate), encoding="utf-8"
            )

        try:
            write_bookmark(
                "iris", "first iris", "iris--first.pythagoras.json"
            )
            page.goto(bookmark_restore_app.url)
            wait_for_shiny_ready(page)
            expect(page.locator(
                "[id^='data_import'][id$='-Name']"
            ).first).to_contain_text(
                "first iris", timeout=20_000
            )

            time.sleep(0.05)
            write_bookmark(
                "wine",
                "newer wine",
                "wine--newer.pythagoras.json",
                section_style="accordion",
            )
            page.reload()
            wait_for_shiny_ready(page)
            expect(page.locator(
                "[id^='data_import'][id$='-Name']"
            ).first).to_contain_text(
                "newer wine", timeout=20_000
            )
            expect(page.locator("#Accordion")).to_be_visible()
        finally:
            shutil.rmtree(BOOKMARK_RESTORE_DIR, ignore_errors=True)

    @pytest.mark.ui
    def test_server_bookmark_is_restored_from_browser_after_refresh(
        self, page: Page, server_bookmark_app: ShinyAppProc
    ):
        page.goto(server_bookmark_app.url)
        wait_for_shiny_ready(page)
        page.get_by_role("tab", name="Dataset based", exact=True).click()
        page.evaluate(
            """
            Shiny.setInputValue(
                "data_import-Dataset",
                "sklearn::iris",
                {priority: "event"},
            );
            """
        )
        expect(page.locator("#data_import-DName")).to_have_value("iris")
        page.locator("#data_import-DName").fill("browser iris")
        expect(page.locator("#data_import-Commit")).to_be_enabled()
        page.locator("#data_import-Commit").click()
        expect(page.locator("#data_import-Name")).to_contain_text(
            "browser iris", timeout=20_000
        )

        page.locator("#SaveConfiguration").click()
        dialog = page.get_by_role("dialog")
        dialog.get_by_role(
            "button", name="Save bookmark", exact=False
        ).click()
        expect(page.get_by_text("Bookmark saved as", exact=False)).to_be_visible()
        data_names = dialog.locator("[id$='-SelectedDataName']")
        saved_times = dialog.locator("[id$='-SelectedBookmarkTime']")
        expect(data_names).to_have_value("browser-iris")
        expect(saved_times).to_have_value(
            re.compile(r"^\d{8}T\d{6}[+-]\d{4}$")
        )
        assert saved_times.locator("option").first.text_content() != (
            saved_times.input_value()
        )

        page.evaluate(
            """
            async () => {
                const database = await new Promise((resolve, reject) => {
                    const request = indexedDB.open("pythagoras-bookmarks", 1);
                    request.onsuccess = () => resolve(request.result);
                    request.onerror = () => reject(request.error);
                });
                const records = await new Promise((resolve, reject) => {
                    const request = database.transaction(
                        "bookmarks", "readonly",
                    ).objectStore("bookmarks").getAll();
                    request.onsuccess = () => resolve(request.result);
                    request.onerror = () => reject(request.error);
                });
                records[0].configuration.settings.section_style = "accordion";
                await new Promise((resolve, reject) => {
                    const request = database.transaction(
                        "bookmarks", "readwrite",
                    ).objectStore("bookmarks").put(records[0]);
                    request.onsuccess = () => resolve();
                    request.onerror = () => reject(request.error);
                });
                database.close();
            }
            """
        )

        page.reload()
        wait_for_shiny_ready(page)
        expect(page.locator(
            "[id^='data_import'][id$='-Name']"
        ).first).to_contain_text("browser iris", timeout=20_000)
        expect(page.locator("#Accordion")).to_be_visible(timeout=20_000)

    @pytest.mark.ui
    def test_empty_section_can_add_a_card_and_then_be_deleted(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        page.get_by_role("tab", name="Data prep", exact=True).click()
        page.locator("#ManageCardSection").click()

        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_text("The current section: Data prep")).to_be_visible()
        dialog.get_by_role("tab", name="New section", exact=True).click()
        page.locator("#NewSectionName").fill("Model review")
        dialog.get_by_role("button", name="Add section", exact=True).click()

        new_tab = page.get_by_role("tab", name="Model review", exact=True)
        expect(new_tab).to_be_visible()
        expect(new_tab).to_have_attribute("aria-selected", "true")
        empty_state = page.locator(".section-empty-state:visible")
        expect(empty_state).to_be_visible()
        delete_section = page.get_by_role(
            "button", name="Delete empty section Model review", exact=True
        )
        expect(delete_section).to_be_visible()

        empty_state.locator(".section-add-card").click()
        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_role("tab", name="New card", exact=True)).to_have_class(
            "nav-link active"
        )
        page.locator("#CardPicker_selected").select_option("data_tabulation")
        dialog.get_by_role("button", name="Add card", exact=True).click()

        added_card = page.locator(
            ".cards-grid[data-section-id] > .card[id^='data_tabulation_']"
        )
        expect(added_card).to_have_count(1)
        expect(empty_state).to_be_hidden()
        expect(delete_section).to_be_hidden()

        added_card.hover()
        added_card.locator(".close-btn").click(force=True)
        page.get_by_role("dialog").get_by_role(
            "button", name="Yes, remove"
        ).click()
        expect(added_card).to_have_count(0)
        expect(delete_section).to_be_visible()

        delete_section.click()
        expect(new_tab).to_have_count(0)
        expect(page.get_by_role("tab", name="Data prep", exact=True)).to_have_attribute(
            "aria-selected", "true"
        )

    @pytest.mark.ui
    def test_section_rename_preserves_cards_and_reactive_flow(
        self, page: Page, app: ShinyAppProc, csv_file
    ):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        original_container = page.locator("#section_0-cards-container")
        expect(original_container.locator("#data_import-Card")).to_be_attached(timeout=20_000)
        page.locator("#ManageCardSection").click()
        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_text("The current section: Data prep")).to_be_visible()
        dialog.get_by_role("tab", name="Rename section", exact=True).click()
        expect(dialog.get_by_role("button", name="Rename section", exact=True)).to_be_visible()
        page.locator("#RenameSectionName").fill("Input data")
        dialog.get_by_role("button", name="Rename section", exact=True).click()
        expect(page.get_by_role("tab", name="Input data", exact=True)).to_be_visible()
        expect(page.get_by_role("tab", name="Data prep", exact=True)).to_have_count(0)
        expect(original_container.locator("#data_import-Card")).to_be_attached()
        # page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        # page.locator("#data_import-Commit").click()
        # expect(page.locator("#var_modify-Name")).to_contain_text("reactive-flow", timeout=20_000)

    @pytest.mark.ui
    def test_renamed_section_is_written_by_save_button(
        self, page: Page, save_app: ShinyAppProc
    ):
        shutil.rmtree(BOOKMARK_TEST_DIR, ignore_errors=True)
        try:
            page.goto(save_app.url)
            wait_for_shiny_ready(page)
            expect(page.locator("#data_import-Card")).to_be_attached(
                timeout=20_000
            )
            page.locator("#ManageCardSection").click()
            dialog = page.get_by_role("dialog")
            dialog.get_by_role("tab", name="Rename section", exact=True).click()
            page.locator("#RenameSectionName").fill("Input data")
            dialog.get_by_role(
                "button", name="Rename section", exact=True
            ).click()
            expect(page.get_by_role(
                "tab", name="Input data", exact=True
            )).to_be_visible()

            page.locator("#ManageCardSection").click()
            dialog = page.get_by_role("dialog")
            dialog.get_by_label("Accordion", exact=True).check()
            expect(dialog.get_by_label(
                "Accordion", exact=True
            )).to_be_checked()
            dialog.get_by_role("button", name="Cancel", exact=True).click()
            expect(page.get_by_role(
                "tab", name="Input data", exact=True
            )).to_be_visible()

            page.locator("#SaveConfiguration").click()
            dialog = page.get_by_role("dialog")
            dialog.locator("[id$='-SaveBookmark']").click()
            expect(page.get_by_text(
                "Bookmark saved as", exact=False
            )).to_be_visible()

            saved = list(BOOKMARK_TEST_DIR.glob("*.pythagoras.json"))
            assert len(saved) == 1
            written = json.loads(saved[0].read_text(encoding="utf-8"))
            assert written["layout"][0]["section"] == "Input data"
            assert all(
                group["section"] != "Data prep"
                for group in written["layout"]
            )
            assert written["settings"]["section_style"] == "accordion"
        finally:
            shutil.rmtree(BOOKMARK_TEST_DIR, ignore_errors=True)

    @pytest.mark.ui
    def test_save_button_writes_data_import_inputs_and_committed_tab(
        self, page: Page, save_app: ShinyAppProc, csv_file
    ):
        shutil.rmtree(BOOKMARK_TEST_DIR, ignore_errors=True)
        try:
            page.goto(save_app.url)
            wait_for_shiny_ready(page)
            # expect(page.locator("#data_import-ServerFile")).to_be_attached(
            #     timeout=20_000
            # )
            # page.locator("#data_import-ServerFile").set_input_files(
            #     str(csv_file)
            # )
            expect(page.locator("#data_import-Commit")).to_be_enabled()
            page.locator("#data_import-Commit").click()
            expect(page.locator("#data_import-Check")).to_contain_text(
                "File import successful"
            )

            page.get_by_role("tab", name="Web based", exact=True).click()
            page.locator("#data_import-UName").fill("uncommitted web draft")
            page.locator("#SaveConfiguration").click()
            dialog = page.get_by_role("dialog")
            dialog.locator("[id$='-SaveBookmark']").click()
            expect(page.get_by_text(
                "Bookmark saved as", exact=False
            )).to_be_visible()

            saved = list(BOOKMARK_TEST_DIR.glob("*.pythagoras.json"))
            assert len(saved) == 1
            written = json.loads(saved[0].read_text(encoding="utf-8"))
            data_import = next(
                card
                for group in written["layout"]
                for card in group["cards"]
                if card["module"] == "data_import"
            )
            state = data_import["state"]
            assert state["last_committed_tab"] == "File based"
            assert state["inputs"]["Navset"] == "Web based"
            assert state["inputs"]["FName"] == "Assmnt"
            assert state["inputs"]["UName"] == "uncommitted web draft"
            assert state["inputs"]["ServerFile"][0]["name"] == ("Assmnt.csv")
        finally:
            shutil.rmtree(BOOKMARK_TEST_DIR, ignore_errors=True)

    @pytest.mark.ui
    def test_committed_data_reacts_through_cards_and_section_boundary(
        self, page: Page, app: ShinyAppProc, csv_file,
    ):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        # expect(page.locator("#data_import-ServerFile")).to_be_attached(timeout=20_000)
        # page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        # expect(page.locator("#data_import-Commit")).to_be_enabled()
        # page.locator("#data_import-Commit").click()
        # for namespace in ("data_tabulation", "role_assignment", "var_modify"):
        #     expect(page.locator(f"#{namespace}-Name")).to_contain_text("reactive-flow", timeout=20_000)
        # page.get_by_role("tab", name="Data cleaning", exact=True).click()
        # expect(page.locator("#obs_duplicates-Name")).to_contain_text("reactive-flow", timeout=20_000)

    @pytest.mark.ui
    def test_removing_a_module_reconnects_the_reactive_chain(
        self, page: Page, app: ShinyAppProc, csv_file, tmp_path,
    ):
        page.goto(app.url)
        wait_for_shiny_ready(page)
        # page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        expect(page.locator("#data_import-Commit")).to_be_enabled()
        page.locator("#data_import-Commit").click()
        # expect(page.locator("#var_modify-Name")).to_contain_text("reactive-flow", timeout=20_000)

        page.locator("#role_assignment-Card").hover()
        page.locator("#role_assignment-CloseButton").click(force=True)
        page.get_by_role("dialog").get_by_role("button", name="Yes, remove").click()
        expect(page.locator("#role_assignment-Card")).to_have_count(0)

        # replacement = tmp_path / "after-removal.csv"
        # replacement.write_text("id,value\n1,100\n2,200\n", encoding="utf-8")
        # # page.locator("#data_import-ServerFile").set_input_files(str(replacement))
        # expect(page.locator("#data_import-Commit")).to_be_enabled()
        # page.locator("#data_import-Commit").click()

        # expect(page.locator("#var_modify-Name")).to_contain_text("after-removal", timeout=20_000)
