from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import ValidationError
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

APP_DIR = Path(__file__).resolve().parent.parent / "app"
APP_FILE = APP_DIR / "app.py"
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


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "reactive-flow.csv"
    path.write_text(
        "id,value,group\n1,10.25,A\n2,20.75,B\n",
        encoding="utf-8",
    )
    return path


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
        self, app_module, sample_config
    ):
        candidate = app_module.configuration_from_card_state(
            sample_config,
            section_order=("Input data", "Missing values"),
            visited_sections=("Input data",),
            section_orders={
                "Input data": ("data_import", "data_tabulation"),
            },
            card_modules={
                "data_import": "data_import",
                "data_tabulation": "data_tabulation",
            },
        )

        assert candidate["layout"][0] == {
            "section": "Input data",
            "cards": [
                {"module": "data_import"},
                {"module": "data_tabulation"},
            ],
        }

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
        expect(page.get_by_text("Pythagoras", exact=True)).to_be_visible()

    @pytest.mark.ui
    def test_configured_tabs_are_visible(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(page.get_by_role("tab", name="Start", exact=True)).to_have_count(0)
        expect(page.locator("#welcome-to-pythagoras")).to_have_count(0)
        expect(page.get_by_role("tab", name="Data prep")).to_be_visible()
        expect(page.get_by_role("tab", name="Missing values")).to_be_visible()

    @pytest.mark.ui
    def test_start_page_renders_welcome_and_defers_card_creation(
        self, page: Page, start_app: ShinyAppProc
    ):
        page.goto(start_app.url)

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
    def test_empty_section_can_add_a_card_and_then_be_deleted(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
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
        original_container = page.locator("#section_0-cards-container")
        expect(original_container.locator("#data_import-Card")).to_be_attached(
            timeout=20_000
        )

        page.locator("#ManageCardSection").click()
        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_text("The current section: Data prep")).to_be_visible()
        dialog.get_by_role("tab", name="Rename section", exact=True).click()
        expect(dialog.get_by_role(
            "button", name="Rename section", exact=True
        )).to_be_visible()
        page.locator("#RenameSectionName").fill("Input data")
        dialog.get_by_role("button", name="Rename section", exact=True).click()

        expect(page.get_by_role("tab", name="Input data", exact=True)).to_be_visible()
        expect(page.get_by_role("tab", name="Data prep", exact=True)).to_have_count(0)
        expect(original_container.locator("#data_import-Card")).to_be_attached()

        page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        page.locator("#data_import-Commit").click()
        expect(page.locator("#var_modify-Name")).to_contain_text(
            "reactive-flow", timeout=20_000
        )

    @pytest.mark.ui
    def test_committed_data_reacts_through_cards_and_section_boundary(
        self, page: Page, app: ShinyAppProc, csv_file,
    ):
        page.goto(app.url)
        expect(page.locator("#data_import-ServerFile")).to_be_attached(
            timeout=20_000,
        )
        page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        expect(page.locator("#data_import-Commit")).to_be_enabled()
        page.locator("#data_import-Commit").click()

        for namespace in ("data_tabulation", "role_assignment", "var_modify"):
            expect(page.locator(f"#{namespace}-Name")).to_contain_text(
                "reactive-flow", timeout=20_000,
            )

        page.get_by_role("tab", name="Data cleaning", exact=True).click()
        expect(page.locator("#obs_duplicates-Name")).to_contain_text(
            "reactive-flow", timeout=20_000,
        )

    @pytest.mark.ui
    def test_removing_a_module_reconnects_the_reactive_chain(
        self, page: Page, app: ShinyAppProc, csv_file, tmp_path,
    ):
        page.goto(app.url)
        page.locator("#data_import-ServerFile").set_input_files(str(csv_file))
        expect(page.locator("#data_import-Commit")).to_be_enabled()
        page.locator("#data_import-Commit").click()
        expect(page.locator("#var_modify-Name")).to_contain_text(
            "reactive-flow", timeout=20_000,
        )

        page.locator("#role_assignment-Card").hover()
        page.locator("#role_assignment-CloseButton").click(force=True)
        page.get_by_role("dialog").get_by_role(
            "button", name="Yes, remove",
        ).click()
        expect(page.locator("#role_assignment-Card")).to_have_count(0)

        replacement = tmp_path / "after-removal.csv"
        replacement.write_text("id,value\n1,100\n2,200\n", encoding="utf-8")
        page.locator("#data_import-ServerFile").set_input_files(str(replacement))
        expect(page.locator("#data_import-Commit")).to_be_enabled()
        page.locator("#data_import-Commit").click()

        expect(page.locator("#var_modify-Name")).to_contain_text(
            "after-removal", timeout=20_000,
        )
