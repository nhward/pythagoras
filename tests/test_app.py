from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc


APP_DIR = Path(__file__).resolve().parent.parent / "app"
APP_FILE = APP_DIR / "app.py"
MODULE_NAME = "pythagoras_app_under_test"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


app = create_app_fixture(app="../app/app.py", scope="function")


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
    def test_section_id_normalises_surrounding_and_internal_spaces(self, app_module):
        assert app_module.section_id("Data prep") == "Data_prep"
        assert app_module.section_id("  Missing values  ") == "Missing_values"

    @pytest.mark.unit
    def test_sections_preserves_configured_order(
        self, app_module, monkeypatch, sample_config
    ):
        monkeypatch.setattr(app_module, "config", sample_config)
        assert app_module.sections() == ["Data prep", "Missing values"]

    @pytest.mark.unit
    def test_tab_sections_create_one_panel_and_container_per_section(
        self, app_module, monkeypatch, sample_config
    ):
        monkeypatch.setattr(app_module, "config", sample_config)

        panels = app_module.create_sections()
        markup = "".join(str(panel.content) for panel in panels)

        assert len(panels) == 2
        assert 'id="Data_prep-cards-container"' in markup
        assert 'id="Missing_values-cards-container"' in markup

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
        assert 'id="Data_prep-cards-container"' in markup
        assert 'id="Missing_values-cards-container"' in markup

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


class TestApplicationBrowser:
    @pytest.mark.ui
    def test_shell_loads(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(page.get_by_text("Pythagoras", exact=True)).to_be_visible()

    @pytest.mark.ui
    def test_configured_tabs_are_visible(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(page.get_by_role("tab", name="Data prep")).to_be_visible()
        expect(page.get_by_role("tab", name="Missing values")).to_be_visible()

    @pytest.mark.ui
    def test_navigation_actions_are_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(page.locator("#AddCard")).to_be_visible()
        expect(page.locator("#FullScreen")).to_be_visible()
        expect(page.locator("#Quit")).to_be_visible()
