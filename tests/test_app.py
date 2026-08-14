import importlib

import pytest
from shiny.pytest import create_app_fixture


@pytest.fixture
def app_module(monkeypatch):
    import app
    importlib.reload(app)
    return app


@pytest.fixture
def sample_config():
    return {
        "version": 1,
        "settings": {
            "section_style": "tab",
            "reuse_cards": True,
            "max_card_height": 400,
            "max_dupl_cards": 10,
        },
        "layout": [
            {
                "section": "Data prep",
                "cards": [
                    {"module": "DataImport"},
                    {"module": "DataTabulation"},
                ],
            },
            {
                "section": "Missing values",
                "cards": [
                    {"module": "DataPlaceholders"},
                ],
            },
        ],
    }


def test_section_id_strips_and_replaces_spaces(app_module):
    assert app_module.section_id("Data prep") == "Data_prep"
    assert app_module.section_id("  Missing values  ") == "Missing_values"


def test_sections_returns_section_names(app_module, monkeypatch, sample_config):
    monkeypatch.setattr(app_module, "config", sample_config)
    assert app_module.sections() == ["Data prep", "Missing values"]


def test_create_sections_tab_returns_one_panel_per_section(
    app_module,
    monkeypatch,
    sample_config,
):
    monkeypatch.setattr(app_module, "config", sample_config)
    panels = app_module.create_sections()
    assert isinstance(panels, list)
    assert len(panels) == 2


def test_create_sections_accordion_returns_single_nav_panel(
    app_module,
    monkeypatch,
    sample_config,
):
    sample_config["settings"]["section_style"] = "accordion"
    monkeypatch.setattr(app_module, "config", sample_config)
    panels = app_module.create_sections()
    assert isinstance(panels, list)
    assert len(panels) == 1


def test_create_sections_unknown_style_returns_empty_list(
    app_module,
    monkeypatch,
    sample_config,
):
    sample_config["settings"]["section_style"] = "nonsense"
    monkeypatch.setattr(app_module, "config", sample_config)
    panels = app_module.create_sections()
    assert panels == []


def test_application_returns_shiny_app(app_module, monkeypatch, sample_config):
    monkeypatch.setattr(app_module, "config", sample_config)
    shiny_app = app_module.application()
    assert shiny_app is not None
    assert shiny_app.__class__.__name__ == "App"


#############################################

app = create_app_fixture("../app.py")

@pytest.mark.playwright

def test_app_loads(page, app):
    page.goto(app.url)
    page.get_by_text("Pythagoras").wait_for()
    assert page.get_by_text("Pythagoras").is_visible()


# Assuming tab mode
@pytest.mark.playwright
def test_tab_sections_exist(page, app):
    page.goto(app.url)
    page.get_by_role("tab", name="Data prep").wait_for()
    page.get_by_role("tab", name="Missing values").wait_for()
    assert page.get_by_role("tab", name="Data prep").is_visible()
    assert page.get_by_role("tab", name="Missing values").is_visible()


@pytest.mark.playwright
def test_navbar_buttons_exist(page, app):
    page.goto(app.url)
    page.locator("#AddCard").wait_for()
    page.locator("#FullScreen").wait_for()
    page.locator("#Quit").wait_for()
    assert page.locator("#AddCard").is_visible()
    assert page.locator("#FullScreen").is_visible()
    assert page.locator("#Quit").is_visible()