import pytest
import asyncio
from unittest.mock import AsyncMock
from pathlib import Path
from card import Card

#############################
# Fixtures
#############################
@pytest.fixture
def card():
    """Basic Card with minimal constructor."""
    return Card(name="TestCard")


@pytest.fixture
def temp_md_file(tmp_path):
    """Create a temporary markdown file for 'information()' tests."""
    md = tmp_path / "TestCard.md"
    md.write_text("# Heading\nSome **markdown** text.", encoding="utf-8")
    return md


#############################
# Constructor-related tests
#############################
def test_card_initialisation(card):
    assert card.name == "TestCard"
    assert card.long_name == "TestCard"   # default fallback
    assert card.allow_full_screen is True
    assert card.max_height == "450px"
    assert card.mutable is False
    assert card.initially_hidden is False
    assert card.description is None
    # Script and CSS additions
    assert "www/pythagoras.js" in card.script_list
    assert "www/markdown_tabsets.js" in card.script_list
    assert "www/animate.css" in card.css_list
    assert "www/pythagoras.css" in card.css_list


#############################
# Boolean helper methods
#############################
def test_hasSidebar_default(card):
    # settings() returns None -> no sidebar
    assert card.settings() is None
    assert card.hasSidebar() is False


def test_hasFooter_default(card):
    assert card.footer() is None
    assert card.hasFooter() is False


def test_hasFlipSide_default(card):
    assert card.back() is None
    assert card.hasFlipSide() is False


def test_override_methods_affect_flags():
    """If a subclass overrides front/back/settings/footer, flags should work."""
    class C(Card):
        def back(self): return "back-ui"
        def settings(self): return "settings-ui"
        def footer(self): return "footer-ui"
    c = C("X")
    assert c.hasFlipSide() is True
    assert c.hasSidebar() is True
    assert c.hasFooter() is True


#############################
# Markdown information tests
#############################
def test_information_returns_none_if_missing(card):
    """No markdown file means information() returns None."""
    # ensure file does not exist
    missing = Path("markdown/TestCard.md")
    if missing.exists():
        missing.unlink()
    assert card.information() is None


def test_information_reads_md(card, tmp_path, monkeypatch):
    # Simulate a project root with markdown/TestCard.md
    project_root = tmp_path
    markdown_dir = project_root / "markdown"
    markdown_dir.mkdir()
    md_file = markdown_dir / "TestCard.md"
    md_file.write_text("# Heading\nSome **markdown** text.", encoding="utf-8")
    # Run the code with CWD set to our fake project root
    monkeypatch.chdir(project_root)
    out = card.information()
    # Now we expect a ui.markdown object
    assert out is not None
    # Case 1: Newer Shiny → Tag object (has .render)
    if hasattr(out, "render"):
        html = out.render()
    else:    # Case 2: Older Shiny → plain HTML string
        html = out
    # Common assertions
    assert "<h1" in html.lower()
    assert "markdown" in html.lower()


#############################
# show() / hide() tests
#############################
def test_show_sends_message(card):
    session = AsyncMock()
    session.ns = lambda x: f"ns-{x}"
    asyncio.run(card.show(session))
    session.send_custom_message.assert_called_once_with(
        "toggle_visibility",
        {"id": "ns-Card", "visible": True}
    )


def test_hide_sends_message(card):
    session = AsyncMock()
    session.ns = lambda x: f"ns-{x}"
    asyncio.run(card.hide(session))
    session.send_custom_message.assert_called_once_with(
        "toggle_visibility",
        {"id": "ns-Card", "visible": False}
    )


#############################
# UI stub methods
#############################
def test_stub_ui_methods(card):
    assert card.front() is None
    assert card.back() is None
    assert card.settings() is None
    assert card.footer() is None


#############################
# Namespace existence
#############################
def test_namespace_exists(card):
    """Inherited from Module — but we can check basic existence."""
    assert hasattr(card, "namespace")
    assert isinstance(card.namespace, str)
