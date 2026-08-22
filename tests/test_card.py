import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from card import Card


#############################
# Fixtures
#############################
@pytest.fixture
def card():
    """Basic Card with minimal constructor."""
    return Card(file="/Users/nickward/Documents/pythagoras/markdown/card.html", long_name = "TestCard")


@pytest.fixture
def temp_md_file(tmp_path):
    """Create a temporary markdown file for 'information()' tests."""
    md = tmp_path / "TestCard.md"
    md.write_text("# Heading\nSome **markdown** text.", encoding="utf-8")
    return md


#############################
# Constructor-related tests
#############################
@pytest.mark.unit
def test_card_initialisation(card):
    assert card.name == "card"
    assert card.long_name == "TestCard"   # default fallback
    assert card.allow_full_screen is True
    assert card.max_height == "450px"
    assert card.mutable is False
    assert card.initially_hidden is False
    assert card.description is None
    # Script and CSS additions
    assert Card.ROOT / "www" / "pythagoras.js" in card.script_list
    assert Card.ROOT / "www" / "animate.css" in card.css_list
    assert Card.ROOT / "www" / "pythagoras.css" in card.css_list

@pytest.mark.unit
def test_hasSidebar_default(card):
    # settings() returns None -> no sidebar
    assert card.settings is None
    assert card.hasSidebar() is False


@pytest.mark.unit
def test_hasFooter_default(card):
    assert card.footer is None
    assert card.hasFooter() is False


@pytest.mark.unit
def test_hasFlipSide_default(card):
    assert card.back is None
    assert card.hasFlipSide() is False


@pytest.mark.unit
def test_assigned_ui_factories_affect_flags():
    c = Card(file="/Users/nickward/Documents/pythagoras/markdown/card.html", long_name = "TestCard")
    c.back = lambda: "back-ui"
    c.settings = lambda: "settings-ui"
    c.footer = lambda: "footer-ui"
    assert c.hasFlipSide() is True
    assert c.hasSidebar() is True
    assert c.hasFooter() is True
    assert c.back == "back-ui"
    assert c.settings == "settings-ui"
    assert c.footer == "footer-ui"


@pytest.mark.unit
def test_clearing_ui_factories_resets_flags():
    c = Card(file="/Users/nickward/Documents/pythagoras/markdown/card.html", long_name = "TestCard")
    c.back = lambda: "back-ui"
    c.settings = lambda: "settings-ui"
    c.footer = lambda: "footer-ui"
    c.back = None
    c.settings = None
    c.footer = None
    assert c.hasFlipSide() is False
    assert c.hasSidebar() is False
    assert c.hasFooter() is False


@pytest.mark.unit
def test_information_returns_message_if_missing(card):
    """No html file means information() returns message."""
    # ensure file does not exist
    missing = Path("markdown/card.html")
    if missing.exists():
        missing.unlink()
    assert card.information() == '<br>File /Users/nickward/Documents/pythagoras/markdown/card.html not found'


#############################
# UI stub methods
#############################
@pytest.mark.unit
def test_stub_ui_methods(card):
    assert card.front is None
    assert card.back is None
    assert card.settings is None
    assert card.footer is None


#############################
# Namespace existence
#############################
@pytest.mark.unit
def test_namespace_exists(card):
    """Inherited from Module — but we can check basic existence."""
    assert hasattr(card, "namespace")
    assert isinstance(card.namespace, str)
