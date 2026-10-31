import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "app"
if path not in sys.path:
    sys.path.insert(0, path)

import pytest
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
    assert card.pipeline_compatible is True
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
    missing = Path("app/www/markdown/card.html")
    if missing.exists():
        missing.unlink()
    assert card.information() == '<br>File /Users/nickward/Documents/pythagoras/app/www/markdown/card.html not found'


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


class FakeModuleSession:
    def __init__(self):
        self.destroy_count = 0

    async def destroy(self):
        self.destroy_count += 1


@pytest.mark.unit
@pytest.mark.parametrize('name, compatible', [
    ('data_import', False), ('var_modify', False), ('var_roles', False),
    ('obs_duplicates', False), ('var_nominal', True), ('miss_map', True),
])
def test_card_factories_declare_pipeline_compatibility(name, compatible, monkeypatch):
    import importlib
    from module import Module
    monkeypatch.setattr(Module, 'Instances', {})
    instance = importlib.import_module(f'cards.{name}').instance()
    assert instance.pipeline_compatible is compatible


@pytest.mark.unit
@pytest.mark.parametrize('mutable', [True, False])
def test_pipeline_compatibility_is_independent_of_mutability(mutable):
    import pandas as pd
    from proxy_data import proxy_data
    from shiny import reactive
    from sklearn.preprocessing import FunctionTransformer

    c = Card(file=f'/tmp/compatibility_{mutable}.py', mutable=mutable, pipeline_compatible=False)
    raw = proxy_data(pd.DataFrame({'x': [1, 2]}))
    step = FunctionTransformer()
    learned = raw.with_pipeline_step(step, name='upstream', preview_frame=step.fit_transform(raw.frame))
    c._imports.set(learned)
    calls = []
    guarded = c.guard_position(lambda: calls.append('export') or raw)
    with reactive.isolate():
        assert guarded() is learned
        assert 'not pipeline-compatible' in c.position_error()
    assert calls == []  # Reject before an export can discard the pipeline.
    c._imports.set(raw)
    with reactive.isolate():
        assert guarded() is raw
        assert c.position_error() is None
    assert calls == ['export']


@pytest.mark.unit
def test_incompatible_import_can_produce_data_without_an_incoming_source():
    from shiny import reactive
    c = Card(file='/tmp/import_without_source.py', mutable=True, pipeline_compatible=False)
    result = object()
    guarded = c.guard_position(lambda: result)
    with reactive.isolate():
        assert guarded() is result
        assert c.position_error() is None
