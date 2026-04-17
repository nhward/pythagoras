import json
import textwrap
import pytest 
from shiny import ui, App, reactive

from pathlib import Path
import sys
# Ensure app root is importable when pytest is run outside the IDE
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module import Module  # noqa: E402


# -------------------------------------------------------------------
# Helper subclass for testing (implements abstract methods)
# -------------------------------------------------------------------
class DummyModule(Module):
    def call_ui(self):
        # minimal valid UI
        return ui.div(f"Module: {self.namespace}")
    def call_server(self, input, output, session):
        # no-op server
        return


# -------------------------------------------------------------------
# Global fixture: keep Module's class-level state clean per test
# and avoid patching real shiny.ui / shinywidgets in tests
# -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_module_class(monkeypatch):
    # Reset class-level state
    Module.Instances.clear()
    Module._ui_patched = False
    Module.script_list = []
    Module.css_list = []
    # Avoid touching real shiny.ui / shinywidgets in tests
    monkeypatch.setattr(Module, "_patch_module", lambda self, module: None)
    yield
    # Clean up after test
    Module.Instances.clear()
    Module._ui_patched = False
    Module.script_list = []
    Module.css_list = []


# -------------------------------------------------------------------
# Basic construction / namespace behaviour
# -------------------------------------------------------------------
@pytest.mark.unit
def test_module_initialises_and_registers_instance():
    m = DummyModule(name = "something")
    assert m.name == "something"
    assert m.namespace == "something"  # first instance uses name
    assert Module.Instances[m.namespace] is m
    # script/css lists populated
    assert Path(f"{m.ROOT}/www/console.js") in Module.script_list
    assert Path(f"{m.ROOT}/www/shepherd.css") in Module.css_list


@pytest.mark.unit
def test_namespace_gets_unique_suffix_when_reused():
    m1 = DummyModule(name = "card")
    m2 = DummyModule(name = "card")
    assert m1.namespace == "card"
    assert m2.namespace.startswith("card_")
    assert m1.namespace != m2.namespace
    assert len(Module.Instances) == 2


@pytest.mark.unit
def test_namespace_limit_raises_after_max_instances():
    # Use the same name many times
    for i in range(Module.MaxInstances):
        DummyModule(name = "dup")
    # Next one should raise
    with pytest.raises(ValueError):
        DummyModule(name = "dup")


@pytest.mark.unit
def test_reset_removes_instance_from_registry():
    m = DummyModule(name = "card")
    assert m.namespace in Module.Instances
    m.reset()
    assert m.namespace not in Module.Instances


@pytest.fixture
def test_close_swallows_reset_errors(monkeypatch, capsys):
    m = DummyModule(name = "card")
    def boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(m, "reset", boom)
    # Should not raise
    m.close()
    # Optional: we know warning() is used, but we don't assert on its text here
    out, err = capsys.readouterr()
    # Just check test runs to completion


# -------------------------------------------------------------------
# Namespace helper
# -------------------------------------------------------------------
@pytest.mark.unit
def test_ns_prefixes_ids_with_namespace():
    m = DummyModule("mycard")
    nid = m.ns("slider1")
    assert "mycard" in nid
    assert nid.endswith("slider1")


# -------------------------------------------------------------------
# Test tour steps payload
# -------------------------------------------------------------------
@pytest.mark.unit
def test_tour_steps_payload_sorts_by_priority_desc():
    m = DummyModule("card")
    m._shepherd_steps = {
        "card-x_wrapper": {
            "title": "X", "text": "x", "position": "bottom", "priority": 1
        },
        "card-y_wrapper": {
            "title": "Y", "text": "y", "position": "top", "priority": 10
        },
    }
    payload = m._tour_steps_payload()
    steps = json.loads(payload)
    # Highest priority first (y then x)
    assert [s["id"] for s in steps] == ["card-y_wrapper", "card-x_wrapper"]
    # and you can add more structure checks:
    assert steps[0]["selector"] == "#card-y_wrapper"
    assert steps[1]["selector"] == "#card-x_wrapper"


#  Problem with async 
 # -------------------------------------------------------------------
# create_run_tour: Shepherd steps + custom message
# -------------------------------------------------------------------
# @pytest.mark.unit
# def test_create_run_tour_sends_sorted_steps():
#     m = GetDummyModule("card")
#     # Fake some steps with different priorities
#     m._shepherd_steps = {
#         "card-x_wrapper": {
#             "title": "X", "text": "x", "position": "bottom", "priority": 1
#         },
#         "card-y_wrapper": {
#             "title": "Y", "text": "y", "position": "top", "priority": 10
#         },
#     }
#     class FakeSession:
#         def __init__(self):
#             self.messages = []
#         async def send_custom_message(self, name, payload):
#             self.messages.append((name, payload))
#     session = FakeSession()
#     run_async_for_test(m.create_run_tour(session))
#     assert len(session.messages) == 1
#     name, payload = session.messages[0]
#     assert name == "create_run_tour"
#     steps = json.loads(payload)
#     assert [s["id"] for s in steps] == ["card-y_wrapper", "card-x_wrapper"]


# -------------------------------------------------------------------
# _make_wrapper: basic wrapping behaviour (using a fake ui func)
# -------------------------------------------------------------------
@pytest.mark.unit
def test_make_wrapper_registers_shepherd_step_and_wraps():
    m = DummyModule("card")

    def fake_input(id, label=None, **kwargs):
        # imitate a Shiny input_* function returning a Tag
        return ui.input_text(id=id, label=label or "Label")

    wrapped = m._make_wrapper(fake_input)
    # Use guide=m so we actually register
    widget = wrapped(
        "myid",
        guide=m,
        label="My label",
        title="Step title",
        text="Help text",
        position="top",
        priority=5,
        foo="bar",
    )
    # Should have registered a step for namespaced wrapper id
    nid = m.ns("myid")
    wid = f"{nid}_wrapper"
    assert wid in m._shepherd_steps
    step = m._shepherd_steps[wid]
    assert step["title"] == "Step title"
    assert step["text"] == "Help text"
    assert step["position"] == "top"
    assert step["priority"] == 5
    # Returned widget should be wrapped in a div with that id
    assert isinstance(widget, ui.Tag)
    assert widget.attrs.get("id") == wid


# -------------------------------------------------------------------
# capture_print decorator
# -------------------------------------------------------------------
@pytest.mark.unit
def test_capture_print_captures_stdout_and_return_value():
    m = DummyModule("card")

    @m.capture_print
    def f(x, y):
        print("hello")
        return x + y

    result = f(2, 3)
    # The decorator returns printed text + stringified result
    assert "hello" in result
    assert "5" in result


# -------------------------------------------------------------------
# suspendable decorator (calc) behaviour
# -------------------------------------------------------------------
@pytest.mark.unit
def test_suspendable_calc_respects_suspend_and_resume():
    m = DummyModule("card")
    calls = {"count": 0}

    @m.suspendable(suspended=True, default=-1)
    def f() -> int:
        calls["count"] += 1
        return 99

    # Starts suspended
    with reactive.isolate():
        assert f() == -1
        assert calls["count"] == 0
    # Resume → now the underlying function should run
    f.resume()
    with reactive.isolate():
        val = f()
        assert val == 99
        assert calls["count"] == 1
    # Suspend again
    f.suspend()
    with reactive.isolate():
        assert f() == -1
        assert calls["count"] == 1
    # Global suspend/resume
    m.suspend()
    with reactive.isolate():
        assert f() == -1
    m.resume()
    with reactive.isolate():
        assert f() == 99
        assert calls["count"] == 2
    # Registered as suspendable
    assert f in m.suspendables


# -------------------------------------------------------------------
# record_code, retrieve_code, code_text
# -------------------------------------------------------------------
@pytest.mark.unit
def test_record_code_and_retrieve_code_store_source():
    m = DummyModule("card")

    @m.record_code
    def foo():
        return 10

    # Call once so code gets recorded
    assert foo() == 10
    src = m.retrieve_code("foo")
    assert "def foo" in src
    # Unknown name raises
    with pytest.raises(ValueError):
        m.retrieve_code("bar")


@pytest.mark.unit
def test_code_text_returns_html_snippet():
    m = DummyModule("card")

    @m.record_code
    def foo():
        # @output and @render.print tags will be sanitised in code_text()
        return 1

    foo()
    html_obj = m.code_text()
    # Check it’s a Shiny HTML tag
    assert isinstance(html_obj, ui.HTML)
    rendered = str(html_obj)
    assert "foo" in rendered
    # Check that decorators have been cleaned up in some way
    assert "<h3># foo</h3>" in rendered


# -------------------------------------------------------------------
# debounce / throttle: basic smoke tests (no tight timing assertions)
# -------------------------------------------------------------------
@pytest.mark.unit
def test_debounce_wraps_callable_without_raising():
    m = DummyModule("test-module")
    calls = {"count": 0}

    @m.debounce(delay_secs=0.01)
    def compute():
        calls["count"] += 1
        return 42

    # It should be callable and behave like a function
    assert callable(compute)
    # Use a lightweight reactive context so the reactive infra is “properly” set up
    with reactive.isolate():
        result = compute()
    assert result == 42
    assert calls["count"] == 1


@pytest.mark.unit
def test_throttle_wraps_callable_without_raising():
    m = DummyModule("card")
    seen = {"count": 0}

    @m.throttle(delay_secs=0.01)
    def f():
        seen["count"] += 1
        return 321

    with reactive.isolate():
        out = f()
    assert callable(f)
    assert out in (321, None)


# -------------------------------------------------------------------
# create_cards: dynamic import from folder
# -------------------------------------------------------------------
@pytest.mark.unit
def test_create_cards_imports_and_instantiates(tmp_path, monkeypatch):
    # Create a temp 'cards' folder with a single Python file
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    # Minimal card file defining instance() that returns a DummyModule
    card_code = textwrap.dedent(
        """
        from module import Module
        from shiny import ui

        class MyCard(Module):
            def call_ui(self):
                return ui.div("card ui")

            def call_server(self, input, output, session):
                pass

        def instance():
            return MyCard("mycard")
        """
    )
    (cards_dir / "mycard.py").write_text(card_code, encoding="utf-8")
    # Use the real create_cards (it expects folder name, not Path)
    # Ensure cwd is tmp_path so "cards" resolves
    monkeypatch.chdir(tmp_path)
    imported = Module.create_cards(folder="cards")
    # Should have exactly one imported card
    assert len(imported) == 1
    (ns, card) = next(iter(imported.items()))
    assert isinstance(card, Module)
    assert card.name == "mycard"
    # Module.Instances should also contain it
    assert card.namespace in Module.Instances


# -------------------------------------------------------------------
# app(): building the Shiny app skeleton
# -------------------------------------------------------------------
@pytest.mark.unit
def test_app_builds_shiny_app():
    m1 = DummyModule("card1")
    m2 = DummyModule("card2")
    modules = {
        m1.namespace: m1,
        m2.namespace: m2,
    }
    app = Module.app(modules)
    assert isinstance(app, App)
    # ui should contain both cards
    html = str(app.ui)
    assert "Module: card1" in html
    assert "Module: card2" in html