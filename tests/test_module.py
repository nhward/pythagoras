import json
import sys
from pathlib import Path
from types import SimpleNamespace

path = str(Path(__file__).resolve().parent.parent / "app")
if path not in sys.path:
    sys.path.insert(0, path)

import module as module_lib
import pytest
from module import BusyTracker, Module
from shiny import reactive, ui
from shiny.types import SilentException


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
    # Module.script_list = []
    # Module.css_list = []
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
    assert m.Instances[m.namespace] is m
    # script/css lists populated
    assert (m.ROOT / "www" / "console.js") in m.script_list
    assert Path(m.ROOT / "www" / "shepherd-15.3.0.css") in  m.css_list


@pytest.mark.unit
@pytest.mark.parametrize(
    ("hostname", "expected"),
    [
        ("localhost", "local"),
        ("127.0.0.1", "local"),
        ("::1", "local"),
        ("pythagoras.example", "server"),
    ],
)
def test_runtime_mode_uses_client_hostname(
    monkeypatch, hostname, expected
):
    monkeypatch.setattr(Module, "IS_SHINYLIVE", False)
    session = SimpleNamespace(
        clientdata=SimpleNamespace(url_hostname=lambda: hostname)
    )

    assert Module.runtime_mode(session) == expected


@pytest.mark.unit
def test_runtime_mode_prefers_shinylive(monkeypatch):
    monkeypatch.setattr(Module, "IS_SHINYLIVE", True)

    assert Module.runtime_mode(SimpleNamespace()) == "shinylive"


@pytest.mark.unit
def test_runtime_mode_supports_legacy_client_data(monkeypatch):
    monkeypatch.setattr(Module, "IS_SHINYLIVE", False)
    session = SimpleNamespace(
        client_data=SimpleNamespace(url_hostname=lambda: "localhost")
    )

    assert Module.runtime_mode(session) == "local"


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
    keys = []
    for key, instance in Module.Instances.items(): 
        if instance is None:
            continue
        keys.append(key)
    assert m.namespace not in keys


@pytest.mark.unit
def test_reset_honours_instance_reuse_setting():
    module = DummyModule(name="card")
    module._reuse_cards = False

    module.reset()

    assert Module.Instances[module.namespace] is None


@pytest.fixture
def test_close_swallows_reset_errors(monkeypatch, capsys):
    m = DummyModule(name = "card")
    def boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(m, "reset", boom)
    # Should not raise
    m.close()
    # Optional: we know warning() is used, but we don't assert on its text here
    _out, _err = capsys.readouterr()
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


@pytest.mark.unit
def test_generic_bookmark_state_restores_and_captures_inputs(caplog):
    module = DummyModule("card")
    module.restore_configuration_state({
        "inputs": {
            "Columns": ["saved", "removed"],
            "RemovedInput": "retained for compatibility",
        },
        "future_card_field": {"mode": "newer"},
    })

    def input_select(id, label, choices, *, selected=None):
        return id, label, choices, selected

    restored = module._configuration_kwargs(
        input_select,
        "Columns",
        {"choices": ["current"], "selected": None},
    )
    assert restored["selected"] == ["saved", "removed"]
    assert restored["choices"] == ["saved", "removed", "current"]

    module._configuration_input = SimpleNamespace(
        Columns=lambda: ("current", "saved"),
    )
    state = module.configuration_state()

    assert state["inputs"]["Columns"] == ["current", "saved"]
    assert state["inputs"]["RemovedInput"] == "retained for compatibility"
    assert state["future_card_field"] == {"mode": "newer"}
    assert "is no longer present" in caplog.text


@pytest.mark.unit
def test_generic_bookmark_uses_defaults_for_new_inputs_and_skips_buttons(caplog):
    module = DummyModule("card")
    module.restore_configuration_state({"inputs": {}})

    def input_text(id, label, value=""):
        return id, label, value

    def input_action_button(id, label):
        return id, label

    text_options = module._configuration_kwargs(
        input_text,
        "NewInput",
        {"value": "default"},
    )
    module._configuration_kwargs(input_action_button, "Run", {})

    assert text_options["value"] == "default"
    assert module._configuration_input_ids == {"NewInput"}
    assert "has no value for new input" in caplog.text


@pytest.mark.unit
def test_generic_bookmark_restores_card_navset_selection():
    module = DummyModule("card")
    module.restore_configuration_state({"inputs": {"Navset": "Evidence"}})

    def navset_tab(*args, id=None, selected=None):
        return args, id, selected

    wrapped = module._make_navset_wrapper(navset_tab)
    with module.configuration_ui_context():
        _args, input_id, selected = wrapped(
            "front",
            "evidence",
            id="Navset",
            selected="front",
        )

    assert input_id == "Navset"
    assert selected == "Evidence"
    assert module._configuration_input_ids == {"Navset"}


@pytest.mark.unit
def test_generic_bookmark_supports_manually_registered_custom_inputs():
    module = DummyModule("card")
    saved_role_map = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
    }
    module.register_configuration_input("role_map")
    module.restore_configuration_state({"inputs": {"role_map": saved_role_map}})

    assert module.restored_configuration_input("role_map") == saved_role_map

    current_role_map = {
        "target": ["y"],
        "predictor": ["x2"],
        "none": ["x1"],
    }
    module._configuration_input = SimpleNamespace(
        role_map=lambda: current_role_map,
    )

    assert module.configuration_state()["inputs"]["role_map"] == current_role_map


@pytest.mark.unit
def test_generic_bookmark_supports_card_owned_state_fields(caplog):
    module = DummyModule("card")
    module.register_configuration_state_field("table_model")
    module.restore_configuration_state({
        "inputs": {},
        "table_model": {"rows": [{"source": "x", "type": "integer"}]},
    })

    restored = module.restored_configuration_field("table_model")
    assert restored == {"rows": [{"source": "x", "type": "integer"}]}
    assert "unrecognised field" not in caplog.text

    module.register_configuration_state_provider(
        "table_model",
        lambda: {"rows": [{"source": "x", "type": "decimal"}]},
    )
    assert module.configuration_state()["table_model"] == {
        "rows": [{"source": "x", "type": "decimal"}]
    }


@pytest.mark.unit
def test_generic_bookmark_excludes_transient_projection_inputs():
    module = DummyModule("card")
    module.exclude_configuration_input("CurrentRowType")
    module.restore_configuration_state({
        "inputs": {"CurrentRowType": "integer", "Persistent": "saved"},
    })

    def input_select(id, label, choices, *, selected=None):
        return id, label, choices, selected

    options = module._configuration_kwargs(
        input_select,
        "CurrentRowType",
        {"choices": ["decimal"], "selected": "decimal"},
    )
    module._configuration_input = SimpleNamespace(
        CurrentRowType=lambda: "text",
        Persistent=lambda: "current",
    )
    module._configuration_input_ids.add("Persistent")

    assert options["selected"] == "decimal"
    assert module.configuration_state()["inputs"] == {"Persistent": "current"}


@pytest.mark.unit
def test_generic_bookmark_defers_dynamic_input_restoration():
    module = DummyModule("card")
    module.defer_configuration_input("DynamicNavset")
    module.restore_configuration_state({
        "inputs": {"DynamicNavset": "created-later"},
    })

    def navset_tab(*args, id=None, selected=None):
        return args, id, selected

    wrapped = module._make_navset_wrapper(navset_tab)
    with module.configuration_ui_context():
        _args, input_id, selected = wrapped(
            "initial-panel",
            id="DynamicNavset",
            selected="initial-panel",
        )

    assert input_id == "DynamicNavset"
    assert selected == "initial-panel"
    assert (
        module.restored_configuration_input("DynamicNavset")
        == "created-later"
    )

    module._configuration_input = SimpleNamespace(
        DynamicNavset=lambda: "current-panel",
    )
    assert module.configuration_state()["inputs"]["DynamicNavset"] == (
        "current-panel"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("choices", "saved", "expected"),
    [
        ([-9999, -999, -1], ["-9999", "-1"], [-9999, -1]),
        (
            [-9999.99, -99.0, -1.0],
            ["-9999.99", "-99.0", "-1.0"],
            [-9999.99, -99.0, -1.0],
        ),
    ],
)
def test_generic_bookmark_coerces_numeric_choice_strings(
    choices, saved, expected
):
    module = DummyModule("card")
    module.restore_configuration_state({"inputs": {"Sentinels": saved}})

    def input_selectize(id, label, choices, *, selected=None):
        return id, label, choices, selected

    restored = module._configuration_kwargs(
        input_selectize,
        "Sentinels",
        {"choices": choices, "selected": choices},
    )
    module._configuration_input = SimpleNamespace(Sentinels=lambda: saved)

    assert restored["selected"] == expected
    assert module.configuration_state()["inputs"]["Sentinels"] == expected


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
# reactable decorator (calc) behaviour
# -------------------------------------------------------------------
@pytest.mark.unit
def test_reactable_calc_respects_suspend_and_resume():
    m = DummyModule("card")
    calls = {"count": 0}

    @m.reactable(suspended=True, default=-1, calc = True)
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
    # Registered as reactable
    assert f in m.reactables


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_reactable_calc_without_default_stops_silently():
    m = DummyModule("card")

    @m.reactable(suspended=True, calc=True)
    async def f() -> int:
        return 99

    with pytest.raises(SilentException):  # noqa: SIM117
        with reactive.isolate():
            await f()

    f.resume()
    with reactive.isolate():
        assert await f() == 99


@pytest.mark.unit
def test_reactable_calc_without_default_is_silent_while_suspended():
    m = DummyModule("card")
    calls = {"count": 0}

    @m.reactable(suspended=True, calc=True)
    def value():
        calls["count"] += 1
        return "available"

    with reactive.isolate(), pytest.raises(SilentException):
        value()
    assert calls["count"] == 0

    value.resume()
    with reactive.isolate():
        assert value() == "available"
    assert calls["count"] == 1

    value.suspend()
    with reactive.isolate(), pytest.raises(SilentException):
        value()
    assert calls["count"] == 1


@pytest.mark.unit
def test_reactable_calc_preserves_explicit_none_default():
    m = DummyModule("card")

    @m.reactable(suspended=True, default=None, calc=True)
    def value():
        return "available"

    with reactive.isolate():
        assert value() is None


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

@pytest.mark.unit
def test_settle_test_bypass_returns_immediately(monkeypatch):
    module = DummyModule("test-module")
    calls = {"count": 0}

    monkeypatch.setattr(
        module,
        "running_under_tests",
        lambda: True,
    )

    @module.settle(seconds=2, bypass_during_tests=True)
    def compute():
        calls["count"] += 1
        return 42

    with reactive.isolate():
        result = compute()

    assert result == 42
    assert calls["count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extended_task_logs_start_and_elapsed_completion(monkeypatch):
    module = DummyModule("timed-card")
    records = []
    module.log = SimpleNamespace(
        debug=lambda message, *args: records.append(
            ("DEBUG", message % args)
        ),
        info=lambda message, *args: records.append(
            ("INFO", message % args)
        ),
    )
    clock = iter((10.0, 12.3456))
    monkeypatch.setattr(module_lib.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(reactive, "extended_task", lambda function: function)

    @module.extended_task
    async def Calculate(value):
        return value * 2

    assert await Calculate(21) == 42
    assert records == [
        ("DEBUG", "Extended task Calculate started"),
        ("INFO", "Extended task Calculate completed in 2.346 seconds"),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extended_task_logs_completion_without_hiding_failure(monkeypatch):
    module = DummyModule("failing-card")
    records = []
    module.log = SimpleNamespace(
        debug=lambda message, *args: records.append(
            ("DEBUG", message % args)
        ),
        info=lambda message, *args: records.append(
            ("INFO", message % args)
        ),
    )
    clock = iter((20.0, 20.5))
    monkeypatch.setattr(module_lib.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(reactive, "extended_task", lambda function: function)

    @module.extended_task
    async def Calculate():
        raise RuntimeError("calculation failed")

    with pytest.raises(RuntimeError, match="calculation failed"):
        await Calculate()
    assert records[-1] == (
        "INFO",
        "Extended task Calculate completed in 0.500 seconds",
    )


@pytest.mark.unit
def test_extended_task_can_be_tracked_by_busy_decorator():
    module = DummyModule("busy-card")
    busy = BusyTracker()

    @busy.track("Calculating…")
    @module.extended_task
    async def Calculate():
        return 1

    assert callable(Calculate.status)
    assert busy._tasks == [(Calculate, "Calculating…")]


class FakeExtendedTask:
    def __init__(self, status="initial"):
        self.current_status = status

    def status(self):
        return self.current_status


@pytest.mark.unit
def test_busy_tracker_decorator_preserves_task_and_tracks_running_status():
    busy = BusyTracker()
    task = FakeExtendedTask()

    decorated = busy.track("Calculating…")(task)

    assert decorated is task
    assert busy.ui() is None

    task.current_status = "running"
    rendered = str(busy.ui())
    assert "Calculating…" in rendered
    assert "spinner-border" in rendered
    assert 'aria-live="polite"' in rendered


@pytest.mark.unit
def test_busy_tracker_rejects_wrong_decorator_order():
    busy = BusyTracker()

    with pytest.raises(TypeError, match="reactive.extended_task"):
        busy.track("Calculating…")(lambda: None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_settle_waits_since_last_change_and_retains_published_value():
    import asyncio
    module = DummyModule('settle-timing')
    source = reactive.Value(('initial',))
    observed = []

    @module.settle(seconds=.12, bypass_during_tests=False)
    def settled():
        return source.get()

    @reactive.effect
    def first_reader():
        observed.append(('first', settled()))

    @reactive.effect
    def second_reader():
        observed.append(('second', settled()))

    try:
        await reactive.flush()
        assert observed == []  # A second reader cannot publish the candidate early.
        await asyncio.sleep(.16)
        await reactive.flush()
        assert observed == [('first', ('initial',)), ('second', ('initial',))]
        source.set(('a',))
        await reactive.flush()
        await asyncio.sleep(.07)
        source.set(('a', 'b'))
        await reactive.flush()
        await asyncio.sleep(.07)  # Past first change's deadline, before the second's.
        await reactive.flush()
        with reactive.isolate():
            assert settled() == ('initial',)
        assert len(observed) == 2
        await asyncio.sleep(.09)
        await reactive.flush()
        assert observed[-2:] == [('first', ('a', 'b')), ('second', ('a', 'b'))]
        # Returning to the committed value creates no redundant publication.
        source.set(('temporary',))
        await reactive.flush()
        source.set(('a', 'b'))
        await reactive.flush()
        await asyncio.sleep(.16)
        await reactive.flush()
        assert len(observed) == 4
    finally:
        first_reader.destroy()
        second_reader.destroy()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_settle_passes_proxy_with_immutable_history_without_deepcopy():
    import asyncio

    import pandas as pd
    from proxy_data import proxy_data

    module = DummyModule('settle-proxy')
    original = proxy_data(pd.DataFrame({'x': [1, 2]}))
    first = original.with_inactive_step(stage='Learning', card='example',
        operation='Example', parameters={'selected': ['a']})
    second = original.with_inactive_step(stage='Learning', card='example',
        operation='Example', parameters={'selected': ['b']})
    source = reactive.Value(first)
    observed = []

    @module.settle(seconds=.05, bypass_during_tests=False)
    def settled():
        # PreparedData-style producers can return a fresh equivalent object.
        return source.get().clone()

    @reactive.effect
    def reader():
        observed.append(settled())

    try:
        await reactive.flush()
        await asyncio.sleep(.09)
        await reactive.flush()
        assert len(observed) == 1 and observed[0].equals(first)
        source.set(second)
        await reactive.flush()
        with reactive.isolate():
            assert settled().equals(first)
        await asyncio.sleep(.09)
        await reactive.flush()
        assert len(observed) == 2 and observed[-1].equals(second)
        assert observed[-1].processing_records == second.processing_records
    finally:
        reader.destroy()


@pytest.mark.unit
def test_settle_snapshots_plain_containers_but_preserves_opaque_values():
    from types import MappingProxyType
    class NotCopyable:
        def __deepcopy__(self, memo):
            raise TypeError('Must not copy an application value')
    opaque = NotCopyable()
    history = MappingProxyType({'value': 1})
    options = {'selected': ['a'], 'nested': ({'enabled': {'x'}},),
               'object': opaque, 'history': history}
    snapshot = module_lib._settle_snapshot(options)
    options['selected'].append('b')
    options['nested'][0]['enabled'].add('y')
    assert snapshot['selected'] == ['a']
    assert snapshot['nested'][0]['enabled'] == {'x'}
    assert snapshot['object'] is opaque
    assert snapshot['history'] is history


@pytest.mark.unit
def test_record_code_outside_reactable_preserves_calc_and_source():
    module = DummyModule('record-reactive')
    calls = []

    @module.reactable(calc=True, suspended=False)
    def calculated():
        calls.append('called')
        return 42

    recorded = module.record_code(calculated)
    assert recorded is calculated
    assert 'def calculated' in module.retrieve_code('calculated')
    with reactive.isolate():
        assert recorded() == recorded() == 42
    assert calls == ['called']
    recorded.suspend()
    with reactive.isolate(), pytest.raises(SilentException):
        recorded()
    recorded.resume()
    with reactive.isolate():
        assert recorded() == 42


@pytest.mark.unit
def test_record_code_plain_shiny_calc_source_unavailable_is_benign():
    module = DummyModule('record-plain-reactive')

    @reactive.calc
    def calculated():
        return 7

    assert module.record_code(calculated) is calculated
    assert module.retrieve_code('calculated') == '<source not available>'
    with reactive.isolate():
        assert calculated() == 7
