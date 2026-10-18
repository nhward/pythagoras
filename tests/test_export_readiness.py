import asyncio
from types import SimpleNamespace

import pytest
from shiny import reactive
from shiny.types import SilentException
from shiny.reactive._core import _reactive_environment

from export_readiness import ExportReadiness
from session_task import SessionExtendedTask


@pytest.fixture(autouse=True)
def fresh_lock(monkeypatch):
    monkeypatch.setattr(_reactive_environment, "_lock", None)


class Session:
    def __init__(self):
        self.callbacks = []
        self.ended = []

    def on_flushed(self, callback, once):
        self.callbacks.append(callback)

    def on_ended(self, callback):
        self.ended.append(callback)

    async def flush(self):
        await reactive.flush()
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            await callback()
        await reactive.flush()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("close_session", [False, True])
async def test_final_publication_triggers_downstream_without_another_browser_event(close_session):
    session = Session()
    effects = []

    def effect():
        def decorate(function):
            item = reactive.effect(function)
            effects.append(item)
            return item
        return decorate

    export = ExportReadiness().bind(SimpleNamespace(reactable=effect), lambda: "ready", {}, session)
    received = []

    @reactive.effect
    def downstream():
        received.append(export())

    try:
        await reactive.flush()
        callbacks, session.callbacks = session.callbacks, []
        for callback in callbacks:
            await callback()
        if close_session:
            for callback in session.ended:
                callback()
        # No input change or explicit reactive.flush: publication must wake it.
        for _ in range(12):
            await asyncio.sleep(0)
        assert received == ([] if close_session else ["ready"])
    finally:
        downstream.destroy()
        for item in effects:
            item.destroy()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_export_waits_for_browser_ack_and_flush_then_accepts_user_edits():
    contract = ExportReadiness()
    contract.require_inputs("Mode")
    contract.expect("Mode", "transform")
    mode = reactive.Value(None)
    inputs = {"Mode": mode}
    session = Session()
    effects = []

    def effect():
        def decorate(function):
            item = reactive.effect(function)
            effects.append(item)
            return item
        return decorate

    export = contract.bind(SimpleNamespace(reactable=effect), lambda: mode(), inputs, session)
    try:
        for intermediate in (None, "none"):
            mode.set(intermediate)
            await session.flush()
            with reactive.isolate(), pytest.raises(SilentException):
                export()
        mode.set("transform")
        await reactive.flush()
        with reactive.isolate(), pytest.raises(SilentException):
            export()  # No publication before the flush acknowledgement.
        await session.flush()
        with reactive.isolate():
            assert export() == "transform"
        mode.set("none")
        await session.flush()
        with reactive.isolate():
            assert export() == "none"
        contract.expect("Mode", "transform")
        await session.flush()
        with reactive.isolate(), pytest.raises(SilentException):
            export()
        mode.set("transform")
        await session.flush()
        with reactive.isolate():
            assert export() == "transform"
    finally:
        for item in effects:
            item.destroy()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_readiness_rejects_old_results_without_waiting_for_diagnostics():
    @SessionExtendedTask
    async def transform(value):
        return value * 2

    @SessionExtendedTask
    async def diagnostic():
        await asyncio.Event().wait()

    contract = ExportReadiness()
    current = reactive.Value(1)
    contract.checks.append(lambda: transform.require_current(current()))
    diagnostic.invoke()
    transform.invoke(1)
    for _ in range(15):
        await asyncio.sleep(0)
    try:
        with reactive.isolate():
            contract.check({})
            assert diagnostic.status() == "running"
        current.set(2)
        with reactive.isolate(), pytest.raises(SilentException):
            contract.check({})
        transform.invoke(2)
        for _ in range(15):
            await asyncio.sleep(0)
        with reactive.isolate():
            contract.check({})
            assert transform.result() == 4
    finally:
        diagnostic.close()
        transform.close()
        for _ in range(15):
            await asyncio.sleep(0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_obsolete_flush_callback_and_export_error_do_not_publish_stale_data():
    contract = ExportReadiness()
    session = Session()
    state = reactive.Value("old")
    effects = []

    def effect():
        def decorate(function):
            item = reactive.effect(function)
            effects.append(item)
            return item
        return decorate

    def source():
        value = state()
        if value == "error":
            raise ValueError("invalid transformation")
        return value

    export = contract.bind(SimpleNamespace(reactable=effect), source, {}, session)
    try:
        await reactive.flush()
        state.set("new")
        await session.flush()
        with reactive.isolate():
            assert export() == "new"
        state.set("error")
        await session.flush()  # The producer effect must not raise.
        with reactive.isolate(), pytest.raises(ValueError, match="invalid transformation"):
            export()
        state.set("recovered")
        await session.flush()
        with reactive.isolate():
            assert export() == "recovered"
    finally:
        for item in effects:
            item.destroy()
