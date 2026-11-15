import asyncio

import pytest
import reactive_timer as timer
from shiny import reactive
from shiny.reactive._core import _reactive_environment

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def isolated_reactive_lock(monkeypatch):
    monkeypatch.setattr(_reactive_environment, '_lock', None)


@pytest.fixture
def tasks(monkeypatch):
    created = []
    original = timer.create_task

    def create(coroutine):
        task = original(coroutine)
        created.append(task)
        return task

    monkeypatch.setattr(timer, 'create_task', create)
    return created


@pytest.mark.parametrize('phase', ['sleep', 'lock'])
@pytest.mark.parametrize('cause', ['context', 'session'])
async def test_expected_cancellation_is_quiet_and_unregisters_cleanup(phase, cause, tasks, monkeypatch, capsys):
    callbacks = []

    class Session:
        def on_ended(self, callback):
            callbacks.append(callback)
            return lambda: callbacks.remove(callback)

    monkeypatch.setattr(timer, 'get_current_session', lambda: Session())
    trigger = reactive.Value(0)

    @reactive.effect
    def effect():
        trigger()
        timer.invalidate_later(3600 if phase == 'sleep' else 0)

    try:
        await reactive.flush()
        async with reactive.lock():
            for _ in range(10):
                await asyncio.sleep(0)
            assert not tasks[0].done()
            if cause == 'context':
                trigger.set(1)
            else:
                callbacks[0]()
            await tasks[0]
        assert not tasks[0].cancelled()
        assert not callbacks
        assert 'CancelledError' not in capsys.readouterr().err
    finally:
        effect.destroy()


async def test_timer_still_invalidates_and_flushes_without_another_browser_event(tasks):
    visits = []

    @reactive.effect
    def effect():
        visits.append('ran')
        if len(visits) == 1:
            timer.invalidate_later(0)

    try:
        await reactive.flush()
        await tasks[0]
        assert visits == ['ran', 'ran']
    finally:
        effect.destroy()


async def test_cancel_before_timer_starts_still_releases_session_callback(tasks, monkeypatch, capsys):
    invalidated, ended = [], []

    class Context:
        on_invalidate = staticmethod(invalidated.append)

    class Session:
        def on_ended(self, callback):
            ended.append(callback)
            return lambda: ended.remove(callback)

    monkeypatch.setattr(timer, 'get_current_context', lambda: Context())
    monkeypatch.setattr(timer, 'get_current_session', lambda: Session())
    timer.invalidate_later(3600)
    invalidated[0]()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    await asyncio.sleep(0)
    assert not ended
    assert 'CancelledError' not in capsys.readouterr().err


async def test_non_cancellation_errors_are_not_swallowed(tasks, monkeypatch):
    async def broken_flush():
        raise RuntimeError('real timer failure')

    effect = reactive.effect(lambda: timer.invalidate_later(0))
    try:
        await reactive.flush()
        monkeypatch.setattr(reactive, 'flush', broken_flush)
        with pytest.raises(RuntimeError, match='real timer failure'):
            await tasks[0]
    finally:
        effect.destroy()
