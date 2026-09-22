"""Real asyncio completion callbacks during reactive/session teardown."""
import asyncio
from types import SimpleNamespace

import pytest
from shiny import reactive
import session_task as m
from shiny.reactive._core import _reactive_environment


@pytest.fixture(autouse=True)
def fresh_reactive_lock(monkeypatch):
    # Each pytest-asyncio test has its own event loop.
    monkeypatch.setattr(_reactive_environment, "_lock", None)


async def drain():
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error', 'cancel'])
@pytest.mark.parametrize('close_session', [False, True])
async def test_completion_after_values_destroyed(monkeypatch, outcome, close_session):
    ended, destroyed = [], []
    monkeypatch.setattr(m, 'get_current_session', lambda: SimpleNamespace(
        on_ended=ended.append, on_destroy=destroyed.append))
    loop = asyncio.get_running_loop()
    failures = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: failures.append(context))
    try:
        @m.SessionExtendedTask
        async def task():
            if outcome == 'error':
                raise RuntimeError('worker failed')
            return 42

        # Hold publication behind the lock, reproducing the deferred callback race.
        async with reactive.lock():
            task.invoke()
            task.invoke()  # Queued work must not restart after teardown.
            if outcome == 'cancel':
                task.cancel()
            await drain()
            if close_session:
                ended[0]()
                destroyed[0]()
            for value in (task.status, task.value, task.error):
                value.destroy()
        await drain()
        assert not failures
        assert task._closed
        assert not task._invocation_queue
        assert task._task is None
        task.invoke()  # Late invocation after closure is a no-op.
        assert task._task is None
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normal_results_errors_cancellation_and_queue():
    calls = []
    @m.SessionExtendedTask
    async def task(value):
        calls.append(value)
        if value < 0:
            raise ValueError('worker failed')
        return value * 2

    task.invoke(1)
    task.invoke(2)
    await drain()
    with reactive.isolate():
        assert task.result() == 4
    assert calls == [1, 2]
    task.invoke(-1)
    await drain()
    with reactive.isolate(), pytest.raises(ValueError, match='worker failed'):
        task.result()
    task.invoke(3)
    task.cancel()
    await drain()
    with reactive.isolate():
        assert task.status() == 'cancelled'
    task.invoke(4)
    await drain()
    with reactive.isolate():
        assert task.result() == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_cancels_running_worker_and_drops_queue():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []

    @m.SessionExtendedTask
    async def task(value):
        calls.append(value)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task.invoke(1)
    task.invoke(2)
    await started.wait()
    task.close()
    for value in (task.status, task.value, task.error):
        value.destroy()
    await drain()
    assert cancelled.is_set()
    assert calls == [1]
    assert task._task is None
    assert not task._invocation_queue
