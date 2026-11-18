"""Real asyncio completion callbacks during reactive/session teardown."""
import asyncio

import pytest
import session_task as m
from shiny import reactive
from shiny.types import SilentException, SilentOperationInProgressException


class IdleSession:
    def __init__(self):
        self.ended, self.destroyed, self.pending = [], [], []

    def on_ended(self, callback):
        self.ended.append(callback)

    def on_destroy(self, callback):
        self.destroyed.append(callback)

    def run_once_when_idle(self, callback):
        self.pending.append(callback)

    async def flush(self):
        while self.pending:
            self.pending.pop(0)()
            await reactive.flush()
            await drain()


async def drain():
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pending_result_clears_output_and_remains_reactive():
    release = asyncio.Event()

    @m.SessionExtendedTask
    async def task():
        await release.wait()
        return 42

    @reactive.calc
    def result():
        return task.result()

    task.invoke()
    with reactive.isolate(), pytest.raises(SilentException) as caught:
        result()
    assert not isinstance(caught.value, SilentOperationInProgressException)
    release.set()
    await drain()
    with reactive.isolate():
        assert result() == 42


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_running_task_drops_queue_and_ignores_late_result():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    @m.SessionExtendedTask
    async def task(value):
        calls.append(value)
        if value == "old":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
        return value

    task.invoke("old")
    await started.wait()
    task.invoke("queued")
    task.cancel()
    task.invoke("new")
    await drain()
    release.set()
    await drain()
    with reactive.isolate():
        assert task.result() == "new"
    assert calls == ["old", "new"]
    assert not task._invocation_queue


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_discards_deferred_result_and_allows_immediate_restart(monkeypatch):
    session = IdleSession()
    monkeypatch.setattr(m, "get_current_session", lambda: session)
    @m.SessionExtendedTask
    async def task(value):
        return value

    task.invoke("old")
    await drain()  # Worker finished, publication waits for the session idle cycle.
    task.cancel()
    with reactive.isolate():
        assert task.status() == "cancelled"
    task.invoke("new")
    await drain()
    await session.flush()
    with reactive.isolate():
        assert task.result() == "new"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error', 'cancel'])
@pytest.mark.parametrize('close_session', [False, True])
async def test_completion_after_values_destroyed(monkeypatch, outcome, close_session):
    session = IdleSession()
    monkeypatch.setattr(m, 'get_current_session', lambda: session)
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

        # Keep publication queued until after the session/values are destroyed.
        task.invoke()
        task.invoke()
        if outcome == 'cancel':
            task.cancel()
        await drain()
        if close_session:
            session.ended[0]()
            session.destroyed[0]()
        for value in (task.status, task.value, task.error):
            value.destroy()
        await session.flush()
        await drain()
        assert not failures
        assert not task._invocation_queue
        assert task._task is None
        task.invoke()  # Also detects destruction when cancellation skipped publication.
        assert task._closed
        assert task._task is None
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normal_results_errors_cancellation_and_queue(caplog):
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
    failures = [record for record in caplog.records if 'Background task failed' in record.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info[0] is ValueError
    assert failures[0].exc_info[2] is not None
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_queued_invocation_waits_for_async_consumers(monkeypatch):
    session = IdleSession()
    monkeypatch.setattr(m, 'get_current_session', lambda: session)
    reading, release = asyncio.Event(), asyncio.Event()
    calls, received = [], []

    @m.SessionExtendedTask
    async def task(value):
        calls.append(value)
        return value

    @reactive.effect
    async def consume():
        value = task.result()
        if value == 1:
            reading.set()
            await release.wait()
        received.append(task.result())

    try:
        task.invoke(1)
        task.invoke(2)
        await drain()
        publishing = asyncio.create_task(session.flush())
        await asyncio.wait_for(reading.wait(), timeout=2)
        assert calls == [1]
        release.set()
        await asyncio.wait_for(publishing, timeout=2)
        assert calls == [1, 2]
        assert received == [1, 2]
    finally:
        release.set()
        consume.destroy()
        task.close()
    assert not task._invocation_queue
