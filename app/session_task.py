"""Extended tasks that stop publishing results when their session closes."""
import asyncio
import logging
import os
from contextvars import ContextVar
from export_readiness import equivalent

from shiny import reactive, req
from shiny.reactive._reactives import DestroyedReactiveError
from shiny.session import get_current_session
from shiny.types import SilentOperationInProgressException

task_restart_context = ContextVar("task_restart_context", default=None)


class SessionExtendedTask(reactive.ExtendedTask):
    """Guard Shiny's deferred completion callback against session destruction.

    Shiny schedules result publication separately from the worker task. Cancelling
    the worker alone therefore cannot prevent a write to destroyed reactive values.
    This adapter mirrors that callback, checking closure inside the reactive lock.
    """

    def __init__(self, func):
        super().__init__(func)
        self._closed = False
        self._generation = 0
        self._restart_owner = None
        self._pending_restart = None
        self._arguments = reactive.Value()
        session = get_current_session()
        if session is not None:
            session.on_ended(self.close)
            session.on_destroy(self.close)

    def close(self):
        self._closed = True
        self.cancel()

    def invoke(self, *args, **kwargs):
        if not self._closed:
            self._restart_owner = task_restart_context.get()
            try:
                super().invoke(*args, **kwargs)
            except DestroyedReactiveError:
                self.close()

    def require_current(self, *args, **kwargs):
        """Require a successful result for these export-producing arguments."""
        req(equivalent(self._arguments(), (args, kwargs)))
        self.result()

    def _invoke(self, *args, **kwargs):
        super()._invoke(*args, **kwargs)
        self._running_arguments = (args, kwargs)

    def suspend(self):
        with reactive.isolate():
            if not self._closed and self.status() == "running":
                self._pending_restart = self._restart_owner
                self.cancel()

    def resume(self):
        owner, self._pending_restart = self._pending_restart, None
        if owner is not None and not self._closed:
            with reactive.isolate():
                owner.set(owner.get() + 1)

    def result(self):
        """Clear pending outputs normally; busy trackers own progress display."""
        try:
            return super().result()
        except SilentOperationInProgressException:
            req(False)

    def cancel(self):
        """Cancel queued/running work and invalidate deferred publication."""
        if os.environ.get("PYTHAGORAS_DIAGNOSTICS") == "1":
            logging.getLogger("pythagoras.tasks").info(
                "cancel task=%s instance=%s reason=%s generation=%s running=%s queued=%s",
                self._func.__qualname__, hex(id(self)),
                "session_closed" if self._closed else (
                    "module_suspended" if self._pending_restart is not None else "cancel_called"
                ), self._generation, self._task is not None, len(self._invocation_queue),
            )
        self._generation += 1
        super().cancel()
        self._task = None
        if not self._closed:
            try:
                with reactive.isolate():
                    if self.status() == "running":
                        self.status.set("cancelled")
            except DestroyedReactiveError:
                self._closed = True

    def _done_callback(self, task):
        if self._task is not task:
            if not task.cancelled():
                task.exception()
            return
        self._task = None
        generation = self._generation
        arguments = self._running_arguments
        # Retrieve failures even when shutdown prevents publishing them.
        error = None if task.cancelled() else task.exception()

        async def publish():
            async with reactive.lock():
                if self._closed or generation != self._generation:
                    return
                try:
                    if task.cancelled():
                        self.status.set("cancelled")
                    elif error is not None:
                        self._arguments.set(arguments)
                        self.error.set(error)
                        self.status.set("error")
                    else:
                        self._arguments.set(arguments)
                        self.value.set(task.result())
                        self.status.set("success")
                except DestroyedReactiveError:
                    # A module scope can be destroyed independently of its session.
                    self.close()
                    return
                await reactive.flush()
                if (not self._closed and generation == self._generation
                        and self._invocation_queue):
                    self._invocation_queue.pop(0)()
                    await reactive.flush()

        asyncio.create_task(publish())
