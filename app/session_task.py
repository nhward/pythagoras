"""Extended tasks that stop publishing results when their session closes."""
import asyncio

from shiny import reactive
from shiny.reactive._reactives import DestroyedReactiveError
from shiny.session import get_current_session


class SessionExtendedTask(reactive.ExtendedTask):
    """Guard Shiny's deferred completion callback against session destruction.

    Shiny schedules result publication separately from the worker task. Cancelling
    the worker alone therefore cannot prevent a write to destroyed reactive values.
    This adapter mirrors that callback, checking closure inside the reactive lock.
    """

    def __init__(self, func):
        super().__init__(func)
        self._closed = False
        session = get_current_session()
        if session is not None:
            session.on_ended(self.close)
            session.on_destroy(self.close)

    def close(self):
        self._closed = True
        self.cancel()

    def invoke(self, *args, **kwargs):
        if not self._closed:
            super().invoke(*args, **kwargs)

    def _done_callback(self, task):
        if self._task is not task:
            return
        self._task = None
        # Retrieve failures even when shutdown prevents publishing them.
        error = None if task.cancelled() else task.exception()

        async def publish():
            async with reactive.lock():
                if self._closed:
                    return
                try:
                    if task.cancelled():
                        self.status.set("cancelled")
                    elif error is not None:
                        self.error.set(error)
                        self.status.set("error")
                    else:
                        self.value.set(task.result())
                        self.status.set("success")
                except DestroyedReactiveError:
                    # A module scope can be destroyed independently of its session.
                    self.close()
                    return
                await reactive.flush()
                if not self._closed and self._invocation_queue:
                    self._invocation_queue.pop(0)()
                    await reactive.flush()

        asyncio.create_task(publish())
