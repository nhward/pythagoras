"""Cancellation-safe timers that invalidate at the next Shiny session idle cycle."""
import asyncio
import time
from asyncio import create_task

from shiny.otel._core import detached_otel_context
from shiny.reactive._core import get_current_context
from shiny.session import get_current_session


def invalidate_later(delay):
    context = get_current_context()
    session = get_current_session()
    deadline = time.monotonic() + delay
    cancellable = True
    unsubscribe = None

    def cleanup():
        nonlocal unsubscribe
        if unsubscribe is not None:
            callback, unsubscribe = unsubscribe, None
            callback()

    async def expire():
        nonlocal cancellable
        try:
            await asyncio.sleep(max(0., deadline - time.monotonic()))
            cancellable = False
            with detached_otel_context():
                if session is not None:
                    session.run_once_when_idle(context.invalidate)
                else:
                    context.invalidate()
        except asyncio.CancelledError:
            return
        finally:
            cleanup()

    task = create_task(expire())

    def cancel():
        if cancellable and not task.done():
            task.cancel()

    context.on_invalidate(cancel)
    if session is not None:
        unsubscribe = session.on_ended(cancel)
    # A task cancelled before its first execution never enters expire/finally.
    task.add_done_callback(lambda _task: cleanup())
