"""Explicit prerequisites for publishing a mutable card's data."""
import asyncio
from card_position import CardPositionError

from shiny import reactive, req
from shiny.reactive._core import get_current_context
from shiny.types import SilentException


def equivalent(left, right):
    if hasattr(left, "tolist") and hasattr(right, "tolist") and not hasattr(left, "equals"):
        return equivalent(left.tolist(), right.tolist())
    if hasattr(left, "equals"):
        return bool(left.equals(right))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    return left == right


class ExportReadiness:
    def __init__(self):
        self.inputs = {}
        self.expected = reactive.Value({})
        self._acknowledged = {}
        self._generation = 0
        self.checks = []
        self.phase = reactive.Value(None)

    def require_inputs(self, *names, allow_none=False, when=lambda: True):
        for name in names:
            self.inputs[name] = (allow_none, when)

    def expect(self, name, value):
        if name not in self.inputs:
            return
        with reactive.isolate():
            previous = self.expected.get()
            old = previous.get(name)
            if old is not None and self._acknowledged.get(name) != old[0] and equivalent(old[1], value):
                return
            self._generation += 1
            self.expected.set({**previous, name: (self._generation, value)})

    def check_inputs(self, inputs):
        expected = self.expected.get()
        acknowledged = []
        for name, (allow_none, when) in self.inputs.items():
            if not when():
                continue
            reader = inputs[name] if isinstance(inputs, dict) else getattr(inputs, name)
            value = reader()
            req(allow_none or value is not None)
            if name in expected:
                generation, desired = expected[name]
                if self._acknowledged.get(name) != generation:
                    req(equivalent(value, desired))
                    acknowledged.append((name, generation))
        # Once acknowledged, a later user edit must not be mistaken for an
        # unfinished restoration. Do not invalidate on acknowledgement itself.
        for name, generation in acknowledged:
            self._acknowledged[name] = generation

    def check(self, inputs):
        self.check_inputs(inputs)
        for check in self.checks:
            check()

    def bind(self, module, source, inputs, session):
        published = reactive.Value()
        ready = reactive.Value(False)
        revision = 0
        closed = False

        def close():
            nonlocal closed
            closed = True

        session.on_ended(close)

        async def flush_publication():
            # on_flushed runs after Shiny has drained its effect queue. Resume
            # that queue after releasing the current reactive lock, even when
            # there are no further browser messages or background completions.
            async with reactive.lock():
                if not closed:
                    await reactive.flush()

        @module.reactable()
        def prepare():
            nonlocal revision
            revision += 1
            current = revision
            ready.set(False)

            def invalidated():
                nonlocal revision
                revision += 1
                ready.set(False)

            get_current_context().on_invalidate(invalidated)
            try:
                self.phase.set("configuration")
                self.check_inputs(inputs)
                self.phase.set("calculation")
                for check in self.checks:
                    check()
                value = source()
                req(value is not None)
            except CardPositionError as error:
                # Publish the positioning failure so Card can explain it and
                # pass through upstream data, rather than wait indefinitely.
                value = error
                self.phase.set(None)
            except SilentException:
                return
            except Exception as error:  # noqa: BLE001
                # Report failures through the public calculation. Letting an
                # exception escape a producer effect would close the session.
                value = error
                self.phase.set(None)

            async def commit():
                if closed or current != revision:
                    return
                with reactive.isolate():
                    if not published.is_set() or not equivalent(published.get(), value):
                        published.set(value)
                ready.set(True)
                self.phase.set(None)
                asyncio.create_task(flush_publication())

            session.on_flushed(commit, once=True)

        @reactive.calc
        def export():
            req(ready())
            value = published()
            if isinstance(value, Exception):
                raise value
            return value

        return export
