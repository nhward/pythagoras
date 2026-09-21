"""Opt-in source recording for calculations shared by cards and workers.

Decorate reusable calculations with ``@recordable`` and enter a card's
``@this.record_context`` inside Shiny/task decorators. Contexts follow asyncio
and ``asyncio.to_thread`` calls without changing module globals or class identity.
Ordinary script/test calls outside a recording context do not collect anything.
"""
from __future__ import annotations

import inspect
import textwrap
from contextvars import ContextVar
from functools import wraps

_ACTIVE_CODE = ContextVar("active_code_registry", default=None)


def source_for(function):
    try:
        return textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
    except (OSError, TypeError):
        return "<source not available>"


def _store(registry, name, source, module):
    # Shared helpers from different modules occasionally have the same name.
    if name in registry and registry[name] != source:
        name = f"{module}.{name}"
    registry[name] = source


def recordable(function):
    """Record a called function, or a used class, in the active card's listing.

    Classes retain their identity for sklearn cloning and pickle. Their own
    methods record the complete class definition, including on fitted instances
    restored from a pipeline. Dataclass-generated methods need no source lookup.
    """
    source = source_for(function)
    name, module = function.__name__, function.__module__

    def decorate(method):
        def record():
            registry = _ACTIVE_CODE.get()
            if registry is not None:
                _store(registry, name, source, module)

        if inspect.iscoroutinefunction(method):
            @wraps(method)
            async def wrapper(*args, **kwargs):
                record()
                return await method(*args, **kwargs)
        else:
            @wraps(method)
            def wrapper(*args, **kwargs):
                record()
                return method(*args, **kwargs)
        # functools.wraps does not copy the C-level lru_cache control methods.
        for attribute in ("cache_info", "cache_clear", "cache_parameters"):
            if hasattr(method, attribute):
                setattr(wrapper, attribute, getattr(method, attribute))
        return wrapper

    if inspect.isclass(function):
        for attribute, method in list(vars(function).items()):
            if isinstance(method, (staticmethod, classmethod)):
                setattr(function, attribute, type(method)(decorate(method.__func__)))
            elif inspect.isfunction(method):
                setattr(function, attribute, decorate(method))
        return function
    return decorate(function)


def recording_context(registry, function):
    """Activate one card's registry for a callback without listing its plumbing."""
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def wrapper(*args, **kwargs):
            token = _ACTIVE_CODE.set(registry)
            try:
                return await function(*args, **kwargs)
            finally:
                _ACTIVE_CODE.reset(token)
    else:
        @wraps(function)
        def wrapper(*args, **kwargs):
            token = _ACTIVE_CODE.set(registry)
            try:
                return function(*args, **kwargs)
            finally:
                _ACTIVE_CODE.reset(token)
    return wrapper


def recorded_job(function, *args, **kwargs):
    """Return a job's value and executed source for a thread/process boundary.

    Joblib workers do not inherit ContextVars. Return plain source strings rather
    than passing a card or registry to another process.
    """
    registry = {}
    return recording_context(registry, function)(*args, **kwargs), registry


def collect_job_code(results):
    """Merge completed job recordings into the caller and return their values."""
    registry = _ACTIVE_CODE.get()
    values = []
    for value, sources in results:
        values.append(value)
        if registry is not None:
            for name, source in sources.items():
                _store(registry, name, source, "worker")
    return values
