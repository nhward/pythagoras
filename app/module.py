###########################
## Class: Module ##
###########################
## This class inherits from abc.ABC
## It provides:
##    Configuration
##        Reading config/pythagoras.json which describes configuration setting and layout
##    Guide services via shepherd:
##       _patch_module() function which patches the shiny.ui.input_* & shiny.ui.output_*
##       _make_wrapper() function used by patch_module
##       _ui_patched class variable prevent repeated calls
##       create_run_tour() function send payload to browser for a shepherd tour
##    Logging services:
##       self.log per-instance logger (with format)
##       cls.log for class level logging (with format)
##    Namespace management:
##       ns() function
##       namespace attribute
##    Instances:
##       Class level registry of cards (ns, object)
##       reset(): to drop an instance
##    Abstract Interfaces:
##       call_ui(): module's ui function
##       call_server(): Module's server function
##       guidedDiv(): Wraps a div as a guidable element
##    reactable
##       @reactable decorator to wrap @reactive.calc, @reactive.event, @reactive.effect (use calc=True for reactive.calc)
##       Instance level suspend() & resume() functions
##       Instance level list of reactables/resumables
##    Output reactive like R's render_print style:
##       @capture_print decorator
##    Code recording:
##       @record_code decorator
##       Instance-level dict repository of function code blocks
##       Code key-retrieval mechanism
##       Long HTML listing of all code blocks
##    Input Value Settling:
##       @settle(seconds: float = 2, bypass_during_tests: bool = True) Delays passing a reactive until it has ceased changing 
##    Create cards method that looks for files in "cards" and imports them and calls their instance() method
##      application(): method that creates the shiny app object
##      Run(): method that either runs the single-card app in the viewer
import ast
import asyncio
import functools
import html
import inspect
import io
import json
import logging
import numbers
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from contextlib import redirect_stdout
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from os import environ
from pathlib import Path
from typing import ClassVar

import shinywidgets as _sw
from code_recording import recordable, recording_context, source_for
from faicons import icon_svg as icon
from jsonschema import ValidationError, validate
from shiny import App, reactive, req, ui
from shiny import ui as _ui
from shiny.types import SilentException

_UNSET = object()


def _settle_snapshot(value):
    """Snapshot plain input containers without copying application objects.

    proxy_data and its immutable processing records contain mappingproxy values;
    they must travel through this generic decorator without pickling/deepcopy.
    Reactive producers should replace application values rather than mutate them.
    """
    if type(value) is dict:
        return {key: _settle_snapshot(item) for key, item in value.items()}
    if type(value) is list:
        return [_settle_snapshot(item) for item in value]
    if type(value) is tuple:
        return tuple(_settle_snapshot(item) for item in value)
    if type(value) is set:
        return value.copy()
    return value

_ACTIVE_CONFIGURATION_MODULE: ContextVar["Module | None"] = ContextVar(
    "active_configuration_module",
    default=None,
)

_NON_PERSISTENT_INPUT_CONSTRUCTORS = frozenset({
    "input_action_button",
    "input_action_link",
    "input_bookmark_button",
    "input_file",
    "input_task_button",
})


class ApplicationLogHandler(logging.Handler):
    """Keep a bounded, thread-safe snapshot of application log records."""

    def __init__(self, capacity: int = 5_000):
        super().__init__(level=logging.DEBUG)
        self._records = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                traceback = formatter.formatException(record.exc_info)
                message = f"{message}\n{traceback}"
            item = {
                "Time": datetime.fromtimestamp(record.created).astimezone(),
                "Level": record.levelname,
                "Logger": record.name,
                "Message": message,
                "Source": record.pathname,
                "Line": record.lineno,
                "Thread": record.threadName,
            }
            self.acquire()
            try:
                self._records.append(item)
            finally:
                self.release()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def snapshot(self) -> list[dict[str, object]]:
        """Return copies so readers cannot mutate the shared buffer."""
        self.acquire()
        try:
            return [record.copy() for record in self._records]
        finally:
            self.release()


class BusyTracker:
    """Collect session-local extended tasks for a reactive busy display."""

    def __init__(self):
        self._tasks: list[tuple[object, str]] = []

    def track(self, message: str):
        """Register an ExtendedTask without changing its invocation or result."""
        def decorator(task):
            status = getattr(task, "status", None)
            if not callable(status):
                raise TypeError(
                    "@busy.track must be placed above @this.extended_task "
                    "or @reactive.extended_task"
                )
            self._tasks.append((task, message))
            return task

        return decorator

    def ui(self):
        """Return a spinner and the messages for tasks currently running."""
        messages = [
            message
            for task, message in self._tasks
            if task.status() == "running"
        ]
        if not messages:
            return None
        return ui.div(
            ui.span(
                class_="spinner-border spinner-border-sm me-2",
                role="status",
                aria_hidden="true",
            ),
            ui.span("; ".join(dict.fromkeys(messages))),
            class_="text-info text-center d-block",
            role="status",
            aria_live="polite",
        )


class Module(ABC):
    """
    Base class relating to shiny modules that:
      - Maintains namespace,
      - Provides Guide services via shepherd (incl. patching input/output calls)
      - Add abstract interfaces (call_ui, call_server)
      - Decorates Reactive functions (record_code, capture_print, debounce, throttle, reactable)
      - Provides logging services
      - Loads from "/cards" folder
      - Creates the shiny App
      - Allows single files to be run in a viewer window
    """
    IS_SHINYLIVE = sys.platform == "emscripten"
    N_JOBS = 1 if IS_SHINYLIVE else -1
    ROOT = Path(__file__).resolve().parent   # pythagorus/app
    ModSession = None
    Instances: ClassVar[dict] = {}  # class level dictionary of all instances keyed by their namespaces (possibly including deleted ones with empty values)
    script_list: ClassVar[list] = [
        ROOT / "www" / "console.js",
        ROOT / "www" / "jquery-ui-1.14.2.min.js",
        ROOT / "www" / "sortable-1.15.7.min.js",
        ROOT / "www" / "pythagoras.js"
    ]
    css_list: ClassVar[list] = [
        ROOT / "www" / "pythagoras.css",
        ROOT / "www" / "shepherd-15.3.0.css",
        ROOT / "www" / "animate.css"
    ]        
    mjs_list: ClassVar[list] = [
        "guide.mjs",
    ]
    _ui_patched = False  # whether patching has been performed
    min_log_level = logging.DEBUG
    log_handler = ApplicationLogHandler()
    log = logging.getLogger("pythagoras")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(h)
        log.propagate = False
        log.setLevel(min_log_level)

        # Load schema
        with open(ROOT / "pythagoras.schema.json") as f:
            schema = json.load(f)
        # Load config
        with open(ROOT / "default.pythagoras.json") as f:
            config = json.load(f)
        try:
            validate(instance=config, schema=schema)
        except ValidationError as e:
            log.exception("Invalid configuration")
            raise ValueError(f"Invalid configuration: {e.message}")
    
    if log_handler not in log.handlers:
        log.addHandler(log_handler)

    MaxInstances = config.get("settings", {}).get("max_dupl_cards")

    @classmethod
    def runtime_mode(cls, session) -> str:
        """Classify the runtime as Shinylive, local, or remote server."""
        if cls.IS_SHINYLIVE:
            return "shinylive"
        client_data = getattr(session, "clientdata", None)
        if client_data is None:
            client_data = session.client_data
        hostname = client_data.url_hostname()
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return "local"
        return "server"


    # Initialiser
    def __init__(self, name, *args, **kwargs): # will be inherited by child classes
        super().__init__(*args, **kwargs)  # play nicely with multiple inheritance
        #namespace
        self.name = name
        ns = self.name
        if ns in self.Instances:
            for i in range(self.MaxInstances-1):
                ns_ = f"{self.name}_{i}"
                if ns_ not in self.Instances:
                    ns = ns_
                    break
            else:
                raise ValueError(f"Too many instances of module '{self.name}': exceeded maximum of {self.MaxInstances}")
        self.Instances[ns] = self
        self.namespace = ns
        # reactives
        self._imports = reactive.Value()
        self._upstream = None
        self.output_data = None
        # Guide
        self._shepherd_steps = {}
        if not Module._ui_patched: # only patch once
            # Patch shiny.ui (inputs/outputs/downloads)
            self._patch_module(_ui)
            # Patch shinywidgets (Plotly/ipywidgets output binding)
            self._patch_module(_sw)
            Module._ui_patched = True
        # Logger
        base = logging.getLogger(self.name)
        if not base.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            ))
            base.addHandler(h)
        if Module.log_handler not in base.handlers:
            base.addHandler(Module.log_handler)
        base.propagate = False
        self.log = base  # or a LoggerAdapter if you want extra fields
        self.log.setLevel(Module.min_log_level)
        # instance registries
        self.reactables = [] # An instance-level list of all reactable reactives
        # Code recording
        self.code_registry = {} # An instance-level Code-Registry
        # Generic bookmark state. Special cards may disable this and provide
        # their own configuration_state/restore_configuration_state methods.
        self.generic_configuration_state = True
        self._configuration_input = None
        self._configuration_input_ids: set[str] = set()
        self._configuration_excluded_input_ids: set[str] = set()
        self._configuration_deferred_input_ids: set[str] = set()
        self._configuration_restored_ids: set[str] = set()
        self._configuration_warned_ids: set[str] = set()
        self._configuration_choice_codecs: dict[
            str, tuple[dict[str, object], type | None]
        ] = {}
        self._generic_restored_inputs: dict[str, object] = {}
        self._generic_extra_state: dict[str, object] = {}
        self._configuration_state_fields: set[str] = set()
        self._configuration_state_providers: dict[str, object] = {}
        self._generic_state_loaded = False

    def configuration_ui_context(self):
        """Make this module discoverable while its input UI is constructed."""
        module = self

        class ConfigurationContext:
            def __enter__(self):
                self.token = _ACTIVE_CONFIGURATION_MODULE.set(module)
                return module

            def __exit__(self, exc_type, exc_value, traceback):
                _ACTIVE_CONFIGURATION_MODULE.reset(self.token)

        return ConfigurationContext()

    @staticmethod
    def _json_value(value):
        """Return a detached strict-JSON representation of an input value."""
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

    def restore_configuration_state(self, state) -> None:
        """Validate and retain generic input state until UI construction."""
        if not self.generic_configuration_state:
            return
        if not isinstance(state, dict):
            self.log.warning(
                "Ignoring malformed non-object bookmark state for %s",
                self.name,
            )
            self._generic_state_loaded = True
            return
        inputs = state.get("inputs", {})
        if not isinstance(inputs, dict):
            self.log.warning(
                "Ignoring malformed input state for %s; current defaults will be used",
                self.name,
            )
            inputs = {}
        extra = sorted(set(state) - {"inputs"})
        unknown_extra = sorted(set(extra) - self._configuration_state_fields)
        if unknown_extra:
            self.log.warning(
                "Bookmark state for %s contains unrecognised field(s) %s; "
                "they will be retained for compatibility",
                self.name,
                ", ".join(unknown_extra),
            )
        self._generic_extra_state = {}
        for name in extra:
            try:
                self._generic_extra_state[name] = self._json_value(state[name])
            except (TypeError, ValueError):
                self.log.warning(
                    "Ignoring non-JSON bookmark field %s.%s",
                    self.name,
                    name,
                )
        restored = {}
        for input_id, value in inputs.items():
            if not isinstance(input_id, str):
                self.log.warning(
                    "Ignoring non-text bookmark input ID %r for %s",
                    input_id,
                    self.name,
                )
                continue
            try:
                restored[input_id] = self._json_value(value)
            except (TypeError, ValueError):
                self.log.warning(
                    "Ignoring non-JSON bookmark value for %s.%s",
                    self.name,
                    input_id,
                )
        self._generic_restored_inputs = restored
        self._generic_state_loaded = True

    def register_configuration_state_field(self, name: str) -> None:
        """Declare a card-owned JSON field outside the generic input mapping."""
        if not isinstance(name, str) or not name or name == "inputs":
            raise ValueError(
                "A configuration state field must be non-empty text other than "
                "'inputs'"
            )
        self._configuration_state_fields.add(name)

    def restored_configuration_field(self, name: str, default=None):
        """Return a detached value for a declared card-owned state field."""
        self.register_configuration_state_field(name)
        if name not in self._generic_extra_state:
            return default
        return self._json_value(self._generic_extra_state[name])

    def register_configuration_state_provider(self, name: str, provider) -> None:
        """Supply the current value of a declared card-owned state field."""
        self.register_configuration_state_field(name)
        if not callable(provider):
            raise TypeError("A configuration state provider must be callable")
        self._configuration_state_providers[name] = provider

    def register_configuration_input(self, input_id: str) -> None:
        """Include a custom Shiny input which has no ``ui.input_*`` widget."""
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("A configuration input ID must be non-empty text")
        self._configuration_input_ids.add(input_id)

    def exclude_configuration_input(self, input_id: str) -> None:
        """Exclude a transient input whose value is represented elsewhere."""
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("A configuration input ID must be non-empty text")
        self._configuration_excluded_input_ids.add(input_id)

    def defer_configuration_input(self, input_id: str) -> None:
        """Persist an input but let its card restore it after dynamic UI exists."""
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("A configuration input ID must be non-empty text")
        self._configuration_deferred_input_ids.add(input_id)
        self._configuration_input_ids.add(input_id)

    def restored_configuration_input(self, input_id: str, default=None):
        """Return detached restored state for a manually registered input."""
        self.register_configuration_input(input_id)
        if input_id not in self._generic_restored_inputs:
            if self._generic_state_loaded and input_id not in self._configuration_warned_ids:
                self.log.warning(
                    "Bookmark has no value for new input %s.%s; using its default",
                    self.name,
                    input_id,
                )
                self._configuration_warned_ids.add(input_id)
            return default
        self._configuration_restored_ids.add(input_id)
        return self._json_value(self._generic_restored_inputs[input_id])

    @staticmethod
    def _declared_choice_values(choices) -> list[object]:
        """Flatten Shiny choices while retaining their declared value types."""
        if isinstance(choices, dict):
            values = []
            for key, label in choices.items():
                if isinstance(label, dict):
                    values.extend(Module._declared_choice_values(label))
                else:
                    values.append(key)
            return values
        elif isinstance(choices, (list, tuple, set)):
            return list(choices)
        return []

    @classmethod
    def _choice_codec(cls, choices) -> tuple[dict[str, object], type | None]:
        values = cls._declared_choice_values(choices)
        mapping = {str(value): value for value in values}
        numeric = [value for value in values if not isinstance(value, bool)]
        if numeric and all(isinstance(value, numbers.Integral) for value in numeric):
            inferred_type = int
        elif numeric and all(isinstance(value, numbers.Real) for value in numeric):
            inferred_type = float
        else:
            inferred_type = None
        return mapping, inferred_type

    @staticmethod
    def _coerce_choice_value(value, codec):
        mapping, inferred_type = codec

        def coerce(item):
            declared = mapping.get(str(item), _UNSET)
            if declared is not _UNSET:
                return declared
            if inferred_type is not None and isinstance(item, str):
                try:
                    return inferred_type(item)
                except ValueError:
                    pass
            return item

        if isinstance(value, (list, tuple)):
            return [coerce(item) for item in value]
        return coerce(value)

    @classmethod
    def _choices_with_restored_value(cls, choices, selected):
        """Make absent saved selections legal until dynamic choices arrive."""
        selected_values = selected if isinstance(selected, list) else [selected]
        existing = {
            str(value) for value in cls._declared_choice_values(choices)
        }
        missing = [value for value in selected_values if str(value) not in existing]
        if not missing:
            return choices
        if isinstance(choices, dict):
            return {**{str(value): str(value) for value in missing}, **choices}
        return [*missing, *(list(choices) if choices is not None else [])]

    def _configuration_kwargs(self, func, input_id: str, kwargs: dict) -> dict:
        """Register an input and apply its bookmarked constructor value."""
        if not self.generic_configuration_state:
            return kwargs
        if input_id in self._configuration_excluded_input_ids:
            return kwargs
        if input_id in self._configuration_deferred_input_ids:
            self._configuration_input_ids.add(input_id)
            return kwargs
        name = getattr(func, "__name__", "")
        if name in _NON_PERSISTENT_INPUT_CONSTRUCTORS:
            return kwargs
        signature = inspect.signature(func)
        if "selected" in signature.parameters:
            parameter = "selected"
        elif "value" in signature.parameters:
            parameter = "value"
        else:
            self.log.warning(
                "Input %s.%s cannot be restored generically; using its default",
                self.name,
                input_id,
            )
            return kwargs
        self._configuration_input_ids.add(input_id)
        codec = None
        if parameter == "selected" and "choices" in signature.parameters:
            codec = self._choice_codec(kwargs.get("choices"))
            self._configuration_choice_codecs[input_id] = codec
        if input_id not in self._generic_restored_inputs:
            if self._generic_state_loaded and input_id not in self._configuration_warned_ids:
                self.log.warning(
                    "Bookmark has no value for new input %s.%s; using its default",
                    self.name,
                    input_id,
                )
                self._configuration_warned_ids.add(input_id)
            return kwargs
        value = self._generic_restored_inputs[input_id]
        if codec is not None:
            value = self._coerce_choice_value(value, codec)
            kwargs["choices"] = self._choices_with_restored_value(
                kwargs.get("choices"),
                value,
            )
        kwargs[parameter] = value
        self._configuration_restored_ids.add(input_id)
        return kwargs

    def configuration_state(self) -> dict[str, object]:
        """Capture JSON-compatible values for the module's declared inputs."""
        if not self.generic_configuration_state:
            return {}
        values = {
            input_id: value
            for input_id, value in self._generic_restored_inputs.items()
            if input_id not in self._configuration_excluded_input_ids
        }
        scoped_input = self._configuration_input
        for input_id in sorted(self._configuration_input_ids):
            if scoped_input is None:
                continue
            try:
                current = getattr(scoped_input, input_id)()
                codec = self._configuration_choice_codecs.get(input_id)
                if codec is not None:
                    current = self._coerce_choice_value(current, codec)
                values[input_id] = self._json_value(current)
            except SilentException:
                continue
            except (AttributeError, TypeError, ValueError):
                self.log.warning(
                    "Input %s.%s is not JSON-compatible and was not updated "
                    "in the bookmark",
                    self.name,
                    input_id,
                )
        self.warn_obsolete_configuration_inputs()
        extra = dict(self._generic_extra_state)
        for name, provider in self._configuration_state_providers.items():
            try:
                extra[name] = self._json_value(provider())
            except SilentException:
                continue
            except (TypeError, ValueError):
                self.log.warning(
                    "Configuration field %s.%s is not JSON-compatible and was "
                    "not updated in the bookmark",
                    self.name,
                    name,
                )
        return {**extra, "inputs": values}

    def warn_obsolete_configuration_inputs(self) -> None:
        """Report saved IDs which no constructor in the current card declared."""
        obsolete = set(self._generic_restored_inputs) - self._configuration_input_ids
        for input_id in sorted(obsolete - self._configuration_warned_ids):
            self.log.warning(
                "Bookmark input %s.%s is no longer present; retaining it for "
                "compatibility",
                self.name,
                input_id,
            )
            self._configuration_warned_ids.add(input_id)


    def reset(self):
        self.log.debug(f"🧹 Cleaning up namespace {self.namespace}")
        self._upstream = None
        self.output_data = None
        reuse_cards = getattr(
            self,
            "_reuse_cards",
            self.config.get("settings", {}).get("reuse_cards"),
        )
        if reuse_cards:
            self.Instances.pop(self.namespace)
        else:
            self.Instances[self.namespace] = None

    def input_data(self):
        """Read this module's connected reactive source or standalone input."""
        if self._upstream is not None:
            source = self._upstream()
            if source is not None:
                value = source()
                req(value is not None)
                return value
        req(self._imports.is_set())
        return self._imports.get()

    def has_input_data(self) -> bool:
        """Return whether the current source can provide a value."""
        try:
            return self.input_data() is not None
        except SilentException:
            return False

    Packet = namedtuple("Packet", "data name")

    def guidedDiv(self, *children, id: str, class_ = None, guide = None, title = None,
        text: str = "", position: str = "bottom", priority: int = 0, **kwargs):
        """
        Create a div that can participate in the Shepherd guide.
        When guide is supplied, the div receives a namespaced id and is registered
        as a Shepherd tour target. No additional wrapper is needed.
        """
        # Build the actual inner div first
        actual_id = guide.ns(id) if guide is not None else id
        widget = _ui.div(*children, id=actual_id, class_=class_, style = "overflow: hidden", **kwargs)
        if guide is None:
            return widget
        # Shepherd attaches to the wrapper, not the inner element
        guide._shepherd_steps[actual_id] = {
            "title": title or id,
            "text": text or "",
            "position": position,
            "priority": priority,
        }
        return widget

    # Namespace function (div ids need this as they are not namespaced by the decorator)
    def ns(self, id): # this is equivilent to what @module.ui does
        return f"{self.namespace}-{id}"

    def section_normalise(name: str) -> str:
        return name.strip().replace(" ", "_")

    @staticmethod
    def running_under_tests():
        return (
            "PYTEST_CURRENT_TEST" in environ or
            "pytest" in sys.modules or
            any("pytest" in arg for arg in sys.argv)
        )
        

    @staticmethod
    def running_directly(name):
        return name == "__main__"

    @staticmethod
    def running_in_background():
        return "ipykernel" in sys.modules


    def _tour_steps_payload(self) -> str:
        """Build the JSON payload for Shepherd from _shepherd_steps."""
        sorted_steps = sorted(
            self._shepherd_steps.items(),
            key=lambda x: x[1].get("priority", 0),
            reverse=True,
        )
        steps = [
            {
                "id": id,
                "selector": f"#{id}",
                "title": step["title"],
                "text": step["text"],
                "position": step.get("position", "auto"),
            }
            for id, step in sorted_steps
        ]
        return json.dumps(steps)

    async def create_run_tour(self, session):
        json_steps = self._tour_steps_payload()
        await session.send_custom_message("create_run_tour", json_steps)

    def extended_task(self, func):
        """Create a Shiny ExtendedTask with consistent lifecycle logging."""
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@extended_task requires an async function")

        @reactive.extended_task
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            task_name = func.__name__
            started = time.perf_counter()
            self.log.debug("Extended task %s started", task_name)
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.log.info(
                    "Extended task %s completed in %.3f seconds",
                    task_name,
                    elapsed,
                )

        return wrapped

    def _make_wrapper(self, func, kind = "input"):
        """
        Wrap a Shiny UI function to register a shepherd step
        guide_instance: the Guide instance whose _shepherd_steps we update
        func: the original ui function
        """
        @functools.wraps(func)
        def wrapped(id, *, guide : Module = None, label = None, title = None, text = None, position = "bottom", priority = 0, **kwargs):
            owner = guide or _ACTIVE_CONFIGURATION_MODULE.get()
            if kind == "input" and isinstance(owner, Module):
                kwargs = owner._configuration_kwargs(func, id, dict(kwargs))
            # Call original Shiny UI function
            sig = inspect.signature(func)
            if "label" in sig.parameters:
                widget = func(id = id, label = label, **kwargs)
            else:
                widget = func(id = id, **kwargs)
            if  guide is None:
                # Then do not guide this input/output ui element.
                return widget
            # guide namespace id
            wid = f"{guide.ns(id)}_wrapper"
            # Register shepherd step in this instance
            # self.log.debug(f"Adding {wid} to {guide.name}")
            guide._shepherd_steps[wid] = {
                "title": title or label,
                "text": text or "",
                "position": position,
                "priority": priority
            }
            # Wrap in a div so Shepherd can safely attach
            return _ui.div(widget, id = wid, class_ = "html-fill-container html-fill-item")
        return wrapped

    def _make_navset_wrapper(self, func):
        """Restore card-local navset selections while preserving its API."""
        @functools.wraps(func)
        def wrapped(*args, id=None, selected=None, **kwargs):
            owner = _ACTIVE_CONFIGURATION_MODULE.get()
            if isinstance(owner, Module) and id is not None:
                restored = owner._configuration_kwargs(
                    func,
                    id,
                    {**kwargs, "selected": selected},
                )
                selected = restored.pop("selected", selected)
                kwargs = restored
            return func(*args, id=id, selected=selected, **kwargs)

        return wrapped

    _ui_patch_lock = threading.Lock()


    def _patch_module(self, module) -> None:
        """
        _patch_module() patches modules "shiny.ui" and "shinywidgets" input/output/download functions.
        Patches each module only once - through a lock; originals stored on module._original_funcs.
        """
        if module is None:
            return
        if getattr(module, "_original_funcs", None):
            return
        with self._ui_patch_lock:
            if getattr(module, "_original_funcs", None):
                return
            module._original_funcs = {}
            for name, obj in list(vars(module).items()):
                if not callable(obj):
                    continue
                if name.startswith("navset_"):
                    module._original_funcs[name] = obj
                    setattr(module, name, self._make_navset_wrapper(obj))
                    continue
                if not name.startswith(("input_", "output_", "download_")):
                    continue
                module._original_funcs[name] = obj
                setattr(module, name, self._make_wrapper(obj, name.split("_", 1)[0]))

    def capture_print(self, func):
        """
        Decorator to capture all print() output from a function
        and return it as a string.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = func(*args, **kwargs)
            printed_text = buf.getvalue()
            if result is not None:
                printed_text += str(result)
            return printed_text
        return wrapper    


    def reactable(
        self,
        *,
        triggers=None,
        suspended: bool = True,
        default=_UNSET,
        calc: bool = False,
    ):
        # Universal reactable decorator
        #   reactable decorator to replace @reactive.calc, @reactive.event, @reactive.effect, @reactive.calc
        #   Suspends all registered reactable reactives.
        #   Auto-detects the nature of the wrapped function and applies the appropriate reactive decorators (use calc=True for @reactive.calc)
        #   Can be suspended/resumed with self.suspend() & self.resume()
        #   It handles async functions
        # Args:
        #   triggers: Optional reactive inputs (for event observers).
        #   suspended: Start suspended (default True).
        #   default: Value returned if a suspended calc is called. When omitted,
        #            a suspended calc stops its consumers with req(False).
        #   calc: Whether the function is a "reactive.calc"
        def decorator(func):
            enabled = reactive.Value(not suspended)
            is_calc = calc
            is_async = asyncio.iscoroutinefunction(func)
            if is_calc:
                # Handle calc: async or sync
                if is_async:
                    @reactive.calc
                    @functools.wraps(func)
                    async def wrapped():
                        if not enabled():
                            if default is _UNSET:
                                req(False)
                            return default
                        return await func()
                else:
                    @reactive.calc
                    @functools.wraps(func)
                    def wrapped():
                        if not enabled():
                            if default is _UNSET:
                                req(False)
                            return default
                        return func()
            else:
                # Handle effect/event
                if triggers:
                    if is_async:
                        @reactive.effect
                        @reactive.event(*triggers)
                        @functools.wraps(func)
                        async def wrapped():
                            if not enabled():
                                return
                            await func()
                    else:
                        @reactive.effect
                        @reactive.event(*triggers)
                        @functools.wraps(func)
                        def wrapped():
                            if not enabled():
                                return
                            func()
                else:
                    if is_async:
                        @reactive.effect
                        @functools.wraps(func)
                        async def wrapped():
                            if not enabled():
                                return
                            await func()
                    else:
                        @reactive.effect
                        @functools.wraps(func)
                        def wrapped():
                            if not enabled():
                                return
                            func()
            # Preserve the source-function link for introspection decorators.
            # Shiny's Calc_ keeps the name but does not provide __wrapped__.
            wrapped.__wrapped__ = func
            # Control methods
            def suspend():
                enabled.set(False)
            def resume():
                enabled.set(True)
            wrapped.suspend = suspend
            wrapped.resume = resume
            self.reactables.append(wrapped)
            return wrapped
        return decorator


    # Utility functions for mass control
    def suspend(self):
        for w in self.reactables:
            w.suspend()
    def resume(self):
        for w in self.reactables:
            w.resume()


    def record_context(self, func):
        """Scope helper recording to this card; do not display UI/task plumbing.

        Place inside reactive, render, settle and extended-task decorators so the
        context is entered when the callback executes, including after awaits.
        """
        return recording_context(self.code_registry, func)

    def record_code(self, func):
        """Record a computational function when called, including its helpers."""
        if not inspect.isroutine(func):
            # Preserve already-created reactive objects and lifecycle controls.
            # Prefer record_code INSIDE the reactive decorator for run-only capture.
            name = getattr(func, "__name__", type(func).__name__)
            self.code_registry[name] = source_for(func)
            return func
        return recording_context(self.code_registry, recordable(func))

    def retrieve_code(self, func_name):
        # Retrieve recorded code
        if func_name not in self.code_registry:
            raise ValueError(f"{func_name} not recorded")
        return self.code_registry[func_name]

    @staticmethod
    def clean_code(source):
        """Prepare script-oriented source without changing the live functions.

        Parse decorator boundaries rather than matching arbitrary text: decorators
        can span lines, and comments/strings may contain the same spellings.
        Computational decorators (for example dataclass or cache) are retained.
        Shiny readiness guards become ordinary assertions, and application log
        statements become pass statements (so suites remain valid). Inputs and
        async bodies are not rewritten: record helpers beneath those layers.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        removed = set()
        framework = {
            "record_code", "record_context", "recordable", "reactable", "settle", "extended_task",
            "capture_print", "capture.print", "busy.track",
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for decorator in node.decorator_list:
                expression = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = ast.unparse(expression)
                owner, _, member = name.partition(".")
                if (name in {"output", "render_widget", "record_code", "recordable"}
                        or owner in {"render", "reactive"}
                        or member in framework):
                    removed.update(range(decorator.lineno, decorator.end_lineno + 1))
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        def position(line, column):
            # AST column offsets are UTF-8 byte offsets, not character offsets.
            return offsets[line - 1] + len(lines[line - 1].encode()[:column].decode())

        edits = [(offsets[line - 1], offsets[line], "") for line in removed]
        for node in ast.walk(tree):
            # Worker recordings are transport metadata, not part of the analysis.
            # Unwrap their collection while leaving the Parallel expression intact.
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "collect_job_code" and len(node.args) == 1
                    and not node.keywords):
                argument = node.args[0]
                edits.extend([
                    (position(node.lineno, node.col_offset),
                     position(argument.lineno, argument.col_offset), ""),
                    (position(argument.end_lineno, argument.end_col_offset),
                     position(node.end_lineno, node.end_col_offset), ""),
                ])
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                    and isinstance(node.func.func, ast.Name) and node.func.func.id == "delayed"
                    and len(node.func.args) == 1 and isinstance(node.func.args[0], ast.Name)
                    and node.func.args[0].id == "recorded_job" and node.args):
                plain = ast.Call(
                    func=ast.Call(func=ast.Name(id="delayed", ctx=ast.Load()),
                                  args=[node.args[0]], keywords=[]),
                    args=node.args[1:], keywords=node.keywords,
                )
                edits.append((position(node.lineno, node.col_offset),
                              position(node.end_lineno, node.end_col_offset), ast.unparse(plain)))
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            name = ast.unparse(call.func)
            if name in {"req", "ui.req", "shiny.req"} and call.args:
                conditions = " and ".join(f"({ast.unparse(arg)})" for arg in call.args)
                replacement = f"assert {conditions}"
            elif name.startswith("this.log."):
                replacement = "pass"
            else:
                continue
            edits.append((position(node.lineno, node.col_offset),
                          position(node.end_lineno, node.end_col_offset), replacement))
        for start, end, replacement in sorted(edits, reverse=True):
            source = source[:start] + replacement + source[end:]
        return source.strip() + "\n"

    def code_text(self):
        lines = []
        for name, code in list(self.code_registry.items()):
            code = html.escape(self.clean_code(code))
            lines.append(f'<h3># {html.escape(name)}</h3><pre>{code}</pre>')
        return _ui.HTML("<hr>".join(lines))

    def settle(self, seconds: float = 2, bypass_during_tests: bool = True):
        """Publish after a quiet interval, retaining the last settled value meanwhile.

        A separate producer owns the input dependencies and timer. Consumers read
        only the published value, so multiple readers cannot release a candidate
        early or invalidate downstream work while a selection is still changing.
        """
        if seconds < 0:
            raise ValueError("Settling delay must be nonnegative")

        def decorator(function):
            published = reactive.Value()
            arguments = reactive.Value()
            producer = None
            candidate = _UNSET
            changed_at = 0.0

            @wraps(function)
            def wrapper(*args, **kwargs):
                nonlocal producer
                if bypass_during_tests and Module.running_under_tests():
                    return function(*args, **kwargs)
                with reactive.isolate():
                    if not arguments.is_set() or arguments.get() != (args, kwargs):
                        arguments.set((args, kwargs))
                if producer is None:
                    @reactive.effect
                    def publish_when_settled():
                        nonlocal candidate, changed_at
                        call_args, call_kwargs = arguments.get()
                        current = function(*call_args, **call_kwargs)
                        now = time.monotonic()
                        if candidate is _UNSET or current != candidate:
                            candidate = _settle_snapshot(current)
                            changed_at = now
                        remaining = seconds - (now - changed_at)
                        if remaining > 0:
                            reactive.invalidate_later(remaining)
                            return
                        with reactive.isolate():
                            if not published.is_set() or published.get() != candidate:
                                published.set(_settle_snapshot(candidate))
                    producer = publish_when_settled
                return published.get()
            return wrapper

        return decorator

    @staticmethod
    def busy() -> BusyTracker:
        """Create a busy tracker for one server session."""
        return BusyTracker()
    

    # Abstract methods
    @abstractmethod
    def call_ui(self):
        pass

    @abstractmethod
    def call_server(self, input, output, session):
        pass

    def application(self):
        # main ui object for the app
        app_ui = ui.page_fillable(
            ui.head_content(
                ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
                [ui.include_js(script, method = "inline") for script in dict.fromkeys(Module.script_list)],  # iterate through unique js scripts
                [ui.include_css(css, method = "inline") for css in dict.fromkeys(Module.css_list)],  # iterate through unique CSS documents
                [ui.tags.script(type = "module", src=script) for script in dict.fromkeys(Module.mjs_list)] # iterate through unique mjs modules
           ),
            ui.busy_indicators.options(spinner_type = "bars2"),
            ui.busy_indicators.use(),
            ui.page_navbar(
                ui.nav_panel(
                    "Data Prep",
                    ui.div(
                        self.call_ui(),
                        id = "cards-container", # Container for cards
                        class_ = "cards-grid"
                    ),
                    value = "Data_prep"
                ),
                ui.nav_spacer(),
                ui.nav_control(
                    ui.tooltip(
                        ui.input_action_button(
                            id = "FullScreen",  
                            label= None, 
                            icon = icon("expand", title = "Toggle full screen", a11y = "sem"),
                            class_ = "btn rounded-pill btn-sm fa-xl",
                            style = "border: 0px; box-shadow: none; display: block;"
                        ),
                        "Toggle full screen",
                        placement = "bottom"
                    )
                ),
                ui.nav_control(
                    ui.tooltip(
                        ui.input_action_button(
                            id = "Quit",  
                            label = None, 
                            icon = icon("stop", title = "Quit session", a11y = "sem"),
                            class_ = "btn rounded-pill btn-sm fa-xl",
                            style = "border: 0px; box-shadow: none; display: block;"
                        ),
                        "Quit session",
                        placement = "bottom"
                    )
                ),

                id = "Navbar",
                title = ui.tooltip(
                    ui.TagList(ui.tags.img(src="favicon.ico", style="height:2em; margin-right:0.5em;"), ui.span("Pythagoras", class_ = "text-primary")),
                    '"All is number."',
                    placement = "bottom"
                ),
                fillable=True
            )
        )

        # main server function for the app
        def server(input, output, session):
            Module.ModSession = session

            @reactive.effect
            @reactive.event(input.FullScreen)
            async def FullScreen():
                Module.log.info("Full-screen app requested")
                await session.send_custom_message("fullscreen_app", None)
            

            @reactive.effect
            @reactive.event(input.Quit)
            async def Quit():
                Module.log.info("Quit app requested")
                await session.send_custom_message("quit_app", None)
                await session.close()  # in case the window close is ignored
            

            # redirect browser console to the python console
            @reactive.effect
            @reactive.event(input.Console_log)
            def redirect():
                message = input.Console_log()
                level = message['level'].upper()
                if level =="ERROR":
                    Module.log.error(msg = f"<javascript> | {message['text']}")
                elif level == "INFO":
                    Module.log.info(msg = f"<javascript> | {message['text']}")
                elif level == "WARNING":
                    Module.log.warning(msg = f"<javascript> | {message['text']}")
                else:
                    Module.log.debug(msg = f"<javascript> | {message['text']}")


            self.call_server(input, output, session)
            
            self.resume()
            async def after_flush(card = self):
                await session.send_custom_message("init_card", {"id": card.ns("Card")})
            session.on_flushed(after_flush, once=True)

        return App(ui = app_ui, server = server, static_assets = self.ROOT / "www")



    def run(self):
        # import threading
        import socket

        def get_free_port():
            s = socket.socket()
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            return port

        myapp = self.application()

        def _run():
            # This will call asyncio.run() inside the new thread (no conflict).
            myapp.run(
                host = "127.0.0.1",
                port = get_free_port(),
                log_level = "info",
                launch_browser = "viewer",
                dev_mode = True,
            )

        if "ipykernel" in sys.modules:
            t = threading.Thread(target=_run, daemon=True)
            print("Shiny viewer (running in background thread)")
            t.start()
        else:
            print("Shiny viewer (running in foreground thread)")
            _run()  # normal script behavior
