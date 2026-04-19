###########################
## Class: Module ##
###########################
## This class inherits from abc.ABC
## It provides:
##    Guide services via shepherd:
##       patch_ui() function which patches the shiny.ui.input_* & shiny.ui.output_*
##       _make_wrapper() function used by patch_ui
##       _ui_patched class variable prevent repeated calls
##       Create_run_tour() function to create and run a shepherd tour
##    Logging services:
##       Instance methods debug(), info(), warn(), error()
##    Namespace management:
##       ns() function
##       namespace attribute
##    Abstract Interfaces:
##       call_ui(): module's ui function
##       call_server(): Module's server function
##    Suspendable"
##       @Suspendable decorator to replace @reactive.calc, @reactive.event, @reactive.effect 
##       Instance level suspend() & resume() functions
##       Instance level list of suspendables/resumables
##    Output reactive like R's render_print style:
##       @capture_print decorator
##    Code recording:
##       @record_code decorator
##       Instance-level dict repository of function code blocks
##       Code key-retrieval mechanism
##       Long HTML listing of all code blocks
##    Reactive.calc rate limiting
##       @debounce decorator 
##       @throttle decorator
##    Create cards method that looks for files in "cards" and imports them and calls their instance() method
##    App method that creates the shiny app object
##    Run method that either runs the app in the browser or the card in the viewer
import shinywidgets as _sw
from abc import ABC, abstractmethod
from functools import wraps
from shiny import reactive, ui as _ui
from os import environ
from collections import namedtuple
from pathlib import Path
import logging
import inspect
import json
import functools
import io
import asyncio
import sys
import textwrap
import re
import time
import threading



class Module(ABC):
    """
    Base class relating to shiny modules that:
      - Maintains namespace,
      - Provides Guide services via shepherd (incl. patching input/output calls)
      - Add abstract interfaces (call_ui, call_server)
      - Decorates Reactive functions (record_code, capture_print, debounce, throttle, Suspendable)
      - Provides logging services
      - Loads from "/cards" folder
      - Creates the shiny App
      - Allows single files to be run in a viewer window
    """
    ROOT = Path(__file__).resolve().parent
    WWW = ROOT / "www"
    ModSession = None
    MaxInstances = 10  # arbitrary limit 
    Instances = {}  # class level dictionary of all instances keyed by their namespaces (including deleted ones with empty values)
    script_list = []
    css_list = []
    _ui_patched = False  # whether patching has been performed
    min_log_level = logging.DEBUG
    log = logging.getLogger("pythagoras.module")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | monitor | %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(h)
        log.propagate = False
        log.setLevel(min_log_level)


    # Initialiser
    def __init__(self, name, logger = None, *args, **kwargs): # will be inhereted by child classes
        super().__init__(*args, **kwargs)  # play nicely with multiple inheritance

        # scripts
        self.script_list.append(self.WWW / "console.js")
        self.script_list.append(self.WWW / "shepherd.js")
        self.script_list.append(self.WWW / "guide.js")
        self.script_list.append(self.WWW / "sortable.min.js")
        self.css_list.append(self.WWW / "shepherd.css")

        # reactives
        self._exports = reactive.Value()
        self._imports = reactive.Value()

        #namespace
        if name is None:
            raise ValueError("Name is required — stopping.")
        self.name = name
        ns = name
        if ns in self.Instances:
            for i in range(self.MaxInstances-1):
                ns_ = f"{name}_{i}"
                if ns_ not in self.Instances:
                    ns = ns_
                    break
            else:
                raise ValueError(f"Too many instances of module '{name}': exceeded maximum of {self.MaxInstances}")
        self.Instances[ns] = self
        self.namespace = ns

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
        base.propagate = False
        self.log = base  # or a LoggerAdapter if you want extra fields
        self.log.setLevel(Module.min_log_level)
        # suspendables
        self.suspendables = [] # An instance-level list of all suspendable reactives
        # Code recording
        self.code_registry = {} # An instnce-level Code-Registry 


    def reset(self):
        self.debug(f"Cleaning up namespace {self.namespace}")
        self.Instances[self.namespace] = None

    Packet = namedtuple("Packet", "data name")

    def guidedDiv(
        self,
        *children,
        id: str,
        class_ = None,
        guide = None,
        title = None,
        text: str = "",
        position: str = "bottom",
        priority: int = 0,
        **kwargs,
    ):
        """
        Create a div that can participate in the Shepherd guide.
        The visible div gets the namespaced id.
        A wrapper div gets the guide step id, so Shepherd has a stable target.
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

    # Logging
    # Convenience methods (don’t use deprecated .warn)
    def debug(self, msg, *args, **kw):     self.log.debug(msg, *args, **kw)
    def info(self, msg, *args, **kw):      self.log.info(msg, *args, **kw)
    def warning(self, msg, *args, **kw):   self.log.warning(msg, *args, **kw)   # not .warn
    def error(self, msg, *args, **kw):     self.log.error(msg, *args, **kw)
    def exception(self, msg, *args, **kw): self.log.exception(msg, *args, **kw)  # includes traceback

    # Namespace function (div ids need this as they are not namespaced by the decorator)
    def ns(self, id): # this is equivilent to what @module.ui does
        return f"{self.namespace}-{id}"


    #@classmethod
    def running_under_tests():
        return (
            "PYTEST_CURRENT_TEST" in environ or
            "pytest" in sys.modules or
            any("pytest" in arg for arg in sys.argv)
        )
    IsSolo = running_under_tests()  # running as a single card
        

    #@classmethod
    def running_directly(name):
        return name == "__main__"

    #@classmethod
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

    def _make_wrapper(self, func, kind = "input"):
        """
        Wrap a Shiny UI function to register a shepherd step
        guide_instance: the Guide instance whose _shepherd_steps we update
        func: the original ui function
        """
        @functools.wraps(func)
        def wrapped(id, *, guide : Module = None, label = None, title = None, text = None, position = "bottom", priority = 0, **kwargs):
            # Call original Shiny UI function
            sig = inspect.signature(func)
            if "label" in sig.parameters:
                widget = func(id = id, label = label, **kwargs)
            else:
                widget = func(id = id, **kwargs)
            if  guide is None:
                # Then do not guide this input/output ui element.
                return widget
            # Namespace id
            nid = guide.ns(id)
            wid = f"{nid}_wrapper"
            # Register shepherd step in this instance
            # self.debug(f"Adding {wid} to {guide.name}")
            guide._shepherd_steps[wid] = {
                "title": title or label,
                "text": text or "",
                "position": position,
                "priority": priority
            }
            # Wrap in a div so Shepherd can safely attach
            return _ui.div(widget, id = wid, class_ = "html-fill-container html-fill-item")
        return wrapped

    _ui_patch_lock = threading.Lock()


    def _patch_module(self, module) -> None:
        """Patch a single module once; originals stored on module._original_funcs."""
        if module is None:
            return
        if getattr(module, "_original_funcs", None):
            return
        with self._ui_patch_lock:
            if getattr(module, "_original_funcs", None):
                return
            module._original_funcs = {}
            for name, obj in vars(module).items():
                if not callable(obj):
                    continue
                if not name.startswith(("input_", "output_", "download_")):
                    continue
                module._original_funcs[name] = obj
                setattr(module, name, self._make_wrapper(obj))

    def capture_print(self, func):
        """
        Decorator to capture all print() output from a function
        and return it as a string.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a buffer to capture stdout
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                # Call the decorated function
                result = func(*args, **kwargs)
            finally:
                # Restore original stdout
                sys.stdout = old_stdout

            # Get all printed text
            printed_text = buf.getvalue()

            # If the function also returned a value, append it
            if result is not None:
                printed_text += str(result)

            return printed_text
        return wrapper


    def suspendable(self, *, triggers = None, suspended = True, default = None):
        # Universal Suspendable decorator
        #   Suspendable decorator to replace @reactive.calc, @reactive.event, @reactive.effect
        #   Auto-detects the nature of the wrapped function and applies the appropriate reactive decorators
        #   Can be suspended/resumed with self.suspend() self.resume()
        #   It handles async functions
        # Args:
        #   triggers: Optional reactive inputs (for event observers).
        #   suspended: Start suspended (default True).
        #   default: Value returned if a suspended calc is called.
        def decorator(func):
            enabled = reactive.Value(not suspended)

            # Check if it's a calc (returns a value)
            is_calc = inspect.signature(func).return_annotation != inspect._empty
            is_async = asyncio.iscoroutinefunction(func)

            if is_calc:
                # Handle calc: async or sync
                if is_async:
                    @reactive.calc
                    @wraps(func)
                    async def wrapped():
                        if not enabled():
                            return default
                        return await func()
                else:
                    @reactive.calc
                    @wraps(func)
                    def wrapped():
                        if not enabled():
                            return default
                        return func()
            else:
                # Handle effect/event
                if triggers:
                    if is_async:
                        @reactive.effect
                        @reactive.event(*triggers)
                        @wraps(func)
                        async def wrapped():
                            if not enabled():
                                return
                            await func()
                    else:
                        @reactive.effect
                        @reactive.event(*triggers)
                        @wraps(func)
                        def wrapped():
                            if not enabled():
                                return
                            func()
                else:
                    if is_async:
                        @reactive.effect
                        @wraps(func)
                        async def wrapped():
                            if not enabled():
                                return
                            await func()
                    else:
                        @reactive.effect
                        @wraps(func)
                        def wrapped():
                            if not enabled():
                                return
                            func()
            # Control methods
            def suspend():
                enabled.set(False)
            def resume():
                enabled.set(True)
            wrapped.suspend = suspend
            wrapped.resume = resume
            self.suspendables.append(wrapped)
            return wrapped
        return decorator


    # Utility functions for mass control
    def suspend(self):
        for w in self.suspendables:
            w.suspend()
    def resume(self):
        for w in self.suspendables:
            w.resume()


    def record_code(self, func):
        """
        Decorator to record source code of a function and store 
        this on the instance.
        """
        try:
            # Grab the source, dedent so it runs cleanly
            source = textwrap.dedent(inspect.getsource(func))
        except OSError:
            source = "<source not available>"
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.code_registry[func.__name__] = source
            return func(*args, **kwargs)
        return wrapper
    
    def retrieve_code(self, func_name):
        # Retrieve recorded code
        if func_name not in self.code_registry:
            raise ValueError(f"{func_name} not recorded")
        return self.code_registry[func_name]

    # be careful of html in comments as this will be enacted
    def code_text(self):
        lines = []
        for name, code in self.code_registry.items():
            code = re.sub(r"@render\.(\w+)", r"# returns a \1", code)
            code = re.sub("@output", "", code)
            code = re.sub("@this.record_code\s*", "", code)
            code = re.sub("@this.capture.print\s*", "", code)
            code = code.replace("<", "&lt;")
            code = code.replace(">", "&gt;")
            lines.append(f'<h3># {name}</h3><pre>{code}</pre>')
        return _ui.HTML("<hr>".join(lines))

    def debounce(self, delay_secs: int = 1):
        def wrapper(f):
            when = reactive.Value(None)
            trigger = reactive.Value(0)

            @reactive.calc
            def cached():
                """
                Just in case f isn't a reactive calc already, wrap it in one. This ensures
                that f() won't execute any more than it needs to.
                """
                return f()

            @reactive.effect(priority=102)
            def primer():
                """
                Whenever cached() is invalidated, set a new deadline for when to let
                downstream know--unless cached() invalidates again
                """
                try:
                    cached()
                except Exception:
                    ...
                finally:
                    when.set(time.monotonic() + delay_secs)

            @reactive.effect(priority=101)
            def timer():
                """
                Watches changes to the deadline and triggers downstream if it's expired; if
                not, use invalidate_later to wait the necessary time and then try again.
                """
                deadline = when()
                if deadline is None:
                    return
                time_left = deadline - time.monotonic()
                if time_left <= 0:
                    # The timer expired
                    with reactive.isolate():
                        when.set(None)
                        trigger.set(trigger() + 1)
                else:
                    reactive.invalidate_later(time_left)

            @reactive.calc
            @reactive.event(trigger, ignore_none=False)
            @functools.wraps(f)
            def debounced():
                return cached()

            return debounced

        return wrapper


    def throttle(self, delay_secs: int = 1):
        def wrapper(f):
            last_signaled = reactive.Value(None)
            last_triggered = reactive.Value(None)
            trigger = reactive.Value(0)

            @reactive.calc
            def cached():
                return f()

            @reactive.effect(priority=102)
            def primer():
                try:
                    cached()
                except Exception:
                    ...
                finally:
                    last_signaled.set(time.monotonic())

            @reactive.effect(priority=101)
            def timer():
                if last_triggered() is not None and last_signaled() < last_triggered():
                    return

                now = time.monotonic()
                if last_triggered() is None or (now - last_triggered()) >= delay_secs:
                    last_triggered.set(now)
                    with reactive.isolate():
                        trigger.set(trigger() + 1)
                else:
                    reactive.invalidate_later(delay_secs - (now - last_triggered()))

            @reactive.calc
            @reactive.event(trigger, ignore_none=False)
            @functools.wraps(f)
            def throttled():
                return cached()

            return throttled

        return wrapper

    # Abstract methods
    @abstractmethod
    def call_ui(self):
        pass

    @abstractmethod
    def call_server(input, output, session):
        pass

    #@classmethod
    def run(modules):
        # import threading
        import socket
        Module.IsSolo = True

        def get_free_port():
            s = socket.socket()
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            return port

        def launcher():
            card = list(modules.values())[0]
            print(card.name)
            if card.name == "roleAssign":
                return True
            else:
                return "viewer"

        if isinstance(modules, Module):
            modules = {modules.namespace : modules}          
