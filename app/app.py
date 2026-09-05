###########################
## application           ##
###########################

## This the app for a shiny application called Pythagoras
## It provides:
##    Creating sections for cards (i.e. a grouping structure)
##    Dynamically creating the card instances for a current section
##    Reactively connecting card outputs to subsequent card inputs
##    Invoking the shiny app either in Positron or in Python via the last lines 


import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#TODO: provide a guide button for the sections and buttons
#TODO: provide a info button for the whole app               

# Ensure local modules and packages are resolved from the app directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cards  # noqa: F401
from faicons import icon_svg as icon
from module import Module
from shiny import App, reactive, req, ui

log = logging.getLogger("pythagoras")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(h)
log.propagate = False
log.setLevel(logging.DEBUG)


jslog = logging.getLogger("<Javascript>")
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
))
jslog.addHandler(h)
jslog.propagate = False
jslog.setLevel(logging.DEBUG)
if Module.log_handler not in jslog.handlers:
    jslog.addHandler(Module.log_handler)

config = Module.config


@dataclass
class CardNode:
    card: Module
    upstream: reactive.Value
    output: Callable[[], object]


def wire_card_nodes(
    order: tuple[str, ...],
    nodes: dict[str, CardNode],
) -> None:
    """Connect card outputs to subsequent inputs without reading any data."""
    source = None
    for namespace in order:
        node = nodes.get(namespace)
        if node is None:
            continue
        node.card.resume()
        node.upstream.set(None if source is None else source.output)
        source = node


def welcome():
    """ 
    Read html from a file 
    """
    html_file = Path("www/markdown/welcome.html")
    if html_file.exists():
        try:
            text = html_file.read_text(encoding="utf-8")
            return ui.HTML(text)
        except Exception:  # noqa: BLE001
            log.error(f"Error reading {html_file} file")
            return ""
    else:
        log.error(f"File {html_file} not found")
        return ""


def sections() -> list[str]:
    """
    return a list of section names
    """
    return [section["section"] for section in config["layout"]
]


def create_sections():
    group_style = config.get("settings", {}).get("section_style")
    panels = []
    if config.get("settings", {}).get("show_start"):
        panels.append(
            ui.nav_panel(
                "Start",
                ui.div(
                    welcome(),
                    ui.input_action_button(
                        id = "GuideButton", 
                        label= "Guide me", 
                        icon = icon("eye", title = "Take a guided tour of this card", a11y = "sem"), 
                        class_ = "btn rounded-pill hover-btn btn-sm guide-btn",
                        style = "border: 0px; box-shadow: none;",
                        aria_label = "Take a guided tour of this card"
                    ),
                    id="Start-cards-container",
                    class_=""
                ),
            )
    )
    if group_style == "tab":
        for name in sections():
            _name = Module.section_normalise(name)
            panels.append(
                ui.nav_panel(
                    name,
                    ui.div(
                        id=f"{_name}-cards-container",
                        class_="cards-grid"
                    ),
                    value=name
                )
            )
    elif group_style == "accordion":
        panels.append(
            ui.nav_panel(
                "",
                ui.accordion(
                    *[
                        ui.accordion_panel(
                            name,
                            ui.div(
                                id=f"{Module.section_normalise(name)}-cards-container",
                                class_="cards-grid",
                            ),
                            value=name
                        )
                        for name in sections()
                    ],
                    multiple=False,
                    id="Accordion",
                ),
            )
        )
    return panels

def application():
    """
    Create a shiny app.
    This involves creating UI (app_ui) and SERVER (server) functions and
    passing these to shiny.app.
    """
    # main ui object for the app
    app_ui = ui.page_fillable(
        ui.head_content(
            ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
            [ui.include_js(script, method="inline") for script in dict.fromkeys(Module.script_list)], # iterate through unique js scripts
            [ui.include_css(css, method="inline") for css in dict.fromkeys(Module.css_list)], # iterate through unique CSS documents
            [ui.tags.script(type = "module", src=script) for script in dict.fromkeys(Module.mjs_list)] # iterate through unique mjs modules
        ),
        ui.busy_indicators.options(spinner_type = "bars2"),
        ui.busy_indicators.use(),
        ui.page_navbar(
            *create_sections(),  # << This is the important call here
            ui.nav_spacer(),
            ui.nav_control(
                ui.input_action_button(
                    id = "AddCard",  
                    label= None, 
                    icon = icon("plus", title = "Add a card", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),
            ui.nav_control(
                ui.input_action_button(
                    id = "FullScreen",  
                    label= None, 
                    icon = icon("expand", title = "Toggle full screen", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),
            ui.nav_control(
                ui.input_action_button(
                    id = "Quit",  
                    label = None, 
                    icon = icon("stop", title = "Quit session", a11y = "sem"),
                    class_ = "btn rounded-pill btn-sm fa-xl",
                    style = "border: 0px; box-shadow: none; display: block;"
                )
            ),

            id = "Navbar",
            title = ui.tooltip(
                ui.TagList(ui.tags.img(src="favicon.ico", style="height:2em; margin-right:0.5em;"), ui.span("Pythagoras", class_ = "text-primary")),
                '"All is number."',
                placement = "bottom"
            )
        )
    )

    # main server function for the app
    def server(input, output, session):
        Module.ModSession = session

        card_nodes: dict[str, CardNode] = {}
        SectionsVisited = reactive.value([])
        TopologyVersion = reactive.value(0)

        def unregister_card(namespace: str):
            card_nodes.pop(namespace, None)
            with reactive.isolate():
                version = TopologyVersion.get()
            TopologyVersion.set(version + 1)

        @reactive.effect
        async def startup():
            if config.get("settings", {}).get("show_start"):
                await session.send_custom_message("animate", {"id" : session.ns("GuideButton"), "animation" : "bounce", "delay" : 500})

        @reactive.calc
        def currentSection():
            """
            Uses the group_style to determine how to assess the current group name.
            The name is tested in case it is not a valid group name (i.e. a nav_bar button)
            """
            section_style = config.get("settings", {}).get("section_style")
            if section_style == "tab":
                current = input.Navbar()
            else:
                current = input.Accordion()
                req(current)
                req(len(current) == 1)
                current = current[0]
            valid =  [section.get("section") for section in config.get("layout", [])]
            #Check that the current tab-item is not a button etc
            req(current in valid)
            log.debug(f"🔀 Section switched to {current!r} using {section_style} style")
            return current

        def create_card(name: str):
            module_name = f"cards.{name}"
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, "instance"):
                    raise AttributeError(f"{module_name} does not define instance()")
                module = module.instance()
                module.log.info(msg=f"✅ Card instantiated ({module.namespace})")
                return module
            except Exception:
                log.exception(f"⚠️ Failed to instantiate card {module_name}")
                return None

        @reactive.effect
        def create_section_cards():
            current = currentSection()
            if current not in SectionsVisited.get():
                model_group = next((group for group in config.get("layout", []) if group.get("section") == current), None)
                req(model_group is not None)
                for card in model_group["cards"]:
                    instance = create_card(card["module"])
                    if instance is None:
                        continue
                    ui.insert_ui(ui = instance.call_ui(), selector = f"#{Module.section_normalise(current)}-cards-container", where = "beforeEnd")
                    upstream = reactive.Value(None)
                    card_output = instance.call_server(
                        input,
                        output,
                        session,
                        upstream=upstream,
                        on_remove=unregister_card,
                    )
                    card_nodes[instance.namespace] = CardNode(
                        card=instance,
                        upstream=upstream,
                        output=card_output,
                    )
                    instance.resume()
                    card_id = instance.ns("Card")
                    instance.section = Module.section_normalise(current)
                    async def after_flush(card_id=card_id):
                        await session.send_custom_message("init_card", {"id": card_id})
                    session.on_flushed(after_flush, once=True)
                async def after_flush2(current=current):
                    _name = Module.section_normalise(current)
                    container = f"{_name}-cards-container"
                    imp_id = f"{_name}_CardOrder"
                    await session.send_custom_message("MakeSortable", {"id": container, "input_id": imp_id})
                session.on_flushed(after_flush2, once=True)
                visited = [*SectionsVisited.get(), current]
                SectionsVisited.set([
                    item for item in sections() if item in visited
                ])


        @reactive.calc
        def available_cards():
            """
            Return {display_name: file_path} for all card files in cards_dir.
            Assumes each card file defines instance().
            This is not filesystem-reactive but instead uses invalidation to pick up file changes (eventually)
            """
            cardDict = {}
            cards_dir = Module.ROOT / "cards"
            for path in sorted(cards_dir.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                cardDict[path.stem] = path
            # re-evaluate every hour
            reactive.invalidate_later(3600)
            return cardDict
        
        def card_picker_modal(cardDict: dict[str, Path]):
            """
            Build the modal UI to pick a new card from the available cards.
            """
            return ui.modal(
                #TODO: make the choices more descriptive - currently just the dict key is used
                ui.input_select(
                    id = "CardPicker_selected",
                    label = "Choose a card to insert",
                    choices = list(cardDict.keys()),
                    selected = next(iter(cardDict.keys())) if cardDict else None,
                ),
                title="Add card",
                footer=ui.div(
                    ui.input_action_button(
                        id = "CardPicker_cancel", 
                        label = "Cancel", 
                        class_ = "btn btn-secondary"),
                    ui.input_action_button(
                        id = "CardPicker_ok", 
                        label = "Add card", 
                        class_ = "btn btn-primary ms-2"),
                    class_ = "d-flex justify-content-end"
                ),
                easy_close=True
            )

        @reactive.effect
        @reactive.event(input.AddCard)
        async def showPicker():
            """
            Respond to the "new" button click by showing the model-dialogue of available cards.
            """
            ui.modal_show(
                card_picker_modal(available_cards())
            )


        @reactive.effect
        @reactive.event(input.CardPicker_cancel)
        async def _cancel_picker():
            ui.modal_remove()

        @reactive.effect
        @reactive.event(input.CardPicker_ok)
        def _confirm_picker():
            """
            React to the choice of a new card to add to the current section.
            """
            ui.modal_remove()
            name = input.CardPicker_selected()
            if not name or name not in available_cards():
                return
            current = currentSection()
            _name = Module.section_normalise(current)
            instance = create_card(name)
            ui.insert_ui(ui = instance.call_ui(), selector = f"#{_name}-cards-container", where = "beforeEnd")
            upstream = reactive.Value(None)
            card_output = instance.call_server(
                input,
                output,
                session,
                upstream=upstream,
                on_remove=unregister_card,
            )
            card_nodes[instance.namespace] = CardNode(
                card=instance,
                upstream=upstream,
                output=card_output,
            )
            instance.section = _name
            instance.resume()
            card_id = instance.ns("Card")
            container = f"{_name}-cards-container"
            imp_id = f"{_name}_CardOrder"
            async def after_flush(card_id=card_id, container = container, container_id = imp_id):
                await session.send_custom_message("init_card", {"id": card_id})
                await session.send_custom_message("UpdateCardOrder", {"id": container, "input_id": container_id})
            session.on_flushed(after_flush, once=True)
        

        @reactive.effect
        @reactive.event(input.FullScreen)
        async def FullScreen():
            """
            This makes the browser go full screen.
            Because full-screen is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("🙏 Full-screen app requested")
            await session.send_custom_message("fullscreen_app", None)
        

        @reactive.effect
        @reactive.event(input.Quit)
        async def Quit():
            """
            This shuts the browser session down - just like a conventional application.
            Because closing tabs is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("🙏 Quit app requested")
            await session.send_custom_message("quit_app", None)
            await session.close()  # in case the window close is ignored
        

        # redirect browser console to the python console
        @reactive.effect
        @reactive.event(input.Console_log)
        def Redirect():
            """
            Java-script console messages are redirected to the python log.
            """
            message = input.Console_log()
            level = message['level'].upper()
            if level =="ERROR":
                jslog.error(msg = f"☢️ {message['text']}")
            elif level == "INFO":
                jslog.info(msg = f"ℹ️ {message['text']}")
            elif level == "WARNING":
                jslog.warning(msg = f"⚠️ {message['text']}")
            else:
                jslog.debug(msg = f"🪲 {message['text']}")


        @reactive.calc
        def FlowOrder() -> tuple[str, ...]:
            TopologyVersion.get()
            visited = set(SectionsVisited.get())
            order: list[str] = []
            for section in sections():
                if section not in visited:
                    continue
                input_id = f"{Module.section_normalise(section)}_CardOrder"
                section_order = req(input[input_id]())
                order.extend(
                    value.removesuffix("-Card") for value in section_order
                )
            return tuple(order)

        @reactive.effect
        @reactive.event(FlowOrder)
        def wire_flow():
            log.debug(msg="𑙬 Card flow topology changed")
            wire_card_nodes(FlowOrder(), card_nodes)

    return App(
        ui = app_ui, 
        server = server, 
        static_assets = {
            "/html": Module.ROOT / "www" / "markdown",
            "/": Module.ROOT / "www"
        }
        )


app = application()  # This MUST be called "app" for shiny-mode of IDE integration

def main():
    app.run(
        host = "127.0.0.1",
        port = 3277,
        log_level = "info",
        launch_browser = True,
        dev_mode = False,
    )


if __name__ == "__main__":
    """
    This avoids IDE thread conflicts when calling as a python file by
    choosing between foreground and background threads.
    """
    if "ipykernel" in sys.modules:
        t = threading.Thread(target=main, daemon=True)
        log.info("🔙 Shiny (running in background thread)")
        t.start()
    else:
        log.info("➬ Shiny (running in foreground thread)")
        main()  # normal script behavior
else:
    app  # noqa: B018
