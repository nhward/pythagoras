###########################
## application           ##
###########################

## This the app for a shiny application call Pythagoras
## It provides:
##    Creating the card instances
##    Managing the cascade of results along the cards 
##    Invoking the shiny app either in Positron or in Python via the last lines 


from shiny import ui, reactive, App, req
import threading
import sys
import logging
import importlib
from pathlib import Path
from module import Module
from faicons import icon_svg as icon

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("pythagoras")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(h)
log.propagate = False
log.setLevel(logging.DEBUG)


sections_populated = []

config = Module.config


def sections() -> list[str]:
    """
    return a list of section names
    """
    return [section["section"] for section in config["layout"]
]


def paths(section_name:str) -> dict[Path]:
    """
    args:
        section_name: 
    returns a list of paths corresponding to the cards of the given section
    """
    cards = next(
        (
            section["cards"]
            for section in config["layout"]
            if section["section"] == section_name
        ),
        []
    )
    return {card['namespace'] : card['module'] for card in cards}


def create_sections():
    group_style = config.get("settings", {}).get("section_style")
    panels = []
    if group_style == "tab":
        for name in sections():
            _name = name.strip().replace(" ", "_")
            log.debug(f"Creating tab panel section {name!r}")
            panel = ui.nav_panel(
                name, 
                ui.div(
                    id = f"{_name}-cards-container", # Container for cards
                    class_ = "cards-grid"
                ),
                value = name
            )
            panels.append(panel)
    else:
        panels = ui.nav_panel(
            "",
            ui.accordion(
                *[
                    ui.accordion_panel(
                        name,
                        ui.div(
                            id=f"{name.strip().replace(' ', '_')}-cards-container",
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
            [ui.include_css(css, method="inline") for css in dict.fromkeys(Module.css_list)] # iterate through unique CSS documents
        ),
        ui.busy_indicators.options(spinner_type = "bars2"),
        ui.busy_indicators.use(),
        ui.page_navbar(
            create_sections(),  # << This is the important call here
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


        @reactive.calc
        def available_cards():
            """
            Return {display_name: file_path} for all card files in cards_dir.
            Assumes each card file defines instance().
            This is not filesystem-reactive but instead uses invalidation to pick up file changes (eventually)
            """
            cards = {}
            cards_dir = Module.ROOT / "cards"
            for path in sorted(cards_dir.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                cards[path.stem] = path
            # re-evaluate every hour
            reactive.invalidate_later(3600)
            return cards
        
        def card_picker_modal(cards: dict[str, Path]):
            """
            Build the modal UI to pick a new card from the available cards.
            """
            return ui.modal(
                #TODO: make the choices more descriptive - currently just the dict key is used
                ui.input_select(
                    id = "CardPicker_selected",
                    label = "Choose a card to insert",
                    choices = list(cards.keys()),
                    selected = list(cards.keys())[0] if cards else None,
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
            log.debug(f"Section switched to {current!r} using style {section_style}")
            return current

        def create_card(name: str):
            module_name = f"cards.{name}"
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, "instance"):
                    raise AttributeError(f"{module_name} does not define instance()")
                module = module.instance()
                module.log.info(msg="✅ Instantiated")
                return module
            except Exception:
                log.exception(f"⚠️ Failed to instantiate card {module_name}")
                return None


        @reactive.effect
        @reactive.event(currentSection)
        def create_section_cards():
            current = currentSection()
            if current not in sections_populated:
                model_group = next((group for group in config.get("layout", []) if group.get("section") == current), None)
                req(model_group is not None)
                for card in model_group["cards"]:
                    instance = create_card(card["module"])
                    ui.insert_ui(ui = instance.call_ui(), selector = f"#{current.strip().replace(' ', '_')}-cards-container", where = "beforeEnd")
                    instance.call_server(input, output, session)
                    instance.resume()
                    card_id = instance.ns("Card")
                    async def after_flush(card_id=card_id):
                        await session.send_custom_message("init_card", {"id": card_id})
                    session.on_flushed(after_flush, once=True)
                async def after_flush2(current=current):
                    _name = current.strip().replace(" ", "_")
                    container = f"{_name}-cards-container"
                    imp_id = f"{_name}_CardOrder"
                    await session.send_custom_message("MakeSortable", {"id": container, "input_id": imp_id})
                session.on_flushed(after_flush2, once=True)
                sections_populated.append(current)



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
            _name = current.strip().replace(" ", "_")
            instance = create_card(name)
            ui.insert_ui(ui = instance.call_ui(), selector = f"#{_name}-cards-container", where = "beforeEnd")
            instance.call_server(input, output, session)
            instance.resume()
            card_id = instance.ns("Card")
            container = f"{_name}-cards-container"
            imp_id = f"{_name}_CardOrder"
            async def after_flush(card_id=card_id):
                await session.send_custom_message("init_card", {"id": card_id})
                await session.send_custom_message("UpdateCardOrder", {"id": container, "input_id": imp_id})
            session.on_flushed(after_flush, once=True)
        

        @reactive.effect
        @reactive.event(input.FullScreen)
        async def FullScreen():
            """
            This makes the browser go full screen.
            Because full-screen is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("Full-screen app requested")
            await session.send_custom_message("fullscreen_app", None)
        

        @reactive.effect
        @reactive.event(input.Quit)
        async def Quit():
            """
            This shuts the browser session down - just like a conventional application.
            Because closing tabs is browser specific this may be unreliable.
            The implememtation is in pythagoras.js
            """
            log.info("Quit app requested")
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
                log.error(msg = f"<javascript> | {message['text']}")
            elif level == "INFO":
                log.info(msg = f"<javascript> | {message['text']}")
            elif level == "WARNING":
                log.warning(msg = f"<javascript> | {message['text']}")
            else:
                log.debug(msg = f"<javascript> | {message['text']}")

        @reactive.calc
        def CurrentTabCardOrder() -> list[str]: #given in namespaces
            section = req(currentSection())
            order = req(input[f"{section.strip().replace(' ', '_')}_CardOrder"]())
            order = [s.removesuffix("-Card") for s in order]
            return order

        @reactive.effect
        def cascade():
            order = req(CurrentTabCardOrder())
            log.debug(msg = "Card flow cascade invoked")
            source  = None
            for cardns in order:
                destination = Module.Instances.get(cardns)
                if destination is None:
                    continue
                if source is None:
                    destination._imports.unset()
                    destination.log.debug(msg="No data source currently attached")
                else:
                    if source._exports.is_set():
                        destination._imports.set(source._exports.get())
                    else:
                        destination._imports.unset()
                source = destination

        # Initialize each module/card's server logic
        for card in list(Module.Instances.values()):
            if card is None:
                continue
            card.call_server(input, output, session)
            card.resume()


    return App(ui = app_ui, server = server, static_assets = ROOT / "www")


# # Load the cards in the "cards" folder
# cards = create_cards(folder = ROOT / "cards")

# TODO: Manage the order of the cards

app = application()  # This MUST be called "app" for shiny-mode of IDE integration


def _run():
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
        t = threading.Thread(target=_run, daemon=True)
        log.info("Shiny (running in background thread)")
        t.start()
    else:
        log.info("Shiny (running in foreground thread)")
        _run()  # normal script behavior
else:
    app
