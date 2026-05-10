###########################
## application           ##
###########################

## This the app for a shiny application call Pythagoras
## It provides:
##    Creating the card instances
##    Managing the cascade of results along the cards 
##    Invoking the shiny app either in Positron or in Python via the last lines 


from card import Card
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

def create_cards(folder: Path | str):
    """
    Import all Python modules inside the given folder (default 'cards')
    and create an instance of each.
    """
    # Ensure the folder exists
    if isinstance(folder, str): 
        cards_path = Path(folder)
    else:
        cards_path = folder
    if not cards_path.is_dir():
        raise FileNotFoundError(f"Folder '{folder}' not found.")
    # If the folder isn’t already a package, make sure it is on sys.path
    if str(cards_path.parent) not in sys.path:
        sys.path.append(str(cards_path.parent))
    imported_cards = {}
    # Iterate through all .py files, skipping __init__.py
    for py_file in cards_path.glob(pattern = "*.py"):
        if py_file.name == "__init__.py":
            continue
        package_name = cards_path.name
        module_name = f"{package_name}.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "instance"):
                raise AttributeError(f"{module_name} does not define instance()")
            card = module.instance()
            imported_cards[card.namespace] = card
            card.log.info(msg="✅ Instantiated")
        except Exception:
            log.exception(f"⚠️ Failed to instantiate card {module_name}")
    return imported_cards


def card_control(cards: dict[str, Card]):
    if True: #TODO: Make this a system setting
        return ui.nav_panel(
            "Data Prep",
            ui.div(
                [module.call_ui() for module in cards.values()],  # iterate through any number of cards - creates and runs a ui module for each
                id = "cards-container", # Container for cards
                class_ = "cards-grid"
            ),
            value = "Data_prep"
        )
    else:
        return ui.accordion(
            ui.accordion_panel(
                "Data prep",
                ui.div(
                    [module.call_ui() for module in cards.values()],  # iterate through any number of forms - creates and runs a ui module for each
                    id = "cards-container", # Container for cards
                    class_ = "cards-grid"
                ),
                id = "Data_prep",
            ),
            id = "Accordion",
            open = "Data_prep"
        )


def application(cards: dict[str, Card]):
    """
    Create a shiny app.
    This involves creating UI (app_ui) and SERVER (server) functions and
    passing these to shiny.app.
    """
    if cards is None:
        return
    
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
            card_control(cards),
            ui.nav_spacer(),
            ui.nav_control(
                ui.tooltip(
                    ui.input_action_button(
                        id = "AddCard",  
                        label= None, 
                        icon = icon("plus", title = "Add a card", a11y = "sem"),
                        class_ = "btn rounded-pill btn-sm fa-xl",
                        style = "border: 0px; box-shadow: none; display: block;"
                    ),
                    "Add a card",
                    placement = "bottom"
                )
            ),
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
            This is NOT currently filesystem-reactive
            """
            cards = {}
            cards_dir = Module.ROOT / "cards"
            for path in sorted(cards_dir.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                cards[path.stem] = path
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
        async def AddCard():
            """
            Respond to the "new" button click by showing the model-dialogue of available cards.
            """
            ui.modal_show(
                card_picker_modal(available_cards())
            )

        def load_fresh_card_instance(card_path: Path):
            module_name = f"cards.{card_path.stem}"
            module = importlib.import_module(module_name)
            if not hasattr(module, "instance"):
                raise AttributeError(f"{card_path.name} does not define instance()")
            return module.instance()

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
            #TODO: make this code section aware
            choice = input.CardPicker_selected()
            if not choice or choice not in available_cards():
                return
            file = available_cards()[choice]
            fresh_instance = load_fresh_card_instance(file)
            ui.modal_remove()
            ui.insert_ui(ui = fresh_instance.call_ui(), selector = "#cards-container", where = "beforeEnd")
            fresh_instance.call_server(input, output, session)
            fresh_instance.resume()
            async def after_flush():
                await session.send_custom_message("init_card", {"id": fresh_instance.ns("Card")})
                await session.send_custom_message("UpdateCardOrder", None)
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
            order = req(input.CardOrder())
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


        async def after_flush():
            """
            Ensure that input.CardOrder() is created and populated with the actual card sequence.
            Do this when the reactivity has settled - to avoid race conditions.
            """
            await session.send_custom_message("UpdateCardOrder", None)
        session.on_flushed(after_flush, once=True)

    return App(ui = app_ui, server = server, static_assets = ROOT / "www")


# Load the cards in the "cards" folder
cards = create_cards(folder = ROOT / "cards")

# TODO: Manage the order of the cards

app = application(cards = cards)  # This MUST be called "app" for shiny-mode of IDE integration


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
