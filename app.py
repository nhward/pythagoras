###########################
## application                   ##
###########################

## This the app for a shiny application call Pythagoras
## It provides:
##    Creating the card instances
##    Managing the cascade of results along the cards 
##    Invoking the shiny app either in Positron or in Python via the last lines 

import pathlib
import sys
import importlib
from shiny import reactive, App, ui, req
from faicons import icon_svg as icon
from pathlib import Path
from module import Module

ROOT = Path(__file__).resolve().parent

def create_cards(folder = ""):
    """Import all Python modules inside the given folder (default 'cards')
    and create an instance of each."""
    # Ensure the folder exists
    cards_path = pathlib.Path(folder)
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
        module_name = f"{folder}.{py_file.stem}"   # e.g. cards.DataImport
        try:
            module = importlib.import_module(module_name)
            card = module.instance()
            imported_cards[card.namespace] = card
            card.info(msg="✅ Instantiated")
        except Exception as e:
            Module.log.error(msg = f"⚠️ Failed to instantiate card {module_name}: {repr(e)}")

    return imported_cards

#@classmethod
def app(modules):
    if modules is None:
        return
    
    # main ui object for the app
    app_ui = ui.page_fillable(
        ui.head_content(
            ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
            [ui.include_js(script, method = "inline") for script in set(Module.script_list)],  # iterate through unique js scripts
            [ui.include_css(css, method = "inline") for css in set(Module.css_list)],  # iterate through unique CSS documents
        ),
        ui.busy_indicators.options(spinner_type = "bars2"),
        ui.busy_indicators.use(),
        ui.page_navbar(
            ui.nav_panel(
                "Data Prep",
                ui.div(
                    [module.call_ui() for module in modules.values()],  # iterate through any number of forms - creates and runs a ui module for each
                    id = "cards-container", # Container for cards
                    class_ = "cards-grid"
                ),
                value = "Data_prep"
            ),
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
            """
            cards = {}
            cards_dir = ROOT / "cards"
            for path in sorted(cards_dir.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                cards[path.stem] = path
            return cards
        
        def card_picker_modal(cards: dict[str, Path]):
            """
            Build the modal UI.
            """
            return ui.modal(
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
            choice = input.CardPicker_selected()
            if not choice:
                return
            if choice not in available_cards():
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
            Module.log.info("Full-screen app requested")
            await session.send_custom_message("fullscreen_app", None)
        

        @reactive.effect
        @reactive.event(input.Quit)
        async def Quit():
            Module.log.info("Quit app requested")
            await session.send_custom_message("quit_app", None)
            session.close()  # in case the window close is ignored
        

        # redirect browser console to the python console
        @reactive.effect
        @reactive.event(input.Console_log)
        def _():
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

        @reactive.calc
        def CurrentTabCardOrder(): #given in namespaces
            req(input.CardOrder())
            order = input.CardOrder()
            order = [s.removesuffix("-Card") for s in order]
            return order


        @reactive.effect()
        def cascade():
            # if Module.IsSolo:
            #     Module.log.warning("Running in standalone mode")
            # req(not Module.IsSolo)
            Module.log.debug(msg = "Cascade called")
            order = CurrentTabCardOrder()
            source  = None
            for cardns in order:
                destination = Module.Instances[cardns]
                if destination is None:
                    continue
                if source is None:
                    destination._imports["name"].unset() 
                    destination._imports["data"].unset()
                    destination.debug(msg="No data source currently attached")
                else:
                    if source._exports["name"].is_set():
                        destination._imports["name"].set(source._exports["name"].get())
                        source.debug(msg=f"Output data '{source._exports['name'].get()}' -> card {destination.name}")
                    else:
                        destination._imports["name"].unset()
                    if source._exports["data"].is_set():
                        destination._imports["data"].set(source._exports["data"].get())
                    else:
                        destination._imports["data"].unset()
                source = destination
        
        # Load and initialise each module file (card)
        for card in Module.Instances.values():
            if card is None:
                continue
            card.call_server(input, output, session)
            card.resume()
            async def after_flush(card = card):
                await session.send_custom_message("init_card", {"id": card.ns("Card")})
            session.on_flushed(after_flush, once=True)


    www = pathlib.Path(__file__).resolve().parent / "www"  # because of the cards folder we need to be explicit about www's location
    return App(ui = app_ui, server = server, static_assets = www)


# Load the cards in the "cards" folder
cards = create_cards(folder = "cards")

# TODO: Manage the order of the cards

app = app(modules = cards)

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 3277, log_level = "info", dev_mode = False)