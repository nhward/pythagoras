###########################
## application                   ##
###########################

## This the app for a shiny application call Pythagoras
## It provides:
##    Creating the card instances
##    Managing the cascade of results along the cards 
##    Invoking the shiny app either in Positron or in Python via the last lines 

import threading
import sys
import importlib
from pathlib import Path
from module import Module

ROOT = Path(__file__).resolve().parent

def create_cards(folder: Path | str):
    """Import all Python modules inside the given folder (default 'cards')
    and create an instance of each."""
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
        module_name = f"{folder}.{py_file.stem}"   # e.g. cards.DataImport
        try:
            module = importlib.import_module(module_name)
            card = module.instance()
            imported_cards[card.namespace] = card
            card.info(msg="✅ Instantiated")
        except Exception as e:
            Module.log.error(msg = f"⚠️ Failed to instantiate card {module_name}: {repr(e)}")

    return imported_cards

# Load the cards in the "cards" folder
cards = create_cards(folder = "cards")

# TODO: Manage the order of the cards

app = Module.app(modules = cards)

if __name__ == "__main__":
    def _run():
        # This will call asyncio.run() inside the new thread (no conflict).
        app.run(
            host = "127.0.0.1",
            port = 3277,
            log_level = "info",
            launch_browser = True,
            dev_mode = False,
        )

    if "ipykernel" in sys.modules:
        t = threading.Thread(target=_run, daemon=True)
        print("Shiny (running in background thread)")
        t.start()
    else:
        print("Shiny (running in foreground thread)")
        _run()  # normal script behavior

