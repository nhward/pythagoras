###########################
## application                   ##
###########################

## This the app for a shiny application call Pythagoras
## It provides:
##    Creating the card instances
##    Managing the cascade of results along the cards 
##    Invoking the shiny app either in Positron or in Python via the last lines 

# import cards.Configuration
# import cards.DataImport
from module import Module

# Load the cards in the "cards" folder
cards = Module.create_cards(folder = "cards")

# TODO: Manage the order of the cards

app = Module.app(modules = cards)

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 3277, log_level = "info", dev_mode = False)