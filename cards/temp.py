
from module import Module
from shiny import ui

class MyCard(Module):
    def call_ui(self):
        return ui.div("card_ui")

    def call_server(self, input, output, session):
        pass

def instance():
    return MyCard("mycard")
