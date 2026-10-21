from shiny.types import SilentException


class CardPositionError(SilentException, RuntimeError):
    """An incompatible card position; silent in effects, explained by Card."""
