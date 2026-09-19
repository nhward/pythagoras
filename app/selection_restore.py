"""Restore dynamic selections while upstream column choices are still settling."""


class SelectionRestore:
    """Keep unavailable saved values pending, unless the user changes selection.

    Values and defaults are lists, including for single-selection widgets.
    Call once per choices update, with the current input read in isolation.
    Also call observe from an input effect to acknowledge browser updates.
    """

    def __init__(self, saved=None):
        self.pending = None if saved is None else list(saved)
        self.initial = True
        self.expected = None
        self.awaiting = None

    def observe(self, current):
        """Distinguish a browser acknowledgement from an actual user edit."""
        current = list(current)
        if current == self.expected:
            self.awaiting = None
        elif self.awaiting is not None and current == self.awaiting:
            return
        elif self.expected is not None:
            self.pending = None
            self.initial = False
            self.awaiting = None

    def resolve(self, current, choices, default):
        current = list(current)
        choices = list(choices)
        self.observe(current)
        if self.pending is not None:
            selected = [value for value in self.pending if value in choices]
            self.initial = False
        elif self.initial and choices:
            selected = list(default)
            self.initial = False
        else:
            selected = [value for value in current if value in choices]
        self.awaiting = current if self.pending is not None and current != selected else None
        self.expected = selected
        return selected
