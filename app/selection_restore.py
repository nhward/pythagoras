"""Restore dynamic selections while upstream column choices are still settling."""


class SelectionRestore:
    """Keep unavailable saved values pending, unless the user changes selection.

    Values and defaults are lists, including for single-selection widgets.
    Call once per choices update, with the current input read in isolation.
    """

    def __init__(self, saved=None):
        self.pending = None if saved is None else list(saved)
        self.initial = True
        self.expected = None

    def resolve(self, current, choices, default):
        current = list(current)
        choices = list(choices)
        if self.expected is not None and current != self.expected:
            self.pending = None
            self.initial = False
        if self.pending is not None:
            selected = [value for value in self.pending if value in choices]
            if all(value in choices for value in self.pending):
                self.pending = None
                self.initial = False
        elif self.initial and choices:
            selected = list(default)
            self.initial = False
        else:
            selected = [value for value in current if value in choices]
        self.expected = selected
        return selected
