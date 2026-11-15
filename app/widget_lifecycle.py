"""Release replaced chart widgets, including inside Shiny module sessions."""
from shiny.session import session_context


def _removed_layout_properties(previous, current, prefix=''):
    removed = {}
    for key, value in previous.items():
        path = f'{prefix}.{key}' if prefix else key
        if key not in current:
            removed[path] = None
        elif isinstance(value, dict) and isinstance(current[key], dict):
            removed.update(_removed_layout_properties(value, current[key], path))
    return removed


def update_figure(widget, figure, config, *, preserve_traces=False):
    """Update a Plotly model in place instead of importing another widget view."""
    config_changed = widget._config != config
    widget._config = config
    if config.get('responsive') and figure.layout.width is None and figure.layout.height is None:
        figure.update_layout(autosize=True)
    with widget.batch_update():
        if preserve_traces:
            wanted = {trace.legendgroup or trace.name: trace for trace in figure.data}
            kept = tuple(trace for trace in widget.data if (trace.legendgroup or trace.name) in wanted)
            if len(kept) != len(widget.data):
                widget.data = kept
            existing = {trace.legendgroup or trace.name: trace for trace in widget.data}
            for key, trace in wanted.items():
                if key in existing:
                    values = trace.to_plotly_json()
                    values.pop('type', None)
                    existing[key].update(values, overwrite=True)
                else:
                    widget.add_trace(trace)
        else:
            widget.data = ()
            widget.add_traces(figure.data)
        if widget.layout.to_json() != figure.layout.to_json():
            # Plotly's layout setter merges properties. Explicitly clear those
            # omitted by the next figure, including placeholder images/text and
            # hidden/fixed axes, rather than leaving them on the browser chart.
            removed = _removed_layout_properties(widget.layout.to_plotly_json(), figure.layout.to_plotly_json())
            if removed:
                widget.plotly_relayout(removed)
                for name in ('annotations', 'images', 'shapes', 'selections', 'sliders', 'updatemenus'):
                    if name in removed:
                        widget.layout[name] = ()
            # Keep the layout object: shinywidgets attaches its responsive-size
            # callbacks to it, and replacing it drops those callbacks.
            widget.update_layout(figure.layout, overwrite=True)
    return config_changed


class WidgetSlot:
    def __init__(self, session):
        self.session = session
        self.current = None
        self.pending = {}
        session.on_ended(self.close)

    def replace(self, widget):
        previous, self.current = self.current, widget
        if previous is not None and previous is not widget:
            self.pending[id(previous)] = previous

            def release():
                if self.pending.pop(id(previous), None) is previous:
                    with session_context(self.session):
                        previous.close()

            # Let the replacement reach the browser before closing its predecessor.
            self.session.on_flushed(release, once=True)
        return widget

    def close(self):
        widgets = list(self.pending.values()) + ([self.current] if self.current is not None else [])
        self.pending = {}
        self.current = None
        with session_context(self.session):
            for widget in widgets:
                widget.close()
