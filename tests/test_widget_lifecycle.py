from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import widget_lifecycle

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('preserve_traces', [False, True])
def test_placeholder_layout_is_removed_when_results_arrive(preserve_traces):
    import plotly.graph_objects as go
    from card import Card
    widget = Card.empty_figure('Waiting for the first complete evaluation.')
    assert widget.layout.annotations and widget.layout.images
    results = go.Figure(go.Scatter(x=[1], y=[4], legendgroup='method'))
    results.update_xaxes(title_text='Trial', range=[.5, 10.5])
    widget_lifecycle.update_figure(widget, results, {}, preserve_traces=preserve_traces)
    assert not widget.layout.annotations and not widget.layout.images
    assert widget.layout.xaxis.visible is None
    assert widget.layout.xaxis.fixedrange is None
    assert widget.layout.yaxis.visible is None
    assert widget.layout.yaxis.fixedrange is None
    widget_lifecycle.update_figure(widget, Card.empty_figure('Waiting again'), {}, preserve_traces=preserve_traces)
    assert widget.layout.annotations[0].text == 'Waiting again'
    assert len(widget.layout.images) == 1


def test_history_updates_preserve_existing_traces():
    import plotly.graph_objects as go
    widget = go.Figure(go.Scatter(x=[1], y=[4], legendgroup='method', name='Method'))
    trace = widget.data[0]
    layout = widget.layout
    config = {'displayModeBar': False, 'responsive': True}
    figure = go.Figure(go.Scatter(x=[1, 2], y=[4, 3], legendgroup='method', name='Method'))
    widget_lifecycle.update_figure(widget, figure, config, preserve_traces=True)
    assert widget.data[0] is trace
    assert widget.layout is layout
    assert widget.layout.autosize is True
    assert list(trace.x) == [1, 2] and list(trace.y) == [4, 3]


def test_replaced_widgets_close_after_flush_and_session_cleanup_is_idempotent(monkeypatch):
    monkeypatch.setattr(widget_lifecycle, 'session_context', lambda session: nullcontext())
    callbacks, ended = [], []
    session = SimpleNamespace(on_ended=ended.append,
                              on_flushed=lambda callback, once: callbacks.append(callback))

    class Widget:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    slot = widget_lifecycle.WidgetSlot(session)
    first, second, third = Widget(), Widget(), Widget()
    assert slot.replace(first) is first
    slot.replace(second)
    assert first.closes == second.closes == 0
    callbacks.pop(0)()
    assert first.closes == 1 and second.closes == 0
    slot.replace(third)
    ended[0]()
    assert second.closes == third.closes == 1
    callbacks.pop(0)()
    slot.close()
    assert first.closes == second.closes == third.closes == 1
