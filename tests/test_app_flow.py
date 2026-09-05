from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shiny import reactive


path = Path(__file__).resolve().parents[1] / "app"
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from app import CardNode, wire_card_nodes


class FakeCard:
    def __init__(self):
        self.resume_count = 0

    def resume(self):
        self.resume_count += 1


def make_node(output):
    return CardNode(
        card=FakeCard(),
        upstream=reactive.Value(None),
        output=output,
    )


@pytest.mark.unit
def test_wiring_builds_a_lazy_reactive_chain_and_supports_reordering():
    root_value = reactive.Value(1)
    root = make_node(root_value)
    add = make_node(None)
    double = make_node(None)

    @reactive.calc
    def add_output():
        source = add.upstream()
        assert source is not None
        return source() + 10

    @reactive.calc
    def double_output():
        source = double.upstream()
        assert source is not None
        return source() * 2

    add.output = add_output
    double.output = double_output
    nodes = {"root": root, "add": add, "double": double}

    wire_card_nodes(("root", "add", "double"), nodes)
    with reactive.isolate():
        assert double.output() == 22
        assert add.upstream() is root.output
        assert double.upstream() is add.output

    wire_card_nodes(("root", "double", "add"), nodes)
    with reactive.isolate():
        assert add.output() == 12
        assert double.upstream() is root.output
        assert add.upstream() is double.output

    root_value.set(5)
    with reactive.isolate():
        assert add.output() == 20


@pytest.mark.unit
def test_wiring_skips_removed_nodes_and_bridges_the_gap():
    root_value = reactive.Value(3)
    root = make_node(root_value)
    final = make_node(None)

    @reactive.calc
    def final_output():
        source = final.upstream()
        assert source is not None
        return source() + 1

    final.output = final_output
    nodes = {"root": root, "final": final}

    wire_card_nodes(("root", "removed", "final"), nodes)

    with reactive.isolate():
        assert final.upstream() is root.output
        assert final.output() == 4
