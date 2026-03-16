"""Smoke tests for arcs.euler — graph construction and Eulerian walk."""
from __future__ import annotations

import pytest
from arcs.euler import CircuitEdge, CircuitGraph


class TestCircuitEdge:
    def test_creation(self):
        edge = CircuitEdge(
            component_type="RESISTOR",
            component_id="R1",
            value=1000.0,
            pin_a="R1_P",
            pin_b="R1_N",
            net_a="NET_0",
            net_b="NET_1",
        )
        assert edge.component_type == "RESISTOR"
        assert edge.value == 1000.0
        assert edge.net_a == "NET_0"


class TestCircuitGraph:
    def test_empty_graph(self):
        g = CircuitGraph()
        assert len(g.edges) == 0
        assert len(g.net_names) == 0

    def test_add_component(self):
        g = CircuitGraph()
        g.add_component(
            component_type="RESISTOR",
            component_id="R1",
            value=1000.0,
            pin_a="R1_P",
            pin_b="R1_N",
            net_a="VIN",
            net_b="GND",
        )
        assert len(g.edges) == 1
        assert "VIN" in g.net_names
        assert "GND" in g.net_names

    def test_graph_property(self):
        g = CircuitGraph()
        g.add_component("RESISTOR", "R1", 1e3, "R1_P", "R1_N", "A", "B")
        g.add_component("CAPACITOR", "C1", 1e-6, "C1_P", "C1_N", "B", "GND")
        mg = g.graph
        assert mg.number_of_nodes() == 3
        assert mg.number_of_edges() == 2

    def test_euler_path_check(self):
        # Two edges forming a path A-B-A (Euler circuit)
        g = CircuitGraph()
        g.add_component("RESISTOR", "R1", 1e3, "R1_P", "R1_N", "A", "B")
        g.add_component("CAPACITOR", "C1", 1e-6, "C1_P", "C1_N", "B", "A")
        assert g.has_euler_circuit()
        assert g.has_euler_path()

    def test_find_euler_paths(self):
        g = CircuitGraph()
        g.add_component("RESISTOR", "R1", 1e3, "R1_P", "R1_N", "A", "B")
        g.add_component("CAPACITOR", "C1", 1e-6, "C1_P", "C1_N", "B", "A")
        paths = g.find_euler_paths(max_paths=5)
        assert len(paths) >= 1
        for path in paths:
            assert len(path) == 2  # Two edges

    def test_make_eulerian(self):
        # Three edges: A-B, B-C, C-D — not Eulerian (4 odd-degree nodes)
        g = CircuitGraph()
        g.add_component("RESISTOR", "R1", 1e3, "R1_P", "R1_N", "A", "B")
        g.add_component("RESISTOR", "R2", 2e3, "R2_P", "R2_N", "B", "C")
        g.add_component("RESISTOR", "R3", 3e3, "R3_P", "R3_N", "C", "D")
        eg = g.make_eulerian()
        assert eg.has_euler_path() or eg.has_euler_circuit()
