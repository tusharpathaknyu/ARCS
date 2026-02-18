"""Eulerian circuit graph representation and augmentation.

Converts circuit netlists into graph form (nodes=pins, edges=components),
finds Eulerian paths/circuits, and generates multiple valid orderings
as data augmentation (following AnalogGenie's approach).

Key insight: A circuit can be serialized as a sequence in many valid orders.
Different Eulerian paths through the circuit graph give different but equivalent
token sequences. Training on all orderings helps the model learn that circuits
are about connections, not sequence position.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np


@dataclass
class CircuitEdge:
    """An edge in the circuit graph = one component."""

    component_type: str  # e.g., "RESISTOR", "MOSFET_N"
    component_id: str    # e.g., "R1", "Q1"
    value: float         # Component value
    pin_a: str           # e.g., "R1_P", "Q1_D"
    pin_b: str           # e.g., "R1_N", "Q1_S"
    net_a: str           # Net name for pin_a, e.g., "NET_0", "input"
    net_b: str           # Net name for pin_b, e.g., "NET_1", "GND"


@dataclass
class CircuitGraph:
    """Graph representation of a circuit.

    Nodes = nets (electrical nodes)
    Edges = components connecting nets
    """

    edges: list[CircuitEdge] = field(default_factory=list)
    net_names: set[str] = field(default_factory=set)
    _multigraph: Optional[nx.MultiGraph] = field(default=None, repr=False)

    def add_component(
        self,
        component_type: str,
        component_id: str,
        value: float,
        pin_a: str,
        pin_b: str,
        net_a: str,
        net_b: str,
    ) -> None:
        """Add a two-terminal component to the graph."""
        edge = CircuitEdge(
            component_type=component_type,
            component_id=component_id,
            value=value,
            pin_a=pin_a,
            pin_b=pin_b,
            net_a=net_a,
            net_b=net_b,
        )
        self.edges.append(edge)
        self.net_names.add(net_a)
        self.net_names.add(net_b)
        self._multigraph = None  # Invalidate cache

    @property
    def graph(self) -> nx.MultiGraph:
        """Build/return NetworkX multigraph (nets as nodes, components as edges)."""
        if self._multigraph is None:
            G = nx.MultiGraph()
            for net in self.net_names:
                G.add_node(net)
            for i, edge in enumerate(self.edges):
                G.add_edge(
                    edge.net_a,
                    edge.net_b,
                    key=i,
                    component_type=edge.component_type,
                    component_id=edge.component_id,
                    value=edge.value,
                    pin_a=edge.pin_a,
                    pin_b=edge.pin_b,
                )
            self._multigraph = G
        return self._multigraph

    def has_euler_circuit(self) -> bool:
        """Check if the graph has an Eulerian circuit (all vertices even degree)."""
        G = self.graph
        if not nx.is_connected(G):
            return False
        return all(d % 2 == 0 for _, d in G.degree())

    def has_euler_path(self) -> bool:
        """Check if the graph has an Eulerian path (0 or 2 odd-degree vertices)."""
        G = self.graph
        if not nx.is_connected(G):
            return False
        odd_count = sum(1 for _, d in G.degree() if d % 2 != 0)
        return odd_count in (0, 2)

    def make_eulerian(self) -> "CircuitGraph":
        """Add minimum edges to make the graph Eulerian if it isn't.

        Uses Hierholzer's approach: add dummy edges between odd-degree nodes.
        Returns a new CircuitGraph with dummy edges added.
        """
        G = self.graph.copy()

        # Find odd-degree nodes
        odd_nodes = [n for n, d in G.degree() if d % 2 != 0]

        if len(odd_nodes) == 0:
            return self  # Already Eulerian

        # Pair up odd nodes and add dummy edges
        new_graph = CircuitGraph()
        new_graph.edges = list(self.edges)
        new_graph.net_names = set(self.net_names)

        for i in range(0, len(odd_nodes) - 1, 2):
            dummy_id = f"DUMMY_{i // 2}"
            new_graph.add_component(
                component_type="WIRE",
                component_id=dummy_id,
                value=0.0,
                pin_a=f"{dummy_id}_A",
                pin_b=f"{dummy_id}_B",
                net_a=odd_nodes[i],
                net_b=odd_nodes[i + 1],
            )

        return new_graph

    def find_euler_paths(self, max_paths: int = 20) -> list[list[CircuitEdge]]:
        """Find multiple Eulerian paths through the circuit graph.

        Each path is a different valid ordering of the same circuit.
        Used for data augmentation.

        Args:
            max_paths: Maximum number of distinct paths to find.

        Returns:
            List of edge orderings, each a valid circuit serialization.
        """
        G = self.graph

        if not nx.is_connected(G):
            # If disconnected, find paths within each component
            return self._find_paths_disconnected(max_paths)

        # Make Eulerian if needed
        euler_graph = self.make_eulerian()
        G = euler_graph.graph

        paths = []
        edge_list = list(euler_graph.edges)

        # Try different starting nodes to get diverse orderings
        nodes = list(G.nodes())
        rng = np.random.default_rng(42)
        rng.shuffle(nodes)

        for start_node in nodes[:max_paths * 2]:  # Try more starts than needed
            if len(paths) >= max_paths:
                break

            try:
                euler_edges = list(nx.eulerian_path(G, source=start_node))
                # Map back to CircuitEdge objects
                path = self._euler_edges_to_circuit_edges(euler_edges, euler_graph)

                # Filter out dummy edges
                path = [e for e in path if e.component_type != "WIRE"]

                # Check if this ordering is genuinely different
                path_sig = tuple(e.component_id for e in path)
                existing_sigs = {tuple(e.component_id for e in p) for p in paths}
                if path_sig not in existing_sigs:
                    paths.append(path)

            except (nx.NetworkXError, StopIteration):
                continue

        return paths if paths else [self.edges]  # Fallback to original ordering

    def _euler_edges_to_circuit_edges(
        self, euler_edges: list[tuple], euler_graph: "CircuitGraph"
    ) -> list[CircuitEdge]:
        """Map NetworkX Euler path edges back to CircuitEdge objects."""
        G = euler_graph.graph
        edge_usage = {}  # Track which parallel edges we've used

        result = []
        for u, v in euler_edges:
            # Find an unused edge between u and v
            edges_between = G[u][v]
            for key, data in edges_between.items():
                if (u, v, key) not in edge_usage and (v, u, key) not in edge_usage:
                    edge_usage[(u, v, key)] = True
                    # Find corresponding CircuitEdge
                    for ce in euler_graph.edges:
                        if ce.component_id == data["component_id"]:
                            result.append(ce)
                            break
                    break

        return result

    def _find_paths_disconnected(self, max_paths: int) -> list[list[CircuitEdge]]:
        """Handle disconnected graphs by finding paths in each component."""
        G = self.graph
        components = list(nx.connected_components(G))

        if len(components) == 1:
            return [self.edges]

        # For each connected component, find local edges
        component_edges = []
        for comp_nodes in components:
            local_edges = [e for e in self.edges if e.net_a in comp_nodes]
            component_edges.append(local_edges)

        # Create orderings by permuting component order
        paths = []
        for perm in itertools.islice(itertools.permutations(range(len(component_edges))), max_paths):
            path = []
            for idx in perm:
                path.extend(component_edges[idx])
            paths.append(path)

        return paths


def circuit_sample_to_graph(sample) -> CircuitGraph:
    """Convert a CircuitSample to a CircuitGraph.

    Maps the flat parameter dict to a graph with proper net connectivity
    based on the topology type.
    """
    from arcs.datagen import CircuitSample

    graph = CircuitGraph()
    topology = sample.topology
    params = sample.parameters

    # Topology-specific graph construction
    if topology == "buck":
        _build_buck_graph(graph, params)
    elif topology == "boost":
        _build_boost_graph(graph, params)
    elif topology in ("cuk", "sepic"):
        _build_dual_inductor_graph(graph, params, topology)
    else:
        # Generic: create a simple series graph
        _build_generic_graph(graph, params)

    return graph


def _build_buck_graph(graph: CircuitGraph, params: dict) -> None:
    """Build graph for buck converter topology."""
    graph.add_component("SWITCH_IDEAL", "SW1", params.get("r_dson", 0.05),
                       "SW1_IN", "SW1_OUT", "input", "sw_node")
    graph.add_component("DIODE", "D1", 0.3,
                       "D1_A", "D1_K", "GND", "sw_node")
    graph.add_component("INDUCTOR", "L1", params["inductance"],
                       "L1_P", "L1_N", "sw_node", "vout")
    graph.add_component("CAPACITOR", "C1", params["capacitance"],
                       "C1_P", "C1_N", "vout", "GND")
    graph.add_component("RESISTOR", "Rload", params.get("r_load", 5.0),
                       "Rload_P", "Rload_N", "vout", "GND")


def _build_boost_graph(graph: CircuitGraph, params: dict) -> None:
    """Build graph for boost converter topology."""
    graph.add_component("INDUCTOR", "L1", params["inductance"],
                       "L1_P", "L1_N", "input", "sw_node")
    graph.add_component("SWITCH_IDEAL", "SW1", params.get("r_dson", 0.05),
                       "SW1_IN", "SW1_OUT", "sw_node", "GND")
    graph.add_component("DIODE", "D1", 0.3,
                       "D1_A", "D1_K", "sw_node", "vout")
    graph.add_component("CAPACITOR", "C1", params["capacitance"],
                       "C1_P", "C1_N", "vout", "GND")
    graph.add_component("RESISTOR", "Rload", params.get("r_load", 24.0),
                       "Rload_P", "Rload_N", "vout", "GND")


def _build_dual_inductor_graph(graph: CircuitGraph, params: dict, topology: str) -> None:
    """Build graph for Ćuk/SEPIC topologies."""
    graph.add_component("INDUCTOR", "L1", params["inductance_1"],
                       "L1_P", "L1_N", "input", "sw_node")
    graph.add_component("SWITCH_IDEAL", "SW1", params.get("r_dson", 0.05),
                       "SW1_IN", "SW1_OUT", "sw_node", "GND")
    graph.add_component("CAPACITOR", "Cc", params["cap_coupling"],
                       "Cc_P", "Cc_N", "sw_node", "mid")
    graph.add_component("INDUCTOR", "L2", params["inductance_2"],
                       "L2_P", "L2_N", "mid", "vout")
    graph.add_component("DIODE", "D1", 0.3,
                       "D1_A", "D1_K", "mid" if topology == "sepic" else "GND", "vout")
    graph.add_component("CAPACITOR", "C1", params["capacitance"],
                       "C1_P", "C1_N", "vout", "GND")
    graph.add_component("RESISTOR", "Rload", params.get("r_load", 5.0),
                       "Rload_P", "Rload_N", "vout", "GND")


def _build_generic_graph(graph: CircuitGraph, params: dict) -> None:
    """Build a generic graph from parameter names."""
    net_counter = 0
    for name, value in params.items():
        if "inductance" in name:
            comp_type = "INDUCTOR"
        elif "capacitance" in name or "cap" in name:
            comp_type = "CAPACITOR"
        elif "r_" in name or "esr" in name:
            comp_type = "RESISTOR"
        else:
            comp_type = "RESISTOR"  # Default

        comp_id = f"{comp_type[0]}{net_counter}"
        graph.add_component(
            comp_type, comp_id, value,
            f"{comp_id}_P", f"{comp_id}_N",
            f"NET_{net_counter}", f"NET_{net_counter + 1}",
        )
        net_counter += 1
