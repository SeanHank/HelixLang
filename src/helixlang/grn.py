"""Gene regulatory network (GRN): sigmoid concentration threshold model.

Each tick updates each gene's expression level; genes with level > 0.5 are triggered to execute their ORF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(slots=True)
class GeneNode:
    name: str
    threshold: float
    level: float = 0.0


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    weight: float


class GRN:
    """Hybrid GRN model with discrete ticks + sigmoid thresholds."""

    DECAY = 0.7  # Protein decay coefficient (smaller values respond faster)

    def __init__(self) -> None:
        self.nodes: dict[str, GeneNode] = {}
        self.edges: list[Edge] = []

    def add_gene(self, name: str, threshold: float,
                 initial_level: float = 0.0) -> None:
        self.nodes[name] = GeneNode(name, threshold, initial_level)

    def add_edge(self, source: str, target: str, weight: float) -> None:
        self.edges.append(Edge(source, target, weight))

    def set_level(self, name: str, level: float) -> None:
        if name in self.nodes:
            self.nodes[name].level = max(0.0, min(1.0, level))

    def step(self) -> list[str]:
        """Advance one tick, returning the names of the genes triggered this tick (level > 0.5)."""
        new_levels: dict[str, float] = {}
        for name, node in self.nodes.items():
            inputs = sum(
                e.weight * self.nodes[e.source].level
                for e in self.edges if e.target == name
            )
            raw = sigmoid(inputs - node.threshold)
            # Decay + new input (not doubled, to avoid self-excitation of genes without inputs)
            blended = self.DECAY * node.level + (1 - self.DECAY) * raw
            new_levels[name] = max(0.0, min(1.0, blended))
        for name, lvl in new_levels.items():
            self.nodes[name].level = lvl
        return [n for n, l in new_levels.items() if l > 0.5]
