"""Gene regulatory network (GRN): sigmoid / Hill concentration
threshold model.

Each tick updates each gene's expression level; genes with level > 0.5
are triggered to execute their ORF.

Models:
- Legacy sigmoid threshold model (default, unchanged)::
      activation = sigmoid(sum(w_i * level_i) - threshold)
      level' = decay * level + (1 - decay) * activation
- Optional per-gene Hill kinetics (activator-saturating)::
      activation = inputs^n / (kd^n + inputs^n),  inputs >= 0
  with kd the half-maximal effector concentration and n the Hill
  coefficient. Real measured Kd values (e.g. lacI repressor Kd ~ 0.1 nM,
  Oehler 1990 EMBO J) can be supplied per gene via ``kd=``.

Per-gene decay follows measured protein half-lives via
:func:`decay_from_half_life_ticks` (E. coli protein half-lives median
~110 min; Mosteller 1980 J Biol Chem, Helbig 2011 Proteomics 11).
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


def hill(x: float, n: float, kd: float) -> float:
    """Hill activation function.

    ``x^n / (kd^n + x^n)``, half-max at ``x = kd``; returns 0 for
    non-positive input (pure activator, no ligand-binding side effect).

    Args:
        x:  effector concentration (regulatory input)
        n:  Hill coefficient (cooperativity)
        kd: dissociation constant at half-maximum activation
    """
    if x <= 0:
        return 0.0
    xn: float = x ** n
    kdn: float = kd ** n
    if kdn <= 0:  # kd == 0 -> unit step at any positive input
        return 1.0
    return xn / (kdn + xn)


def decay_from_half_life_ticks(half_life_ticks: float) -> float:
    """Per-tick decay coefficient from a protein half-life.

    ``decay = 0.5 ** (1 / half_life_ticks)`` is the fraction of protein
    remaining after one tick given a half-life expressed in ticks
    (first-order degradation: level halves every ``half_life_ticks``).

    E. coli protein half-lives are ~60-600 min (median ~110 min;
    Mosteller 1980, Helbig 2011), so with one tick per minute a typical
    decay is ~0.994, far slower than the legacy universal 0.7.

    Args:
        half_life_ticks: protein half-life in simulation ticks

    Returns:
        decay coefficient in [0, 1)
    """
    if half_life_ticks <= 0:
        raise ValueError("half_life_ticks must be > 0")
    return float(0.5 ** (1.0 / half_life_ticks))


@dataclass(slots=True)
class GeneNode:
    name: str
    threshold: float
    level: float = 0.0
    decay: float | None = None
    hill_n: float | None = None
    kd: float | None = None


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    weight: float


class GRN:
    """Hybrid GRN model with discrete ticks + sigmoid/Hill thresholds.

    ``DECAY`` is the universal legacy decay used when a gene has no
    per-gene ``decay=``. It is a dimensionless heuristic (smaller values
    respond faster); per-gene decay should be derived from measured
    protein half-lives via :func:`decay_from_half_life_ticks`.
    """

    DECAY = 0.7  # legacy universal decay (heuristic, not measured)

    def __init__(self) -> None:
        self.nodes: dict[str, GeneNode] = {}
        self.edges: list[Edge] = []
        # Per-target incoming-edge index (target -> edges), so ``step()`` is
        # O(N + E) instead of scanning all edges per node (O(N·E)).
        self._incoming: dict[str, list[Edge]] = {}
        self._edge_count = 0

    def add_gene(self, name: str, threshold: float,
                 initial_level: float = 0.0,
                 decay: float | None = None,
                 hill_n: float | None = None,
                 kd: float | None = None) -> None:
        """Add a gene node.

        Args:
            name: gene name
            threshold: activation threshold for the legacy sigmoid path
                (ignored when ``hill_n`` is set and ``kd`` is given)
            initial_level: starting expression level in [0, 1]
            decay: per-gene decay coefficient (default: :attr:`DECAY`);
                derive from a measured half-life with
                :func:`decay_from_half_life_ticks`
            hill_n: optional Hill coefficient; when set, activation uses
                Hill kinetics instead of the sigmoid
            kd: dissociation constant for Hill activation (default:
                ``threshold`` when omitted)
        """
        self.nodes[name] = GeneNode(name, threshold, initial_level,
                                    decay, hill_n, kd)
        self._incoming.setdefault(name, [])

    def add_edge(self, source: str, target: str, weight: float) -> None:
        self.edges.append(Edge(source, target, weight))
        self._incoming.setdefault(target, []).append(self.edges[-1])
        self._edge_count += 1

    def _rebuild_incoming(self) -> None:
        """Rebuild the incoming-edge index after direct ``edges`` mutation."""
        self._incoming = {}
        for e in self.edges:
            self._incoming.setdefault(e.target, []).append(e)
        self._edge_count = len(self.edges)

    def set_level(self, name: str, level: float) -> None:
        if name in self.nodes:
            self.nodes[name].level = max(0.0, min(1.0, level))

    def step(self) -> list[str]:
        """Advance one tick, returning the names of the genes triggered this tick (level > 0.5)."""
        # In-place weight updates (OP_REGULATE) reuse the same Edge objects, so
        # the index stays valid; guard against external direct ``edges`` appends.
        if self._edge_count != len(self.edges):
            self._rebuild_incoming()
        incoming = self._incoming
        nodes = self.nodes
        new_levels: dict[str, float] = {}
        for name, node in nodes.items():
            inputs: float = 0
            for e in incoming.get(name, ()):
                inputs += e.weight * nodes[e.source].level
            if node.hill_n is not None:
                kd = node.kd if node.kd is not None else node.threshold
                raw = hill(inputs, node.hill_n, kd)
            else:
                raw = sigmoid(inputs - node.threshold)
            decay = node.decay if node.decay is not None else self.DECAY
            # Decay + new input (not doubled, to avoid self-excitation of genes without inputs)
            blended = decay * node.level + (1 - decay) * raw
            new_levels[name] = max(0.0, min(1.0, blended))
        for name, lvl in new_levels.items():
            self.nodes[name].level = lvl
        return [n for n, l in new_levels.items() if l > 0.5]
