"""Spatial-omics-guided multi-scale models (T3.1, gap G8).

MiMICS-style (Walsh et al. 2024, PLoS Comput Biol 20(4):e1012031)
transcriptomics-guided ABM: expression matrices are imported, clustered
into discrete cell states, and each state is mapped onto (a) a distinct
GRN parameter set and (b) a distinct FBA bound set, so the simulated
heterogeneity can be calibrated against spatial transcriptomics at
Par-seqFISH scale (Dar et al. 2021, Nat Biotechnol 39:313-319).

Public API:
- :class:`ExpressionMatrix`: cells x genes expression with optional
  spatial coordinates; importers :func:`from_arrays`,
  :func:`read_expression_matrix` (TSV/CSV).
- :meth:`ExpressionMatrix.normalized`, ``barcode``, ``cluster``:
  per-gene scaling, binarization, k-means state calling.
- :func:`expression_to_grn_states`: state centroids -> per-gene GRN
  thresholds and initial levels for :func:`build_state_grn`.
- :func:`expression_to_fba_bounds` / :func:`apply_fba_bounds`: state
  expression -> uptake / reaction bound sets (transcriptomics-guided
  metabolic switching).
- :class:`SpatialAtlas`: spatial spot coordinates + states;
  :meth:`SpatialAtlas.state_at` nearest-spot assignment for population
  cells.
- :func:`compare_heterogeneity`: adjusted Rand index / state-match
  fraction between simulated and observed spatial states.

References:
- Walsh et al. 2024. MiMICS: a Bayesian network-based ABM. PLoS Comput
  Biol 20(4):e1012031.
- Dar et al. 2021. Spatial transcriptomics of planktonic and sessile
  bacterial populations. Nat Biotechnol 39:313-319.
- Karr et al. 2012. A whole-cell computational model of M. genitalium.
  Cell 150:389-401 (data-calibrated parameter sets).
"""
from __future__ import annotations

import csv
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is the standard env
    _HAS_NUMPY = False

from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


# ============================================================================
# Expression matrix
# ============================================================================

@dataclass
class ExpressionMatrix:
    """Cells x genes expression matrix with optional spatial coordinates.

    Attributes:
        genes: gene names (columns).
        cells: cell / spot identifiers (rows).
        values: ``values[c][g]`` = expression of gene ``g`` in cell ``c``.
        x, y, z: optional spatial coordinates per cell (for spatial
            transcriptomics, e.g. Par-seqFISH; defaults to ``None``).
    """

    genes: list[str]
    cells: list[str]
    values: list[list[float]]
    x: list[float] | None = None
    y: list[float] | None = None
    z: list[float] | None = None

    def __post_init__(self) -> None:
        n_cells = len(self.cells)
        n_genes = len(self.genes)
        if len(self.values) != n_cells:
            raise ValueError("values rows must match cells")
        if any(len(row) != n_genes for row in self.values):
            raise ValueError("values columns must match genes")
        if self.x is not None and len(self.x) != n_cells:
            raise ValueError("x coordinates must match cells")
        if self.y is not None and len(self.y) != n_cells:
            raise ValueError("y coordinates must match cells")
        if self.z is not None and len(self.z) != n_cells:
            raise ValueError("z coordinates must match cells")

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_cells, n_genes)``."""
        return len(self.cells), len(self.genes)

    def cell_vector(self, cell_id: str) -> list[float]:
        """Expression row of one cell."""
        i = self.cells.index(cell_id)
        return list(self.values[i])

    def gene_profile(self, gene: str) -> list[float]:
        """Expression of one gene across all cells."""
        g = self.genes.index(gene)
        return [row[g] for row in self.values]

    def normalized(self, method: str = "max") -> ExpressionMatrix:
        """Per-gene scaled copy with values in [0, 1].

        ``method="max"`` divides every gene by its maximum across cells;
        ``method="sum"`` divides by the total across cells (fractional
        abundance, MiMICS-style state signatures).
        """
        out = [[0.0] * len(self.genes) for _ in self.cells]
        for g in range(len(self.genes)):
            col = [row[g] for row in self.values]
            denom = max(col) if method == "max" else sum(col)
            if denom == 0:
                continue
            for c in range(len(self.cells)):
                out[c][g] = col[c] / denom
        return ExpressionMatrix(list(self.genes), list(self.cells), out,
                                None if self.x is None else list(self.x),
                                None if self.y is None else list(self.y),
                                None if self.z is None else list(self.z))

    def barcode(self, threshold: float = 0.5) -> ExpressionMatrix:
        """Binarized copy (``>= threshold`` -> 1, else 0)."""
        out = [[1.0 if v >= threshold else 0.0 for v in row]
               for row in self.values]
        return ExpressionMatrix(list(self.genes), list(self.cells), out,
                                None if self.x is None else list(self.x),
                                None if self.y is None else list(self.y),
                                None if self.z is None else list(self.z))

    def cluster(self, k: int, seed: int = 0, iters: int = 100) -> tuple[list[int], list[list[float]]]:
        """k-means over cells; returns ``(state_ids, centroids)``.

        ``state_ids[c]`` is the cluster label of cell ``c`` and
        ``centroids[s]`` the mean expression vector of state ``s``.
        Vectorized with numpy when available; pure-Python fallback
        (nearest-neighbor over centroids) otherwise.  State labels are
        deterministic for a given ``seed``.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        n = len(self.cells)
        if n == 0:
            raise ValueError("no cells to cluster")
        rng = random.Random(seed)
        k = min(k, n)
        # deterministic seed selection via random choice of distinct cells
        idx = list(range(n))
        rng.shuffle(idx)
        centroids = [list(self.values[idx[s]]) for s in range(k)]
        state_ids = [0] * n
        for _ in range(iters):
            if _HAS_NUMPY:
                vals = np.asarray(self.values, dtype=float)
                cents = np.asarray(centroids, dtype=float)
                d = ((vals[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
                new_ids = [int(i) for i in d.argmin(axis=1)]
            else:  # pragma: no cover - numpy is the standard env
                new_ids = [min(range(k),
                               key=lambda s: _euclidean(self.values[c],
                                                        centroids[s]))
                           for c in range(n)]
            new_cents: list[list[float]] = []
            for s in range(k):
                members = [c for c, sid in enumerate(new_ids) if sid == s]
                if not members:
                    new_cents.append(list(centroids[s]))
                else:
                    d = len(members)
                    new_cents.append([
                        sum(self.values[c][g] for c in members) / d
                        for g in range(len(self.genes))])
            if new_ids == state_ids:
                state_ids = new_ids
                centroids = new_cents
                break
            state_ids, centroids = new_ids, new_cents
        return state_ids, centroids

    def state_centroids(self, state_ids: Sequence[int]) -> list[list[float]]:
        """Mean expression vector per state given per-cell ``state_ids``."""
        k = max(state_ids) + 1 if state_ids else 0
        sums: list[list[float]] = [[0.0] * len(self.genes) for _ in range(k)]
        counts = [0] * k
        for c, s in enumerate(state_ids):
            counts[s] += 1
            for g in range(len(self.genes)):
                sums[s][g] += self.values[c][g]
        return [[v / counts[s] if counts[s] else 0.0 for v in sums[s]]
                for s in range(k)]


def from_arrays(genes: Sequence[str], cells: Sequence[str],
                values: Sequence[Sequence[float]],
                x: Sequence[float] | None = None,
                y: Sequence[float] | None = None,
                z: Sequence[float] | None = None) -> ExpressionMatrix:
    """Build an :class:`ExpressionMatrix` from in-memory arrays."""
    return ExpressionMatrix(list(genes), list(cells),
                            [list(row) for row in values],
                            None if x is None else list(x),
                            None if y is None else list(y),
                            None if z is None else list(z))


def read_expression_matrix(path: str, delimiter: str = "\t",
                           transpose: bool = False,
                           coords_columns: Sequence[str] = (),
                           cell_label: str = "cell") -> ExpressionMatrix:
    """Import a TSV (default) / CSV expression matrix.

    The header row names the genes; the first column is the cell / spot
    label (default ``cell``).  Columns named in ``coords_columns`` are
    read as spatial coordinates (x, y, z in header order) and removed
    from the gene set.  ``transpose=True`` reads a genes x cells table
    (first column = gene names).  Blank lines and ``#`` comments are
    skipped.
    """
    with open(path, newline="") as fh:
        rows = [[c.strip() for c in line]
                for line in csv.reader(fh, delimiter=delimiter)
                if line and not line[0].startswith("#")]
    if not rows:
        raise ValueError("empty expression matrix")
    header, body = rows[0], rows[1:]
    if transpose:
        gene_cols = [row[0] for row in body]
        cell_cols = header[1:]
        values = [[float(row[i]) for row in body]
                  for i in range(1, len(header))]
        coords: dict[str, list[float]] = {}
        for name in coords_columns:
            if name not in header:
                continue
            i = header.index(name)
            coords[name] = [float(row[i]) for row in body]
        cells = cell_cols
        genes = gene_cols
    else:
        cell_cols = [row[0] for row in body]
        gene_cols = [c for c in header[1:] if c not in coords_columns]
        gene_idx = [i for i in range(1, len(header))
                    if header[i] not in coords_columns]
        values = [[float(row[i]) for i in gene_idx] for row in body]
        coords = {name: [float(row[header.index(name)]) for row in body]
                  for name in coords_columns if name in header}
        cells = cell_cols
        genes = gene_cols
    if cell_label in coords_columns:
        raise ValueError(f"{cell_label!r} is reserved for the id column")
    return ExpressionMatrix(
        genes, cells, values,
        coords.get("x"),
        coords.get("y"),
        coords.get("z"))


# ============================================================================
# Expression -> GRN parameter sets
# ============================================================================

def expression_to_grn_states(
        matrix: ExpressionMatrix, k: int, seed: int = 0,
        base_threshold: float = 0.5,
        min_threshold: float = 0.2) -> tuple[list[int], list[list[float]], list[dict[str, dict]]]:
    """Map an expression matrix onto ``k`` GRN parameter sets.

    Cells are clustered into ``k`` states; every state gets a per-gene
    parameter dict used by :func:`build_state_grn`:

    - ``initial_level``: the state's normalized mean expression of the
      gene (clamped to [0, 1]);
    - ``threshold``: ``base_threshold * (1 - expr)`` floored at
      ``min_threshold``, so genes strongly expressed in a state become
      easier to activate (lower threshold), and silent genes keep the
      base threshold.

    Returns:
        ``(state_ids, centroids, state_params)`` where
        ``state_params[s][gene] = {"initial_level": ..., "threshold": ...}``.
    """
    norm = matrix.normalized(method="max")
    state_ids, centroids = norm.cluster(k, seed=seed)
    centroids_norm = matrix.state_centroids(state_ids)
    state_params: list[dict[str, dict]] = []
    for s in range(k):
        params: dict[str, dict] = {}
        for g, gene in enumerate(matrix.genes):
            expr = _clamp(centroids_norm[s][g])
            params[gene] = {
                "initial_level": expr,
                "threshold": max(min_threshold,
                                 base_threshold * (1.0 - expr)),
            }
        state_params.append(params)
    return state_ids, centroids_norm, state_params


def build_state_grn(grn: GRN, params: dict[str, dict]) -> GRN:
    """Copy a template GRN with one state's parameter set applied.

    Preserves edges, weights, decays and noise settings; overrides each
    gene's ``initial_level`` and ``threshold`` from ``params``.  The
    template's nodes are used as the gene list (genes absent from
    ``params`` keep their template values).
    """
    out = GRN(noise_enabled=grn.noise_enabled)
    for name, node in grn.nodes.items():
        p = params.get(name, {})
        out.add_gene(
            name,
            threshold=p.get("threshold", node.threshold),
            initial_level=p.get("initial_level", node.level),
            decay=node.decay,
            hill_n=node.hill_n,
            kd=node.kd,
            noise=node.noise)
    for e in grn.edges:
        out.add_edge(e.source, e.target, e.weight)
    return out


# ============================================================================
# Expression -> FBA bound sets
# ============================================================================

def expression_to_fba_bounds(
        matrix: ExpressionMatrix, state_ids: Sequence[int],
        gene_to_reaction: dict[str, str],
        base_bound: float,
        direction: dict[str, int] | None = None,
        min_bound: float = 0.0) -> list[dict[str, float]]:
    """Map expression states onto distinct FBA bound sets.

    For each state the mean expression of every gene is turned into a
    reaction bound ``clamp(base_bound * expr, min_bound, base_bound)``
    for the reaction named in ``gene_to_reaction``.  A negative entry in
    ``direction`` inverts the mapping (repressed reactions become
    constrained when the gene is expressed, MiMICS-style metabolic
    switching).

    Args:
        matrix: expression matrix (columns matched by ``gene_to_reaction``).
        state_ids: per-cell state labels (as from
            :meth:`ExpressionMatrix.cluster`).
        gene_to_reaction: gene name -> reaction id (an ``EX_*`` exchange
            id, e.g. ``EX_glc``, or any other reaction in the model).
        base_bound: bound applied at unit expression.
        direction: optional per-gene sign (default all +1).
        min_bound: floor for the derived bound.

    Returns:
        one dict ``reaction_id -> bound`` per state.
    """
    norm = matrix.normalized(method="max")
    centroids = norm.state_centroids(list(state_ids))
    direction = direction or {}
    k = len(centroids)
    out: list[dict[str, float]] = []
    for s in range(k):
        bounds: dict[str, float] = {}
        for gene, rxn in gene_to_reaction.items():
            g = matrix.genes.index(gene)
            sign = direction.get(gene, 1)
            expr = _clamp(centroids[s][g])
            v = base_bound * (expr if sign > 0 else (1.0 - expr))
            bounds[rxn] = max(min_bound, min(base_bound, v))
        out.append(bounds)
    return out


def apply_fba_bounds(fba: FluxBalanceAnalysis, bounds: dict[str, float]) -> None:
    """Apply a bound set to a :class:`FluxBalanceAnalysis` in place.

    ``EX_<metabolite>`` exchange reactions set the metabolite uptake
    limit (``FluxBalanceAnalysis.set_uptake``); every other reaction's
    ``upper_bound`` is set directly.  Reaction ids not present in the
    model are ignored (lenient to missing reactions).
    """
    model = fba.model
    ex_met: dict[str, str] = {}
    for rid, rxn in model.reactions.items():
        if rxn.subsystem == "exchange" and rid.startswith("EX_"):
            # positive-coefficient metabolite is the exchanged species
            mets = [met for met, coef in rxn.stoichiometry.items()
                    if coef > 0]
            if mets:
                ex_met[rid] = mets[0]
    for rid, bound in bounds.items():
        if rid in ex_met:
            fba.set_uptake(ex_met[rid], bound)
        elif rid in model.reactions:
            model.reactions[rid].upper_bound = float(bound)


# ============================================================================
# Spatial atlas + heterogeneity comparison
# ============================================================================

class SpatialAtlas:
    """Spatial transcriptomics atlas: spots at (x, y, z) with state ids.

    Assigns population cells to the nearest spatial spot's state, the
    Par-seqFISH-scale analogue of the MiMICS transcriptomics-guided
    state field.
    """

    def __init__(self, spots: Sequence[tuple[float, float, float]],
                 state_ids: Sequence[int]) -> None:
        if len(spots) != len(state_ids):
            raise ValueError("spots and state_ids must have equal length")
        self.spots = [tuple(s) for s in spots]
        self.state_ids = [int(s) for s in state_ids]

    def state_at(self, x: float, y: float, z: float = 0.0) -> int:
        """State id of the nearest spot."""
        if not self.spots:
            raise ValueError("empty atlas")
        best, best_d2 = 0, float("inf")
        for i, (sx, sy, sz) in enumerate(self.spots):
            d2 = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
            if d2 < best_d2:
                best, best_d2 = i, d2
        return self.state_ids[best]

    def assign_cell_states(self, cells: Sequence) -> list[int]:
        """Nearest-spot state for each cell (uses ``cell.x/y/z``)."""
        return [self.state_at(getattr(c, "x", 0.0),
                              getattr(c, "y", 0.0),
                              getattr(c, "z", 0.0)) for c in cells]


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand index between two cluster labelings (Hubert & Arabie 1985).

    Compares two independent partitions of the same cells (e.g. simulated
    per-cell states vs. spatial-transcriptomics states) on a [-1, 1] scale
    where 1 = identical partition.  Computed from the contingency table
    ``n_ij`` (cells in state i of ``a`` and state j of ``b``):

        ARI = (index - expected) / (max_index - expected)

    with ``index = sum_ij C(n_ij, 2)`` and the usual
    :math:`sum_a = sum_i C(a_i, 2)`, :math:`sum_b = sum_j C(b_j, 2)`.
    """
    if len(a) != len(b):
        raise ValueError("labelings must have equal length")
    n = len(a)
    if n == 0:
        return 0.0
    a_labels = sorted(set(a))
    b_labels = sorted(set(b))
    cont: dict[tuple[int, int], int] = {}
    for i in range(n):
        cont[(a[i], b[i])] = cont.get((a[i], b[i]), 0) + 1
    n_ij = [[cont.get((i, j), 0) for j in b_labels] for i in a_labels]
    a_sizes = [sum(row) for row in n_ij]
    b_sizes = [sum(n_ij[i][j] for i in range(len(a_labels)))
               for j in range(len(b_labels))]
    def choose2(x: int) -> int:
        return x * (x - 1) // 2
    index = sum(choose2(n_ij[i][j]) for i in range(len(a_labels))
                for j in range(len(b_labels)))
    sum_a = sum(choose2(s) for s in a_sizes)
    sum_b = sum(choose2(s) for s in b_sizes)
    total = choose2(n)
    if total == 0:
        return 0.0
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2.0
    if max_index - expected == 0:
        return 1.0 if index == expected else 0.0
    return (index - expected) / (max_index - expected)


def compare_heterogeneity(simulated: Sequence[int],
                          observed: Sequence[int]) -> dict[str, float]:
    """Agreement between simulated and observed per-cell states.

    Returns ``{"ari": ..., "state_match": ...}`` where ``state_match``
    is the best-over-permutation fraction of cells assigned to a
    matching observed state (Hungarian-free greedy: each simulated state
    is matched to its most abundant observed counterpart).
    """
    ari = adjusted_rand_index(list(simulated), list(observed))
    n = len(simulated)
    if n == 0:
        return {"ari": 1.0, "state_match": 1.0}
    sim_states = sorted(set(simulated))
    obs_states = sorted(set(observed))
    used: set[int] = set()
    matched = 0
    for s in sim_states:
        best_obs = max(
            (b for b in obs_states if b not in used),
            default=None,
            key=lambda b: sum(1 for i in range(n)
                              if simulated[i] == s and observed[i] == b))
        if best_obs is not None:
            used.add(best_obs)
            matched += sum(1 for i in range(n)
                           if simulated[i] == s and observed[i] == best_obs)
    return {"ari": ari, "state_match": matched / n}


__all__ = [
    "ExpressionMatrix",
    "SpatialAtlas",
    "adjusted_rand_index",
    "apply_fba_bounds",
    "build_state_grn",
    "compare_heterogeneity",
    "expression_to_fba_bounds",
    "expression_to_grn_states",
    "from_arrays",
    "read_expression_matrix",
]
