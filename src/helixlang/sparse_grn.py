"""Sparse genome-scale GRN (doc/18-programmable-cell-population-simulation.md §13 Design 5, task 2).

A dense weight matrix for G = 4300 E. coli MG1655-scale genes costs
``4300^2 * 8 B ~ 148 MB`` per template — and the old per-cell deep-copy
scheme would multiply that by the cell count.  A real regulatory net
(RegulonDB scale) has only ~10^4 edges, so the matrix is stored in CSR
form and only the nonzero entries are touched:

    inputs[c, t] = sum_{s} W[t, s] * levels[c, s]

with the same per-target summation order as :meth:`GRN.step`, so
:meth:`SparseGRN.step` reproduces the scalar recurrence for every cell
row in one vectorized call (the VectorizedGRN cross-validation pattern,
``test_vectorized.py``).  scipy is used when present (fast CSR matmul);
a pure-numpy CSR matmul fallback keeps the module dependency-free.

The **active-gene budget** (:meth:`SparseGRN.step_budgeted`) is the
performance path: at any instant only ~10% of E. coli genes are
significantly transcribed, so targets whose level and all incoming
sources are near-silent are advanced by pure decay ``level' = decay *
level`` and edges from globally silent sources are skipped.  When the
requested budget is below the number of materially-driven targets, the
top-``budget`` by raw input magnitude are kept (deterministic).

References:
- Martínez-Antonio & Collado-Vides 2003, Curr Opin Microbiol 6:482
  (E. coli TF network is hierarchical, global regulators on top).
- Martínez-Antonio, Janga & Thieffry 2008, J Mol Biol 381:238
  (sparse ~10^4-edge functional network).
- Salgado et al. 2013, NAR 41:D203 (RegulonDB scale).
- Alon 2007; vectorized GRN matrix semantics (as vectorized.py).
- Kick et al. 2019 NUFEB (large-scale agent-based runtime).
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is the standard env
    _HAS_NUMPY = False

try:
    import scipy.sparse  # noqa: F401  (imported for type annotations)
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional extra
    _HAS_SCIPY = False

from helixlang.grn import GRN, _activation_raw

#: expression level below which a gene counts as "silent" for the
#: active-gene budget (E. coli: only ~10% of genes significantly
#: transcribed at any instant; Martínez-Antonio et al. 2008).
SILENT_LEVEL = 0.01


def _vectorized_activation(inputs: np.ndarray, thresholds: np.ndarray,
                           hill_n: np.ndarray, kd: np.ndarray,
                           has_hill: bool) -> np.ndarray:
    """Vectorized per-gene activation (identical to ``grn._activation_raw``).

    Mirrors ``VectorizedGRN.activation`` so sigmoid and Hill paths stay
    byte-for-byte aligned with the scalar recurrence.
    """
    if has_hill:
        act: np.ndarray = 1.0 / (
            1.0 + np.exp(-(inputs - thresholds[None, :])))
        mask = hill_n > 0
        x = np.clip(inputs, 0.0, None)
        xn = x ** hill_n[None, :]
        kdn = kd[None, :] ** hill_n[None, :]
        kdnz = np.where(kdn > 0.0, kdn, 1.0)
        hill = np.where(kdn > 0.0, xn / (kdnz + xn),
                        (x > 0.0).astype(float))
        out: np.ndarray = np.where(mask[None, :], hill, act)
        return out
    sigmoid_act: np.ndarray = 1.0 / (
        1.0 + np.exp(-(inputs - thresholds[None, :])))
    return sigmoid_act


@dataclass(slots=True)
class SparseGRN:
    """Genome-scale GRN in sparse (CSR) form over a shared template.

    The topology (weights / thresholds / decays) is built once and shared
    by every cell; per-cell expression state is a row of ``levels``
    ``(N_cells, G)`` (doc/18-programmable-cell-population-simulation.md Design 5, task 3).  Advances any number of
    cells in one call.
    """

    names: list[str]
    data: np.ndarray      # edge weights, CSR target-major (NNZ,)
    col_indices: np.ndarray  # source index of each edge (NNZ,)
    row_ptr: np.ndarray   # CSR row pointers over targets (G+1,)
    thresholds: np.ndarray
    decays: np.ndarray
    hill_n: np.ndarray
    kd: np.ndarray
    noise_fano: np.ndarray       # (G,) Fano factor, 0 = no noise
    noise_expression_scale: np.ndarray  # (G,)
    noise_seed: int | None = None
    _idx: dict[str, int] = field(default_factory=dict, init=False)
    _noise_mask: np.ndarray = field(init=False)
    _noise_gen: object | None = field(default=None, init=False)
    _edge_rows: np.ndarray = field(init=False)
    _csr: object | None = field(default=None, init=False)
    _csr_t: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        g = len(self.names)
        if len(self.data) != len(self.col_indices):
            raise ValueError("CSR data/col length mismatch")
        if len(self.row_ptr) != g + 1:
            raise ValueError("CSR row_ptr must have G+1 entries")
        self._idx: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self._noise_mask = np.asarray(self.noise_fano > 0.0)
        self._csr = None
        self._csr_t = None
        # target index of every edge (precomputed for the budgeted step)
        self._edge_rows = np.repeat(
            np.arange(g), np.diff(self.row_ptr)).astype(np.int64)
        if _HAS_SCIPY:
            from scipy.sparse import csr_matrix
            w = csr_matrix((self.data, self.col_indices, self.row_ptr),
                           shape=(g, g))
            self._csr = w
            self._csr_t = w.T.tocsr()
        if _HAS_NUMPY and self._noise_mask.any():
            self._noise_gen = np.random.default_rng(self.noise_seed)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_grn(cls, grn: GRN, noise_seed: int | None = None
                 ) -> SparseGRN:
        """Build a sparse GRN from a scalar :class:`~helixlang.grn.GRN`.

        Targets are ordered by insertion order (same as ``GRN.step``);
        each target row lists its incoming edges in insertion order, so
        the input summation order matches the scalar recurrence exactly.
        """
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        names = list(grn.nodes.keys())
        idx = {n: i for i, n in enumerate(names)}
        data: list[float] = []
        cols: list[int] = []
        row_ptr = [0]
        for tname in names:
            for e in grn._incoming.get(tname, ()):  # noqa: SLF001
                data.append(e.weight)
                cols.append(idx[e.source])
            row_ptr.append(len(data))
        thresholds = np.array([grn.nodes[n].threshold for n in names],
                              dtype=float)
        decays = np.array(
            [grn.nodes[n].decay if grn.nodes[n].decay is not None
             else grn._default_decay for n in names], dtype=float)  # noqa: SLF001
        hill_n = np.array(
            [grn.nodes[n].hill_n if grn.nodes[n].hill_n is not None
             else 0.0 for n in names], dtype=float)
        kd = np.array(
            [grn.nodes[n].kd if grn.nodes[n].kd is not None
             else grn.nodes[n].threshold for n in names], dtype=float)
        fano: list[float] = []
        scale: list[float] = []
        for n in names:
            noise = grn.nodes[n].noise
            if noise is not None:
                fano.append(noise.fano_factor())
                scale.append(noise.expression_scale)
            else:
                fano.append(0.0)
                scale.append(1.0)
        return cls(
            names=names,
            data=np.asarray(data, dtype=float),
            col_indices=np.asarray(cols, dtype=np.int64),
            row_ptr=np.asarray(row_ptr, dtype=np.int64),
            thresholds=thresholds,
            decays=decays,
            hill_n=hill_n,
            kd=kd,
            noise_fano=np.asarray(fano, dtype=float),
            noise_expression_scale=np.asarray(scale, dtype=float),
            noise_seed=noise_seed,
        )

    @property
    def n_genes(self) -> int:
        return len(self.names)

    @property
    def n_edges(self) -> int:
        return int(len(self.data))

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def new_state(self, n_cells: int, seed: int | None = None,
                  initial_genes: tuple[str, ...] = ()) -> np.ndarray:
        """All-silent expression state ``(n_cells, G)``.

        ``initial_genes`` are seeded at level 1.0 (a minimal essential
        gene set, Karr 2012 style) so the GRN has something to transcribe
        from; the rest start quiet.
        """
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        levels = np.zeros((n_cells, self.n_genes), dtype=float)
        for name in initial_genes:
            i = self._idx.get(name)
            if i is not None:
                levels[:, i] = 1.0
        return levels

    def triggered(self, levels: np.ndarray, threshold: float = 0.5
                  ) -> np.ndarray:
        """Boolean ``(N, G)`` mask of genes above ``threshold``."""
        return np.asarray(levels) > threshold

    def n_active(self, levels: np.ndarray, threshold: float = SILENT_LEVEL
                 ) -> np.ndarray:
        """Per-cell active-gene counts (level above ``threshold``)."""
        counts: np.ndarray = (np.asarray(levels) > threshold).sum(axis=1)
        return counts

    # ------------------------------------------------------------------
    # exact step (bitwise-equivalent to GRN.step, all cells at once)
    # ------------------------------------------------------------------
    def _inputs(self, levels: np.ndarray) -> np.ndarray:
        """``inputs[c, t] = sum_s W[t,s] * levels[c,s]`` over CSR edges."""
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        if self._csr is not None:
            return np.asarray(levels) @ self._csr_t  # type: ignore[no-any-return]
        # pure-numpy CSR matmul (target-major accumulation)
        inputs: np.ndarray = np.zeros_like(np.asarray(levels), dtype=float)
        ptr = self.row_ptr
        for t in range(self.n_genes):
            a, b = int(ptr[t]), int(ptr[t + 1])
            for k in range(a, b):
                inputs[:, t] += self.data[k] * levels[:, self.col_indices[k]]
        return inputs

    def step(self, levels: np.ndarray) -> np.ndarray:
        """Advance one tick for every cell row of ``levels`` ``(N, G)``.

        Returns the updated level matrix (clipped to [0, 1]) — the
        exact :meth:`GRN.step` recurrence, vectorized across cells.
        When the source GRN carried telegraph ``noise=`` promoters and
        the scaffold enables noise, per-gene zero-mean Gaussian noise
        matching the two-state Fano factor is drawn once for all cells.
        """
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        levels = np.asarray(levels, dtype=float)
        inputs = self._inputs(levels)
        act = _vectorized_activation(
            inputs, self.thresholds, self.hill_n, self.kd,
            bool((self.hill_n > 0).any()))
        blended = (self.decays[None, :] * levels
                   + (1.0 - self.decays)[None, :] * act)
        if self._noise_mask.any():
            # one vectorized sampling for the whole population (Design 5 gate 5)
            std = np.zeros_like(blended)
            m = blended[:, self._noise_mask]
            dec = self.decays[self._noise_mask]
            std[:, self._noise_mask] = np.sqrt(
                (1.0 - dec * dec)
                * self.noise_fano[self._noise_mask]
                * np.maximum(0.0, m)
                / self.noise_expression_scale[self._noise_mask])
            ng = self._noise_gen
            blended = blended + ng.normal(0.0, std)  # type: ignore[attr-defined]
        clipped: np.ndarray = np.clip(blended, 0.0, 1.0)
        return clipped

    # ------------------------------------------------------------------
    # active-gene-budget step (performance path)
    # ------------------------------------------------------------------
    def step_budgeted(self, levels: np.ndarray, budget: int | None = None,
                      silent_level: float = SILENT_LEVEL) -> np.ndarray:
        """Budgeted tick: only materially-driven targets are recomputed.

        - Edges whose source is silent in *every* cell are skipped.
        - Targets whose own level and all incoming sources are silent
          are advanced by pure decay (input ``0``).
        - When more targets than ``budget`` carry nonzero input, the
          top-``budget`` by |input| are updated; the rest decay.

        This is the doc's active-gene-budget design: with ~10% of genes
        active, the edge pass collapses from NNZ to the active-source
        subgraph.  Results match :meth:`step` up to the floating-point
        truncation of dropped near-silent contributions (asserted in
        ``tests/test_genome_scale.py``).
        """
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        levels = np.asarray(levels, dtype=float)
        g = self.n_genes
        # globally silent sources (no cell above the level) -> their edges
        # are dropped; targets with no surviving input decay silently
        src_active: np.ndarray = np.asarray(
            (levels > silent_level).any(axis=0))
        if _HAS_SCIPY and src_active.any():
            from scipy.sparse import csr_matrix
            keep = src_active[self.col_indices]
            f_data = self.data[keep]
            f_col = self.col_indices[keep]
            rows = self._edge_rows[keep]
            if rows.size:
                counts = np.bincount(rows, minlength=g)
                f_ptr = np.concatenate(([0], np.cumsum(counts)))
                w = csr_matrix((f_data, f_col, f_ptr), shape=(g, g))
                inputs = np.asarray(levels) @ w.T
            else:
                inputs = np.zeros_like(levels)
        else:
            # pure-numpy reduced accumulation (source-major)
            src_ids = np.flatnonzero(src_active)
            keep = np.isin(self.col_indices, src_ids)
            rows = self._edge_rows[keep]
            f_data = self.data[keep]
            f_col = self.col_indices[keep]
            inputs = np.zeros_like(levels)
            for r, c, v in zip(rows, f_col, f_data, strict=True):
                inputs[:, r] += v * levels[:, c]
        if budget is not None and budget < g:
            # restrict the budget cut to columns that actually carry input
            mag_col = np.abs(inputs).sum(axis=0)
            cand = np.flatnonzero(mag_col > 0)
            if cand.size > budget:
                sub = np.abs(inputs[:, cand]).astype(np.float32)
                keep_local = np.argpartition(
                    sub, -budget, axis=1)[:, -budget:]
                keep_cols = cand[keep_local]
                mask = np.ones(inputs.shape, dtype=bool)
                rows_all = np.arange(inputs.shape[0])[:, None]
                mask[rows_all, keep_cols] = False
                inputs[mask] = 0.0
        act = _vectorized_activation(
            inputs, self.thresholds, self.hill_n, self.kd,
            bool((self.hill_n > 0).any()))
        blended = (self.decays[None, :] * levels
                   + (1.0 - self.decays)[None, :] * act)
        if self._noise_mask.any():
            std = np.zeros_like(blended)
            m = blended[:, self._noise_mask]
            dec = self.decays[self._noise_mask]
            std[:, self._noise_mask] = np.sqrt(
                (1.0 - dec * dec)
                * self.noise_fano[self._noise_mask]
                * np.maximum(0.0, m)
                / self.noise_expression_scale[self._noise_mask])
            ng = self._noise_gen
            blended = blended + ng.normal(0.0, std)  # type: ignore[attr-defined]
        clipped: np.ndarray = np.clip(blended, 0.0, 1.0)
        return clipped

    def scalar_inputs(self, levels: np.ndarray) -> np.ndarray:
        """Reference inputs via the scalar recurrence (test oracle).

        Rebuilds a throwaway :class:`GRN` and runs :meth:`GRN.step`
        input summation per target so the sparse path can be compared
        bit-for-bit with the scalar implementation.
        """
        if not _HAS_NUMPY:
            raise ImportError("SparseGRN requires numpy")
        grn = self.to_grn()
        out = np.zeros_like(np.asarray(levels, dtype=float))
        for c, row in enumerate(np.asarray(levels, dtype=float)):
            for i, name in enumerate(self.names):
                node = grn.nodes[name]
                inputs = 0.0
                for e in grn._incoming.get(name, ()):  # noqa: SLF001
                    inputs += e.weight * row[self._idx[e.source]]
                out[c, i] = _activation_raw(node, inputs)
        return out

    def to_grn(self) -> GRN:
        """Materialize a scalar :class:`~helixlang.grn.GRN` template.

        Used for cross-validation and for wiring the engineered `.helix`
        layer on top of the background genome.
        """
        grn = GRN()
        for i, name in enumerate(self.names):
            grn.add_gene(name, float(self.thresholds[i]),
                         decay=float(self.decays[i]),
                         hill_n=(float(self.hill_n[i])
                                 if self.hill_n[i] > 0 else None),
                         kd=(float(self.kd[i]) if self.hill_n[i] > 0
                             else None))
        ptr = self.row_ptr
        for t in range(self.n_genes):
            a, b = int(ptr[t]), int(ptr[t + 1])
            for k in range(a, b):
                grn.add_edge(self.names[self.col_indices[k]], self.names[t],
                             float(self.data[k]))
        return grn


def sparse_from_edges(names: list[str], edges: list[tuple[str, str, float]],
                      thresholds: list[float] | None = None,
                      decays: list[float] | None = None,
                      noise_seed: int | None = None) -> SparseGRN:
    """Build a :class:`SparseGRN` directly from a name + edge list.

    Args:
        names: gene names in matrix order.
        edges: ``(source, target, weight)`` triples.
        thresholds: per-gene activation thresholds (default all 0.5).
        decays: per-gene decay (default :attr:`GRN.DECAY`).
    """
    if not _HAS_NUMPY:
        raise ImportError("SparseGRN requires numpy")
    grn = GRN()
    idx = {n: i for i, n in enumerate(names)}
    default_decay = GRN.DECAY
    for i, name in enumerate(names):
        thr = 0.5 if thresholds is None else thresholds[i]
        dec = default_decay if decays is None else decays[i]
        grn.add_gene(name, thr, decay=dec)
    for s, t, w in edges:
        if s not in idx or t not in idx:
            raise ValueError(f"edge references unknown gene: {s!r} -> {t!r}")
        grn.add_edge(s, t, w)
    return SparseGRN.from_grn(grn, noise_seed=noise_seed)


__all__ = [
    "SILENT_LEVEL",
    "SparseGRN",
    "sparse_from_edges",
    "_HAS_SCIPY",
    "_HAS_NUMPY",
]
