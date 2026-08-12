"""Large-scale vectorized runtime (T3.3, gap G8).

Scale to 10^6+ agents by pushing the per-agent loop into array
operations:

- :class:`VectorizedGRN`: the :class:`~helixlang.grn.GRN` discrete
  recurrence as a matrix multiply, so an entire population's gene
  networks advance in one vectorized step (identical semantics to
  ``GRN.step``, byte-for-byte same activation function).
- :func:`sort_cells`: cache-friendly spatial ordering of a population
  (NUFEB lesson: process neighbors together).
- :func:`iter_snapshots`: streaming snapshot generator with optional
  incremental JSONL output, so long runs never hold the full history in
  memory.
- :data:`_HAS_NUMBA` / :func:`optional_jit`: optional JIT hot-path
  wrapper (no-op when numba is absent).

Grid diffusion is already vectorized (numpy path in
:mod:`helixlang.environment` / :mod:`helixlang.population`); parallel
grid decomposition is left for a distributed backend.

References:
- Kick et al. 2019 (NUFEB). Commun Comput Phys 27:1882-1908.
- Alon 2007 (threshold/sigmoid GRN dynamics; vectorized via matrix
  algebra, e.g. GPUsim / CuDNN-based GRN solvers).
"""
from __future__ import annotations

import json
from typing import Iterator, Sequence

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is the standard env
    _HAS_NUMPY = False

try:
    import numba  # type: ignore
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - optional extra
    _HAS_NUMBA = False

from helixlang.grn import GRN


def optional_jit(**kwargs):
    """JIT wrapper: returns a numba ``njit``-compiled function when numba
    is available, otherwise the plain Python function (no-op)."""
    def deco(fn):
        if _HAS_NUMBA:
            try:
                return numba.njit(**kwargs)(fn)
            except Exception:  # pragma: no cover - exotic numba errors
                return fn
        return fn
    return deco


@optional_jit()
def _sigmoid_array(x):
    """Numerically stable vectorized sigmoid (matches ``grn.sigmoid``)."""
    z = np.exp(-x)
    return 1.0 / (1.0 + z)


class VectorizedGRN:
    """Across-cell GRN in matrix form (one numpy step for all cells).

    Builds the weight matrix ``W`` (target x source) and per-gene
    threshold / decay / Hill parameters once, then advances any number of
    cells with

        inputs    = L @ W.T
        activation = sigma(inputs - threshold)      (sigmoid path)
        L'        = decay * L + (1 - decay) * activation

    which is exactly the discrete recurrence of :meth:`GRN.step`, so a
    vectorized population reproduces the scalar per-cell dynamics
    (used to cross-check one against the other in tests).
    """

    def __init__(self, grn: GRN) -> None:
        self.names: list[str] = list(grn.nodes.keys())
        n = len(self.names)
        idx = {name: i for i, name in enumerate(self.names)}
        if _HAS_NUMPY:
            W = np.zeros((n, n), dtype=float)
            for e in grn.edges:
                W[idx[e.target], idx[e.source]] = e.weight
            thresholds = np.array(
                [grn.nodes[name].threshold for name in self.names],
                dtype=float)
            decays = np.array(
                [grn.nodes[name].decay if grn.nodes[name].decay is not None
                 else grn._default_decay for name in self.names],
                dtype=float)
            hill_n = np.array(
                [grn.nodes[name].hill_n if grn.nodes[name].hill_n is not None
                 else 0.0 for name in self.names], dtype=float)
            kd = np.array(
                [grn.nodes[name].kd if grn.nodes[name].kd is not None
                 else grn.nodes[name].threshold for name in self.names],
                dtype=float)
            self._W = W
            self._thresholds = thresholds
            self._decays = decays
            self._hill_n = hill_n
            self._kd = kd
            self._has_hill = bool((hill_n > 0).any())
        else:  # pragma: no cover - numpy is the standard env
            self._W = [[0.0] * n for _ in range(n)]
            for e in grn.edges:
                self._W[idx[e.target]][idx[e.source]] = e.weight
            self._thresholds = [
                grn.nodes[name].threshold for name in self.names]
            self._decays = [
                grn.nodes[name].decay if grn.nodes[name].decay is not None
                else grn._default_decay for name in self.names]
            self._hill_n = [grn.nodes[name].hill_n for name in self.names]
            self._kd = [grn.nodes[name].kd for name in self.names]
            self._has_hill = any(
                h is not None and h > 0 for h in self._hill_n)

    @property
    def n_genes(self) -> int:
        return len(self.names)

    def activation(self, inputs) -> "np.ndarray":
        """Vectorized per-gene activation ``(N, G)`` for inputs ``(N, G)``.

        Genes with ``hill_n > 0`` use Hill kinetics ``x^n/(kd^n + x^n)``;
        all others use the sigmoid threshold model — identical to
        ``grn._activation_raw``.
        """
        if not _HAS_NUMPY:  # pragma: no cover - numpy is the standard env
            return self._activation_python(inputs)
        if self._has_hill:
            act = _sigmoid_array(inputs - self._thresholds[None, :])
            mask = self._hill_n > 0
            x = np.clip(inputs, 0.0, None)
            xn = x ** self._hill_n[None, :]
            kdn = self._kd[None, :] ** self._hill_n[None, :]
            kdnz = np.where(kdn > 0.0, kdn, 1.0)
            hill = np.where(kdn > 0.0, xn / (kdnz + xn), (x > 0.0).astype(float))
            return np.where(mask[None, :], hill, act)
        return _sigmoid_array(inputs - self._thresholds[None, :])

    def _activation_python(self, inputs):  # pragma: no cover - fallback
        out = []
        for row in inputs:
            r = []
            for g, x in enumerate(row):
                if self._hill_n[g] and self._hill_n[g] > 0:
                    kd = self._kd[g] if self._kd[g] is not None \
                        else self._thresholds[g]
                    xn = x ** self._hill_n[g]
                    kdn = kd ** self._hill_n[g]
                    r.append(1.0 if kdn <= 0 and x > 0
                             else (0.0 if x <= 0 else xn / (kdn + xn)))
                else:
                    from helixlang.grn import sigmoid
                    r.append(sigmoid(x - self._thresholds[g]))
            out.append(r)
        return out

    def step(self, levels: "np.ndarray") -> "np.ndarray":
        """Advance one tick for every cell row of ``levels`` ``(N, G)``.

        Returns the updated level matrix (clipped to [0, 1]).
        """
        if not _HAS_NUMPY:  # pragma: no cover - numpy is the standard env
            out = [[0.0] * self.n_genes for _ in levels]
            for c, row in enumerate(levels):
                inputs = [sum(self._W[g][s] * row[s]
                              for s in range(self.n_genes))
                          for g in range(self.n_genes)]
                act = self.activation([inputs])[0]
                for g in range(self.n_genes):
                    out[c][g] = max(0.0, min(1.0,
                        self._decays[g] * row[g]
                        + (1 - self._decays[g]) * act[g]))
            return out
        inputs = np.asarray(levels, dtype=float) @ self._W.T
        act = self.activation(inputs)
        new = self._decays[None, :] * np.asarray(levels, dtype=float) \
            + (1.0 - self._decays)[None, :] * act
        return np.clip(new, 0.0, 1.0)

    def triggered(self, levels: "np.ndarray", threshold: float = 0.5) -> "np.ndarray":
        """Boolean ``(N, G)`` mask of genes above ``threshold``."""
        return np.asarray(levels) > threshold


def sort_cells(cells: Sequence, keys: Sequence[str] = ("x", "y", "z")) -> list:
    """Stable spatial ordering of a population (cache-friendly processing).

    Sorts by the given cell attributes in order, so neighboring agents
    are visited together (NUFEB-style spatial locality).
    """
    def keyfn(c):
        return tuple(getattr(c, k, 0.0) for k in keys)
    return sorted(cells, key=keyfn)


def iter_snapshots(population, n_steps: int, interval: int = 1,
                   path: str | None = None) -> Iterator[dict]:
    """Stream per-tick population snapshots.

    Yields a snapshot dict every ``interval`` steps while running
    ``n_steps`` ticks of ``population.step()``:

    ``{"step", "alive", "cells": [{id, x, y, z, energy, alive}]}``

    When ``path`` is given, snapshots are appended incrementally as
    JSONL so arbitrarily long runs need no in-memory history.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if interval < 1:
        raise ValueError("interval must be >= 1")
    fh = None
    try:
        fh = open(path, "a", encoding="utf-8") if path else None
        for s in range(n_steps):
            population.step()
            if s % interval == 0:
                snap = _make_snapshot(population, s + 1)
                if fh is not None:
                    fh.write(json.dumps(snap) + "\n")
                    fh.flush()
                yield snap
    finally:
        if fh is not None:
            fh.close()


def _make_snapshot(population, step: int) -> dict:
    cells = population.cells
    return {
        "step": step,
        "alive": sum(1 for c in cells if c.alive),
        "cells": [
            {"id": c.id, "x": c.x, "y": c.y, "z": getattr(c, "z", 0),
             "energy": round(c.energy, 3), "alive": c.alive}
            for c in cells
        ],
    }


__all__ = [
    "VectorizedGRN",
    "_HAS_NUMBA",
    "_HAS_NUMPY",
    "iter_snapshots",
    "optional_jit",
    "sort_cells",
]
