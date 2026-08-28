"""Simplex pivoting hot loop (doc/36 P2 / §5).

Isolates the core two-phase-simplex pivot loop so it can be backed by multiple
equivalent-fidelity implementations with byte-identical numerics.  The math
mirrors :func:`helixlang.plugins.runtime.metabolism._simplex_max` (Bland's rule, reduced-cost
entering test, ratio test with smallest-index tie-break, pivot normalization +
column elimination) so swapping backends never changes results — a pure speed
switch, never a fidelity switch (§3ξ.5).

Public API (shared by every ``impl_*``)::

    run(tableau, basis, obj, n_vars, eps=1e-9, max_iter=10000,
        forbidden=None) -> "optimal" | "unbounded" | "max_iter"

``tableau`` (n_rows x n_vars+1, RHS in last column) and ``basis`` are mutated
in place; returns the termination status.  Because a whole simplex run is the
unit of work (rather than a single pivot), ``run`` is the hot-loop step here.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot


def run(tableau, basis, obj, n_vars, eps=1e-9, max_iter=10000, forbidden=None):
    """Alias to the fastest available equivalent-fidelity implementation."""
    mod = load_hot("helixlang._accel.simplex")
    return mod.run(tableau, basis, obj, n_vars, eps, max_iter, forbidden)
