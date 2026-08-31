"""Numba-jit implementation of the simplex pivot loop (doc/39 O7).

Same numerics as ``impl_python``/``impl_numpy`` (Bland's rule, reduced-cost
entering test, ratio test with smallest-index tie-break, in-place pivot
normalization + elimination).  The inner pivot loop is ``@njit``-compiled on a
2-D float array (the only structure numba can JIT-own mutably without
reflection limits), so pure-wheel / native-less installs gain compiled speed
without a C toolchain when ``numba`` is present.  Absence of numba is **not** a
silent fallback: :func:`run` raises :class:`RuntimeError` unless ``numba`` is
importable.
"""
from __future__ import annotations

import math

try:  # pragma: no cover - numba is optional
    from numba import njit
except ImportError:  # pragma: no cover - numba is optional
    njit = None  # type: ignore[assignment]

_EPS = 1e-9


def _run_nb(tableau, basis, n_rows, n_vars, obj, eps, max_iter, rhs_col,
            forbidden):
    """njit-compiled core.  ``tableau`` is ``(n_rows, n_vars+1)`` float64;
    ``basis``/``obj``/``forbidden`` are int / float64 1-D arrays.  Returns the
    status code (0 optimal, 1 unbounded, 2 max_iter) and mutates in place."""
    n_forb = len(forbidden)
    for _ in range(max_iter):
        cB = obj[basis]
        entering = -1
        for j in range(n_vars):
            in_basis = False
            for bi in range(n_rows):
                if basis[bi] == j:
                    in_basis = True
                    break
            if in_basis:
                continue
            is_forbidden = False
            for fi in range(n_forb):
                if forbidden[fi] == j:
                    is_forbidden = True
                    break
            if is_forbidden:
                continue
            rc_j = obj[j]
            for i in range(n_rows):
                rc_j -= cB[i] * tableau[i, j]
            if rc_j > eps:
                entering = j
                break
        if entering == -1:
            return 0  # optimal
        leaving_row = -1
        min_ratio = math.inf
        min_basis_idx = n_vars + 1
        for i in range(n_rows):
            pivot = tableau[i, entering]
            if pivot > eps:
                ratio = tableau[i, rhs_col] / pivot
                if (ratio < min_ratio - eps
                        or (abs(ratio - min_ratio) <= eps
                            and basis[i] < min_basis_idx)):
                    min_ratio = ratio
                    leaving_row = i
                    min_basis_idx = basis[i]
        if leaving_row == -1:
            return 1  # unbounded
        pivot_val = tableau[leaving_row, entering]
        inv_pivot = 1.0 / pivot_val
        for k in range(n_vars + 1):
            tableau[leaving_row, k] *= inv_pivot
        for i in range(n_rows):
            if i == leaving_row:
                continue
            factor = tableau[i, entering]
            if abs(factor) < eps:
                continue
            for k in range(n_vars + 1):
                tableau[i, k] -= factor * tableau[leaving_row, k]
        basis[leaving_row] = entering
    return 2  # max_iter


if njit is not None:
    _RUN_NB = njit(cache=True)(_run_nb)
else:  # pragma: no cover - exercised only in pure-wheel-without-numba CI
    _RUN_NB = _run_nb


def run(tableau, basis, obj, n_vars, eps=_EPS, max_iter=10000, forbidden=None):
    """See ``backend`` docstring.  Mutates ``tableau``/``basis`` in place."""
    if njit is None:  # pragma: no cover - numba is optional
        raise RuntimeError(
            "numba is required for the numba simplex stack but is not installed"
        )
    n_rows = len(tableau)
    if n_rows == 0:
        return "optimal"
    rhs_col = n_vars
    basis_before = list(basis)
    import numpy as np

    table = np.asarray(
        [list(map(float, row)) for row in tableau], dtype=np.float64
    )
    basis_arr = np.asarray(basis_before, dtype=np.int64)
    obj_arr = np.asarray([float(o) for o in obj], dtype=np.float64)
    forbidden_arr = np.asarray(
        [int(i) for i in (forbidden or ())], dtype=np.int64
    )
    status = _RUN_NB(table, basis_arr, n_rows, n_vars, obj_arr, eps, max_iter,
                     rhs_col, forbidden_arr)
    for i, _row in enumerate(table):
        for k in range(n_vars + 1):
            tableau[i][k] = float(table[i, k])
    for i in range(n_rows):
        basis[i] = int(basis_arr[i])
    return ("optimal", "unbounded", "max_iter")[status]
