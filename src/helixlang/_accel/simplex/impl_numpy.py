"""NumPy batch implementation of the simplex pivot loop.

Same numerics as ``impl_python`` but vectorized (reduced-cost via ``cB @ T``
and pivot transformation via outer products).  Mirrors
``metabolism._simplex_max_numpy`` exactly.  Accepts list or ndarray ``tableau``
/``obj`` and mutates the supplied structure in place.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def run(tableau, basis, obj, n_vars, eps=_EPS, max_iter=10000, forbidden=None):
    """See ``backend`` docstring.  Mutates ``tableau``/``basis`` in place."""
    n_rows = len(tableau)
    if n_rows == 0:
        return "optimal"
    rhs_col = n_vars
    basis_arr = np.asarray(basis, dtype=np.intp)
    forbidden_mask = np.zeros(n_vars, dtype=bool)
    if forbidden:
        for j in forbidden:
            forbidden_mask[j] = True
    obj = np.asarray(obj, dtype=np.float64)
    tab = tableau if isinstance(tableau, np.ndarray) else np.asarray(
        tableau, dtype=np.float64)

    for _ in range(max_iter):
        cB = obj[basis_arr]
        col_contrib = cB @ tab[:, :n_vars]
        rc = obj - col_contrib
        in_current_basis = np.zeros(n_vars, dtype=bool)
        in_current_basis[basis_arr] = True
        eligible = (~in_current_basis) & (~forbidden_mask) & (rc > eps)
        candidates = np.nonzero(eligible)[0]
        if candidates.size == 0:
            return "optimal"
        entering = int(candidates[0])

        col = tab[:, entering]
        valid = col > eps
        if not np.any(valid):
            return "unbounded"
        rhs = tab[:, rhs_col]
        ratios = np.where(valid, rhs / np.where(valid, col, 1.0), np.inf)
        min_ratio = ratios.min()
        tied = np.abs(ratios - min_ratio) <= eps
        tied_rows = np.nonzero(tied)[0]
        if tied_rows.size > 0:
            leaving_row = int(tied_rows[np.argmin(basis_arr[tied_rows])])
        else:
            leaving_row = int(np.argmin(ratios))

        pivot_val = tab[leaving_row, entering]
        tab[leaving_row, :] /= pivot_val
        factor_col = tab[:, entering].copy()
        factor_col[leaving_row] = 0.0
        tab -= np.outer(factor_col, tab[leaving_row, :])
        basis[leaving_row] = entering
        basis_arr[leaving_row] = entering
    return "max_iter"
