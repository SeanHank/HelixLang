"""Pure-Python reference implementation of the two-phase simplex pivot loop.

Equivalent fidelity with the numpy/Cython paths — identical numerics.
Mirrors ``metabolism._simplex_max`` exactly (Bland's rule, ratio test with
smallest-index tie-break, in-place pivot normalization + elimination).
"""
from __future__ import annotations

_INF = float("inf")
_EPS = 1e-9


def run(tableau, basis, obj, n_vars, eps=_EPS, max_iter=10000, forbidden=None):
    """See ``backend`` docstring.  Mutates ``tableau``/``basis`` in place."""
    n_rows = len(tableau)
    rhs_col = n_vars
    if forbidden is None:
        forbidden = set()
    for _ in range(max_iter):
        cB = [obj[basis[i]] for i in range(n_rows)]
        entering = -1
        for j in range(n_vars):
            if j in basis or j in forbidden:
                continue
            rc = obj[j] - sum(cB[i] * tableau[i][j] for i in range(n_rows))
            if rc > eps:
                entering = j
                break
        if entering == -1:
            return "optimal"
        leaving_row = -1
        min_ratio = _INF
        min_basis_idx = n_vars + 1
        for i in range(n_rows):
            pivot = tableau[i][entering]
            if pivot > eps:
                ratio = tableau[i][rhs_col] / pivot
                if (ratio < min_ratio - eps
                        or (abs(ratio - min_ratio) <= eps
                            and basis[i] < min_basis_idx)):
                    min_ratio = ratio
                    leaving_row = i
                    min_basis_idx = basis[i]
        if leaving_row == -1:
            return "unbounded"
        pivot_val = tableau[leaving_row][entering]
        inv_pivot = 1.0 / pivot_val
        for k in range(n_vars + 1):
            tableau[leaving_row][k] *= inv_pivot
        for i in range(n_rows):
            if i == leaving_row:
                continue
            factor = tableau[i][entering]
            if abs(factor) < eps:
                continue
            for k in range(n_vars + 1):
                tableau[i][k] -= factor * tableau[leaving_row][k]
        basis[leaving_row] = entering
    return "max_iter"
