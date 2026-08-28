# cython: boundscheck=False, wraparound=False, cdivision=False
"""Cython implementation of the simplex pivot loop (doc/36 §5.2).

Byte-identical to ``impl_python``: Bland's rule, ratio test with smallest-index
tie-break, in-place pivot normalization + elimination, same IEEE-754 op order.
Only the loop plumbing is compiled; arithmetic is ordinary doubles, so results
match the reference exactly — a pure speed switch.  Built into ``*.so`` by
``python -m helixlang._accel.build`` and selected under the ``native`` tag.
"""
from __future__ import annotations

INF = float("inf")
EPS = 1e-9


def run(tableau, basis, obj, n_vars, eps=EPS, max_iter=10000, forbidden=None):
    cdef Py_ssize_t n_rows, rhs_col, i, j, k
    cdef double min_ratio, pivot, ratio, pivot_val, inv_pivot, factor, rc, acc
    n_rows = len(tableau)
    rhs_col = n_vars
    if forbidden is None:
        forbidden = set()
    cdef int _max_iter = max_iter
    cdef double _eps = eps
    # local aliases for speed
    tab = tableau
    b = basis
    ob = obj
    for _iter in range(_max_iter):
        # reduced cost entering test (Bland: smallest index)
        entering = -1
        for j in range(n_vars):
            if j in b or j in forbidden:
                continue
            acc = 0.0
            for i in range(n_rows):
                acc += ob[b[i]] * tab[i][j]
            rc = ob[j] - acc
            if rc > _eps:
                entering = j
                break
        if entering == -1:
            return "optimal"
        leaving_row = -1
        min_ratio = INF
        min_basis_idx = n_vars + 1
        for i in range(n_rows):
            pivot = tab[i][entering]
            if pivot > _eps:
                ratio = tab[i][rhs_col] / pivot
                if (ratio < min_ratio - _eps
                        or (abs(ratio - min_ratio) <= _eps
                            and b[i] < min_basis_idx)):
                    min_ratio = ratio
                    leaving_row = i
                    min_basis_idx = b[i]
        if leaving_row == -1:
            return "unbounded"
        pivot_val = tab[leaving_row][entering]
        inv_pivot = 1.0 / pivot_val
        for k in range(n_vars + 1):
            tab[leaving_row][k] *= inv_pivot
        for i in range(n_rows):
            if i == leaving_row:
                continue
            factor = tab[i][entering]
            if abs(factor) < _eps:
                continue
            for k in range(n_vars + 1):
                tab[i][k] -= factor * tab[leaving_row][k]
        b[leaving_row] = entering
    return "max_iter"
