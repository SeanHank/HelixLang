"""Dispatch alias: exposes ``run_quota`` via the uniform hot-loop loader.

For now the only available implementation is the pure-Python reference in
``impl_python``; ``impl_cext``/``impl_cython`` are added under the ``[native]``
build and selected automatically by
:func:`helixlang._accel._loaders.choose_backend`.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot


def run_quota(code, constants, *, quota: int = 4096, gene_table=None):
    """Execute ``quota`` ops of ``code`` with the fastest available backend."""
    mod = load_hot("helixlang._accel.dispatch")
    return mod.run_quota(code, constants, quota=quota, gene_table=gene_table)


def run_many(code, constants, *, quota: int = 4096, n_cells: int = 1,
             gene_table=None):
    """Population dispatch of ``code`` over ``n_cells`` with the best backend."""
    mod = load_hot("helixlang._accel.dispatch")
    return mod.run_many(code, constants, quota=quota, n_cells=n_cells,
                        gene_table=gene_table)
