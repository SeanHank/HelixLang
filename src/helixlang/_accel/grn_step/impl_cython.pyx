# cython: language_level=3
# distutils: language=c
"""Cython native implementation of the GRN discrete-step recurrence
(doc/36 §4.2 / §5.1).

Same numerics as ``impl_python`` (accumulate edge weights, sigmoid threshold,
decay blend, clip) — a pure speed switch, never a fidelity switch (§3ξ.5).
Compiled into ``helixlang/_accel/grn_step/*.so`` by the ``[native]`` build;
picked by the ``_accel`` loader ahead of the numpy/python stacks.
"""
from __future__ import annotations

from libc.math cimport exp


def step(list levels, list src, list dst, list weights,
         list decays, list thresholds, double default_decay):
    """See ``backend`` docstring.  ``decays`` uses None for per-gene default."""
    cdef Py_ssize_t e, i, n = len(levels)
    cdef double x, z, raw, blended, v, dec, w, lvl
    cdef list acc = [0.0] * n

    for e in range(len(src)):
        acc[dst[e]] += <double>weights[e] * <double>levels[src[e]]

    new_levels = []
    triggered = []
    for i in range(n):
        x = acc[i] - <double>thresholds[i]
        if x >= 0.0:
            z = exp(-x)
            raw = 1.0 / (1.0 + z)
        else:
            z = exp(x)
            raw = z / (1.0 + z)
        dec = decays[i]
        if dec is None:
            dec = default_decay
        blended = dec * <double>levels[i] + (1.0 - dec) * raw
        v = blended if blended > 0.0 else 0.0
        if v > 1.0:
            v = 1.0
        new_levels.append(v)
        if v > 0.5:
            triggered.append(i)
    return new_levels, triggered
