"""Numba-JIT implementation of the GRN discrete-step recurrence (doc/36 §4.2).

Optional stack: only used when the operator sets ``HELIX_ACCEL`` to include
``numba`` — JIT warmup is ignored by default so ``import helixlang`` / the
default ``native,numpy,python`` resolution stays startup-light.

Same numerics as ``impl_python`` (sigmoid-threshold, decay blend, clip) — a pure
speed switch, never a fidelity switch (§3ξ.5).  Absence of numba is **not** a
silent fallback: :func:`step` raises
:class:`~helixlang.core.errors.NativeBackendError` with the rebuild hint.
"""
from __future__ import annotations

import math
from typing import Any, cast

from helixlang.core.errors import NativeBackendError

try:
    import numpy as np
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is optional
    np = cast(Any, None)
    njit = cast(Any, None)
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _step_nb(levels, src, dst, weights, decays, thresholds, default_decay):
        """Numba kernel; mirrors ``impl_python._sigmoid`` arithmetic exactly."""
        n = levels.shape[0]
        acc = np.zeros(n, dtype=np.float64)
        for e in range(src.shape[0]):
            acc[dst[e]] += weights[e] * levels[src[e]]
        new_levels = np.zeros(n, dtype=np.float64)
        triggered = []
        for i in range(n):
            x = acc[i] - thresholds[i]
            if x >= 0.0:
                raw = 1.0 / (1.0 + math.exp(-x))
            else:
                z = math.exp(x)
                raw = z / (1.0 + z)
            dec = decays[i]
            if dec != dec:  # NaN (expanded None) -> default
                dec = default_decay
            blended = dec * levels[i] + (1.0 - dec) * raw
            v = blended if blended > 0.0 else 0.0
            if v > 1.0:
                v = 1.0
            new_levels[i] = v
            if v > 0.5:
                triggered.append(i)
        return new_levels, triggered


def step(levels, src, dst, weights, decays, thresholds, default_decay):
    """See ``backend`` docstring.  ``decays`` uses None for per-gene default."""
    if not _HAS_NUMBA:
        raise NativeBackendError(
            "numba is required for the numba GRN stack but is not installed.",
            rebuild="pip install helixlang[native]",
        )
    arr = np.asarray(levels, dtype=np.float64)
    src_i = np.asarray(src, dtype=np.int64)
    dst_i = np.asarray(dst, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    dec = np.asarray([d if d is not None else float("nan")
                      for d in decays], dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)
    new_levels, triggered = _step_nb(arr, src_i, dst_i, w, dec, thr,
                                     float(default_decay))
    return new_levels.tolist(), list(triggered)
