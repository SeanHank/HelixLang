"""Numba-JIT implementation of the Gray-Scott diffusion update (doc/36 §5.3).

Optional stack: only used when the operator sets ``HELIX_ACCEL`` to include
``numba`` — JIT warmup is ignored by default.  Same numerics as ``impl_python``
(5-point Laplacian, U·V² reaction, clamp, borders preserved) — a pure speed
switch.  Absence of numba is **not** a silent fallback: :func:`step` raises
:class:`~helixlang.core.errors.NativeBackendError`.
"""
from __future__ import annotations

from helixlang.core.errors import NativeBackendError

try:
    import numpy as np
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is optional
    np = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _step_nb(u, v, F, k, Du, Dv):
        """Numba kernel; mirrors ``impl_python`` interior update + clamp."""
        n = u.shape[0]
        nu = np.empty_like(u)
        nv = np.empty_like(v)
        nu[0, :] = u[0, :]
        nu[-1, :] = u[-1, :]
        nv[0, :] = v[0, :]
        nv[-1, :] = v[-1, :]
        for i in range(1, n - 1):
            nu[i, 0] = u[i, 0]
            nu[i, -1] = u[i, -1]
            nv[i, 0] = v[i, 0]
            nv[i, -1] = v[i, -1]
            for j in range(1, n - 1):
                lu = (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]
                      - 4.0 * u[i, j]) * 0.25
                lv = (v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1]
                      - 4.0 * v[i, j]) * 0.25
                uij = u[i, j]
                vij = v[i, j]
                uvv = uij * vij * vij
                x = uij + (Du * lu - uvv + F * (1.0 - uij))
                y = vij + (Dv * lv + uvv - (F + k) * vij)
                x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
                nu[i, j] = x
                nv[i, j] = y
        return nu, nv

    def step(u, v, F, k, Du, Dv):
        ua = np.ascontiguousarray(u, dtype=np.float64)
        va = np.ascontiguousarray(v, dtype=np.float64)
        nu, nv = _step_nb(ua, va, float(F), float(k), float(Du), float(Dv))
        return nu, nv

else:  # pragma: no cover - numba is optional

    def step(u, v, F, k, Du, Dv):
        raise NativeBackendError(
            "numba is required for the numba diffusion stack but is not "
            "installed.",
            rebuild="pip install helixlang[native]",
        )
