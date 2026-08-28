"""Pure-Python reference implementation of the Gray-Scott diffusion update.

Equivalent fidelity with the numpy/numba batch paths — identical numerics,
Python loops.  Mirrors ``GrayScott._step_py`` exactly (interior update, clamp
to [0,1], borders preserved).
"""
from __future__ import annotations


def step(u, v, F, k, Du, Dv):
    """See ``backend`` docstring.  ``u``/``v`` are 2-D lists; borders preserved."""
    n = len(u)
    nu = [row[:] for row in u]
    nv = [row[:] for row in v]
    for i in range(1, n - 1):
        um = u[i - 1]
        up = u[i + 1]
        vm = v[i - 1]
        vp = v[i + 1]
        ui = u[i]
        vi = v[i]
        nui = nu[i]
        nvi = nv[i]
        for j in range(1, n - 1):
            lu = (um[j] + up[j] + ui[j - 1] + ui[j + 1] - 4.0 * ui[j]) * 0.25
            lv = (vm[j] + vp[j] + vi[j - 1] + vi[j + 1] - 4.0 * vi[j]) * 0.25
            uij = ui[j]
            vij = vi[j]
            uvv = uij * vij * vij
            nui[j] = uij + (Du * lu - uvv + F * (1.0 - uij))
            nvi[j] = vij + (Dv * lv + uvv - (F + k) * vij)
            if nui[j] < 0.0:
                nui[j] = 0.0
            elif nui[j] > 1.0:
                nui[j] = 1.0
            if nvi[j] < 0.0:
                nvi[j] = 0.0
            elif nvi[j] > 1.0:
                nvi[j] = 1.0
    return nu, nv
