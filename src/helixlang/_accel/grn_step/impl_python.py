"""Pure-Python reference implementation of the GRN discrete-step recurrence.

Equivalent fidelity with a numpy batch path — identical numerics, Python loops.
"""
from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    # MathJax-clean; matches helixlang.plugins.runtime.grn.sigmoid for the sigmoid-threshold path.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def step(levels, src, dst, weights, decays, thresholds, default_decay):
    """See ``backend`` docstring.  ``decays`` uses None for per-gene default."""
    n = len(levels)
    new_levels = [0.0] * n
    # aggregate incoming weighted sums
    acc = [0.0] * n
    for s, d, w in zip(src, dst, weights, strict=True):
        acc[d] += w * levels[s]
    triggered: list[int] = []
    for i in range(n):
        raw = _sigmoid(acc[i] - thresholds[i])
        dec = decays[i] if decays[i] is not None else default_decay
        blended = dec * levels[i] + (1.0 - dec) * raw
        v = blended if blended > 0.0 else 0.0
        if v > 1.0:
            v = 1.0
        new_levels[i] = v
        if v > 0.5:
            triggered.append(i)
    return new_levels, triggered


def _hill(x: float, n: float, kd: float) -> float:
    """Hill activation, mirroring ``helixlang.plugins.runtime.grn.hill`` exactly."""
    if x <= 0:
        return 0.0
    xn = x ** n
    kdn = kd ** n
    if kdn <= 0:  # kd == 0 -> unit step at any positive input
        return 1.0
    return float(xn / (kdn + xn))


def step_mixed(levels, src, dst, weights, decays, thresholds, default_decay,
               hill_ns, kds):
    """Mixed sigmoid/Hill discrete-step recurrence (doc/39 O6).

    Mirrors :meth:`helixlang.plugins.runtime.grn.GRN.step` byte-for-byte on the
    noiseless path: each gene either follows the sigmoid-threshold activation
    (``hill_ns[i]`` is ``None``) or Hill kinetics (``hill_ns[i]`` is the Hill
    coefficient, ``kds[i]`` the dissociation constant, falling back to the gene
    threshold when ``kds[i]`` is ``None``).  Returns ``(new_levels, triggered)``.
    """
    n = len(levels)
    acc = [0.0] * n
    for s, d, w in zip(src, dst, weights, strict=True):
        acc[d] += w * levels[s]
    triggered: list[int] = []
    new_levels = [0.0] * n
    for i in range(n):
        hn = hill_ns[i]
        if hn is not None:
            kd = kds[i] if kds[i] is not None else thresholds[i]
            raw = _hill(acc[i], hn, kd)
        else:
            raw = _sigmoid(acc[i] - thresholds[i])
        dec = decays[i] if decays[i] is not None else default_decay
        blended = dec * levels[i] + (1.0 - dec) * raw
        v = blended if blended > 0.0 else 0.0
        if v > 1.0:
            v = 1.0
        new_levels[i] = v
        if v > 0.5:
            triggered.append(i)
    return new_levels, triggered
