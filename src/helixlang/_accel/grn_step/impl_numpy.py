"""Numpy batch implementation of the GRN discrete-step recurrence.

Same numerics as ``impl_python`` (accumulate edge weights, sigmoid threshold,
decay, clip), vectorized over nodes.  This is only a speed switch — it must not
change rounding semantics relative to the reference within float tolerance.
"""
from __future__ import annotations

import numpy as np


def step(levels, src, dst, weights, decays, thresholds, default_decay):
    arr = np.asarray(levels, dtype=float)
    src_i = np.asarray(src, dtype=np.intp)
    dst_i = np.asarray(dst, dtype=np.intp)
    w = np.asarray(weights, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    dec = np.asarray(decays, dtype=float)
    dec = np.where(np.isnan(dec), default_decay, dec)
    acc = np.bincount(dst_i, weights=w * arr[src_i], minlength=len(arr))
    x = acc - thr
    raw = 1.0 / (1.0 + np.exp(-x.clip(min=-700.0, max=700.0)))
    blended = dec * arr + (1.0 - dec) * raw
    new = np.clip(blended, 0.0, 1.0)
    return new.tolist(), np.nonzero(new > 0.5)[0].tolist()


def step_mixed(levels, src, dst, weights, decays, thresholds, default_decay,
               hill_ns, kds):
    """Mixed sigmoid/Hill kernel (doc/39 O6); vectorized, same numerics as
    ``impl_python.step_mixed`` within float tolerance."""
    arr = np.asarray(levels, dtype=float)
    src_i = np.asarray(src, dtype=np.intp)
    dst_i = np.asarray(dst, dtype=np.intp)
    w = np.asarray(weights, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    dec = np.asarray(decays, dtype=float)
    dec = np.where(np.isnan(dec), default_decay, dec)
    acc = np.bincount(dst_i, weights=w * arr[src_i], minlength=len(arr))

    # Hill coefficient per gene: NaN marks the sigmoid path.  kd falls back to
    # the gene threshold where the caller left it None.
    hn = np.asarray([float("nan") if h is None else h for h in hill_ns],
                    dtype=float)
    kd_eff = np.asarray([k if k is not None else thresholds[i]
                         for i, k in enumerate(kds)], dtype=float)

    mask = ~np.isnan(hn)
    pos = acc > 0
    m = mask & pos
    x = acc - thr
    sig = 1.0 / (1.0 + np.exp(-x.clip(min=-700.0, max=700.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = np.where(m, acc, 1.0) ** np.where(m, hn, 1.0)
        kdn = np.where(m, kd_eff, 1.0) ** np.where(m, hn, 1.0)
    hill_term = np.where(kdn <= 0, 1.0, xn / (kdn + xn))
    hill_term = np.where(m, hill_term, 0.0)
    raw = np.where(mask, hill_term, sig)
    blended = dec * arr + (1.0 - dec) * raw
    new = np.clip(blended, 0.0, 1.0)
    return new.tolist(), np.nonzero(new > 0.5)[0].tolist()
