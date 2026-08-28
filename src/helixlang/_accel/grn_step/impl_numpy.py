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
