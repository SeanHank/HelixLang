"""Numpy batch implementation of the Gray-Scott diffusion update.

Same numerics as ``impl_python`` (5-point Laplacian, U·V² reaction, clamp,
borders preserved), vectorized over the field.
"""
from __future__ import annotations

import numpy as np


def step(u, v, F, k, Du, Dv):
    """See ``backend`` docstring.  ``u``/``v`` may be lists or ndarrays."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    lap_u = np.zeros_like(u)
    lap_v = np.zeros_like(v)
    lap_u[1:-1, 1:-1] = (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        - 4.0 * u[1:-1, 1:-1]) * 0.25
    lap_v[1:-1, 1:-1] = (
        v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]
        - 4.0 * v[1:-1, 1:-1]) * 0.25
    uvv = u * v * v
    new_u = np.clip(u + (Du * lap_u - uvv + F * (1.0 - u)), 0.0, 1.0)
    new_v = np.clip(v + (Dv * lap_v + uvv - (F + k) * v), 0.0, 1.0)
    new_u[0, :] = u[0, :]
    new_u[-1, :] = u[-1, :]
    new_u[:, 0] = u[:, 0]
    new_u[:, -1] = u[:, -1]
    new_v[0, :] = v[0, :]
    new_v[-1, :] = v[-1, :]
    new_v[:, 0] = v[:, 0]
    new_v[:, -1] = v[:, -1]
    return new_u, new_v
