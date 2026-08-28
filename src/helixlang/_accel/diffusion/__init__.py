"""Gray-Scott diffusion hot loop (doc/36 P2 / §5).

Isolates the per-step reaction-diffusion update so it can be backed by multiple
equivalent-fidelity implementations.  The math mirrors
:class:`helixlang.plugins.runtime.reaction_diffusion.GrayScott` (5-point Laplacian, U·V²
reaction, clamp to [0,1], borders preserved) so swapping backends never changes
results — a pure speed switch, never a fidelity switch (§3ξ.5).

Public API (shared by every ``impl_*``)::

    step(u, v, F, k, Du, Dv) -> (new_u, new_v)

``u``/``v`` are the concentration fields (2-D list or ndarray); borders are
returned unchanged.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot


def step(u, v, F, k, Du, Dv):
    """Alias to the fastest available equivalent-fidelity implementation."""
    mod = load_hot("helixlang._accel.diffusion")
    return mod.step(u, v, F, k, Du, Dv)
