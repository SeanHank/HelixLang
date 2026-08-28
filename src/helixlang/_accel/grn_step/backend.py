"""GRN discrete-tick step hot loop (doc/36 P1).

Isolates the per-tick GRN recurrence so it can be backed by multiple
equivalent-fidelity implementations.  The math here must stay byte-for-byte
compatible with :meth:`helixlang.plugins.runtime.grn.GRN.step` (sigmoid-threshold path, no
noise) so swapping backends never changes results.

Public API (shared by every ``impl_*``):

    step(levels, src, dst, weights, decays, thresholds, default_decay)
        -> (new_levels, triggered_indices)

where:
- ``levels``: list[float] length N (current levels)
- ``src/dst/weights``: parallel edge arrays (length E)
- ``decays``: list[float|None] length N (per-gene decay)
- ``thresholds``: list[float] length N
- ``default_decay``: float used where ``decays[i]`` is None

Aliases are normalized so callers pass explicit ``decay`` and ``threshold``
arrays; None decays are expanded by the Python wrapper before the hot loop.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot


def step(levels, src, dst, weights, decays, thresholds, default_decay):
    """Alias to the fastest available equivalent-fidelity implementation."""
    mod = load_hot("helixlang._accel.grn_step")
    return mod.step(levels, src, dst, weights, decays, thresholds, default_decay)
