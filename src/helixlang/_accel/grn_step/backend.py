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

Mixed sigmoid/Hill activation (doc/39 O6):

    step_mixed(levels, src, dst, weights, decays, thresholds, default_decay,
               hill_ns, kds)
        -> (new_levels, triggered_indices)

with ``hill_ns[i]`` the Hill coefficient or ``None`` for the sigmoid path and
``kds[i]`` the dissociation constant (``None`` falls back to ``thresholds[i]``).

Aliases are normalized so callers pass explicit ``decay`` and ``threshold``
arrays; None decays are expanded by the Python wrapper before the hot loop.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot
from helixlang._accel.grn_step import impl_python


def step(levels, src, dst, weights, decays, thresholds, default_decay,
         prefer: str | None = None):
    """Alias to the fastest available equivalent-fidelity implementation.

    ``prefer`` pins the backend tag (doc/37 §3.4): ``"python"`` selects the
    byte-identical reference kernel; ``None`` lets the loader pick
    ``native > numpy > python``.
    """
    mod = load_hot("helixlang._accel.grn_step", prefer=prefer)
    return mod.step(levels, src, dst, weights, decays, thresholds, default_decay)


def step_mixed(levels, src, dst, weights, decays, thresholds, default_decay,
               hill_ns, kds, prefer: str | None = None):
    """Alias for the mixed sigmoid/Hill kernel (doc/39 O6).

    Same selection/fidelity contract as :func:`step`; ``prefer`` pins the tag.
    Compiled ``impl_cext``/``impl_cython`` artifacts that predate this hook
    fall back to the byte-identical ``impl_python`` kernel.
    """
    mod = load_hot("helixlang._accel.grn_step", prefer=prefer)
    if hasattr(mod, "step_mixed"):
        return mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                              default_decay, hill_ns, kds)
    return impl_python.step_mixed(levels, src, dst, weights, decays, thresholds,
                                  default_decay, hill_ns, kds)
    return impl_python.step_mixed(levels, src, dst, weights, decays, thresholds,
                                  default_decay, hill_ns, kds)
