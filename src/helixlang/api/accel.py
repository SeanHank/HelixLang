"""Accelerated-kernel surface (doc/38 §6.2 ``api.accel``).

Thin, stdlib-only adapter for the strict-fidelity GRN step kernel so plugins
never import the private ``helixlang._accel.*`` paths.  The kernel backend is
resolved lazily inside each call (and preserves the accel layer's fidelity
opt-in gate).
"""
from __future__ import annotations

from typing import Any


def grn_step(*args: Any, **kwargs: Any) -> Any:
    """Run one GRN propagation step through the accelerated equivalent-fidelity
    kernel, if the declared backend is available."""
    from helixlang._accel.grn_step.backend import step
    return step(*args, **kwargs)


def grn_step_mixed(*args: Any, **kwargs: Any) -> Any:
    """Run one GRN step on mixed sigmoid/Hill activation (doc/39 O6).

    Same selection/fidelity contract as :func:`grn_step`; compiled native
    artifacts that predate the hook fall back to the byte-identical
    ``impl_python`` kernel.
    """
    from helixlang._accel.grn_step.backend import step_mixed
    return step_mixed(*args, **kwargs)


__all__ = ["grn_step", "grn_step_mixed"]
