"""Accelerated-kernel surface (doc/38 §6.2 ``api.accel``).

Thin, stdlib-only adapter for the strict-fidelity GRN step kernel so plugins
never import the private ``helixlang._accel.*`` paths.  The kernel backend is
resolved lazily inside each call (and preserves the accel layer's fidelity
opt-in gate).
"""
from __future__ import annotations

from typing import Any, cast


def grn_step(levels: Any, src: Any, dst: Any, weights: Any, decays: Any,
             thresholds: Any, default_decay: Any,
             prefer: str | None = None) -> Any:
    """Run one GRN propagation step through the accelerated equivalent-fidelity
    kernel, if the declared backend is available.

    ``prefer`` pins the backend tag (e.g. ``"python"`` for bit-identical numerics
    — doc/37 §3.4; ``None`` lets the loader pick ``native > numpy > python``).
    """
    from helixlang._accel.grn_step.backend import step
    return step(levels, src, dst, weights, decays, thresholds, default_decay,
                prefer=prefer)


def grn_step_mixed(levels: Any, src: Any, dst: Any, weights: Any, decays: Any,
                   thresholds: Any, default_decay: Any, hill_ns: Any, kds: Any,
                   prefer: str | None = None) -> Any:
    """Run one GRN step on mixed sigmoid/Hill activation (doc/39 O6).

    Same selection/fidelity contract as :func:`grn_step`; ``prefer`` pins the
    backend tag.  Compiled native artifacts that predate the hook fall back to
    the byte-identical ``impl_python`` kernel.
    """
    from helixlang._accel.grn_step.backend import step_mixed
    return step_mixed(levels, src, dst, weights, decays, thresholds,
                      default_decay, hill_ns, kds, prefer=prefer)


def simplex_run(tableau: Any, basis: Any, obj: Any, n_vars: int,
                eps: float = 1e-9, max_iter: int = 10000,
                forbidden: Any | None = None) -> str:
    """Run a full two-phase simplex pivot loop through the accelerated
    equivalent-fidelity kernel (doc/42 Phase C PF-2).

    ``tableau`` (n_rows x n_vars+1, RHS last column) and ``basis`` are mutated
    in place; returns ``"optimal" | "unbounded" | "max_iter"``.  The kernel is
    selected by the accel loader ("native > numpy > python").  Native compiled
    pigeon-holes to the reference numerics to within ~1e-14, so callers that
    require *bit-identical* accepted-state results (e.g. golden-verifiable FBA)
    should keep the reference ``simplex`` loop and switch explicitly.
    """
    from helixlang._accel.simplex import run
    return cast(str, run(tableau, basis, obj, n_vars, eps, max_iter, forbidden))


__all__ = ["grn_step", "grn_step_mixed", "simplex_run"]
