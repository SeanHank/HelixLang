#!/usr/bin/env python3
"""Benchmark 11: Performance comparison — HelixLang vs COBRApy FBA solve times."""
from __future__ import annotations

import io
import json
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"


def _suppress_tqdm() -> None:
    try:
        import tqdm as _tqdm_mod
        _orig = _tqdm_mod.tqdm
        _tqdm_mod.tqdm = lambda *a, **kw: _orig(*a, **{**kw, "disable": True})
    except ImportError:
        pass


def _load_cobra_model(model_id: str):
    """Load a COBRApy model with stdout suppressed."""
    import cobra
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model = cobra.io.load_model(model_id)
    finally:
        sys.stdout = old_stdout
    model.solver = "glpk"
    ex_rxn = model.reactions.get_by_id("EX_glc__D_e")
    ex_rxn.lower_bound = -10.0
    ex_rxn.upper_bound = 0.0
    return model


def _benchmark_model(model_id: str, n_solves: int = 100) -> dict | None:
    """Benchmark COBRApy vs HelixLang for a single model."""
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis, _from_cobra_model

    _suppress_tqdm()

    try:
        cobra_model = _load_cobra_model(model_id)
    except Exception:
        return None

    # Convert to HelixLang
    helix_model = _from_cobra_model(cobra_model)
    fba = FluxBalanceAnalysis(helix_model)
    fba.set_uptake("glc_e", 10.0)

    # Warm up both solvers (1 solve each, discard)
    cobra_model.optimize()
    fba.solve(objective="biomass")

    # Time COBRApy solves
    t0 = time.perf_counter()
    for _ in range(n_solves):
        cobra_model.optimize()
    cobrapy_time = time.perf_counter() - t0

    # Time HelixLang solves
    t0 = time.perf_counter()
    for _ in range(n_solves):
        fba.solve(objective="biomass")
    helixlang_time = time.perf_counter() - t0

    speedup = cobrapy_time / helixlang_time if helixlang_time > 0 else float("inf")

    return {
        "n_reactions": len(cobra_model.reactions),
        "cobrapy_100_solves_s": round(cobrapy_time, 4),
        "helixlang_100_solves_s": round(helixlang_time, 4),
        "speedup_ratio": round(speedup, 4),
        "cobrapy_avg_ms": round(cobrapy_time / n_solves * 1000, 4),
        "helixlang_avg_ms": round(helixlang_time / n_solves * 1000, 4),
    }


def run() -> dict:
    t0 = time.perf_counter()
    try:
        import cobra  # noqa: F401 — verify cobra is available

        from helixlang.plugins.runtime.metabolism import _from_cobra_model  # noqa: F401

        _suppress_tqdm()

        result: dict = {
            "id": "11_performance_comparison",
            "status": "PASS",
            "ecoli_core": None,
            "iml1515": None,
        }

        # Benchmark e_coli_core
        core = _benchmark_model("e_coli_core")
        if core is None:
            result["status"] = "FAIL"
            result["error"] = "Failed to load e_coli_core"
        else:
            result["ecoli_core"] = core

        # Benchmark iML1515 (optional)
        iml = _benchmark_model("iML1515")
        if iml is not None:
            result["iml1515"] = iml

        result["note"] = "Speedup > 1 means HelixLang is faster"
        result["runtime_seconds"] = time.perf_counter() - t0
        return result

    except ImportError:
        return {
            "id": "11_performance_comparison",
            "status": "SKIP",
            "reason": "COBRApy not installed",
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as exc:
        return {
            "id": "11_performance_comparison",
            "status": "FAIL",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
