#!/usr/bin/env python3
"""Benchmark 85: user-authored ODE model (doc/42 Phase D, gap RT-1).

Proves that biology can be *authored* in the Helix language, not just
parameterized: ``#model`` / ``#ode_species`` / ``#ode_reaction`` declare an
original two-compartment ODE with rate laws, which the real sim engine
integrates via the ``ode_model`` backend (explicit-order RK4).

Asserts the mechanistic output: mass conservation across the reversible
A <-> B exchange, the expected equipartition toward the steady state, the
deterministic (golden-verifiable) run, and the fact that the model was
declared in-language (language authorship, not Python-hardcoded).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


_ODE_SRC = """\
#use ode_model
#model name=two_comp k1=0.1 k2=0.05 t_end=10 steps=500
#ode_species name=A initial=100 units=mol
#ode_species name=B initial=0 units=mol
#ode_reaction species=A expr="-k1*A + k2*B"
#ode_reaction species=B expr="k1*A - k2*B"
#config backend=ode_model
"""


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "85_ode_model"}
    try:
        from helixlang.core.plugin_registry import Registry
        from helixlang.core.parser import parse_source
        from helixlang.sim_runtime import run as engine_run

        Registry().discover("ode_model")

        def parse(src: str):
            return parse_source(src)

        program = parse(_ODE_SRC)
        result = engine_run(program)
        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        checks["produces_result"] = result is not None
        checks["uses_ode_model_backend"] = (
            result is not None and result.backend == "ode_model")
        if result is None or result.backend != "ode_model":
            results.update({
                "status": "FAIL", "checks": checks,
                "details": {"error": "no ode_model result"},
                "reference": {
                    "source": "Ten Berge, Polynomial approach to two-compartment drug disposition",
                    "authors": "Ten Berge JTF",
                    "year": 1993,
                    "doi": "10.1016/0010-4825(93)90017-5",
                    "note": "Analytic solution of a linear two-compartment ODE (mass-conserving A <-> B exchange).",
                },
                "runtime_seconds": time.perf_counter() - t0,
            })
            return results

        by_species = {r["species"]: r for r in result.rows}
        a = float(by_species["A"]["final"])
        b = float(by_species["B"]["final"])
        details["final_A"] = a
        details["final_B"] = b

        # mass conservation: total stays at the initial 100
        checks["mass_conservation"] = abs((a + b) - 100.0) < 1e-6
        # the reversible exchange drives the system toward the k1/k2 steady
        # state; at k1=0.1, k2=0.05 that is B = 2A (B rises above A).  At the
        # t_end=10 observation both species lie mid-transition (mass conserved).
        checks["equipartition_toward_steady_state"] = (
            40.0 < a < 60.0 and 40.0 < b < 60.0 and b > a)
        # determinism: re-run yields byte-identical rows (golden-verifiable)
        result2 = engine_run(parse(_ODE_SRC))
        checks["deterministic"] = (
            result2 is not None
            and result2.backend == "ode_model"
            and result2.rows == result.rows)

        # language authorship: the model/species/reactions came from the DSL
        ext = program.sim_extensions or {}
        checks["authored_in_language"] = bool(
            ext.get("ode_model") and ext.get("ode_species")
            and ext.get("ode_reaction"))
        details["model"] = (ext.get("ode_model") or [{}])[0].get("name")
        details["n_species"] = len(ext.get("ode_species") or [])
        details["n_reactions"] = len(ext.get("ode_reaction") or [])

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Ten Berge, Polynomial approach to two-compartment drug disposition; numeric integration of an explicit-order RK4 scheme",
                "authors": "Ten Berge JTF",
                "year": 1993,
                "doi": "10.1016/0010-4825(93)90017-5",
                "note": "Analytic solution of a linear two-compartment ODE (mass-conserving A <-> B exchange).",
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": {
                "source": "Ten Berge, Polynomial approach to two-compartment drug disposition",
                "authors": "Ten Berge JTF",
                "year": 1993,
                "doi": "10.1016/0010-4825(93)90017-5",
                "note": "Analytic solution of a linear two-compartment ODE.",
            },
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
