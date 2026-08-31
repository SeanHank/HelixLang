#!/usr/bin/env python3
"""Benchmark 45: Provenance — the doc/41 §7 unified 8-field contract.

Every simulation result post-run carries the normative Model Provenance
record: ``source_hash``, ``model_version``, ``parameter_set``,
``literature_references``, ``backend_implementation``, ``solver``,
``random_seed``, ``fidelity_mode`` — plus the legacy doc/34 fields that
benchmark 16 relies on (kept as an identical superset per doc/41 §7.2.4).

This benchmark runs a real FBA program through the production engine
(``helixlang.sim_runtime.run``), so the engine-level auto-attach is what is
actually being validated — not a synthetic provenance dict.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

CONTRACT_KEYS = (
    "source_hash",
    "model_version",
    "parameter_set",
    "literature_references",
    "backend_implementation",
    "solver",
    "random_seed",
    "fidelity_mode",
)

# Benchmark 16's required set (16_cli_server_provenance/run.py:23) stays a
# subset of every provenance dict produced here.
LEGACY_SUPERSET_KEYS = (
    "helix_version",
    "seed",
    "backend",
    "parameters",
    "dependencies",
    "timestamp",
    "source_hash",
)


def _provenance_of(result: Any) -> dict | None:
    prov = getattr(result, "provenance", None)
    if prov is None and isinstance(result, dict):
        prov = result.get("provenance")
    return prov if isinstance(prov, dict) else None


def _stable(prov: dict) -> dict:
    """Provenance minus the wall-clock timestamp (doc/41 §7.3 byte-identity)."""
    p = dict(prov)
    p.pop("timestamp", None)
    return p


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.core.lexer import Lexer
        from helixlang.core.parser import Parser
        from helixlang.core.provenance import complete_provenance
        from helixlang.sim_runtime import run as engine_run

        def parse(src: str):
            return Parser(list(Lexer(src).tokens())).parse()

        fba_src = (
            "#media nutrient=GLC concentration=10.0\n"
            "#config backend=fba\n"
            "#config seed=42\n"
            "#config output=BIOMASS,EX_glc\n"
        )

        # ── S1: production engine auto-attach on a real FBA run ────────────
        result = engine_run(parse(fba_src))
        checks["engine_run_produces_fba_result"] = (
            result is not None and result.backend == "fba")
        fba_prov = _provenance_of(result)
        checks["fba_result_has_provenance"] = fba_prov is not None
        details["fba_provenance"] = fba_prov
        if fba_prov is not None:
            missing = [k for k in CONTRACT_KEYS if k not in fba_prov]
            checks["provenance_has_8_field_contract"] = not missing
            details["missing_contract_keys"] = missing

            superset_missing = [k for k in LEGACY_SUPERSET_KEYS if k not in fba_prov]
            checks["provenance_has_benchmark16_superset"] = not superset_missing
            details["missing_legacy_keys"] = superset_missing

            checks["model_version_field_present"] = "model_version" in fba_prov
            details["model_version"] = fba_prov.get("model_version", "")

            ps = fba_prov.get("parameter_set")
            checks["parameter_set_has_fingerprint"] = (
                isinstance(ps, dict) and ps.get("fields") is not None
                and str(ps.get("fingerprint", "")).startswith("sha256:"))
            details["parameter_set_fingerprint_prefix"] = (
                str(ps.get("fingerprint", ""))[:11] if isinstance(ps, dict) else "")

            bi = fba_prov.get("backend_implementation")
            checks["backend_impl_descriptor"] = (
                isinstance(bi, dict) and bi.get("name") == "fba"
                and isinstance(bi.get("native"), bool))
            details["backend_implementation"] = bi

            checks["solver_field_present"] = "solver" in fba_prov
            details["solver"] = fba_prov.get("solver")

            rs = fba_prov.get("random_seed")
            checks["random_seed_tracks_seed"] = (
                isinstance(rs, dict) and rs.get("seed") == 42)
            details["random_seed"] = rs

            checks["fidelity_mode_defaults_full"] = (
                fba_prov.get("fidelity_mode") == "full")
            details["fidelity_mode"] = fba_prov.get("fidelity_mode")

        # ── S2: identical runs are byte-identical (modulo timestamp) ───────
        if fba_prov is not None:
            result2 = engine_run(parse(fba_src))
            prov2 = _provenance_of(result2)
            checks["identical_runs_byte_identical_provenance"] = (
                prov2 is not None and _stable(fba_prov) == _stable(prov2))

        # ── S3: complete_provenance fills the contract, preserves values ───
        existing = {"backend": "custom", "fidelity": "reduced"}
        merged = complete_provenance(existing, seed=1, backend_name="custom")
        checks["complete_preserves_existing_values"] = (
            merged["backend"] == "custom" and merged["fidelity"] == "reduced")
        checks["complete_fills_contract"] = all(k in merged for k in CONTRACT_KEYS)

        # ── S4: parameter-set fingerprint is order-stable ──────────────────
        a = complete_provenance({}, parameters={"x": 1, "y": 2})
        b = complete_provenance({}, parameters={"y": 2, "x": 1})
        checks["parameter_set_fingerprint_stable"] = (
            a["parameter_set"]["fingerprint"] == b["parameter_set"]["fingerprint"])

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "45_provenance_completeness",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "45_provenance_completeness",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
