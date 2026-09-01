#!/usr/bin/env python3
"""Benchmark 84: Cardiology closed-loop cardiac cycle (doc/42 Phase E-2).

Drives a ``#cardiac_cycle`` annotation end-to-end through the real sim engine
(``sim_runtime.run`` with the ``cardiology`` backend), so the directive produces
actual hemodynamic output from the Phase B RL-1 closed-loop cardiovascular core
rather than only validating a period.

Asserts the mechanistic outputs against literature (Guyton & Hall) and verifies
the run is deterministic (two identical runs → golden-verifiable).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "84_cardiac_cycle"}
    try:
        from helixlang.core.plugin_registry import Registry
        from helixlang.core.parser import parse_source
        from helixlang.sim_runtime import run as engine_run

        # register the bundled cardiology grammar so the lexer treats
        # #cardiac_cycle as an annotation (the CLI does this at startup).
        Registry().discover("cardiology")

        def parse(src: str):
            return parse_source(src)

        src = (
            "#use cardiology\n"
            "#cardiac_cycle period=0.8 conduction=normal\n"
            "#config backend=cardiology\n"
        )

        result = engine_run(parse(src))
        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        checks["produces_result"] = result is not None
        checks["uses_cardiology_backend"] = (
            result is not None and result.backend == "cardiology")
        if result is None or result.backend != "cardiology":
            results.update({
                "status": "FAIL", "checks": checks,
                "details": {"error": "no cardiology result"}, "runtime_seconds": time.perf_counter() - t0,
            })
            return results

        row = result.rows[0]
        co = float(row["cardiac_output_l_min"])
        hr = float(row["heart_rate_bpm"])
        sv = float(row["stroke_volume_ml"])
        ci = float(row["cardiac_index_l_min_m2"])
        map_mmhg = float(row["map_mmhg"])
        sbp = float(row["systolic_bp_mmhg"])
        dbp = float(row["diastolic_bp_mmhg"])
        details["row"] = {k: row[k] for k in row}

        # flow balance: CO = HR x SV / 1000 (mass/flow-balanced, not target-tracking);
        # tolerance accommodates the 3-decimal rounding of CO in the result row.
        checks["co_from_flow_balance"] = abs(co - hr * sv / 1000.0) < 1e-3
        # cardiac index in the normal adult range (2.5 - 4.0 L/min/m2)
        checks["ci_normal_range"] = 2.5 <= ci <= 4.0
        # healthy resting HR
        checks["hr_physiologic"] = 55.0 <= hr <= 120.0
        # MAP consistent with a functioning pressure/flow balance
        checks["map_physiologic"] = 70.0 <= map_mmhg <= 110.0
        # SBP > DBP always (Windkessel pulse pressure)
        checks["sbp_gt_dbp"] = sbp > dbp and sbp > 90.0 and dbp > 40.0

        # determinism: re-run yields byte-identical rows
        result2 = engine_run(parse(src))
        checks["deterministic"] = (
            result2 is not None
            and result2.backend == "cardiology"
            and result2.rows == result.rows)
        details["hr_bpm"] = round(hr, 2)
        details["co_l_min"] = round(co, 3)
        details["map_mmhg"] = round(map_mmhg, 2)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Guyton & Hall, Textbook of Medical Physiology (closed-loop cardiovascular control)",
                "authors": "Guyton AC, Hall JE",
                "year": 2016,
                "doi": "10.1016/B978-0-323-39335-5.00001-1",
                "note": "Closed-loop cardiac output / MAP regulation from the Frank-Starling and baroreflex control of the circulation (Guyton & Hall, 13th ed.).",
            },
            "experimental_comparison": {
                "cardiac_index_l_min_m2": {
                    "reference_min": 2.5,
                    "reference_max": 4.0,
                    "unit": "L/min/m2",
                    "note": "Normal adult cardiac index.",
                },
                "heart_rate_bpm": {
                    "reference_min": 55.0,
                    "reference_max": 120.0,
                    "unit": "bpm",
                    "note": "Normal resting adult heart rate.",
                },
                "mean_arterial_pressure_mmhg": {
                    "reference_min": 70.0,
                    "reference_max": 110.0,
                    "unit": "mmHg",
                    "note": "Normal resting mean arterial pressure.",
                },
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": {
                "source": "Guyton & Hall, Textbook of Medical Physiology",
                "authors": "Guyton AC, Hall JE",
                "year": 2016,
                "doi": "10.1016/B978-0-323-39335-5.00001-1",
                "note": "Closed-loop cardiovascular control.",
            },
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
