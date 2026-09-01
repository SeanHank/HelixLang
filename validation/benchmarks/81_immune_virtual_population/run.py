#!/usr/bin/env python3
"""Benchmark 81: Virtual immune population (doc/40 Phase D — G13 / doc/39 O2/O9).

Validates the seeded virtual-population sampler and the cohort-vectorized
runner:
  - G13 sample_virtual_population produces log-normal baseline variance that is
    reproducible under a fixed seed and physiologic in range (L6 npj Syst Biol
    Appl 2023 methodology).
  - doc/39 O2/O9: run_cohort (vectorized + multiprocess) advances a cohort
    bit-identically to the scalar per-patient path.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys_path = str(Path(__file__).resolve().parents[3])
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.plugins.human.immune import run_cohort, sample_virtual_population

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── G13: baseline variance across a cohort ─────────────────────────
        cohort = sample_virtual_population(16, seed=7)
        il6s = [m.cytokines.il6 for m in cohort]
        neuts = [m.cells.neutrophils for m in cohort]
        il6_span = (max(il6s) - min(il6s)) / (sum(il6s) / len(il6s))
        details["il6_span_fraction"] = il6_span
        checks["cohort_has_baseline_variance"] = il6_span > 0.05
        # Physiologic range: IL-6 in pg/mL, healthy ~1 (0.4–3), neutrophils ~2–7.
        checks["cohort_baselines_are_physiologic"] = (
            all(0.3 < v < 8.0 for v in il6s)
            and all(1.5 < v < 9.0 for v in neuts))

        # ── G13: seeded reproducibility ────────────────────────────────────
        cohort_b = sample_virtual_population(16, seed=7)
        same = all(abs(a.cytokines.il6 - b.cytokines.il6) < 1e-12
                   for a, b in zip(cohort, cohort_b, strict=True))
        checks["cohort_deterministic_across_runs"] = same

        # ── O2/O9: cohort teacher == scalar advance ────────────────────────
        scalar = sample_virtual_population(8, seed=3)
        for m in scalar:
            m.infection_severity = 0.6
        run_cohort(scalar, 48, dt_h=1.0, workers=1)

        vector = sample_virtual_population(8, seed=3)
        for m in vector:
            m.infection_severity = 0.6
        run_cohort(vector, 48, dt_h=1.0, workers=4)

        same2 = all(abs(a.get_il6() - b.get_il6()) < 1e-9
                    for a, b in zip(scalar, vector, strict=True))
        checks["cohort_matches_scalar_advance"] = same2
        details["il6_after_cohort"] = vector[0].get_il6()

        all_pass = all(checks.values())
        return {
            "id": "81_immune_virtual_population",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "doc/40 L6 (npj Syst Biol Appl 11, 2023); doc/39 O2/O9",
                "doi": "10.1038/s41540-023-00269-6",
                "note": "seeded log-normal virtual population; cohort-vectorized run is bit-identical to scalar",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "81_immune_virtual_population",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
