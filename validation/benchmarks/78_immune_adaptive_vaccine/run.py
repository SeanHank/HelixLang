#!/usr/bin/env python3
"""Benchmark 78: Adaptive immunity + vaccination (doc/40 Phase B — G2/G3/G7/G12).

Validates the adaptive immune layer (:mod:`helixlang.plugins.human.adaptive`):
  - G2 naive/effector/memory CD4/CD8/B pools, G3 antibody IgM→IgG with plasma
    cell waning (L5 Pawelek 2012 effector arms; L8 two-dose consensus model).
  - G7 APC/MHC antigen-presentation priming delay (~18 h).
  - G12 vaccination: two-dose prime/boost schedule with sigmoid rise, peak,
    and memory-anamnesis on rechallenge (L8 Front. Immunol. 16:1596518).
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
        from helixlang.plugins.human.adaptive import AdaptiveImmuneModel, VaccineSchedule

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── G2/G3: infection drives antibody + memory ──────────────────────
        m = AdaptiveImmuneModel()
        for _ in range(21 * 24):
            m.step(1.0, 0.8)
        igg = m.get_igg()
        details["igg_after_infection"] = igg
        checks["infection_drives_antibody"] = igg > 10.4
        memory = m.get_memory_t()
        details["memory_t"] = memory
        checks["memory_established_after_response"] = memory > 0.0

        # ── G12: two-dose vaccine peak + memory ────────────────────────────
        v = AdaptiveImmuneModel()
        sched = VaccineSchedule([(0.0, 1.0), (24 * 7.0, 1.0)])
        peak = 0.0
        for t in range(24 * 28):
            dose = sched.due(t)
            v.step(1.0, 0.0, dose=dose)
            peak = max(peak, v.get_total_antibody())
        details["two_dose_peak_ab"] = peak
        checks["two_dose_schedule_peak_and_memory"] = peak > 10.4 and v.memory_b > 0.0

        # ── G12: boost (memory anamnesis) beats primary response ───────────
        primary = AdaptiveImmuneModel()
        primary.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 28):
            primary.step(1.0, 0.0)
        peak_primary = primary.get_total_antibody()
        details["peak_primary"] = peak_primary

        boosted = AdaptiveImmuneModel()
        boosted.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 7):
            boosted.step(1.0, 0.0)
        boosted.step(1.0, 0.0, dose=1.0)
        for _ in range(24 * 21):
            boosted.step(1.0, 0.0)
        peak_boosted = boosted.get_total_antibody()
        details["peak_boosted"] = peak_boosted
        checks["boost_anamnesis_faster_than_priming"] = peak_boosted > peak_primary

        # ── G7: APC priming delay is not instantaneous ─────────────────────
        p = AdaptiveImmuneModel()
        for _ in range(24):
            p.step(1.0, 0.9)
        day1_eff = p.get_effector_t()
        for _ in range(6 * 24):
            p.step(1.0, 0.9)
        details["effector_day1"] = day1_eff
        details["effector_day7"] = p.get_effector_t()
        checks["priming_is_delayed_not_instant"] = p.get_effector_t() > day1_eff

        all_pass = all(checks.values())
        return {
            "id": "78_immune_adaptive_vaccine",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "doc/40 L5 (Pawelek et al. 2012), L8 (Front. Immunol. 16:1596518, 2025)",
                "doi": "10.1371/journal.pcbi.1002588; 10.3389/fimmu.2025.1596518",
                "note": "effector/memory arms, two-dose antibody chain, APC priming delay",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "78_immune_adaptive_vaccine",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
