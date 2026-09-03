#!/usr/bin/env python3
"""Benchmark 79: Complement + cell-granule pools (doc/40 Phase C — G5/G6).

Validates:
  - G5 reduced complement cascade (ComplementCascade): C3→C3b/C3a opsonization,
    C5→C5a/C5b-9 MAC, anti-C5 blocker spares opsonization while suppressing MAC
    (L7 Zewde & Morikis 2018).
  - G6 additive NK/mast/eosinophil/basophil pools with IgE→histamine
    anaphylaxis release (L1 BIS entity set; anaphylaxis mediator spike that
    clears after the trigger).
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
        from helixlang.plugins.human.complement import (
            N_L7_PARAMS,
            ComplementCascade,
            FullL7Complement,
        )
        from helixlang.plugins.human.immune import InnateImmuneModel, create_immune_model

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── G5: signal drives opsonization + MAC ───────────────────────────
        c = ComplementCascade()
        for _ in range(24):
            c.step(1.0, 0.8)
        details["opsonization"] = c.get_opsonization()
        details["c3a"] = c.get_c3a()
        details["mac"] = c.get_mac()
        details["c3"] = c.c3
        checks["signal_drives_opsonization_and_mac"] = (
            c.get_opsonization() > 0.0 and c.get_c3a() > 0.0
            and c.get_mac() > 0.0 and c.c3 < 1.0)

        # ── G5: anti-C5 suppresses MAC, spares opsonization ────────────────
        placebo = ComplementCascade()
        for _ in range(24):
            placebo.step(1.0, 0.8)
        treated = ComplementCascade()
        treated.anti_c5_dose = 1.0
        for _ in range(24):
            treated.step(1.0, 0.8)
        details["placebo_mac"] = placebo.get_mac()
        details["treated_mac"] = treated.get_mac()
        details["treated_opsonization"] = treated.get_opsonization()
        checks["anti_c5_suppresses_mac_spares_opsonization"] = (
            treated.get_mac() < placebo.get_mac() * 0.1
            and treated.get_opsonization() > 0.0)
        # Full L7 network is importable and exposes the real parameter table
        # (61 dynamics-referenced constants; no inert placeholder padding).
        full = FullL7Complement()
        details["l7_n_params"] = float(N_L7_PARAMS)
        checks["full_l7_complement_importable"] = (
            N_L7_PARAMS >= 50 and full.n_params() == N_L7_PARAMS
            and all(not k.startswith("rate_") for k in full.p))

        # ── G6: NK rises with innate signal ────────────────────────────────
        m, _crp = create_immune_model()
        m.infection_severity = 0.8
        for _ in range(24):
            m.step(1.0)
        details["nk_24h"] = m.get_nk_cells()
        checks["nk_rises_with_innate_signal"] = m.get_nk_cells() > 0.25

        # ── G6: IgE drive → histamine release then clearance ───────────────
        an = InnateImmuneModel()
        an.cells.igE_signal = 1.0
        peak = 0.0
        for _ in range(24):
            an.step(1.0)
            peak = max(peak, an.get_histamine())
        details["histamine_peak"] = peak
        checks["ige_drive_releases_histamine"] = peak > 5.0

        an.cells.igE_signal = 0.0
        for _ in range(24):
            an.step(1.0)
        details["histamine_after_clear"] = an.get_histamine()
        checks["histamine_clears_after_signal_removed"] = an.get_histamine() < 5.0

        all_pass = all(checks.values())
        return {
            "id": "79_immune_complement",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "doc/40 L7 (Zewde & Morikis 2018), L1 (BIS entity set)",
                "doi": "10.1371/journal.pone.0198644; 10.1186/1742-4682-8-9",
                "note": "reduced complement cascade + anti-C5 PD; NK/mast/eosinophil/basophil pools and histamine anaphylaxis",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "79_immune_complement",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
