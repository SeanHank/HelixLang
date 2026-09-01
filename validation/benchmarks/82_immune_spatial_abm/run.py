#!/usr/bin/env python3
"""Benchmark 82: Spatial immune agent-based model (doc/40 Phase F — G15).

Validates :mod:`helixlang.plugins.human.spatial_abm` under the numpy backend
(always available; the optional jax kernel is a performance path, per doc/40):
  - agent seeding matches configuration counts;
  - steps advance chemokine diffusion and migration;
  - chemokine-gradient-guided migration + contact-dependent T-cell/APC
    signaling activates T cells;
  - deterministic replay under a fixed seed (doc/39 §5 determinism).
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
        from helixlang.plugins.human.spatial_abm import (
            SpatialABMConfig,
            SpatialAgentGrid,
            run_spatial_abm,
        )

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── Seeding matches config ─────────────────────────────────────────
        cfg = SpatialABMConfig(seed=0)
        g = SpatialAgentGrid(cfg)
        counts = g.cell_counts()
        details["n_tcell"] = float(counts["tcell"])
        details["n_apc"] = float(counts["apc"])
        checks["agent_counts_match_config"] = (
            counts["tcell"] == cfg.n_tcell and counts["apc"] == cfg.n_apc
            and len(g.agents) == 60 + 12 + 40 + 20 + 15)

        # ── Stepping advances + bulk chemokine diffuses above 0 ────────────
        g2 = SpatialAgentGrid(SpatialABMConfig(seed=1, max_steps=10))
        for _ in range(5):
            g2.step()
        details["step_index"] = float(g2.step_index)
        details["mean_chemokine"] = g2.mean_chemokine()
        checks["step_advances_and_diffuses"] = g2.step_index == 5 and g2.mean_chemokine() > 0.0

        # ── Chemokine-guided migration: cells approach the wound source ────
        g3 = SpatialAgentGrid(SpatialABMConfig(seed=2, max_steps=40,
                                               tissue_heterogeneity=True))
        cx, cy = g3.W // 2, g3.H // 2
        d0 = sum(abs(a.x - cx) + abs(a.y - cy) for a in g3.agents) / max(1, len(g3.agents))
        for _ in range(20):
            g3.step()
        d1 = sum(abs(a.x - cx) + abs(a.y - cy) for a in g3.agents) / max(1, len(g3.agents))
        details["mean_dist_initial"] = d0
        details["mean_dist_after_migration"] = d1
        checks["chemokine_guides_migration"] = d1 < d0

        # ── Contact signaling activates T cells ────────────────────────────
        g4 = run_spatial_abm(SpatialABMConfig(seed=4, max_steps=30), steps=20)
        details["activated_tcells"] = float(g4.activated_tcells())
        details["cell_counts"] = float(sum(g4.cell_counts().values()))
        checks["contact_signaling_activates_tcells"] = g4.activated_tcells() > 0

        # ── Deterministic replay ───────────────────────────────────────────
        a = SpatialAgentGrid(SpatialABMConfig(seed=11, max_steps=12))
        b = SpatialAgentGrid(SpatialABMConfig(seed=11, max_steps=12))
        for _ in range(8):
            a.step()
            b.step()
        checks["deterministic_replay"] = a.state_histogram() == b.state_histogram()

        all_pass = all(checks.values())
        return {
            "id": "82_immune_spatial_abm",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "doc/40 G15; doc/31 §2.4 (BIS agent taxonomy, spatial ABM design)",
                "doi": "10.1186/1742-4682-8-9",
                "note": "agent grid + chemokine diffusion + contact signaling, deterministic replay under numpy backend",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "82_immune_spatial_abm",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
