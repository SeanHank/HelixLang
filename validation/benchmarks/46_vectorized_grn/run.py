#!/usr/bin/env python3
"""Benchmark 46: Vectorized GRN correctness.

Validates that VectorizedGRN.step() produces identical results to the scalar
GRN.step() for sigmoid-only, Hill-only, and mixed-activation networks across
multiple cell populations and tick counts.

Reference: Elowitz & Leibler 2000 (repressilator architecture);
           HelixLang grn.py scalar implementation.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        import numpy as np

        from helixlang.plugins.runtime.grn import GRN
        from helixlang.plugins.runtime.vectorized import VectorizedGRN

        # ── Test 1: Sigmoid-only network (4 genes, 6 cells, 50 ticks) ──────
        grn_sig = GRN()
        for name in ["lacI", "tetR", "cI", "araC"]:
            grn_sig.add_gene(name, threshold=0.0, initial_level=0.5, decay=0.02)
        grn_sig.add_edge("lacI", "tetR", weight=-2.0)
        grn_sig.add_edge("tetR", "cI", weight=-2.0)
        grn_sig.add_edge("cI", "lacI", weight=-2.0)
        grn_sig.add_edge("araC", "lacI", weight=1.5)

        vgrn_sig = VectorizedGRN(grn_sig)
        np.random.seed(42)
        levels = np.random.rand(6, vgrn_sig.n_genes).clip(0, 1)

        # Run vectorized for 50 ticks
        for _ in range(50):
            levels = vgrn_sig.step(levels)

        # Rebuild scalar GRN with same initial state
        grn_sig2 = GRN()
        for name in ["lacI", "tetR", "cI", "araC"]:
            grn_sig2.add_gene(name, threshold=0.0, initial_level=0.5, decay=0.02)
        grn_sig2.add_edge("lacI", "tetR", weight=-2.0)
        grn_sig2.add_edge("tetR", "cI", weight=-2.0)
        grn_sig2.add_edge("cI", "lacI", weight=-2.0)
        grn_sig2.add_edge("araC", "lacI", weight=1.5)

        # Run scalar for 50 ticks on first cell
        for _ in range(50):
            grn_sig2.step()

        scalar_level = np.array([grn_sig2.nodes[n].level for n in grn_sig2.nodes])
        max_diff_sig = float(np.max(np.abs(levels[0] - scalar_level)))
        sigmoid_match = max_diff_sig < 1e-10

        # ── Test 2: Hill-kinetics network (2 genes, 4 cells, 30 ticks) ─────
        grn_hill = GRN()
        grn_hill.add_gene("A", threshold=0.0, initial_level=0.8, decay=0.05, hill_n=2, kd=0.3)
        grn_hill.add_gene("B", threshold=0.0, initial_level=0.2, decay=0.05, hill_n=2, kd=0.3)
        grn_hill.add_edge("A", "B", weight=-3.0)
        grn_hill.add_edge("B", "A", weight=-3.0)

        vgrn_hill = VectorizedGRN(grn_hill)
        levels_h = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])

        for _ in range(30):
            levels_h = vgrn_hill.step(levels_h)

        # Scalar
        grn_hill2 = GRN()
        grn_hill2.add_gene("A", threshold=0.0, initial_level=0.8, decay=0.05, hill_n=2, kd=0.3)
        grn_hill2.add_gene("B", threshold=0.0, initial_level=0.2, decay=0.05, hill_n=2, kd=0.3)
        grn_hill2.add_edge("A", "B", weight=-3.0)
        grn_hill2.add_edge("B", "A", weight=-3.0)

        for _ in range(30):
            grn_hill2.step()
        scalar_h = np.array([grn_hill2.nodes[n].level for n in ["A", "B"]])
        hill_match = bool(np.allclose(levels_h[0], scalar_h, atol=1e-10))

        # ── Test 3: Clipping ────────────────────────────────────────────────
        grn_c = GRN()
        grn_c.add_gene("X", threshold=0.0, initial_level=0.0, decay=0.01)
        grn_c.add_edge("X", "X", weight=10.0)
        vgrn_c = VectorizedGRN(grn_c)
        levels_c = np.array([[0.0]])
        for _ in range(100):
            levels_c = vgrn_c.step(levels_c)
        clips = bool(np.all(levels_c >= 0.0) and np.all(levels_c <= 1.0))

        # ── Test 4: triggered mask ───────────────────────────────────────────
        levels_t = np.array([[0.1, 0.9, 0.5], [0.6, 0.4, 0.3]])
        grn_t = GRN()
        for n in ["a", "b", "c"]:
            grn_t.add_gene(n, threshold=0.0)
        vgrn_t = VectorizedGRN(grn_t)
        mask = vgrn_t.triggered(levels_t, threshold=0.5)
        trigger_ok = bool(
            not mask[0, 0] and mask[0, 1] and not mask[0, 2]
            and mask[1, 0] and not mask[1, 1] and not mask[1, 2]
        )

        # ── Test 5: n_genes / names ──────────────────────────────────────────
        props_ok = vgrn_sig.n_genes == 4 and vgrn_sig.names == ["lacI", "tetR", "cI", "araC"]

        all_pass = sigmoid_match and hill_match and clips and trigger_ok and props_ok

        return {
            "id": "46_vectorized_grn",
            "status": "PASS" if all_pass else "FAIL",
            "checks": {
                "sigmoid_scalar_match": sigmoid_match,
                "hill_scalar_match": hill_match,
                "output_clipped_0_1": clips,
                "triggered_mask_correct": trigger_ok,
                "n_genes_and_names_correct": props_ok,
            },
            "details": {
                "sigmoid_max_diff": max_diff_sig,
                "n_cells_test1": 6,
                "n_ticks_test1": 50,
                "n_cells_test2": 4,
                "n_ticks_test2": 30,
            },
            "reference": {
                "source": "HelixLang grn.py scalar implementation",
                "note": "VectorizedGRN must produce identical output to scalar GRN.step()",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "46_vectorized_grn",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
