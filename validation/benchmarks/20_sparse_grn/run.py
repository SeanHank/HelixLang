#!/usr/bin/env python3
"""Benchmark 20: Sparse GRN — compare SparseGRN vs dense GRN output."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "20_sparse_grn"}
    try:
        import numpy as np

        from helixlang.grn import GRN
        from helixlang.sparse_grn import SparseGRN

        # ── Build a 10-gene GRN with sparse topology ────────────────
        names = [f"G{i}" for i in range(10)]
        grn = GRN()
        for name in names:
            grn.add_gene(name, threshold=0.5, decay=0.9)

        # Sparse edges: each gene regulated by 1-2 others (12 edges)
        edges = [
            ("G0", "G1", 1.5),
            ("G1", "G2", -1.0),
            ("G2", "G3", 2.0),
            ("G3", "G4", -0.5),
            ("G4", "G5", 1.0),
            ("G5", "G6", -1.5),
            ("G6", "G7", 0.8),
            ("G7", "G8", -1.2),
            ("G8", "G9", 1.0),
            ("G9", "G0", -0.7),
            ("G0", "G5", 0.3),
            ("G3", "G8", 0.6),
        ]
        for src, tgt, w in edges:
            grn.add_edge(src, tgt, w)

        sparse_grn = SparseGRN.from_grn(grn)

        # ── Validation 1: sparse GRN has fewer edges than dense ─────
        n_genes = sparse_grn.n_genes
        n_edges = sparse_grn.n_edges
        max_dense = n_genes * (n_genes - 1)
        fewer_edges = n_edges < max_dense

        # ── Validation 2: both produce same output for same input ───
        rng = np.random.default_rng(42)
        init_levels = rng.random((3, n_genes))  # 3 cells, 10 genes

        # Dense GRN: set levels, step, read back (scalar, per-cell)
        dense_out = np.zeros_like(init_levels)
        for c in range(init_levels.shape[0]):
            for i, name in enumerate(names):
                grn.set_level(name, float(init_levels[c, i]))
            grn.step()
            for i, name in enumerate(names):
                dense_out[c, i] = grn.nodes[name].level

        # Sparse GRN step (vectorized across cells)
        sparse_out = sparse_grn.step(init_levels.copy())

        # Compare (allow small floating-point tolerance)
        max_diff = float(np.max(np.abs(dense_out - sparse_out)))
        outputs_match = max_diff < 1e-6

        # ── Validation 3: sparse GRN round-trips to GRN ─────────────
        reconstructed = sparse_grn.to_grn()
        rt_names = list(reconstructed.nodes.keys())
        rt_ok = rt_names == names

        all_ok = fewer_edges and outputs_match and rt_ok

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok else "FAIL",
            "validation": {
                "fewer_edges_than_dense": fewer_edges,
                "sparse_matches_dense_output": outputs_match,
                "roundtrip_to_grn": rt_ok,
            },
            "topology": {
                "n_genes": n_genes,
                "n_edges_sparse": n_edges,
                "n_edges_dense_max": max_dense,
                "sparsity_ratio": round(n_edges / max_dense, 4),
            },
            "output_comparison": {
                "max_abs_difference": round(max_diff, 10),
                "tolerance": 1e-6,
                "n_cells_tested": init_levels.shape[0],
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
