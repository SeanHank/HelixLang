#!/usr/bin/env python3
"""Benchmark 51: 3D L-system morphogenesis.

Validates morphology_3d.py:
  - Rodrigues rotation correctness (known angles)
  - Branching: F[F]F produces 3 line segments
  - Preset tree3d produces 3D branching (z-extent > 0)
  - Bounding box correctness
  - Iteration scaling (more iterations -> more lines)

Reference: Lindenmayer A 1968, J Theor Biol 18:280-299;
           Prusinkiewicz & Lindenmayer 1990, The Algorithmic Beauty of Plants.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.plugins.runtime.morphology_3d import (
            LSystem3D,
            PLANT_PRESETS,
            Point3D,
            rotate_vector,
        )

        checks: dict[str, bool] = {}
        details: dict[str, float | dict] = {}

        # ── Test 1: Rodrigues rotation — 90 deg around Z ─────────────────────
        v = Point3D(1.0, 0.0, 0.0)
        axis = Point3D(0.0, 0.0, 1.0)
        rotated = rotate_vector(v, axis, math.pi / 2)
        r90_ok = abs(rotated.x) < 1e-10 and abs(rotated.y - 1.0) < 1e-10 and abs(rotated.z) < 1e-10
        checks["rotation_90z"] = r90_ok
        details["rotation_90z_result"] = f"({rotated.x:.4f}, {rotated.y:.4f}, {rotated.z:.4f})"

        # ── Test 2: Zero rotation is identity ─────────────────────────────────
        v2 = Point3D(3.0, 4.0, 5.0)
        v2_rot = rotate_vector(v2, axis, 0.0)
        zero_rot = abs(v2_rot.x - 3.0) < 1e-10 and abs(v2_rot.y - 4.0) < 1e-10
        checks["rotation_zero_identity"] = zero_rot

        # ── Test 3: Rotation preserves norm ───────────────────────────────────
        v3 = Point3D(1.0, 2.0, 3.0)
        v3_rot = rotate_vector(v3, Point3D(0.0, 1.0, 0.0), 1.23)
        norm_orig = v3.norm()
        norm_rot = v3_rot.norm()
        checks["rotation_preserves_norm"] = abs(norm_orig - norm_rot) < 1e-10

        # ── Test 4: Single F produces 1 line from origin to (0,step,0) ────────
        ls = LSystem3D(axiom="F", rules={}, angle=45.0, step=1.0)
        lines = ls.draw(0)
        checks["single_f_one_line"] = len(lines) == 1
        if lines:
            end = lines[0].end
            single_f_ok = abs(end.x) < 1e-10 and abs(end.y - 1.0) < 1e-10 and abs(end.z) < 1e-10
            checks["single_f_endpoint"] = single_f_ok

        # ── Test 5: F[F]F produces 3 lines ────────────────────────────────────
        ls2 = LSystem3D(axiom="F[F]F", rules={}, angle=90.0, step=1.0)
        lines2 = ls2.draw(0)
        checks["branch_3_lines"] = len(lines2) == 3
        details["branch_line_count"] = len(lines2)

        # ── Test 6: tree3d preset produces 3D branching (z != 0) ──────────────
        p = PLANT_PRESETS["tree3d"]
        ls3 = LSystem3D(axiom=p["axiom"], rules=p["rules"], angle=p["angle"], step=p["step"])
        lines3 = ls3.draw(2)
        points3 = ls3.get_points(iterations=2)
        z_values = [pt.z for pt in points3]
        z_extent = max(z_values) - min(z_values)
        checks["tree3d_has_z_extent"] = z_extent > 0
        details["tree3d_line_count"] = len(lines3)
        details["tree3d_z_extent"] = z_extent

        # ── Test 7: Iteration scaling ─────────────────────────────────────────
        lines_1 = ls3.draw(1)
        lines_2 = ls3.draw(2)
        lines_3 = ls3.draw(3)
        checks["iteration_scaling"] = len(lines_1) < len(lines_2) < len(lines_3)
        details["iteration_line_counts"] = [len(lines_1), len(lines_2), len(lines_3)]

        # ── Test 8: Bounding box ──────────────────────────────────────────────
        bounds = ls3.get_bounds(iterations=2)
        checks["bounds_has_keys"] = all(k in bounds for k in ("min", "max", "center", "size"))
        if "size" in bounds:
            sz = bounds["size"]
            checks["bounds_size_positive"] = sz.x > 0 and sz.y > 0

        # ── Test 9: Bush preset ───────────────────────────────────────────────
        pb = PLANT_PRESETS["bush"]
        ls4 = LSystem3D(axiom=pb["axiom"], rules=pb["rules"], angle=pb["angle"], step=pb["step"])
        lines4 = ls4.draw(2)
        checks["bush_produces_lines"] = len(lines4) > 0
        details["bush_line_count_2iter"] = len(lines4)

        all_pass = all(checks.values())

        return {
            "id": "51_morphology_3d",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Lindenmayer A 1968, J Theor Biol 18:280",
                "doi": "10.1016/0022-5193(68)90079-7",
                "authors": "Lindenmayer A",
                "year": 1968,
                "note": "L-system 3D turtle interpretation with Rodrigues rotation",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "51_morphology_3d",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
