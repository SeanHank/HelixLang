"""D3Q19 3D lattice-Boltzmann performance and gate benchmark (doc/18-programmable-cell-population-simulation.md §13 gate 7).

The 3D counterpart of the interactive Level-2 2D timing checks: measures the
``LatticeBoltzmann3D`` single-tick cost on the acceptance-gate lattice
(100 x 100 x 50, gate 7e target < 2 s/tick single core) and verifies the
rectangular-duct gate 7a numerically (peak/mean = 2.096 within 1 %, wall-drag
vs body-force balance). Pure standard library.

Usage::

    python benchmarks/bench_lbm3d.py             # full run
    python benchmarks/bench_lbm3d.py --fast      # reduced matrix

Methodology follows ``bench_helix.py``: best of several repeats with the GC
disabled during timing, a warm-up pass before each measurement, and the flow
fields kept in numpy so the streaming/collision loop is measured (not the
Python-level field bookkeeping).
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helixlang.plugins.apps.lattice_boltzmann_3d import LatticeBoltzmann3D  # noqa: E402
from helixlang.plugins.runtime.flow import channel_poiseuille_3d  # noqa: E402


def best_time(fn: Callable[[], Any], *, number: int = 1,
              repeat: int = 5) -> float:
    """Best wall time (seconds) across ``repeat`` runs of ``fn``."""
    fn()  # warm-up
    best = float("inf")
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = (time.perf_counter() - t0) / number
        gc.enable()
        if dt < best:
            best = dt
    gc.enable()
    return best


def make_duct_solver(size: int) -> LatticeBoltzmann3D:
    """Periodic body-force duct used by the gate 7a measurements."""
    return LatticeBoltzmann3D(
        size, size, size, omega=1.0, periodic_x=True,
        body_force=(1.0e-4, 0.0, 0.0))


def gate_duct(size: int, steps: int) -> dict[str, float]:
    """Rectangular-duct gate 7a numbers (peak/mean, correlation, drag)."""
    import numpy as np

    lbm = make_duct_solver(size)
    lbm.run(steps)
    u, _, _ = lbm.velocity_fields()
    profile = u[1:-1, 1:-1, size // 2]
    ratio = float(profile.max() / profile.mean())
    ref = np.asarray(channel_poiseuille_3d(size, size, size, 1.0, "E").u)
    corr = float(np.corrcoef(profile.ravel(),
                             ref[1:-1, 1:-1, size // 2].ravel())[0, 1])
    fx = lbm.force_x
    wall_drag = (fx[:, 0, :].sum() + fx[:, -1, :].sum()
                 + fx[0, :, :].sum() + fx[-1, :, :].sum())
    body = 0.5 * 1.0e-4 * (size - 2) * (size - 2) * size
    return {
        "ratio": ratio,
        "ratio_analytic": 2.096,
        "corr": corr,
        "wall_drag_over_body": wall_drag / body,
    }


def bench_duct_gate() -> dict[str, Any]:
    size, steps = 21, 1500
    res = gate_duct(size, steps)
    res["size"] = size
    res["steps"] = steps
    res["ratio_rel_err"] = abs(res["ratio"] - 2.096) / 2.096
    res["passed"] = (res["ratio_rel_err"] < 0.01
                     and res["corr"] > 0.99
                     and abs(res["wall_drag_over_body"] - 1.0) < 0.02)
    return res


def bench_tick_per_site(sizes: list[int], steps: int) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        lbm = make_duct_solver(size)
        lbm.run(200)  # settle caches / numpy blocks before timing
        sites = size * size * size

        def tick(_lbm: LatticeBoltzmann3D = lbm) -> None:
            _lbm.step()

        per_tick = best_time(tick, number=5, repeat=3)
        rows.append({
            "size": size,
            "sites": sites,
            "us_per_site": per_tick * 1e6 / sites,
            "ms_per_tick": per_tick * 1e3,
        })
    return rows


def bench_gate_lattice() -> dict[str, Any]:
    """Single-tick cost on the 100 x 100 x 50 gate-7e lattice."""
    w, h, d = 100, 100, 50
    lbm = LatticeBoltzmann3D(w, h, d, omega=1.0, periodic_x=True,
                             body_force=(1.0e-4, 0.0, 0.0))
    lbm.run(200)

    def tick() -> None:
        lbm.step()

    per_tick = best_time(tick, number=5, repeat=3)
    return {
        "width": w, "height": h, "depth": d,
        "sites": w * h * d,
        "ms_per_tick": per_tick * 1e3,
        "target_ms_per_tick": 2.0e3,
        "passed": per_tick < 2.0,
    }


def emit_markdown(results: dict[str, Any]) -> str:
    gate = results["gate"]
    lines = [
        "# D3Q19 3D lattice-Boltzmann benchmark",
        "",
        f"Platform: {results['platform']}",
        "",
        "## Gate 7a — rectangular-duct solution (21^3, 1500 steps)",
        "",
        "| quantity | value | analytic | rel. err |",
        "|---|---|---|---|",
        f"| peak/mean | {gate['ratio']:.4f} | 2.096 | "
        f"{gate['ratio_rel_err']:.2%} |",
        f"| correlation | {gate['corr']:.4f} | > 0.99 | — |",
        f"| wall drag / body force | {gate['wall_drag_over_body']:.4f} "
        "| 1.0 | — |",
        "",
        f"Gate 7a passed: {gate['passed']}",
        "",
        "## Single-tick cost",
        "",
        "| lattice | sites | µs/site/tick | ms/tick |",
        "|---|---|---|---|",
    ]
    for row in results["scaling"]:
        lines.append(
            f"| {row['size']}x{row['size']}x{row['size']} | {row['sites']} "
            f"| {row['us_per_site']:.2f} | {row['ms_per_tick']:.2f} |")
    gl = results["gate_lattice"]
    lines += [
        f"| {gl['width']}x{gl['height']}x{gl['depth']} | {gl['sites']} | "
        f"{gl['ms_per_tick'] * 1e3 / gl['sites']:.2f} | "
        f"{gl['ms_per_tick']:.1f} |",
        "",
        f"Gate 7e (100x100x50 < 2000 ms/tick): {gl['passed']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", action="store_true",
                   help="run a reduced measurement matrix")
    p.add_argument("--json", metavar="OUT", default=None,
                   help="also write the raw results as JSON")
    args = p.parse_args(argv)

    print(f"D3Q19 benchmark: {'fast' if args.fast else 'full'} matrix "
          f"({platform.python_implementation()} "
          f"{platform.python_version()})", file=sys.stderr)

    sizes = [21, 30] if args.fast else [21, 30, 40]
    results: dict[str, Any] = {
        "platform": f"{platform.python_implementation()} "
                    f"{platform.python_version()}",
        "gate": bench_duct_gate(),
        "scaling": bench_tick_per_site(sizes, 200),
        "gate_lattice": bench_gate_lattice(),
    }

    print(emit_markdown(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
