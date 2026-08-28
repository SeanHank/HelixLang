"""Cross-stack hot-loop benchmark matrix (doc/36 Phase 5 item 2).

Benchmarks every hot-loop backend at the SAME workload and reports a
stack-vs-stack matrix: which implementation (``impl_python`` / ``impl_numpy`` /
``impl_numba`` / ``impl_cext`` / ``impl_cython``) is fastest for each kernel,
and by how much.  Uses the public loading/selection path
(:mod:`helixlang._accel._loaders`) so it reflects exactly what production code
sees, and prints a markdown table ready to paste into
``doc/13-performance-report.md`` §3.8 (cross-stack matrix).

Kernels (each is a pure speed switch among equivalent-fidelity impls, doc/36
§3ξ.5 — swapping backend never changes numerics):

* ``dispatch``  — VM kernel ``run_quota`` (python / cext when present)
* ``grn_step``  — GRN discrete-tick recurrence ``step`` (python / numpy / cext / cython)
* ``simplex``   — two-phase simplex pivot ``run`` (python / numpy / cython)
* ``diffusion`` — Gray-Scott reaction-diffusion ``step`` (python / numpy / numba)

Methodology mirrors ``bench_helix.py``: best-of-N repeats with GC disabled and
a warm-up pass; pure stdlib apart from the optional numpy used as an input type
for the numpy backends.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helixlang._accel._loaders import _importable  # noqa: E402


# ============================================================================
# Timing helpers
# ============================================================================
def _best(fn, *, repeat=5):
    """Best-of-``repeat`` per-call times (each call repeats ``fn`` once)."""
    gc.disable()
    try:
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best
    finally:
        gc.enable()


def _bench(fn, *, repeat=5):
    fn()  # warm-up
    return _best(fn, repeat=repeat)


# ============================================================================
# Workload builders
# ============================================================================
def _dispatch_workload(rng, length=128):
    ops = [_PUSH_CONST, 0, _PUSH_CONST, 1, _MUL]
    code = (ops * (length // len(ops))) + [_HALT_OP]
    consts = [rng.uniform(-10.0, 10.0) for _ in range(2)]
    return code, consts


def _grn_workload(rng, n=256, e=4096):
    levels = [rng.random() for _ in range(n)]
    src = [rng.randrange(n) for _ in range(e)]
    dst = [rng.randrange(n) for _ in range(e)]
    weights = [rng.uniform(-1.0, 1.0) for _ in range(e)]
    decays = [0.5] * n
    thresholds = [0.0] * n
    return levels, src, dst, weights, decays, thresholds, 0.5


def _simplex_workload(rng, n_rows=40, n_vars=60):
    # Build a feasible LP whose first `n_rows` columns are an identity basis.
    import numpy as np
    r = np.random.default_rng(202608)
    A = r.uniform(-0.01, 0.01, size=(n_rows, n_vars))
    A[:, :n_rows] = 0.0
    A[np.arange(n_rows), np.arange(n_rows)] = 1.0
    b = r.uniform(0.5, 1.0, size=n_rows)
    tableau = np.hstack([A, b.reshape(-1, 1)])
    basis = list(range(n_rows))
    obj = np.zeros(n_vars)
    obj[n_rows:] = -r.uniform(1.0, 2.0, size=n_vars - n_rows)
    return tableau, basis, obj.tolist(), n_vars


def _diffusion_workload(rng, grid=64):
    import numpy as np
    u = np.full((grid, grid), 1.0)
    v = np.zeros((grid, grid))
    u[grid // 2, grid // 2] = 1.0
    v[grid // 2 + 1, grid // 2] = 1.0
    return u, v, 0.04, 0.06, 0.1, 0.05


# opcodes for the dispatch kernel
_PUSH_CONST = 0x20
_MUL = 0x92
_HALT_OP = 0x11


def _make_runner(mod_name, kind, workload):
    """Return a zero-arg callable that runs one unit of the workload on a
    backend module freshly imported, so each measurement reflects THAT module."""
    mod = importlib.import_module(f"helixlang._accel.{kind}.{mod_name}")
    if kind == "dispatch":
        code, consts = workload
        return lambda: mod.run_quota(code, consts, quota=4096)
    if kind == "grn_step":
        levels, src, dst, weights, decays, thresholds, default = workload
        return lambda: mod.step(levels, src, dst, weights, decays,
                                thresholds, default)
    if kind == "simplex":
        tableau, basis, obj, n_vars = workload
        # copy so repeated runs don't grind to "optimal"/unbounded state
        def run():
            t = [list(row) for row in tableau]
            b = list(basis)
            return mod.run(t, b, obj, n_vars)
        return run
    if kind == "diffusion":
        u, v, F, k, Du, Dv = workload
        return lambda: mod.step(u, v, F, k, Du, Dv)
    raise ValueError(kind)


def _available_modules(kind, expected):
    """Return the importable backend module names for ``kind`` (in loader
    priority order: native, numpy, numba, python)."""
    order = ["impl_cext", "impl_cython", "impl_pyomod",
             "impl_numpy", "impl_numba", "impl_python"]
    found = []
    for name in order:
        if name in expected and _importable(f"helixlang._accel.{kind}", name):
            found.append(name)
    # always ensure the pure-python reference is present and last
    if "impl_python" in expected and "impl_python" not in found:
        found.append("impl_python")
    return found


_KERNELS = {
    "dispatch":  ["impl_cext", "impl_python"],
    "grn_step":  ["impl_cext", "impl_cython", "impl_numpy", "impl_python"],
    "simplex":   ["impl_cython", "impl_numpy", "impl_python"],
    "diffusion": ["impl_numpy", "impl_numba", "impl_python"],
}


# ============================================================================
# Matrix
# ============================================================================
def run_matrix() -> dict:
    rng = random.Random(202608)
    results = {}
    for kind, expected in _KERNELS.items():
        mods = _available_modules(kind, expected)
        units = {}
        if kind == "dispatch":
            units = {"length=128": _dispatch_workload(rng)}
        elif kind == "grn_step":
            units = {"N=256,E=4096": _grn_workload(rng)}
        elif kind == "simplex":
            units = {"40x60": _simplex_workload(rng)}
        elif kind == "diffusion":
            units = {"64x64": _diffusion_workload(rng)}
        kind_rows = []
        for unit_name, workload in units.items():
            row = {"kernel": kind, "workload": unit_name, "backends": {}}
            for m in mods:
                runner = _make_runner(m, kind, workload)
                row["backends"][m] = _bench(runner)
            kind_rows.append(row)
        results[kind] = kind_rows
    return results


def _fmt_sec(v: float) -> str:
    if v >= 1.0:
        return f"{v:.3f}s"
    if v >= 1e-3:
        return f"{v * 1e3:.3f}ms"
    return f"{v * 1e6:.2f}us"


def render_markdown(results: dict) -> str:
    out = []
    out.append("### 3.8 Cross-stack hot-loop matrix (doc/36 Phase 5)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out.append(f"\n_Measured {now} with `python benchmarks/bench_accel_matrix.py`; "
               f"best-of-5, GC disabled, warm-up each._")
    out.append("\nBest per kernel/row is **bolded**; `n/a` = impl not built for "
               "this interpreter.")
    for kind, rows in results.items():
        all_backends = []
        for r in rows:
            for b in r["backends"]:
                if b not in all_backends:
                    all_backends.append(b)
        out.append(f"\n**`{kind}`**\n")
        hdr = "| workload | " + " | ".join(all_backends) + " |"
        out.append(hdr)
        sep = "|---|---" + "---|" * len(all_backends)
        out.append(sep)
        for r in rows:
            cells = []
            for b in all_backends:
                v = r["backends"].get(b)
                if v is None:
                    cells.append("n/a")
                    continue
                text = _fmt_sec(v)
                # bold the fastest present backend for the row
                present = {bb: t for bb, t in r["backends"].items()}
                if present and min(present.values()) == v:
                    text = f"**{text}**"
                cells.append(text)
            out.append(f"| {r['workload']} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse as _ap
    ap = _ap.ArgumentParser(prog="bench_accel_matrix")
    ap.add_argument("--json", metavar="PATH", help="also write raw JSON")
    args = ap.parse_args(argv)
    pk = platform.python_implementation()
    print(f"Benchmarking accel hot-loops on {pk} "
          f"{platform.python_version()} {platform.machine()} ...\n")
    results = run_matrix()
    print(render_markdown(results))
    if args.json:
        json.dump(results, open(args.json, "w"), indent=2)
        print(f"\nraw JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
