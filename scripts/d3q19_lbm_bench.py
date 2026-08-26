"""D3Q19 periodic-channel benchmark (gate 7a: duct profile + wall drag).

Runs the body-force periodic 3D channel and checks the two closed-form
predictions of gate 7a: the interior cross-section peak/mean = 2.096 within
1 % and the wall drag balancing the Guo-2002 body force, plus a raw
sites/step throughput number for the solver.

Run with:  python examples/d3q19_lbm_bench.py
"""
from __future__ import annotations

import time

import numpy as np

from helixlang.apps.lattice_boltzmann_3d import LatticeBoltzmann3D
from helixlang.flow import channel_poiseuille_3d

SIZE, STEPS, FORCE = 21, 1500, 1.0e-4


def main() -> None:
    lbm = LatticeBoltzmann3D(
        SIZE, SIZE, SIZE, omega=1.0, periodic_x=True,
        body_force=(FORCE, 0.0, 0.0))
    t0 = time.perf_counter()
    lbm.run(STEPS)
    elapsed = time.perf_counter() - t0

    u, _, _ = lbm.velocity_fields()
    profile = u[1:-1, 1:-1, SIZE // 2]
    ratio = profile.max() / profile.mean()
    ref = np.asarray(channel_poiseuille_3d(SIZE, SIZE, SIZE, 1.0, "E").u)
    corr = float(np.corrcoef(profile.ravel(),
                             ref[1:-1, 1:-1, SIZE // 2].ravel())[0, 1])

    fx = lbm.force_x
    wall_drag = (fx[:, 0, :].sum() + fx[:, -1, :].sum()
                 + fx[0, :, :].sum() + fx[-1, :, :].sum())
    body = 0.5 * FORCE * (SIZE - 2) * (SIZE - 2) * SIZE

    print(f"D3Q19 periodic channel benchmark ({SIZE}^3, {STEPS} steps)")
    print(f"  peak/mean            : {ratio:.4f}  "
          f"(analytic 2.096, rel err {abs(ratio - 2.096) / 2.096:.2%})")
    print(f"  profile correlation  : {corr:.4f}  (> 0.99 required)")
    print(f"  wall drag / body     : {wall_drag / body:.4f}  (1.0 expected)")
    print(f"  throughput           : {elapsed * 1e3 / STEPS:.2f} ms/tick "
          f"({elapsed * 1e3 / STEPS / (SIZE ** 3) * 1e3:.2f} "
          "us/site/tick)")
    ok = (abs(ratio - 2.096) / 2.096 < 0.01 and corr > 0.99
          and abs(wall_drag / body - 1.0) < 0.02)
    print(f"  gate 7a              : {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
