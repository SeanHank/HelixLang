"""D3Q19 3D pressure-driven microfluidic channel + colony (Design 6 Level 2 3D).

A D3Q19 lattice-Boltzmann simulation of laminar flow through a rectangular
microchannel with a small 3D cell colony as a no-slip obstacle, plus the
resulting flow field advecting a dissolved-substrate volume (3D nutrient
transport demo).

Two stages:

1. a clean rectangular duct verifies the Boussinesq solution of doc/18-programmable-cell-population-simulation.md gate
   7a (interior peak/mean = 2.096 within 1 %),
2. a channel with a 5 x 5 x 5 colony shows the Ladd momentum-exchange drag
   and drives 3D substrate advection.

Run with:  python examples/d3q19_lbm_pressure_channel.py
"""
from __future__ import annotations

import numpy as np

from helixlang.apps.lattice_boltzmann_3d import LatticeBoltzmann3D
from helixlang.environment import ConcentrationField3D
from helixlang.flow import channel_poiseuille_3d

HEIGHT, DEPTH = 17, 17
OMEGA = 1.0
INLET_DENSITY, OUTLET_DENSITY = 1.0005, 0.9995


def duct_inlet_profile(width: int) -> np.ndarray:
    """Duct cross-section profile (analytic series) scaled to a low speed."""
    ref = np.asarray(channel_poiseuille_3d(width, HEIGHT, DEPTH, 1.0, "E").u)
    return ref[:, :, width // 2] / ref.max() * 0.05


def main() -> None:
    # stage 1: clean rectangular duct -> gate 7a
    clean_width = 40
    clean = LatticeBoltzmann3D(
        clean_width, HEIGHT, DEPTH, omega=OMEGA,
        inlet_velocity=duct_inlet_profile(clean_width),
        inlet_density=INLET_DENSITY, outlet_density=OUTLET_DENSITY)
    clean.run(1500)
    u, _, _ = clean.velocity_fields()
    profile = u[1:-1, 1:-1, clean_width // 2]
    ratio = profile.max() / profile.mean()

    print("D3Q19 pressure-driven 3D channel")
    print(f"  clean duct          : {clean_width} x {HEIGHT} x {DEPTH}, "
          f"1500 steps")
    print(f"    gate 7a peak/mean = {ratio:.4f} "
          f"(analytic 2.096, rel err {abs(ratio - 2.096) / 2.096:.2%})")

    # stage 2: channel with a 5 x 5 x 5 colony block
    width = 50
    lbm = LatticeBoltzmann3D(
        width, HEIGHT, DEPTH, omega=OMEGA,
        inlet_velocity=duct_inlet_profile(width),
        inlet_density=INLET_DENSITY, outlet_density=OUTLET_DENSITY)
    occ = np.zeros((DEPTH, HEIGHT, width), dtype=bool)
    occ[6:11, 6:11, 22:27] = True
    lbm.set_occupancy(occ)
    lbm.run(2000)

    u, v, w = lbm.velocity_fields()
    cross = max(abs(v[1:-1, 1:-1, :]).max(), abs(w[1:-1, 1:-1, :]).max())
    drag_x = float(lbm.force_x[6:11, 6:11, 22:27].sum())
    drag_y = float(lbm.force_y[6:11, 6:11, 22:27].sum())
    drag_z = float(lbm.force_z[6:11, 6:11, 22:27].sum())
    print(f"  channel + colony    : {width} x {HEIGHT} x {DEPTH}, "
          f"colony at x=22..26, 2000 steps")
    print(f"    max cross-flow |v|,|w| = {cross:.2e} sites/step")
    print(f"    colony drag  Fx = {drag_x:.5f} (downstream)   "
          f"Fy = {drag_y:.2e}   Fz = {drag_z:.2e}")

    # advect a substrate volume through the developed downstream region
    field = lbm.flow_field(substeps=1)
    substrate = ConcentrationField3D(
        "oxygen", width, HEIGHT, DEPTH, diffusion_um2_s=100.0)
    substrate.concentration = [[[1.0 if 34 <= x < 40 else 0.0
                                for x in range(width)]
                               for _ in range(HEIGHT)]
                              for _ in range(DEPTH)]
    mass0 = substrate.total_mm()
    com0 = sum(plane[y][x] * x
               for plane in substrate.concentration
               for y in range(HEIGHT) for x in range(width)) / mass0
    substrate.advect_3d(field)
    mass1 = substrate.total_mm()
    com1 = sum(plane[y][x] * x
               for plane in substrate.concentration
               for y in range(HEIGHT) for x in range(width)) / mass1
    print(f"  3D substrate advection: peak flow "
          f"{field.max_magnitude():.4f} sites/tick")
    print(f"    mass drift          : {abs(mass1 - mass0) / mass0:.2e} "
          "(relative)")
    print(f"    centre of mass shift: {com1 - com0:+.4f} sites (downstream)")


if __name__ == "__main__":
    main()
