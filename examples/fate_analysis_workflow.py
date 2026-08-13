"""Example 28: cell-fate decision analysis (S7).

A two-gene mutual-repression toggle switch (Gardner, Cantor & Collins
2000 Nature 403:339) is the canonical binary cell-fate decision: two
differentiated attractors (fate A / fate B) separated by an unstable
boundary state.  This script reproduces the three complementary views
of the decision machinery from helixlang.apps.fate_analysis:

  1. bistability_scan    -- deterministic bifurcation diagram: the
     saddle-node birth of the two fates as repression strength w grows.
  2. switching_rate      -- telegraph-promoter noise (Peccoud & Ycart
     1995) drives spontaneous fate flips; a shared translation-resource
     pool (Goetz et al. 2025) throttles both genes, collapses the
     barrier and amplifies switching.
  3. critical_slowing_down -- near the bifurcation the return rate to
     the fixed point vanishes: the lag-1 autocorrelation of a(t)
     approaches 1 (Scheffer et al. 2009 Nature 461:53), an
     early-warning signal before the fate is committed.

Annotated `.helix` form of this workflow:
`examples/28_fate_analysis.helix`

Run with:  python examples/fate_analysis_workflow.py
"""
from helixlang.apps.fate_analysis import (
    bistability_scan,
    critical_slowing_down,
    switching_rate,
)


def main() -> None:
    scan = bistability_scan((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    print("bifurcation scan (toggle switch, Gardner 2000):")
    for p in scan:
        print(f"  w = {p.parameter:>4}   stable fates = {p.n_stable_states}",
              f"(A-high {p.stable_states[-1].a:.2f}, "
              f"B-high {p.stable_states[0].a:.2f})")

    print("\nstochastic switching (w = 7, shared resource pool):")
    for res in (0.0, 0.5, 1.0):
        print(f"  resource={res}: switching rate = "
              f"{switching_rate(7.0, resource_strength=res):.2f}")

    print("\ncritical slowing down (lag-1 autocorrelation of a(t)):")
    for w in (3.0, 5.0, 5.3, 5.5):
        print(f"  w = {w}: lag-1 autocorrelation = "
              f"{critical_slowing_down(w):.3f}")


if __name__ == "__main__":
    main()
