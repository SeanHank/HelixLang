"""Example 30: virtual-cell calibration -> prediction workflow (S9).

The whole-cell-modeling validation loop in miniature, driven by the
Python API: a VirtualCell (central dogma + GRN + FBA metabolism +
cell-cycle energy budget, Karr et al. 2012) is *calibrated* on one
growth condition and *predicts* an independent condition -- the
"calibrate then predict" protocol of the Virtual Cell Challenge 2025.

Annotated `.helix` form of this workflow:
`examples/30_virtual_cell.helix`

Run with:  python examples/virtual_cell_workflow.py
"""
from helixlang.plugins.apps.virtual_cell_bench import (
    VirtualCellBench,
    VirtualCellBenchConfig,
    run_virtual_cell_benchmark,
)


def main() -> None:
    bench = VirtualCellBench(VirtualCellBenchConfig(
        truth_biomass_to_atp=5.0e6,
        truth_maintenance_atp_per_min=2.5e7,
        calibration_uptake={"GLC": 10.0},
        prediction_uptake={"GLC": 20.0},
        calibration_minutes=20,
        prediction_minutes=60,
    ))

    print(f"calibration condition   : GLC "
          f"{bench.config.calibration_uptake['GLC']:.0f} mM, "
          f"{bench.config.calibration_minutes} min")
    print(f"observed energy @ t={bench.config.calibration_minutes}: "
          f"{bench.observed_energy[-1]:,.0f} ATP")

    fit = bench.calibrate()
    fitted = fit["best"]["biomass_to_atp"]
    print(f"fitted biomass_to_atp   : {fitted:.3e} "
          f"(truth {bench.config.truth_biomass_to_atp:.3e}, "
          f"rel err {bench.config.truth_biomass_to_atp and abs(fitted - bench.config.truth_biomass_to_atp) / bench.config.truth_biomass_to_atp:.2%})")
    print(f"fit evaluations         : {fit['n_samples']}")

    truth = bench.run_prediction(bench.config.truth_biomass_to_atp)
    prediction = bench.run_prediction(fitted)
    print(f"\nprediction condition    : GLC "
          f"{bench.config.prediction_uptake['GLC']:.0f} mM, "
          f"{bench.config.prediction_minutes} min")
    print(f"ground-truth energy     : {truth['energy']:,.0f} ATP "
          f"(alive={truth['alive']}, proteins={truth['proteins']})")
    print(f"fitted-model energy     : {prediction['energy']:,.0f} ATP "
          f"(alive={prediction['alive']}, proteins={prediction['proteins']})")

    result = run_virtual_cell_benchmark()
    print(f"\nbenchmark passed        : {result['passed']} "
          f"(calibrated={result['calibration_recovered']}, "
          f"predicted={result['prediction_matches']})")


if __name__ == "__main__":
    main()
