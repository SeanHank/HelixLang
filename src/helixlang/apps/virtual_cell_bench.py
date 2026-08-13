"""Virtual-cell calibration -> prediction benchmark (S9).

The whole-cell-modeling validation loop in miniature: a
:class:`~helixlang.virtual_cell.VirtualCell` (central dogma + GRN +
FBA metabolism + cell-cycle energy budget, Karr et al. 2012 Cell
150:389) is *calibrated* on one growth condition and *predicts* an
independent condition -- the "calibrate then predict" protocol at the
heart of the Virtual Cell Challenge 2025 and of predictive whole-cell
efforts (Macklin et al. 2020, ``vivek_aerobic``; Thornburg et al.
2022).

:class:`VirtualCellBench` runs the loop:

1. **calibrate** -- build a ground-truth cell with a hidden energy-
   coupling constant (``biomass_to_atp``, the unknown ATP gain per FBA
   biomass-flux unit), record its energy trajectory over the calibration
   condition, and recover the constant with
   :func:`~helixlang.virtual_cell.fit_parameters`.
2. **predict** -- rebuild the cell with the fitted constant and run it
   under a *different* substrate condition and a longer horizon; compare
   the predicted final energy / alive status against the ground truth.

:func:`run_virtual_cell_benchmark` packages the whole loop into a
benchmark dict with explicit pass criteria (parameter recovered within
5%, prediction within 5% relative energy error).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from helixlang.grn import GRN
from helixlang.virtual_cell import (
    VirtualCell,
    VirtualCellConfig,
    encode_gene,
    fit_parameters,
)

#: small three-gene test genome (transcribed + translated by the cell)
_DEFAULT_GENOME: dict[str, str] = {
    "lacZ": encode_gene("MAQILARVFFDDV"),
    "galK": encode_gene("MSSRPQAAASSWW"),
    "tetR": encode_gene("MSRLDKSVINS"),
}


def _default_grn() -> GRN:
    """Three-gene GRN: TetR represses lacZ, lacZ activates galK."""
    grn = GRN()
    for name in _DEFAULT_GENOME:
        grn.add_gene(name, 0.5)
    grn.add_edge("tetR", "lacZ", -2.0)
    grn.add_edge("lacZ", "galK", 2.0)
    grn.nodes["lacZ"].level = 1.0
    return grn


@dataclass(slots=True)
class VirtualCellBenchConfig:
    """Benchmark scenario (calibration + prediction conditions)."""

    genome: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_GENOME))
    grn_builder: Callable[[], GRN] = _default_grn
    truth_biomass_to_atp: float = 5.0e6
    truth_maintenance_atp_per_min: float = 2.5e7
    calibration_uptake: dict[str, float] = field(default_factory=lambda: {"GLC": 10.0})
    prediction_uptake: dict[str, float] = field(default_factory=lambda: {"GLC": 20.0})
    calibration_minutes: int = 20
    prediction_minutes: int = 60
    fit_range: tuple[float, float] = (1e5, 1e8)
    n_samples: int = 150
    fit_seed: int = 0
    refine_rounds: int = 3


class VirtualCellBench:
    """Calibrate-then-predict whole-cell benchmark."""

    def __init__(self, config: VirtualCellBenchConfig | None = None) -> None:
        self.config = config or VirtualCellBenchConfig()
        self.truth_cell = self._build(
            self.config.truth_biomass_to_atp,
            self.config.truth_maintenance_atp_per_min,
            self.config.calibration_uptake,
        )
        self.truth_cell.run(self.config.calibration_minutes)
        self.observed_energy: list[float] = [
            h["energy"] for h in self.truth_cell.history
        ]

    def _build(self, biomass_to_atp: float, maintenance_atp_per_min: float,
               uptake: dict[str, float]) -> VirtualCell:
        cfg = VirtualCellConfig(
            uptake=dict(uptake),
            biomass_to_atp=biomass_to_atp,
            maintenance_atp_per_min=maintenance_atp_per_min,
        )
        return VirtualCell(self.config.genome, self.config.grn_builder(),
                           config=cfg)

    def calibrate(self) -> dict:
        """Recover ``biomass_to_atp`` from the calibration trajectory."""

        def predict(biomass_to_atp: float) -> list[float]:
            cell = self._build(
                biomass_to_atp,
                self.config.truth_maintenance_atp_per_min,
                self.config.calibration_uptake,
            )
            cell.run(self.config.calibration_minutes)
            return [h["energy"] for h in cell.history]

        lo, hi = self.config.fit_range
        return fit_parameters(
            predict,
            self.observed_energy,
            {"biomass_to_atp": (lo, hi)},
            n_samples=self.config.n_samples,
            seed=self.config.fit_seed,
            refine_rounds=self.config.refine_rounds,
        )

    def run_prediction(self, biomass_to_atp: float) -> dict:
        """Run the cell under the prediction condition."""
        cell = self._build(
            biomass_to_atp,
            self.config.truth_maintenance_atp_per_min,
            self.config.prediction_uptake,
        )
        cell.run(self.config.prediction_minutes)
        return {
            "energy": cell.energy,
            "alive": cell.alive,
            "divisions": cell.divisions,
            "proteins": dict(cell.proteins),
        }

    def run(self) -> dict:
        """Full closed loop: calibrate, predict, and score the errors."""
        truth = self.run_prediction(self.config.truth_biomass_to_atp)
        fit = self.calibrate()
        fitted_btp = float(fit["best"]["biomass_to_atp"])
        prediction = self.run_prediction(fitted_btp)
        energy_err = abs(prediction["energy"] - truth["energy"]) / max(
            1.0, abs(truth["energy"])
        )
        btp_err = abs(fitted_btp - self.config.truth_biomass_to_atp) / (
            self.config.truth_biomass_to_atp
        )
        calibration_recovered = btp_err < 0.05
        prediction_matches = (
            prediction["alive"] == truth["alive"] and energy_err < 0.05
        )
        return {
            "fit": fit,
            "fitted_biomass_to_atp": fitted_btp,
            "truth_prediction": truth,
            "fitted_prediction": prediction,
            "energy_rel_error": energy_err,
            "biomass_to_atp_rel_error": btp_err,
            "calibration_recovered": calibration_recovered,
            "prediction_matches": prediction_matches,
            "passed": calibration_recovered and prediction_matches,
        }


def run_virtual_cell_benchmark(
    config: VirtualCellBenchConfig | None = None,
) -> dict:
    """One-shot calibration -> prediction benchmark (see :class:`VirtualCellBench`)."""
    return VirtualCellBench(config).run()


__all__ = [
    "VirtualCellBench",
    "VirtualCellBenchConfig",
    "run_virtual_cell_benchmark",
]
