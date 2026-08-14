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

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from helixlang.apps.whole_cell_scale import essentiality_screen
from helixlang.environment import Environment, EnvironmentConfig
from helixlang.grn import GRN
from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
)
from helixlang.population import (
    CellPopulation,
    PopulationCell,
    PopulationConfig,
)
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


# ============================================================================
# 4-gate whole-cell benchmark (Phase 5 exit gate, doc §8.3-8.4)
# ============================================================================

#: E. coli aerobic glucose batch doubling time (h) from Mahadevan 2002
#: (the dyFBA reference batch used throughout Phase 5).
BATCH_DOUBLING_REFERENCE_H = 0.5

#: adder-slope bound (Taheri-Araghi 2015 reports ~ -0.1 .. 0)
ADDER_SLOPE_TOL = 0.2

#: essentiality-accuracy floor (EcoCyc reference subset)
ESSENTIALITY_ACCURACY_FLOOR = 0.95

#: solid-colony density-profile tolerances (iDynoMiCS 2.0 BM3 profile:
#: uniform interior volume fraction, no hollow core, no edge-ring)
DENSITY_INNER_FLOOR = 0.5
DENSITY_OUTER_FLOOR = 0.5


def _adder_slope(steps: int = 500) -> tuple[float, float]:
    """Regression slope of added-size vs birth-size under the adder rule."""
    cfg = VirtualCellConfig(
        division_rule="adder",
        maintenance_atp_per_min=0.0,
        transcription_atp_per_nt=0.0,
        translation_atp_per_aa=0.0,
        uptake={"GLC": 10.0},
        adder_noise_std=0.2,
        seed=7,
    )
    vc = VirtualCell(_DEFAULT_GENOME, _default_grn(), config=cfg)
    for _ in range(steps):
        vc.step()
    events: list[tuple[float, float]] = []
    prev = vc.history[0]
    for entry in vc.history[1:]:
        if entry["divisions"] > prev["divisions"]:
            added = prev["volume_um3"] - prev["volume_birth_um3"]
            events.append((prev["volume_birth_um3"], added))
        prev = entry
    births = [b for b, _ in events]
    addeds = [a for _, a in events]
    if len(events) < 3:
        return 0.0, 0.0
    bmean = sum(births) / len(births)
    amean = sum(addeds) / len(addeds)
    cov = sum((b - bmean) * (a - amean) for b, a in events)
    var = sum((b - bmean) ** 2 for b in births)
    slope = cov / var if var > 0 else 0.0
    return slope, amean - 1.6


def _colony_density_profile() -> dict:
    """Area-normalized radial density profile of a dFBA disk colony.

    Each radial band's cell count is divided by the band's annulus area
    (``2k+1``), giving a volume-fraction proxy -- flat inside a solid
    colony, as in the iDynoMiCS 2.0 BM3 density profiles.
    """
    center, radius = 10, 3
    cells: list[PopulationCell] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                cells.append(PopulationCell(
                    id=len(cells), energy=1e5,
                    x=center + dx, y=center + dy))
    env = Environment(EnvironmentConfig(
        width=21, height=21,
        glucose_initial_mm=10.0, oxygen_initial_mm=0.25))
    cfg = PopulationConfig(grid_width=21, grid_height=21, environment=env,
                           dfba_enabled=True, division_threshold=1e9)
    pop = CellPopulation(cells, cfg)
    for _ in range(3):
        pop.step()
    counts = pop.colony_observables()["radial_density"]
    n = len(counts)
    normalized = [counts[k] / (2 * k + 1) for k in range(n)]
    inner = normalized[:n // 2]
    outer = normalized[n // 2:]
    global_mean = sum(normalized) / n
    inner_mean = sum(inner) / len(inner)
    outer_mean = sum(outer) / len(outer)
    inner_ratio = inner_mean / global_mean if global_mean > 0 else 0.0
    outer_ratio = (outer_mean / inner_mean if inner_mean > 0 else 0.0)
    return {
        "counts": counts,
        "normalized_density": normalized,
        "inner_half_mean": inner_mean,
        "outer_half_mean": outer_mean,
        "inner_ratio": inner_ratio,
        "outer_ratio": outer_ratio,
    }


def run_whole_cell_benchmark() -> dict:
    """Phase-5 4-gate whole-cell benchmark (doc §8.3-8.4).

    Scores, and requires all four of:

    1. **essentiality accuracy** -- EcoCyc FBA knockout screen ≥ 0.95;
    2. **batch doubling-time fidelity** -- the core-model aerobic glucose
       batch doubling time matches the Mahadevan 2002 reference (~0.5 h);
    3. **adder slope** -- added-size vs birth-size regression slope ≈ 0
       (size homeostasis, Taheri-Araghi 2015);
    4. **BM3-style colony density profile** -- a dFBA disk colony shows a
       uniform interior volume fraction with no hollow core and no
       edge-ring depletion (iDynoMiCS 2.0 profile).
    """
    ess = essentiality_screen()
    ess_accuracy = float(ess["accuracy"])
    ess_n_tested = int(ess["n_tested"])
    ess_n_matched = int(ess["n_matched"])

    fba = DynamicFluxBalance(
        model=ECOLI_CORE_MODEL,
        config=DynamicFBAConfig(
            dt_h=0.25, initial_biomass_gdw=0.05, initial_glucose_mm=10.0))
    mu_max = max(e["growth_rate"] for e in fba.run(duration_h=2.0))
    doubling_h = math.log(2.0) / mu_max
    doubling_err = abs(doubling_h - BATCH_DOUBLING_REFERENCE_H) / (
        BATCH_DOUBLING_REFERENCE_H)

    adder_slope, adder_bias = _adder_slope()

    density = _colony_density_profile()

    scores = {
        "essentiality_accuracy": ess_accuracy,
        "essentiality_n_matched": ess_n_matched,
        "essentiality_n_tested": ess_n_tested,
        "batch_doubling_h": doubling_h,
        "batch_doubling_reference_h": BATCH_DOUBLING_REFERENCE_H,
        "batch_doubling_rel_error": doubling_err,
        "adder_slope": adder_slope,
        "adder_mean_bias_um3": adder_bias,
        "density_inner_ratio": density["inner_ratio"],
        "density_outer_ratio": density["outer_ratio"],
    }
    passed = {
        "essentiality": ess_accuracy >= ESSENTIALITY_ACCURACY_FLOOR,
        "batch_doubling": doubling_err <= 0.2,
        "adder_slope": abs(adder_slope) <= ADDER_SLOPE_TOL,
        "density_profile": (
            density["inner_ratio"] >= DENSITY_INNER_FLOOR
            and density["outer_ratio"] >= DENSITY_OUTER_FLOOR),
    }
    return {"scores": scores, "passed": passed, "all_passed": all(passed.values())}


__all__ = [
    "VirtualCellBench",
    "VirtualCellBenchConfig",
    "run_virtual_cell_benchmark",
    "run_whole_cell_benchmark",
]
