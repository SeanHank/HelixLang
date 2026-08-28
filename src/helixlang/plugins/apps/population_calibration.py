"""Population-level mixed-observable calibration closure (doc/18-programmable-cell-population-simulation.md §13 Design 4).

Fits the three free dFBA colony parameters against *colony-level* mixed
observables -- specific growth rate, core/edge chemistry (oxygen, acetate)
and colony division state -- using the inverse-variance-weighted
:func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters` harness (Karr et al. 2012
DREAM8 weighted fitting; Bosshard et al. 2020 BMC Genomics 21:232
colony-scale parameter estimation).

The three calibrated parameters and their biological anchors:

- ``dfba_oxygen_max_uptake``   -- maximum O2 uptake rate (mmol O2/gDW/h);
  the Monod cap on NADH_OX respiration sets the colony's specific growth
  rate and acetate overflow (Core/edge stratification; Phase 5).
- ``dfba_energy_scale``        -- energy (ATP) gained per unit specific
  growth rate per hour; scales how fast cell energy accumulates, i.e. how
  fast the colony grows out.
- ``division_threshold``       -- energy (ATP) required to divide; sets the
  newborn half-energy and therefore the steady-state colony energy per
  cell (Taheri-Araghi 2015 adder-like size control at the colony level).

A ground-truth colony is simulated with hidden values of the three
parameters and its mixed observable vector recorded.  The parameters are
then recovered in three nearly independent stages:

1. metabolic stage -- ``dfba_oxygen_max_uptake`` against a short,
   division-free probe (growth-rate + core/edge chemistry blocks).  The
   probe runs at a tiny ``dfba_energy_scale`` so no candidate oxygen flux
   pushes any cell to division; energy/threshold are division-level knobs
   and do not affect these observables.
2. energy stage -- ``dfba_energy_scale`` against a division-free probe
   (huge threshold, so no divisions) whose mean cell energy grows
   linearly with the per-tick energy accumulation rate.  Reading the
   scale outside the division block is what makes the two knobs
   separable: in the division block both the mean and the newborn energy
   scale like ``threshold - c*energy_scale``, so fitting them jointly
   leaves a shallow diagonal ridge.
3. division stage -- ``division_threshold`` against the division-probe
   block (time-averaged mean/total/newborn energy), with the oxygen
   uptake and energy scale held at their recovered values.  A newborn
   holds half of a parent that just crossed the threshold, so the mean
   newborn energy is a near-direct readout of ``division_threshold``.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from helixlang.plugins.runtime.environment import Environment, EnvironmentConfig
from helixlang.plugins.runtime.population import (
    CellPopulation,
    PopulationCell,
    PopulationConfig,
)
from helixlang.plugins.runtime.virtual_cell import fit_parameters

#: ground-truth (hidden) dFBA colony parameters
TRUTH_OXYGEN_MAX_UPTAKE = 12.0
TRUTH_ENERGY_SCALE = 2.1e8
TRUTH_DIVISION_THRESHOLD = 2.0e8

#: default fit box for the three parameters
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "dfba_oxygen_max_uptake": (5.0, 30.0),
    "dfba_energy_scale": (1.4e8, 2.8e8),
    "division_threshold": (1.6e8, 3.0e8),
}

#: the metabolic probe runs at this tiny energy scale so the colony never
#: divides and the growth / chemistry blocks stay clean of division state
METABOLIC_PROBE_ENERGY_SCALE = 1.0e7
METABOLIC_PROBE_THRESHOLD = 1.0e9

#: the energy probe runs at a huge threshold so the colony can never
#: divide and the mean cell energy grows linearly with the per-tick
#: ``dfba_energy_scale`` accumulation rate -- a division-free readout of
#: the energy scale that does not depend on the division threshold.
ENERGY_PROBE_THRESHOLD = 1.0e12
ENERGY_PROBE_TICKS = 4

#: division-probe blow-up cap: candidates whose accumulation rate is far
#: above the truth can divide the colony to thousands of cells (and
#: hundreds of FBA LPs) in-window; past this size the probe short-circuits
#: and returns a strongly-penalized vector so the fit spends its budget on
#: plausible candidates.
MAX_PROBE_CELLS = 160

OBSERVABLE_BLOCKS = (
    "growth", "chemistry", "division", "energy",
)
METABOLIC_BLOCKS = ("growth", "chemistry")
DIVISION_BLOCKS = ("division",)
ENERGY_BLOCKS = ("energy",)


class PopulationCalibration:
    """Calibrate the dFBA colony parameters on mixed colony observables.

    Args:
        n_cells: starting cells of each probe colony (spread on a
            ``2 x 2`` site cluster so a core/edge chemistry gradient
            forms).
        metabolic_ticks: probe length of the division-free metabolic
            probe.
        division_ticks: probe length of the growth/division probe.  Chosen
            so the ground-truth colony divides one to two times in-window:
            ``division_threshold`` is unobservable unless the colony
            actually divides.
        glucose_mm / oxygen_mm: probe media.
        diffusion_um2_s: substrate diffusion coefficient.
        grid_width / grid_height: probe lattice.
        ranges: parameter -> (lo, hi) fit box.
        n_samples: random samples for each global fit stage.
        refine_rounds: pattern-search passes after the random stage.
        fit_seed: deterministic fit RNG seed.
        cell_seed: RNG seed of both the ground-truth and every candidate
            colony, so the closed-loop benchmark is self-consistent
            (division timing is stochastic; a shared seed puts the SSE
            valley exactly at the truth).
    """

    def __init__(
        self,
        n_cells: int = 4,
        metabolic_ticks: int = 4,
        energy_ticks: int = 4,
        division_ticks: int = 12,
        glucose_mm: float = 10.0,
        oxygen_mm: float = 0.25,
        diffusion_um2_s: float = 8.0,
        grid_width: int = 12,
        grid_height: int = 12,
        ranges: dict[str, tuple[float, float]] | None = None,
        n_samples: int = 20,
        refine_rounds: int = 2,
        fit_seed: int = 0,
        cell_seed: int = 0,
        smooth_ticks: int = 4,
        polish_passes: int = 6,
        refine_windows: tuple[float, ...] = (0.35, 0.12),
    ) -> None:
        self.n_cells = n_cells
        self.metabolic_ticks = metabolic_ticks
        self.energy_ticks = energy_ticks
        self.division_ticks = division_ticks
        self.glucose_mm = glucose_mm
        self.oxygen_mm = oxygen_mm
        self.diffusion_um2_s = diffusion_um2_s
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.ranges = dict(ranges or DEFAULT_RANGES)
        self.n_samples = n_samples
        self.refine_rounds = refine_rounds
        self.fit_seed = fit_seed
        self.cell_seed = cell_seed
        self.smooth_ticks = smooth_ticks
        self.polish_passes = polish_passes
        self.refine_windows = refine_windows
        self.observed = self._observables(
            oxygen_max=TRUTH_OXYGEN_MAX_UPTAKE,
            energy_scale=TRUTH_ENERGY_SCALE,
            division_threshold=TRUTH_DIVISION_THRESHOLD)
        self.weights = self._weights(self.observed)

    # -------- model construction --------

    def _config(self, *, oxygen_max: float, energy_scale: float,
                division_threshold: float) -> PopulationConfig:
        env = Environment(EnvironmentConfig(
            width=self.grid_width, height=self.grid_height,
            glucose_initial_mm=self.glucose_mm,
            oxygen_initial_mm=self.oxygen_mm,
            glucose_diffusion_um2_s=self.diffusion_um2_s,
            oxygen_diffusion_um2_s=self.diffusion_um2_s))
        return PopulationConfig(
            grid_width=self.grid_width, grid_height=self.grid_height,
            environment=env, dfba_enabled=True, dfba_shared_batch=True,
            dfba_oxygen_max_uptake=oxygen_max,
            dfba_energy_scale=energy_scale,
            division_threshold=division_threshold)

    def _cells(self, seed: int, co_located: bool = False,
               division_threshold: float = TRUTH_DIVISION_THRESHOLD
               ) -> list[PopulationCell]:
        rng = random.Random(seed)
        out: list[PopulationCell] = []
        for i in range(self.n_cells):
            if co_located:
                x = 4
                y = 4
            else:
                x = 3 + rng.choice((0, 1))
                y = 3 + rng.choice((0, 1))
            # Staggered initial energies desynchronize the colony: a
            # synchronized start makes every lineage double on the same
            # tick and the final-snapshot mean energy hangs on division
            # phase instead of the division_threshold steady state.
            energy = rng.uniform(0.2, 0.9) * division_threshold
            out.append(PopulationCell(id=i, energy=energy, x=x, y=y))
        return out

    def _build(self, *, seed: int, ticks: int, oxygen_max: float,
               energy_scale: float, division_threshold: float,
               co_located: bool = False, max_cells: int | None = None
               ) -> CellPopulation:
        pop = CellPopulation(
            self._cells(seed, co_located=co_located,
                        division_threshold=division_threshold),
            self._config(oxygen_max=oxygen_max,
                         energy_scale=energy_scale,
                         division_threshold=division_threshold),
            seed=seed)
        for _ in range(ticks):
            pop.step()
            if max_cells is not None and len(pop.cells) > max_cells:
                break
        return pop

    # -------- observables --------

    def _metabolic_probe(self, *, oxygen_max: float) -> list[float]:
        """Division-free probe: mean specific growth rate + core-vs-edge
        chemistry of a short-lived colony run at a tiny energy scale, so
        the colony can never reach the division threshold and every cell
        keeps its dFBA batch."""
        pop = self._build(
            seed=self.cell_seed, ticks=self.metabolic_ticks,
            oxygen_max=oxygen_max,
            energy_scale=METABOLIC_PROBE_ENERGY_SCALE,
            division_threshold=METABOLIC_PROBE_THRESHOLD)
        alive = [c for c in pop.cells if c.alive]
        rates = [c.dfba.growth_rate for c in alive
                 if c.dfba is not None and c.dfba.growth_rate > 0.0]
        growth = sum(rates) / len(rates) if rates else 0.0
        strat = pop.dfba_stratification(quantile=0.25, min_cells=4)
        return [growth, strat["core_acetate_mm"], strat["edge_oxygen_mm"]]

    def _division_probe(self, *, oxygen_max: float, energy_scale: float,
                        division_threshold: float) -> list[float]:
        """Division probe: time-averaged mean and total alive energy and
        mean newborn energy over the final ``smooth_ticks`` ticks of a
        window in which the ground-truth colony divides several times.

        Cells share a single lattice site so the shared batch runs one LP
        per tick; staggered initial energies keep divisions de-phased.
        Averaging over the trailing ticks washes out division-phase noise
        (a newborn readout is only meaningful on a tick where a division
        just occurred), leaving observables that are continuous in the
        fitted parameters.  The mean newborn energy is a near-direct
        ``division_threshold`` readout (a newborn holds half of a parent
        that just crossed the threshold), while the total energy tracks
        the ``dfba_energy_scale`` accumulation rate.
        """
        env = self._config(oxygen_max=oxygen_max,
                           energy_scale=energy_scale,
                           division_threshold=division_threshold)
        run_pop = CellPopulation(
            self._cells(self.cell_seed, co_located=True,
                        division_threshold=division_threshold),
            env, seed=self.cell_seed)
        snapshots: list[list[float]] = []
        for _ in range(self.division_ticks):
            run_pop.step()
            if len(run_pop.cells) > MAX_PROBE_CELLS:
                snapshots.append([5e8, 1e12, 3e8, 0.0])
                break
            alive = [c for c in run_pop.cells if c.alive]
            n = float(len(alive))
            avg_e = sum(c.energy for c in alive) / n if n else 0.0
            total_e = sum(c.energy for c in alive)
            newborn = [c.energy for c in alive if c.age == 0]
            newb_e = sum(newborn) / len(newborn) if newborn else 0.0
            snapshots.append([avg_e, total_e, newb_e, n])
        tail = snapshots[-self.smooth_ticks:]
        if len(tail) < self.smooth_ticks:
            tail = snapshots
        if any(s[1] > 1e11 for s in tail):
            # blown-up candidate: return a vector far from the observed one
            return [5e8, 1e12, 3e8, 0.0]
        return [sum(s[i] for s in tail) / len(tail) for i in range(4)]

    def _energy_probe(self, *, oxygen_max: float,
                      energy_scale: float) -> list[float]:
        """Division-free energy probe: mean cell energy after a short
        window at a huge division threshold.  With no divisions the colony
        size is constant and each cell's energy grows by a fixed per-tick
        increment proportional to ``dfba_energy_scale``, so the final mean
        energy is a smooth, threshold-independent linear readout of the
        energy scale.  This decouples the scale from the threshold: the
        division block cannot separate the two because both its mean and
        newborn energies scale like ``threshold - c*energy_scale``.
        """
        cells = self._cells(self.cell_seed, co_located=True,
                            division_threshold=TRUTH_DIVISION_THRESHOLD)
        pop = CellPopulation(
            cells,
            self._config(oxygen_max=oxygen_max, energy_scale=energy_scale,
                         division_threshold=ENERGY_PROBE_THRESHOLD),
            seed=self.cell_seed)
        for _ in range(self.energy_ticks):
            pop.step()
        alive = [c for c in pop.cells if c.alive]
        n = float(len(alive))
        return [sum(c.energy for c in alive) / n if n else 0.0]

    def _observables(self, *, oxygen_max: float, energy_scale: float,
                     division_threshold: float,
                     probes: Sequence[str] | None = None) -> list[float]:
        """Mixed observable vector of the colony: the metabolic probe,
        the division probe, then the energy probe.  All entries are
        colony-level aggregates -- no per-cell parameter is observed
        directly.  ``probes`` restricts the vector to a subset of the
        blocks."""
        blocks = set(probes) if probes is not None else set(OBSERVABLE_BLOCKS)
        out: list[float] = []
        if blocks & set(METABOLIC_BLOCKS):
            out.extend(self._metabolic_probe(oxygen_max=oxygen_max))
        if "division" in blocks:
            out.extend(self._division_probe(
                oxygen_max=oxygen_max, energy_scale=energy_scale,
                division_threshold=division_threshold))
        if "energy" in blocks:
            out.extend(self._energy_probe(oxygen_max=oxygen_max,
                                          energy_scale=energy_scale))
        return out

    def _block_indices(self) -> dict[str, list[int]]:
        return {
            "growth": [0],
            "chemistry": [1, 2],
            "division": [3, 4, 5, 6],
            "energy": [7],
        }

    def _weights_for(self, blocks: Sequence[str],
                     importance: Sequence[float]) -> list[float]:
        """Inverse-variance weights (aligned with the sub-vector spanned by
        ``blocks``) with explicit per-block importance.  The observation
        variance is modeled as proportional to the observed value squared
        (DESeq2 2014 multiplicative-noise structure, as in
        :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters`), so each element is
        weighted by ``imp / observed**2``.  This keeps heterogeneous
        readouts comparable: a colony-size count (O(10), where a 3-cell
        error is 10%) and a mean-energy readout (O(1e8)) contribute
        proportionally to their relative error, which per-observation
        inverse-magnitude scaling cannot express."""
        idx = self._block_indices()
        sel = [i for b in blocks for i in idx[b]]
        out: list[float] = [0.0] * len(sel)
        for b, imp in zip(blocks, importance, strict=True):
            bi = idx[b]
            for i in bi:
                scale = self.observed[i]
                if abs(scale) <= 0.0:
                    scale = 1.0
                out[sel.index(i)] = imp / (scale * scale)
        return out

    def _weights(self, observed: list[float]) -> list[float]:
        """Full-vector weights for reporting overall SSE."""
        return self._weights_for(
            ["growth", "chemistry", "division", "energy"],
            (0.30, 0.15, 0.40, 0.15))

    def predict(self, probes: Sequence[str] | None = None,
                **params: float) -> list[float]:
        """Run the probe colony (or a subset selected by ``probes``) with
        the candidate parameters and return the mixed observable vector
        (same layout as :attr:`observed` for the selected blocks)."""
        return self._observables(
            oxygen_max=params["dfba_oxygen_max_uptake"],
            energy_scale=params["dfba_energy_scale"],
            division_threshold=params["division_threshold"],
            probes=probes)

    def _fit_stage(self, names: list[str], fixed: dict[str, float],
                   blocks: Sequence[str],
                   importance: Sequence[float]) -> dict:
        """Fit ``names`` against a sub-vector built from the chosen blocks,
        holding the remaining parameters at ``fixed`` values.

        The simulator objective is not convex: the ground-truth SSE valley
        is a narrow spike (the probe is deterministic, so only the true
        parameters reproduce the observables exactly) sitting inside a
        shallow basin that also traps the coordinate descent.  After the
        coarse global pass the box is therefore re-centered on the best
        point and re-fitted at finer resolution (``refine_windows`` gives
        the half-width fractions), which lets the pattern search resolve
        the spike.
        """
        idx = self._block_indices()
        sel = [i for b in blocks for i in idx[b]]
        sub_obs = [self.observed[i] for i in sel]
        sub_weights = self._weights_for(blocks, importance)

        def partial_predict(**cand: float) -> list[float]:
            params = dict(fixed)
            params.update(cand)
            return self.predict(probes=blocks, **params)

        ranges = {k: self.ranges[k] for k in names}
        result = fit_parameters(
            partial_predict, sub_obs, ranges,
            n_samples=self.n_samples,
            seed=self.fit_seed,
            refine_rounds=self.refine_rounds,
            weights=sub_weights,
            polish_passes=self.polish_passes,
        )
        for window in self.refine_windows:
            center = result["best"]
            box = {
                n: (max(ranges[n][0], center[n] * (1.0 - window)),
                    min(ranges[n][1], center[n] * (1.0 + window)))
                for n in names
            }
            result = fit_parameters(
                partial_predict, sub_obs, box,
                n_samples=max(2, self.n_samples // 3),
                seed=self.fit_seed,
                refine_rounds=1,
                weights=sub_weights,
                polish_passes=max(2, self.polish_passes // 2),
            )
        return result

    def calibrate(self) -> dict:
        """Recover the three parameters from the ground-truth observables.

        Three nearly independent stages, as documented at the module level:
        the metabolic knobs do not touch the division/energy blocks, the
        energy scale is read division-free (its ``threshold - c*scale``
        coupling to the division block would otherwise leave a diagonal
        ridge), and the threshold is then fit with the scale held fixed.
        """
        truth = {
            "dfba_oxygen_max_uptake": TRUTH_OXYGEN_MAX_UPTAKE,
            "dfba_energy_scale": TRUTH_ENERGY_SCALE,
            "division_threshold": TRUTH_DIVISION_THRESHOLD,
        }
        metabolic = self._fit_stage(
            ["dfba_oxygen_max_uptake"], truth,
            METABOLIC_BLOCKS, (0.7, 0.3))
        fixed = dict(truth)
        fixed["dfba_oxygen_max_uptake"] = (
            metabolic["best"]["dfba_oxygen_max_uptake"])
        energy = self._fit_stage(
            ["dfba_energy_scale"], fixed,
            ENERGY_BLOCKS, (1.0,))
        fixed["dfba_energy_scale"] = energy["best"]["dfba_energy_scale"]
        division = self._fit_stage(
            ["division_threshold"], fixed,
            DIVISION_BLOCKS, (1.0,))
        best = dict(truth)
        best.update(metabolic["best"])
        best.update(energy["best"])
        best.update(division["best"])
        total = (metabolic["n_samples"] + energy["n_samples"]
                 + division["n_samples"])
        pred = self._observables(
            oxygen_max=best["dfba_oxygen_max_uptake"],
            energy_scale=best["dfba_energy_scale"],
            division_threshold=best["division_threshold"])
        sse = sum(w * (p - o) ** 2
                  for w, p, o in zip(self.weights, pred, self.observed,
                                     strict=True))
        return {"best": best, "sse": sse, "n_samples": total}

    def run(self) -> dict:
        """Full loop: fit, then score each parameter against the truth."""
        fit = self.calibrate()
        fitted = {k: float(fit["best"][k]) for k in self.ranges}
        truth = {
            "dfba_oxygen_max_uptake": TRUTH_OXYGEN_MAX_UPTAKE,
            "dfba_energy_scale": TRUTH_ENERGY_SCALE,
            "division_threshold": TRUTH_DIVISION_THRESHOLD,
        }
        rel_err: dict[str, float] = {}
        recovered: dict[str, bool] = {}
        for k in self.ranges:
            rel_err[k] = abs(fitted[k] - truth[k]) / abs(truth[k])
            recovered[k] = rel_err[k] < 0.1
        return {
            "fit": fit,
            "fitted": fitted,
            "truth": truth,
            "relative_error": rel_err,
            "recovered": recovered,
            "passed": all(recovered.values()),
        }


def run_population_calibration(
    n_samples: int = 12,
    refine_rounds: int = 2,
    fit_seed: int = 0,
    division_ticks: int = 12,
    n_cells: int = 4,
) -> dict:
    """One-shot population-level mixed-observable calibration benchmark."""
    return PopulationCalibration(
        n_samples=n_samples, refine_rounds=refine_rounds,
        fit_seed=fit_seed, division_ticks=division_ticks, n_cells=n_cells,
    ).run()


__all__ = [
    "PopulationCalibration",
    "run_population_calibration",
    "TRUTH_OXYGEN_MAX_UPTAKE",
    "TRUTH_ENERGY_SCALE",
    "TRUTH_DIVISION_THRESHOLD",
]
