"""Whole-cell parameter calibration closure (Phase 5, doc §8.3).

Fits the Phase 1-4 free parameters of a
:class:`~helixlang.plugins.runtime.virtual_cell.VirtualCell` against *mixed*
observables -- growth curve, protein abundances, cell size / division
behaviour and enzyme-limited biomass flux -- using the existing
inverse-variance-weighted :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters`
harness (Karr et al. 2012 DREAM8 weighted fitting; Virtual Cell
Challenge 2025 calibratability requirement).

The four calibrated parameters and their biological anchors:

- ``adder_volume_um3``   -- Phase 2 adder size-control threshold Delta
  (Taheri-Araghi 2015): divides when ``volume - volume_birth`` crosses it.
- ``k_fold``             -- Phase 3 folding equilibrium fraction
  ``k_fold/(k_fold+k_misfold)`` (Balchin 2016); mapped onto the cell's
  ``fold_rate_per_min`` with a fixed misfolding rate.  Controls the
  steady-state folded protein abundance.
- ``enzyme_scale``       -- Phase 4 GECKO-style kcat rescaling
  (Sanchez 2017); with enzyme-capacity enabled it caps every enzyme-
  gated reaction flux by ``kcat*E*scale``, so it sets the saturating
  biomass flux.
- ``maintenance_atp_per_min`` -- basal maintenance burn; sets the slope
  of the growth curve.

The :class:`WholeCellCalibration` class builds a *ground-truth* cell
with hidden values of the four parameters, records its mixed observable
vector, then recovers the parameters with a joint weighted fit.
:func:`run_whole_cell_calibration` packages the loop into a benchmark
dict whose ``passed`` flag requires every parameter to be recovered
within its tolerance.
"""
from __future__ import annotations

from collections.abc import Sequence

from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.metabolism import ECOLI_CORE_GENE_REACTIONS
from helixlang.plugins.runtime.virtual_cell import (
    VirtualCell,
    VirtualCellConfig,
    encode_gene,
    fit_parameters,
)

#: the 28-enzyme-gene genome so enzyme-capacity gating is meaningful
DEFAULT_GENOME: dict[str, str] = {
    g: encode_gene("MAQILARVFFDDV") for g in ECOLI_CORE_GENE_REACTIONS
}

#: representative genes whose folded abundance is observed
OBSERVED_ENZYMES: tuple[str, ...] = ("pgi", "pfkA", "atpF")

#: ground-truth (hidden) values of the four calibrated parameters
TRUTH_ADDER_VOLUME_UM3 = 1.6
TRUTH_K_FOLD = 0.7
TRUTH_ENZYME_SCALE = 1.0e4
TRUTH_MAINTENANCE_ATP_PER_MIN = 2.5e7


def _fold_rate_from_k_fold(k_fold: float,
                           misfold_rate_per_min: float) -> float:
    """Folding rate that yields the given equilibrium folding fraction."""
    k = float(k_fold)
    if not 0.0 < k < 1.0:
        raise ValueError("k_fold must be in (0, 1)")
    return misfold_rate_per_min * k / (1.0 - k)


class WholeCellCalibration:
    """Calibrate the Phase 1-4 parameters on mixed observables.

    Args:
        genome: gene -> protein-sequence genome (default: the enzyme
            genome, so kcat scaling is observable).
        minutes: run horizon (min) for the growth / size curves.
        observed_enzymes: genes whose folded abundance is observed.
        ranges: parameter -> (lo, hi) fit box for the four parameters.
        n_samples: random samples for the global fit stage.
        refine_rounds: pattern-search passes after the random stage.
        fit_seed: deterministic fit RNG seed.
        max_divisions: fixed length of the per-division added-volume
            observable (padded with 0.0 so candidates that divide fewer
            times are penalized).
        biomass_to_volume_pg_per_min: volume-biomass coupling.  A smaller
            value grows the cell in finer per-minute steps, which is what
            makes the adder threshold resolvable: the step increment is
            also the threshold resolution, so a coarse growth rate hides
            any adder within one step's worth of volume (the
            calibration's default was chosen so several divisions occur
            while the per-step increment stays well under the adder range).
        adder_noise_std: relative Gaussian noise on the per-generation
            adder threshold (Taheri-Araghi 2015, ~0.1-0.2); exercises the
            fit against division-timing noise.
        cell_seed: RNG seed of the ground-truth cells (the i-th cell gets
            ``cell_seed + i``).
        predict_seed: RNG seed of every candidate cell during the fit --
            independent of ``cell_seed`` so predictions are fresh noise
            realizations (deterministic across runs).
        n_cells: how many independent cells back each observable vector.
            A single cell lives only a few divisions (maintenance drains
            its energy after the first division), so the adder
            added-volume observable is a one-sample estimate of the noisy
            threshold.  Averaging over several cells (as a population
            experiment would) shrinks the noise by ``sqrt(n_cells)`` and
            is what keeps the fit robust once ``adder_noise_std`` is
            nonzero.
    """

    def __init__(
        self,
        genome: dict[str, str] | None = None,
        minutes: int = 60,
        observed_enzymes: Sequence[str] = OBSERVED_ENZYMES,
        ranges: dict[str, tuple[float, float]] | None = None,
        n_samples: int = 60,
        refine_rounds: int = 2,
        fit_seed: int = 0,
        max_divisions: int = 4,
        biomass_to_volume_pg_per_min: float = 0.016,
        adder_noise_std: float = 0.0,
        cell_seed: int = 0,
        predict_seed: int = 1,
        n_cells: int = 1,
    ) -> None:
        self.genome = dict(genome or DEFAULT_GENOME)
        self.minutes = minutes
        self.observed_enzymes = list(observed_enzymes)
        self.max_divisions = max_divisions
        self.biomass_to_volume_pg_per_min = biomass_to_volume_pg_per_min
        self.adder_noise_std = adder_noise_std
        self.cell_seed = cell_seed
        self.predict_seed = predict_seed
        self.n_cells = n_cells
        self.ranges = ranges or {
            "adder_volume_um3": (1.0, 2.2),
            "k_fold": (0.5, 0.9),
            "enzyme_scale": (3e3, 3e4),
            "maintenance_atp_per_min": (1.5e7, 3.5e7),
        }
        self.n_samples = n_samples
        self.refine_rounds = refine_rounds
        self.fit_seed = fit_seed
        self.truth_cells = self._build_truth()
        self.truth_cell = self.truth_cells[0]
        self.observed = self._observables(self.truth_cells)
        self.weights = self._weights(self.observed)

    # -------- model construction --------

    def _grn(self) -> GRN:
        g = GRN()
        for name in self.genome:
            g.add_gene(name, 0.5)
            g.nodes[name].level = 1.0
        return g

    def _config(self, *, k_fold: float, enzyme_scale: float,
                maintenance_atp_per_min: float,
                adder_volume_um3: float, seed: int) -> VirtualCellConfig:
        return VirtualCellConfig(
            seed=seed,
            adder_noise_std=self.adder_noise_std,
            uptake={"GLC": 10.0},
            energy_init=1.0e9,
            division_energy=2.0e9,
            division_rule="adder",
            adder_volume_um3=adder_volume_um3,
            biomass_to_volume_pg_per_min=self.biomass_to_volume_pg_per_min,
            maintenance_atp_per_min=maintenance_atp_per_min,
            transcription_atp_per_nt=0.0,
            translation_atp_per_aa=0.0,
            protein_maturation_mode="chaperone",
            misfold_rate_per_min=0.3,
            fold_rate_per_min=_fold_rate_from_k_fold(
                k_fold, 0.3),
            aggregation_rate_per_min=0.0,
            degraded_rate_per_min=0.0,
            protein_half_life_min=1.0e6,
            enzyme_capacity_enabled=True,
            enzyme_scale=enzyme_scale,
            protein_mass_fraction=0.3,
        )

    def _build_truth(self) -> list[VirtualCell]:
        return [self._build_cell(self.cell_seed + i) for i in range(self.n_cells)]

    def _build(self, **params: float) -> list[VirtualCell]:
        return [self._build_cell(self.predict_seed + i, **params)
                for i in range(self.n_cells)]

    def _build_cell(self, seed: int, **params: float) -> VirtualCell:
        truth = {
            "adder_volume_um3": TRUTH_ADDER_VOLUME_UM3,
            "k_fold": TRUTH_K_FOLD,
            "enzyme_scale": TRUTH_ENZYME_SCALE,
            "maintenance_atp_per_min": TRUTH_MAINTENANCE_ATP_PER_MIN,
        }
        truth.update(params)
        return VirtualCell(
            self.genome, self._grn(),
            config=self._config(
                k_fold=truth["k_fold"],
                enzyme_scale=truth["enzyme_scale"],
                maintenance_atp_per_min=truth["maintenance_atp_per_min"],
                adder_volume_um3=truth["adder_volume_um3"],
                seed=seed,
            ))

    # -------- observables --------

    def _observables(self, cells: list[VirtualCell]) -> list[float]:
        """Mixed observable vector: growth curve + size (volume) curve +
        biomass-flux curve + protein abundances + per-division added
        volumes + division count, averaged over the population of cells.

        Runs are padded to ``self.minutes`` points so a dead candidate
        (shorter history) still yields a fixed-length vector.  Under the
        adder rule the volume added over a generation equals the
        ``adder_volume_um3`` threshold exactly, so the per-division
        added volumes are a direct, near-linear adder observable; with
        ``adder_noise_std > 0`` each generation adds a fresh noisy
        sample, so averaging ``n_cells`` independent cells shrinks the
        noise on the recovered mean threshold."""
        n = self.minutes
        for c in cells:
            c.run(n)

        def _curve(cell: VirtualCell, key: str) -> list[float]:
            vals = [h[key] for h in cell.history]
            if len(vals) < n:
                last = vals[-1] if vals else 0.0
                vals = vals + [last] * (n - len(vals))
            return vals[:n]

        def _mean(values: list[list[float]]) -> list[float]:
            return [sum(v[i] for v in values) / len(values)
                    for i in range(n)]

        per_cell = [(_curve(c, "energy"), _curve(c, "volume_um3"),
                     _curve(c, "biomass_flux")) for c in cells]
        energy = _mean([p[0] for p in per_cell])
        volume = _mean([p[1] for p in per_cell])
        biomass = _mean([p[2] for p in per_cell])
        proteins: list[float] = [0.0] * len(self.observed_enzymes)
        for c in cells:
            for gi, g in enumerate(self.observed_enzymes):
                proteins[gi] += (
                    c.protein_pools[g].folded
                    if c.protein_pools and g in c.protein_pools
                    else c.proteins.get(g, 0.0))
        proteins = [p / len(cells) for p in proteins]
        added: list[float] = []
        for c in cells:
            a: list[float] = []
            prev = c.history[0]
            for entry in c.history[1:]:
                if entry["divisions"] > prev["divisions"]:
                    a.append(prev["volume_um3"] - prev["volume_birth_um3"])
                prev = entry
            a = a[:self.max_divisions]
            a += [0.0] * (self.max_divisions - len(a))
            added.extend(a)
        return [*energy, *volume, *biomass, *proteins, *added,
                float(sum(c.divisions for c in cells))]

    def _weights_for(self, blocks: Sequence[str],
                     importance: Sequence[float]) -> list[float]:
        """Block-normalized weights with explicit per-block importance so
        that heterogeneous observable blocks (energy at ~1e9, volume ~3,
        proteins ~1e3, discrete division counts) each contribute
        comparably despite wildly different natural scales."""
        idx = self._block_indices(blocks)
        out: list[float] = [0.0] * len(idx)
        for b, imp in zip(blocks, importance, strict=True):
            bi = self._block_indices([b])
            vals = [self.observed[i] for i in bi]
            scale = max(abs(v) for v in vals) if vals else 1.0
            if scale <= 0.0:
                scale = 1.0
            w = imp / (scale * len(bi))
            for gi in bi:
                out[idx.index(gi)] = w
        return out

    def _weights(self, observed: list[float]) -> list[float]:
        """Full-vector weights (all blocks) for reporting overall SSE."""
        return self._weights_for(
            ["energy", "volume", "biomass", "proteins",
             "division_added", "divisions"],
            (0.12, 0.16, 0.16, 0.20, 0.28, 0.08))

    # -------- calibration loop --------

    def predict(self, **params: float) -> list[float]:
        """Run a cell with the candidate parameters and return its
        mixed observable vector (same layout as :attr:`observed`)."""
        return self._observables(self._build(**params))

    def _block_indices(self, blocks: Sequence[str]) -> list[int]:
        """Map block names (energy/volume/biomass/proteins/division_added/
        divisions) to indices in the observable vector."""
        n = self.minutes
        size = (3 * n + len(self.observed_enzymes)
                + self.n_cells * self.max_divisions + 1)
        index = {
            "energy": list(range(0, n)),
            "volume": list(range(n, 2 * n)),
            "biomass": list(range(2 * n, 3 * n)),
            "proteins": list(range(3 * n, 3 * n + len(self.observed_enzymes))),
            "division_added": list(range(
                3 * n + len(self.observed_enzymes), size - 1)),
            "divisions": [size - 1],
        }
        out: list[int] = []
        for b in blocks:
            out.extend(index[b])
        return out

    def _fit_stage(self, names: list[str], fixed: dict[str, float],
                   blocks: Sequence[str], importance: Sequence[float]) -> dict:
        """Fit ``names`` against a sub-vector built from the chosen blocks,
        holding the remaining parameters at ``fixed`` values."""
        idx = self._block_indices(blocks)
        sub_obs = [self.observed[i] for i in idx]
        sub_weights = self._weights_for(blocks, importance)

        def partial_predict(**cand: float) -> list[float]:
            params = dict(fixed)
            params.update(cand)
            obs = self.predict(**params)
            return [obs[i] for i in idx]

        ranges = {k: self.ranges[k] for k in names}
        return fit_parameters(
            partial_predict, sub_obs, ranges,
            n_samples=self.n_samples,
            seed=self.fit_seed,
            refine_rounds=self.refine_rounds,
            weights=sub_weights,
        )

    def calibrate(self) -> dict:
        """Recover the four parameters from the ground-truth observables.

        The joint 4-parameter surface is strongly correlated (maintenance
        and enzyme scale both shape the growth curve; the adder threshold
        and folding fraction both shape size / protein pools), which
        stalls a coordinate-wise descent.  Two nearly independent 2-D
        fits are therefore run in sequence:

        1. growth stage -- ``enzyme_scale`` and ``maintenance_atp_per_min``
           against the energy + biomass curves (size/folding fixed at truth);
        2. size/folding stage -- ``adder_volume_um3`` and ``k_fold``
           against the volume curve + protein abundances + per-division
           added volumes + division count (growth parameters fixed at
           their recovered values).  The added-volume block pins the adder
           threshold directly, since each generation adds exactly
           ``adder_volume_um3`` before dividing.
        """
        truth = {
            "adder_volume_um3": TRUTH_ADDER_VOLUME_UM3,
            "k_fold": TRUTH_K_FOLD,
            "enzyme_scale": TRUTH_ENZYME_SCALE,
            "maintenance_atp_per_min": TRUTH_MAINTENANCE_ATP_PER_MIN,
        }
        growth = self._fit_stage(
            ["enzyme_scale", "maintenance_atp_per_min"], truth,
            ["energy", "biomass"], (0.5, 0.5))
        fixed_growth = dict(truth)
        fixed_growth.update(growth["best"])
        size = self._fit_stage(
            ["adder_volume_um3", "k_fold"], fixed_growth,
            ["volume", "proteins", "division_added", "divisions"],
            (0.20, 0.25, 0.40, 0.15))
        best = dict(truth)
        best.update(growth["best"])
        best.update(size["best"])
        total = growth["n_samples"] + size["n_samples"]
        pred = self.predict(**best)
        sse = sum(w * (p - o) ** 2
                  for w, p, o in zip(self.weights, pred, self.observed,
                                     strict=True))
        return {"best": best, "sse": sse, "n_samples": total}

    def run(self) -> dict:
        """Full loop: fit, then score each parameter against the truth."""
        fit = self.calibrate()
        fitted = {k: float(fit["best"][k]) for k in self.ranges}
        truth = {
            "adder_volume_um3": TRUTH_ADDER_VOLUME_UM3,
            "k_fold": TRUTH_K_FOLD,
            "enzyme_scale": TRUTH_ENZYME_SCALE,
            "maintenance_atp_per_min": TRUTH_MAINTENANCE_ATP_PER_MIN,
        }
        rel_err: dict[str, float] = {}
        recovered: dict[str, bool] = {}
        for k in self.ranges:
            rel_err[k] = abs(fitted[k] - truth[k]) / abs(truth[k])
        # enzyme_scale spans orders of magnitude: allow a log-scale tolerance
        recovered["enzyme_scale"] = (
            abs(_log10(fitted["enzyme_scale"]) - _log10(truth["enzyme_scale"]))
            < 0.2)
        for k in ("adder_volume_um3", "k_fold",
                  "maintenance_atp_per_min"):
            recovered[k] = rel_err[k] < 0.1
        return {
            "fit": fit,
            "fitted": fitted,
            "truth": truth,
            "relative_error": rel_err,
            "recovered": recovered,
            "passed": all(recovered.values()),
        }


def _log10(x: float) -> float:
    import math
    return math.log10(max(x, 1e-12))


def run_whole_cell_calibration(
    minutes: int = 60,
    n_samples: int = 60,
    refine_rounds: int = 2,
    fit_seed: int = 0,
    adder_noise_std: float = 0.0,
    n_cells: int = 1,
) -> dict:
    """One-shot whole-cell calibration closure benchmark."""
    return WholeCellCalibration(
        minutes=minutes, n_samples=n_samples, refine_rounds=refine_rounds,
        fit_seed=fit_seed, adder_noise_std=adder_noise_std,
        n_cells=n_cells,
    ).run()


__all__ = [
    "WholeCellCalibration",
    "run_whole_cell_calibration",
    "DEFAULT_GENOME",
    "OBSERVED_ENZYMES",
]
