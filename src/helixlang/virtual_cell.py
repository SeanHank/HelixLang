"""Virtual-cell integration and validation benchmarks (T3.4, gap G8).

Ties the four modeling layers into one cell-cycle budget model and
publishes standardized benchmark cases:

- :class:`VirtualCell`: a Karr-2012-style cell that couples
  :mod:`helixlang.central_dogma` (transcription/translation -> protein),
  :mod:`helixlang.grn` (regulation decides which genes are expressed),
  :mod:`helixlang.metabolism` (FBA biomass flux -> energy budget) and a
  cell-cycle energy budget (maintenance + division gate).
- :func:`fit_parameters`: a parameter-estimation harness (randomized
  search + coordinate refinement) that fits model parameters to observed
  data (e.g. doubling time, protein levels).
- :func:`run_biofilm_benchmark`: BM3-style uniform-biofilm growth metrics
  over a :class:`~helixlang.population.CellPopulation`.
- :func:`perturbation_response`: perturbation-response benchmark for a
  GRN (gene knockout, continuous-time response and settling metrics).

References:
- Karr et al. 2012. A whole-cell computational model of M. genitalium.
  Cell 150:389-401.
- Virtual Cell Challenge 2025 (integrated whole-cell benchmarks).
- iDynoMiCS 2.0 / BM3 biofilm benchmark (biofilm growth metrics).
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from helixlang.bio_data import ECOLI_CODON_USAGE
from helixlang.cell import INITIAL_CELL_ENERGY
from helixlang.central_dogma import (
    PROTEINS_PER_MRNA_LIFETIME,
    transcribe,
    translate,
)
from helixlang.grn import GRN, integrate_grn
from helixlang.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
from helixlang.population import CellPopulation

#: translation cost per amino acid (ATP; Karr et al. 2012)
VIRTUAL_TRANSLATION_ATP_PER_AA = 4.0
#: transcription cost per nucleotide (ATP; Karr et al. 2012)
VIRTUAL_TRANSCRIPTION_ATP_PER_NT = 1.0
#: basal maintenance cost (ATP/min; ~2.5e7 for a newborn E. coli,
#: Orth 2010 + Alberts dry mass)
VIRTUAL_MAINTENANCE_ATP_PER_MIN = 2.5e7
#: energy budget required to divide (ATP)
VIRTUAL_DIVISION_ENERGY = 2.0e9
#: ATP gained per unit biomass flux per minute (coupling constant
#: between FBA biomass flux and the whole-cell energy budget)
VIRTUAL_BIOMASS_TO_ATP = 1.0e6


# ============================================================================
# Gene encoding
# ============================================================================

#: preferred E. coli codon per amino acid (most frequent, ECOLI_CODON_USAGE)
_PREFERRED_CODON: dict[str, str] = {}
for _codon, _info in ECOLI_CODON_USAGE.items():
    _PREFERRED_CODON.setdefault(_info[0], _codon)


def encode_gene(protein: str, rbs: str = "AGGAGG") -> str:
    """Build a translatable CDS DNA sequence from a protein sequence.

    ``RBS + spacer + ATG(start) + codons(rest) + TAA`` using the most
    frequent E. coli codon for each amino acid (ECOLI_CODON_USAGE); the
    RBS sits within 15 nt of the start codon so :func:`translate`
    detects it.  The start codon supplies the N-terminal methionine, so
    ``encode_gene("MA...")`` and ``encode_gene("A...")`` both translate
    to a protein beginning with ``M``.
    """
    if not protein:
        raise ValueError("protein sequence must be non-empty")
    # the leading residue is encoded by the ATG start codon when it is
    # already a methionine; otherwise add the start codon explicitly
    if protein[0] == "M":
        codons = "".join(_PREFERRED_CODON.get(aa, "GCT")
                         for aa in protein[1:])
        return rbs + "GACC" + "ATG" + codons + "TAA"
    codons = "".join(_PREFERRED_CODON.get(aa, "GCT") for aa in protein)
    return rbs + "GACC" + "ATG" + codons + "TAA"


# ============================================================================
# Integrated virtual cell
# ============================================================================

@dataclass(slots=True)
class VirtualCellConfig:
    """Whole-cell budget parameters (Karr 2012-style).

    Args:
        energy_init: starting energy budget (ATP molecules).
        division_energy: energy at which the cell divides (halving).
        maintenance_atp_per_min: basal maintenance burn (ATP/min).
        biomass_to_atp: energy gain per unit FBA biomass flux per minute.
        translation_atp_per_aa: translation cost (ATP/amino acid).
        transcription_atp_per_nt: transcription cost (ATP/nucleotide).
        protein_yield_per_mrna: protein molecules per transcribed mRNA.
        minutes_per_step: physical time of one ``step()``.
        uptake: initial FBA exchange uptake bounds (metabolite -> rate),
            applied to the solver at construction.
    """

    energy_init: float = INITIAL_CELL_ENERGY
    division_energy: float = VIRTUAL_DIVISION_ENERGY
    maintenance_atp_per_min: float = VIRTUAL_MAINTENANCE_ATP_PER_MIN
    biomass_to_atp: float = VIRTUAL_BIOMASS_TO_ATP
    translation_atp_per_aa: float = VIRTUAL_TRANSLATION_ATP_PER_AA
    transcription_atp_per_nt: float = VIRTUAL_TRANSCRIPTION_ATP_PER_NT
    protein_yield_per_mrna: float = PROTEINS_PER_MRNA_LIFETIME
    minutes_per_step: float = 1.0
    uptake: dict[str, float] = field(default_factory=dict)


class VirtualCell:
    """Integrated cell: central dogma + GRN + metabolism + cell budget.

    Each :meth:`step` (one minute):
    1. advances the GRN (which genes cross the activation threshold);
    2. transcribes + translates each triggered gene, crediting protein
       and debiting transcription/translation ATP;
    3. solves the FBA for the biomass flux and credits the energy budget
       (``flux * biomass_to_atp``);
    4. pays basal maintenance; divides when the budget allows.

    The model is deliberately simple but couples the four layers so a
    calibration harness (e.g. :func:`fit_parameters` on doubling time)
    can tune the energy-coupling constants against data.
    """

    def __init__(self, genome: dict[str, str], grn: GRN,
                 fba: FluxBalanceAnalysis | None = None,
                 config: VirtualCellConfig | None = None,
                 name: str = "virtual-cell") -> None:
        self.genome = dict(genome)
        self.grn = grn
        self.config = config or VirtualCellConfig()
        self.name = name
        self.fba = fba or FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        for met, rate in self.config.uptake.items():
            self.fba.set_uptake(met, rate)
        self.energy: float = self.config.energy_init
        self.age: int = 0
        self.divisions: int = 0
        self.alive: bool = True
        self.proteins: dict[str, float] = {}
        self.mrna: dict[str, float] = {}
        self.mass: float = 1.0  # relative cell mass (biomass flux units)
        self.history: list[dict] = []

    # -------- internals --------

    def _express(self, gene: str) -> None:
        """Transcribe + translate one gene and pay the energy cost."""
        if gene not in self.genome:
            return
        dna = self.genome[gene]
        node = self.grn.nodes.get(gene)
        strength = max(0.0, min(1.0, node.level if node else 1.0))
        transcript = transcribe(dna, promoter_strength=strength)
        result = translate(transcript)
        cost = (len(dna) * self.config.transcription_atp_per_nt
                + len(result.protein) * self.config.translation_atp_per_aa)
        self.energy -= cost
        self.mrna[gene] = self.mrna.get(gene, 0.0) + 1.0
        self.proteins[gene] = (self.proteins.get(gene, 0.0)
                               + self.config.protein_yield_per_mrna)

    def _metabolism(self) -> float:
        """Return the FBA biomass flux (mmol/gDW/h)."""
        bm = self.fba.model.biomass_reaction
        if bm is None:
            return 0.0
        return self.fba.solve().get(bm, 0.0)

    # -------- public API --------

    def step(self) -> dict:
        """Advance one minute; returns the history entry appended."""
        cfg = self.config
        triggered = self.grn.step()
        for gene in triggered:
            self._express(gene)
        flux = self._metabolism()
        self.energy += flux * cfg.biomass_to_atp * cfg.minutes_per_step
        self.energy -= cfg.maintenance_atp_per_min * cfg.minutes_per_step
        self.mass += max(0.0, flux) * 0.01
        self.age += 1
        if self.energy <= 0.0:
            self.alive = False
        elif self.energy >= cfg.division_energy:
            self.energy /= 2.0
            self.divisions += 1
        entry = {
            "age": self.age,
            "energy": self.energy,
            "alive": self.alive,
            "divisions": self.divisions,
            "mass": self.mass,
            "biomass_flux": flux,
            "proteins": dict(self.proteins),
            "triggered": triggered,
        }
        self.history.append(entry)
        return entry

    def run(self, n_steps: int) -> list[dict]:
        """Run ``n_steps`` minutes; returns :attr:`history`."""
        for _ in range(n_steps):
            if not self.alive:
                break
            self.step()
        return self.history


# ============================================================================
# Parameter estimation harness
# ============================================================================

def fit_parameters(predict: Callable[..., list[float]], observed: list[float],
                   ranges: dict[str, tuple[float, float]],
                 n_samples: int = 500, seed: int = 0,
                 refine_rounds: int = 5, n_grid: int = 50,
                 weights: Sequence[float] | None = None) -> dict:
    """Fit model parameters to observed data.

    ``predict(**params) -> list[float]`` is evaluated at randomized
    parameter points inside ``ranges``; the best point is then refined
    in two stages: a coordinate-wise pattern search (full-box scan per
    axis with exponentially doubling resolution, which locates the
    valley) followed by parabolic-interpolation polish (which slides
    along narrow ridges such as ``a + b*x``).

    Args:
        predict: callable mapping parameters to a predicted vector.
        observed: target vector (same length as the prediction).
        ranges: parameter name -> (lower, upper) box.
        n_samples: random samples for the global search stage.
        seed: RNG seed (deterministic runs).
        refine_rounds: pattern-search passes after the random search;
            each pass scans every axis over its full range at
            ``2**(round+2)+1`` grid points.
        n_grid: kept for compatibility (resolution doubling is fixed);
            ignored by the current implementation.
        weights: optional per-observation weights (same length as
            ``observed``) giving the objective ``sum(w_i (p_i - o_i)^2)``.
            Supports multi-scale omics calibration: observations from
            heterogeneous readouts (mRNA vs protein, high- vs low-count
            perturb-seq conditions) are jointly fitted with inverse-
            variance weights (DESeq2 2014 variance structure
            ``Var = mu + dispersion*mu^2``; Karr et al. 2012 DREAM8
            weighted fitting). ``None`` = unit weights.

    Returns:
        ``{"best": {param: value}, "sse": float, "n_samples": int}``.
    """
    if not ranges:
        raise ValueError("ranges must be non-empty")
    if not observed:
        raise ValueError("observed must be non-empty")
    if weights is not None and len(weights) != len(observed):
        raise ValueError("weights must have the same length as observed")
    rng = random.Random(seed)
    names = list(ranges)

    def sse(params: dict) -> float:
        try:
            pred = predict(**params)
        except TypeError as exc:
            raise ValueError(
                f"predict must accept the fitted parameters {names!r} "
                f"as keyword arguments: {exc}") from exc
        if len(pred) != len(observed):
            raise ValueError(
                "prediction length must match observed length")
        if weights is not None:
            return sum(w * (p - o) ** 2
                       for w, p, o in zip(weights, pred, observed,
                                          strict=True))
        return sum((p - o) ** 2 for p, o in zip(pred, observed, strict=True))

    best = {n: rng.uniform(*ranges[n]) for n in names}
    best_sse = sse(best)
    total = 0
    for _ in range(n_samples):
        params = {n: rng.uniform(*ranges[n]) for n in names}
        total += 1
        s = sse(params)
        if s < best_sse:
            best, best_sse = params, s
    # stage 1: coordinate-wise pattern search over the full box with
    # exponentially doubling resolution per round. Finds the valley of a
    # correlated objective without window-halving stalls.
    for rnd in range(refine_rounds):
        for n in names:
            lo, hi = ranges[n]
            steps = 2 ** (rnd + 2)
            cand, cand_s = None, best_sse
            for i in range(steps + 1):
                v = lo + (hi - lo) * i / steps
                params = dict(best)
                params[n] = v
                total += 1
                s = sse(params)
                if s < cand_s:
                    cand_s, cand = s, v
            if cand is not None:
                best[n] = cand
                best_sse = cand_s
    # stage 2: parabolic-interpolation polish on each axis. The discrete
    # grid cannot slide along a narrow ridge (e.g. a + b*x), so fit a
    # parabola through three samples and jump to its vertex.
    for _ in range(64):
        improved = False
        for n in names:
            lo, hi = ranges[n]
            delta = max((hi - lo) / 100.0, 1e-9)
            params = dict(best)
            f0 = best_sse
            params[n] = min(hi, max(lo, best[n] - delta))
            f1 = sse(params)
            params[n] = min(hi, max(lo, best[n] + delta))
            f2 = sse(params)
            denom = f1 - 2.0 * f0 + f2
            if abs(denom) < 1e-30:
                continue
            vertex = best[n] - delta * (f2 - f1) / (2.0 * denom)
            vertex = min(hi, max(lo, vertex))
            params = dict(best)
            params[n] = vertex
            total += 1
            fv = sse(params)
            if fv < best_sse:
                best, best_sse = params, fv
                improved = True
        if not improved:
            break
    return {"best": best, "sse": best_sse, "n_samples": total}


# ============================================================================
# Standardized benchmarks
# ============================================================================

def run_biofilm_benchmark(population: CellPopulation, n_steps: int = 120,
                          interval: int = 10) -> dict:
    """BM3-style uniform-biofilm growth benchmark.

    Runs a :class:`~helixlang.population.CellPopulation` (or
    ``CellPopulation3D``) for ``n_steps`` and reports standardized
    biofilm metrics: biomass (alive-cell) time series, final biomass,
    spatial extent and an estimated doubling interval.

    Returns:
        a dict with ``biomass`` (timeseries), ``final_biomass``,
        ``max_extent``, ``doubling_ticks`` and ``growth_rate_per_tick``.
    """
    biomass: list[int] = []
    extent: list[float] = []
    for s in range(n_steps):
        population.step()
        if s % interval == 0 or s == n_steps - 1:
            alive = [c for c in population.cells if c.alive]
            biomass.append(len(alive))
            if alive:
                xs = [c.x for c in alive]
                ys = [c.y for c in alive]
                extent.append(max(max(xs) - min(xs),
                                  max(ys) - min(ys)))
            else:
                extent.append(0.0)
    growth: float = 0.0
    if len(biomass) >= 2 and biomass[0] > 0:
        growth = (biomass[-1] - biomass[0]) / biomass[0] / max(1, n_steps)
    doubling_ticks: float | None = None
    for i in range(1, len(biomass)):
        if biomass[i] >= 2 * biomass[0] and biomass[0] > 0:
            doubling_ticks = i * interval
            break
    return {
        "biomass": biomass,
        "final_biomass": biomass[-1] if biomass else 0,
        "max_extent": max(extent) if extent else 0.0,
        "doubling_ticks": doubling_ticks,
        "growth_rate_per_tick": growth,
    }


def _clone_grn(grn: GRN) -> GRN:
    out = GRN(noise_enabled=grn.noise_enabled)
    for name, node in grn.nodes.items():
        out.add_gene(name, node.threshold, initial_level=node.level,
                     decay=node.decay, hill_n=node.hill_n, kd=node.kd,
                     noise=node.noise)
    for e in grn.edges:
        out.add_edge(e.source, e.target, e.weight)
    return out


def perturbation_response(grn: GRN, target: str,
                          knockout: str | None = None,
                          t_span: tuple[float, float] = (0.0, 600.0),
                          n_points: int = 300) -> dict:
    """Perturbation-response benchmark (continuous-time GRN).

    Integrates the GRN ODE (T2.2 DOPRI5) once unperturbed and once with
    ``knockout``'s outgoing edges set to zero, then reports the response
    of ``target``: final fold change, settling time (first time the
    trajectory is within 5% of its final value) and the response curves.

    Returns:
        a dict with ``control_final``, ``perturbed_final``,
        ``fold_change``, ``settling_time`` (minutes), ``times`` and
        ``response`` (perturbed trajectory of ``target``).
    """
    control = integrate_grn(grn, t_span, n_points=n_points,
                            method="rk45").trajectory(target)
    if knockout is not None:
        grn_p = _clone_grn(grn)
        for e in grn_p.edges:
            if e.source == knockout:
                e.weight = 0.0
        grn_p._rebuild_incoming()
    else:
        grn_p = grn
    result = integrate_grn(grn_p, t_span, n_points=n_points, method="rk45")
    perturbed = result.trajectory(target)
    control_final = control[-1]
    perturbed_final = perturbed[-1]
    fold = (perturbed_final / control_final if control_final > 0
            else float("inf"))
    settling = None
    tol = 0.05 * abs(perturbed_final)
    for t, v in zip(result.times, perturbed, strict=True):
        if abs(v - perturbed_final) <= tol:
            settling = t
            break
    return {
        "control_final": control_final,
        "perturbed_final": perturbed_final,
        "fold_change": fold,
        "settling_time": settling,
        "times": result.times,
        "response": perturbed,
    }


__all__ = [
    "VirtualCell",
    "VirtualCellConfig",
    "encode_gene",
    "fit_parameters",
    "perturbation_response",
    "run_biofilm_benchmark",
    "VIRTUAL_BIOMASS_TO_ATP",
    "VIRTUAL_DIVISION_ENERGY",
    "VIRTUAL_MAINTENANCE_ATP_PER_MIN",
    "VIRTUAL_TRANSCRIPTION_ATP_PER_NT",
    "VIRTUAL_TRANSLATION_ATP_PER_AA",
]
